"""One self-improvement cycle: harvest a diverse diet, retrain, hand to the gate.

Harvest legs replay one decision per actor and compare every legal action family,
averaging repeated stochastic continuations for each family:

* the heuristic champion (or an explicit ``--on-policy`` candidate) five-handed
  against the calibrated archetype lineup;
* heads-up against the permanent shover, the station, and the nit (the
  exposure cycle one lacked -- its candidate regressed against shoves);
* optional sparring against a learned candidate (champion side recorded).

The harvest mixes with foreign teacher rows for behavior warm-up only. Reward
training uses log-percentage counterfactual action values.
``--examples-out`` persists the assembled corpus before training;
``--examples-in`` retrains from saved corpora without harvesting.
Candidate state only; promotion stays a separate evaluation decision.

Example (module mode puts the repository root on the import path):
    python -m tools.self_play_cycle --model-version candidate-mixed-0004 \
        --foreign-csv "foreign play data/.../top15_decisions.csv" \
        --sparring artifacts/candidates/candidate-mixed-0003.manifest.json \
        --return-scale-pct 20
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path

from devfun_poker_playground.decision_engine import SharedEquityCache
from devfun_poker_playground.foreign_data import load_foreign_training_examples
from devfun_poker_playground.learned_policy import load_policy
from devfun_poker_playground.offline_trainer import (
    print_branch_summary,
    TRAINING_OBJECTIVE,
    TrainingConfig,
    train_candidate,
    validate_training_device,
)
from devfun_poker_playground.poker_policy import build_policy
from devfun_poker_playground.table_simulator import (
    calibrated_lineup,
    RecordingPolicy,
    run_sessions,
    ScriptedAgent,
    TexturedAgent,
)
from devfun_poker_playground.training_telemetry import (
    load_training_corpus,
    save_training_corpus,
)


def _hero(equity_trials: int, on_policy: str | None) -> RecordingPolicy:
    # equity_cache: harvest-only memoization, measured at an 80.1% hit rate
    # here (branches x rollouts revisit the pinned decision state) and 0.0%
    # on the serve path, which is why live construction never passes one.
    #
    # hyper_aggression_chance=0.0: the anti-modeling roll is switched OFF for
    # harvesting, and only for harvesting. Its whole purpose is to stop an
    # opponent modelling us, but every harvest opponent is a ScriptedAgent or
    # TexturedAgent -- stateless seeded RNG with no memory of us and no
    # capacity to model anyone. Against a non-modelling opponent the noise is
    # all cost: it is excluded from training labels by construction, so it
    # burns harvest decisions outright, and it perturbs the trajectories that
    # the surrounding labelled decisions are conditioned on. Live play keeps
    # its floor (HYPER_AGGRESSION_CHANCE), where opponents can adapt.
    policy = (
        load_policy(
            on_policy,
            equity_trials=equity_trials,
            equity_cache=SharedEquityCache(),
            hyper_aggression_chance=0.0,
        )
        if on_policy
        else build_policy(
            aggressive=True,
            equity_trials=equity_trials,
            equity_cache=SharedEquityCache(),
            hyper_aggression_chance=0.0,
        )
    )
    return RecordingPolicy(policy)


@dataclass(frozen=True)
class _LegSpec:
    """A harvest leg described by value, so a worker process can rebuild it.

    Opponents are described rather than passed: agents hold policy objects
    that are expensive to ship between processes, and every archetype is a
    pure function of its seeded parameters.
    """

    name: str
    hands: int
    seed: int
    equity_trials: int
    starting_stack: int
    on_policy: str | None
    counterfactual_rollouts: int
    lineup_seed: int | None = None
    archetype: tuple[str, float, float, float, int] | None = None
    sparring: str | None = None
    sparring_record_both: bool = False
    # Build the archetype as a TexturedAgent rather than a ScriptedAgent
    # (precondition P3): its fold frequency responds to the price being laid
    # and to board texture, so expected value stops being linear in bet size.
    # A flag rather than a sixth tuple slot, because every existing leg's
    # archetype tuple must keep meaning exactly what it meant before.
    textured: bool = False

    def opponents(self) -> list:
        if self.lineup_seed is not None:
            return calibrated_lineup(self.lineup_seed)
        if self.archetype is not None:
            label, aggression, fold_vs_bet, shove_rate, seed = self.archetype
            factory = TexturedAgent if self.textured else ScriptedAgent
            return [(label, factory(label, aggression, fold_vs_bet, shove_rate, seed))]
        return []


def _sparring_partner(spec: _LegSpec) -> RecordingPolicy:
    """Build the sparring seat; recording it is opt-in via the leg spec.

    With ``sparring_record_both`` the literal partner value ``champion``
    means the built-in heuristic champion rather than a manifest path.
    """

    if spec.sparring == "champion" and spec.sparring_record_both:
        policy = build_policy(
            aggressive=True,
            equity_trials=spec.equity_trials,
            equity_cache=SharedEquityCache(),
        )
    else:
        policy = load_policy(spec.sparring, equity_trials=spec.equity_trials)
    return RecordingPolicy(policy, record_examples=spec.sparring_record_both)


def _run_leg(spec: _LegSpec) -> tuple[str, list]:
    """Run one leg and return its summary line and leg-tagged examples.

    Every leg carries its own seed, so results do not depend on whether
    legs run sequentially or concurrently.
    """

    if spec.hands <= 0:
        return "", []
    if spec.sparring is not None:
        agents = [
            ("hero", lambda: _hero(spec.equity_trials, spec.on_policy)),
            ("sparring", lambda: _sparring_partner(spec)),
        ]
    else:
        agents = [
            ("hero", lambda: _hero(spec.equity_trials, spec.on_policy)),
            *[
                (label, (lambda agent=agent: agent))
                for label, agent in spec.opponents()
            ],
        ]
    result = run_sessions(
        agents,
        target_hands=spec.hands,
        seed=spec.seed,
        starting_stack=spec.starting_stack,
        collect_examples=True,
        collect_counterfactuals=True,
        counterfactual_rollouts=spec.counterfactual_rollouts,
    )
    examples = [replace(example, harvest_leg=spec.name) for example in result.examples]
    summary = (
        f"{spec.name}: {len(examples)} examples over {result.hands} hands in "
        f"{result.sessions} carry-over sessions "
        f"(hero {result.bb_per_100('hero'):+.1f} bb/100, "
        f"busted {result.busts.get('hero', 0)}x)"
    )
    return summary, examples


def _harvest_workers(requested: int, legs: int) -> int:
    """Resolve the worker count; 0 means one process per leg within reason."""

    if requested == 1 or legs <= 1:
        return 1
    if requested > 1:
        return min(requested, legs)
    return max(1, min(legs, (os.cpu_count() or 2) - 1))


def harvest_specs(args) -> list[_LegSpec]:
    """The harvest legs in their fixed order, independent of scheduling."""

    common = {
        "equity_trials": args.equity_trials,
        "starting_stack": args.starting_stack,
        "on_policy": args.on_policy,
        "counterfactual_rollouts": args.counterfactual_rollouts,
    }
    specs = [
        _LegSpec(
            name="five-max lineup",
            hands=args.lineup_hands,
            seed=args.seed,
            lineup_seed=args.seed + 1,
            **common,
        )
    ]
    for offset, (label, aggression, fold_vs_bet, shove_rate, bump) in enumerate(
        (
            ("shover", 0.0, 0.0, 1.0, 10),
            ("station", 0.15, 0.05, 0.0, 11),
            ("nit", 0.05, 0.85, 0.0, 12),
        )
    ):
        specs.append(
            _LegSpec(
                name=f"heads-up vs {label}",
                hands=getattr(args, f"{label}_hands"),
                seed=args.seed + 20 + offset,
                archetype=(
                    label,
                    aggression,
                    fold_vs_bet,
                    shove_rate,
                    args.seed + bump,
                ),
                **common,
            )
        )
    # P3 phase two. Added as a NEW leg rather than by re-parameterising an
    # existing one, so every other leg's contribution to the corpus is
    # unchanged and the corpus stays comparable to candidate-v7-0001's.
    # Note this ends vs-textured's held-out status in the batteries: once the
    # model trains against this archetype, that battery measures fit, not
    # generalisation.
    specs.append(
        _LegSpec(
            name="heads-up vs textured",
            hands=args.textured_hands,
            seed=args.seed + 23,
            archetype=("textured", 0.226, 0.5, 0.0, args.seed + 13),
            textured=True,
            **common,
        )
    )
    if args.sparring and args.sparring_hands > 0:
        specs.append(
            _LegSpec(
                name="sparring",
                hands=args.sparring_hands,
                seed=args.seed + 30,
                sparring=args.sparring,
                sparring_record_both=getattr(args, "sparring_record_both", False),
                **common,
            )
        )
    return specs


def harvest(args) -> list:
    """Run every leg and concatenate examples in the fixed leg order.

    Legs are independent and separately seeded, so running them in worker
    processes produces the same examples in the same order as running them
    one after another; only wall-clock changes.
    """

    specs = [spec for spec in harvest_specs(args) if spec.hands > 0]
    workers = _harvest_workers(getattr(args, "harvest_workers", 1), len(specs))
    if workers == 1:
        results = [_run_leg(spec) for spec in specs]
    else:
        print(f"harvesting {len(specs)} legs across {workers} worker processes")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_leg, specs))
    examples: list = []
    for summary, leg_examples in results:
        if summary:
            print(summary)
        examples += leg_examples
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineup-hands", type=int, default=3_000)
    parser.add_argument("--shover-hands", type=int, default=2_000)
    parser.add_argument("--station-hands", type=int, default=1_000)
    parser.add_argument("--nit-hands", type=int, default=1_000)
    parser.add_argument(
        "--textured-hands",
        type=int,
        default=1_000,
        help=(
            "heads-up hands against the card-aware TexturedAgent, whose fold "
            "frequency responds to bet size and board texture (precondition "
            "P3); 0 disables the leg and reproduces the pre-P3 harvest mix"
        ),
    )
    parser.add_argument("--sparring", help="candidate manifest to spar against")
    parser.add_argument(
        "--on-policy",
        help="current candidate manifest used as the learning hero",
    )
    parser.add_argument("--sparring-hands", type=int, default=2_000)
    parser.add_argument(
        "--sparring-record-both",
        action="store_true",
        help=(
            "record the sparring seat's examples too; with this flag the "
            "sparring value 'champion' means the built-in heuristic champion "
            "instead of a manifest path"
        ),
    )
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--equity-trials", type=int, default=80)
    parser.add_argument(
        "--harvest-workers",
        type=int,
        default=0,
        help=(
            "processes for the independent harvest legs: 0 picks one per leg "
            "within the core count (default), 1 runs them one after another. "
            "Legs are separately seeded, so the examples and their order are "
            "verified identical either way; the measured speedup is 2.47x."
        ),
    )
    parser.add_argument(
        "--examples-out",
        help=(
            "write the assembled corpus (gzip float32 features plus a "
            ".meta.json sidecar) after the harvest, before training"
        ),
    )
    parser.add_argument(
        "--examples-in",
        action="append",
        default=[],
        help=(
            "train from saved corpora instead of harvesting; repeatable, "
            "concatenated in order"
        ),
    )
    parser.add_argument(
        "--starting-stack",
        type=int,
        default=6_000,
        help="chips per seat at 50/100 blinds; 6,000 matches arena depth",
    )
    parser.add_argument("--foreign-csv", action="append", default=[])
    parser.add_argument("--output-dir", default="artifacts/candidates")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=17,
        help="seed for the train/validation split, kept apart from --init-seed",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=17,
        help="seed for weight initialization, kept apart from --split-seed",
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--architecture",
        choices=("v6", "v7"),
        default="v6",
        help=(
            "v6 keeps the frozen three-output path; v7 trains the format-2 "
            "two-branch network with the size-conditioned value head (CUDA)"
        ),
    )
    parser.add_argument("--baseline-warmup-epochs", type=int, default=0)
    parser.add_argument("--behavior-warmup-epochs", type=int, default=1)
    parser.add_argument("--return-scale-pct", type=float, default=20.0)
    parser.add_argument("--reinforcement-multiplier", type=float, default=1.5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--counterfactual-rollouts",
        type=int,
        default=1,
        help="stochastic continuations averaged per legal family and selected state",
    )
    parser.add_argument(
        "--train-risk-head",
        action="store_true",
        help="train learned sizing; off by default until action values pass",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the session plan without harvesting or training",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.counterfactual_rollouts < 1:
        parser.error("--counterfactual-rollouts must be positive")
    if args.baseline_warmup_epochs != 0:
        parser.error("--baseline-warmup-epochs must be zero for v6")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.examples_in and args.foreign_csv:
        parser.error(
            "--examples-in trains from the saved corpus alone; "
            "persist the foreign rows into a corpus instead"
        )
    device_name = validate_training_device(args.device)

    harvesting = not args.examples_in
    if args.dry_run:
        foreign_rows = {}
        for csv_path in args.foreign_csv:
            foreign_rows[csv_path] = len(load_foreign_training_examples(csv_path))
        corpus_rows = {}
        for corpus_path in args.examples_in:
            corpus_rows[corpus_path] = len(load_training_corpus(corpus_path))
        if (
            args.sparring
            and harvesting
            and not (args.sparring == "champion" and args.sparring_record_both)
        ):
            load_policy(args.sparring, equity_trials=args.equity_trials)
        if args.on_policy and harvesting:
            load_policy(args.on_policy, equity_trials=args.equity_trials)
        output_dir = Path(args.output_dir).expanduser().resolve()
        targets = [
            output_dir / f"{args.model_version}.manifest.json",
            output_dir / f"{args.model_version}.weights.json",
        ]
        if any(path.exists() for path in targets):
            parser.error(f"candidate artifact already exists for {args.model_version}")
        print(
            json.dumps(
                {
                    "mode": "dry-run; no simulation or training performed",
                    "objective": TRAINING_OBJECTIVE,
                    "model_version": args.model_version,
                    "harvest_hands": {
                        "five_max": args.lineup_hands if harvesting else 0,
                        "shover": args.shover_hands if harvesting else 0,
                        "station": args.station_hands if harvesting else 0,
                        "nit": args.nit_hands if harvesting else 0,
                        "champion_only_sparring": (
                            args.sparring_hands if args.sparring and harvesting else 0
                        ),
                    },
                    "harvest_workers": args.harvest_workers,
                    "foreign_rows": foreign_rows,
                    "foreign_row_total": sum(foreign_rows.values()),
                    "examples_in_rows": corpus_rows,
                    "examples_in_total": sum(corpus_rows.values()),
                    "examples_out": args.examples_out,
                    "sparring_manifest": args.sparring,
                    "sparring_record_both": args.sparring_record_both,
                    "on_policy_manifest": args.on_policy,
                    "state_policy": args.on_policy or "heuristic-aggressive-v6",
                    "training": {
                        "epochs": args.epochs,
                        "split_seed": args.split_seed,
                        "init_seed": args.init_seed,
                        "learning_rate": args.learning_rate,
                        "baseline_warmup_epochs": args.baseline_warmup_epochs,
                        "behavior_warmup_epochs": args.behavior_warmup_epochs,
                        "return_scale_pct": args.return_scale_pct,
                        "reinforcement_multiplier": args.reinforcement_multiplier,
                        "gradient_clip": args.gradient_clip,
                        "device": args.device,
                        "device_name": device_name,
                        "batch_size": args.batch_size,
                        "train_risk_head": args.train_risk_head,
                        "validation_split": "whole hands by table_id",
                        "counterfactual_rollouts": args.counterfactual_rollouts,
                    },
                    "output_dir": str(output_dir),
                    "evaluation_command": (
                        "python -m tools.evaluate_policies --include-heuristic "
                        f"--candidate {targets[0]} --seeds 2 --json"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(f"training device: {args.device} ({device_name})")
    if args.examples_in:
        examples = []
        for corpus_path in args.examples_in:
            corpus = load_training_corpus(corpus_path)
            print(f"loaded {len(corpus)} corpus examples from {corpus_path}")
            examples.extend(corpus)
    else:
        print(f"state policy: {args.on_policy or 'heuristic-aggressive-v6'}")
        foreign_examples = []
        for csv_path in args.foreign_csv:
            foreign = load_foreign_training_examples(csv_path)
            print(f"loaded {len(foreign)} foreign teacher examples from {csv_path}")
            foreign_examples.extend(foreign)
        print(f"foreign teacher examples: {len(foreign_examples)}")
        examples = list(harvest(args))
        examples.extend(foreign_examples)
    if args.examples_out:
        save_training_corpus(args.examples_out, examples)
        print(f"wrote {len(examples)} corpus examples to {args.examples_out}")

    counterfactual_examples = [
        example for example in examples if example.counterfactual
    ]
    print(
        f"counterfactual reward examples: {len(counterfactual_examples)}; "
        f"behavior-only examples: {len(examples) - len(counterfactual_examples)}"
    )
    config_fields = {field.name for field in fields(TrainingConfig)}
    config_kwargs = {
        "epochs": args.epochs,
        "model_version": args.model_version,
        "baseline_warmup_epochs": args.baseline_warmup_epochs,
        "behavior_warmup_epochs": args.behavior_warmup_epochs,
        "return_scale_fraction": args.return_scale_pct / 100.0,
        "reinforcement_multiplier": args.reinforcement_multiplier,
        "gradient_clip": args.gradient_clip,
        "device": args.device,
        "batch_size": args.batch_size,
        "counterfactual_rollouts": args.counterfactual_rollouts,
        "train_risk_head": args.train_risk_head,
    }
    if "learning_rate" in config_fields:
        config_kwargs["learning_rate"] = args.learning_rate
    if "split_seed" in config_fields and "init_seed" in config_fields:
        config_kwargs["split_seed"] = args.split_seed
        config_kwargs["init_seed"] = args.init_seed
    elif "seed" in config_fields:
        # Until the trainer separates the split and init seeds, its single
        # seed drives both; the init seed wins the fallback.
        config_kwargs["seed"] = args.init_seed
    if "architecture" in config_fields:
        config_kwargs["architecture"] = args.architecture
    config = TrainingConfig(**config_kwargs)
    summary = train_candidate(tuple(examples), args.output_dir, config)
    print(f"examples: {summary.examples}")
    print(f"train_loss: {summary.train_loss:.6f}")
    print(f"validation_loss: {summary.validation_loss:.6f}")
    print_branch_summary(summary)
    print(f"manifest: {summary.manifest_path}")
    print("state: candidate (promotion requires the evaluation gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
