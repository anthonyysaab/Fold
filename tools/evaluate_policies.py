"""Simulated evaluation gauntlet for live and learned policies.

Runs each policy through the standard batteries -- heads-up against the
calibrated median, a nit, a station, and the permanent shover, plus a
five-handed table against the calibrated lineup -- and duels the loaded
policies against each other. Scores are BB/100 across seeded matches with
fresh policy instances per match (session opponent models start cold, as
they would live). Every arm also reports ruin (busts per 100 hands, bust
rate per session); duels run each seed in both seat orders to cancel the
structural first-seat button advantage; with 2+ seeds each arm reports
paired per-seed differences against the first-listed policy. Entirely
offline; no Arena requests.

Example (module mode puts the repository root on the import path):
    python -m tools.evaluate_policies --include-heuristic \
        --candidate artifacts/candidates/<v>.manifest.json --json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

from devfun_poker_playground.learned_policy import load_policy
from devfun_poker_playground.poker_policy import build_policy
from devfun_poker_playground.table_simulator import (
    MatchResult,
    RecordingPolicy,
    run_sessions,
    ScriptedAgent,
    TableSimulator,
)

# (battery name, archetype builder, heads-up hands at scale 1.0)
_BATTERIES: tuple[tuple[str, Callable[[int], ScriptedAgent], int], ...] = (
    ("vs-median", lambda seed: ScriptedAgent("median", 0.226, 0.5, 0.0, seed), 1_000),
    ("vs-nit", lambda seed: ScriptedAgent("nit", 0.05, 0.85, 0.0, seed), 1_000),
    ("vs-station", lambda seed: ScriptedAgent("station", 0.15, 0.05, 0.0, seed), 1_000),
    ("vs-shover", lambda seed: ScriptedAgent("shover", 0.0, 0.0, 1.0, seed), 1_500),
)


def paired_stats(diffs: Sequence[float]) -> dict:
    """Mean, sample SD, SE, and t statistic for per-seed paired differences."""

    count = len(diffs)
    if count < 2:
        raise ValueError("paired statistics need at least two seeds")
    mean = sum(diffs) / count
    variance = sum((diff - mean) ** 2 for diff in diffs) / (count - 1)
    sd = math.sqrt(variance)
    se = sd / math.sqrt(count)
    return {
        "diffs": [round(diff, 2) for diff in diffs],
        "mean": round(mean, 2),
        "sd": round(sd, 2),
        "se": round(se, 2),
        "t": round(mean / se, 2) if se > 0 else None,
    }


def ruin_stats(results: Sequence[MatchResult], agent_id: str) -> dict:
    """Aggregate ruin metrics for one agent across per-seed match results.

    Median session length and survival-conditional BB/100 are not computable:
    MatchResult carries only per-match totals, not per-session hand counts or
    per-session chip deltas. In reset-per-hand mode busts stay zero because a
    reset stack cannot bust.
    """

    hands = sum(result.hands for result in results)
    sessions = sum(result.sessions for result in results)
    busts = sum(result.busts.get(agent_id, 0) for result in results)
    chips = sum(result.chip_deltas.get(agent_id, 0) for result in results)
    big_blind = results[0].big_blind
    return {
        "sessions": sessions,
        "busts": busts,
        "busts_per_100_hands": round(100.0 * busts / hands, 4) if hands else 0.0,
        "bust_rate_per_session": round(busts / sessions, 4) if sessions else 0.0,
        "mean_session_hands": round(hands / sessions, 1) if sessions else 0.0,
        "bb_per_100": round(100.0 * chips / (big_blind * hands), 2) if hands else 0.0,
    }


def _policy_label(policy: object, fallback: str) -> str:
    return str(getattr(policy, "policy_version", fallback))


def _lineup(seed: int) -> list[tuple[str, ScriptedAgent]]:
    return [
        ("median-bot", ScriptedAgent("median-bot", 0.226, 0.5, 0.0, seed)),
        ("tight-bot", ScriptedAgent("tight-bot", 0.10, 0.75, 0.0, seed + 1)),
        ("wild-bot", ScriptedAgent("wild-bot", 0.35, 0.30, 0.02, seed + 2)),
        ("station-bot", ScriptedAgent("station-bot", 0.15, 0.05, 0.0, seed + 3)),
    ]


def _match(
    hero_factory: Callable[[], object],
    opponents: list[tuple[str, object]],
    hands: int,
    seed: int,
    stack: int,
    reset_stacks: bool,
) -> MatchResult:
    if reset_stacks:
        simulator = TableSimulator(seed=seed, starting_stack=stack)
        return simulator.play_match(
            [("hero", RecordingPolicy(hero_factory())), *opponents], hands=hands
        )
    factories = [
        ("hero", lambda: RecordingPolicy(hero_factory())),
        *[(name, (lambda agent=agent: agent)) for name, agent in opponents],
    ]
    return run_sessions(factories, target_hands=hands, seed=seed, starting_stack=stack)


def _battery_entry(hands: int, results: Sequence[MatchResult]) -> dict:
    values = [result.bb_per_100("hero") for result in results]
    return {
        "hands_per_seed": hands,
        "bb_per_100": round(sum(values) / len(values), 2),
        "seeds": [round(value, 2) for value in values],
        "ruin": ruin_stats(results, "hero"),
    }


def evaluate_policy(
    name: str,
    factory: Callable[[], object],
    *,
    seeds: tuple[int, ...],
    scale: float,
    stack: int,
    reset_stacks: bool,
) -> dict:
    report: dict = {}
    for battery, archetype, base_hands in _BATTERIES:
        hands = max(50, round(base_hands * scale))
        results = [
            _match(
                factory,
                [(battery, archetype(13 + index))],
                hands,
                100 + index,
                stack,
                reset_stacks,
            )
            for index, _ in enumerate(seeds)
        ]
        report[battery] = _battery_entry(hands, results)
    hands = max(50, round(1_000 * scale))
    results = [
        _match(factory, _lineup(23 + index), hands, 200 + index, stack, reset_stacks)
        for index, _ in enumerate(seeds)
    ]
    report["five-max-lineup"] = _battery_entry(hands, results)
    return report


def _duel_match(
    first: tuple[str, Callable[[], object]],
    second: tuple[str, Callable[[], object]],
    hands: int,
    seed: int,
    stack: int,
    reset_stacks: bool,
) -> MatchResult:
    if reset_stacks:
        simulator = TableSimulator(seed=seed, starting_stack=stack)
        return simulator.play_match(
            [
                (first[0], RecordingPolicy(first[1]())),
                (second[0], RecordingPolicy(second[1]())),
            ],
            hands=hands,
        )
    return run_sessions(
        [
            (first[0], lambda: RecordingPolicy(first[1]())),
            (second[0], lambda: RecordingPolicy(second[1]())),
        ],
        target_hands=hands,
        seed=seed,
        starting_stack=stack,
    )


def duel(
    name_a: str,
    factory_a: Callable[[], object],
    name_b: str,
    factory_b: Callable[[], object],
    *,
    seeds: tuple[int, ...],
    scale: float,
    stack: int,
    reset_stacks: bool,
) -> dict:
    """Duel two policies with each seed played in both seat orders.

    Every value under ``orientations`` and ``seeds`` is ``name_a``'s BB/100:
    ``a_first``/``b_first`` are the raw orientations (who took seat 0 and its
    structural button advantage), ``seeds`` is their per-seed seat-mean, and
    ``paired`` treats those seat-means as per-seed A-minus-B comparisons with
    a null of zero.
    """

    hands = max(50, round(2_000 * scale))
    a_first: list[float] = []
    b_first: list[float] = []
    results: list[MatchResult] = []
    for index, _ in enumerate(seeds):
        seed = 300 + index
        forward = _duel_match(
            (name_a, factory_a), (name_b, factory_b), hands, seed, stack, reset_stacks
        )
        swapped = _duel_match(
            (name_b, factory_b), (name_a, factory_a), hands, seed, stack, reset_stacks
        )
        a_first.append(forward.bb_per_100(name_a))
        b_first.append(swapped.bb_per_100(name_a))
        results.extend((forward, swapped))
    seat_means = [
        round((first + second) / 2, 2)
        for first, second in zip(a_first, b_first, strict=True)
    ]
    data = {
        "hands_per_seed": hands,
        f"{name_a}_bb_per_100": round(sum(seat_means) / len(seat_means), 2),
        "seeds": seat_means,
        "orientations": {
            "a_first": [round(value, 2) for value in a_first],
            "b_first": [round(value, 2) for value in b_first],
        },
        "ruin": {
            name_a: ruin_stats(results, name_a),
            name_b: ruin_stats(results, name_b),
        },
    }
    if len(seat_means) >= 2:
        data["paired"] = paired_stats(seat_means)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="candidate manifest to evaluate (repeatable)",
    )
    parser.add_argument(
        "--include-heuristic",
        action="store_true",
        help="also evaluate the live heuristic aggressive policy",
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument(
        "--duel-seeds",
        type=int,
        help="duel seed count, oversampling duels independently of the "
        "batteries (defaults to --seeds)",
    )
    parser.add_argument(
        "--ablate-sizing",
        action="store_true",
        help="also evaluate each candidate's action head with heuristic sizing",
    )
    parser.add_argument(
        "--hybrid-min-advantage",
        type=float,
        help=(
            "also evaluate an action-value candidate as a heuristic correction "
            "layer; override only above this predicted advantage"
        ),
    )
    parser.add_argument("--hybrid-max-abs-z", type=float, default=5.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--equity-trials", type=int, default=80)
    parser.add_argument(
        "--starting-stack",
        type=int,
        default=6_000,
        help=(
            "chips per seat at 50/100 blinds; the 6,000 default is 60bb, "
            "matching the audited arena depth (99.5%% of decisions at 50bb+)"
        ),
    )
    parser.add_argument(
        "--reset-stacks",
        action="store_true",
        help=(
            "reset stacks every hand (fixed-depth measurement mode); the "
            "default plays carry-over sessions that end on busts, like live"
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--output",
        help="also write the JSON report to this path as UTF-8 without BOM",
    )
    args = parser.parse_args(argv)

    policies: list[tuple[str, Callable[[], object]]] = []
    if args.include_heuristic:

        def heuristic_factory() -> object:
            return build_policy(aggressive=True, equity_trials=args.equity_trials)

        policies.append(
            (_policy_label(heuristic_factory(), "heuristic-v5"), heuristic_factory)
        )
    for manifest in args.candidate:
        loaded = load_policy(manifest, equity_trials=args.equity_trials)
        version = _policy_label(loaded, manifest)

        def factory(path: str = manifest) -> object:
            return load_policy(path, equity_trials=args.equity_trials)

        policies.append((version, factory))
        if args.ablate_sizing:
            policies.append(
                (
                    f"{version}-heuristic-sizing",
                    lambda path=manifest: load_policy(
                        path,
                        equity_trials=args.equity_trials,
                        use_learned_sizing=False,
                    ),
                )
            )
        if args.hybrid_min_advantage is not None:
            policies.append(
                (
                    f"{version}-hybrid",
                    lambda path=manifest: load_policy(
                        path,
                        equity_trials=args.equity_trials,
                        use_learned_sizing=False,
                        hybrid_min_value_advantage=args.hybrid_min_advantage,
                        hybrid_max_abs_z=args.hybrid_max_abs_z,
                    ),
                )
            )
    if not policies:
        parser.error("nothing to evaluate; pass --include-heuristic or --candidate")

    seeds = tuple(range(args.seeds))
    duel_seeds = tuple(
        range(args.duel_seeds if args.duel_seeds is not None else args.seeds)
    )
    baseline = policies[0][0]
    report: dict = {
        "batteries": {},
        "duels": {},
        "starting_stack": args.starting_stack,
        "stacks": "reset-per-hand" if args.reset_stacks else "carry-over-sessions",
        "baseline": baseline,
        "battery_seed_count": len(seeds),
        "duel_seed_count": len(duel_seeds),
    }
    for name, factory in policies:
        report["batteries"][name] = evaluate_policy(
            name,
            factory,
            seeds=seeds,
            scale=args.scale,
            stack=args.starting_stack,
            reset_stacks=args.reset_stacks,
        )
    if len(seeds) >= 2:
        for name, _ in policies[1:]:
            for battery, data in report["batteries"][name].items():
                reference = report["batteries"][baseline][battery]["seeds"]
                data["paired"] = {
                    "baseline": baseline,
                    **paired_stats(
                        [
                            value - ref
                            for value, ref in zip(data["seeds"], reference, strict=True)
                        ]
                    ),
                }
    for index_a in range(len(policies)):
        for index_b in range(index_a + 1, len(policies)):
            name_a, factory_a = policies[index_a]
            name_b, factory_b = policies[index_b]
            report["duels"][f"{name_a} vs {name_b}"] = duel(
                name_a,
                factory_a,
                name_b,
                factory_b,
                seeds=duel_seeds,
                scale=args.scale,
                stack=args.starting_stack,
                reset_stacks=args.reset_stacks,
            )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for name, batteries in report["batteries"].items():
            print(f"{name}:")
            for battery, data in batteries.items():
                line = (
                    f"  {battery:<16} {data['bb_per_100']:+8.1f} bb/100"
                    f"  (seeds {data['seeds']})"
                    f"  busts/100h {data['ruin']['busts_per_100_hands']:.2f}"
                    f"  bust/session {data['ruin']['bust_rate_per_session']:.2f}"
                )
                paired = data.get("paired")
                if paired and paired.get("t") is not None:
                    line += f"  t {paired['t']:+.2f}"
                print(line)
        for label, data in report["duels"].items():
            key = next(k for k in data if k.endswith("_bb_per_100"))
            line = (
                f"{label}: {data[key]:+.1f} bb/100 seat-mean ({data['seeds']})"
                f"  a-first {data['orientations']['a_first']}"
                f"  b-first {data['orientations']['b_first']}"
            )
            paired = data.get("paired")
            if paired and paired.get("t") is not None:
                line += f"  t {paired['t']:+.2f}"
            print(line)
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
