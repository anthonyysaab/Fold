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
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from devfun_poker_playground.learned_policy import load_policy
from devfun_poker_playground.poker_policy import build_policy
from devfun_poker_playground.table_simulator import (
    MatchResult,
    RecordingPolicy,
    run_sessions,
    ScriptedAgent,
    TableSimulator,
    TexturedAgent,
)

# (battery name, archetype builder, heads-up hands at scale 1.0)
#
# vs-textured is the held-out leg: a TexturedAgent whose fold rate prices the
# wager and the board (precondition P3, phase one). No candidate has ever
# trained against it, and unlike the four legs above it punishes oversized
# aggression, so it is the falsification test NEXT.md item 3 asks for. Keep it
# OUT of the harvest legs while it is doing that job, or it stops being
# held-out (phase two moves it into the harvest, knowingly).
_BATTERIES: tuple[tuple[str, Callable[[int], ScriptedAgent], int], ...] = (
    ("vs-median", lambda seed: ScriptedAgent("median", 0.226, 0.5, 0.0, seed), 1_000),
    ("vs-nit", lambda seed: ScriptedAgent("nit", 0.05, 0.85, 0.0, seed), 1_000),
    ("vs-station", lambda seed: ScriptedAgent("station", 0.15, 0.05, 0.0, seed), 1_000),
    ("vs-shover", lambda seed: ScriptedAgent("shover", 0.0, 0.0, 1.0, seed), 1_500),
    (
        "vs-textured",
        lambda seed: TexturedAgent("textured", 0.226, 0.5, 0.0, seed),
        1_000,
    ),
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

    # Hands this agent was actually seated for. A busted agent stops being
    # dealt in while the table plays on, so the table-wide count would divide
    # its busts by hands it never played and understate ruin.
    hands = sum(
        (getattr(result, "hands_by_agent", None) or {}).get(agent_id, result.hands)
        for result in results
    )
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


@dataclass(frozen=True)
class _PolicySpec:
    """A policy described by value, so a worker process can rebuild it.

    Matches are dispatched across processes; policy factories are closures
    and cannot be pickled, while a manifest path and its load options can.
    """

    label: str
    kind: str
    equity_trials: int
    manifest: str | None = None
    use_learned_sizing: bool | None = None
    hybrid_min_advantage: float | None = None
    hybrid_max_abs_z: float = 5.0

    def build(self) -> object:
        if self.kind == "heuristic":
            return build_policy(aggressive=True, equity_trials=self.equity_trials)
        options: dict = {"equity_trials": self.equity_trials}
        if self.use_learned_sizing is not None:
            options["use_learned_sizing"] = self.use_learned_sizing
        if self.hybrid_min_advantage is not None:
            options["hybrid_min_value_advantage"] = self.hybrid_min_advantage
            options["hybrid_max_abs_z"] = self.hybrid_max_abs_z
        return load_policy(self.manifest, **options)


@dataclass(frozen=True)
class _BatteryTask:
    """One battery match: a policy against one seeded opponent set."""

    policy: _PolicySpec
    battery: str
    opponent_seed: int
    hands: int
    seed: int
    stack: int
    reset_stacks: bool

    def opponents(self) -> list[tuple[str, object]]:
        if self.battery == "five-max-lineup":
            return _lineup(self.opponent_seed)
        builder = next(build for name, build, _ in _BATTERIES if name == self.battery)
        return [(self.battery, builder(self.opponent_seed))]

    def run(self) -> MatchResult:
        return _match(
            self.policy.build,
            self.opponents(),
            self.hands,
            self.seed,
            self.stack,
            self.reset_stacks,
        )


@dataclass(frozen=True)
class _DuelTask:
    """One duel orientation: `first` takes seat 0 and its button advantage."""

    first: _PolicySpec
    second: _PolicySpec
    hands: int
    seed: int
    stack: int
    reset_stacks: bool

    def run(self) -> MatchResult:
        return _duel_match(
            (self.first.label, self.first.build),
            (self.second.label, self.second.build),
            self.hands,
            self.seed,
            self.stack,
            self.reset_stacks,
        )


def _run_task(task: _BatteryTask | _DuelTask) -> MatchResult:
    """Module-level entry point so tasks survive pickling to a worker."""

    return task.run()


def _run_tasks(
    tasks: Sequence[_BatteryTask | _DuelTask], workers: int
) -> list[MatchResult]:
    """Run matches, preserving order. Every task carries its own seed, so
    results do not depend on whether they ran concurrently."""

    if workers == 1 or len(tasks) <= 1:
        return [task.run() for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_task, tasks))


def _gauntlet_workers(requested: int, tasks: int) -> int:
    """Resolve the worker count; 0 means fill the cores within reason."""

    if requested == 1 or tasks <= 1:
        return 1
    if requested > 1:
        return min(requested, tasks)
    return max(1, min(tasks, (os.cpu_count() or 2) - 1))


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


def battery_tasks(
    policy: _PolicySpec,
    *,
    seeds: tuple[int, ...],
    scale: float,
    stack: int,
    reset_stacks: bool,
) -> list[tuple[str, int, _BatteryTask]]:
    """Every battery match for one policy, tagged by battery and hand count."""

    tasks: list[tuple[str, int, _BatteryTask]] = []
    for battery, _, base_hands in _BATTERIES:
        hands = max(50, round(base_hands * scale))
        for index, _ in enumerate(seeds):
            tasks.append(
                (
                    battery,
                    hands,
                    _BatteryTask(
                        policy=policy,
                        battery=battery,
                        opponent_seed=13 + index,
                        hands=hands,
                        seed=100 + index,
                        stack=stack,
                        reset_stacks=reset_stacks,
                    ),
                )
            )
    hands = max(50, round(1_000 * scale))
    for index, _ in enumerate(seeds):
        tasks.append(
            (
                "five-max-lineup",
                hands,
                _BatteryTask(
                    policy=policy,
                    battery="five-max-lineup",
                    opponent_seed=23 + index,
                    hands=hands,
                    seed=200 + index,
                    stack=stack,
                    reset_stacks=reset_stacks,
                ),
            )
        )
    return tasks


def evaluate_policy(
    name: str,
    factory: Callable[[], object],
    *,
    seeds: tuple[int, ...],
    scale: float,
    stack: int,
    reset_stacks: bool,
) -> dict:
    """Sequential single-policy battery sweep, kept for direct callers."""

    del name
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


def duel_tasks(
    first: _PolicySpec,
    second: _PolicySpec,
    *,
    seeds: tuple[int, ...],
    scale: float,
    stack: int,
    reset_stacks: bool,
) -> list[_DuelTask]:
    """Both seat orientations for every duel seed, forward then swapped."""

    hands = max(50, round(2_000 * scale))
    tasks: list[_DuelTask] = []
    for index, _ in enumerate(seeds):
        seed = 300 + index
        tasks.append(_DuelTask(first, second, hands, seed, stack, reset_stacks))
        tasks.append(_DuelTask(second, first, hands, seed, stack, reset_stacks))
    return tasks


def duel_entry(
    name_a: str,
    name_b: str,
    hands: int,
    results: Sequence[MatchResult],
) -> dict:
    """Assemble a duel report from forward/swapped results in task order."""

    a_first = [result.bb_per_100(name_a) for result in results[0::2]]
    b_first = [result.bb_per_100(name_a) for result in results[1::2]]
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
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "processes for the independent matches: 0 (default) fills the "
            "cores, 1 runs them one after another. Every match is separately "
            "seeded, so the report is identical either way"
        ),
    )
    args = parser.parse_args(argv)

    policies: list[_PolicySpec] = []
    if args.include_heuristic:
        label = _policy_label(
            build_policy(aggressive=True, equity_trials=args.equity_trials),
            "heuristic-v5",
        )
        policies.append(
            _PolicySpec(label=label, kind="heuristic", equity_trials=args.equity_trials)
        )
    for manifest in args.candidate:
        loaded = load_policy(manifest, equity_trials=args.equity_trials)
        version = _policy_label(loaded, manifest)
        policies.append(
            _PolicySpec(
                label=version,
                kind="candidate",
                equity_trials=args.equity_trials,
                manifest=manifest,
            )
        )
        if args.ablate_sizing:
            policies.append(
                _PolicySpec(
                    label=f"{version}-heuristic-sizing",
                    kind="candidate",
                    equity_trials=args.equity_trials,
                    manifest=manifest,
                    use_learned_sizing=False,
                )
            )
        if args.hybrid_min_advantage is not None:
            policies.append(
                _PolicySpec(
                    label=f"{version}-hybrid",
                    kind="candidate",
                    equity_trials=args.equity_trials,
                    manifest=manifest,
                    use_learned_sizing=False,
                    hybrid_min_advantage=args.hybrid_min_advantage,
                    hybrid_max_abs_z=args.hybrid_max_abs_z,
                )
            )
    if not policies:
        parser.error("nothing to evaluate; pass --include-heuristic or --candidate")

    seeds = tuple(range(args.seeds))
    duel_seeds = tuple(
        range(args.duel_seeds if args.duel_seeds is not None else args.seeds)
    )
    baseline = policies[0].label
    report: dict = {
        "batteries": {},
        "duels": {},
        "starting_stack": args.starting_stack,
        "stacks": "reset-per-hand" if args.reset_stacks else "carry-over-sessions",
        "baseline": baseline,
        "battery_seed_count": len(seeds),
        "duel_seed_count": len(duel_seeds),
    }

    # Build every match up front. They are separately seeded and mutually
    # independent, so one pool runs the whole gauntlet and the results are
    # reassembled in task order -- identical output, a fraction of the
    # wall clock.
    battery_plan: list[tuple[str, str, int, _BatteryTask]] = []
    for policy in policies:
        for battery, hands, task in battery_tasks(
            policy,
            seeds=seeds,
            scale=args.scale,
            stack=args.starting_stack,
            reset_stacks=args.reset_stacks,
        ):
            battery_plan.append((policy.label, battery, hands, task))
    duel_plan: list[tuple[str, str, int, list[_DuelTask]]] = []
    for index_a in range(len(policies)):
        for index_b in range(index_a + 1, len(policies)):
            first, second = policies[index_a], policies[index_b]
            tasks = duel_tasks(
                first,
                second,
                seeds=duel_seeds,
                scale=args.scale,
                stack=args.starting_stack,
                reset_stacks=args.reset_stacks,
            )
            duel_plan.append((first.label, second.label, tasks[0].hands, tasks))

    all_tasks: list[_BatteryTask | _DuelTask] = [item[3] for item in battery_plan]
    for _, _, _, tasks in duel_plan:
        all_tasks.extend(tasks)
    workers = _gauntlet_workers(args.workers, len(all_tasks))
    if workers > 1:
        print(
            f"running {len(all_tasks)} matches across {workers} worker processes",
            flush=True,
        )
    outcomes = _run_tasks(all_tasks, workers)

    cursor = 0
    grouped: dict[tuple[str, str], tuple[int, list[MatchResult]]] = {}
    for label, battery, hands, _ in battery_plan:
        entry = grouped.setdefault((label, battery), (hands, []))
        entry[1].append(outcomes[cursor])
        cursor += 1
    for (label, battery), (hands, results) in grouped.items():
        report["batteries"].setdefault(label, {})[battery] = _battery_entry(
            hands, results
        )
    for name_a, name_b, hands, tasks in duel_plan:
        results = outcomes[cursor : cursor + len(tasks)]
        cursor += len(tasks)
        report["duels"][f"{name_a} vs {name_b}"] = duel_entry(
            name_a, name_b, hands, results
        )

    if len(seeds) >= 2:
        for policy in policies[1:]:
            for battery, data in report["batteries"][policy.label].items():
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
