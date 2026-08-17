"""Gauntlet a format-3 (v8 composed-value) candidate through the frozen instrument.

Additive wrapper around ``tools.evaluate_policies`` — the measurement logic
(seat-swapped duels, paired per-seed t statistics, battery construction,
ruin denominators) is imported from there unchanged; this module only adds
what the v7 tool cannot field:

1. a policy spec whose ``build()`` goes through
   ``learned_policy_v8.load_policy_v8`` (format 3) instead of the v7 loader;
2. the trivial-baseline floor (``always_fold`` / ``always_check_call`` /
   ``always_aggress_small`` / ``always_aggress_large`` /
   ``uniform_random_legal``) run through the *same* battery channels, seeds,
   and session mode as the candidate, so the floor and the candidate are
   scored by one instrument.  The aggress baselines size at the E6
   stack-aware branch targets and the ``always_fold`` intent follows the v7
   absorption convention (a fold the table would give away for free is
   played as the free check — that *is* what the engine would execute);
3. per-street strength separation (DECISIONS §5) from a dedicated
   recording simulation, on the canonical ``strength_metric`` scale;
4. a self-duel null check: the candidate dueled against itself must be an
   exact rename-mirror (paired diffs identically zero), which validates the
   task wiring before any real number is read — measure the instrument
   before the result.

Every stage writes its fragment to ``--workdir`` as soon as it completes,
and the final report is assembled from whichever fragments exist, so a
crashed or skipped stage never throws away finished compute.

Entirely offline; no Arena requests, no promotion, ``artifacts/approved.json``
is never read or written.

Example (module mode puts the repository root on the import path):
    python -m tools.evaluate_v8 --workers 6 ^
        --output artifacts/evaluations/candidate-v8-0001-gauntlet.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devfun_poker_playground.feature_extract_v8 import _BRANCH_LARGE, _BRANCH_SMALL
from devfun_poker_playground.game_state import effective_stack_chips
from devfun_poker_playground.learned_policy_v8 import load_policy_v8
from devfun_poker_playground.strength_aware_opponent import (
    DEFAULT_FIT_PATH as P3_DEFAULT_FIT_PATH,
)
from devfun_poker_playground.strength_aware_opponent import StrengthAwareAgent
from devfun_poker_playground.strength_metric import strength_percentile
from devfun_poker_playground.table_simulator import (
    MatchResult,
    RecordingPolicy,
    ScriptedAgent,
    _family_of,
    run_sessions,
)
from tools.evaluate_policies import (
    _BATTERIES,
    _battery_entry,
    _gauntlet_workers,
    _lineup,
    _PolicySpec,
    _run_tasks,
    battery_tasks,
    duel_entry,
    duel_tasks,
    paired_stats,
)

DEFAULT_CANDIDATE = "artifacts/candidates/candidate-v8-0001.manifest.json"
DEFAULT_INCUMBENT = "artifacts/candidates/candidate-v7-0001c.approved.manifest.json"
DEFAULT_NOISE_FLOOR = "artifacts/evaluations/noise-floor-2026-08-15.json"

# The caller-facing rule from the reset (DECISIONS §4.7): a duel margin
# below the measured seed spread resolves nothing and must be reported as
# UNRESOLVED, never as a win or a loss.
KNOWN_SEED_SPREAD_BB_PER_100 = 16.78

TRIVIAL_MODES = (
    "always_fold",
    "always_check_call",
    "always_aggress_small",
    "always_aggress_large",
    "uniform_random_legal",
)

_STAGES = (
    "nullcheck",
    "trivial",
    "battery",
    "champion",
    "p3",
    "duel",
    "strength",
)


# ---------------------------------------------------------------------------
# Policy specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V8PolicySpec:
    """A format-3 policy described by value, rebuildable in a worker process.

    Mirrors ``evaluate_policies._PolicySpec``'s duck interface (``label``,
    ``build()``) so the existing battery/duel task machinery fields it
    unchanged.
    """

    label: str
    manifest: str
    equity_trials: int

    def build(self) -> object:
        return load_policy_v8(self.manifest, equity_trials=self.equity_trials)


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------


@dataclass
class TrivialAgent:
    """Constant-intent baseline speaking the Arena snapshot dialect.

    ``always_fold`` plays the free check when checking costs nothing (the
    v7 absorption convention: the engine would never execute a fold that a
    check dominates for free, so scoring the raw fold would mis-score the
    floor).  The two aggress modes size at the E6 stack-aware branch
    targets and fall through all-in -> check/call -> fold exactly like the
    counterfactual family payloads.  ``uniform_random_legal`` picks a legal
    family uniformly with a seeded, snapshot-keyed RNG (deterministic per
    seed, like ``ScriptedAgent``) and sizes aggress uniformly in the stated
    legal range.
    """

    mode: str
    seed: int = 977

    # Card-blind by construction, like every scripted archetype.
    reads_cards = False

    def __post_init__(self) -> None:
        if self.mode not in TRIVIAL_MODES:
            raise ValueError(f"unknown trivial mode {self.mode!r}")
        self.policy_version = f"trivial-{self.mode}"

    def _aggress_payload(
        self, table: Mapping[str, Any], allowed: Mapping[str, Any], available: set[str]
    ) -> dict | None:
        pot_fraction, stack_fraction = (
            _BRANCH_SMALL if self.mode == "always_aggress_small" else _BRANCH_LARGE
        )
        pot = int(table.get("potChips") or 0)
        to_call = int(allowed.get("callChips") or 0)
        hero = next(
            seat
            for seat in table["seats"]
            if seat["seatNumber"] == table["selfSeatNumber"]
        )
        contribution = int(hero["currentBetChips"])
        target = min(
            to_call + pot_fraction * (pot + to_call),
            stack_fraction * effective_stack_chips(table),
        )
        for action, range_name in (("raise", "raiseRange"), ("bet", "betRange")):
            amount_range = allowed.get(range_name)
            if action in available and amount_range is not None:
                low, high = int(amount_range["min"]), int(amount_range["max"])
                to_amount = min(high, max(low, int(round(contribution + target))))
                return {"action": action, "amount": to_amount, "message": "trivial"}
        if "all-in" in available:
            return {
                "action": "all-in",
                "amount": int(allowed["allInToAmount"]),
                "message": "trivial",
            }
        return None

    def _passive_payload(self, available: set[str]) -> dict:
        if "check" in available:
            return {"action": "check", "message": "trivial"}
        if "call" in available:
            return {"action": "call", "message": "trivial"}
        return {"action": "fold", "message": "trivial"}

    def decide(self, table: Mapping[str, Any]) -> dict:
        allowed = table["allowedActions"]
        available = set(allowed["availableActions"])
        if self.mode == "always_fold":
            if "check" in available:
                return {"action": "check", "message": "trivial"}
            return {"action": "fold", "message": "trivial"}
        if self.mode == "always_check_call":
            return self._passive_payload(available)
        if self.mode in ("always_aggress_small", "always_aggress_large"):
            payload = self._aggress_payload(table, allowed, available)
            return payload if payload is not None else self._passive_payload(available)
        # uniform_random_legal: seeded, snapshot-keyed, like ScriptedAgent.
        key = (
            f"{self.seed}:{self.mode}:{table['tableId']}:{table['street']}"
            f":{table['potChips']}:{table['selfSeatNumber']}"
        )
        roll = random.Random(key)
        families = ["fold"] if "fold" in available else []
        if "check" in available or "call" in available:
            families.append("check_call")
        can_size = any(
            action in available and allowed.get(range_name) is not None
            for action, range_name in (("bet", "betRange"), ("raise", "raiseRange"))
        )
        if can_size or "all-in" in available:
            families.append("aggress")
        family = families[roll.randrange(len(families))]
        if family == "fold":
            return {"action": "fold", "message": "trivial"}
        if family == "check_call":
            return self._passive_payload(available)
        for action, range_name in (("raise", "raiseRange"), ("bet", "betRange")):
            amount_range = allowed.get(range_name)
            if action in available and amount_range is not None:
                low, high = int(amount_range["min"]), int(amount_range["max"])
                return {
                    "action": action,
                    "amount": roll.randint(low, high),
                    "message": "trivial",
                }
        return {
            "action": "all-in",
            "amount": int(allowed["allInToAmount"]),
            "message": "trivial",
        }


@dataclass(frozen=True)
class TrivialBatteryTask:
    """One trivial-baseline battery match, mirroring ``_BatteryTask`` exactly.

    Same channels, same opponent seeds (13+/23+), same match seeds
    (100+/200+), same carry-over session mode — the floor must be measured
    by the same instrument as the candidate or it is not a floor.  The only
    difference is that the hero is a :class:`TrivialAgent`, which has no
    diagnostics and therefore skips the ``RecordingPolicy`` wrapper.
    """

    mode: str
    battery: str
    opponent_seed: int
    hands: int
    seed: int
    stack: int

    def run(self) -> MatchResult:
        if self.battery == "five-max-lineup":
            opponents = _lineup(self.opponent_seed)
        else:
            builder = next(
                build for name, build, _ in _BATTERIES if name == self.battery
            )
            opponents = [(self.battery, builder(self.opponent_seed))]
        mode = self.mode
        factories = [
            ("hero", lambda: TrivialAgent(mode)),
            *[(name, (lambda agent=agent: agent)) for name, agent in opponents],
        ]
        return run_sessions(
            factories,
            target_hands=self.hands,
            seed=self.seed,
            starting_stack=self.stack,
        )


def trivial_battery_plan(
    mode: str, *, seeds: Sequence[int], scale: float, stack: int
) -> list[tuple[str, int, TrivialBatteryTask]]:
    """The full battery plan for one trivial mode, tagged by channel."""

    plan: list[tuple[str, int, TrivialBatteryTask]] = []
    for battery, _, base_hands in _BATTERIES:
        hands = max(50, round(base_hands * scale))
        for index, _ in enumerate(seeds):
            plan.append(
                (
                    battery,
                    hands,
                    TrivialBatteryTask(
                        mode=mode,
                        battery=battery,
                        opponent_seed=13 + index,
                        hands=hands,
                        seed=100 + index,
                        stack=stack,
                    ),
                )
            )
    hands = max(50, round(1_000 * scale))
    for index, _ in enumerate(seeds):
        plan.append(
            (
                "five-max-lineup",
                hands,
                TrivialBatteryTask(
                    mode=mode,
                    battery="five-max-lineup",
                    opponent_seed=23 + index,
                    hands=hands,
                    seed=200 + index,
                    stack=stack,
                ),
            )
        )
    return plan


# ---------------------------------------------------------------------------
# The P3 channel — the first battery seat that can see its own cards
# ---------------------------------------------------------------------------
#
# ``PLAYERS.md`` ("What is missing, and what to build first") and
# ``V8_DESIGN.md`` §6.7: every existing battery archetype is card-blind, so
# none of them can punish indiscriminate aggression, and P3 enters the
# batteries as a **held-out gate before any training uses it**.
#
# The seat is deliberately the *exact structural twin* of ``vs-median``:
# identical aggression, identical nominal ``fold_vs_bet`` constant, identical
# opponent-seed and match-seed conventions.  The single difference is that its
# folding is the fitted card-aware model instead of the constant.  That makes
# ``vs-median`` the controlled reference for anything ``vs-p3`` shows, and it
# means a vs-p3 minus vs-median difference is attributable to one edit.

P3_CHANNEL = "vs-p3"
P3_CHANNEL_HANDS = 1_000
#: (name, aggression, fold_vs_bet, shove_rate) — see ``vs-median`` in
#: ``evaluate_policies._BATTERIES``, which this mirrors term for term.
P3_SEAT: tuple[str, float, float, float] = ("p3-median", 0.226, 0.5, 0.0)


class StrictStrengthAwareAgent(StrengthAwareAgent):
    """A P3 seat that refuses to silently become the card-blind archetype.

    ``StrengthAwareAgent`` degrades to its constant ``fold_vs_bet`` whenever a
    snapshot hides its cards, counting the event.  In a *battery* that
    degradation is not a graceful fallback, it is a silent channel swap: the
    arm believes it is measuring a strength-aware opponent and is in fact
    measuring ``vs-median``.  Here it is an error, so a gate can never report
    a vs-p3 number that was partly produced by a different opponent.
    """

    def _degrade(self, reason: str) -> float:  # pragma: no cover - guard
        raise AssertionError(
            f"{self.name}: the P3 seat degraded to the card-blind constant "
            f"({reason}); this arm would be measuring vs-median under the "
            "vs-p3 label and its number must not be read"
        )


def p3_opponents(
    opponent_seed: int,
    fit_path: str | Path = P3_DEFAULT_FIT_PATH,
    *,
    strict: bool = True,
) -> list[tuple[str, StrengthAwareAgent]]:
    """The vs-p3 seat, seeded exactly as ``vs-median``'s seat is."""

    name, aggression, fold_vs_bet, shove_rate = P3_SEAT
    factory = StrictStrengthAwareAgent if strict else StrengthAwareAgent
    return [
        (
            name,
            factory(
                name,
                aggression,
                fold_vs_bet,
                shove_rate,
                opponent_seed,
                fit_path=str(fit_path),
            ),
        )
    ]


def channel_opponents(
    channel: str,
    opponent_seed: int,
    fit_path: str | Path = P3_DEFAULT_FIT_PATH,
    *,
    strict_p3: bool = True,
) -> list[tuple[str, Any]]:
    """Opponent seats for any channel name, P3 or the frozen instrument's."""

    if channel == P3_CHANNEL:
        return list(p3_opponents(opponent_seed, fit_path, strict=strict_p3))
    if channel == "five-max-lineup":
        return list(_lineup(opponent_seed))
    builder = next((build for name, build, _ in _BATTERIES if name == channel), None)
    if builder is None:
        raise ValueError(f"unknown battery channel {channel!r}")
    return [(channel, builder(opponent_seed))]


def channel_hands(channel: str) -> int:
    """Hands per seed at scale 1.0 for any channel name."""

    if channel == P3_CHANNEL:
        return P3_CHANNEL_HANDS
    if channel == "five-max-lineup":
        return 1_000
    return next(hands for name, _, hands in _BATTERIES if name == channel)


@dataclass(frozen=True)
class TrivialSpec:
    """A trivial floor described by value, so a worker can rebuild it.

    Gives the constant-intent baselines the same ``label``/``build()`` duck
    interface as :class:`V8PolicySpec` and ``_PolicySpec``, so one task class
    can field every arm of a gate through one instrument.
    """

    mode: str

    @property
    def label(self) -> str:
        return f"trivial-{self.mode}"

    def build(self) -> object:
        return TrivialAgent(self.mode)


def _wrap_hero(policy: Any) -> Any:
    """Mirror the frozen instrument's hero wrapping, exactly.

    ``evaluate_policies._match`` wraps a ``DecisionEngine`` policy in
    ``RecordingPolicy``; ``TrivialBatteryTask`` leaves a bare agent alone
    because it has no diagnostics.  The rule is "wrap iff the object speaks
    the diagnostics protocol", so a mixed-arm gate reproduces both published
    conventions without a special case per arm.
    """

    return (
        RecordingPolicy(policy)
        if hasattr(policy, "decide_with_diagnostics")
        else policy
    )


@dataclass(frozen=True)
class ChannelOutcome:
    """One battery match plus the guard rails that decide if it is readable."""

    result: MatchResult
    chips_conserved: bool


@dataclass(frozen=True)
class ChannelBatteryTask:
    """One battery match on any channel for any arm, seeded as the instrument.

    Hands, opponent seed, match seed and the carry-over session mode are the
    published ``battery_tasks`` conventions verbatim, so a channel added here
    is scored by the same instrument as every frozen number, and two arms run
    at the same seed index meet the same cards in the same order.
    """

    hero: Any  # anything with .label and .build()
    channel: str
    opponent_seed: int
    hands: int
    seed: int
    stack: int
    fit_path: str = str(P3_DEFAULT_FIT_PATH)

    def run(self) -> ChannelOutcome:
        opponents = channel_opponents(self.channel, self.opponent_seed, self.fit_path)
        hero = self.hero
        factories = [
            ("hero", lambda: _wrap_hero(hero.build())),
            *[(name, (lambda agent=agent: agent)) for name, agent in opponents],
        ]
        result = run_sessions(
            factories,
            target_hands=self.hands,
            seed=self.seed,
            starting_stack=self.stack,
        )
        return ChannelOutcome(
            result=result,
            chips_conserved=sum(result.chip_deltas.values()) == 0,
        )


def channel_battery_plan(
    hero: Any,
    channels: Sequence[str],
    *,
    seeds: Sequence[int],
    scale: float,
    stack: int,
    fit_path: str | Path = P3_DEFAULT_FIT_PATH,
) -> list[tuple[str, int, ChannelBatteryTask]]:
    """Every (channel, seed) match for one arm, tagged by channel and hands."""

    plan: list[tuple[str, int, ChannelBatteryTask]] = []
    for channel in channels:
        hands = max(50, round(channel_hands(channel) * scale))
        base_seed = 200 if channel == "five-max-lineup" else 100
        base_opponent = 23 if channel == "five-max-lineup" else 13
        for index, _ in enumerate(seeds):
            plan.append(
                (
                    channel,
                    hands,
                    ChannelBatteryTask(
                        hero=hero,
                        channel=channel,
                        opponent_seed=base_opponent + index,
                        hands=hands,
                        seed=base_seed + index,
                        stack=stack,
                        fit_path=str(fit_path),
                    ),
                )
            )
    return plan


# ---------------------------------------------------------------------------
# Strength separation
# ---------------------------------------------------------------------------


class StrengthRecorder:
    """Wrap a DecisionEngine policy; record (street, family, strength) per decision.

    The recorded family is the **submitted** action's family (post bluff
    conversion, post safety gates) — the action the table actually saw —
    and strength is the canonical ``strength_metric.strength_percentile``
    of the hero's holding on the revealed board.
    """

    def __init__(self, policy: Any, records: list[tuple[str, str, float]]) -> None:
        self.policy = policy
        self.policy_version = getattr(policy, "policy_version", "policy")
        self.records = records

    def decide(self, table: Mapping[str, Any]) -> dict:
        payload = self.policy.decide_with_diagnostics(table).to_payload()
        family = _family_of(str(payload.get("action")))
        hero = next(
            seat
            for seat in table["seats"]
            if seat["seatNumber"] == table["selfSeatNumber"]
        )
        hole = hero.get("holeCards") or ()
        board = table.get("boardCards") or ()
        if family is not None and len(hole) == 2:
            street = str(table.get("street") or "preflop").casefold()
            self.records.append(
                (street, family, strength_percentile(hole, board))
            )
        return payload


@dataclass(frozen=True)
class StrengthTask:
    """One recording simulation: a policy spec against one opponent arena."""

    spec: Any  # V8PolicySpec | _PolicySpec — anything with .build()/.label
    arena: str  # "vs-median-hu" | "five-max-lineup"
    hands: int
    seed: int
    opponent_seed: int
    stack: int

    def run(self) -> list[tuple[str, str, float]]:
        records: list[tuple[str, str, float]] = []
        if self.arena == "five-max-lineup":
            opponents = _lineup(self.opponent_seed)
        else:
            opponents = [
                ("median", ScriptedAgent("median", 0.226, 0.5, 0.0, self.opponent_seed))
            ]
        spec = self.spec
        factories = [
            ("hero", lambda: StrengthRecorder(spec.build(), records)),
            *[(name, (lambda agent=agent: agent)) for name, agent in opponents],
        ]
        run_sessions(
            factories,
            target_hands=self.hands,
            seed=self.seed,
            starting_stack=self.stack,
        )
        return records


_SEPARATION_STREETS = ("preflop", "flop", "turn", "river")


def separation_report(records: Sequence[tuple[str, str, float]]) -> dict:
    """Per-street and pooled strength separation: mean(aggress) − mean(fold).

    Alongside the separation: per-family counts and means, the standard
    error of the difference of means, and a 95% normal CI — the design's
    preflop acceptance asks specifically whether the CI excludes zero.
    """

    def _bucket(street: str | None) -> dict:
        chosen = [
            record for record in records if street is None or record[0] == street
        ]
        families: dict[str, list[float]] = {}
        for _, family, strength in chosen:
            families.setdefault(family, []).append(strength)
        entry: dict[str, Any] = {"decisions": len(chosen), "families": {}}
        for family, values in sorted(families.items()):
            mean = sum(values) / len(values)
            entry["families"][family] = {
                "count": len(values),
                "mean_strength": round(mean, 4),
            }
        aggress = families.get("aggress", [])
        folds = families.get("fold", [])
        if len(aggress) >= 2 and len(folds) >= 2:
            mean_a = sum(aggress) / len(aggress)
            mean_f = sum(folds) / len(folds)
            var_a = sum((v - mean_a) ** 2 for v in aggress) / (len(aggress) - 1)
            var_f = sum((v - mean_f) ** 2 for v in folds) / (len(folds) - 1)
            se = math.sqrt(var_a / len(aggress) + var_f / len(folds))
            separation = mean_a - mean_f
            entry["separation_aggress_minus_fold"] = round(separation, 4)
            entry["se"] = round(se, 4)
            entry["ci95"] = [
                round(separation - 1.96 * se, 4),
                round(separation + 1.96 * se, 4),
            ]
        else:
            entry["separation_aggress_minus_fold"] = None
        return entry

    report = {"all_streets": _bucket(None)}
    for street in _SEPARATION_STREETS:
        report[street] = _bucket(street)
    return report


# ---------------------------------------------------------------------------
# Invariants — measure the instrument before the result
# ---------------------------------------------------------------------------


def _assert_chip_conservation(results: Sequence[MatchResult], stage: str) -> None:
    for index, result in enumerate(results):
        total = sum(result.chip_deltas.values())
        if total != 0:
            raise AssertionError(
                f"{stage}: chip conservation violated in match {index}: {total}"
            )


def _assert_duel_zero_sum(
    results: Sequence[MatchResult], name_a: str, name_b: str, stage: str
) -> None:
    for index, result in enumerate(results):
        if result.chip_deltas[name_a] != -result.chip_deltas[name_b]:
            raise AssertionError(f"{stage}: duel match {index} is not zero-sum")
        if result.hands_by_agent.get(name_a) != result.hands_by_agent.get(name_b):
            raise AssertionError(
                f"{stage}: duel match {index} has unequal hand counts"
            )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_nullcheck(v8_spec: V8PolicySpec, args: argparse.Namespace) -> dict:
    """Self-duel the candidate; the paired diffs must be exactly zero."""

    mirror_a = V8PolicySpec("v8-null-A", v8_spec.manifest, v8_spec.equity_trials)
    mirror_b = V8PolicySpec("v8-null-B", v8_spec.manifest, v8_spec.equity_trials)
    tasks = duel_tasks(
        mirror_a,
        mirror_b,
        seeds=tuple(range(args.null_seeds)),
        scale=args.null_scale,
        stack=args.starting_stack,
        reset_stacks=False,
    )
    results = _run_tasks(tasks, _gauntlet_workers(args.workers, len(tasks)))
    _assert_chip_conservation(results, "nullcheck")
    _assert_duel_zero_sum(results, "v8-null-A", "v8-null-B", "nullcheck")
    entry = duel_entry("v8-null-A", "v8-null-B", tasks[0].hands, results)
    diffs = entry.get("paired", {}).get("diffs", entry["seeds"])
    mirror_exact = all(diff == 0.0 for diff in diffs)
    if not mirror_exact:
        raise AssertionError(
            f"nullcheck: self-duel is not an exact mirror: {diffs} — the task "
            "wiring is broken and no other stage's numbers may be read"
        )
    return {
        "mirror_exact": mirror_exact,
        "seeds": len(diffs),
        "hands_per_seed": tasks[0].hands,
        "detail": "candidate self-duel paired diffs identically zero",
        "report": entry,
    }


def stage_trivial(args: argparse.Namespace) -> dict:
    seeds = tuple(range(args.trivial_seeds))
    plans = {
        mode: trivial_battery_plan(
            mode, seeds=seeds, scale=args.scale, stack=args.starting_stack
        )
        for mode in TRIVIAL_MODES
    }
    all_tasks = [task for plan in plans.values() for _, _, task in plan]
    results = _run_tasks(all_tasks, _gauntlet_workers(args.workers, len(all_tasks)))
    _assert_chip_conservation(results, "trivial")
    report: dict[str, Any] = {"seed_count": len(seeds), "floors": {}}
    cursor = 0
    for mode, plan in plans.items():
        grouped: dict[str, tuple[int, list[MatchResult]]] = {}
        for battery, hands, _ in plan:
            entry = grouped.setdefault(battery, (hands, []))
            entry[1].append(results[cursor])
            cursor += 1
        report["floors"][mode] = {
            battery: _battery_entry(hands, channel_results)
            for battery, (hands, channel_results) in grouped.items()
        }
    return report


def stage_battery(v8_spec: V8PolicySpec, args: argparse.Namespace) -> dict:
    seeds = tuple(range(args.battery_seeds))
    plan = battery_tasks(
        v8_spec,
        seeds=seeds,
        scale=args.scale,
        stack=args.starting_stack,
        reset_stacks=False,
    )
    tasks = [task for _, _, task in plan]
    results = _run_tasks(tasks, _gauntlet_workers(args.workers, len(tasks)))
    _assert_chip_conservation(results, "battery")
    grouped: dict[str, tuple[int, list[MatchResult]]] = {}
    for (battery, hands, _), result in zip(plan, results, strict=True):
        entry = grouped.setdefault(battery, (hands, []))
        entry[1].append(result)
    return {
        "seed_count": len(seeds),
        "channels": {
            battery: _battery_entry(hands, channel_results)
            for battery, (hands, channel_results) in grouped.items()
        },
    }


#: The policy whose battery means ``noise-floor-2026-08-15.json`` publishes.
#: It is **not** the duel incumbent — the noise floor's own ``champion`` field
#: says ``heuristic-aggressive-v6`` — and conflating the two turns a
#: policy difference into a fake "staleness" number.
NOISE_FLOOR_CHAMPION_LABEL = "heuristic-aggressive-v6"


def stage_champion(
    incumbent_spec: _PolicySpec, args: argparse.Namespace
) -> dict:
    """Rebuild both reference battery arms on this machine, this run.

    Two arms, because the project has two different things called "champion":

    * ``heuristic-aggressive-v6`` — the **noise-floor champion**, the policy
      whose per-channel means and MDEs ``noise-floor-2026-08-15.json``
      publishes and which ``battery_comparisons`` has always scored against.
    * the **duel incumbent** (``--incumbent``, ``candidate-v7-0001c``) — the
      policy the seat-swapped duel is fought against.

    Rebuilding the noise-floor champion is what makes the staleness question
    answerable *like for like*: the fresh arm and the published arm are the
    same policy, so any difference between them is the instrument moving
    (the 2026-08-16 absorption-path repair), not a policy difference. That
    comparison is reported as ``published_reproduction`` and is the gate to
    read before any battery verdict — an arm that reproduces the published
    per-seed values exactly means the published means are still usable; one
    that does not means they are stale by the amount shown.

    Both arms use the *same* ``battery_tasks`` seed arguments as
    ``stage_battery`` (``opponent_seed = 13 + index``, ``seed = 100 + index``,
    identical hand counts), so every arm is seed-paired with the candidate by
    construction.

    The stage is candidate-independent: its fragment can be copied between
    workdirs and reused across seeds of the same candidate family.
    """

    seeds = tuple(range(args.battery_seeds))
    champion_spec = _PolicySpec(
        label=NOISE_FLOOR_CHAMPION_LABEL,
        kind="heuristic",
        equity_trials=args.equity_trials,
    )
    arms = [champion_spec, incumbent_spec]
    plans = {
        arm.label: battery_tasks(
            arm,
            seeds=seeds,
            scale=args.scale,
            stack=args.starting_stack,
            reset_stacks=False,
        )
        for arm in arms
    }
    tasks = [task for plan in plans.values() for _, _, task in plan]
    results = _run_tasks(tasks, _gauntlet_workers(args.workers, len(tasks)))
    _assert_chip_conservation(results, "champion")

    report: dict[str, Any] = {
        "noise_floor_champion_label": champion_spec.label,
        "incumbent_label": incumbent_spec.label,
        "seed_count": len(seeds),
        "seed_pairing": "battery_tasks seeds are policy-independent by construction",
        "arms": {},
    }
    cursor = 0
    for label, plan in plans.items():
        grouped: dict[str, tuple[int, list[MatchResult]]] = {}
        for battery, hands, _ in plan:
            entry = grouped.setdefault(battery, (hands, []))
            entry[1].append(results[cursor])
            cursor += 1
        report["arms"][label] = {
            battery: _battery_entry(hands, channel_results)
            for battery, (hands, channel_results) in grouped.items()
        }
    return report


def published_reproduction(
    champion: Mapping[str, Any] | None, noise_floor: Mapping[str, Any] | None
) -> dict:
    """Does the rebuilt noise-floor champion still post its published numbers?

    Same policy, same channels, same seeds — so this isolates the instrument.
    Reported per channel as the per-seed max absolute difference; anything
    non-zero means the published means in ``battery_comparisons`` describe a
    simulator that no longer exists.
    """

    if not champion or not noise_floor:
        return {}
    label = champion.get("noise_floor_champion_label")
    arm = champion.get("arms", {}).get(label)
    if not arm:
        return {}
    published_channels = noise_floor.get("battery_channels", {})
    out: dict[str, Any] = {
        "arm": label,
        "published_champion": noise_floor.get("champion"),
        "channels": {},
    }
    for channel, data in arm.items():
        published = published_channels.get(channel)
        if not published:
            continue
        fresh_seeds = data["seeds"]
        published_seeds = published.get("per_seed_bb_per_100", [])[: len(fresh_seeds)]
        entry: dict[str, Any] = {
            "fresh_bb_per_100": data["bb_per_100"],
            "published_mean_bb_per_100": published.get("mean_bb_per_100"),
            "published_mde_bb_per_100": published.get("mde_bb_per_100", {}).get(
                f"{len(fresh_seeds)}_seeds"
            ),
        }
        if len(published_seeds) == len(fresh_seeds):
            diffs = [
                round(fresh - old, 4)
                for fresh, old in zip(fresh_seeds, published_seeds, strict=True)
            ]
            entry["per_seed_diffs"] = diffs
            entry["max_abs_per_seed_diff"] = max(abs(diff) for diff in diffs)
            entry["reproduces_published"] = entry["max_abs_per_seed_diff"] == 0.0
        out["channels"][channel] = entry
    reproducing = [
        entry.get("reproduces_published") for entry in out["channels"].values()
    ]
    out["all_channels_reproduce"] = bool(reproducing) and all(reproducing)
    return out


def stage_p3(
    v8_spec: V8PolicySpec, incumbent_spec: _PolicySpec, args: argparse.Namespace
) -> dict:
    """The held-out P3 channel: candidate, incumbent, and every trivial floor.

    ``V8_DESIGN.md`` §6.7 — "P3 holds out before it trains anything: it enters
    the batteries as a gate first."  The trivial-baseline floor applies to a
    new channel exactly as it does to the old ones, so the floors run here on
    the same seeds and the same hands as the real arms.
    """

    arms: list[Any] = [v8_spec, incumbent_spec, *(TrivialSpec(m) for m in TRIVIAL_MODES)]
    # Seed count defaults to --battery-seeds (unchanged behaviour) but can be
    # raised on its own: the published vs-p3 MDE is derived at 16 seeds, and
    # scoring against it at 8 would compare an effect to the wrong threshold.
    seed_count = getattr(args, "p3_seeds", None) or args.battery_seeds
    seeds = tuple(range(seed_count))
    plans = {
        arm.label: channel_battery_plan(
            arm,
            (P3_CHANNEL,),
            seeds=seeds,
            scale=args.scale,
            stack=args.starting_stack,
            fit_path=args.p3_fit,
        )
        for arm in arms
    }
    tasks = [task for plan in plans.values() for _, _, task in plan]
    outcomes = _run_tasks(tasks, _gauntlet_workers(args.workers, len(tasks)))
    if not all(outcome.chips_conserved for outcome in outcomes):
        raise AssertionError("p3: chip conservation violated")
    report: dict[str, Any] = {
        "channel": P3_CHANNEL,
        "fit": str(args.p3_fit),
        "seed_count": len(seeds),
        "arms": {},
    }
    cursor = 0
    for label, plan in plans.items():
        hands = plan[0][1]
        results = [outcomes[cursor + i].result for i in range(len(plan))]
        cursor += len(plan)
        report["arms"][label] = _battery_entry(hands, results)
    reference = report["arms"][v8_spec.label]["seeds"]
    report["paired_vs_candidate"] = {
        label: paired_stats(
            [value - ref for value, ref in zip(entry["seeds"], reference, strict=True)]
        )
        for label, entry in report["arms"].items()
        if label != v8_spec.label and len(entry["seeds"]) >= 2
    }
    return report


def stage_duel(
    v8_spec: V8PolicySpec, incumbent_spec: _PolicySpec, args: argparse.Namespace
) -> dict:
    tasks = duel_tasks(
        v8_spec,
        incumbent_spec,
        seeds=tuple(range(args.duel_seeds)),
        scale=args.scale,
        stack=args.starting_stack,
        reset_stacks=False,
    )
    results = _run_tasks(tasks, _gauntlet_workers(args.workers, len(tasks)))
    _assert_chip_conservation(results, "duel")
    _assert_duel_zero_sum(results, v8_spec.label, incumbent_spec.label, "duel")
    entry = duel_entry(v8_spec.label, incumbent_spec.label, tasks[0].hands, results)
    paired = entry.get("paired", {})
    mean = paired.get("mean")
    sd = paired.get("sd")
    count = len(paired.get("diffs", []))
    empirical_mde = (
        round(2.0 * sd / math.sqrt(count), 2) if sd is not None and count else None
    )
    resolvable = max(
        KNOWN_SEED_SPREAD_BB_PER_100,
        empirical_mde if empirical_mde is not None else 0.0,
    )
    verdict = "UNRESOLVED"
    if mean is not None and abs(mean) >= resolvable:
        verdict = "v8 wins" if mean > 0 else "incumbent wins"
    return {
        "opponent": incumbent_spec.label,
        "seeds": count,
        "report": entry,
        "empirical_mde_bb_per_100": empirical_mde,
        "known_seed_spread_bb_per_100": KNOWN_SEED_SPREAD_BB_PER_100,
        "resolvable_spread_bb_per_100": round(resolvable, 2),
        "verdict": verdict,
        "verdict_rule": (
            "a margin below the resolvable spread (max of the known 16.78 "
            "BB/100 seed spread and 2·sd/√n) is UNRESOLVED, never a win/loss"
        ),
    }


def stage_strength(
    v8_spec: V8PolicySpec, incumbent_spec: _PolicySpec, args: argparse.Namespace
) -> dict:
    hu_hands = max(50, round(args.strength_hu_hands * args.scale))
    lineup_hands = max(50, round(args.strength_lineup_hands * args.scale))
    tasks = [
        StrengthTask(v8_spec, "vs-median-hu", hu_hands, 400, 41, args.starting_stack),
        StrengthTask(
            v8_spec, "five-max-lineup", lineup_hands, 500, 43, args.starting_stack
        ),
        StrengthTask(
            incumbent_spec, "vs-median-hu", hu_hands, 400, 41, args.starting_stack
        ),
        StrengthTask(
            incumbent_spec,
            "five-max-lineup",
            lineup_hands,
            500,
            43,
            args.starting_stack,
        ),
    ]
    outcomes = _run_tasks(tasks, _gauntlet_workers(min(args.workers, 4), len(tasks)))
    by_policy: dict[str, list[tuple[str, str, float]]] = {}
    for task, records in zip(tasks, outcomes, strict=True):
        by_policy.setdefault(task.spec.label, []).extend(records)
    return {
        "metric": "strength_metric.strength_percentile (canonical percentile scale)",
        "arenas": {
            "vs-median-hu": {"hands": hu_hands, "seed": 400, "opponent_seed": 41},
            "five-max-lineup": {
                "hands": lineup_hands,
                "seed": 500,
                "opponent_seed": 43,
            },
        },
        "caveat": (
            "every battery opponent is card-blind (DECISIONS §5): separation "
            "measured here says what the policy does, not what a real field "
            "would let it get away with. The field benchmark on THIS metric "
            "is +0.386 (artifacts/evaluations/"
            "field-separation-canonical-2026-08-16.json, 8,986 field "
            "decisions). The older +0.150 was a different scale and is "
            "superseded -- do not compare canonical numbers to it"
        ),
        "policies": {
            label: separation_report(records)
            for label, records in by_policy.items()
        },
    }


# ---------------------------------------------------------------------------
# Comparisons and assembly
# ---------------------------------------------------------------------------


def battery_comparisons(
    battery: Mapping[str, Any],
    trivial: Mapping[str, Any] | None,
    noise_floor: Mapping[str, Any] | None,
    champion: Mapping[str, Any] | None = None,
) -> dict:
    """Score the candidate's battery channels against every available reference.

    Per channel: the published champion mean and per-seed pairing (same
    match seeds by construction), the published MDE at the seed count used,
    and paired per-seed differences against each trivial floor (again, same
    seeds by construction).

    When ``champion`` (the ``stage_champion`` fragment) is supplied, each of
    its freshly rebuilt arms is paired seed-for-seed against the candidate:
    the noise-floor champion under ``fresh_champion_*`` and the duel incumbent
    under ``fresh_incumbent_*``. Those are the references to read. The
    published columns are kept beside them, never overwritten, so the
    staleness of the published means is visible as a number rather than
    asserted.
    """

    seed_count = None
    comparisons: dict[str, Any] = {}
    channels = battery.get("channels", {})
    floor_channels = (trivial or {}).get("floors", {})
    noise_channels = (noise_floor or {}).get("battery_channels", {})
    champion = champion or {}
    fresh_arms = champion.get("arms", {})
    arm_prefixes = {
        champion.get("noise_floor_champion_label"): "fresh_champion",
        champion.get("incumbent_label"): "fresh_incumbent",
    }
    for channel, data in channels.items():
        seeds = data["seeds"]
        seed_count = len(seeds)
        entry: dict[str, Any] = {"bb_per_100": data["bb_per_100"]}
        for arm_label, prefix in arm_prefixes.items():
            if not arm_label:
                continue
            fresh = fresh_arms.get(arm_label, {}).get(channel)
            if not fresh:
                continue
            fresh_seeds = fresh["seeds"][:seed_count]
            entry[f"{prefix}_label"] = arm_label
            entry[f"{prefix}_bb_per_100"] = fresh["bb_per_100"]
            if len(fresh_seeds) == seed_count and seed_count >= 2:
                paired = paired_stats(
                    [
                        value - reference
                        for value, reference in zip(seeds, fresh_seeds, strict=True)
                    ]
                )
                entry[f"paired_vs_{prefix}"] = paired
                sd = paired.get("sd")
                if sd is not None and seed_count:
                    entry[f"{prefix}_empirical_mde_bb_per_100"] = round(
                        2.0 * sd / math.sqrt(seed_count), 2
                    )
        noise = noise_channels.get(channel)
        if noise:
            mde_key = f"{seed_count}_seeds"
            entry["published_mde_bb_per_100"] = noise.get("mde_bb_per_100", {}).get(
                mde_key
            )
            entry["champion_mean_bb_per_100"] = noise.get("mean_bb_per_100")
            champion_seeds = noise.get("per_seed_bb_per_100", [])[:seed_count]
            if len(champion_seeds) == seed_count and seed_count >= 2:
                entry["paired_vs_champion"] = paired_stats(
                    [
                        value - reference
                        for value, reference in zip(seeds, champion_seeds, strict=True)
                    ]
                )
        floors: dict[str, Any] = {}
        for mode, floor in floor_channels.items():
            floor_data = floor.get(channel)
            if not floor_data:
                continue
            floor_seeds = floor_data["seeds"][:seed_count]
            comparison: dict[str, Any] = {
                "floor_bb_per_100": floor_data["bb_per_100"],
                "beats_floor_mean": data["bb_per_100"] > floor_data["bb_per_100"],
            }
            if len(floor_seeds) == seed_count and seed_count >= 2:
                comparison["paired_vs_floor"] = paired_stats(
                    [
                        value - reference
                        for value, reference in zip(seeds, floor_seeds, strict=True)
                    ]
                )
            floors[mode] = comparison
        if floors:
            entry["trivial_floors"] = floors
        comparisons[channel] = entry
    return {"seed_count": seed_count, "channels": comparisons}


def _write_fragment(workdir: Path, stage: str, fragment: Mapping[str, Any]) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / f"{stage}.json").write_text(
        json.dumps(fragment, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_fragment(workdir: Path, stage: str) -> dict | None:
    path = workdir / f"{stage}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assemble_report(
    workdir: Path,
    *,
    candidate_manifest: Mapping[str, Any],
    candidate_label: str,
    incumbent_label: str,
    noise_floor: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> dict:
    fragments = {stage: _read_fragment(workdir, stage) for stage in _STAGES}
    training = candidate_manifest.get("training", {})
    report: dict[str, Any] = {
        "study": "candidate-v8-0001-gauntlet",
        "candidate": candidate_label,
        "incumbent": incumbent_label,
        "config": dict(config),
        "trainer_context": {
            "init_seed_exported": training.get("init_seed"),
            "init_seeds_evaluated": training.get("init_seeds_evaluated"),
            "seed_selection": training.get("seed_selection"),
            "note": (
                "only the exported init seed has an artifact; the sibling "
                "seeds were not persisted, so this gauntlet covers one of "
                "the three trained seeds (V8_DESIGN §6.1 asks for all three)"
            ),
        },
        "phase_a_calibration": candidate_manifest.get("evaluation", {}).get(
            "calibration"
        ),
        "nullcheck": fragments["nullcheck"],
        "trivial_floors": fragments["trivial"],
        "batteries": fragments["battery"],
        "champion_batteries": fragments["champion"],
        "published_reproduction": published_reproduction(
            fragments["champion"], noise_floor
        ),
        "p3_channel": fragments["p3"],
        "duel": fragments["duel"],
        "strength_separation": fragments["strength"],
    }
    if fragments["battery"]:
        report["battery_comparisons"] = battery_comparisons(
            fragments["battery"],
            fragments["trivial"],
            noise_floor,
            fragments["champion"],
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--incumbent", default=DEFAULT_INCUMBENT)
    parser.add_argument("--noise-floor", default=DEFAULT_NOISE_FLOOR)
    parser.add_argument(
        "--p3-fit",
        default=str(P3_DEFAULT_FIT_PATH),
        help="fitted strength-aware opponent artifact for the vs-p3 channel",
    )
    parser.add_argument(
        "--stages",
        default=",".join(_STAGES),
        help=f"comma-separated subset of {_STAGES}; skipped stages reuse any "
        "fragment already in --workdir",
    )
    parser.add_argument("--battery-seeds", type=int, default=8)
    parser.add_argument(
        "--p3-seeds",
        type=int,
        default=None,
        help="seeds for the vs-p3 channel (default: --battery-seeds); the "
        "published vs-p3 MDE is derived at 16",
    )
    parser.add_argument("--duel-seeds", type=int, default=16)
    parser.add_argument("--trivial-seeds", type=int, default=8)
    parser.add_argument("--null-seeds", type=int, default=2)
    parser.add_argument(
        "--null-scale",
        type=float,
        default=0.05,
        help="hands scale for the self-duel wiring check (the true effect "
        "is zero by construction at any length)",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--equity-trials", type=int, default=80)
    parser.add_argument("--starting-stack", type=int, default=6_000)
    parser.add_argument("--strength-hu-hands", type=int, default=2_000)
    parser.add_argument("--strength-lineup-hands", type=int, default=1_200)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output", default="artifacts/evaluations/candidate-v8-0001-gauntlet.json"
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="stage-fragment directory (default: <output>.stages)",
    )
    args = parser.parse_args(argv)

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = sorted(set(stages) - set(_STAGES))
    if unknown:
        parser.error(f"unknown stages: {unknown}")

    output = Path(args.output)
    workdir = Path(args.workdir) if args.workdir else output.with_suffix(".stages")

    # Load both policies once, fail-loud, for labels and validation; the
    # workers rebuild from the manifests independently.
    candidate_manifest = json.loads(
        Path(args.candidate).read_text(encoding="utf-8")
    )
    v8_policy = load_policy_v8(args.candidate, equity_trials=args.equity_trials)
    v8_spec = V8PolicySpec(
        label=str(v8_policy.policy_version),
        manifest=args.candidate,
        equity_trials=args.equity_trials,
    )
    del v8_policy
    from devfun_poker_playground.learned_policy import load_policy

    incumbent_label = str(
        load_policy(args.incumbent, equity_trials=args.equity_trials).policy_version
    )
    incumbent_spec = _PolicySpec(
        label=incumbent_label,
        kind="candidate",
        equity_trials=args.equity_trials,
        manifest=args.incumbent,
    )
    noise_floor = None
    noise_path = Path(args.noise_floor)
    if noise_path.exists():
        noise_floor = json.loads(noise_path.read_text(encoding="utf-8"))

    stage_runners = {
        "nullcheck": lambda: stage_nullcheck(v8_spec, args),
        "trivial": lambda: stage_trivial(args),
        "battery": lambda: stage_battery(v8_spec, args),
        "champion": lambda: stage_champion(incumbent_spec, args),
        "p3": lambda: stage_p3(v8_spec, incumbent_spec, args),
        "duel": lambda: stage_duel(v8_spec, incumbent_spec, args),
        "strength": lambda: stage_strength(v8_spec, incumbent_spec, args),
    }
    timings: dict[str, float] = {}
    for stage in _STAGES:
        if stage not in stages:
            continue
        print(f"[evaluate_v8] stage {stage} starting", flush=True)
        started = time.monotonic()
        fragment = stage_runners[stage]()
        timings[stage] = round(time.monotonic() - started, 1)
        _write_fragment(workdir, stage, fragment)
        print(
            f"[evaluate_v8] stage {stage} done in {timings[stage]}s", flush=True
        )

    config = {
        "candidate_manifest": args.candidate,
        "incumbent_manifest": args.incumbent,
        "p3_fit": args.p3_fit,
        "battery_seeds": args.battery_seeds,
        "p3_seeds": args.p3_seeds or args.battery_seeds,
        "duel_seeds": args.duel_seeds,
        "trivial_seeds": args.trivial_seeds,
        "scale": args.scale,
        "equity_trials": args.equity_trials,
        "starting_stack": args.starting_stack,
        "stacks": "carry-over-sessions",
        "workers": args.workers,
        "stage_elapsed_seconds": timings,
        "stages_run": stages,
    }
    report = assemble_report(
        workdir,
        candidate_manifest=candidate_manifest,
        candidate_label=v8_spec.label,
        incumbent_label=incumbent_label,
        noise_floor=noise_floor,
        config=config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"[evaluate_v8] report written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
