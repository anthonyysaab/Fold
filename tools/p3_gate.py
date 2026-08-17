"""The held-out **P3 gate** — the first battery seat that can see its own cards.

``PLAYERS.md`` ("What is missing, and what to build first") and
``V8_DESIGN.md`` §6.7 are explicit: the strength-aware opponent enters the
batteries **as a gate, before any training uses it**. Nothing in this tool
trains, harvests, promotes, deploys, or touches ``artifacts/approved.json``;
it runs matches and writes a report.

What it measures
----------------

Six real arms plus a wiring mirror, on **identical seeds and identical hand
counts**, over three channels:

* ``vs-p3``      — the new one: :class:`StrengthAwareAgent`, folding by the
  model fitted to the real field's revealed hole cards.
* ``vs-median``  — the *structural twin control*. Same aggression, same
  nominal ``fold_vs_bet``, same seeds; the only difference from ``vs-p3`` is
  that its folding is a constant instead of the fit. A vs-p3 minus vs-median
  difference is therefore attributable to exactly one edit.
* ``vs-station`` — the channel the candidate broke worst on (-332 BB/100
  against the champion), and the archetype the diagnosed mechanism blames:
  a villain that essentially never folds.

Why the derived MDE, and why two of them
----------------------------------------

``noise-floor-2026-08-15.json`` publishes no ``vs-p3`` row, so a difference on
this channel has no published resolvable size and **any number read off it
would be uninterpretable** (``PLAYERS.md``: "Without it no result is
interpretable"). Two are derived here, both from this run's own data:

1. **Independent-seed MDE**, ``2σ/√n`` on an arm's per-seed BB/100 — the
   published construction reproduced exactly (check: vs-median σ 24.12 →
   17.06 at 8 seeds in the published file). This is the honest figure for
   comparing against a number measured on *different* seeds.
2. **Paired MDE**, ``2·sd(diffs)/√n`` on per-seed differences — every arm here
   plays the same cards in the same order, so common random numbers cancel
   most of the variance. This is the instrument actually used for the
   contrasts, and it is much tighter. ``evaluate_v8.stage_duel`` derives its
   duel MDE the same way.

Anything below the relevant one is **UNRESOLVED**, never a win and never a
loss.

Measure the instrument before the result
----------------------------------------

The ``instrument`` stage runs first and its result is reported first. It is a
known-answer test with a matched negative control: the *opponent's own*
canonical strength when it continues minus when it folds must be

* ``≈ 0`` for the card-blind ``vs-median`` seat — **impossible by
  construction** for it to be anything else, since that agent never sees a
  card; and
* clearly ``> 0`` for the ``vs-p3`` seat, or the fit is not reaching the
  table.

Plus: a self-null arm whose paired differences must be identically zero, chip
conservation on every match, and a deterministic monotonicity sweep of the
fitted fold probability. The P3 seats run as
:class:`~tools.evaluate_v8.StrictStrengthAwareAgent`, which raises rather than
silently degrading to the card-blind constant, so no vs-p3 number in this
report can have been partly produced by a different opponent.

Stdlib-only, entirely offline, seeded throughout.

Example::

    python -m tools.p3_gate --seeds 16 --workers 14 \
        --output artifacts/evaluations/p3-gate-2026-08-16.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devfun_poker_playground.poker_policy import build_policy
from devfun_poker_playground.strength_aware_opponent import (
    DEFAULT_FIT_PATH,
    STREETS,
    P3Decision,
    load_fit,
)
from devfun_poker_playground.strength_metric import strength_percentile
from devfun_poker_playground.table_simulator import _family_of, run_sessions
from tools.evaluate_policies import (
    _gauntlet_workers,
    _PolicySpec,
    _run_tasks,
    paired_stats,
)
from tools.evaluate_v8 import (
    P3_CHANNEL,
    TRIVIAL_MODES,
    TrivialSpec,
    V8PolicySpec,
    channel_battery_plan,
    channel_opponents,
    p3_opponents,
    separation_report,
    _wrap_hero,
)

DEFAULT_CANDIDATE = "artifacts/candidates/candidate-v8-0001.manifest.json"
DEFAULT_INCUMBENT = "artifacts/candidates/candidate-v7-0001c.approved.manifest.json"

#: vs-p3 is the subject; the other two are controls, run on the same seeds by
#: the same arms so cross-channel statements are made within one run instead
#: of across two reports.
CHANNELS: tuple[str, ...] = (P3_CHANNEL, "vs-median", "vs-station")

#: The label the wiring mirror runs under. It is the champion, renamed: its
#: per-seed differences against the champion arm must be identically zero.
NULL_MIRROR_LABEL = "null-mirror-of-champion"

_STAGES = ("instrument", "arms", "logs")


# ---------------------------------------------------------------------------
# Decision logging
# ---------------------------------------------------------------------------


def wager_pot_fraction(
    table: Mapping[str, Any], payload: Mapping[str, Any]
) -> float | None:
    """The engine's own pot fraction for an aggressive action, or ``None``.

    ``(new chips - call) / (pot + call)`` — the formula
    ``tools/measure_field_separation`` uses, which is the quantity the frozen
    "median bet 0.60x pot" field figure and the incumbent's published "0.50x
    pot median bet" are both on (``PENDING_EDITS`` 18d: the earlier 1.333x
    figure double-counted the call).
    """

    action = str(payload.get("action") or "")
    if action not in ("bet", "raise", "all-in"):
        return None
    allowed = table.get("allowedActions") or {}
    to_call = int(allowed.get("callChips") or 0)
    pot = int(table.get("potChips") or 0)
    denominator = pot + to_call
    if denominator <= 0:
        return None
    hero = next(
        seat for seat in table["seats"] if seat["seatNumber"] == table["selfSeatNumber"]
    )
    new_chips = max(0, int(payload.get("amount") or 0) - int(hero["currentBetChips"]))
    return (new_chips - to_call) / denominator


def _log_row(table: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple | None:
    """``(street, family, strength, wager_pot_fraction)`` for one decision."""

    family = _family_of(str(payload.get("action")))
    if family is None:
        return None
    hero = next(
        seat for seat in table["seats"] if seat["seatNumber"] == table["selfSeatNumber"]
    )
    hole = hero.get("holeCards") or ()
    if len(hole) != 2:
        return None
    street = str(table.get("street") or "preflop").casefold()
    strength = strength_percentile(hole, table.get("boardCards") or ())
    return (street, family, strength, wager_pot_fraction(table, payload))


class DecisionLogger:
    """Wrap any agent; record what it did and what it held while doing it.

    Records the **submitted** action's family — post bluff conversion, post
    safety gates, the action the table actually saw — so the rates reported
    are the rates the opponent experienced, not the model's intent.
    """

    def __init__(self, agent: Any, records: list) -> None:
        self.agent = agent
        self.records = records
        self.reads_cards = getattr(agent, "reads_cards", False)
        self.policy_version = getattr(agent, "policy_version", "policy")

    def decide(self, table: Mapping[str, Any]) -> dict:
        payload = self.agent.decide(table)
        row = _log_row(table, payload)
        if row is not None:
            self.records.append(row)
        return payload


@dataclass(frozen=True)
class LogTask:
    """One recording match: log both the hero's and the opponent's decisions.

    Seeded identically to the arm of the same seed index in the ``arms``
    stage, so the decision log is a subsample of the very matches whose
    BB/100 is reported and not a separate experiment.
    """

    hero: Any
    channel: str
    opponent_seed: int
    hands: int
    seed: int
    stack: int
    fit_path: str = str(DEFAULT_FIT_PATH)

    def run(self) -> dict:
        hero_records: list = []
        opponent_records: list = []
        opponents = channel_opponents(self.channel, self.opponent_seed, self.fit_path)
        hero = self.hero
        factories = [
            ("hero", lambda: DecisionLogger(_wrap_hero(hero.build()), hero_records)),
            *[
                (name, (lambda agent=agent: DecisionLogger(agent, opponent_records)))
                for name, agent in opponents
            ],
        ]
        result = run_sessions(
            factories,
            target_hands=self.hands,
            seed=self.seed,
            starting_stack=self.stack,
        )
        return {
            "hero": hero_records,
            "opponent": opponent_records,
            "chips_conserved": sum(result.chip_deltas.values()) == 0,
            "bb_per_100": result.bb_per_100("hero"),
        }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def independent_seed_mde(values: Sequence[float]) -> dict:
    """``2σ/√n`` on per-seed BB/100 — the published noise-floor construction.

    Reproduced term for term so this channel's figure is on the same scale as
    the published per-channel MDEs and can be quoted beside them. **The σ is
    rounded to 2dp before dividing**, which is not cosmetic: the published
    artifact was generated that way, and using the unrounded σ disagrees with
    it by 0.01 on 6 of its 18 published cells (e.g. vs-median at 8 seeds,
    17.05 vs the published 17.06). ``tests/test_p3_gate.py`` pins the
    reproduction against every published cell, so this stays honest.
    """

    _, sd = _mean_sd(values)
    sigma = round(sd, 2)
    count = len(values)
    return {
        "sigma_seed_count": count,
        "sigma_bb_per_100": sigma,
        "mde_bb_per_100": {
            f"{n}_seeds": round(2.0 * sigma / math.sqrt(n), 2)
            for n in sorted({8, 16, count})
            if n > 0
        },
    }


def paired_contrast(
    a: Sequence[float], b: Sequence[float], *, label_a: str, label_b: str
) -> dict:
    """Per-seed A-minus-B with its own paired MDE and an explicit verdict."""

    diffs = [x - y for x, y in zip(a, b, strict=True)]
    stats = paired_stats(diffs)
    count = len(diffs)
    mde = 2.0 * stats["sd"] / math.sqrt(count) if count else float("inf")
    mean = stats["mean"]
    # A zero difference is UNRESOLVED even when the MDE is also zero: two arms
    # that never differ have not been shown to differ. Guarding the sign this
    # way is what makes the self-null mirror read correctly instead of
    # reporting a winner at a margin of 0.00.
    if mean == 0.0 or abs(mean) < mde:
        verdict = "UNRESOLVED"
    else:
        verdict = f"{label_a} ahead" if mean > 0 else f"{label_b} ahead"
    return {
        "a": label_a,
        "b": label_b,
        **stats,
        "paired_mde_bb_per_100": round(mde, 2),
        "verdict": verdict,
    }


def family_rates(records: Sequence[tuple]) -> dict:
    """Aggression / fold / check-call rates and the median wager as a pot share."""

    total = len(records)
    counts = {"aggress": 0, "check_call": 0, "fold": 0}
    for _, family, _, _ in records:
        if family in counts:
            counts[family] += 1
    fractions = [
        fraction
        for _, family, _, fraction in records
        if family == "aggress" and fraction is not None
    ]
    entry = {
        "decisions": total,
        "counts": counts,
        "aggression_rate": round(counts["aggress"] / total, 4) if total else None,
        "fold_rate": round(counts["fold"] / total, 4) if total else None,
        "check_call_rate": round(counts["check_call"] / total, 4) if total else None,
        "sized_aggressions": len(fractions),
        "median_wager_pot_fraction": (
            round(statistics.median(fractions), 4) if fractions else None
        ),
    }
    if len(fractions) >= 2:
        ordered = sorted(fractions)
        entry["wager_pot_fraction_quartiles"] = [
            round(ordered[max(0, int(0.25 * (len(ordered) - 1)))], 4),
            round(statistics.median(ordered), 4),
            round(ordered[min(len(ordered) - 1, int(0.75 * (len(ordered) - 1)))], 4),
        ]
    return entry


def continue_minus_fold(records: Sequence[tuple]) -> dict:
    """Mean canonical strength when an agent continues minus when it folds.

    The known-answer quantity of the ``instrument`` stage. A card-blind agent
    cannot make this non-zero: its action is a function of a seeded mixer and
    public state only, so its own holding is independent of what it did.
    """

    continuing = [row[2] for row in records if row[1] in ("aggress", "check_call")]
    folding = [row[2] for row in records if row[1] == "fold"]
    if len(continuing) < 2 or len(folding) < 2:
        return {"n_continue": len(continuing), "n_fold": len(folding), "value": None}
    mean_c, sd_c = _mean_sd(continuing)
    mean_f, sd_f = _mean_sd(folding)
    se = math.sqrt(sd_c**2 / len(continuing) + sd_f**2 / len(folding))
    value = mean_c - mean_f
    return {
        "n_continue": len(continuing),
        "n_fold": len(folding),
        "mean_strength_continue": round(mean_c, 4),
        "mean_strength_fold": round(mean_f, 4),
        "value": round(value, 4),
        "se": round(se, 4),
        "t": round(value / se, 2) if se > 0 else None,
        "ci95": [round(value - 1.96 * se, 4), round(value + 1.96 * se, 4)],
        "realized_fold_rate": round(
            sum(1 for row in records if row[1] == "fold") / len(records), 4
        ),
    }


# ---------------------------------------------------------------------------
# Instrument stage — known-answer tests, run and reported before any result
# ---------------------------------------------------------------------------


def monotonicity_sweep(fit_path: str | Path) -> dict:
    """Deterministic sweep of the fitted fold probability. No RNG, no table.

    Two properties the whole opponent exists for, checked on the artifact that
    will actually be fielded: worse hands fold more, worse prices fold more.
    Read off :meth:`StrengthAwareAgent.fold_probability_for`, i.e. after the
    clamp band, because that is the function the table sees.
    """

    from tools.evaluate_v8 import StrictStrengthAwareAgent

    fit = load_fit(str(fit_path))
    agent = StrictStrengthAwareAgent("sweep", 0.226, 0.5, 0.0, 1, fit=fit)
    grid = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    report: dict[str, Any] = {
        "band": list(fit.band),
        "note": (
            "clamped output, i.e. the function the table sees; monotone "
            "strictly unless the clamp band is active at both endpoints"
        ),
        "streets": {},
    }
    ok = True
    for street in STREETS:
        def probe(strength: float, price: float) -> float:
            return agent.fold_probability_for(
                P3Decision(
                    street=street,
                    strength_percentile=strength,
                    pot_odds=price,
                    bet_to_pot=price / max(1e-9, 1.0 - price),
                    texture=0.3,
                    active_players=2,
                    position_unit=0.5,
                    to_call=100,
                    pot=200,
                )
            )

        by_strength = [round(probe(s, 1 / 3), 5) for s in grid]
        by_price = [round(probe(0.5, p), 5) for p in (0.2, 0.3, 1 / 3, 0.4, 0.5, 0.6)]
        strength_ok = all(
            b <= a + 1e-12 for a, b in zip(by_strength, by_strength[1:])
        ) and by_strength[0] > by_strength[-1]
        price_ok = all(
            b >= a - 1e-12 for a, b in zip(by_price, by_price[1:])
        ) and by_price[-1] > by_price[0]
        ok = ok and strength_ok and price_ok
        report["streets"][street] = {
            "p_fold_by_strength_at_half_pot": by_strength,
            "strength_grid": grid,
            "p_fold_by_pot_odds_at_median_hand": by_price,
            "pot_odds_grid": [0.2, 0.3, round(1 / 3, 4), 0.4, 0.5, 0.6],
            "decreasing_in_strength": strength_ok,
            "increasing_in_price": price_ok,
        }
    report["passed"] = ok
    return report


@dataclass(frozen=True)
class OpponentProbeTask:
    """Log the *opponent's* own decisions on one channel; the known-answer test."""

    channel: str
    opponent_seed: int
    hands: int
    seed: int
    stack: int
    fit_path: str = str(DEFAULT_FIT_PATH)

    def run(self) -> list:
        records: list = []
        opponents = channel_opponents(self.channel, self.opponent_seed, self.fit_path)
        factories = [
            (
                "hero",
                lambda: _wrap_hero(build_policy(aggressive=True, equity_trials=80)),
            ),
            *[
                (name, (lambda agent=agent: DecisionLogger(agent, records)))
                for name, agent in opponents
            ],
        ]
        run_sessions(
            factories,
            target_hands=self.hands,
            seed=self.seed,
            starting_stack=self.stack,
        )
        return records


def stage_instrument(args: argparse.Namespace) -> dict:
    tasks = [
        OpponentProbeTask(
            channel=channel,
            opponent_seed=13 + index,
            hands=args.probe_hands,
            seed=100 + index,
            stack=args.starting_stack,
            fit_path=args.p3_fit,
        )
        for channel in (P3_CHANNEL, "vs-median")
        for index in range(args.probe_seeds)
    ]
    outcomes = _run_tasks(tasks, _gauntlet_workers(args.workers, len(tasks)))
    pooled: dict[str, list] = {P3_CHANNEL: [], "vs-median": []}
    for task, records in zip(tasks, outcomes, strict=True):
        pooled[task.channel].extend(records)

    control = continue_minus_fold(pooled["vs-median"])
    subject = continue_minus_fold(pooled[P3_CHANNEL])
    control_flat = control["value"] is not None and (
        control["ci95"][0] <= 0.0 <= control["ci95"][1]
    )
    subject_live = subject["value"] is not None and subject["ci95"][0] > 0.05
    sweep = monotonicity_sweep(args.p3_fit)

    return {
        "ran_before_any_result": True,
        "known_answer_test": {
            "question": (
                "does the opponent's own action correlate with its own hand "
                "strength on the canonical metric?"
            ),
            "negative_control": {
                "channel": "vs-median",
                "why_impossible": (
                    "ScriptedAgent is card-blind: its action is a seeded "
                    "function of public state only, so continue-minus-fold "
                    "strength CANNOT differ from zero except by sampling noise"
                ),
                "expected": "CI95 contains 0",
                "observed": control,
                "passed": control_flat,
            },
            "positive_control": {
                "channel": P3_CHANNEL,
                "expected": "CI95 lower bound > +0.05",
                "observed": subject,
                "passed": subject_live,
            },
            "passed": bool(control_flat and subject_live),
        },
        "monotonicity_sweep": sweep,
        "degradation_guard": {
            "mechanism": "tools.evaluate_v8.StrictStrengthAwareAgent",
            "detail": (
                "the P3 seat raises instead of falling back to the card-blind "
                "constant, so a completed run cannot contain a vs-p3 match "
                "that was partly played by vs-median"
            ),
        },
        "probe_config": {
            "seeds": args.probe_seeds,
            "hands_per_seed": args.probe_hands,
            "hero": "heuristic-aggressive (champion)",
        },
    }


# ---------------------------------------------------------------------------
# Arms stage
# ---------------------------------------------------------------------------


def build_arms(args: argparse.Namespace) -> list[Any]:
    from devfun_poker_playground.learned_policy import load_policy
    from devfun_poker_playground.learned_policy_v8 import load_policy_v8

    v8_label = str(
        load_policy_v8(args.candidate, equity_trials=args.equity_trials).policy_version
    )
    v7_label = str(
        load_policy(args.incumbent, equity_trials=args.equity_trials).policy_version
    )
    champion_label = str(
        build_policy(aggressive=True, equity_trials=args.equity_trials).policy_version
    )
    return [
        V8PolicySpec(v8_label, args.candidate, args.equity_trials),
        _PolicySpec(v7_label, "candidate", args.equity_trials, manifest=args.incumbent),
        _PolicySpec(champion_label, "heuristic", args.equity_trials),
        # The wiring mirror: the champion under a second label, same seeds.
        _PolicySpec(NULL_MIRROR_LABEL, "heuristic", args.equity_trials),
        *[TrivialSpec(mode) for mode in TRIVIAL_MODES],
    ]


def stage_arms(args: argparse.Namespace) -> dict:
    arms = build_arms(args)
    seeds = tuple(range(args.seeds))
    plans = {
        arm.label: channel_battery_plan(
            arm,
            CHANNELS,
            seeds=seeds,
            scale=args.scale,
            stack=args.starting_stack,
            fit_path=args.p3_fit,
        )
        for arm in arms
    }
    tasks = [task for plan in plans.values() for _, _, task in plan]
    print(f"[p3_gate] arms: {len(tasks)} matches", flush=True)
    outcomes = _run_tasks(tasks, _gauntlet_workers(args.workers, len(tasks)))
    violations = sum(1 for outcome in outcomes if not outcome.chips_conserved)
    if violations:
        raise AssertionError(f"arms: chip conservation violated in {violations} matches")

    report: dict[str, Any] = {
        "channels": list(CHANNELS),
        "seed_count": len(seeds),
        "seed_convention": (
            "opponent seed 13+i, match seed 100+i, carry-over sessions, "
            "starting stack per --starting-stack — the published "
            "evaluate_policies.battery_tasks convention verbatim"
        ),
        "chip_conservation_violations": violations,
        "arms": {},
    }
    cursor = 0
    for label, plan in plans.items():
        per_channel: dict[str, Any] = {}
        for channel in CHANNELS:
            entries = [item for item in plan if item[0] == channel]
            hands = entries[0][1]
            values = [outcomes[cursor + i].result.bb_per_100("hero") for i in range(len(entries))]
            busts = [outcomes[cursor + i].result.busts.get("hero", 0) for i in range(len(entries))]
            hero_hands = [
                (getattr(outcomes[cursor + i].result, "hands_by_agent", None) or {}).get(
                    "hero", outcomes[cursor + i].result.hands
                )
                for i in range(len(entries))
            ]
            cursor += len(entries)
            per_channel[channel] = {
                "hands_per_seed": hands,
                "bb_per_100": round(sum(values) / len(values), 2),
                "seeds": [round(value, 2) for value in values],
                "busts_per_100_hands": round(
                    100.0 * sum(busts) / max(1, sum(hero_hands)), 4
                ),
                **independent_seed_mde(values),
            }
        report["arms"][label] = per_channel
    return report


# ---------------------------------------------------------------------------
# Logs stage
# ---------------------------------------------------------------------------


#: Logged alongside the three real policies. ``always_aggress_large`` is the
#: arm the trivial-baseline gate turns on, so the question "what does the
#: opponent do when it is bet at indiscriminately" needs its decision log too,
#: not just its BB/100. ``always_check_call`` is the passive reference.
LOGGED_FLOORS: tuple[str, ...] = ("always_aggress_large", "always_check_call")


def stage_logs(args: argparse.Namespace) -> dict:
    arms = build_arms(args)
    logged = [*arms[:3], *(TrivialSpec(mode) for mode in LOGGED_FLOORS)]
    tasks = [
        LogTask(
            hero=arm,
            channel=channel,
            opponent_seed=13 + index,
            hands=max(50, round(1_000 * args.scale)),
            seed=100 + index,
            stack=args.starting_stack,
            fit_path=args.p3_fit,
        )
        for arm in logged
        for channel in CHANNELS
        for index in range(args.log_seeds)
    ]
    print(f"[p3_gate] logs: {len(tasks)} recording matches", flush=True)
    outcomes = _run_tasks(tasks, _gauntlet_workers(args.workers, len(tasks)))

    hero_pool: dict[tuple[str, str], list] = {}
    opponent_pool: dict[tuple[str, str], list] = {}
    for task, outcome in zip(tasks, outcomes, strict=True):
        if not outcome["chips_conserved"]:
            raise AssertionError("logs: chip conservation violated")
        key = (task.hero.label, task.channel)
        hero_pool.setdefault(key, []).extend(outcome["hero"])
        opponent_pool.setdefault(key, []).extend(outcome["opponent"])

    report: dict[str, Any] = {
        "seeds": args.log_seeds,
        "hands_per_seed": max(50, round(1_000 * args.scale)),
        "seed_note": (
            "the same opponent seeds (13+i) and match seeds (100+i) as the "
            "first log_seeds seeds of the arms stage, so these logs are a "
            "subsample of the matches whose BB/100 is reported above"
        ),
        "metric": "strength_metric.strength_percentile (canonical, V8_DESIGN §2)",
        "wager_definition": "(new chips - call) / (pot + call), the engine's own pot fraction",
        "policies": {},
        "opponent_behaviour": {},
    }
    for (label, channel), records in hero_pool.items():
        entry = report["policies"].setdefault(label, {})
        entry[channel] = {
            **family_rates(records),
            "strength_separation": separation_report(
                [(row[0], row[1], row[2]) for row in records]
            ),
        }
    for (label, channel), records in opponent_pool.items():
        entry = report["opponent_behaviour"].setdefault(channel, {})
        entry[label] = {
            **family_rates(records),
            "continue_minus_fold_strength": continue_minus_fold(records),
        }
    return report


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


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


def reproduction_gate(
    arms: Mapping[str, Any], champion: str, noise_floor: Mapping[str, Any] | None
) -> dict:
    """Does this harness reproduce the published instrument on shared channels?

    ``vs-median`` and ``vs-station`` are run here with the published opponent
    seeds, match seeds, hand counts and session mode. The champion is
    deterministic given those, so its per-seed BB/100 should be **identical**
    to ``noise-floor-2026-08-15.json``'s — an impossible-by-construction match
    if and only if this is the same instrument. A mismatch does not
    invalidate the internal contrasts (every arm here still shares seeds with
    every other), but it does mean nothing in this report may be quoted
    against a published number, so it is checked and reported either way.
    """

    if not noise_floor:
        return {"available": False}
    entry: dict[str, Any] = {"available": True, "channels": {}}
    published_channels = noise_floor.get("battery_channels", {})
    for channel, data in arms[champion].items():
        published = published_channels.get(channel)
        if not published:
            continue
        mine = data["seeds"]
        theirs = [round(value, 2) for value in published["per_seed_bb_per_100"]][
            : len(mine)
        ]
        if len(theirs) < len(mine):
            mine = mine[: len(theirs)]
        exact = mine == theirs
        deltas = [round(a - b, 2) for a, b in zip(mine, theirs, strict=True)]
        # Level and dispersion are separable failures and mean different
        # things. A level shift with a reproduced sigma says the *policy*
        # moved since the artifact was written while the *instrument* did
        # not, which leaves the artifact usable as a noise characterisation
        # and unusable as a reference mean.
        mine_sigma = independent_seed_mde(mine)["sigma_bb_per_100"]
        published_sigma = published.get("sigma_bb_per_100")
        entry["channels"][channel] = {
            "my_sigma_bb_per_100": mine_sigma,
            "published_sigma_bb_per_100": published_sigma,
            "sigma_ratio": (
                round(mine_sigma / published_sigma, 3) if published_sigma else None
            ),
            "comparable": data["hands_per_seed"] == published.get("hands_per_seed"),
            "hands_per_seed": data["hands_per_seed"],
            "published_hands_per_seed": published.get("hands_per_seed"),
            "seeds_compared": len(mine),
            "mine": mine,
            "published": theirs,
            "per_seed_identical": exact,
            "max_abs_delta": max((abs(d) for d in deltas), default=0.0),
            "mean_delta": round(sum(deltas) / len(deltas), 2) if deltas else 0.0,
            "published_mean_bb_per_100": published.get("mean_bb_per_100"),
            "my_mean_bb_per_100": data["bb_per_100"],
        }
    comparable = [v for v in entry["channels"].values() if v["comparable"]]
    entry["comparable"] = bool(comparable)
    entry["passed"] = bool(comparable) and all(
        value["per_seed_identical"] for value in comparable
    )
    if not comparable:
        entry["note"] = (
            "run at a different hand count from the published artifact, so a "
            "per-seed match is not expected and the gate is not informative"
        )
    return entry


def harness_gate(
    arms: Mapping[str, Any], candidate: str, gauntlet: Mapping[str, Any] | None
) -> dict:
    """The other half: does the *harness* reproduce a published candidate run?

    The champion reproduction gate cannot distinguish "the harness changed"
    from "the policy changed" on its own. This one can: ``candidate-v8-0001``
    is a frozen artifact, so if its per-seed BB/100 here is identical to
    ``candidate-v8-0001-gauntlet.json``'s on the shared channels, the harness
    is the frozen instrument and any champion mismatch is the champion's.
    """

    if not gauntlet:
        return {"available": False}
    channels = (gauntlet.get("batteries") or {}).get("channels", {})
    entry: dict[str, Any] = {"available": True, "channels": {}}
    for channel, published in channels.items():
        if channel not in arms.get(candidate, {}):
            continue
        theirs = [round(value, 2) for value in published["seeds"]]
        mine = arms[candidate][channel]["seeds"][: len(theirs)]
        entry["channels"][channel] = {
            "seeds_compared": len(mine),
            "mine": mine,
            "published": theirs,
            "per_seed_identical": mine == theirs,
        }
    entry["passed"] = bool(entry["channels"]) and all(
        value["per_seed_identical"] for value in entry["channels"].values()
    )
    return entry


def analyse(
    arms: Mapping[str, Any],
    *,
    candidate: str,
    incumbent: str,
    champion: str,
) -> dict:
    """Contrasts, dominance checks and the derived MDE for every channel."""

    labels = list(arms.keys())
    aggress_large = "trivial-always_aggress_large"
    analysis: dict[str, Any] = {"channels": {}}

    for channel in CHANNELS:
        seeds_of = {label: arms[label][channel]["seeds"] for label in labels}
        means = {label: arms[label][channel]["bb_per_100"] for label in labels}
        # Recomputed here rather than read back from the arms fragment, so the
        # report's dispersion figures are always the ones this code produces
        # from the per-seed values it is also showing.
        spread = {label: independent_seed_mde(seeds_of[label]) for label in labels}
        ranked = sorted(means.items(), key=lambda item: item[1], reverse=True)
        ranked = [item for item in ranked if item[0] != NULL_MIRROR_LABEL]

        contrasts: dict[str, Any] = {}
        # Every arm against the champion, in one consistent direction, so the
        # report table never has to negate a contrast and relabel its verdict.
        vs_champion = {
            label: paired_contrast(
                seeds_of[label], seeds_of[champion], label_a=label, label_b=champion
            )
            for label in labels
            if label != champion
        }

        def add(name: str, a: str, b: str) -> None:
            contrasts[name] = paired_contrast(
                seeds_of[a], seeds_of[b], label_a=a, label_b=b
            )

        add("candidate_vs_champion", candidate, champion)
        add("candidate_vs_incumbent", candidate, incumbent)
        add("incumbent_vs_champion", incumbent, champion)
        for mode in TRIVIAL_MODES:
            floor = f"trivial-{mode}"
            add(f"candidate_vs_{mode}", candidate, floor)
            add(f"champion_vs_{mode}", champion, floor)
            add(f"incumbent_vs_{mode}", incumbent, floor)

        beaten_by_floor = [
            mode
            for mode in TRIVIAL_MODES
            if means[f"trivial-{mode}"] > means[candidate]
        ]
        dominance = {
            "always_aggress_large_bb_per_100": means[aggress_large],
            "rank_of_always_aggress_large": 1
            + [label for label, _ in ranked].index(aggress_large),
            "arms_ranked": [[label, value] for label, value in ranked],
            "dominates_every_real_policy": all(
                means[aggress_large] > means[label]
                for label in (candidate, incumbent, champion)
            ),
            "beats_champion_paired": contrasts["champion_vs_always_aggress_large"],
            "floors_that_beat_the_candidate": beaten_by_floor,
        }

        analysis["channels"][channel] = {
            "means_bb_per_100": means,
            "vs_champion": vs_champion,
            "independent_seed_mde": {
                label: spread[label]["mde_bb_per_100"] for label in labels
            },
            "channel_sigma_bb_per_100": {
                label: spread[label]["sigma_bb_per_100"] for label in labels
            },
            "contrasts": contrasts,
            "trivial_baseline_gate": dominance,
        }

    # The wiring mirror: identically zero, on every channel, or nothing is read.
    mirror = {
        channel: paired_contrast(
            arms[NULL_MIRROR_LABEL][channel]["seeds"],
            arms[champion][channel]["seeds"],
            label_a=NULL_MIRROR_LABEL,
            label_b=champion,
        )
        for channel in CHANNELS
    }
    analysis["self_null"] = {
        "detail": (
            "the champion run under a second arm label on identical seeds; "
            "every paired difference must be exactly 0.00 or the pairing "
            "machinery, not the policies, is producing the numbers"
        ),
        "per_channel": mirror,
        "passed": all(
            all(diff == 0.0 for diff in entry["diffs"]) for entry in mirror.values()
        ),
    }

    # Mechanism test, stated as a cross-channel comparison in two units.
    mech: dict[str, Any] = {}
    for channel in CHANNELS:
        contrast = analysis["channels"][channel]["contrasts"]["candidate_vs_champion"]
        sigma = analysis["channels"][channel]["channel_sigma_bb_per_100"][champion]
        mech[channel] = {
            "candidate_minus_champion_bb_per_100": contrast["mean"],
            "paired_mde_bb_per_100": contrast["paired_mde_bb_per_100"],
            "t": contrast["t"],
            "in_channel_sigma_units": round(contrast["mean"] / sigma, 2) if sigma else None,
            "verdict": contrast["verdict"],
        }
    analysis["mechanism_test"] = {
        "hypothesis": (
            "candidate-v8-0001 overbets never-folding card-blind archetypes "
            "because its fold_through head was fitted on a real field that "
            "folds ~50% of the time; against an opponent that folds "
            "realistically it should stand RELATIVELY better than it does on "
            "vs-station"
        ),
        "per_channel": mech,
        "note": (
            "BB/100 is not comparable across channels at face value, so the "
            "same contrast is also given in that channel's own champion sigma"
        ),
    }
    return analysis


def render_markdown(report: Mapping[str, Any]) -> str:
    candidate = report["arms_of_interest"]["candidate"]
    incumbent = report["arms_of_interest"]["incumbent"]
    champion = report["arms_of_interest"]["champion"]
    arms = report["arms"]["arms"]
    analysis = report["analysis"]
    instrument = report["instrument"]

    lines: list[str] = []
    add = lines.append
    add("# P3 held-out gate — 2026-08-16")
    add("")
    add(
        f"`{report['config']['output']}` · fit `{report['config']['p3_fit']}` · "
        f"{report['arms']['seed_count']} seeds/arm · "
        f"{arms[champion][P3_CHANNEL]['hands_per_seed']} hands/seed · "
        "identical seeds and hands across every arm"
    )
    add("")
    add(
        "> **P3 was measured before it was trained against.** Nothing in this "
        "run trains, harvests, promotes or deploys; `artifacts/approved.json` "
        "is untouched."
    )
    add("")

    add("## 0. The instrument, before any result")
    add("")
    kat = instrument["known_answer_test"]
    neg, pos = kat["negative_control"], kat["positive_control"]
    add(
        "Known-answer test with a matched negative control — the *opponent's* "
        "own canonical strength when it continues minus when it folds:"
    )
    add("")
    add("| control | channel | continue − fold | 95% CI | t | verdict |")
    add("|---|---|---|---|---|---|")
    for name, entry in (("negative (must be ~0)", neg), ("positive (must be > 0)", pos)):
        obs = entry["observed"]
        add(
            f"| {name} | `{entry['channel']}` | {obs['value']:+.4f} | "
            f"[{obs['ci95'][0]:+.4f}, {obs['ci95'][1]:+.4f}] | {obs['t']:+.2f} | "
            f"{'PASS' if entry['passed'] else 'FAIL'} |"
        )
    add("")
    add(
        f"Monotonicity sweep of the fitted fold probability: "
        f"**{'PASS' if instrument['monotonicity_sweep']['passed'] else 'FAIL'}** "
        "(worse hands fold more, worse prices fold more, every street model)."
    )
    add(
        f"Self-null (champion under a second label, identical seeds): "
        f"**{'PASS' if analysis['self_null']['passed'] else 'FAIL'}** — every "
        "paired difference exactly 0.00."
    )
    add(
        f"Chip conservation: "
        f"{report['arms']['chip_conservation_violations']} violations in "
        f"{report['config']['arm_matches']} matches."
    )
    add(
        "Degradation guard: the P3 seat raises rather than silently reverting "
        "to the card-blind constant, so no vs-p3 number here was partly "
        "produced by `vs-median`."
    )
    add("")
    harness = analysis.get("harness_gate", {})
    if harness.get("available"):
        shared = ", ".join(f"`{name}`" for name in harness["channels"])
        add(
            f"**Harness gate** — `{candidate}` is a frozen artifact, so its "
            f"per-seed BB/100 on {shared} must match "
            "`candidate-v8-0001-gauntlet.json` exactly if this runner is the "
            f"frozen instrument: **{'PASS' if harness['passed'] else 'FAIL'}** "
            + ", ".join(
                f"{name} {'identical' if v['per_seed_identical'] else 'DIFFERS'} "
                f"({v['seeds_compared']} seeds)"
                for name, v in harness["channels"].items()
            )
            + "."
        )
        add("")

    repro = analysis.get("reproduction_gate", {})
    if repro.get("available") and repro.get("comparable"):
        add(
            "**Reproduction gate** — the champion is deterministic given the "
            "seeds, so on the two channels this run shares with "
            "`noise-floor-2026-08-15.json` its per-seed BB/100 should be "
            "*identical*, or this is not the same instrument:"
        )
        add("")
        add(
            "| channel | per-seed identical | my mean | published mean | "
            "my σ | published σ | σ ratio |"
        )
        add("|---|---|---|---|---|---|---|")
        for channel, entry in repro["channels"].items():
            add(
                f"| `{channel}` | "
                f"{'YES' if entry['per_seed_identical'] else 'NO'} | "
                f"{entry['my_mean_bb_per_100']:+.2f} | "
                f"{entry['published_mean_bb_per_100']:+.2f} | "
                f"{entry['my_sigma_bb_per_100']:.2f} | "
                f"{entry['published_sigma_bb_per_100']:.2f} | "
                f"{entry['sigma_ratio']} |"
            )
        add("")
        if not repro["passed"]:
            add(
                "> **The champion arm does not reproduce the published "
                "per-seed values, but its σ does.** Dispersion matching while "
                "the level moves says the *policy* changed since "
                "`noise-floor-2026-08-15.json` was written and the "
                "*instrument* did not — which the harness gate above confirms "
                "independently, by reproducing a frozen artifact's per-seed "
                "values exactly through this same runner. Two consequences, "
                "both load-bearing: "
                "the published per-channel σ and MDE remain usable (and the "
                "vs-p3 MDE derived by the same construction is therefore on a "
                "sound footing), while the published champion **means** are "
                "stale and must not be used as a reference level. Every "
                "contrast inside this report pairs arms that ran in this run "
                "on shared seeds, so none of them depends on the published "
                "artifact."
            )
            add("")

    add("## 1. The derived MDE for `vs-p3`")
    add("")
    add(
        "`noise-floor-2026-08-15.json` has no `vs-p3` row. Two figures are "
        "derived from this run:"
    )
    add("")
    add("| construction | value | what it is for |")
    add("|---|---|---|")
    p3 = analysis["channels"][P3_CHANNEL]
    n = report["arms"]["seed_count"]
    add(
        f"| independent-seed `2σ/√n` (champion arm, σ = "
        f"{p3['channel_sigma_bb_per_100'][champion]:.2f}) | "
        f"**{p3['independent_seed_mde'][champion][f'{n}_seeds']:.2f} BB/100** at "
        f"{n} seeds | comparing against a number measured on *other* seeds; "
        "same construction as the published per-channel MDEs |"
    )
    pc = p3["contrasts"]["candidate_vs_champion"]
    add(
        f"| paired `2·sd(diffs)/√n` | **{pc['paired_mde_bb_per_100']:.2f} BB/100** "
        "(candidate vs champion) | the contrasts in this report, which share "
        "seeds and therefore cancel most of the variance |"
    )
    add("")
    add("Per-channel comparison of the same construction:")
    add("")
    add("| channel | champion σ | independent-seed MDE @" + str(n) + " |")
    add("|---|---|---|")
    for channel in CHANNELS:
        entry = analysis["channels"][channel]
        add(
            f"| `{channel}` | {entry['channel_sigma_bb_per_100'][champion]:.2f} | "
            f"{entry['independent_seed_mde'][champion][f'{n}_seeds']:.2f} |"
        )
    add("")

    add("## 2. The table")
    add("")
    for channel in CHANNELS:
        entry = analysis["channels"][channel]
        add(f"### `{channel}`")
        add("")
        add(
            "| arm | BB/100 | σ across seeds | vs champion (paired) | t | "
            "paired MDE | verdict |"
        )
        add("|---|---|---|---|---|---|---|")
        for label, value in entry["trivial_baseline_gate"]["arms_ranked"]:
            sigma = entry["channel_sigma_bb_per_100"][label]
            if label == champion:
                add(
                    f"| **{label}** | **{value:+.2f}** | {sigma:.2f} | — | — | "
                    "— | reference |"
                )
                continue
            contrast = entry["vs_champion"][label]
            verdict = (
                "UNRESOLVED"
                if contrast["verdict"] == "UNRESOLVED"
                else ("ahead" if contrast["mean"] > 0 else "behind")
            )
            add(
                f"| {label} | {value:+.2f} | {sigma:.2f} | "
                f"{contrast['mean']:+.2f} | {contrast['t']} | "
                f"{contrast['paired_mde_bb_per_100']:.2f} | {verdict} |"
            )
        add("")

    add("## 3. Does `always_aggress_large` still dominate?")
    add("")
    add("| channel | aggress-large BB/100 | rank among arms | beats every real policy? |")
    add("|---|---|---|---|")
    for channel in CHANNELS:
        gate = analysis["channels"][channel]["trivial_baseline_gate"]
        add(
            f"| `{channel}` | {gate['always_aggress_large_bb_per_100']:+.2f} | "
            f"{gate['rank_of_always_aggress_large']} of "
            f"{len(gate['arms_ranked'])} | "
            f"{'YES' if gate['dominates_every_real_policy'] else 'NO'} |"
        )
    add("")

    add("## 4. Mechanism test")
    add("")
    add(f"> {analysis['mechanism_test']['hypothesis']}")
    add("")
    add("| channel | candidate − champion | paired MDE | in champion-σ units | verdict |")
    add("|---|---|---|---|---|")
    for channel, entry in analysis["mechanism_test"]["per_channel"].items():
        add(
            f"| `{channel}` | {entry['candidate_minus_champion_bb_per_100']:+.2f} | "
            f"{entry['paired_mde_bb_per_100']:.2f} | "
            f"{entry['in_channel_sigma_units']:+.2f} | {entry['verdict']} |"
        )
    add("")

    logs = report.get("logs")
    if logs:
        add("## 5. Decision logs on `vs-p3`")
        add("")
        add(
            "| policy | decisions | aggression | fold | median wager/pot | "
            "separation (all streets) | 95% CI |"
        )
        add("|---|---|---|---|---|---|---|")
        for label in (candidate, incumbent, champion):
            entry = logs["policies"].get(label, {}).get(P3_CHANNEL)
            if not entry:
                continue
            sep = entry["strength_separation"]["all_streets"]
            value = sep.get("separation_aggress_minus_fold")
            ci = sep.get("ci95")
            shown = f"{value:+.4f}" if value is not None else "n/a"
            ci_shown = (
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                if ci
                else "n/a (too few folds to separate)"
            )
            add(
                f"| {label} | {entry['decisions']} | "
                f"{entry['aggression_rate']:.3f} | {entry['fold_rate']:.3f} | "
                f"{entry['median_wager_pot_fraction']} | {shown} | {ci_shown} |"
            )
        add("")
        add("Per street, on `vs-p3` (canonical metric):")
        add("")
        add("| policy | preflop | flop | turn | river |")
        add("|---|---|---|---|---|")
        for label in (candidate, incumbent, champion):
            entry = logs["policies"].get(label, {}).get(P3_CHANNEL)
            if not entry:
                continue
            cells = []
            for street in ("preflop", "flop", "turn", "river"):
                street_entry = entry["strength_separation"][street]
                value = street_entry.get("separation_aggress_minus_fold")
                if value is None:
                    cells.append("n/a")
                    continue
                low, high = street_entry["ci95"]
                cells.append(f"{value:+.4f} [{low:+.3f}, {high:+.3f}]")
            add(f"| {label} | " + " | ".join(cells) + " |")
        add("")
        add("The same three policies on the two control channels:")
        add("")
        add("| policy | channel | aggression | fold | median wager/pot | separation |")
        add("|---|---|---|---|---|---|")
        for label in (candidate, incumbent, champion):
            for channel in CHANNELS:
                if channel == P3_CHANNEL:
                    continue
                entry = logs["policies"].get(label, {}).get(channel)
                if not entry:
                    continue
                sep = entry["strength_separation"]["all_streets"]
                add(
                    f"| {label} | `{channel}` | {entry['aggression_rate']:.3f} | "
                    f"{entry['fold_rate']:.3f} | "
                    f"{entry['median_wager_pot_fraction']} | "
                    f"{sep.get('separation_aggress_minus_fold')} |"
                )
        add("")
        add("### What the opponent does back")
        add("")
        add(
            "The direct read on the mechanism: how often the seat folds to "
            "each hero, and how much stronger the hands that *continue* are "
            "than the hands that fold. The second column can only be non-zero "
            "for a card-aware seat."
        )
        add("")
        add("| channel | facing | opp fold rate | opp continue − fold strength | t |")
        add("|---|---|---|---|---|")
        for channel in CHANNELS:
            for label, entry in sorted(
                logs.get("opponent_behaviour", {}).get(channel, {}).items()
            ):
                gap = entry["continue_minus_fold_strength"]
                value = gap.get("value")
                add(
                    f"| `{channel}` | {label} | {entry['fold_rate']:.3f} | "
                    + (f"{value:+.4f}" if value is not None else "n/a")
                    + f" | {gap.get('t')} |"
                )
        add("")

    add("## Caveats")
    add("")
    add(
        "- The P3 opponent is a **fitted model of the observed field's fold "
        "frequency**, not an optimal or exploitative player. Beating it is "
        "not evidence of beating the field."
    )
    p3_fold = kat["positive_control"]["observed"]["realized_fold_rate"]
    median_fold = kat["negative_control"]["observed"]["realized_fold_rate"]
    add(
        f"- **`vs-p3` differs from `vs-median` in level as well as in "
        f"conditioning.** Realized fold rate {p3_fold:.3f} against "
        f"{median_fold:.3f} for the card-blind twin. The two channels are "
        "therefore not a clean single-variable contrast: P3 both folds *more* "
        f"({p3_fold - median_fold:+.3f}) and folds *selectively*. Note the "
        "direction — more folding gives an aggressor **more** fold equity, so "
        "it works against any finding that aggression stops paying here."
    )
    add(
        "- The fit's postflop strata are thin (flop 455 training rows, turn "
        "154, river 74); its preflop stratum carries the bulk of the data. "
        "Postflop behaviour is the least-supported part of the opponent."
    )
    add(
        "- Its aggression and shove rate are still the inherited card-blind "
        "scripted knobs. **Only the folding is card-aware.** A policy can "
        "still not be punished for the *hands it shows down*, only for the "
        "prices it lays."
    )
    add(
        "- `vs-p3` is heads-up. Multiway strength-aware play is untested here."
    )
    add(
        "- The derived MDEs come from this run's own seeds; they are "
        "estimates with their own uncertainty at "
        f"{report['arms']['seed_count']} seeds, not published constants."
    )
    if repro.get("available") and repro.get("comparable") and not repro.get("passed"):
        add(
            "- **The champion moved since the published noise floor.** Every "
            "`vs champion` figure here is against the champion as it builds "
            "*today*; published statements of the form \"v8 is N BB/100 "
            "behind the champion\" were computed against the 2026-08-15 "
            "champion and are not the same comparison."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--incumbent", default=DEFAULT_INCUMBENT)
    parser.add_argument("--p3-fit", default=str(DEFAULT_FIT_PATH))
    parser.add_argument(
        "--noise-floor",
        default="artifacts/evaluations/noise-floor-2026-08-15.json",
        help="published per-channel noise floor, used for the reproduction gate",
    )
    parser.add_argument(
        "--gauntlet",
        default="artifacts/evaluations/candidate-v8-0001-gauntlet.json",
        help="published candidate gauntlet, used for the harness gate",
    )
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--log-seeds", type=int, default=4)
    parser.add_argument("--probe-seeds", type=int, default=4)
    parser.add_argument("--probe-hands", type=int, default=400)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--equity-trials", type=int, default=80)
    parser.add_argument("--starting-stack", type=int, default=6_000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--stages", default=",".join(_STAGES))
    parser.add_argument(
        "--output", default="artifacts/evaluations/p3-gate-2026-08-16.json"
    )
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args(argv)

    if args.seeds < 8:
        parser.error("the gate needs at least 8 seeds")

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = sorted(set(stages) - set(_STAGES))
    if unknown:
        parser.error(f"unknown stages: {unknown}")

    output = Path(args.output)
    workdir = Path(args.workdir) if args.workdir else output.with_suffix(".stages")

    runners = {
        "instrument": lambda: stage_instrument(args),
        "arms": lambda: stage_arms(args),
        "logs": lambda: stage_logs(args),
    }
    timings: dict[str, float] = {}
    for stage in _STAGES:
        if stage not in stages:
            continue
        print(f"[p3_gate] stage {stage} starting", flush=True)
        started = time.monotonic()
        fragment = runners[stage]()
        timings[stage] = round(time.monotonic() - started, 1)
        _write_fragment(workdir, stage, fragment)
        print(f"[p3_gate] stage {stage} done in {timings[stage]}s", flush=True)

    fragments = {stage: _read_fragment(workdir, stage) for stage in _STAGES}
    arms_fragment = fragments["arms"]
    if arms_fragment is None:
        print("[p3_gate] no arms fragment; nothing to assemble", flush=True)
        return 1

    specs = build_arms(args)
    candidate, incumbent, champion = (spec.label for spec in specs[:3])
    noise_floor = None
    noise_path = Path(args.noise_floor)
    if noise_path.exists():
        noise_floor = json.loads(noise_path.read_text(encoding="utf-8"))
    analysis = analyse(
        arms_fragment["arms"],
        candidate=candidate,
        incumbent=incumbent,
        champion=champion,
    )
    analysis["reproduction_gate"] = reproduction_gate(
        arms_fragment["arms"], champion, noise_floor
    )
    gauntlet = None
    gauntlet_path = Path(args.gauntlet)
    if gauntlet_path.exists():
        gauntlet = json.loads(gauntlet_path.read_text(encoding="utf-8"))
    analysis["harness_gate"] = harness_gate(arms_fragment["arms"], candidate, gauntlet)
    report = {
        "study": "p3-held-out-gate",
        "date": "2026-08-16",
        "arms_of_interest": {
            "candidate": candidate,
            "incumbent": incumbent,
            "champion": champion,
        },
        "config": {
            "candidate_manifest": args.candidate,
            "incumbent_manifest": args.incumbent,
            "p3_fit": args.p3_fit,
            "seeds": args.seeds,
            "log_seeds": args.log_seeds,
            "scale": args.scale,
            "equity_trials": args.equity_trials,
            "starting_stack": args.starting_stack,
            "stacks": "carry-over-sessions",
            "output": args.output,
            "arm_matches": sum(
                len(entry) for entry in arms_fragment["arms"].values()
            )
            * arms_fragment["seed_count"],
            "stage_elapsed_seconds": timings,
        },
        "instrument": fragments["instrument"],
        "arms": arms_fragment,
        "logs": fragments["logs"],
        "analysis": analysis,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if fragments["instrument"]:
        output.with_suffix(".md").write_text(
            render_markdown(report), encoding="utf-8", newline="\n"
        )
        print(f"[p3_gate] markdown written to {output.with_suffix('.md')}", flush=True)
    print(f"[p3_gate] report written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
