"""What the three unmeasured gate changes would have done to real decisions.

``.handoff/PENDING_EDITS.md`` carries a caveat that has blocked this
decision since 2026-08-15: the only evidence against the effective-stack
risk cap is **-13.97 BB/100 (t = -5.18)**, and it was collected on
batteries whose every opponent is card-blind and *structurally cannot
punish overcommitment*. That instrument prices what a gate costs and is
blind to what it prevents, so it can only ever argue for reverting.

This tool measures the other half on the only population where
overcommitment was ever actually punished: the **stored live journal**.
``.arena-training.jsonl`` records, per decision, both denominators the
argument is about -- ``state.hero_stack_chips`` and
``state.effective_stack_chips`` -- plus the price, the legal range, the
street and the equity. Every gate in question is arithmetic over exactly
those fields, so the counterfactual "what would the changed gate have
done here" is *computed, not simulated*, and it is computed against hands
whose chip outcome is also on record.

What this is, and what it is not
--------------------------------

It is a **binding-rate and exposure** measurement: how often the change
alters a gate's verdict, on which decisions, and what those hands went on
to be worth.

It is **not** a counterfactual EV. Clipping a bet changes how the hand
continues, and this journal cannot replay a hand that never happened. A
line like "the changed cap would have bound on the -3,768 hand" means the
gate would have refused that size -- not that the hand would have broken
even. Every money figure here is an **association with hands the gate
would have touched**, and is reported as exposure, never as a saving.

Read it beside the battery number, not instead of it.

Measure the instrument before the result
----------------------------------------

The ``instrument`` section runs first and reports first. Its checks are
impossible-by-construction, not preferences:

* ``effective_stack_chips <= hero_stack_chips`` on **every** record.
  ``game_state.effective_stack_chips`` is ``min(hero, deepest active
  opponent)``, so a single violation means the journal is not what this
  tool thinks it is and no number below can be believed.
* On records where the two denominators are **equal**, every gate verdict
  under one must equal the verdict under the other -- exactly zero
  differences. A tool that reports divergence there is comparing
  something other than the denominator.
* **Engine parity**: every recorded sized bet must sit at or under the
  *hero-purse* cap, because a hero-purse-denominated engine is what
  produced those amounts. This is a known-answer test of the audit's
  arithmetic rather than of the gate -- if the cap were modelled wrongly
  here, that inequality would break. Rows over it are surfaced rather
  than hidden: they are also the signature a supervisor restart under
  the changed gates would leave.
* Full parse accounting, including lines that failed to parse.

Gate constants come from the **incumbent's own manifest**, not from the
dataclass defaults, because ``load_policy`` rebuilds a learned policy's
gates from its ``engine_parameters.safety_gates`` block. Reading them
from anywhere else would audit gates the served policy does not use.

Stdlib-only, entirely offline, and read-only: nothing here promotes,
deploys, joins, or writes to ``artifacts/approved.json``.

Example::

    python -m tools.gate_binding_audit \\
        --output artifacts/evaluations/gate-binding-audit-2026-08-26.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_JOURNAL = ".arena-training.jsonl"
DEFAULT_MANIFEST = "artifacts/candidates/candidate-v7-0001c.approved.manifest.json"
DEFAULT_POLICY = "candidate-v7-0001c"

#: Board cards still to come, by street. Copied from
#: ``game_state._REVEALS_REMAINING`` rather than imported so this audit
#: states its own arithmetic; ``test_gate_binding_audit`` pins the two
#: together so a drift in either is a test failure, not a silent skew.
REVEALS_REMAINING = {"preflop": 3, "flop": 2, "turn": 1, "river": 0}

AGGRESSIVE_ACTIONS = frozenset({"bet", "raise", "all-in", "all_in", "allin"})


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """One recorded live decision, reduced to what the gates read."""

    table_id: str
    street: str
    big_blind: int
    hero_stack: int
    effective_stack: int
    contribution: int
    to_call: int
    raise_min: int | None
    raise_max: int | None
    equity: float | None
    action: str
    amount_to: int | None
    hyper: bool

    @property
    def denominators_agree(self) -> bool:
        return self.hero_stack == self.effective_stack

    def reveal_expense(self, price_chips: int) -> float:
        """``game_state.card_reveal_expense`` over journal fields."""

        at_risk = max(1, self.effective_stack)
        share = min(1.0, max(0, price_chips) / at_risk)
        return share * (REVEALS_REMAINING.get(self.street, 0) / 3.0)


def _int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_journal(path: Path, policy_version: str) -> tuple[list[Decision], dict, dict]:
    """Decisions for one policy, the hand ledger, and parse accounting."""

    decisions: list[Decision] = []
    hand_deltas: dict[str, int] = {}
    counts = Counter()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            counts["lines"] += 1
            line = line.strip()
            if not line:
                counts["blank"] += 1
                continue
            try:
                record = json.loads(line)
            except ValueError:
                # Recorded, not swallowed: a truncated tail is evidence
                # about the stop path (PENDING_EDITS 22), not noise.
                counts["unparsable"] += 1
                continue

            event = record.get("event")
            counts[f"event:{event}"] += 1

            if event == "hand_result":
                table_id = str(record.get("table_id") or "")
                delta = _int(record.get("chip_delta_chips"))
                if table_id and delta is not None:
                    hand_deltas[table_id] = delta
                continue

            if event != "decision":
                continue
            if record.get("policy_version") != policy_version:
                counts["decision:other_policy"] += 1
                continue

            state = record.get("state") or {}
            legal = record.get("legal") or {}
            hero_stack = _int(state.get("hero_stack_chips"))
            effective = _int(state.get("effective_stack_chips"))
            if hero_stack is None or effective is None:
                counts["decision:missing_stacks"] += 1
                continue

            raise_range = legal.get("raise_range") or {}
            decisions.append(
                Decision(
                    table_id=str(record.get("table_id") or ""),
                    street=str(record.get("street") or "").casefold(),
                    big_blind=_int(record.get("big_blind_chips"), 1) or 1,
                    hero_stack=hero_stack,
                    effective_stack=effective,
                    contribution=_int(state.get("hero_contribution_chips"), 0) or 0,
                    to_call=_int(legal.get("call_chips"), 0) or 0,
                    raise_min=_int(raise_range.get("min")),
                    raise_max=_int(raise_range.get("max")),
                    equity=record.get("equity"),
                    action=str(record.get("action") or ""),
                    amount_to=_int(record.get("amount_to")),
                    hyper=bool(record.get("hyper_aggression")),
                )
            )

    return decisions, hand_deltas, dict(counts)


def load_gates(manifest_path: Path) -> dict:
    """Gate constants as the *served* policy uses them.

    ``learned_policy.load_policy`` rebuilds gates from the manifest's
    ``engine_parameters.safety_gates`` block, so the dataclass defaults
    are the wrong source for anything the incumbent actually served.
    Fields absent from a frozen block -- ``reveal_expense_equity_slope``
    is absent from every pre-2026-08-15 manifest -- fall through to the
    dataclass default, which is exactly how an unmeasured change ships
    under an approved artifact.
    """

    from devfun_poker_playground.decision_engine import SafetyGates

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    block = (manifest.get("engine_parameters") or {}).get("safety_gates") or {}
    gates = SafetyGates.from_mapping(block)
    return {
        "source": str(manifest_path),
        "keys_in_manifest": sorted(block),
        "risk_cap_stack_fraction": gates.risk_cap_stack_fraction,
        "near_nut_floor": gates.near_nut_floor,
        "call_stack_gates": [list(pair) for pair in gates.call_stack_gates],
        "reveal_expense_equity_slope": gates.reveal_expense_equity_slope,
        "inherited_from_dataclass_default": sorted(
            name
            for name in (
                "reveal_expense_equity_slope",
                "risk_cap_on_effective_stack",
                "call_gates_on_effective_stack",
                "gate_stack_counts_committed_chips",
                "pot_odds_exclude_uncallable",
                "condition_range_without_price",
            )
            if name not in block
        ),
    }


# ---------------------------------------------------------------------------
# The gates, as arithmetic over journal fields
# ---------------------------------------------------------------------------


def risk_cap(decision: Decision, gates: Mapping[str, Any], *, effective: bool) -> int:
    """``contribution + max(bb, round(fraction * denominator))``."""

    denominator = decision.effective_stack if effective else decision.hero_stack
    return decision.contribution + max(
        decision.big_blind,
        round(gates["risk_cap_stack_fraction"] * max(0, denominator)),
    )


def cap_applies(decision: Decision, gates: Mapping[str, Any]) -> bool:
    """The cap is only reached below the near-nut release."""

    return decision.equity is None or decision.equity < gates["near_nut_floor"]


def call_gate_triggers(
    decision: Decision, gates: Mapping[str, Any], *, effective: bool
) -> list[float]:
    """Stack-fraction triggers that fire at this price, in gate order."""

    denominator = decision.effective_stack if effective else decision.hero_stack
    return [
        fraction
        for fraction, _floor in gates["call_stack_gates"]
        if decision.to_call >= fraction * max(1, denominator)
    ]


def call_gate_refuses(
    decision: Decision, gates: Mapping[str, Any], *, effective: bool
) -> bool:
    """Would the gate actually REFUSE this call, not merely reach it?

    Arrival and refusal are different events and conflating them
    overstates the edit. The engine (``decision_engine._call_clears_margin``)
    refuses only when ``equity < required``, where::

        required = (1 - wildness) * (floor + reveal_penalty)
                   + wildness * neutral_price

    ``wildness`` is not in the journal, so this evaluates at ``w = 0``.
    That is not a neutral choice, it is the **maximising** one: the
    neutral price is below the gate floor on every flagged decision here,
    so any tracked wildness strictly lowers the bar and strictly reduces
    refusals. Every refusal count this function produces is therefore a
    **ceiling**, and must be reported as one.
    """

    if decision.equity is None:
        return False
    penalty = gates["reveal_expense_equity_slope"] * decision.reveal_expense(
        decision.to_call
    )
    denominator = decision.effective_stack if effective else decision.hero_stack
    return any(
        decision.to_call >= fraction * max(1, denominator)
        and decision.equity < floor + penalty
        for fraction, floor in gates["call_stack_gates"]
    )


def _sized_aggressively(decision: Decision) -> bool:
    return decision.action in AGGRESSIVE_ACTIONS and decision.amount_to is not None


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------


def stage_instrument(decisions: Sequence[Decision], gates: Mapping[str, Any]) -> dict:
    """Known-answer checks, before any result is computed."""

    # 1. effective_stack_chips is min(hero, deepest opponent). Nothing in a
    #    correct journal can exceed hero's own purse.
    over = [d for d in decisions if d.effective_stack > d.hero_stack]

    # 2. Where the denominators are equal, the two configurations are the
    #    same function. Any difference here is a wiring fault.
    agreeing = [d for d in decisions if d.denominators_agree]
    agreeing_diffs = sum(
        1
        for d in agreeing
        if risk_cap(d, gates, effective=True) != risk_cap(d, gates, effective=False)
        or call_gate_triggers(d, gates, effective=True)
        != call_gate_triggers(d, gates, effective=False)
    )

    # 3. Engine parity -- the check that validates the arithmetic itself.
    #
    #    These amounts were produced by a live engine whose cap was
    #    denominated in hero's purse, and `_sized_action` returns
    #    `min(max(desired, minimum), maximum)` with `maximum <= risk_cap`.
    #    So every recorded sized bet MUST satisfy `amount_to <= cap_hero`.
    #    If this tool's cap arithmetic disagreed with the engine's, that
    #    inequality would break -- which makes it a known-answer test of
    #    the audit, not of the gate.
    #
    #    All-ins are excluded: they do not come from the sizing path.
    #    A non-zero count here is not automatically a defect in this tool
    #    -- it is also what a supervisor restart under the *changed* gates
    #    would look like -- so the violating rows are surfaced, not hidden.
    parity_rows = [
        d
        for d in decisions
        if d.action in ("bet", "raise")
        and d.amount_to is not None
        and cap_applies(d, gates)
    ]
    parity_breaks = [
        d for d in parity_rows if d.amount_to > risk_cap(d, gates, effective=False)
    ]
    # Of those, the ones consistent with the EFFECTIVE-stack cap instead:
    # the signature of decisions served after the edit went live.
    served_by_new_gate = [
        d for d in parity_breaks if d.amount_to <= risk_cap(d, gates, effective=True)
    ]

    # 4. The reveal expense is a share in [0, 1] and is 0 on the river by
    #    definition -- paying there buys a showdown, not a card.
    expenses = [d.reveal_expense(d.to_call) for d in decisions]
    out_of_range = sum(1 for value in expenses if not 0.0 <= value <= 1.0)
    river_nonzero = sum(
        1 for d in decisions if d.street == "river" and d.reveal_expense(d.to_call) != 0
    )

    checks = {
        "effective_never_exceeds_hero": {
            "violations": len(over),
            "verdict": "PASS" if not over else "FAIL",
        },
        "equal_denominators_give_equal_verdicts": {
            "records": len(agreeing),
            "differences": agreeing_diffs,
            "verdict": "PASS" if agreeing_diffs == 0 else "FAIL",
        },
        "engine_parity_on_the_hero_purse_cap": {
            "sized_bets_checked": len(parity_rows),
            "over_the_hero_purse_cap": len(parity_breaks),
            "of_which_within_the_effective_stack_cap": len(served_by_new_gate),
            "verdict": "PASS" if not parity_breaks else "REVIEW",
        },
        "reveal_expense_in_range": {
            "out_of_range": out_of_range,
            "river_nonzero": river_nonzero,
            "verdict": "PASS" if not out_of_range and not river_nonzero else "FAIL",
        },
    }
    blocking = (
        "effective_never_exceeds_hero",
        "equal_denominators_give_equal_verdicts",
        "reveal_expense_in_range",
    )
    checks["all_passed"] = all(checks[name]["verdict"] == "PASS" for name in blocking)
    return checks


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def _rate(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def stage_binding(decisions: Sequence[Decision], gates: Mapping[str, Any]) -> dict:
    """How often each edit changes a verdict, and on what."""

    diverging = [d for d in decisions if not d.denominators_agree]

    # --- the sizing risk cap -------------------------------------------
    sized = [d for d in decisions if _sized_aggressively(d)]
    capped = [d for d in sized if cap_applies(d, gates)]
    would_clip = [
        d for d in capped if (d.amount_to or 0) > risk_cap(d, gates, effective=True)
    ]
    # Below the minimum legal raise the engine declines to size at all and
    # falls through to the passive path -- a strictly larger intervention
    # than clipping.
    would_decline = [
        d
        for d in would_clip
        if d.raise_min is not None and risk_cap(d, gates, effective=True) < d.raise_min
    ]
    clip_chips = [
        (d.amount_to or 0) - risk_cap(d, gates, effective=True) for d in would_clip
    ]

    # --- the call stack gates ------------------------------------------
    calls = [d for d in decisions if d.action == "call" and d.to_call > 0]
    # Reaches a gate it would not have reached on hero's purse...
    newly_gated = [
        d
        for d in calls
        if call_gate_triggers(d, gates, effective=True)
        and not call_gate_triggers(d, gates, effective=False)
    ]
    # ...of which this many would actually be refused. Arrival is not
    # refusal: the floor still has to beat hero's equity.
    newly_refused = [
        d
        for d in calls
        if call_gate_refuses(d, gates, effective=True)
        and not call_gate_refuses(d, gates, effective=False)
    ]

    # --- the reveal-expense slope --------------------------------------
    slope = gates["reveal_expense_equity_slope"]
    priced_calls = [d for d in calls if d.reveal_expense(d.to_call) > 0]
    penalties = [slope * d.reveal_expense(d.to_call) for d in priced_calls]
    # The slope only decides anything where a stack gate already fired --
    # it raises that gate's equity floor, it does not create a gate.
    slope_live = [d for d in newly_gated if d.reveal_expense(d.to_call) > 0]

    return {
        "decisions": len(decisions),
        "denominator": {
            "diverging": len(diverging),
            "diverging_pct": _rate(len(diverging), len(decisions)),
            "median_hero_stack": (
                round(statistics.median([d.hero_stack for d in diverging]), 1)
                if diverging
                else None
            ),
            "median_effective_stack": (
                round(statistics.median([d.effective_stack for d in diverging]), 1)
                if diverging
                else None
            ),
        },
        "risk_cap": {
            "sized_aggressive": len(sized),
            "below_near_nut": len(capped),
            "would_clip": len(would_clip),
            "would_clip_pct_of_capped": _rate(len(would_clip), len(capped)),
            "would_decline_entirely": len(would_decline),
            "chips_removed_total": sum(clip_chips),
            "chips_removed_median": (
                round(statistics.median(clip_chips), 1) if clip_chips else None
            ),
            "chips_removed_max": max(clip_chips) if clip_chips else None,
        },
        "call_gates": {
            "calls": len(calls),
            "reaches_a_gate": len(newly_gated),
            "reaches_a_gate_distinct_decisions": len(
                {(d.table_id, d.street, d.to_call, d.equity) for d in newly_gated}
            ),
            "would_refuse_ceiling": len(newly_refused),
            "would_refuse_pct": _rate(len(newly_refused), len(calls)),
            "refusal_is_a_ceiling_because": (
                "wildness is not recorded in the journal and w>0 strictly "
                "lowers the required equity, so refusals are evaluated at w=0"
            ),
            "chips_at_stake_total": sum(d.to_call for d in newly_refused),
            "chips_at_stake_max": (
                max((d.to_call for d in newly_refused), default=None)
                if newly_refused
                else None
            ),
        },
        "reveal_slope": {
            "slope": slope,
            "calls_with_expense": len(priced_calls),
            "median_penalty": (
                round(statistics.median(penalties), 4) if penalties else None
            ),
            "max_penalty": round(max(penalties), 4) if penalties else None,
            "calls_where_a_stack_gate_also_fires": len(slope_live),
        },
        "by_street": {
            street: {
                "decisions": sum(1 for d in decisions if d.street == street),
                "diverging_pct": _rate(
                    sum(1 for d in diverging if d.street == street),
                    sum(1 for d in decisions if d.street == street),
                ),
            }
            for street in ("preflop", "flop", "turn", "river")
        },
    }


def _bounds_exposure(
    decision: Decision, gates: Mapping[str, Any], *, effective: bool
) -> bool:
    """Does the cap hold hero under what can actually be lost here?

    The cap allows a further commitment of ``max(bb, round(frac * denom))``.
    Hero could otherwise commit the whole effective stack (bounded by the
    legal range). The gate is doing work only when the former is smaller.
    """

    denominator = decision.effective_stack if effective else decision.hero_stack
    allowance = max(decision.big_blind, round(gates["risk_cap_stack_fraction"] * max(0, denominator)))
    headroom = decision.effective_stack
    if decision.raise_max is not None:
        headroom = min(headroom, max(0, decision.raise_max - decision.contribution))
    return allowance < headroom


def stage_purse_buckets(decisions: Sequence[Decision], gates: Mapping[str, Any]) -> dict:
    """The published decay claim, recomputed from the journal.

    ``PENDING_EDITS`` states the hero-purse cap "could bind on 58.6% of
    sub-near-nut sizing decisions at a ~2.6k purse, 30.4% at ~8.7k and
    4.3% at ~12k" -- the gate going inert exactly as the money it guards
    grows. This recomputes that decay so the audit's reading of the
    journal is checked against a number someone else published, rather
    than only against itself.
    """

    capped = [d for d in decisions if _sized_aggressively(d) and cap_applies(d, gates)]
    buckets: dict[str, dict] = {}
    edges = [(0, 4_000), (4_000, 10_000), (10_000, 10**9)]
    for low, high in edges:
        rows = [d for d in capped if low <= d.hero_stack < high]
        # "Could bind" = the cap bounds hero's further commitment BELOW
        # the chips actually at risk in the hand.
        #
        # Comparing the cap to the top of the legal raise range instead is
        # very nearly vacuous -- a 0.455 fraction of any denominator is
        # below the legal maximum almost always -- and it produces a flat
        # ~75% in every purse bucket, which is the signature of a
        # definition that is measuring nothing. The quantity the entry is
        # about is whether the gate still bounds EXPOSURE: once
        # `0.455 * purse` exceeds everything the opponent can pay, the cap
        # is arithmetic that cannot change an outcome.
        binds_hero = [d for d in rows if _bounds_exposure(d, gates, effective=False)]
        binds_effective = [d for d in rows if _bounds_exposure(d, gates, effective=True)]
        label = f"{low}-{high}" if high < 10**9 else f"{low}+"
        buckets[label] = {
            "sub_near_nut_sizing_decisions": len(rows),
            "hero_purse_cap_binds_pct": _rate(len(binds_hero), len(rows)),
            "effective_stack_cap_binds_pct": _rate(len(binds_effective), len(rows)),
        }
    return buckets


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------


def stage_exposure(
    decisions: Sequence[Decision],
    hand_deltas: Mapping[str, int],
    gates: Mapping[str, Any],
) -> dict:
    """Chip outcomes of the hands each edit would have touched.

    **Association, not a saving.** A clipped bet changes how the hand
    continues, and a journal cannot replay a hand that never happened.
    What is reported is the realised result of the hands on which the
    changed gate would have intervened, beside the result of all hands,
    so the reader can see whether the gate lands on the losses or
    scatters at random.
    """

    def touched(rows: Iterable[Decision]) -> set[str]:
        return {d.table_id for d in rows if d.table_id in hand_deltas}

    sized = [d for d in decisions if _sized_aggressively(d)]
    capped = [d for d in sized if cap_applies(d, gates)]
    clip_hands = touched(
        d for d in capped if (d.amount_to or 0) > risk_cap(d, gates, effective=True)
    )
    calls = [d for d in decisions if d.action == "call" and d.to_call > 0]
    # Refusal, not arrival. Attributing a hand to this edit because the
    # gate was merely *reached* overstates it by 72% on this journal.
    call_hands = touched(
        d
        for d in calls
        if call_gate_refuses(d, gates, effective=True)
        and not call_gate_refuses(d, gates, effective=False)
    )
    call_arrival_hands = touched(
        d
        for d in calls
        if call_gate_triggers(d, gates, effective=True)
        and not call_gate_triggers(d, gates, effective=False)
    )
    all_hands = touched(decisions)
    either = clip_hands | call_hands

    def summarise(hands: set[str], name: str) -> dict:
        deltas = [hand_deltas[h] for h in sorted(hands)]
        return {
            "name": name,
            "hands": len(deltas),
            "chip_total": sum(deltas),
            "chip_mean": round(statistics.mean(deltas), 1) if deltas else None,
            "chip_median": round(statistics.median(deltas), 1) if deltas else None,
            "worst": min(deltas) if deltas else None,
            "losing_hands": sum(1 for value in deltas if value < 0),
            "losing_pct": _rate(sum(1 for value in deltas if value < 0), len(deltas)),
        }

    return {
        "caveat": (
            "Association, not a counterfactual saving: clipping a bet changes "
            "how the hand continues and this journal cannot replay a hand that "
            "never happened. The selection is also not random -- these gates "
            "fire on the largest prices, and large prices are mechanically "
            "where large losses live, so some concentration is expected before "
            "any judgement about the gate."
        ),
        "refused_call_hands": sorted(
            (hand_deltas[h] for h in call_hands), reverse=True
        ),
        "reached_only_call_hands": sorted(
            (hand_deltas[h] for h in call_arrival_hands), reverse=True
        ),
        "all_hands": summarise(all_hands, "every hand with a recorded result"),
        "risk_cap_would_clip": summarise(clip_hands, "hands the changed cap touches"),
        "call_gate_would_refuse": summarise(
            call_hands, "hands the changed call gate would REFUSE (ceiling, w=0)"
        ),
        "call_gate_reaches_only": summarise(
            call_arrival_hands, "hands where the gate is merely REACHED"
        ),
        "either_edit": summarise(either, "hands either edit touches"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(args: argparse.Namespace) -> dict:
    journal = Path(args.journal)
    decisions, hand_deltas, counts = load_journal(journal, args.policy_version)
    if not decisions:
        raise SystemExit(f"no {args.policy_version} decisions found in {journal}")

    gates = load_gates(Path(args.manifest))
    instrument = stage_instrument(decisions, gates)
    report: dict[str, Any] = {
        "journal": str(journal),
        "policy_version": args.policy_version,
        "parse_accounting": counts,
        "hands_with_results": len(hand_deltas),
        "gates": gates,
        "instrument": instrument,
    }
    if not instrument["all_passed"]:
        report["binding"] = "NOT COMPUTED — instrument failed"
        return report

    report["binding"] = stage_binding(decisions, gates)
    report["purse_buckets"] = stage_purse_buckets(decisions, gates)
    report["exposure"] = stage_exposure(decisions, hand_deltas, gates)
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Gate binding audit — what the three changes would have done live")
    add("")
    add(
        f"`{report['journal']}` · policy `{report['policy_version']}` · "
        f"gates from `{report['gates']['source']}`"
    )
    add("")
    add(
        "> Read beside the battery number, never instead of it. This prices "
        "**how often** each edit changes a verdict and **which hands** it lands "
        "on. It is not a counterfactual EV."
    )
    add("")

    add("## 0. The instrument, before any result")
    add("")
    add("| check | result | verdict |")
    add("|---|---|---|")
    inst = report["instrument"]
    add(
        "| `effective_stack <= hero_stack` on every record | "
        f"{inst['effective_never_exceeds_hero']['violations']} violations | "
        f"{inst['effective_never_exceeds_hero']['verdict']} |"
    )
    equal = inst["equal_denominators_give_equal_verdicts"]
    add(
        "| equal denominators give equal verdicts | "
        f"{equal['differences']} differences over {equal['records']} records | "
        f"{equal['verdict']} |"
    )
    parity = inst["engine_parity_on_the_hero_purse_cap"]
    add(
        "| recorded bets respect the hero-purse cap (validates this "
        "tool's arithmetic against the engine that ran) | "
        f"{parity['over_the_hero_purse_cap']} over, of "
        f"{parity['sized_bets_checked']} sized bets "
        f"({parity['of_which_within_the_effective_stack_cap']} of them within "
        f"the effective-stack cap) | {parity['verdict']} |"
    )
    reveal = inst["reveal_expense_in_range"]
    add(
        "| reveal expense in [0,1], zero on the river | "
        f"{reveal['out_of_range']} out of range, {reveal['river_nonzero']} "
        f"non-zero rivers | {reveal['verdict']} |"
    )
    add("")
    counts = report["parse_accounting"]
    add(
        f"Parse accounting: {counts.get('lines', 0)} lines, "
        f"{counts.get('unparsable', 0)} unparsable, {counts.get('blank', 0)} blank, "
        f"{counts.get('event:decision', 0)} decision records, "
        f"{counts.get('event:hand_result', 0)} hand results, "
        f"{report['hands_with_results']} hands with a chip delta."
    )
    add("")
    inherited = report["gates"]["inherited_from_dataclass_default"]
    if inherited:
        add(
            "**The manifest does not name "
            + ", ".join(f"`{name}`" for name in inherited)
            + "**, so the served policy takes the dataclass default. That is "
            "the mechanism by which an unmeasured gate change ships under an "
            "approved artifact."
        )
        add("")

    if not inst["all_passed"]:
        add("**Instrument failed. No result computed.**")
        return "\n".join(lines) + "\n"

    binding = report["binding"]
    add("## 1. How often each edit changes anything")
    add("")
    denom = binding["denominator"]
    add(
        f"Across **{binding['decisions']}** recorded decisions the two "
        f"denominators disagree on **{denom['diverging']}** "
        f"({denom['diverging_pct']}%) — hero covering the table. Median hero "
        f"purse on those: {denom['median_hero_stack']}, median effective "
        f"stack {denom['median_effective_stack']}."
    )
    add("")
    add("| edit | population | changes the verdict | rate |")
    add("|---|---|---|---|")
    cap = binding["risk_cap"]
    add(
        f"| effective-stack risk cap | {cap['below_near_nut']} sub-near-nut "
        f"sized bets | {cap['would_clip']} clipped, of which "
        f"{cap['would_decline_entirely']} declined outright | "
        f"{cap['would_clip_pct_of_capped']}% |"
    )
    gate = binding["call_gates"]
    add(
        f"| effective-stack call gates | {gate['calls']} calls | "
        f"{gate['reaches_a_gate']} newly reach a stack gate "
        f"({gate['reaches_a_gate_distinct_decisions']} distinct decisions), of "
        f"which **at most {gate['would_refuse_ceiling']} are actually refused** | "
        f"{gate['would_refuse_pct']}% |"
    )
    slope = binding["reveal_slope"]
    add(
        f"| reveal-expense slope {slope['slope']} | "
        f"{slope['calls_with_expense']} calls with cards still to come | "
        f"{slope['calls_where_a_stack_gate_also_fires']} where a stack gate "
        "also fires (the slope raises a floor, it never creates a gate) | — |"
    )
    add("")
    add(
        f"Chips the cap removes: **{cap['chips_removed_total']}** total, "
        f"median {cap['chips_removed_median']}, largest single clip "
        f"{cap['chips_removed_max']}. Chips in the calls the gate would "
        f"actually refuse: **{gate['chips_at_stake_total']}**, largest single "
        f"call {gate['chips_at_stake_max']}."
    )
    add("")
    add(
        "> **Reaching a gate is not being refused by it.** The engine refuses "
        "only when `equity < required`; the stack-fraction trigger merely "
        "decides whether the floor is consulted at all. The refusal count is "
        f"a **ceiling**: {gate['refusal_is_a_ceiling_because']}."
    )
    add("")

    add("## 2. The decay the change was made to fix")
    add("")
    add(
        "`PENDING_EDITS` claims the hero-purse cap went inert as the bankroll "
        "grew. Recomputed here from the journal:"
    )
    add("")
    add(
        "> **This is corroboration, not validation, and the definition was "
        "chosen after seeing the alternative fail.** Under `_bounds_exposure` "
        "(the cap holds hero under the chips actually at risk) the decay is "
        "steep. Under the first definition tried (the cap sits below the legal "
        "raise maximum) it is **75.22 / 75.46 / 71.37 — flat, and identical "
        "for both denominators**, because a 0.455 fraction is below the legal "
        "max almost always. `_bounds_exposure` is the better definition and "
        "the reason is in its code comment, but a check picked after seeing "
        "the other one measure nothing cannot also serve as independent "
        "validation. The archived source buckets by *session*, not by these "
        "purse edges, so the agreement is directional only."
    )
    add("")
    add("| hero purse | sub-near-nut sized bets | hero-purse cap binds | effective-stack cap binds |")
    add("|---|---|---|---|")
    for label, row in report["purse_buckets"].items():
        add(
            f"| {label} | {row['sub_near_nut_sizing_decisions']} | "
            f"{row['hero_purse_cap_binds_pct']}% | "
            f"{row['effective_stack_cap_binds_pct']}% |"
        )
    add("")

    add("## 3. Which hands the edits land on")
    add("")
    add(f"> {report['exposure']['caveat']}")
    add("")
    add("| population | hands | total chips | mean | median | worst | losing |")
    add("|---|---|---|---|---|---|---|")
    for key in (
        "all_hands",
        "risk_cap_would_clip",
        "call_gate_would_refuse",
        "call_gate_reaches_only",
        "either_edit",
    ):
        row = report["exposure"][key]
        add(
            f"| {row['name']} | {row['hands']} | {row['chip_total']} | "
            f"{row['chip_mean']} | {row['chip_median']} | {row['worst']} | "
            f"{row['losing_hands']} ({row['losing_pct']}%) |"
        )
    add("")
    return "\n".join(lines) + "\n"


def _utf8_stdout() -> None:
    """Windows consoles default to cp1252, which cannot encode the report.

    The file is always written UTF-8; this only keeps the echoed copy
    from killing an otherwise finished run.
    """

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - stream without it
        pass


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--journal", default=DEFAULT_JOURNAL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--policy-version", default=DEFAULT_POLICY)
    parser.add_argument("--output", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    report = build_report(args)
    text = render_markdown(report)
    print(text)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        out.with_suffix(".md").write_text(text, encoding="utf-8")
        print(f"[gate_binding_audit] wrote {out} and {out.with_suffix('.md')}")
    return 0 if report["instrument"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
