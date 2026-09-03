"""Raw journal -> per-table ledger and gate-margin reconstruction, one policy.

Bust forensics need the live journal read **raw**. The official reader
(``engine.training_telemetry.load_training_examples``) refuses this
journal (defect 25: 104 duplicate ``hand_result`` rows), so a post-mortem
must parse the JSONL itself, tolerate and **count** the rows it cannot
use, and join decisions to settlements by ``table_id``.
``tools.gate_binding_audit.load_journal`` does that inline for one audit;
this module makes the reader reusable and adds the two reconstructions
every bust report needs:

* the **per-table ledger** -- which hands lost the money, street by
  street, joined to their ``hand_result`` chip deltas;
* the **proposal-versus-execution verdict** -- whether the engine
  executed the composed layer's ``proposed_branch`` literally, rendered
  it as a different action of the same family, or *overrode* it, and
  which record field witnesses the override; and
* the **rails gate margins** -- which safety gates bound and by how
  much, and whether the all-in denominator collapse recurred
  (``effective_stack_chips`` clamped to 1 while the hero still holds a
  stack -- the signature of ``engine.decision_engine._gate_stack``'s
  ``max(1, ...)`` when every active opponent is all-in).

**Why the override verdict is its own mechanism.** The first draft of
this reader compared ``proposed_branch`` against ``action`` only as a
frequency table and never named the fields that can move one off the
other. It therefore reported the S17 bust hand as "executed literally on
all four streets", which is false: the preflop action was a bluff
override (``bluff_kind == "steal"``,
``engine/decision_engine.py``, the bluff-advisor block of
``decide_with_diagnostics``), and ``bluff_kind`` was
not read at all. Two further traps the frequency table walks into:

* ``proposed_family`` is **not** a proposal. It is
  ``DecisionResult.family``, which the bluff path rewrites to
  ``"aggress"`` *before* the record is built (the bluff-advisor block
  of ``decide_with_diagnostics``), so on an overridden row it already
  equals the executed family. ``proposed_branch`` is the only recorded proposal.
* the branch's action is state-dependent. ``active`` is a call at a
  price and a bet unprovoked, so "which action IS this branch" must come
  from ``engine.branch_contract_v9.branch_action``, never from a fixed
  branch-to-action table.

:data:`OVERRIDE_FIELDS` enumerates every field on the schema-3 decision
record that can move the executed action off the proposal, each with the
engine site that writes it; the classification consults all of them and
counts the rows no field explains.

Gate constants come from ``engine.decision_engine.SafetyGates`` -- the
served defaults (STATUS section 4: no manifest names them) -- and the
per-street call margins are copied from
``engine.decision_engine._CALL_MARGINS`` and pinned by
``tests/test_session_postmortem.py`` so a drift in either is a test
failure, not a silent skew. Board tiers come from
``engine.hand_strength.board_improvement``, the same pure helper the
engine uses.

This is a **reader only**: read-only on the journal, no Arena, no
promotion, stdlib-only, and it never imports the telemetry module's
official reader.

Example::

    python -m tools.session_postmortem \\
        --journal .arena-training.jsonl \\
        --policy-version candidate-v9-0003b \\
        --output artifacts/evaluations/s17-bust-postmortem-2026-09-03.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.branch_contract_v9 import (
    BranchContractError,
    branch_action,
    branch_engine_family,
)
from engine.decision_engine import SafetyGates
from engine.hand_strength import board_improvement

DEFAULT_JOURNAL = ".arena-training.jsonl"

#: Per-street call margin added to pot odds by
#: ``DecisionEngine._call_clears_margin``. Copied from
#: ``engine.decision_engine._CALL_MARGINS`` rather than imported so this
#: module states its own arithmetic; the test pins the two together.
STREET_CALL_MARGINS = {"preflop": 0.0, "flop": 0.02, "turn": 0.05, "river": 0.08}

#: Board cards still to come, by street, for the reveal-expense slope.
REVEALS_REMAINING = {"preflop": 3, "flop": 2, "turn": 1, "river": 0}

#: Executed action -> the frozen engine family. Copied from
#: ``engine.training_telemetry.action_family`` rather than imported so
#: this module states its own arithmetic (as with the call margins); the
#: test pins the two together.
ACTION_FAMILY = {
    "fold": "fold",
    "check": "check_call",
    "call": "check_call",
    "bet": "aggress",
    "raise": "aggress",
    "all-in": "aggress",
}

#: Families ordered by how much of the stack they commit, so an override
#: has a direction: a demotion is safe-ward, a promotion is not.
_FAMILY_RANK = {"fold": 0, "check_call": 1, "aggress": 2}


@dataclass(frozen=True, slots=True)
class OverrideField:
    """One decision-record field that can move execution off the proposal.

    ``engine_site`` is where the field is written, so a reader can check
    the claim instead of trusting this table. ``changes_action`` says
    whether the field can change *which* action is submitted (as opposed
    to only its size, its inputs, or its fate after submission).
    """

    name: str
    engine_site: str
    effect: str
    changes_action: bool


#: Every field on the schema-3 decision record
#: (``engine/training_telemetry.py::make_decision_record``) that can move
#: the executed action off the composed proposal. Enumerated from that
#: function and the sites that write each value -- not guessed. A row
#: whose verdict is ``override`` and whose witness list is empty is
#: counted as ``unexplained``: the rails' own demotion ladder
#: (``_aggressive_action`` -> ``_passive_action`` -> fold) leaves nothing
#: on the record, which is itself a reportable instrument gap.
OVERRIDE_FIELDS: tuple[OverrideField, ...] = (
    OverrideField(
        name="bluff_kind",
        engine_site=(
            "engine/decision_engine.py::DecisionEngine."
            "decide_with_diagnostics, the bluff-advisor block "
            "(lines 1339-1353 on 2026-09-03)"
        ),
        effect=(
            "a non-aggress composed family may be replaced by a sized "
            "bet/raise from the bluff advisor; family is relabelled "
            "aggress, so this is the ONLY witness on the record"
        ),
        changes_action=True,
    ),
    OverrideField(
        name="fallback_reason",
        engine_site=(
            "engine/decision_engine.py::DecisionEngine."
            "decide_with_diagnostics, the deadline branch, and "
            "engine/training_telemetry.py::make_decision_record's "
            "deadline_fallback promotion "
            "(lines 1263-1279 and 418-419 on 2026-09-03)"
        ),
        effect=(
            "the deadline path executes _deadline_action and returns "
            "family 'deadline' with proposed_branch None: the submitted "
            "action is not the composition's at all"
        ),
        changes_action=True,
    ),
    OverrideField(
        name="rule_verdicts",
        engine_site=(
            "engine/decision_engine.py::DecisionEngine._record_verdict "
            "and _take_rule_verdicts "
            "(lines 1137-1152, 1371 on 2026-09-03)"
        ),
        effect=(
            "the C1-C5 dials record what fired with their inputs; a "
            "fired dial is the record's only witness that a rule moved "
            "the action or its size"
        ),
        changes_action=True,
    ),
    OverrideField(
        name="hyper_aggression",
        engine_site=(
            "engine/decision_engine.py::DecisionEngine._hyper_roll and "
            "its readers in _consider_bluff / _equity_family / "
            "_sized_action (lines 853, 1253, 1409, 1656-1657 on "
            "2026-09-03)"
        ),
        effect=(
            "anti-modeling noise: drops the aggression floor, targets "
            "the full pot for sizing and swaps the bluff settings"
        ),
        changes_action=True,
    ),
    OverrideField(
        name="proposed_branch",
        engine_site=(
            "engine/decision_engine.py::DecisionEngine."
            "decide_with_diagnostics, the per-decision reset and the "
            "decide_forced pin (lines 1260, 1300-1308 on 2026-09-03)"
        ),
        effect=(
            "null means no composed proposal is on the row -- the "
            "composition never ran, or decide_forced's pin replaced the "
            "argmax and nulled it; such a row is unclassifiable, never "
            "'executed literally'"
        ),
        changes_action=True,
    ),
    OverrideField(
        name="proposed_risk_fraction",
        engine_site=(
            "engine/training_telemetry.py::make_decision_record, the "
            "submitted_risk_fraction computation and the two record "
            "keys (lines 400, 557-560 on 2026-09-03)"
        ),
        effect=(
            "the size the composition asked for against "
            "submitted_risk_fraction, the size that survived the risk "
            "cap and the legality clamps; size only, never the action"
        ),
        changes_action=False,
    ),
    OverrideField(
        name="accepted",
        engine_site=(
            "engine/training_telemetry.py::make_decision_record, the "
            "accepted computation and the response keys "
            "(lines 420, 563-567 on 2026-09-03)"
        ),
        effect=(
            "with action_status and response_error: the Arena's verdict "
            "on the submitted action, so a non-2xx row names an action "
            "the table never executed"
        ),
        changes_action=False,
    ),
    OverrideField(
        name="belief_degraded",
        engine_site=(
            "engine/decision_engine.py::DecisionEngine."
            "decide_with_diagnostics, the belief-degrade reset and "
            "carry-out (lines 1261, 1377-1378 on 2026-09-03)"
        ),
        effect=(
            "with belief_degrade_reason: the serving belief provider "
            "degraded, so the composition's inputs were not the nominal "
            "ones; it moves WHICH branch is proposed, and the proposal "
            "on the row already reflects it"
        ),
        changes_action=False,
    ),
)


def _int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class Decision:
    """One recorded live decision, reduced to what the ledger and gates read."""

    table_id: str
    recorded_at_ms: int
    street: str
    big_blind: int
    hero_stack: int
    effective_stack: int
    contribution: int
    pot_chips: int
    to_call: int
    equity: float | None
    action: str
    amount_to: int | None
    proposed_branch: str | None
    #: ``DecisionResult.family`` AFTER the bluff path may have rewritten
    #: it to ``aggress`` (the bluff-advisor block of
    #: ``DecisionEngine.decide_with_diagnostics``). Never read
    #: it as the proposal -- ``proposed_branch`` is the proposal.
    proposed_family: str | None
    submitted_risk_fraction: float | None
    proposed_risk_fraction: float | None
    hole_cards: tuple[str, ...]
    board_cards: tuple[str, ...]
    legal_actions: tuple[str, ...]
    rule_verdicts_present: bool
    fallback_reason: str | None
    action_status: int | None
    accepted: bool
    hyper_aggression: bool
    belief_degraded: bool
    training_eligible: bool
    # Override witnesses (OVERRIDE_FIELDS). Defaulted so the older
    # keyword construction still builds a row, and always passed
    # explicitly by ``load_journal``.
    bluff_kind: str | None = None
    rule_verdict_count: int = 0
    belief_degrade_reason: str | None = None
    response_error: str | None = None
    identity_verified: bool = True


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


def load_journal(
    path: str | Path, policy_version: str
) -> tuple[list[Decision], dict[str, dict[str, object]], dict[str, int]]:
    """Raw decisions for one policy, the hand ledger, and parse accounting.

    Modelled on ``tools.gate_binding_audit.load_journal``: malformed lines
    are skipped **with a count**, never swallowed silently. Duplicate
    ``hand_result`` rows (defect 25) are counted and resolved last-write-
    wins, matching the audit's behaviour. Returns decisions for the one
    policy, the full hand ledger (every settled table, any policy), and
    the accounting counters.
    """

    decisions: list[Decision] = []
    hand_results: dict[str, dict[str, object]] = {}
    counts: Counter[str] = Counter()

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            counts["lines"] += 1
            line = line.strip()
            if not line:
                counts["blank"] += 1
                continue
            try:
                record = json.loads(line)
            except ValueError:
                counts["unparsable"] += 1
                continue

            event = record.get("event")
            counts[f"event:{event}"] += 1

            if event == "hand_result":
                table_id = str(record.get("table_id") or "")
                delta = _int(record.get("chip_delta_chips"))
                if table_id and delta is not None:
                    if table_id in hand_results:
                        counts["hand_result:duplicate"] += 1
                    hand_results[table_id] = {
                        "table_id": table_id,
                        "chip_delta_chips": delta,
                        "reward_bb": record.get("reward_bb"),
                        "settled_at_ms": _int(record.get("settled_at_ms")),
                        "big_blind_chips": _int(record.get("big_blind_chips")),
                    }
                continue

            if event != "decision":
                counts["event:other"] += 1
                continue
            if record.get("policy_version") != policy_version:
                counts["decision:other_policy"] += 1
                continue

            state = record.get("state") or {}
            legal = record.get("legal") or {}
            hero_stack = _int(state.get("hero_stack_chips"))
            effective = _int(state.get("effective_stack_chips"))
            pot_chips = _int(state.get("pot_chips"))
            recorded_at = _int(record.get("recorded_at_ms"))
            if hero_stack is None or effective is None or pot_chips is None:
                counts["decision:missing_stacks"] += 1
                continue
            if recorded_at is None:
                counts["decision:missing_timestamp"] += 1
                continue

            decisions.append(
                Decision(
                    table_id=str(record.get("table_id") or ""),
                    recorded_at_ms=recorded_at,
                    street=str(record.get("street") or "").casefold(),
                    big_blind=_int(record.get("big_blind_chips"), 1) or 1,
                    hero_stack=hero_stack,
                    effective_stack=effective,
                    contribution=_int(state.get("hero_contribution_chips"), 0) or 0,
                    pot_chips=pot_chips,
                    to_call=_int(legal.get("call_chips"), 0) or 0,
                    equity=record.get("equity"),
                    action=str(record.get("action") or ""),
                    amount_to=_int(record.get("amount_to")),
                    proposed_branch=record.get("proposed_branch"),
                    proposed_family=record.get("proposed_family"),
                    submitted_risk_fraction=record.get("submitted_risk_fraction"),
                    proposed_risk_fraction=record.get("proposed_risk_fraction"),
                    hole_cards=tuple(state.get("hole_cards") or ()),
                    board_cards=tuple(state.get("board_cards") or ()),
                    legal_actions=tuple(str(a) for a in (legal.get("actions") or [])),
                    rule_verdicts_present=bool(record.get("rule_verdicts")),
                    fallback_reason=record.get("fallback_reason"),
                    action_status=_int(record.get("action_status")),
                    accepted=bool(record.get("accepted")),
                    hyper_aggression=bool(record.get("hyper_aggression")),
                    belief_degraded=bool(record.get("belief_degraded")),
                    training_eligible=bool(record.get("training_eligible")),
                    bluff_kind=record.get("bluff_kind"),
                    rule_verdict_count=len(record.get("rule_verdicts") or ()),
                    belief_degrade_reason=record.get("belief_degrade_reason"),
                    response_error=record.get("response_error"),
                    identity_verified=bool(record.get("identity_verified")),
                )
            )

    return decisions, hand_results, dict(counts)


# ---------------------------------------------------------------------------
# Proposal versus execution
# ---------------------------------------------------------------------------


def override_witnesses(decision: Decision) -> list[str]:
    """Which :data:`OVERRIDE_FIELDS` are actually set on this row.

    Only the fields that can change WHICH action was submitted; the
    size-only and post-submission fields are reported separately by
    :func:`override_analysis` so they are never mistaken for a cause.
    """

    witnesses: list[str] = []
    if decision.bluff_kind:
        witnesses.append(f"bluff_kind={decision.bluff_kind}")
    if decision.fallback_reason:
        witnesses.append(f"fallback_reason={decision.fallback_reason}")
    if decision.rule_verdict_count:
        witnesses.append(f"rule_verdicts={decision.rule_verdict_count}")
    if decision.hyper_aggression:
        witnesses.append("hyper_aggression=true")
    return witnesses


def classify_execution(decision: Decision) -> dict[str, object]:
    """Did the engine execute the proposed branch, render it, or override it?

    ``literal`` -- the executed action IS the branch's action at this
    price (``branch_action``). ``rendering`` -- a different action of the
    same engine family: the Arena names an unprovoked wager ``raise`` at
    a blind-option spot, and an escalation at stack-reaching size renders
    as ``all-in``; neither changes what the branch decided to do.
    ``override`` -- a different engine family, i.e. the composition asked
    for one thing and something else was submitted. ``unclassifiable`` --
    no proposal on the row (``proposed_branch`` null), or a branch this
    state masks, which the contract refuses to project.
    """

    branch = decision.proposed_branch
    executed = decision.action
    executed_family = ACTION_FAMILY.get(executed)
    row: dict[str, object] = {
        "table_id": decision.table_id,
        "street": decision.street,
        "recorded_at_ms": decision.recorded_at_ms,
        "proposed_branch": branch,
        "to_call_chips": decision.to_call,
        "executed_action": executed,
        "executed_family": executed_family,
        "amount_to": decision.amount_to,
        # Post-override: equals the executed family on a bluff row. Kept
        # so a reader can SEE it disagree with the branch.
        "proposed_family": decision.proposed_family,
        "bluff_kind": decision.bluff_kind,
        "witnesses": override_witnesses(decision),
    }
    if not branch:
        row.update(
            canonical_action=None,
            canonical_family=None,
            verdict="unclassifiable",
            reason="proposed_branch is null: no composed proposal on this row",
            direction=None,
        )
        return row
    try:
        canonical = branch_action(branch, decision.to_call)
        canonical_family = branch_engine_family(branch, decision.to_call)
    except BranchContractError as exc:
        row.update(
            canonical_action=None,
            canonical_family=None,
            verdict="unclassifiable",
            reason=f"branch masked by this state: {exc}",
            direction=None,
        )
        return row

    row["canonical_action"] = canonical
    row["canonical_family"] = canonical_family
    if executed == canonical:
        row.update(verdict="literal", reason="executed action is the branch's action", direction=None)
        return row
    if executed_family == canonical_family:
        row.update(
            verdict="rendering",
            reason=(
                f"same engine family {canonical_family!r}: the Arena named "
                f"this action {executed!r}, not {canonical!r}"
            ),
            direction=None,
        )
        return row

    rank_from = _FAMILY_RANK.get(canonical_family, -1)
    rank_to = _FAMILY_RANK.get(executed_family, -1)
    row.update(
        verdict="override",
        reason=(
            f"branch {branch!r} executes {canonical!r} ({canonical_family}) "
            f"at to_call {decision.to_call}; submitted {executed!r} "
            f"({executed_family})"
        ),
        direction="promotion" if rank_to > rank_from else "demotion",
    )
    return row


def override_analysis(decisions: list[Decision]) -> dict[str, object]:
    """Proposal-versus-execution verdicts, with the witness accounting.

    Every non-literal row is listed in full: an override the record can
    explain (a witness field is set) and an override it cannot
    (``unexplained``) are different findings, and collapsing them is how
    the first draft of this report missed the bust hand's bluff.
    """

    rows = [classify_execution(decision) for decision in decisions]
    non_literal = [row for row in rows if row["verdict"] != "literal"]
    overrides = [row for row in rows if row["verdict"] == "override"]

    by_witness: Counter[str] = Counter()
    for row in overrides:
        witnesses = row["witnesses"]
        if witnesses:
            for witness in witnesses:
                by_witness[witness] += 1
        else:
            by_witness["unexplained"] += 1

    # The bluff signature, stated independently of ``bluff_kind``: the
    # record's two "proposal" fields disagree, because the family was
    # rewritten after the branch was recorded.
    family_disagreements = [
        row
        for row in rows
        if row["canonical_family"] is not None
        and row["proposed_family"] != row["canonical_family"]
    ]

    return {
        "fields_enumerated_from": (
            "engine/training_telemetry.py::make_decision_record "
            "(telemetry_schema_version 3)"
        ),
        "override_fields": [
            {
                "name": field.name,
                "engine_site": field.engine_site,
                "effect": field.effect,
                "changes_action": field.changes_action,
                "rows_set": _rows_set(field.name, decisions),
            }
            for field in OVERRIDE_FIELDS
        ],
        "verdicts": _counted(Counter(row["verdict"] for row in rows)),
        "override_directions": _counted(
            Counter(row["direction"] for row in overrides)
        ),
        "override_by_witness": _counted(by_witness),
        "overrides": len(overrides),
        "overrides_unexplained": sum(1 for row in overrides if not row["witnesses"]),
        "proposed_family_disagrees_with_branch": {
            "count": len(family_disagreements),
            "tables": sorted({str(row["table_id"]) for row in family_disagreements}),
        },
        "non_literal_rows": non_literal,
    }


def _counted(values: Counter[object]) -> dict[str, int]:
    """A counter as JSON keys, with ``None`` named rather than dropped.

    ``json.dumps(..., sort_keys=True)`` raises on a dict mixing ``None``
    and ``str`` keys, and a counter over an optional field (``bluff_kind``
    is null on most rows) is exactly that. The absent bucket is reported
    as ``"none"`` -- never omitted, because "354 rows had no bluff" is
    the accounting.
    """

    return {("none" if key is None else str(key)): count for key, count in values.items()}


def _rows_set(name: str, decisions: list[Decision]) -> int:
    """How many rows carry a non-default value for one override field."""

    if name == "bluff_kind":
        return sum(1 for d in decisions if d.bluff_kind)
    if name == "fallback_reason":
        return sum(1 for d in decisions if d.fallback_reason)
    if name == "rule_verdicts":
        return sum(1 for d in decisions if d.rule_verdict_count)
    if name == "hyper_aggression":
        return sum(1 for d in decisions if d.hyper_aggression)
    if name == "proposed_branch":
        # The reportable state is the ABSENT one.
        return sum(1 for d in decisions if not d.proposed_branch)
    if name == "proposed_risk_fraction":
        return sum(1 for d in decisions if d.proposed_risk_fraction is not None)
    if name == "accepted":
        return sum(
            1
            for d in decisions
            if not d.accepted
            or d.action_status is None
            or not 200 <= d.action_status < 300
            or d.response_error
        )
    if name == "belief_degraded":
        return sum(1 for d in decisions if d.belief_degraded or d.belief_degrade_reason)
    raise KeyError(f"no row counter for override field {name!r}")


# ---------------------------------------------------------------------------
# Ledger and windows
# ---------------------------------------------------------------------------


def build_ledger(
    decisions: list[Decision], hand_results: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    """One row per table: decisions in time order, joined to its settlement."""

    tables: dict[str, list[Decision]] = {}
    for decision in decisions:
        tables.setdefault(decision.table_id, []).append(decision)

    ledger: list[dict[str, object]] = []
    for table_id, table_decisions in tables.items():
        ordered = sorted(table_decisions, key=lambda d: d.recorded_at_ms)
        settlement = hand_results.get(table_id)
        ledger.append(
            {
                "table_id": table_id,
                "first_decision_at_ms": ordered[0].recorded_at_ms,
                "last_decision_at_ms": ordered[-1].recorded_at_ms,
                "n_decisions": len(ordered),
                "hand_delta_chips": settlement.get("chip_delta_chips")
                if settlement
                else None,
                "reward_bb": settlement.get("reward_bb") if settlement else None,
                "settled_at_ms": settlement.get("settled_at_ms") if settlement else None,
                "decisions": [_decision_row(d) for d in ordered],
            }
        )
    ledger.sort(key=lambda row: row["first_decision_at_ms"])
    return ledger


def _decision_row(decision: Decision) -> dict[str, object]:
    execution = classify_execution(decision)
    return {
        "recorded_at_ms": decision.recorded_at_ms,
        "street": decision.street,
        "action": decision.action,
        "amount_to": decision.amount_to,
        # Proposal-vs-execution, on every ledger row: reading a hand
        # street by street must show an override where there was one.
        "execution_verdict": execution["verdict"],
        "canonical_action": execution["canonical_action"],
        "override_direction": execution["direction"],
        "override_witnesses": execution["witnesses"],
        "bluff_kind": decision.bluff_kind,
        "rule_verdict_count": decision.rule_verdict_count,
        "belief_degrade_reason": decision.belief_degrade_reason,
        "response_error": decision.response_error,
        "identity_verified": decision.identity_verified,
        "pot_chips": decision.pot_chips,
        "to_call_chips": decision.to_call,
        "hero_stack_chips": decision.hero_stack,
        "effective_stack_chips": decision.effective_stack,
        "hero_contribution_chips": decision.contribution,
        "equity": decision.equity,
        "proposed_branch": decision.proposed_branch,
        "proposed_family": decision.proposed_family,
        "submitted_risk_fraction": decision.submitted_risk_fraction,
        "proposed_risk_fraction": decision.proposed_risk_fraction,
        "hole_cards": list(decision.hole_cards),
        "board_cards": list(decision.board_cards),
        "legal_actions": list(decision.legal_actions),
        "rule_verdicts_present": decision.rule_verdicts_present,
        "fallback_reason": decision.fallback_reason,
        "action_status": decision.action_status,
        "accepted": decision.accepted,
        "hyper_aggression": decision.hyper_aggression,
        "belief_degraded": decision.belief_degraded,
        "training_eligible": decision.training_eligible,
    }


def _parse_iso(value: str) -> int:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return int(stamp.timestamp() * 1000)


def assign_windows(
    ledger: list[dict[str, object]], windows: list[tuple[str, str, str]]
) -> list[dict[str, object]]:
    """Assign each table to a window by its first decision; sum its delta.

    ``windows`` is a list of ``(name, start_iso, end_iso)``. Tables are
    assigned by ``first_decision_at_ms``, matching how a session log
    attributes its actions. The per-window delta sum is over the tables'
    ``hand_result`` rows, so a table whose settlement lands after the
    window edge still contributes to the window that saw its decisions.
    """

    bounds = [(name, _parse_iso(start), _parse_iso(end)) for name, start, end in windows]
    rows: list[dict[str, object]] = []
    for (name, start_ms, end_ms), (_, start_iso, end_iso) in zip(bounds, windows):
        tables = [
            row
            for row in ledger
            if start_ms <= row["first_decision_at_ms"] < end_ms
        ]
        deltas = [
            row["hand_delta_chips"] for row in tables if row["hand_delta_chips"] is not None
        ]
        rows.append(
            {
                "name": name,
                "start": start_iso,
                "end": end_iso,
                "n_decisions": sum(row["n_decisions"] for row in tables),
                "n_tables": len(tables),
                "n_settled": len(deltas),
                "delta_sum_chips": sum(deltas),
                "tables_without_settlement": sum(
                    1 for row in tables if row["hand_delta_chips"] is None
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Gate reconstruction
# ---------------------------------------------------------------------------


def reconstruct_gates(
    decisions: list[Decision], gates: SafetyGates | None = None
) -> dict[str, object]:
    """Per-decision rails margins, recomputed from what the record carries.

    Only the served gates are reconstructed (STATUS section 4; C1-C5 rule
    layer is dark, so ``rule_verdicts`` is expected null). Two engine
    inputs are NOT in the record -- the temperature-shaping boldness and
    the opponent tracker's wildness -- so the price margins below are the
    NEUTRAL form (no temperature shift, no wildness blend); both only
    move a margin, and the report says so.
    """

    if gates is None:
        gates = SafetyGates()

    calls: list[dict[str, object]] = []
    risk_caps: list[dict[str, object]] = []
    all_ins: list[dict[str, object]] = []
    collapses: list[dict[str, object]] = []

    for decision in decisions:
        row_base = {
            "table_id": decision.table_id,
            "street": decision.street,
            "recorded_at_ms": decision.recorded_at_ms,
            "equity": decision.equity,
            "action": decision.action,
            "amount_to": decision.amount_to,
            "hero_stack_chips": decision.hero_stack,
            "effective_stack_chips": decision.effective_stack,
            "hole_cards": list(decision.hole_cards),
            "board_cards": list(decision.board_cards),
            # A gate row that does not say whether the action it gated
            # was the proposal or an override cannot answer "did the
            # rails execute this literally?".
            "proposed_branch": decision.proposed_branch,
            "execution_verdict": classify_execution(decision)["verdict"],
            "bluff_kind": decision.bluff_kind,
        }
        equity = decision.equity
        to_call = decision.to_call
        effective = max(1, decision.effective_stack)

        if decision.action == "call" and equity is not None and to_call > 0:
            pot_odds = to_call / max(decision.pot_chips + to_call, 1)
            tier = board_improvement(decision.hole_cards, decision.board_cards)
            board_margin = gates.board_margin(tier)
            price = (
                pot_odds
                + STREET_CALL_MARGINS[decision.street]
                + board_margin
            )
            reveal = (
                gates.reveal_expense_equity_slope
                * min(1.0, to_call / effective)
                * (REVEALS_REMAINING[decision.street] / 3.0)
            )
            stack_gates = [
                {
                    "stack_fraction": fraction,
                    "equity_floor": floor,
                    "triggered": to_call >= fraction * effective,
                    "margin_vs_required": equity - (floor + reveal),
                }
                for fraction, floor in gates.call_stack_gates
            ]
            board_gate = gates.board_stack_gate(tier)
            board_gate_row = None
            if board_gate is not None:
                trigger, floor = board_gate
                board_gate_row = {
                    "tier": tier,
                    "stack_fraction": trigger,
                    "equity_floor": floor,
                    "triggered": to_call >= trigger * effective,
                    "margin_vs_required": equity - (floor + reveal),
                }
            calls.append(
                {
                    **row_base,
                    "to_call_chips": to_call,
                    "pot_chips": decision.pot_chips,
                    "board_tier": tier,
                    "pot_odds": pot_odds,
                    "price": price,
                    "margin_vs_price": equity - price,
                    "stack_gates": stack_gates,
                    "board_stack_gate": board_gate_row,
                }
            )

        if (
            decision.action in ("bet", "raise")
            and equity is not None
            and equity < gates.near_nut_floor
            and decision.amount_to is not None
        ):
            ceiling = decision.contribution + max(
                decision.big_blind,
                round(gates.risk_cap_stack_fraction * effective),
            )
            margin_chips = ceiling - decision.amount_to
            risk_caps.append(
                {
                    **row_base,
                    "risk_cap_stack_fraction": gates.risk_cap_stack_fraction,
                    "cap_ceiling_chips": ceiling,
                    "margin_chips": margin_chips,
                    "at_ceiling": margin_chips == 0,
                    "above_ceiling": margin_chips < 0,
                }
            )

        if decision.action == "all-in" and equity is not None:
            all_ins.append(
                {
                    **row_base,
                    "near_nut_floor": gates.near_nut_floor,
                    "margin_vs_near_nut": equity - gates.near_nut_floor,
                    "legal_actions": list(decision.legal_actions),
                }
            )

        if decision.effective_stack == 1 and decision.hero_stack > 1:
            collapses.append(
                {
                    "table_id": decision.table_id,
                    "street": decision.street,
                    "recorded_at_ms": decision.recorded_at_ms,
                    "action": decision.action,
                    "hero_stack_chips": decision.hero_stack,
                    "effective_stack_chips": decision.effective_stack,
                    "to_call_chips": decision.to_call,
                    "submitted_risk_fraction": decision.submitted_risk_fraction,
                    "equity": decision.equity,
                    "proposed_branch": decision.proposed_branch,
                    "execution_verdict": classify_execution(decision)["verdict"],
                    "bluff_kind": decision.bluff_kind,
                }
            )

    return {
        "calls": calls,
        "risk_caps": risk_caps,
        "all_ins": all_ins,
        "denominator_collapses": collapses,
        "served_gates": {
            "risk_cap_stack_fraction": gates.risk_cap_stack_fraction,
            "near_nut_floor": gates.near_nut_floor,
            "call_stack_gates": list(gates.call_stack_gates),
            "reveal_expense_equity_slope": gates.reveal_expense_equity_slope,
            "board_stackoff_kicker": list(gates.board_stackoff_kicker),
            "board_stackoff_thin": list(gates.board_stackoff_thin),
        },
        "not_in_record": [
            "temperature shaping boldness (moves the call margin)",
            "opponent tracker wildness (blends gate floors)",
            "engine equity top_fraction (record equity is the post-conditioning estimate)",
        ],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    journal: str | Path,
    policy_version: str,
    windows: list[tuple[str, str, str]] | None = None,
) -> dict[str, object]:
    """Full report: parse accounting, counters, windows, ledger, gates."""

    decisions, hand_results, counts = load_journal(journal, policy_version)
    ledger = build_ledger(decisions, hand_results)
    gate_analysis = reconstruct_gates(decisions)
    overrides = override_analysis(decisions)

    played_tables = {row["table_id"] for row in ledger}
    stray_settlements = [
        settlement
        for table_id, settlement in hand_results.items()
        if table_id not in played_tables
    ]
    stray_delta = sum(
        settlement["chip_delta_chips"] for settlement in stray_settlements
    )

    counters = {
        "decisions": len(decisions),
        "tables": len(ledger),
        "proposed_branch": _counted(Counter(d.proposed_branch for d in decisions)),
        # Post-override (the bluff-advisor block). Never a proposal on a
        # bluff row -- see OVERRIDE_FIELDS and override_analysis.
        "proposed_family_post_override": _counted(
            Counter(d.proposed_family for d in decisions)
        ),
        "executed_action": _counted(Counter(d.action for d in decisions)),
        "execution_verdict": overrides["verdicts"],
        "bluff_kind": _counted(Counter(d.bluff_kind for d in decisions)),
        "rule_verdicts_present": sum(1 for d in decisions if d.rule_verdicts_present),
        "proposed_risk_fraction_present": sum(
            1 for d in decisions if d.proposed_risk_fraction is not None
        ),
        "fallback_reason_present": sum(
            1 for d in decisions if d.fallback_reason is not None
        ),
        "accepted": sum(1 for d in decisions if d.accepted),
        "rejected": sum(1 for d in decisions if not d.accepted),
        "action_status_not_2xx": sum(
            1 for d in decisions if d.action_status is None or not 200 <= d.action_status < 300
        ),
        "hyper_aggression": sum(1 for d in decisions if d.hyper_aggression),
        "belief_degraded": sum(1 for d in decisions if d.belief_degraded),
        "training_eligible": sum(1 for d in decisions if d.training_eligible),
    }

    report: dict[str, object] = {
        "policy_version": policy_version,
        "journal": str(Path(journal).resolve()),
        "parse_accounting": counts,
        "counters": counters,
        "hand_result_rows_without_decision_table": {
            "count": len(stray_settlements),
            "delta_sum_chips": stray_delta,
        },
        "override_analysis": overrides,
        "gate_analysis": gate_analysis,
    }
    if windows:
        report["windows"] = assign_windows(ledger, windows)
    report["ledger"] = ledger
    return report


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - stream without it
        pass


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--journal", default=DEFAULT_JOURNAL)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument(
        "--windows",
        nargs="*",
        metavar="NAME:START_ISO:END_ISO",
        help="optional session windows, e.g. session-001:2026-09-02T05:33:17Z:2026-09-02T11:33:22Z",
    )
    parser.add_argument("--output", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    windows = None
    if args.windows:
        windows = []
        for spec in args.windows:
            name, rest = spec.split(":", 1)
            split_at = rest.index("Z:") + 1
            start, end = rest[:split_at], rest[split_at + 1 :]
            windows.append((name, start, end))

    report = build_report(args.journal, args.policy_version, windows)
    summary = (
        f"decisions={report['counters']['decisions']} tables="
        f"{report['counters']['tables']} unparsable="
        f"{report['parse_accounting'].get('unparsable', 0)} "
        f"duplicate_hand_results="
        f"{report['parse_accounting'].get('hand_result:duplicate', 0)} "
        f"rule_verdicts_present={report['counters']['rule_verdicts_present']} "
        f"collapses={len(report['gate_analysis']['denominator_collapses'])}"
    )
    print(f"[session_postmortem] {summary}")
    overrides = report["override_analysis"]
    print(
        f"[session_postmortem] execution={overrides['verdicts']} "
        f"overrides={overrides['overrides']} "
        f"unexplained={overrides['overrides_unexplained']} "
        f"by_witness={overrides['override_by_witness']}"
    )
    if report.get("windows"):
        for row in report["windows"]:
            print(
                f"  window {row['name']}: decisions={row['n_decisions']} "
                f"tables={row['n_tables']} settled={row['n_settled']} "
                f"delta_sum_chips={row['delta_sum_chips']}"
            )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[session_postmortem] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
