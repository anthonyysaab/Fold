"""Checks for the session post-mortem journal-to-ledger reader.

The reader restates the served call/risk-cap gates as arithmetic over
stored journal fields instead of driving the engine, and it reads the
live journal raw because the official reader refuses it (defect 25).
These tests buy the restated arithmetic back: every copied constant is
pinned against the module it came from, and every reconstruction is
verified on a synthetic journal in a temp dir whose sums are known
before the tool runs -- an impossible-by-construction check, not a
preference.

:class:`OverrideDetectionTests` exists because the first draft of the
reader could not see a bluff override and the report it produced said
the S17 bust hand "executed the proposal literally". Its rows are the
shape of that hand: a ``proposed_branch`` of ``active`` facing a price
(so the branch's action is a call) executed as a raise with
``bluff_kind == "steal"``. Every one of those tests fails on a reader
that does not carry ``bluff_kind`` and does not project the branch
through ``engine.branch_contract_v9.branch_action``.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from engine import game_state, training_telemetry
from engine.decision_engine import DecisionResult, SafetyGates
from tools.session_postmortem import (
    ACTION_FAMILY,
    OVERRIDE_FIELDS,
    REVEALS_REMAINING,
    STREET_CALL_MARGINS,
    Decision,
    assign_windows,
    build_ledger,
    build_report,
    classify_execution,
    load_journal,
    main,
    override_analysis,
    override_witnesses,
    reconstruct_gates,
)


def _decision_record(
    table_id: str = "t1",
    recorded_at_ms: int = 1_000,
    street: str = "preflop",
    hero_stack: int = 1_600,
    effective_stack: int = 1_600,
    contribution: int = 0,
    pot_chips: int = 10,
    call_chips: int = 0,
    equity: float | None = 0.5,
    action: str = "fold",
    amount_to: int | None = None,
    proposed_branch: str | None = "fatal",
    proposed_family: str | None = "fold",
    policy_version: str = "candidate-v9-0003b",
    hole_cards: list[str] | None = None,
    board_cards: list[str] | None = None,
    rule_verdicts: list | None = None,
    submitted_risk_fraction: float | None = 0.0,
    **extra: object,
) -> dict:
    record = {
        "telemetry_schema_version": 3,
        "event": "decision",
        "table_id": table_id,
        "policy_version": policy_version,
        "recorded_at_ms": recorded_at_ms,
        "street": street,
        "big_blind_chips": 2,
        "equity": equity,
        "state": {
            "hero_stack_chips": hero_stack,
            "effective_stack_chips": effective_stack,
            "hero_contribution_chips": contribution,
            "pot_chips": pot_chips,
            "hole_cards": hole_cards if hole_cards is not None else ["As", "Kd"],
            "board_cards": board_cards if board_cards is not None else [],
        },
        "legal": {
            "call_chips": call_chips,
            "actions": ["fold", "call", "bet", "raise"],
        },
        "proposed_family": proposed_family,
        "proposed_branch": proposed_branch,
        "action": action,
        "amount_to": amount_to,
        "proposed_risk_fraction": None,
        "submitted_risk_fraction": submitted_risk_fraction,
        "rule_verdicts": rule_verdicts,
        "fallback_reason": None,
        "action_status": 200,
        "accepted": True,
        "hyper_aggression": False,
        "belief_degraded": False,
        "belief_degrade_reason": None,
        "bluff_kind": None,
        "identity_verified": True,
        "response_error": None,
        "training_eligible": True,
    }
    record.update(extra)
    return record


#: A minimal but real Arena table snapshot, enough for
#: ``training_telemetry.make_decision_record`` to build a record. Used
#: only to pin OVERRIDE_FIELDS against the keys the engine really writes.
_TABLE_FIXTURE = {
    "tableId": "t1",
    "street": "preflop",
    "bigBlindChips": 2,
    "potChips": 12,
    "currentBet": 3,
    "boardCards": [],
    "selfSeatNumber": 1,
    "seats": [
        {
            "seatNumber": 1,
            "status": "active",
            "stackChips": 1_634,
            "currentBetChips": 0,
            "totalCommittedChips": 0,
            "holeCards": ["As", "4h"],
        },
        {
            "seatNumber": 2,
            "status": "active",
            "stackChips": 1_500,
            "currentBetChips": 3,
            "totalCommittedChips": 3,
        },
    ],
    "allowedActions": {
        "availableActions": ["all-in", "call", "fold", "raise"],
        "callChips": 3,
        "amountSemantics": "toAmount",
    },
    "recentEvents": [],
}


def _hand_result_record(
    table_id: str, delta: int, settled_at_ms: int = 2_000, reward_bb: float = 0.0
) -> dict:
    return {
        "telemetry_schema_version": 3,
        "event": "hand_result",
        "table_id": table_id,
        "chip_delta_chips": delta,
        "settled_at_ms": settled_at_ms,
        "reward_bb": reward_bb,
    }


class LoadJournalTests(unittest.TestCase):
    def test_filters_by_policy_and_counts_everything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(_decision_record(table_id="a")) + "\n")
                handle.write(
                    json.dumps(_decision_record(table_id="b", policy_version="v7"))
                    + "\n"
                )
                handle.write(json.dumps(_hand_result_record("a", -10)) + "\n")
                handle.write(json.dumps(_hand_result_record("b", 5)) + "\n")
                handle.write(json.dumps({"event": "unknown_thing"}) + "\n")
                handle.write("\n")

            decisions, hand_results, counts = load_journal(path, "candidate-v9-0003b")
            self.assertEqual([d.table_id for d in decisions], ["a"])
            self.assertEqual(counts["decision:other_policy"], 1)
            self.assertEqual(counts["event:hand_result"], 2)
            self.assertEqual(counts["event:unknown_thing"], 1)
            self.assertEqual(counts["blank"], 1)
            self.assertEqual(counts["event:other"], 1)
            self.assertEqual(len(hand_results), 2)

    def test_unparsable_lines_skipped_with_a_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write("{not json\n")
                handle.write(json.dumps(_decision_record()) + "\n")
                handle.write("also not json\n")

            decisions, _, counts = load_journal(path, "candidate-v9-0003b")
            self.assertEqual(len(decisions), 1)
            self.assertEqual(counts["unparsable"], 2)
            self.assertEqual(counts["lines"], 3)

    def test_duplicate_hand_results_counted_and_last_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(_decision_record(table_id="t")) + "\n")
                handle.write(json.dumps(_hand_result_record("t", -10)) + "\n")
                handle.write(json.dumps(_hand_result_record("t", -20)) + "\n")

            _, hand_results, counts = load_journal(path, "candidate-v9-0003b")
            self.assertEqual(hand_results["t"]["chip_delta_chips"], -20)
            self.assertEqual(counts["hand_result:duplicate"], 1)

    def test_decisions_missing_stacks_skipped_with_a_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                bad = _decision_record()
                bad["state"] = {}
                handle.write(json.dumps(bad) + "\n")
                handle.write(json.dumps(_decision_record()) + "\n")

            decisions, _, counts = load_journal(path, "candidate-v9-0003b")
            self.assertEqual(len(decisions), 1)
            self.assertEqual(counts["decision:missing_stacks"], 1)


class LedgerTests(unittest.TestCase):
    def test_ledger_joins_settlements_and_sorts_decisions(self) -> None:
        decisions = [
            Decision(
                table_id="t", recorded_at_ms=20, street="river", big_blind=2,
                hero_stack=100, effective_stack=100, contribution=0,
                pot_chips=50, to_call=10, equity=0.5, action="call",
                amount_to=None, proposed_branch="active",
                proposed_family="check_call", submitted_risk_fraction=0.1,
                proposed_risk_fraction=None, hole_cards=("As", "Kd"),
                board_cards=("2c", "3d", "4h"), legal_actions=("call", "fold"),
                rule_verdicts_present=False, fallback_reason=None,
                action_status=200, accepted=True, hyper_aggression=False,
                belief_degraded=False, training_eligible=True,
            ),
            Decision(
                table_id="t", recorded_at_ms=10, street="preflop", big_blind=2,
                hero_stack=120, effective_stack=120, contribution=0,
                pot_chips=10, to_call=0, equity=0.6, action="raise",
                amount_to=10, proposed_branch="aggressive",
                proposed_family="aggress", submitted_risk_fraction=0.08,
                proposed_risk_fraction=None, hole_cards=("As", "Kd"),
                board_cards=(), legal_actions=("raise", "fold"),
                rule_verdicts_present=False, fallback_reason=None,
                action_status=200, accepted=True, hyper_aggression=False,
                belief_degraded=False, training_eligible=True,
            ),
        ]
        ledger = build_ledger(decisions, {"t": {"chip_delta_chips": -30, "settled_at_ms": 30, "reward_bb": -15.0}})
        self.assertEqual(len(ledger), 1)
        row = ledger[0]
        self.assertEqual(row["hand_delta_chips"], -30)
        self.assertEqual(row["n_decisions"], 2)
        self.assertEqual(row["decisions"][0]["street"], "preflop")
        self.assertEqual(row["decisions"][1]["street"], "river")

    def test_ledger_marks_tables_without_settlement(self) -> None:
        decisions = [
            Decision(
                table_id="t", recorded_at_ms=10, street="preflop", big_blind=2,
                hero_stack=100, effective_stack=100, contribution=0,
                pot_chips=10, to_call=0, equity=None, action="fold",
                amount_to=None, proposed_branch="fatal",
                proposed_family="fold", submitted_risk_fraction=0.0,
                proposed_risk_fraction=None, hole_cards=("As", "Kd"),
                board_cards=(), legal_actions=("fold",),
                rule_verdicts_present=False, fallback_reason=None,
                action_status=200, accepted=True, hyper_aggression=False,
                belief_degraded=False, training_eligible=True,
            )
        ]
        ledger = build_ledger(decisions, {})
        self.assertIsNone(ledger[0]["hand_delta_chips"])
        self.assertIsNone(ledger[0]["settled_at_ms"])


class WindowTests(unittest.TestCase):
    def test_windows_sum_known_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(_decision_record(table_id="a", recorded_at_ms=1000)) + "\n")
                handle.write(json.dumps(_decision_record(table_id="b", recorded_at_ms=2000)) + "\n")
                handle.write(json.dumps(_hand_result_record("a", 977)) + "\n")
                handle.write(json.dumps(_hand_result_record("b", -977)) + "\n")
            decisions, hand_results, _ = load_journal(path, "candidate-v9-0003b")
            ledger = build_ledger(decisions, hand_results)
            windows = [
                ("first", "1970-01-01T00:00:00Z", "1970-01-01T00:00:01.5Z"),
                ("second", "1970-01-01T00:00:01.5Z", "1970-01-01T00:00:03Z"),
            ]
            rows = assign_windows(ledger, windows)
            self.assertEqual(rows[0]["n_decisions"], 1)
            self.assertEqual(rows[0]["delta_sum_chips"], 977)
            self.assertEqual(rows[1]["n_decisions"], 1)
            self.assertEqual(rows[1]["delta_sum_chips"], -977)


class GateReconstructionTests(unittest.TestCase):
    def test_risk_cap_ceiling_and_bind_flags(self) -> None:
        gates = SafetyGates()
        decisions = [
            Decision(
                table_id="t", recorded_at_ms=10, street="turn", big_blind=2,
                hero_stack=131, effective_stack=131, contribution=0,
                pot_chips=60, to_call=0, equity=0.348, action="bet",
                amount_to=60, proposed_branch="active",
                proposed_family="aggress", submitted_risk_fraction=0.45,
                proposed_risk_fraction=None, hole_cards=("As", "Kd"),
                board_cards=("2c", "3d", "4h", "5s"),
                legal_actions=("bet", "check"), rule_verdicts_present=False,
                fallback_reason=None, action_status=200, accepted=True,
                hyper_aggression=False, belief_degraded=False,
                training_eligible=True,
            )
        ]
        analysis = reconstruct_gates(decisions, gates)
        self.assertEqual(len(analysis["risk_caps"]), 1)
        row = analysis["risk_caps"][0]
        expected_ceiling = 0 + max(2, round(gates.risk_cap_stack_fraction * 131))
        self.assertEqual(row["cap_ceiling_chips"], expected_ceiling)
        self.assertTrue(row["at_ceiling"])
        self.assertFalse(row["above_ceiling"])
        self.assertEqual(row["margin_chips"], 0)

    def test_call_price_and_stack_gate_margins(self) -> None:
        gates = SafetyGates()
        decisions = [
            Decision(
                table_id="t", recorded_at_ms=10, street="river", big_blind=2,
                hero_stack=500, effective_stack=500, contribution=0,
                pot_chips=100, to_call=250, equity=0.60, action="call",
                amount_to=None, proposed_branch="passive",
                proposed_family="check_call", submitted_risk_fraction=0.5,
                proposed_risk_fraction=None, hole_cards=("As", "Kd"),
                board_cards=("2c", "3d", "4h", "5s", "9c"),
                legal_actions=("call", "fold"), rule_verdicts_present=False,
                fallback_reason=None, action_status=200, accepted=True,
                hyper_aggression=False, belief_degraded=False,
                training_eligible=True,
            )
        ]
        analysis = reconstruct_gates(decisions, gates)
        call = analysis["calls"][0]
        # to_call 250 >= 0.455 * 500 -> second gate triggers; first does not.
        self.assertFalse(call["stack_gates"][0]["triggered"])
        self.assertTrue(call["stack_gates"][1]["triggered"])
        expected_pot_odds = 250 / 350
        self.assertAlmostEqual(call["pot_odds"], expected_pot_odds)
        # river street margin 0.08, unpaired board -> fresh -> board margin 0.
        self.assertAlmostEqual(call["price"], expected_pot_odds + 0.08)
        required = gates.call_stack_gates[1][1] + gates.reveal_expense_equity_slope * 0.5 * (0 / 3.0)
        self.assertAlmostEqual(call["stack_gates"][1]["margin_vs_required"], 0.60 - required)

    def test_shove_near_nut_margin_and_collapse_detection(self) -> None:
        gates = SafetyGates()
        decisions = [
            Decision(
                table_id="t", recorded_at_ms=10, street="river", big_blind=2,
                hero_stack=1243, effective_stack=1, contribution=0,
                pot_chips=2429, to_call=1243, equity=0.7765, action="all-in",
                amount_to=1243, proposed_branch="aggressive",
                proposed_family="aggress", submitted_risk_fraction=1.0,
                proposed_risk_fraction=None, hole_cards=("As", "4h"),
                board_cards=("2h", "3c", "5c", "9s", "Qc"),
                legal_actions=("all-in", "call", "fold"),
                rule_verdicts_present=False, fallback_reason=None,
                action_status=200, accepted=True, hyper_aggression=False,
                belief_degraded=False, training_eligible=True,
            )
        ]
        analysis = reconstruct_gates(decisions, gates)
        shove = analysis["all_ins"][0]
        self.assertAlmostEqual(shove["margin_vs_near_nut"], 0.7765 - gates.near_nut_floor)
        self.assertEqual(len(analysis["denominator_collapses"]), 1)
        collapse = analysis["denominator_collapses"][0]
        self.assertEqual(collapse["hero_stack_chips"], 1243)
        self.assertEqual(collapse["effective_stack_chips"], 1)

    def test_no_collapse_when_hero_really_short(self) -> None:
        gates = SafetyGates()
        decisions = [
            Decision(
                table_id="t", recorded_at_ms=10, street="preflop", big_blind=2,
                hero_stack=1, effective_stack=1, contribution=0,
                pot_chips=10, to_call=1, equity=0.5, action="call",
                amount_to=None, proposed_branch="active",
                proposed_family="check_call", submitted_risk_fraction=1.0,
                proposed_risk_fraction=None, hole_cards=("As", "Kd"),
                board_cards=(), legal_actions=("call", "fold"),
                rule_verdicts_present=False, fallback_reason=None,
                action_status=200, accepted=True, hyper_aggression=False,
                belief_degraded=False, training_eligible=True,
            )
        ]
        analysis = reconstruct_gates(decisions, gates)
        self.assertEqual(analysis["denominator_collapses"], [])


def _override_decision(**changes: object) -> Decision:
    """The S17 bust hand's preflop row: active branch, raise, bluff steal."""

    fields: dict[str, object] = dict(
        table_id="cmtk4s7kymtizot1wknrcpjws",
        recorded_at_ms=1_788_355_690_596,
        street="preflop",
        big_blind=2,
        hero_stack=1_634,
        effective_stack=1_634,
        contribution=0,
        pot_chips=12,
        to_call=3,
        equity=0.3567,
        action="raise",
        amount_to=20,
        proposed_branch="active",
        # The engine relabels the family to aggress on the bluff path
        # BEFORE the record is written, so the record's own
        # "proposed_family" already agrees with the executed action.
        proposed_family="aggress",
        submitted_risk_fraction=0.012,
        proposed_risk_fraction=None,
        hole_cards=("As", "4h"),
        board_cards=(),
        legal_actions=("all-in", "call", "fold", "raise"),
        rule_verdicts_present=False,
        fallback_reason=None,
        action_status=200,
        accepted=True,
        hyper_aggression=False,
        belief_degraded=False,
        training_eligible=True,
        bluff_kind="steal",
    )
    fields.update(changes)
    return Decision(**fields)  # type: ignore[arg-type]


class OverrideDetectionTests(unittest.TestCase):
    """A bluff override must be reported as one, never as a literal.

    Fails on the pre-2026-09-03 reader, which never read ``bluff_kind``
    and compared ``proposed_branch`` to ``action`` only as a frequency
    table.
    """

    def test_bluff_steal_over_an_active_call_is_an_override(self) -> None:
        row = classify_execution(_override_decision())
        self.assertEqual(row["verdict"], "override")
        self.assertEqual(row["canonical_action"], "call")
        self.assertEqual(row["canonical_family"], "check_call")
        self.assertEqual(row["executed_action"], "raise")
        self.assertEqual(row["executed_family"], "aggress")
        self.assertEqual(row["direction"], "promotion")
        self.assertEqual(row["bluff_kind"], "steal")
        self.assertEqual(row["witnesses"], ["bluff_kind=steal"])

    def test_the_override_is_not_hidden_by_proposed_family(self) -> None:
        # proposed_family is post-override: on this row it equals the
        # executed family, so a family-vs-action comparison sees nothing.
        decision = _override_decision()
        self.assertEqual(decision.proposed_family, "aggress")
        row = classify_execution(decision)
        self.assertNotEqual(row["proposed_family"], row["canonical_family"])
        self.assertEqual(row["verdict"], "override")

    def test_bluff_row_survives_the_journal_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            path.write_text(
                json.dumps(
                    _decision_record(
                        table_id="bust",
                        street="preflop",
                        action="raise",
                        amount_to=20,
                        proposed_branch="active",
                        proposed_family="aggress",
                        call_chips=3,
                        bluff_kind="steal",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            decisions, _, _ = load_journal(path, "candidate-v9-0003b")
            self.assertEqual(decisions[0].bluff_kind, "steal")
            analysis = override_analysis(decisions)
            self.assertEqual(analysis["overrides"], 1)
            self.assertEqual(analysis["overrides_unexplained"], 0)
            self.assertEqual(analysis["override_by_witness"], {"bluff_kind=steal": 1})
            self.assertEqual(analysis["verdicts"], {"override": 1})
            self.assertEqual(
                analysis["proposed_family_disagrees_with_branch"]["count"], 1
            )

    def test_report_and_ledger_carry_the_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            path.write_text(
                json.dumps(
                    _decision_record(
                        table_id="bust",
                        action="raise",
                        amount_to=20,
                        proposed_branch="active",
                        proposed_family="aggress",
                        call_chips=3,
                        bluff_kind="steal",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            report = build_report(path, "candidate-v9-0003b")
            self.assertEqual(report["counters"]["execution_verdict"], {"override": 1})
            self.assertEqual(report["counters"]["bluff_kind"], {"steal": 1})
            ledger_row = report["ledger"][0]["decisions"][0]
            self.assertEqual(ledger_row["execution_verdict"], "override")
            self.assertEqual(ledger_row["canonical_action"], "call")
            self.assertEqual(ledger_row["override_witnesses"], ["bluff_kind=steal"])
            self.assertEqual(ledger_row["bluff_kind"], "steal")

    def test_literal_execution_is_still_literal(self) -> None:
        row = classify_execution(
            _override_decision(action="call", amount_to=None, bluff_kind=None,
                               proposed_family="check_call")
        )
        self.assertEqual(row["verdict"], "literal")
        self.assertEqual(row["witnesses"], [])
        self.assertIsNone(row["direction"])

    def test_same_family_rendering_is_not_an_override(self) -> None:
        # Escalation at stack-reaching size renders as all-in: the branch
        # still executed, so this is a rendering, not an override.
        row = classify_execution(
            _override_decision(
                street="river", proposed_branch="aggressive", action="all-in",
                amount_to=1_243, to_call=1_243, bluff_kind=None,
                proposed_family="aggress", legal_actions=("all-in", "call", "fold"),
            )
        )
        self.assertEqual(row["verdict"], "rendering")
        self.assertEqual(row["canonical_action"], "raise")
        self.assertIsNone(row["direction"])

    def test_safe_direction_demotion_is_an_unexplained_override(self) -> None:
        # active facing a price executes a call; a fold instead is a
        # demotion the record carries no field for.
        decision = _override_decision(
            action="fold", amount_to=None, bluff_kind=None, proposed_family="fold"
        )
        row = classify_execution(decision)
        self.assertEqual(row["verdict"], "override")
        self.assertEqual(row["direction"], "demotion")
        self.assertEqual(row["witnesses"], [])
        analysis = override_analysis([decision])
        self.assertEqual(analysis["overrides_unexplained"], 1)
        self.assertEqual(analysis["override_by_witness"], {"unexplained": 1})

    def test_null_branch_is_unclassifiable_not_literal(self) -> None:
        row = classify_execution(
            _override_decision(proposed_branch=None, action="fold",
                               amount_to=None, bluff_kind=None,
                               fallback_reason="deadline", proposed_family=None)
        )
        self.assertEqual(row["verdict"], "unclassifiable")
        self.assertIn("fallback_reason=deadline", row["witnesses"])

    def test_masked_branch_is_unclassifiable(self) -> None:
        # aggressive is escalation-only; at to_call 0 the contract
        # refuses to project it rather than guessing an action.
        row = classify_execution(
            _override_decision(proposed_branch="aggressive", to_call=0,
                               action="bet", amount_to=10, bluff_kind=None,
                               proposed_family="aggress")
        )
        self.assertEqual(row["verdict"], "unclassifiable")
        self.assertIn("masked", str(row["reason"]))

    def test_every_action_changing_witness_is_reported(self) -> None:
        witnesses = override_witnesses(
            _override_decision(
                bluff_kind="steal", fallback_reason="deadline",
                rule_verdict_count=2, hyper_aggression=True,
            )
        )
        self.assertEqual(
            witnesses,
            [
                "bluff_kind=steal",
                "fallback_reason=deadline",
                "rule_verdicts=2",
                "hyper_aggression=true",
            ],
        )


class PinnedToTheEngineTests(unittest.TestCase):
    """Anything the reader restates must equal what it restated."""

    def test_street_call_margins_match_engine(self) -> None:
        from engine import decision_engine

        self.assertEqual(STREET_CALL_MARGINS, decision_engine._CALL_MARGINS)

    def test_reveals_remaining_matches_game_state(self) -> None:
        self.assertEqual(REVEALS_REMAINING, game_state._REVEALS_REMAINING)

    def test_action_family_matches_the_telemetry_module(self) -> None:
        self.assertEqual(
            ACTION_FAMILY,
            {
                action: training_telemetry.action_family(action)
                for action in ACTION_FAMILY
            },
        )
        # And nothing the engine can submit is missing from the copy.
        for action in ("fold", "check", "call", "bet", "raise", "all-in"):
            self.assertIn(action, ACTION_FAMILY)

    def test_override_fields_are_keys_of_a_real_decision_record(self) -> None:
        """Every enumerated name is a key the engine actually writes.

        Built by calling ``make_decision_record`` itself, so a rename or
        a removal in the telemetry schema fails here instead of leaving
        the reader silently blind to that override channel again.
        """

        record = training_telemetry.make_decision_record(
            competition_id="c1",
            policy_version="candidate-v9-0003b",
            table=_TABLE_FIXTURE,
            payload={"action": "fold"},
            decision=None,
            deadline_budget_s=10.0,
            fallback_reason=None,
            action_status=200,
            identity_verified=True,
            recorded_at_ms=1_000,
        )
        result_fields = {f.name for f in dataclasses.fields(DecisionResult)}
        for field in OVERRIDE_FIELDS:
            with self.subTest(field=field.name):
                self.assertIn(field.name, record)
                self.assertTrue(field.engine_site.startswith("engine/"))
                self.assertTrue(field.effect)
        # And the action-changing ones all originate on DecisionResult or
        # on the record builder's own signature.
        record_parameters = set(
            inspect.signature(training_telemetry.make_decision_record).parameters
        )
        for field in OVERRIDE_FIELDS:
            if not field.changes_action:
                continue
            with self.subTest(field=field.name, origin=True):
                self.assertTrue(
                    field.name in result_fields or field.name in record_parameters,
                    f"{field.name} changes the action but is neither a "
                    "DecisionResult field nor a make_decision_record parameter",
                )

    def test_bluff_kind_is_enumerated_and_action_changing(self) -> None:
        by_name = {field.name: field for field in OVERRIDE_FIELDS}
        self.assertIn("bluff_kind", by_name)
        self.assertTrue(by_name["bluff_kind"].changes_action)
        # DecisionResult must still carry it: if the engine drops the
        # field, this reader's central witness is gone and the test says
        # so instead of the report silently going blind again.
        self.assertIn(
            "bluff_kind", {f.name for f in dataclasses.fields(DecisionResult)}
        )


class ReportAndCliTests(unittest.TestCase):
    def test_build_report_accounts_for_stray_settlements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(_decision_record(table_id="a")) + "\n")
                handle.write(json.dumps(_hand_result_record("a", -10)) + "\n")
                handle.write(json.dumps(_hand_result_record("untracked", 3)) + "\n")
            report = build_report(path, "candidate-v9-0003b")
            stray = report["hand_result_rows_without_decision_table"]
            self.assertEqual(stray["count"], 1)
            self.assertEqual(stray["delta_sum_chips"], 3)
            self.assertEqual(report["counters"]["decisions"], 1)
            self.assertEqual(report["parse_accounting"].get("unparsable", 0), 0)

    def test_main_writes_reloadable_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal.jsonl"
            journal.write_text(
                json.dumps(_decision_record(table_id="a")) + "\n"
                + json.dumps(_hand_result_record("a", -10)) + "\n",
                encoding="utf-8",
            )
            out = Path(tmp) / "report.json"
            code = main(
                [
                    "--journal", str(journal),
                    "--policy-version", "candidate-v9-0003b",
                    "--windows", "s1:1970-01-01T00:00:00Z:1970-01-01T00:00:02Z",
                    "--output", str(out),
                ]
            )
            self.assertEqual(code, 0)
            loaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(loaded["policy_version"], "candidate-v9-0003b")
            self.assertEqual(loaded["windows"][0]["delta_sum_chips"], -10)
            self.assertEqual(loaded["ledger"][0]["hand_delta_chips"], -10)


if __name__ == "__main__":
    unittest.main()
