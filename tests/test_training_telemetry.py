"""Checks for append-only decisions and settled-hand rewards."""

from __future__ import annotations

import json
import tempfile
import unittest
from array import array
from dataclasses import replace
from pathlib import Path

from risk_temperature import measure_risk_temperature

from engine.decision_engine import ArenaAction, DecisionResult
from engine.learning_contract import LEARNING_INPUT_SIZE
from engine.training_telemetry import (
    action_response_matches,
    load_training_corpus,
    load_training_examples,
    make_decision_record,
    parse_replay_receipts,
    ReplayReceipt,
    save_training_corpus,
    TelemetryError,
    TelemetryLog,
    TrainingExample,
)


def _table() -> dict:
    return {
        "id": "table-1",
        "tableId": "table-1",
        "competitionId": "comp-1",
        "street": "flop",
        "potChips": 200,
        "currentBet": 0,
        "boardCards": ["2c", "7d", "9h"],
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": 1,
        "seats": [
            {
                "seatNumber": 1,
                "status": "Active",
                "stackChips": 1_000,
                "currentBetChips": 0,
                "totalCommittedChips": 0,
                "holeCards": ["Ah", "Ad"],
            },
            {
                "seatNumber": 2,
                "status": "Active",
                "stackChips": 1_000,
                "currentBetChips": 0,
                "holeCards": None,
            },
        ],
        "allowedActions": {
            "canFold": True,
            "canCheck": True,
            "canCall": False,
            "canBet": True,
            "canRaise": False,
            "canAllIn": False,
            "callChips": 0,
            "minBet": 100,
            "minRaiseTo": None,
            "betRange": {"min": 100, "max": 1_000},
            "raiseRange": None,
            "availableActions": ["fold", "check", "bet"],
            "reasoningRequired": False,
        },
        "recentEvents": [],
    }


def _decision() -> DecisionResult:
    return DecisionResult(
        action=ArenaAction("bet", 300, "measured pressure"),
        family="aggress",
        equity=0.8,
        situation_temperature=measure_risk_temperature(
            hand_strength=80,
            purse=1_000,
            bet=0,
            street="flop",
            players=2,
        ),
        learning_features=(0.0,) * LEARNING_INPUT_SIZE,
        behavior_probabilities=(0.0, 0.0, 1.0),
        opponent_evidence_confidence=0.75,
    )


def _example(**overrides) -> TrainingExample:
    values = {
        "table_id": "sim-7-3",
        "policy_version": "heuristic-aggressive-v5",
        "features": (0.1, -2.5, 1 / 3, 7.0),
        "action_family_index": 2,
        "behavior_probabilities": (0.2, 0.3, 0.5),
        "submitted_risk_fraction": 0.25,
        "purse_bb": 60.0,
        "reward_bb": -1.5,
        "counterfactual": True,
        "opponent_confidence": 0.75,
        "decision_id": "sim-7-3:hero:4",
        "harvest_leg": "heads-up vs shover",
        "inclusion_count": 3,
    }
    values.update(overrides)
    return TrainingExample(**values)


class TrainingCorpusTests(unittest.TestCase):
    def test_round_trip_preserves_every_field_at_float32_features(self) -> None:
        examples = (
            _example(),
            _example(
                table_id="sim-7-4",
                features=(0.0, 1e-8, -7.25, 3.14159),
                counterfactual=False,
                decision_id=None,
                harvest_leg=None,
                inclusion_count=None,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.gz"
            save_training_corpus(path, examples)
            self.assertTrue(Path(str(path) + ".meta.json").exists())
            loaded = load_training_corpus(path)
        expected = tuple(
            replace(example, features=tuple(array("f", example.features)))
            for example in examples
        )
        self.assertEqual(loaded, expected)

    def test_branch_absorption_survives_the_corpus_round_trip(self) -> None:
        """A persisted Option A corpus must not read back as a legacy one.

        `branch_absorption` was added to `TrainingExample` but not to the
        sidecar, so 117,447 of 117,447 rows in the candidate-v7-0003 corpus
        lost it on the way to disk. Consumers treat an absent map as
        pre-Option-A and fall back to `min(returns)`, which under-scores the
        `always_aggress_*` trivial baselines — the BLOCKING promotion gate.
        The round-trip test above compares whole dataclasses and still missed
        it, because its fixture never populated the field.
        """

        absorption = (
            ("aggress_half_pot", "aggress_half_pot"),
            ("aggress_pot", "aggress_half_pot"),
            ("check_call", "check_call"),
            ("fold", "check_call"),
        )
        examples = (
            _example(action_branch="aggress_half_pot", branch_absorption=absorption),
            _example(action_branch="check_call", branch_absorption=absorption),
            # A legacy row carries no map and must stay None, not become ().
            _example(action_branch=None, branch_absorption=None, counterfactual=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.gz"
            save_training_corpus(path, examples)
            loaded = load_training_corpus(path)
        self.assertEqual(loaded[0].branch_absorption, absorption)
        self.assertEqual(loaded[1].branch_absorption, absorption)
        self.assertIsNone(loaded[2].branch_absorption)
        # An Option A corpus must be distinguishable from a legacy one.
        self.assertTrue(any(row.branch_absorption for row in loaded))

    def test_sidecar_row_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.gz"
            save_training_corpus(path, (_example(),))
            sidecar = Path(str(path) + ".meta.json")
            document = json.loads(sidecar.read_text(encoding="utf-8"))
            document["rows"] = []
            sidecar.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(TelemetryError):
                load_training_corpus(path)

    def test_ragged_feature_rows_are_rejected_at_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.gz"
            with self.assertRaises(TelemetryError):
                save_training_corpus(path, (_example(), _example(features=(1.0, 2.0))))

    def test_training_example_constructs_without_harvest_fields(self) -> None:
        example = TrainingExample(
            table_id="table-1",
            policy_version="heuristic-test-v1",
            features=(0.0,),
            action_family_index=0,
            behavior_probabilities=(1.0, 0.0, 0.0),
            submitted_risk_fraction=0.0,
            purse_bb=10.0,
            reward_bb=0.0,
        )
        self.assertIsNone(example.harvest_leg)
        self.assertIsNone(example.inclusion_count)


class TrainingTelemetryTests(unittest.TestCase):
    def test_identity_verified_decision_joins_once_to_big_blind_reward(self) -> None:
        decision = _decision()
        record = make_decision_record(
            competition_id="comp-1",
            policy_version="heuristic-test-v1",
            table=_table(),
            payload=decision.to_payload(),
            decision=decision,
            deadline_budget_s=5.0,
            fallback_reason=None,
            action_status=200,
            identity_verified=True,
            recorded_at_ms=1,
        )
        self.assertTrue(record["training_eligible"])
        self.assertEqual(record["submitted_risk_fraction"], 0.3)
        self.assertEqual(record["behavior_probabilities"], [0.0, 0.0, 1.0])
        self.assertEqual(record["opponent_evidence_confidence"], 0.75)
        self.assertEqual(record["state"]["hole_cards"], ["Ah", "Ad"])
        self.assertEqual(record["state"]["hero_purse_chips"], 1_000)
        self.assertEqual(record["legal"]["bet_range"], {"min": 100, "max": 1_000})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            telemetry = TelemetryLog(path)
            telemetry.append(record)
            receipt = ReplayReceipt("hand-1", "table-1", 2, -50)
            self.assertEqual(telemetry.append_settlements("comp-1", (receipt,)), 1)
            self.assertEqual(telemetry.append_settlements("comp-1", (receipt,)), 0)
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            examples = load_training_examples(path)

        self.assertEqual(lines[1]["event"], "hand_result")
        self.assertEqual(lines[1]["reward_bb"], -0.5)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].action_family_index, 2)
        self.assertEqual(examples[0].submitted_risk_fraction, 0.3)
        self.assertEqual(examples[0].purse_bb, 10.0)
        self.assertEqual(examples[0].reward_bb, -0.5)
        self.assertFalse(examples[0].counterfactual)
        self.assertEqual(examples[0].opponent_confidence, 0.75)

    def test_schema_two_records_carry_the_betting_that_led_here(self) -> None:
        """Schema 2 persists the snapshot's recentEvents into the record.

        Without them no live hand can be attributed: "ran into the top of
        their range" and "was outplayed" read identically, which is where
        the 2026-08-15 drawdown forensics had to stop.
        """

        table = _table()
        table["recentEvents"] = [
            {
                "street": "Preflop",
                "summary": {"seatNumber": 2, "action": "raise", "amount": 300},
            },
            {"street": "Preflop", "summary": {"seatNumber": 1, "action": "call"}},
            {
                "street": "flop",
                "summary": {"seatNumber": 2, "action": "check", "amount": None},
            },
            "garbage",  # malformed entries are skipped, never fatal
            {"street": "flop", "summary": None},
            {
                "street": "flop",
                "summary": {"seatNumber": None, "action": "bet", "amount": 150},
            },
        ]
        decision = _decision()
        record = make_decision_record(
            competition_id="comp-1",
            policy_version="heuristic-test-v1",
            table=table,
            payload=decision.to_payload(),
            decision=decision,
            deadline_budget_s=5.0,
            fallback_reason=None,
            action_status=200,
            identity_verified=True,
            recorded_at_ms=1,
        )
        self.assertEqual(record["telemetry_schema_version"], 3)
        self.assertEqual(
            record["state"]["recent_actions"],
            [
                {
                    "street": "preflop",
                    "seat_number": 2,
                    "action": "raise",
                    "amount": 300,
                },
                {"street": "preflop", "seat_number": 1, "action": "call"},
                {"street": "flop", "seat_number": 2, "action": "check"},
                {"street": "flop", "seat_number": None, "action": "bet", "amount": 150},
            ],
        )

    def test_additive_v9_diagnostics_are_null_for_legacy_decisions(self) -> None:
        """proposed_branch and the belief-degrade pair are additive: a v7/v8
        decision carries null/false, never a fabricated value."""

        record = make_decision_record(
            competition_id="comp-1",
            policy_version="heuristic-test-v1",
            table=_table(),
            payload=_decision().to_payload(),
            decision=_decision(),
            deadline_budget_s=5.0,
            fallback_reason=None,
            action_status=200,
            identity_verified=True,
        )
        self.assertIsNone(record["proposed_branch"])
        self.assertFalse(record["belief_degraded"])
        self.assertIsNone(record["belief_degrade_reason"])

    def test_proposed_branch_and_belief_degrade_are_recorded(self) -> None:
        decision = replace(
            _decision(),
            proposed_branch="aggressive",
            belief_degraded=True,
            belief_degrade_reason="ValueError: corrupt event",
        )
        record = make_decision_record(
            competition_id="comp-1",
            policy_version="candidate-v9-test",
            table=_table(),
            payload=decision.to_payload(),
            decision=decision,
            deadline_budget_s=5.0,
            fallback_reason=None,
            action_status=200,
            identity_verified=True,
        )
        self.assertEqual(record["proposed_branch"], "aggressive")
        self.assertTrue(record["belief_degraded"])
        self.assertEqual(record["belief_degrade_reason"], "ValueError: corrupt event")

    def test_schema_one_journals_still_load(self) -> None:
        """The 4.7 MB live journal predates schema 2 and must stay loadable.

        A reader that accepts only the writer's version turns a schema bump
        into silent loss of the only record of how the deployed policy has
        actually played.
        """

        decision = _decision()
        record = make_decision_record(
            competition_id="comp-1",
            policy_version="heuristic-test-v1",
            table=_table(),
            payload=decision.to_payload(),
            decision=decision,
            deadline_budget_s=5.0,
            fallback_reason=None,
            action_status=200,
            identity_verified=True,
            recorded_at_ms=1,
        )
        # Regress the record to schema 1: exactly what every existing line
        # of the live journal looks like.
        legacy = dict(record, telemetry_schema_version=1)
        legacy["state"] = {
            key: value
            for key, value in record["state"].items()
            if key != "recent_actions"
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(legacy) + "\n")
            telemetry = TelemetryLog(path)  # re-index must accept schema 1
            receipt = ReplayReceipt("hand-1", "table-1", 2, -50)
            self.assertEqual(telemetry.append_settlements("comp-1", (receipt,)), 1)
            examples = load_training_examples(path)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].submitted_risk_fraction, 0.3)

        # An actually unknown version must still be rejected loudly.
        # (3 became a readable version with the rule_verdicts field, so the
        # unknown-version case moves to 4.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(dict(legacy, telemetry_schema_version=4)) + "\n"
                )
            with self.assertRaises(TelemetryError):
                load_training_examples(path)

    def test_unverified_or_fallback_actions_are_not_training_samples(self) -> None:
        record = make_decision_record(
            competition_id="comp-1",
            policy_version="heuristic-test-v1",
            table=_table(),
            payload=_decision().to_payload(),
            decision=_decision(),
            deadline_budget_s=5.0,
            fallback_reason="exception",
            action_status=200,
            identity_verified=False,
        )
        self.assertFalse(record["training_eligible"])

    def test_replays_are_validated_sorted_and_deduplicated(self) -> None:
        receipts = parse_replay_receipts(
            [
                {
                    "handId": "hand-2",
                    "tableId": "table-2",
                    "settledAt": 20,
                    "chipDelta": 5,
                },
                {
                    "handId": "hand-1",
                    "tableId": "table-1",
                    "settledAt": 10,
                    "chipDelta": -5,
                },
            ]
        )
        self.assertEqual(
            [receipt.hand_id for receipt in receipts], ["hand-1", "hand-2"]
        )
        with self.assertRaises(TelemetryError):
            parse_replay_receipts(
                [
                    {
                        "handId": "hand-1",
                        "tableId": "table-1",
                        "settledAt": 10,
                        "chipDelta": 0,
                    },
                    {
                        "handId": "hand-1",
                        "tableId": "table-2",
                        "settledAt": 11,
                        "chipDelta": 0,
                    },
                ]
            )
        with self.assertRaises(TelemetryError):
            parse_replay_receipts(
                [
                    {
                        "handId": "hand-1",
                        "tableId": "table-1",
                        "settledAt": 10,
                        "chipDelta": 2**63,
                    }
                ]
            )

    def test_action_response_requires_all_three_identities(self) -> None:
        response = {
            "table": {"tableId": "table-1", "competitionId": "comp-1"},
            "participant": {"competitionId": "comp-1", "agentId": "agent-1"},
        }
        self.assertTrue(
            action_response_matches(
                response,
                table_id="table-1",
                competition_id="comp-1",
                agent_id="agent-1",
            )
        )
        response["participant"]["agentId"] = "other"
        self.assertFalse(
            action_response_matches(
                response,
                table_id="table-1",
                competition_id="comp-1",
                agent_id="agent-1",
            )
        )


class Schema4RecordTests(unittest.TestCase):
    """The additive schema-4 columns, and their contract.

    ``features`` stays the 142-input schema-2 vector every stored journal
    and the v7 offline trainer read. ``features_v9`` rides alongside so a
    v9 deployment's own hands are usable for v9 training and for
    ``head_degeneracy_audit``, which selects rows by feature width.
    """

    @staticmethod
    def _record(decision) -> dict:
        return make_decision_record(
            competition_id="comp-1",
            policy_version="candidate-v9-test",
            table=_table(),
            payload=decision.to_payload(),
            decision=decision,
            deadline_budget_s=5.0,
            fallback_reason=None,
            action_status=200,
            identity_verified=True,
            recorded_at_ms=1,
        )

    def test_a_v9_decision_records_the_schema_4_columns(self) -> None:
        from engine.learning_contract import FEATURE_SCHEMA_VERSION
        from engine.schema4 import INPUT_SIZE_V9, SCHEMA_VERSION_V9

        decision = replace(
            _decision(), learning_features_v9=(0.25,) * INPUT_SIZE_V9
        )
        record = self._record(decision)

        self.assertEqual(len(record["features_v9"]), INPUT_SIZE_V9)
        self.assertEqual(record["features_v9_schema_version"], SCHEMA_VERSION_V9)
        # The legacy columns are untouched.
        self.assertEqual(len(record["features"]), LEARNING_INPUT_SIZE)
        self.assertEqual(record["feature_schema_version"], FEATURE_SCHEMA_VERSION)

    def test_a_v7_decision_records_nulls_not_absences(self) -> None:
        record = self._record(_decision())
        self.assertIsNone(record["features_v9"])
        self.assertIsNone(record["features_v9_schema_version"])
        self.assertEqual(len(record["features"]), LEARNING_INPUT_SIZE)

    def test_a_wrong_width_v9_vector_is_refused(self) -> None:
        from engine.schema4 import INPUT_SIZE_V9

        for width in (INPUT_SIZE_V9 - 1, INPUT_SIZE_V9 + 1, LEARNING_INPUT_SIZE):
            with self.subTest(width=width):
                decision = replace(
                    _decision(), learning_features_v9=(0.0,) * width
                )
                with self.assertRaises(TelemetryError):
                    self._record(decision)

    def test_a_non_finite_v9_vector_is_refused(self) -> None:
        from engine.schema4 import INPUT_SIZE_V9

        vector = [0.0] * INPUT_SIZE_V9
        vector[7] = float("nan")
        decision = replace(_decision(), learning_features_v9=tuple(vector))
        with self.assertRaises(TelemetryError):
            self._record(decision)


if __name__ == "__main__":
    unittest.main()
