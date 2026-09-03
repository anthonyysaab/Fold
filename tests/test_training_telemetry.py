"""Checks for append-only decisions and settled-hand rewards."""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
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
    load_training_examples_with_summary,
    make_decision_record,
    parse_replay_receipts,
    ReplayReceipt,
    save_training_corpus,
    TELEMETRY_SCHEMA_VERSION,
    TelemetryError,
    TelemetryLog,
    TelemetryWarning,
    TrainingExample,
    TrainingLoadSummary,
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
        the 2026-08-15 drawdown forensics had to stop. Schema 4 additionally
        persists `to_amount` from the Arena's raise-to `toAmount` -- the
        price aggressive events actually carry (their `amount` is null),
        which was defect 23: every stored aggressive action was sizeless.
        """

        table = _table()
        table["recentEvents"] = [
            {
                "street": "Preflop",
                "summary": {
                    "seatNumber": 2,
                    "action": "raise",
                    "amount": None,
                    "toAmount": 300,
                },
            },
            {
                "street": "Preflop",
                "summary": {"seatNumber": 1, "action": "call", "amount": 100},
            },
            {
                "street": "flop",
                "summary": {"seatNumber": 2, "action": "check", "amount": None},
            },
            "garbage",  # malformed entries are skipped, never fatal
            {"street": "flop", "summary": None},
            {
                "street": "flop",
                "summary": {
                    "seatNumber": None,
                    "action": "bet",
                    "amount": 150,
                    "toAmount": 150,
                },
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
        self.assertEqual(record["telemetry_schema_version"], 4)
        self.assertEqual(
            record["state"]["recent_actions"],
            [
                {
                    "street": "preflop",
                    "seat_number": 2,
                    "action": "raise",
                    "to_amount": 300,
                },
                {"street": "preflop", "seat_number": 1, "action": "call", "amount": 100},
                {"street": "flop", "seat_number": 2, "action": "check"},
                {
                    "street": "flop",
                    "seat_number": None,
                    "action": "bet",
                    "amount": 150,
                    "to_amount": 150,
                },
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
        # (4 became a readable version with the to_amount / showdown fields,
        # so the unknown-version case moves to 5.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(dict(legacy, telemetry_schema_version=5)) + "\n"
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


class JournalReaderResilienceTests(unittest.TestCase):
    """Skip-and-count reading: defects J and 25.

    One malformed line must not block startup or loading (defect J), one
    duplicate settlement must not refuse the production journal (defect 25:
    104 duplicates in 2,458 rows). Skips are counted, warned and the
    journal is never rewritten.
    """

    @staticmethod
    def _decision_record() -> dict:
        decision = _decision()
        return make_decision_record(
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

    @staticmethod
    def _hand_result(table_id="table-1", reward=1.0, schema=None) -> dict:
        return {
            "telemetry_schema_version": (
                schema if schema is not None else TELEMETRY_SCHEMA_VERSION
            ),
            "event": "hand_result",
            "competition_id": "comp-1",
            "table_id": table_id,
            "reward_bb": reward,
        }

    @staticmethod
    def _warning_texts(caught) -> list[str]:
        return [
            str(item.message)
            for item in caught
            if issubclass(item.category, TelemetryWarning)
        ]

    def test_a_malformed_line_is_skipped_counted_and_warned(self) -> None:
        record = self._decision_record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.write("{this is not valid json\n")
                handle.write(json.dumps(self._hand_result()) + "\n")
            original = path.read_bytes()

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                telemetry = TelemetryLog(path)
            self.assertEqual(telemetry.malformed_lines, 1)
            self.assertTrue(any("line 2" in text for text in self._warning_texts(caught)))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                summary = load_training_examples_with_summary(path)
            self.assertIsInstance(summary, TrainingLoadSummary)
            self.assertEqual(summary.malformed_lines, 1)
            self.assertEqual(summary.duplicate_hand_results, 0)
            self.assertEqual(len(summary.examples), 1)
            self.assertTrue(any("line 2" in text for text in self._warning_texts(caught)))
            # The legacy wrapper still returns the bare tuple.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(len(load_training_examples(path)), 1)
            # The reader never rewrites the journal.
            self.assertEqual(path.read_bytes(), original)

    def test_a_duplicate_hand_result_keeps_the_first_and_counts(self) -> None:
        record = self._decision_record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.write(json.dumps(self._hand_result(reward=1.25)) + "\n")
                handle.write(json.dumps(self._hand_result(reward=-9.0)) + "\n")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                summary = load_training_examples_with_summary(path)
        self.assertEqual(summary.duplicate_hand_results, 1)
        self.assertEqual(summary.malformed_lines, 0)
        self.assertEqual(len(summary.examples), 1)
        self.assertEqual(summary.examples[0].reward_bb, 1.25)
        self.assertTrue(
            any(
                "duplicate hand result" in text
                for text in self._warning_texts(caught)
            )
        )

    def test_a_duplicate_hand_result_does_not_block_the_index(self) -> None:
        record = self._decision_record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.write(json.dumps(self._hand_result(reward=1.0)) + "\n")
                handle.write(json.dumps(self._hand_result(reward=2.0)) + "\n")
            telemetry = TelemetryLog(path)
        self.assertEqual(telemetry.malformed_lines, 0)

    def test_a_mixed_schema_three_and_four_journal_loads(self) -> None:
        """Schema-4 rows ride alongside schema-3 rows; the reader takes both."""

        legacy = dict(self._decision_record(), telemetry_schema_version=3)
        legacy["state"] = {
            key: value
            for key, value in legacy["state"].items()
            if key != "recent_actions"
        }
        legacy["table_id"] = "table-old"
        current = dict(self._decision_record())
        current["table_id"] = "table-new"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(legacy) + "\n")
                handle.write(
                    json.dumps(self._hand_result("table-old", 0.5, schema=3)) + "\n"
                )
                handle.write(json.dumps(current) + "\n")
                handle.write(json.dumps(self._hand_result("table-new", -0.25)) + "\n")
            summary = load_training_examples_with_summary(path)
        self.assertEqual(len(summary.examples), 2)
        self.assertEqual(summary.malformed_lines, 0)
        self.assertEqual(summary.duplicate_hand_results, 0)
        self.assertEqual(
            {example.table_id: example.reward_bb for example in summary.examples},
            {"table-old": 0.5, "table-new": -0.25},
        )

    def test_a_schema_one_row_without_the_purse_key_loads(self) -> None:
        """Schema-1 rows predate `hero_purse_chips`; the reader reconstructs
        the writer's own purse formula (stack + chips committed) instead of
        refusing them -- 11 such rows exist in the production journal."""

        legacy = dict(self._decision_record(), telemetry_schema_version=1)
        state = legacy["state"]
        del state["hero_purse_chips"]
        del state["recent_actions"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(legacy) + "\n")
                handle.write(json.dumps(self._hand_result(schema=1)) + "\n")
            summary = load_training_examples_with_summary(path)
        self.assertEqual(len(summary.examples), 1)
        expected_purse = state["hero_stack_chips"] + state["hero_contribution_chips"]
        self.assertEqual(summary.examples[0].purse_bb, expected_purse / 100.0)

    def test_a_row_without_any_purse_fields_is_still_rejected(self) -> None:
        """Reconstruction needs recorded chips; their absence is a contract
        violation and must stay loud instead of inventing a purse."""

        legacy = dict(self._decision_record(), telemetry_schema_version=1)
        state = legacy["state"]
        del state["hero_purse_chips"]
        del state["hero_stack_chips"]
        del state["hero_contribution_chips"]
        del state["recent_actions"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(legacy) + "\n")
                handle.write(json.dumps(self._hand_result(schema=1)) + "\n")
            with self.assertRaises(TelemetryError):
                load_training_examples_with_summary(path)


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


class Schema4WriterTests(unittest.TestCase):
    """The additive journal schema-4 writer fields (defects 23 and 20).

    Aggressive `recent_actions` entries carry the Arena's raise-to total in
    `to_amount` (their `amount` is null, so schema-3 rows are sizeless), and
    `hand_result` rows carry whatever showdown information the replay
    receipt carried, null when it carried none. Nothing on the decision
    path changes.
    """

    @staticmethod
    def _record(decision) -> dict:
        return make_decision_record(
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

    def test_aggressive_events_persist_the_raise_to_total(self) -> None:
        table = _table()
        table["recentEvents"] = [
            {
                "street": "Preflop",
                "summary": {
                    "seatNumber": 2,
                    "action": "raise",
                    "amount": None,
                    "toAmount": 300,
                },
            },
            {
                "street": "Preflop",
                "summary": {
                    "seatNumber": 2,
                    "action": "raise",
                    "amount": "junk",
                    "toAmount": "junk",
                },
            },
        ]
        record = make_decision_record(
            competition_id="comp-1",
            policy_version="heuristic-test-v1",
            table=table,
            payload=_decision().to_payload(),
            decision=_decision(),
            deadline_budget_s=5.0,
            fallback_reason=None,
            action_status=200,
            identity_verified=True,
            recorded_at_ms=1,
        )
        self.assertEqual(record["telemetry_schema_version"], 4)
        first, second = record["state"]["recent_actions"]
        self.assertNotIn("amount", first)
        self.assertEqual(first["to_amount"], 300)
        self.assertNotIn("amount", second)
        self.assertNotIn("to_amount", second)

    def test_hand_result_rows_carry_the_receipts_showdown_data(self) -> None:
        record = self._record(_decision())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            telemetry = TelemetryLog(path)
            telemetry.append(record)
            receipt = ReplayReceipt(
                "hand-1",
                "table-1",
                2,
                -50,
                board_cards=("2c", "7d", "9h", "3s", "Kd"),
                hero_hole_cards=("Ah", "Ad"),
                revealed_opponent_holes=((2, ("Qc", "Jh")),),
            )
            self.assertEqual(telemetry.append_settlements("comp-1", (receipt,)), 1)
            lines = [json.loads(line) for line in path.read_text().splitlines()]
        result = lines[1]
        self.assertEqual(result["event"], "hand_result")
        self.assertEqual(result["telemetry_schema_version"], 4)
        self.assertEqual(result["board_cards"], ["2c", "7d", "9h", "3s", "Kd"])
        self.assertEqual(result["hero_hole_cards"], ["Ah", "Ad"])
        self.assertEqual(result["revealed_opponent_holes"], [[2, ["Qc", "Jh"]]])

    def test_a_bare_receipt_writes_the_null_showdown_vocabulary(self) -> None:
        record = self._record(_decision())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            telemetry = TelemetryLog(path)
            telemetry.append(record)
            receipt = ReplayReceipt("hand-1", "table-1", 2, -50)
            self.assertEqual(telemetry.append_settlements("comp-1", (receipt,)), 1)
            lines = [json.loads(line) for line in path.read_text().splitlines()]
        result = lines[1]
        self.assertIsNone(result["board_cards"])
        self.assertIsNone(result["hero_hole_cards"])
        self.assertIsNone(result["revealed_opponent_holes"])

    def test_parse_replay_receipts_carries_showdown_fields_defensively(self) -> None:
        receipts = parse_replay_receipts(
            [
                {
                    "handId": "hand-1",
                    "tableId": "table-1",
                    "settledAt": 10,
                    "chipDelta": -5,
                    "boardCards": ["2c", "7d", "9h"],
                    "holeCards": ["Ah", "Ad"],
                    "selfSeatNumber": 1,
                    "seats": [
                        {"seatNumber": 1, "holeCards": ["Ah", "Ad"]},
                        {"seatNumber": 2, "holeCards": ["Qc", "Jh"]},
                        {"seatNumber": 3, "holeCards": None},
                        {"seatNumber": 4, "holeCards": ["not-a-card"]},
                    ],
                },
                {
                    "handId": "hand-2",
                    "tableId": "table-2",
                    "settledAt": 11,
                    "chipDelta": 0,
                    "boardCards": "garbage",
                    "holeCards": ["As"],
                    "seats": "garbage",
                },
            ]
        )
        first, second = receipts
        self.assertEqual(first.board_cards, ("2c", "7d", "9h"))
        self.assertEqual(first.hero_hole_cards, ("Ah", "Ad"))
        self.assertEqual(first.revealed_opponent_holes, ((2, ("Qc", "Jh")),))
        self.assertIsNone(second.board_cards)
        self.assertIsNone(second.hero_hole_cards)
        self.assertIsNone(second.revealed_opponent_holes)

    def test_seats_without_a_self_seat_are_not_attributed(self) -> None:
        receipts = parse_replay_receipts(
            [
                {
                    "handId": "hand-1",
                    "tableId": "table-1",
                    "settledAt": 10,
                    "chipDelta": 0,
                    "seats": [{"seatNumber": 2, "holeCards": ["Qc", "Jh"]}],
                }
            ]
        )
        self.assertIsNone(receipts[0].revealed_opponent_holes)


if __name__ == "__main__":
    unittest.main()
