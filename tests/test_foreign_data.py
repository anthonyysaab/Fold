"""Checks for the foreign-CSV training boundary and the audit summary."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from engine.foreign_data import (
    FEATURE_COLUMNS,
    ForeignDataError,
    load_foreign_training_examples,
)
from engine.learning_contract import LEARNING_INPUT_SIZE
from tools.audit_foreign_play_data import summarize_rows

_META = {
    "season_number": "13",
    "agent_id": "agent-a",
    "table_id": "table-1",
    "street": "flop",
    "action": "bet",
    "action_family": "aggress",
    "action_new_chips": "60",
    "action_effective_stack_fraction": "0.06",
    "hand_reward_bb": "2.5",
    "teacher_eligible": "True",
    "selected_action_is_legal": "True",
    "equity_top_fraction": "0.8",
    "call_chips": "0",
    "pot_before_chips": "100",
    "big_blind_chips": "100",
    "hero_stack_before_chips": "1000",
    "hero_total_committed_before_chips": "0",
    "effective_stack_before_chips": "1000",
    "active_player_count": "2",
}


def _row(**overrides) -> dict:
    row = dict(_META)
    row.update({column: "0.5" for column in FEATURE_COLUMNS})
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class ForeignDataLoaderTests(unittest.TestCase):
    def _load(self, rows: list[dict], **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            _write_csv(path, rows)
            return load_foreign_training_examples(path, **kwargs)

    def test_eligible_rows_become_contract_examples(self) -> None:
        rows = [
            _row(),
            _row(
                table_id="table-2",
                action_family="fold",
                action_effective_stack_fraction="0.0",
                hand_reward_bb="-1.0",
            ),
            _row(table_id="table-3", teacher_eligible="False", action="timeout"),
        ]

        examples = self._load(rows)
        self.assertEqual(len(examples), 2)
        first = examples[0]
        self.assertEqual(first.policy_version, "foreign-teacher-s13-agent-a")
        self.assertEqual(len(first.features), LEARNING_INPUT_SIZE)
        self.assertEqual(first.action_family_index, 2)
        self.assertEqual(first.behavior_probabilities, (0.0, 0.0, 1.0))
        self.assertEqual(first.submitted_risk_fraction, 0.06)
        self.assertEqual(first.purse_bb, 10.0)
        self.assertEqual(first.reward_bb, 2.5)
        self.assertFalse(first.counterfactual)
        self.assertEqual(first.opponent_confidence, 0.0)
        self.assertEqual(examples[1].action_family_index, 0)

        everything = self._load(rows, eligible_only=False)
        self.assertEqual(len(everything), 3)

    def test_beyond_effective_stack_commitments_clamp_to_one(self) -> None:
        examples = self._load([_row(action_effective_stack_fraction="90.2")])
        self.assertEqual(examples[0].submitted_risk_fraction, 1.0)

    def test_malformed_rows_are_refused(self) -> None:
        for overrides in (
            {"action_family": "limp"},
            {"action_effective_stack_fraction": "-0.1"},
            {"hand_reward_bb": "nan"},
            {FEATURE_COLUMNS[0]: "not-a-number"},
            {"selected_action_is_legal": "False"},
            {"agent_id": ""},
        ):
            with self.assertRaises(ForeignDataError, msg=f"no error for {overrides}"):
                self._load([_row(**overrides)])

    def test_missing_feature_columns_are_refused(self) -> None:
        row = _row()
        del row[FEATURE_COLUMNS[-1]]
        with self.assertRaises(ForeignDataError):
            self._load([row])


class AuditSummaryTests(unittest.TestCase):
    def test_summary_counts_and_calibration_rates(self) -> None:
        rows = [
            _row(),  # aggress, unopened flop
            _row(action_family="fold", call_chips="50", street="turn"),
            _row(action_family="check_call", call_chips="50", street="turn"),
            _row(teacher_eligible="False"),
        ]

        summary = summarize_rows(rows)
        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["eligible_rows"], 3)
        self.assertEqual(summary["family_mix"]["aggress"]["count"], 1)
        turn = summary["facing_bet_response_by_street"]["turn"]
        self.assertEqual(turn["decisions"], 2)
        self.assertEqual(turn["fold_rate"], 0.5)
        self.assertEqual(
            summary["unopened_postflop_by_street"]["flop"]["bet_rate"], 1.0
        )
        self.assertEqual(
            summary["aggressive_sizing_pot_fraction"]["flop"]["median"], 0.6
        )


if __name__ == "__main__":
    unittest.main()
