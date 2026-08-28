from __future__ import annotations

import unittest

from engine.decision_engine import DecisionEngine
from engine.learning_contract import LEARNING_FEATURE_NAMES
from tools.collect_foreign_play_data import _decision_row, _receipt_from_table


class ForeignPlayCollectorTests(unittest.TestCase):
    def test_paginated_table_summary_becomes_a_settlement_receipt(self) -> None:
        receipt = _receipt_from_table(
            {
                "id": "table-1",
                "status": "Completed",
                "endedAt": "2026-08-12T08:00:00.000Z",
                "winners": [{"agentId": "villain"}],
                "seats": [
                    {
                        "agentId": "hero",
                        "agentHandle": "hero-handle",
                        "chipDelta": -4,
                    },
                    {
                        "agentId": "villain",
                        "agentHandle": "winner-handle",
                        "chipDelta": 4,
                    },
                ],
            },
            "hero",
        )

        self.assertEqual(receipt["handId"], "table-1")
        self.assertEqual(receipt["chipDelta"], -4)
        self.assertEqual(receipt["winnerHandle"], "winner-handle")
        self.assertEqual(receipt["settledAt"], 1_786_521_600_000)

    def test_action_row_reconstructs_pre_action_state_and_features(self) -> None:
        seats = [
            {
                "seatNumber": 1,
                "agentId": "hero",
                "agentName": "Teacher",
                "status": "Active",
                "stackChips": 94,
                "currentBetChips": 6,
                "totalCommittedChips": 6,
                "holeCards": ["As", "Kd"],
            },
            {
                "seatNumber": 2,
                "agentId": "villain",
                "agentName": "Opponent",
                "status": "Active",
                "stackChips": 98,
                "currentBetChips": 2,
                "totalCommittedChips": 2,
                "holeCards": ["Qc", "Jh"],
            },
        ]
        action = {
            "sequence": 5,
            "type": "ActionTaken",
            "street": "Preflop",
            "occurredAt": 1_700_000_001_000,
            "agentId": "hero",
            "payload": {
                "action": "raise",
                "toAmount": 6,
                "pot": 3,
                "callAmount": 1,
                "seatNumber": 1,
                "stackBefore": 99,
                "currentBetBefore": 2,
                "dealerSeatNumber": 1,
                "minRaiseToBefore": 4,
                "actorCurrentBetBefore": 1,
                "allowedActions": {
                    "availableActions": ["fold", "call", "raise", "all-in"],
                    "canFold": True,
                    "canCheck": False,
                    "canCall": True,
                    "canBet": False,
                    "canRaise": True,
                    "canAllIn": True,
                    "callChips": 1,
                    "callToAmount": 2,
                    "minRaiseTo": 4,
                    "maxCommit": 100,
                    "raiseRange": {"min": 4, "max": 100},
                    "allInToAmount": 100,
                },
            },
            "snapshot": {
                "street": "Preflop",
                "potChips": 8,
                "currentBet": 6,
                "minRaiseTo": 10,
                "boardCards": [],
                "seats": seats,
            },
        }
        replay = {
            "table": {
                "id": "table-1",
                "tableId": "table-1",
                "tableNumber": 1,
                "competitionId": "competition-1",
                "startedAt": 1_700_000_000_000,
                "smallBlindChips": 1,
                "bigBlindChips": 2,
                "winners": [{"agentId": "hero", "handName": "Uncontested"}],
            },
            "events": [
                {
                    "sequence": 1,
                    "type": "BlindPosted",
                    "street": "Preflop",
                    "payload": {"seatNumber": 1, "amount": 1},
                },
                {
                    "sequence": 2,
                    "type": "BlindPosted",
                    "street": "Preflop",
                    "payload": {"seatNumber": 2, "amount": 2},
                },
                action,
            ],
        }
        row = _decision_row(
            replay,
            action,
            {
                "rank": 1,
                "id": "hero",
                "name": "Teacher",
                "totalScore": 1200,
                "totalSubmissions": 50,
            },
            {
                "handId": "hand-1",
                "tableId": "table-1",
                "settledAt": 1_700_000_002_000,
                "chipDelta": 5,
                "winnerHandle": "teacher",
            },
            DecisionEngine(equity_trials=10, seed=7),
        )

        self.assertEqual(row["pot_before_chips"], 3)
        self.assertEqual(row["hero_stack_before_chips"], 99)
        self.assertEqual(row["hero_bet_before_chips"], 1)
        self.assertEqual(row["action_new_chips"], 5)
        self.assertEqual(row["action_family"], "aggress")
        self.assertTrue(row["teacher_eligible"])
        self.assertEqual(
            len([name for name in row if name.startswith("feature_")]),
            len(LEARNING_FEATURE_NAMES),
        )


if __name__ == "__main__":
    unittest.main()
