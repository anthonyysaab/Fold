"""Checks for the hardcoded anti-modeling hyper-aggression dice roll."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.poker_policy import AggressivePokerPolicy
from engine.training_telemetry import make_decision_record


def _table(
    *,
    street: str = "flop",
    board: list[str] | None = None,
    hole_cards: list[str] | None = None,
    available: list[str] | None = None,
    call_chips: int = 0,
    recent_events: list[dict] | None = None,
) -> dict:
    board = board if board is not None else ["2c", "7d", "9h"]
    available = available or ["check", "raise"]
    can_raise = "raise" in available
    can_bet = "bet" in available
    return {
        "id": "hyper-test",
        "tableId": "hyper-test",
        "street": street,
        "potChips": 400,
        "currentBet": call_chips,
        "boardCards": board,
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": 1,
        "seats": [
            {
                "seatNumber": number,
                "status": "Active",
                "stackChips": 1_000,
                "currentBetChips": 0,
                "holeCards": (hole_cards or ["Ah", "Ad"]) if number == 1 else None,
            }
            for number in (1, 2)
        ],
        "allowedActions": {
            "canFold": "fold" in available,
            "canCheck": "check" in available,
            "canCall": "call" in available,
            "canBet": can_bet,
            "canRaise": can_raise,
            "canAllIn": False,
            "callAmount": call_chips,
            "callChips": call_chips,
            "callToAmount": call_chips,
            "minBet": 100 if can_bet else None,
            "minRaiseTo": 100 if can_raise else None,
            "betRange": {"min": 100, "max": 1_000} if can_bet else None,
            "raiseRange": {"min": 100, "max": 1_000} if can_raise else None,
            "allInToAmount": None,
            "availableActions": available,
            "amountSemantics": "toAmount",
            "reasoningRequired": False,
        },
        "recentEvents": recent_events or [],
    }


def _policy(chance: float) -> AggressivePokerPolicy:
    return AggressivePokerPolicy(
        equity_trials=1,
        hyper_aggression_chance=chance,
    )


def _decide(policy, table, equity):
    with patch.object(policy, "_equity", return_value=equity):
        return policy.decide_with_diagnostics(table)


class HyperAggressionTests(unittest.TestCase):
    def test_hyper_turns_a_passive_spot_into_full_pot_pressure(self) -> None:
        calm = _decide(_policy(0.0), _table(), 0.30)
        wild = _decide(_policy(1.0), _table(), 0.30)

        self.assertEqual(calm.to_payload()["action"], "check")
        self.assertFalse(calm.hyper_aggression)
        self.assertEqual(wild.to_payload()["action"], "raise")
        self.assertEqual(wild.to_payload()["amount"], 400)  # full pot
        self.assertTrue(wild.hyper_aggression)
        # The private flag never leaks toward Arena.
        self.assertNotIn("hyper", str(wild.to_payload()))

    def test_hard_gates_still_veto_hyper_decisions(self) -> None:
        board_owned = _table(
            street="turn",
            board=["7s", "7c", "Ks", "7h"],
            hole_cards=["Ac", "3c"],
        )
        decision = _decide(_policy(1.0), board_owned, 0.70)
        self.assertEqual(decision.to_payload()["action"], "check")

        war = _table(
            available=["fold", "call", "raise"],
            call_chips=300,
            recent_events=[
                {
                    "type": "ActionTaken",
                    "street": "flop",
                    "summary": {"seatNumber": 1, "action": "raise", "amount": 100},
                }
            ],
        )
        raise_back = _decide(_policy(1.0), war, 0.50)
        self.assertNotEqual(raise_back.to_payload()["action"], "raise")

    def test_risk_cap_still_bounds_hyper_sizing(self) -> None:
        table = _table()
        table["potChips"] = 2_000  # full pot would be 2,000, cap is 455
        decision = _decide(_policy(1.0), table, 0.30)
        self.assertEqual(decision.to_payload()["action"], "raise")
        self.assertEqual(decision.to_payload()["amount"], 455)

    def test_roll_is_deterministic_for_a_snapshot(self) -> None:
        policy = _policy(0.05)
        first = _decide(policy, _table(), 0.30).hyper_aggression
        second = _decide(policy, _table(), 0.30).hyper_aggression
        self.assertEqual(first, second)

    def test_hyper_decisions_are_marked_and_training_ineligible(self) -> None:
        decision = _decide(_policy(1.0), _table(), 0.30)
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
        self.assertTrue(record["hyper_aggression"])
        self.assertFalse(record["training_eligible"])

        calm = _decide(_policy(0.0), _table(), 0.30)
        calm_record = make_decision_record(
            competition_id="comp-1",
            policy_version="heuristic-test-v1",
            table=_table(),
            payload=calm.to_payload(),
            decision=calm,
            deadline_budget_s=5.0,
            fallback_reason=None,
            action_status=200,
            identity_verified=True,
            recorded_at_ms=1,
        )
        self.assertFalse(calm_record["hyper_aggression"])
        self.assertTrue(calm_record["training_eligible"])

    def test_chance_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            _policy(1.5)


if __name__ == "__main__":
    unittest.main()
