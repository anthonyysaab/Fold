"""Checks for the bounded temperature response of the decision engine."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.decision_engine import (
    DecisionEngine,
    DEFAULT_TEMPERATURE_SHAPING,
    NEUTRAL_TEMPERATURE_SHAPING,
    TemperatureShaping,
)
from engine.poker_policy import AggressivePokerPolicy
from engine.policy_features import FEATURE_NAMES


def _weights_favoring_fold() -> dict:
    return {
        "input_size": len(FEATURE_NAMES),
        "hidden_size": 1,
        "feature_names": list(FEATURE_NAMES),
        "labels": ["fold", "check_call", "aggress"],
        "w1": [[0.0] * len(FEATURE_NAMES)],
        "b1": [0.0],
        "w2": [[0.0], [0.0], [0.0]],
        "b2": [5.0, 0.0, 0.0],
    }


def _table(
    *,
    players: int,
    street: str,
    board: list[str],
    available: list[str],
    pot: int = 400,
    call_chips: int = 0,
    hole_cards: list[str] | None = None,
) -> dict:
    can_raise = "raise" in available
    return {
        "id": "temperature-shaping-test",
        "tableId": "temperature-shaping-test",
        "street": street,
        "potChips": pot,
        "currentBet": call_chips,
        "boardCards": board,
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": 1,
        "seats": [
            {
                "seatNumber": seat_number,
                "status": "Active",
                "stackChips": 1_000,
                "currentBetChips": 0,
                "holeCards": (hole_cards or ["Ah", "Ad"]) if seat_number == 1 else None,
            }
            for seat_number in range(1, players + 1)
        ],
        "allowedActions": {
            "canFold": "fold" in available,
            "canCheck": "check" in available,
            "canCall": "call" in available,
            "canBet": False,
            "canRaise": can_raise,
            "canAllIn": False,
            "callAmount": call_chips,
            "callChips": call_chips,
            "callToAmount": call_chips,
            "minBet": None,
            "minRaiseTo": 100 if can_raise else None,
            "betRange": None,
            "raiseRange": {"min": 100, "max": 1_000} if can_raise else None,
            "allInToAmount": None,
            "availableActions": available,
            "amountSemantics": "toAmount",
            "reasoningRequired": False,
        },
        "recentEvents": [],
    }


def _policy(shaping: TemperatureShaping | None = None) -> AggressivePokerPolicy:
    return AggressivePokerPolicy(
        weights=_weights_favoring_fold(),
        equity_trials=1,
        temperature_shaping=shaping,
        hyper_aggression_chance=0.0,
    )


def _decide(policy: AggressivePokerPolicy, table: dict, equity: float) -> dict:
    with patch.object(policy, "_equity", return_value=equity):
        return policy.decide(table)


class TemperatureShapingTests(unittest.TestCase):
    def test_validation_rejects_out_of_range_parameters(self) -> None:
        for kwargs in (
            {"setpoint": 101.0},
            {"span": 1.0},
            {"aggression_floor_shift": 0.2},
            {"call_margin_shift": 0.2},
            {"sizing_span": 0.6},
        ):
            with self.assertRaises(ValueError):
                TemperatureShaping(**kwargs)

    def test_boldness_is_signed_and_clamped(self) -> None:
        shaping = DEFAULT_TEMPERATURE_SHAPING
        self.assertEqual(shaping.boldness(shaping.setpoint), 0.0)
        self.assertEqual(shaping.boldness(0.0), 1.0)
        self.assertEqual(shaping.boldness(100.0), -1.0)
        self.assertLess(shaping.boldness(80.0), 0.0)
        self.assertGreater(shaping.boldness(20.0), 0.0)

    def test_engine_defaults_to_active_shaping(self) -> None:
        self.assertIs(
            DecisionEngine().temperature_shaping, DEFAULT_TEMPERATURE_SHAPING
        )

    def test_cold_situation_lowers_the_aggression_floor(self) -> None:
        # Equity 0.41 sits between the shifted floor (~0.3946 at reading
        # 33.6) and the aggressive base floor (0.42), so only the shaped
        # policy raises.
        table = _table(
            players=2,
            street="flop",
            board=["2c", "7d", "9h"],
            available=["check", "raise"],
        )

        self.assertEqual(_decide(_policy(), table, 0.41)["action"], "raise")
        self.assertEqual(
            _decide(_policy(NEUTRAL_TEMPERATURE_SHAPING), table, 0.41)["action"],
            "check",
        )

    def test_hot_situation_tightens_the_call_margin(self) -> None:
        # Six players, a weak hand, and a 30%-of-stack bet read 55.0, so the
        # shaped margin grows to ~0.0311 and the borderline call becomes a
        # fold; the neutral policy still calls at the legacy margin.
        table = _table(
            players=6,
            street="turn",
            board=["2c", "7d", "9h", "Js"],
            available=["fold", "call"],
            pot=623,
            call_chips=300,
        )

        self.assertEqual(
            _decide(_policy(NEUTRAL_TEMPERATURE_SHAPING), table, 0.35)["action"],
            "call",
        )
        self.assertEqual(_decide(_policy(), table, 0.35)["action"], "fold")

    def test_cold_situation_sizes_up_within_the_legal_range(self) -> None:
        # Reading 18.0 -> boldness +0.771 -> the half-pot target grows to
        # ~0.650 of the pot: 260 chips instead of the neutral 200.
        table = _table(
            players=2,
            street="flop",
            board=["2c", "7d", "9h"],
            available=["check", "raise"],
        )

        shaped = _decide(_policy(), table, 0.80)
        neutral = _decide(_policy(NEUTRAL_TEMPERATURE_SHAPING), table, 0.80)
        self.assertEqual(shaped["amount"], 260)
        self.assertEqual(neutral["amount"], 200)

    def test_hard_gates_do_not_shift_when_cold(self) -> None:
        # Board-only strength keeps its softened 0.724 aggression floor even
        # though the reading is ice cold; the shaped policy still checks.
        table = _table(
            players=2,
            street="turn",
            board=["7s", "7c", "Ks", "7h"],
            available=["check", "raise"],
            hole_cards=["Ac", "3c"],
        )

        self.assertEqual(_decide(_policy(), table, 0.70)["action"], "check")

    def test_boldness_is_reported_and_stays_out_of_the_payload(self) -> None:
        table = _table(
            players=2,
            street="turn",
            board=["2c", "7d", "9h", "Js"],
            available=["fold", "call"],
            pot=100,
            call_chips=20,
        )
        policy = _policy()

        with patch.object(policy, "_equity", return_value=0.60):
            decision = policy.decide_with_diagnostics(table)
        self.assertAlmostEqual(decision.temperature_boldness, (45.0 - 21.6) / 35.0)
        self.assertNotIn("temperature_boldness", decision.to_payload())

        with patch.object(policy, "_equity", return_value=0.60):
            deadline = policy.decide_with_diagnostics(table, deadline_s=1.0)
        self.assertIsNone(deadline.temperature_boldness)
        self.assertTrue(deadline.deadline_fallback)


if __name__ == "__main__":
    unittest.main()
