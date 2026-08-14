"""Checks for the injectable SafetyGates parameter object."""

from __future__ import annotations

import dataclasses
import json
import unittest
from unittest.mock import patch

from devfun_poker_playground.decision_engine import (
    DecisionEngine,
    DEFAULT_SAFETY_GATES,
    DEFAULT_TEMPERATURE_SHAPING,
    SafetyGates,
    TemperatureShaping,
    UNSOFTENED_SAFETY_GATES,
)
from devfun_poker_playground.poker_policy import (
    AGGRESSIVE_SAFETY_GATES,
    AggressivePokerPolicy,
)
from devfun_poker_playground.policy_features import FEATURE_NAMES


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
    recent_events: list[dict] | None = None,
) -> dict:
    can_raise = "raise" in available
    return {
        "id": "safety-gates-test",
        "tableId": "safety-gates-test",
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
        "recentEvents": recent_events or [],
    }


def _policy(gates: SafetyGates | None = None) -> AggressivePokerPolicy:
    return AggressivePokerPolicy(
        weights=_weights_favoring_fold(),
        equity_trials=1,
        safety_gates=gates,
        hyper_aggression_chance=0.0,
    )


def _decide(policy: AggressivePokerPolicy, table: dict, equity: float) -> dict:
    with patch.object(policy, "_equity", return_value=equity):
        return policy.decide(table)


class SafetyGatesTests(unittest.TestCase):
    def test_defaults_are_the_softened_values(self) -> None:
        gates = DEFAULT_SAFETY_GATES
        self.assertEqual(gates.near_nut_floor, 0.654)
        self.assertEqual(gates.risk_cap_stack_fraction, 0.455)
        self.assertEqual(gates.board_aggression_floor_kicker, 0.724)
        self.assertEqual(gates.call_stack_gates, ((0.78, 0.626), (0.455, 0.584)))

    def test_unsoftened_constant_preserves_the_original_values(self) -> None:
        gates = UNSOFTENED_SAFETY_GATES
        self.assertEqual(gates.near_nut_floor, 0.72)
        self.assertEqual(gates.risk_cap_stack_fraction, 0.35)
        self.assertEqual(gates.board_stackoff_kicker, (0.18, 0.75))
        self.assertEqual(gates.rescue_call_margin, 0.15)

    def test_validation_rejects_out_of_range_gates(self) -> None:
        for kwargs in (
            {"near_nut_floor": 0.45},  # below break-even equity
            {"board_margin_kicker": 0.5},
            {"risk_cap_stack_fraction": 0.0},
            {"board_stackoff_thin": (0.0, 0.71)},
            {"call_stack_gates": ((0.5, 0.4),)},  # floor below 0.5
            {"rescue_call_floor": 1.2},
        ):
            with self.assertRaises(ValueError, msg=f"no error for {kwargs}"):
                SafetyGates(**kwargs)

    def test_mapping_round_trip_is_lossless_and_json_ready(self) -> None:
        for gates in (DEFAULT_SAFETY_GATES, UNSOFTENED_SAFETY_GATES):
            mapping = json.loads(json.dumps(gates.to_mapping()))
            self.assertEqual(SafetyGates.from_mapping(mapping), gates)
        shaping = DEFAULT_TEMPERATURE_SHAPING
        mapping = json.loads(json.dumps(shaping.to_mapping()))
        self.assertEqual(TemperatureShaping.from_mapping(mapping), shaping)
        with self.assertRaises(TypeError):
            SafetyGates.from_mapping({"unknown_gate": 1.0})

    def test_engine_and_policies_pick_their_default_gates(self) -> None:
        self.assertIs(DecisionEngine().safety_gates, DEFAULT_SAFETY_GATES)
        self.assertIs(_policy().safety_gates, AGGRESSIVE_SAFETY_GATES)
        self.assertEqual(AGGRESSIVE_SAFETY_GATES.call_stack_gates, ((0.65, 0.584),))
        explicit = _policy(UNSOFTENED_SAFETY_GATES)
        self.assertIs(explicit.safety_gates, UNSOFTENED_SAFETY_GATES)

    def test_injected_near_nut_floor_changes_the_war_decision(self) -> None:
        table = _table(
            players=2,
            street="flop",
            board=["2c", "7d", "9h"],
            available=["fold", "call", "raise"],
            call_chips=20,
            recent_events=[
                {
                    "type": "ActionTaken",
                    "street": "flop",
                    "summary": {"seatNumber": 1, "action": "raise", "amount": 100},
                }
            ],
        )

        # 0.65 equity sits below the default 0.654 war floor but above a
        # custom 0.60 floor, so only the custom-gate policy re-raises.
        self.assertEqual(_decide(_policy(), table, 0.65)["action"], "call")
        custom = dataclasses.replace(AGGRESSIVE_SAFETY_GATES, near_nut_floor=0.60)
        self.assertEqual(_decide(_policy(custom), table, 0.65)["action"], "raise")

    def test_unsoftened_gates_restore_the_stricter_board_tier(self) -> None:
        table = _table(
            players=2,
            street="turn",
            board=["7s", "7c", "Ks", "7h"],
            available=["check", "raise"],
            hole_cards=["Ac", "3c"],
        )

        # 0.75 equity clears the softened 0.724 kicker floor but not the
        # original 0.82 one.
        self.assertEqual(_decide(_policy(), table, 0.75)["action"], "raise")
        self.assertEqual(
            _decide(_policy(UNSOFTENED_SAFETY_GATES), table, 0.75)["action"],
            "check",
        )


if __name__ == "__main__":
    unittest.main()
