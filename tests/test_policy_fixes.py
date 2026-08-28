"""Regression checks for multiway routing and policy-specific thresholds."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.game_state import ArenaSnapshotError
from engine.opponent_model import (
    AggressionTracker,
    TrackerSettings,
)
from engine.poker_policy import AggressivePokerPolicy
from engine.learning_contract import LEARNING_INPUT_SIZE
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
    recent_events: list[dict] | None = None,
) -> dict:
    can_raise = "raise" in available
    return {
        "id": "policy-fix-test",
        "tableId": "policy-fix-test",
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


class PolicyFixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = AggressivePokerPolicy(
            weights=_weights_favoring_fold(),
            equity_trials=1,
            hyper_aggression_chance=0.0,  # pinned decisions need no dice
        )

    def _decide_with_equity(self, table: dict, equity: float) -> dict:
        with patch.object(self.policy, "_equity", return_value=equity):
            return self.policy.decide(table)

    def _diagnose_with_equity(self, table: dict, equity: float):
        with patch.object(self.policy, "_equity", return_value=equity):
            return self.policy.decide_with_diagnostics(table)

    def test_six_player_table_uses_aggressive_policy_instead_of_network(self) -> None:
        table = _table(
            players=6,
            street="flop",
            board=["2c", "7d", "9h"],
            available=["check", "raise"],
        )

        self.assertEqual(self._decide_with_equity(table, 0.80)["action"], "raise")

    def test_aggressive_threshold_is_not_rechecked_by_tighter_base_gate(self) -> None:
        table = _table(
            players=2,
            street="flop",
            board=["2c", "7d", "9h"],
            available=["check", "raise"],
        )

        self.assertEqual(self._decide_with_equity(table, 0.45)["action"], "raise")

    def test_folded_seats_do_not_raise_the_aggression_threshold(self) -> None:
        table = _table(
            players=6,
            street="flop",
            board=["2c", "7d", "9h"],
            available=["check", "raise"],
        )
        for seat in table["seats"][2:]:
            seat["status"] = "Folded"

        self.assertEqual(self._decide_with_equity(table, 0.45)["action"], "raise")

    def test_aggressive_call_margin_remains_authoritative_during_action(self) -> None:
        table = _table(
            players=2,
            street="turn",
            board=["2c", "7d", "9h", "Js"],
            available=["fold", "call"],
            pot=100,
            call_chips=20,
        )

        self.assertEqual(self._decide_with_equity(table, 0.20)["action"], "call")

    def test_board_only_strength_still_blocks_aggression(self) -> None:
        table = _table(
            players=2,
            street="turn",
            board=["7s", "7c", "Ks", "7h"],
            available=["check", "raise"],
            hole_cards=["Ac", "3c"],
        )

        # The kicker-tier aggression floor is 0.724 after the 2026-08-12
        # 30% softening (was 0.82): below it still checks, above now bets.
        self.assertEqual(self._decide_with_equity(table, 0.70)["action"], "check")
        self.assertEqual(self._decide_with_equity(table, 0.75)["action"], "raise")

    def test_raise_back_still_requires_near_nut_equity(self) -> None:
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
                    "summary": {
                        "seatNumber": 1,
                        "action": "raise",
                        "amount": 100,
                    },
                }
            ],
        )

        self.assertEqual(self._decide_with_equity(table, 0.65)["action"], "call")

    def test_null_call_to_amount_in_check_spot_still_sizes_raises(self) -> None:
        # Live 2026-08-13 regression: Arena nulls callToAmount whenever
        # nothing is left to call (all 1,117 no-call decisions in the foreign
        # corpus), and the raise-sizing path rejected the whole snapshot.
        table = _table(
            players=2,
            street="flop",
            board=["2c", "7d", "9h"],
            available=["check", "raise"],
        )
        table["allowedActions"]["callToAmount"] = None
        table["allowedActions"]["callAmount"] = None

        payload = self._decide_with_equity(table, 0.80)

        self.assertEqual(payload["action"], "raise")
        raise_range = table["allowedActions"]["raiseRange"]
        self.assertGreaterEqual(payload["amount"], raise_range["min"])
        self.assertLessEqual(payload["amount"], raise_range["max"])

    def test_null_call_to_amount_facing_a_live_bet_still_fails(self) -> None:
        # With chips left to call, a missing call target is an inconsistent
        # snapshot and must keep failing validation.
        table = _table(
            players=2,
            street="flop",
            board=["2c", "7d", "9h"],
            available=["fold", "call", "raise"],
            call_chips=20,
        )
        table["allowedActions"]["callToAmount"] = None

        with self.assertRaises(ArenaSnapshotError):
            self._decide_with_equity(table, 0.95)

    def test_temperature_is_logged_without_leaking_into_arena_payload(self) -> None:
        table = _table(
            players=2,
            street="turn",
            board=["2c", "7d", "9h", "Js"],
            available=["fold", "call"],
            pot=100,
            call_chips=20,
        )

        decision = self._diagnose_with_equity(table, 0.60)

        self.assertEqual(decision.situation_temperature.temperature, 21.6)
        self.assertEqual(len(decision.learning_features), LEARNING_INPUT_SIZE)
        self.assertNotIn("temperature", decision.to_payload())


# The 2026-08-13 bust hand's flop war, verbatim from V5_BUST_POSTMORTEM.md:
# hero bets 7, then four villain raises escalate to 912 before the all-in.
_BUST_HAND_WAR = (
    (1, "bet", 7),
    (2, "raise", 14),
    (1, "raise", 39),
    (2, "raise", 64),
    (1, "raise", 156),
    (2, "raise", 248),
    (1, "raise", 580),
    (2, "raise", 912),
)


def _war_events(triples) -> list[dict]:
    return [
        {
            "type": "ActionTaken",
            "street": "flop",
            "summary": {"seatNumber": seat, "action": action, "amount": amount},
        }
        for seat, action, amount in triples
    ]


class RangeFloorEscalationTests(unittest.TestCase):
    """2026-08-13 bust regression: escalation now outranks the global floor."""

    def _policy(self) -> AggressivePokerPolicy:
        policy = AggressivePokerPolicy(
            weights=_weights_favoring_fold(),
            equity_trials=1,
            opponent_tracker=AggressionTracker(TrackerSettings(decay=1.0)),
            hyper_aggression_chance=0.0,
        )
        # One observed history hand: 21 aggressive of 46 actions pins the
        # villain's global frequency at 21 / (46 + 4) = 0.42, the bust
        # villain's tracked shape.
        history = _table(
            players=2,
            street="preflop",
            board=[],
            available=["fold", "call"],
            pot=200,
            call_chips=100,
            hole_cards=["Ah", "9h"],
            recent_events=[
                {
                    "type": "ActionTaken",
                    "street": "preflop",
                    "summary": {"seatNumber": 2, "action": action, "amount": amount},
                }
                for action, base, count in (("raise", 10, 21), ("call", 500, 25))
                for amount in range(base, base + count)
            ],
        )
        history["id"] = history["tableId"] = "history"
        history["seats"][1]["agentId"] = "attacker"
        policy.opponent_tracker.observe(history)
        return policy

    def _width(self, policy: AggressivePokerPolicy, table: dict) -> float:
        # decide() observes the snapshot before conditioning; mirror that.
        policy.opponent_tracker.observe(table)
        return policy._call_top_fraction(table, table["allowedActions"])

    def test_bust_shape_models_the_escalating_villain_below_030(self) -> None:
        policy = self._policy()
        table = _table(
            players=3,
            street="flop",
            board=["Ac", "Th", "Tc"],
            available=["fold", "call", "raise"],
            pot=1_500,
            call_chips=800,
            hole_cards=["Td", "6c"],
            recent_events=_war_events(_BUST_HAND_WAR),
        )
        table["seats"][1]["agentId"] = "attacker"

        width = self._width(policy, table)

        # The flat floor held this spot at the villain's global 0.4237 on
        # the live hand; four raises of escalation now outrank it.
        self.assertLess(width, 0.30)
        self.assertLess(width, policy.opponent_tracker.range_floor("attacker"))

    def test_one_raise_this_hand_still_pays_the_full_global_floor(self) -> None:
        policy = self._policy()
        table = _table(
            players=2,
            street="flop",
            board=["7s", "7c", "7h"],
            available=["fold", "call", "raise"],
            pot=300,
            call_chips=200,
            hole_cards=["Ac", "3c"],
            recent_events=_war_events(((2, "raise", 100),)),
        )
        table["seats"][1]["agentId"] = "attacker"

        width = self._width(policy, table)

        # Kicker-tier board and a pot-sized raise push the raw conditioning
        # to 0.27; a single raise leaves the ~0.43 global floor in charge.
        self.assertAlmostEqual(width, policy.opponent_tracker.range_floor("attacker"))
        self.assertGreater(width, 0.40)

    def test_hard_conditioning_minimum_survives_the_decay(self) -> None:
        policy = self._policy()
        table = _table(
            players=2,
            street="flop",
            board=["7s", "7c", "7h"],
            available=["fold", "call", "raise"],
            pot=300,
            call_chips=200,
            hole_cards=["Ac", "3c"],
            recent_events=_war_events(
                (
                    (2, "raise", 100),
                    (2, "raise", 220),
                    (2, "raise", 470),
                    (2, "raise", 940),
                )
            ),
        )
        table["seats"][1]["agentId"] = "attacker"

        # Raw conditioning and the decayed floor both land under 0.20 here;
        # the clamp is applied after the decay and holds the hard minimum.
        self.assertAlmostEqual(self._width(policy, table), 0.20)


if __name__ == "__main__":
    unittest.main()
