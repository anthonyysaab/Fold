"""Checks for the bluff advisor's wiring into the decision engine."""

from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import patch

from bluff import DEFAULT_BLUFF_SETTINGS

from engine.opponent_model import (
    AggressionTracker,
    TrackerSettings,
)
from engine.poker_policy import AggressivePokerPolicy
from engine.policy_features import FEATURE_NAMES
from engine.training_telemetry import make_decision_record

ALWAYS = dataclasses.replace(
    DEFAULT_BLUFF_SETTINGS,
    steal_frequency=1.0,
    continuation_frequency=1.0,
    semi_bluff_frequency=1.0,
    barrel_frequency=1.0,
    probe_frequency=1.0,
    raise_bluff_frequency=1.0,
    river_frequency=1.0,
)

NEVER = dataclasses.replace(DEFAULT_BLUFF_SETTINGS, bluff_density=0.0)


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
    street: str,
    board: list[str],
    hole_cards: list[str],
    available: list[str],
    pot: int = 400,
    call_chips: int = 0,
    hero_seat: int = 1,
    villain_agent: str | None = None,
    recent_events: list[dict] | None = None,
) -> dict:
    villain: dict = {
        "seatNumber": 2 if hero_seat == 1 else 1,
        "status": "Active",
        "stackChips": 1_000,
        "currentBetChips": call_chips,
        "holeCards": None,
    }
    if villain_agent is not None:
        villain["agentId"] = villain_agent
    hero = {
        "seatNumber": hero_seat,
        "status": "Active",
        "stackChips": 1_000,
        "currentBetChips": 0,
        "holeCards": hole_cards,
    }
    return {
        "id": "bluff-wiring-test",
        "tableId": "bluff-wiring-test",
        "street": street,
        "potChips": pot,
        "currentBet": call_chips,
        "boardCards": board,
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": hero_seat,
        "seats": sorted([hero, villain], key=lambda seat: seat["seatNumber"]),
        "allowedActions": {
            "canFold": "fold" in available,
            "canCheck": "check" in available,
            "canCall": "call" in available,
            "canBet": "bet" in available,
            "canRaise": "raise" in available,
            "canAllIn": False,
            "callAmount": call_chips,
            "callChips": call_chips,
            "callToAmount": call_chips,
            "minBet": 100 if "bet" in available else None,
            "minRaiseTo": 100 if "raise" in available else None,
            "betRange": {"min": 100, "max": 1_000} if "bet" in available else None,
            "raiseRange": {"min": 100, "max": 1_000} if "raise" in available else None,
            "allInToAmount": None,
            "availableActions": available,
            "amountSemantics": "toAmount",
            "reasoningRequired": False,
        },
        "recentEvents": recent_events or [],
    }


def _policy(bluff_settings=ALWAYS, tracker=None) -> AggressivePokerPolicy:
    return AggressivePokerPolicy(
        weights=_weights_favoring_fold(),
        equity_trials=1,
        bluff_settings=bluff_settings,
        opponent_tracker=tracker,
        hyper_aggression_chance=0.0,
    )


def _decide(policy, table, equity):
    with patch.object(policy, "_equity", return_value=equity):
        return policy.decide_with_diagnostics(table)


def _combo_check_spot() -> dict:
    # Checked to the hero on a dry flop with a big combo draw and no
    # showdown value: the model bluff spot.
    return _table(
        street="flop",
        board=["7h", "2h", "9c"],
        hole_cards=["Ah", "Kh"],
        available=["check", "bet"],
    )


class BluffWiringTests(unittest.TestCase):
    def test_passive_spot_becomes_a_priced_semi_bluff(self) -> None:
        decision = _decide(_policy(), _combo_check_spot(), 0.30)

        self.assertEqual(decision.to_payload()["action"], "bet")
        self.assertEqual(decision.to_payload()["amount"], 240)  # 0.6 pot
        self.assertEqual(decision.family, "aggress")
        self.assertEqual(decision.bluff_kind, "probe")
        self.assertIsNotNone(decision.lead_position)

    def test_zero_density_keeps_the_legacy_check(self) -> None:
        decision = _decide(_policy(bluff_settings=NEVER), _combo_check_spot(), 0.30)

        self.assertEqual(decision.to_payload()["action"], "check")
        self.assertIsNone(decision.bluff_kind)

    def test_paired_board_tier_gate_blocks_engine_bluffs(self) -> None:
        table = _table(
            street="turn",
            board=["7s", "7c", "Ks", "7h"],
            hole_cards=["Ac", "3c"],
            available=["check", "bet"],
        )

        decision = _decide(_policy(), table, 0.30)
        self.assertEqual(decision.to_payload()["action"], "check")
        self.assertIsNone(decision.bluff_kind)

    def test_raising_wars_are_never_bluffed(self) -> None:
        table = _table(
            street="flop",
            board=["7h", "2h", "9c"],
            hole_cards=["Ah", "Kh"],
            available=["fold", "call", "raise"],
            call_chips=300,
            recent_events=[
                {
                    "type": "ActionTaken",
                    "street": "flop",
                    "summary": {"seatNumber": 1, "action": "raise", "amount": 100},
                },
                {
                    "type": "ActionTaken",
                    "street": "flop",
                    "summary": {"seatNumber": 2, "action": "raise", "amount": 300},
                },
            ],
        )

        decision = _decide(_policy(), table, 0.20)
        self.assertEqual(decision.to_payload()["action"], "fold")
        self.assertIsNone(decision.bluff_kind)

    def test_observed_maniacs_are_not_bluffed(self) -> None:
        def river_spot() -> dict:
            return _table(
                street="river",
                board=["7h", "2h", "9h", "Jc", "3s"],
                hole_cards=["Ah", "Qd"],
                available=["check", "bet"],
                pot=100,
                hero_seat=2,
                villain_agent="maniac",
                recent_events=[
                    {
                        "type": "ActionTaken",
                        "street": "flop",
                        "summary": {"seatNumber": 2, "action": "bet", "amount": 50},
                    }
                ],
            )

        # Against an unknown opponent the blocker barrel fires.
        fresh = _decide(_policy(), river_spot(), 0.30)
        self.assertEqual(fresh.bluff_kind, "barrel")
        self.assertEqual(fresh.to_payload()["action"], "bet")

        # The same spot against a proven perma-shover is abandoned.
        tracker = AggressionTracker(TrackerSettings(decay=1.0))
        shove_event = [
            {
                "type": "ActionTaken",
                "street": "preflop",
                "summary": {"seatNumber": 1, "action": "all-in", "amount": 1_000},
            }
        ]
        for index in range(10):
            history = _table(
                street="flop",
                board=["7h", "2h", "9c"],
                hole_cards=["Ah", "Kh"],
                available=["check", "bet"],
                hero_seat=2,
                villain_agent="maniac",
                recent_events=shove_event,
            )
            history["tableId"] = f"history-{index}"
            tracker.observe(history)

        informed = _decide(_policy(tracker=tracker), river_spot(), 0.30)
        self.assertIsNone(informed.bluff_kind)
        self.assertEqual(informed.to_payload()["action"], "check")

    def test_bluffs_are_marked_in_telemetry_records(self) -> None:
        decision = _decide(_policy(), _combo_check_spot(), 0.30)
        record = make_decision_record(
            competition_id="comp-1",
            policy_version="heuristic-test-v1",
            table=_combo_check_spot(),
            payload=decision.to_payload(),
            decision=decision,
            deadline_budget_s=5.0,
            fallback_reason=None,
            action_status=200,
            identity_verified=True,
            recorded_at_ms=1,
        )

        self.assertEqual(record["bluff_kind"], "probe")
        self.assertEqual(record["opponent_range_width"], 1.0)
        self.assertIsNotNone(record["lead_position"])
        self.assertTrue(record["training_eligible"])


if __name__ == "__main__":
    unittest.main()
