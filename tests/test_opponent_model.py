"""Checks for aggression tracking, escalation decay, and the perma-shove remedy."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.opponent_model import (
    AggressionTracker,
    TrackerSettings,
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


def _shove_table(*, table_id: str, hole_cards: list[str], agent_id: str | None) -> dict:
    """Heads-up snapshot: the opponent has open-shoved for 1,000 chips."""

    villain: dict = {
        "seatNumber": 2,
        "status": "Active",
        "stackChips": 0,
        "currentBetChips": 1_000,
        "holeCards": None,
    }
    if agent_id is not None:
        villain["agentId"] = agent_id
    return {
        "id": table_id,
        "tableId": table_id,
        "street": "preflop",
        "potChips": 1_100,
        "currentBet": 900,
        "boardCards": [],
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": 1,
        "seats": [
            {
                "seatNumber": 1,
                "status": "Active",
                "stackChips": 1_000,
                "currentBetChips": 100,
                "holeCards": hole_cards,
            },
            villain,
        ],
        "allowedActions": {
            "canFold": True,
            "canCheck": False,
            "canCall": True,
            "canBet": False,
            "canRaise": False,
            "canAllIn": False,
            "callAmount": 900,
            "callChips": 900,
            "callToAmount": 1_000,
            "minBet": None,
            "minRaiseTo": None,
            "betRange": None,
            "raiseRange": None,
            "allInToAmount": None,
            "availableActions": ["fold", "call"],
            "amountSemantics": "toAmount",
            "reasoningRequired": False,
        },
        "recentEvents": [
            {
                "type": "ActionTaken",
                "street": "preflop",
                "summary": {"seatNumber": 2, "action": "all-in", "amount": 1_000},
            }
        ],
    }


_HANDS = (
    ["Ah", "9h"],
    ["Kd", "8c"],
    ["Qs", "Jc"],
    ["7d", "6d"],
    ["Tc", "9s"],
    ["Ad", "4c"],
    ["Kh", "Qc"],
    ["8s", "8d"],
    ["Js", "9d"],
    ["Ac", "Th"],
)


class TrackerTests(unittest.TestCase):
    def test_settings_validate_and_round_trip(self) -> None:
        settings = TrackerSettings(prior_opportunities=6.0, decay=0.95)
        self.assertEqual(TrackerSettings.from_mapping(settings.to_mapping()), settings)
        for kwargs in ({"prior_opportunities": 0.0}, {"decay": 0.5}, {"decay": 1.1}):
            with self.assertRaises(ValueError, msg=f"no error for {kwargs}"):
                TrackerSettings(**kwargs)

    def test_floor_starts_at_zero_and_converges_to_shove_frequency(self) -> None:
        tracker = AggressionTracker(TrackerSettings(decay=1.0))
        self.assertEqual(tracker.range_floor("shover"), 0.0)
        for index, hole in enumerate(_HANDS):
            tracker.observe(
                _shove_table(table_id=f"t{index}", hole_cards=hole, agent_id="shover")
            )
        # Ten shoves in ten observed actions: floor = 10 / (10 + 4).
        self.assertAlmostEqual(tracker.range_floor("shover"), 10.0 / 14.0)

    def test_same_hand_polls_do_not_double_count(self) -> None:
        tracker = AggressionTracker(TrackerSettings(decay=1.0))
        table = _shove_table(table_id="t1", hole_cards=["Ah", "9h"], agent_id="s")
        tracker.observe(table)
        tracker.observe(table)  # second poll of the same pending action
        self.assertEqual(tracker.opportunities("s"), 1.0)

    def test_new_hole_cards_reset_the_hand_bookkeeping(self) -> None:
        tracker = AggressionTracker(TrackerSettings(decay=1.0))
        tracker.observe(
            _shove_table(table_id="t1", hole_cards=["Ah", "9h"], agent_id="s")
        )
        tracker.observe(
            _shove_table(table_id="t1", hole_cards=["Kd", "8c"], agent_id="s")
        )
        self.assertEqual(tracker.opportunities("s"), 2.0)

    def test_stickiness_separates_stations_from_nits(self) -> None:
        tracker = AggressionTracker(TrackerSettings(decay=1.0))
        self.assertEqual(tracker.stickiness("anyone"), 0.0)  # unknown != sticky
        for index, hole in enumerate(_HANDS):
            table = _shove_table(
                table_id=f"t{index}", hole_cards=hole, agent_id="station"
            )
            table["recentEvents"] = [
                {
                    "type": "ActionTaken",
                    "street": "preflop",
                    "summary": {"seatNumber": 2, "action": "call", "amount": 100},
                }
            ]
            tracker.observe(table)
            nit_table = _shove_table(
                table_id=f"n{index}", hole_cards=hole, agent_id="nit"
            )
            nit_table["recentEvents"] = [
                {
                    "type": "ActionTaken",
                    "street": "preflop",
                    "summary": {"seatNumber": 2, "action": "fold"},
                }
            ]
            tracker.observe(nit_table)
        # Ten calls vs ten folds: 10/(10+4) against 0/(10+4).
        self.assertAlmostEqual(tracker.stickiness("station"), 10.0 / 14.0)
        self.assertEqual(tracker.stickiness("nit"), 0.0)
        # Both stay indistinguishable to the aggression-frequency floor.
        self.assertEqual(tracker.range_floor("station"), 0.0)
        self.assertEqual(tracker.range_floor("nit"), 0.0)

    def test_pressure_profile_uses_one_actor_and_reports_confidence(self) -> None:
        tracker = AggressionTracker(TrackerSettings(decay=1.0))
        for index, hole in enumerate(_HANDS):
            tracker.observe(
                _shove_table(table_id=f"wild-{index}", hole_cards=hole, agent_id="wild")
            )
            station = _shove_table(
                table_id=f"station-{index}", hole_cards=hole, agent_id="station"
            )
            station["recentEvents"][0]["summary"]["action"] = "call"
            tracker.observe(station)

        pressured = _shove_table(
            table_id="current", hole_cards=["Ah", "Kd"], agent_id="station"
        )
        pressured["seats"].append(
            {
                "seatNumber": 3,
                "agentId": "wild",
                "status": "Active",
                "stackChips": 1_000,
                "currentBetChips": 0,
                "holeCards": None,
            }
        )
        wildness, stickiness, confidence = tracker.pressure_actor_profile(pressured)

        self.assertEqual(wildness, 0.0)
        self.assertAlmostEqual(stickiness, 10.0 / 14.0)
        self.assertAlmostEqual(confidence, 10.0 / 14.0)

    def test_passive_opponents_keep_a_zero_floor(self) -> None:
        tracker = AggressionTracker(TrackerSettings(decay=1.0))
        table = _shove_table(table_id="t1", hole_cards=["Ah", "9h"], agent_id="caller")
        table["recentEvents"] = [
            {
                "type": "ActionTaken",
                "street": "preflop",
                "summary": {"seatNumber": 2, "action": "call", "amount": 100},
            }
        ]
        tracker.observe(table)
        self.assertEqual(tracker.opportunities("caller"), 1.0)
        self.assertEqual(tracker.range_floor("caller"), 0.0)

    def test_anonymous_seats_fall_back_to_per_table_keys(self) -> None:
        tracker = AggressionTracker(TrackerSettings(decay=1.0))
        tracker.observe(
            _shove_table(table_id="t1", hole_cards=["Ah", "9h"], agent_id=None)
        )
        tracker.observe(
            _shove_table(table_id="t2", hole_cards=["Kd", "8c"], agent_id=None)
        )
        self.assertEqual(tracker.opportunities("t1:2"), 1.0)
        self.assertEqual(tracker.opportunities("t2:2"), 1.0)


def _action_events(
    seat_number: int, action: str, count: int, *, start_amount: int = 10
) -> list[dict]:
    """Events with distinct amounts so per-hand dedup keys never collide."""

    return [
        {
            "type": "ActionTaken",
            "street": "preflop",
            "summary": {
                "seatNumber": seat_number,
                "action": action,
                "amount": start_amount + index,
            },
        }
        for index in range(count)
    ]


def _tracked_history(
    tracker: AggressionTracker, *, agent_id: str, raises: int, calls: int
) -> None:
    """Feed one observed hand establishing an opponent's global frequency."""

    table = _shove_table(
        table_id=f"history-{agent_id}", hole_cards=["Ah", "9h"], agent_id=agent_id
    )
    table["recentEvents"] = _action_events(2, "raise", raises) + _action_events(
        2, "call", calls, start_amount=500
    )
    tracker.observe(table)


class EscalationDecayTests(unittest.TestCase):
    """The 2026-08-13 bust fix: within-hand escalation decays the floor."""

    def _tracker_with_042_attacker(self) -> AggressionTracker:
        tracker = AggressionTracker(TrackerSettings(decay=1.0))
        # 21 aggressive of 46 observed actions: floor = 21 / (46 + 4) = 0.42,
        # the bust villain's tracked frequency.
        _tracked_history(tracker, agent_id="attacker", raises=21, calls=25)
        return tracker

    def _current_hand(self, raise_count: int) -> dict:
        table = _shove_table(
            table_id="current", hole_cards=["Ah", "Kd"], agent_id="attacker"
        )
        table["recentEvents"] = _action_events(2, "raise", raise_count, start_amount=40)
        return table

    def test_one_raise_this_hand_leaves_the_global_floor_unchanged(self) -> None:
        tracker = self._tracker_with_042_attacker()
        self.assertAlmostEqual(tracker.range_floor("attacker"), 0.42)
        self.assertAlmostEqual(
            tracker.range_floor_for_table(self._current_hand(1)), 0.42
        )

    def test_three_raises_this_hand_cut_the_floor_by_at_least_half(self) -> None:
        tracker = self._tracker_with_042_attacker()
        floor = tracker.range_floor_for_table(self._current_hand(3))
        self.assertLessEqual(floor, 0.42 / 2)
        self.assertAlmostEqual(floor, 0.42 * 0.7**2)

    def test_five_raises_this_hand_drop_a_042_opponent_below_015(self) -> None:
        tracker = self._tracker_with_042_attacker()
        floor = tracker.range_floor_for_table(self._current_hand(5))
        self.assertLess(floor, 0.15)
        self.assertAlmostEqual(floor, 0.42 * 0.7**4)

    def test_escalation_is_attributed_per_opponent(self) -> None:
        tracker = self._tracker_with_042_attacker()
        # 9 aggressive of 11 observed actions: floor = 9 / (11 + 4) = 0.6.
        _tracked_history(tracker, agent_id="steady", raises=9, calls=2)
        table = self._current_hand(3)  # the attacker escalates three times
        table["seats"].append(
            {
                "seatNumber": 3,
                "agentId": "steady",
                "status": "Active",
                "stackChips": 1_000,
                "currentBetChips": 0,
                "holeCards": None,
            }
        )
        table["recentEvents"] += _action_events(3, "raise", 1, start_amount=900)
        # steady's single raise keeps its full 0.6 floor; pooling all four
        # raises onto one actor would wrongly decay it to 0.6 * 0.7 ** 3.
        self.assertAlmostEqual(tracker.range_floor_for_table(table), 0.6)


class PermaShoveRemedyTests(unittest.TestCase):
    """The exploit that bled Fold-ver-3 dry: fold, fold, fold to any shove."""

    def _decide(self, policy, table):
        # Equity grows with the simulated range width, mirroring reality:
        # a decent hand fares better against random than against the nuts.
        def fake_equity(hole, board, opponents, trials, seed, top_fraction=1.0):
            return 0.42 + 0.28 * top_fraction

        with patch(
            "engine.decision_engine.estimate_equity",
            side_effect=fake_equity,
        ):
            return policy.decide_with_diagnostics(table)

    def test_repeated_shoves_widen_the_range_until_the_bot_calls(self) -> None:
        policy = AggressivePokerPolicy(
            weights=_weights_favoring_fold(),
            equity_trials=1,
            opponent_tracker=AggressionTracker(TrackerSettings(decay=1.0)),
            hyper_aggression_chance=0.0,
        )

        first = self._decide(
            policy,
            _shove_table(table_id="t0", hole_cards=_HANDS[0], agent_id="attacker"),
        )
        # One shove still reads as strength: narrow range, low equity, fold.
        self.assertEqual(first.to_payload()["action"], "fold")
        self.assertAlmostEqual(first.opponent_range_width, 0.45)

        decisions = [first]
        for index, hole in enumerate(_HANDS[1:], start=1):
            decisions.append(
                self._decide(
                    policy,
                    _shove_table(
                        table_id=f"t{index}", hole_cards=hole, agent_id="attacker"
                    ),
                )
            )

        widths = [decision.opponent_range_width for decision in decisions]
        self.assertEqual(widths, sorted(widths))  # evidence only widens it
        last = decisions[-1]
        self.assertGreater(last.opponent_range_width, 0.64)
        # The tenth identical shove gets called instead of folded through.
        self.assertEqual(last.to_payload()["action"], "call")


if __name__ == "__main__":
    unittest.main()
