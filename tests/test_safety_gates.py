"""Checks for the injectable SafetyGates parameter object."""

from __future__ import annotations

import dataclasses
import json
import unittest
from collections.abc import Sequence
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
from devfun_poker_playground.game_state import card_reveal_expense
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


def _stacked_table(
    *,
    hero_stack: int,
    opponent_stacks: Sequence[int],
    pot: int,
    call_chips: int,
    raise_min: int,
    raise_max: int,
    hero_contribution: int = 0,
    folded_stacks: Sequence[int] = (),
    street: str = "flop",
) -> dict:
    """A snapshot with per-seat stacks, so hero and the table can differ.

    ``_table`` above gives every seat the same purse, which is exactly the
    case where hero's own stack and the effective stack agree. The risk-cap
    tests need them to disagree.
    """

    seats: list[dict] = [
        {
            "seatNumber": 1,
            "status": "Active",
            "stackChips": hero_stack,
            "currentBetChips": hero_contribution,
            "holeCards": ["Ah", "Ad"],
        }
    ]
    for stack in opponent_stacks:
        seats.append(
            {
                "seatNumber": len(seats) + 1,
                "status": "Active",
                "stackChips": stack,
                "currentBetChips": 0,
                "holeCards": None,
            }
        )
    for stack in folded_stacks:
        seats.append(
            {
                "seatNumber": len(seats) + 1,
                "status": "Folded",
                "stackChips": stack,
                "currentBetChips": 0,
                "holeCards": None,
            }
        )
    available = ["fold", "call", "raise"] if call_chips else ["check", "raise"]
    return {
        "id": "risk-cap-test",
        "tableId": "risk-cap-test",
        "street": street,
        "potChips": pot,
        "currentBet": hero_contribution + call_chips,
        "boardCards": ["2c", "7d", "9h"],
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": 1,
        "seats": seats,
        "allowedActions": {
            "canFold": bool(call_chips),
            "canCheck": not call_chips,
            "canCall": bool(call_chips),
            "canBet": False,
            "canRaise": True,
            "canAllIn": False,
            "callAmount": call_chips,
            "callChips": call_chips,
            "callToAmount": (hero_contribution + call_chips) if call_chips else None,
            "minBet": None,
            "minRaiseTo": raise_min,
            "betRange": None,
            "raiseRange": {"min": raise_min, "max": raise_max},
            "allInToAmount": None,
            "availableActions": available,
            "amountSemantics": "toAmount",
            "reasoningRequired": False,
        },
        "recentEvents": [],
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


class RiskCapEffectiveStackTests(unittest.TestCase):
    """The sizing risk cap keys on the effective stack, not hero's purse.

    Live forensics, 2026-08-15: the cap was computed as
    ``risk_cap_stack_fraction * hero stackChips``, so it went inert exactly
    as the bankroll grew -- it could bind on 58.6% of sub-near-nut decisions
    at a 2.5k purse, 30.4% at 8.7k, and 4.3% at 12k.
    """

    def _sized(self, policy, table, equity):
        return policy._sized_action("raise", table, table["allowedActions"], equity)

    def test_deep_hero_versus_a_short_opponent_cannot_size_at_all(self) -> None:
        # The live case: hero 11,842 behind, one active opponent with 1,133.
        # Hero's own stack gives a nominal cap of 5,388 (0.455 * 11,842),
        # which is 4.75x everything the opponent can ever pay -- so the cap
        # could not bind and the engine raised past the whole effective
        # stack on 10% equity. On the effective stack the cap is 516, below
        # the 1,200 minimum legal raise, so sizing must decline entirely.
        table = _stacked_table(
            hero_stack=11_842,
            opponent_stacks=[1_133],
            pot=1_000,
            call_chips=600,
            raise_min=1_200,
            raise_max=11_842,
        )
        self.assertIsNone(self._sized(_policy(), table, 0.10))

    def test_deep_hero_regression_case_falls_through_to_the_passive_path(self) -> None:
        table = _stacked_table(
            hero_stack=11_842,
            opponent_stacks=[1_133],
            pot=1_000,
            call_chips=600,
            raise_min=1_200,
            raise_max=11_842,
        )
        policy = _policy()
        with (
            patch.object(policy, "_equity", return_value=0.10),
            patch.object(policy, "_equity_family", return_value="aggress"),
        ):
            decision = policy.decide(table)
        # Declining to size is not declining to act: the engine must still
        # return a legal passive action, and 10% equity does not price a
        # 600-chip call into a 1,000-chip pot.
        self.assertNotEqual(decision["action"], "raise")
        self.assertEqual(decision["action"], "fold")

    def test_hero_as_the_short_stack_still_caps_on_its_own_purse(self) -> None:
        # The bankroll protection the cap was written for is unchanged:
        # effective_stack_chips clamps to hero's stack, so when hero is the
        # short one the denominator is hero's stack exactly as before.
        table = _stacked_table(
            hero_stack=1_133,
            opponent_stacks=[11_842],
            pot=1_000,
            call_chips=600,
            raise_min=100,
            raise_max=1_133,
        )
        self.assertEqual(self._sized(_policy(), table, 0.10), ("raise", 516))

    def test_multiway_cap_uses_the_deepest_active_opponent(self) -> None:
        # Multiway, "effective" is ambiguous. The engine takes the deepest
        # active opponent (bounded by hero) -- the most any single opponent
        # can make hero pay -- so a short third player does not shrink a
        # value bet a deeper opponent can still call.
        table = _stacked_table(
            hero_stack=11_842,
            opponent_stacks=[1_133, 4_000],
            pot=4_000,
            call_chips=600,
            raise_min=100,
            raise_max=11_842,
        )
        self.assertEqual(self._sized(_policy(), table, 0.10), ("raise", 1_820))

    def test_folded_seats_do_not_set_the_cap(self) -> None:
        table = _stacked_table(
            hero_stack=11_842,
            opponent_stacks=[4_000],
            folded_stacks=[12_000],
            pot=4_000,
            call_chips=600,
            raise_min=100,
            raise_max=11_842,
        )
        self.assertEqual(self._sized(_policy(), table, 0.10), ("raise", 1_820))

    def test_equal_stacks_keep_the_pre_existing_cap(self) -> None:
        # Where the two definitions agree -- every seat the same depth, the
        # ordinary case the cap was tuned on -- behaviour is byte-identical.
        table = _stacked_table(
            hero_stack=1_000,
            opponent_stacks=[1_000],
            pot=2_000,
            call_chips=0,
            raise_min=100,
            raise_max=1_000,
        )
        self.assertEqual(self._sized(_policy(), table, 0.30), ("raise", 455))

    def test_near_nut_equity_still_releases_the_cap(self) -> None:
        table = _stacked_table(
            hero_stack=11_842,
            opponent_stacks=[1_133],
            pot=1_000,
            call_chips=600,
            raise_min=1_200,
            raise_max=11_842,
        )
        sized = self._sized(_policy(), table, 0.90)
        assert sized is not None
        self.assertEqual(sized[0], "raise")
        self.assertGreaterEqual(sized[1], 1_200)


class CallGateEffectiveStackTests(unittest.TestCase):
    """Call gates key on the effective stack, and price the next card.

    Every one of the deployment's largest live losses is a CALL, and calls
    were governed by gates denominated in hero's own purse -- so they went
    inert as the bankroll outgrew the table, exactly as the sizing risk cap
    did. Reconstructed from the worst hand on record (2026-08-15,
    table cmsudjx3y8zyhs5zvadxfca1n, -3,768 chips).
    """

    def _calls(self, policy, table, equity) -> bool:
        return policy._call_clears_margin(table, table["allowedActions"], equity)

    def _worst_hand_flop(self, **overrides) -> dict:
        # QsJc on KcJd3c: hero purse 9,143, but the opponent covers only
        # 2,207, and the call is the whole of it. As a share of hero's own
        # stack that is 24% and trips nothing; as a share of what can
        # actually be lost it is 100%.
        values = dict(
            hero_stack=9_143,
            opponent_stacks=[2_207],
            pot=2_326,
            call_chips=2_207,
            raise_min=4_414,
            raise_max=9_143,
            street="flop",
        )
        values.update(overrides)
        return _stacked_table(**values)

    def test_a_call_of_the_whole_effective_stack_is_refused_at_the_live_equity(
        self,
    ) -> None:
        # Hero's own estimate at the moment of the real call was 0.53.
        self.assertFalse(self._calls(_policy(), self._worst_hand_flop(), 0.53))

    def test_the_gate_would_not_fire_on_heros_own_stack(self) -> None:
        """Proves the test is measuring the scoping, not just a tight gate.

        With every seat equally deep the call is a small share of both
        denominators, so the same equity that is refused above must pass --
        otherwise this class would pass for the wrong reason.
        """

        deep = self._worst_hand_flop(hero_stack=9_143, opponent_stacks=[9_143])
        self.assertTrue(self._calls(_policy(), deep, 0.53))

    def test_card_reveal_expense_scales_with_price_and_street(self) -> None:
        flop = self._worst_hand_flop()
        # Whole effective stack on the flop: two cards still to come.
        self.assertAlmostEqual(card_reveal_expense(flop, 2_207), 2 / 3, places=6)
        # Same price on the turn buys one card.
        turn = self._worst_hand_flop(street="turn")
        self.assertAlmostEqual(card_reveal_expense(turn, 2_207), 1 / 3, places=6)
        # On the river there is no card left to buy, so no expense.
        river = self._worst_hand_flop(street="river")
        self.assertEqual(card_reveal_expense(river, 2_207), 0.0)
        # A cheap look costs proportionally less.
        self.assertAlmostEqual(
            card_reveal_expense(flop, 220), (220 / 2_207) * (2 / 3), places=6
        )
        # Never exceeds the whole stack, and a free look is free.
        self.assertEqual(card_reveal_expense(flop, 99_999), 2 / 3)
        self.assertEqual(card_reveal_expense(flop, 0), 0.0)

    def test_the_reveal_penalty_is_what_separates_flop_from_river(self) -> None:
        """Isolates the new term from the scoping fix.

        Priced identically as a share of the effective stack, a call is
        stricter with cards to come than with none. Setting the slope to
        zero must collapse the two, or the parameter is not doing the work
        the comment claims.
        """

        equity = 0.60
        priced = SafetyGates()
        flop = self._worst_hand_flop(
            hero_stack=4_000, opponent_stacks=[4_000], call_chips=1_900, pot=2_400
        )
        river = self._worst_hand_flop(
            hero_stack=4_000,
            opponent_stacks=[4_000],
            call_chips=1_900,
            pot=2_400,
            street="river",
        )
        self.assertFalse(self._calls(_policy(priced), flop, equity))
        self.assertTrue(self._calls(_policy(priced), river, equity))

        neutral = SafetyGates(reveal_expense_equity_slope=0.0)
        self.assertTrue(self._calls(_policy(neutral), flop, equity))

    def test_slope_is_validated(self) -> None:
        SafetyGates(reveal_expense_equity_slope=0.0)
        SafetyGates(reveal_expense_equity_slope=0.5)
        with self.assertRaises(ValueError):
            SafetyGates(reveal_expense_equity_slope=-0.01)
        with self.assertRaises(ValueError):
            SafetyGates(reveal_expense_equity_slope=0.51)


if __name__ == "__main__":
    unittest.main()
