"""Checks for the injectable SafetyGates parameter object."""

from __future__ import annotations

import dataclasses
import json
import unittest
from collections.abc import Sequence
from unittest.mock import patch

from engine.decision_engine import (
    DecisionEngine,
    DEFAULT_SAFETY_GATES,
    DEFAULT_TEMPERATURE_SHAPING,
    SafetyGates,
    TemperatureShaping,
    UNSOFTENED_SAFETY_GATES,
)
from engine.poker_policy import (
    AGGRESSIVE_SAFETY_GATES,
    AggressivePokerPolicy,
)
from engine.game_state import (
    card_reveal_expense,
    contested_stack_chips,
    effective_stack_chips,
)


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
        equity_trials=1,
        safety_gates=gates,
        hyper_aggression_chance=0.0,
    )


#: The two gate configurations, both named explicitly. Which one is the
#: DEFAULT has flipped twice on 2026-08-26 (reverted after measurement, then
#: rolled back after the deployed revert busted the live bankroll), so no test
#: below infers a configuration from the default -- exactly one test asserts
#: what the default currently is, and everything else asks for what it means
#: to exercise. A future flip should change that one test, not six.
EFFECTIVE_STACK_GATES = dataclasses.replace(
    AGGRESSIVE_SAFETY_GATES,
    risk_cap_on_effective_stack=True,
    call_gates_on_effective_stack=True,
    reveal_expense_equity_slope=0.12,
)
#: The pre-2026-08-15 form: both gates on hero's own purse, no reveal slope.
HERO_PURSE_GATES = dataclasses.replace(
    AGGRESSIVE_SAFETY_GATES,
    risk_cap_on_effective_stack=False,
    call_gates_on_effective_stack=False,
    reveal_expense_equity_slope=0.0,
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
    """What the effective-stack risk cap does, when it is asked for.

    **Reverted out of the default on 2026-08-26**; these cases pin the
    behaviour so the option stays honest, not because it ships.

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
        self.assertIsNone(self._sized(_policy(EFFECTIVE_STACK_GATES), table, 0.10))

    def test_deep_hero_regression_case_falls_through_to_the_passive_path(self) -> None:
        table = _stacked_table(
            hero_stack=11_842,
            opponent_stacks=[1_133],
            pot=1_000,
            call_chips=600,
            raise_min=1_200,
            raise_max=11_842,
        )
        policy = _policy(EFFECTIVE_STACK_GATES)
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
        self.assertEqual(self._sized(_policy(EFFECTIVE_STACK_GATES), table, 0.10), ("raise", 516))

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
        self.assertEqual(self._sized(_policy(EFFECTIVE_STACK_GATES), table, 0.10), ("raise", 1_820))

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
        self.assertEqual(self._sized(_policy(EFFECTIVE_STACK_GATES), table, 0.10), ("raise", 1_820))

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
        self.assertEqual(self._sized(_policy(EFFECTIVE_STACK_GATES), table, 0.30), ("raise", 455))

    def test_near_nut_equity_still_releases_the_cap(self) -> None:
        table = _stacked_table(
            hero_stack=11_842,
            opponent_stacks=[1_133],
            pot=1_000,
            call_chips=600,
            raise_min=1_200,
            raise_max=11_842,
        )
        sized = self._sized(_policy(EFFECTIVE_STACK_GATES), table, 0.90)
        assert sized is not None
        self.assertEqual(sized[0], "raise")
        self.assertGreaterEqual(sized[1], 1_200)


class CallGateEffectiveStackTests(unittest.TestCase):
    """What the effective-stack call gates do, when they are asked for.

    **Reverted out of the default on 2026-08-26**; pinned, not shipped.

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
        self.assertFalse(self._calls(_policy(EFFECTIVE_STACK_GATES), self._worst_hand_flop(), 0.53))

    def test_the_gate_would_not_fire_on_heros_own_stack(self) -> None:
        """Proves the test is measuring the scoping, not just a tight gate.

        With every seat equally deep the call is a small share of both
        denominators, so the same equity that is refused above must pass --
        otherwise this class would pass for the wrong reason.
        """

        deep = self._worst_hand_flop(hero_stack=9_143, opponent_stacks=[9_143])
        self.assertTrue(self._calls(_policy(EFFECTIVE_STACK_GATES), deep, 0.53))

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
        priced = dataclasses.replace(
            SafetyGates(),
            risk_cap_on_effective_stack=True,
            call_gates_on_effective_stack=True,
            reveal_expense_equity_slope=0.12,
        )
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

        neutral = dataclasses.replace(priced, reveal_expense_equity_slope=0.0)
        self.assertTrue(self._calls(_policy(neutral), flop, equity))

    def test_slope_is_validated(self) -> None:
        SafetyGates(reveal_expense_equity_slope=0.0)
        SafetyGates(reveal_expense_equity_slope=0.5)
        with self.assertRaises(ValueError):
            SafetyGates(reveal_expense_equity_slope=-0.01)
        with self.assertRaises(ValueError):
            SafetyGates(reveal_expense_equity_slope=0.51)


class GateDenominatorDialTests(unittest.TestCase):
    """The gate dials, after the 2026-08-26 revert.

    All three 2026-08-15 edits were reverted on 2026-08-26 following
    ``artifacts/evaluations/gate-decision-2026-08-26.md``: on ``vs-p3``,
    the first battery channel whose opponent folds by its own cards,
    reverting is worth +7.58 BB/100 for the cap, +16.02 for the call
    gates and +16.49 for all three, every one resolved against a paired
    MDE. Card-awareness narrows the gates' measured cost and does not
    reverse it, which is the question the earlier card-blind -13.97
    BB/100 left open. Busts sit inside ``bb_per_100``, so those figures
    are already net of the ruin the gates buy.

    The dials remain so the decision stays reversible and re-measurable.
    """

    def _sized(self, policy, table, equity):
        return policy._sized_action("raise", table, table["allowedActions"], equity)

    def _calls(self, policy, table, equity) -> bool:
        return policy._call_clears_margin(table, table["allowedActions"], equity)

    def test_the_current_default_is_the_effective_stack_configuration(self) -> None:
        """The single test that pins which configuration ships.

        Rolled back to this on 2026-08-26 after the deployed revert busted
        the live bankroll (1,000 -> 0 in 36 hands). Everything else in this
        file names its configuration explicitly, so a future flip is a
        one-line change here rather than a rewrite.
        """

        for gates in (DEFAULT_SAFETY_GATES, AGGRESSIVE_SAFETY_GATES):
            self.assertTrue(gates.risk_cap_on_effective_stack)
            self.assertTrue(gates.call_gates_on_effective_stack)
            self.assertEqual(gates.reveal_expense_equity_slope, 0.12)

    def test_a_frozen_manifest_inherits_whatever_the_default_is(self) -> None:
        """The gap that made all of this possible, in both directions.

        ``candidate-v7-0001c``'s frozen ``safety_gates`` block names none
        of the three fields, so ``from_mapping`` fills them from the
        dataclass defaults. That is how three unmeasured changes shipped
        under an approved artifact -- and on 2026-08-26 it is also why a
        pointer rollback alone was behaviourally inert, because the
        approved artifact simply inherited the flipped defaults. The gap
        is closed only by pinning the fields in the manifest.
        """

        frozen = {
            "call_stack_gates": [[0.78, 0.626], [0.455, 0.584]],
            "risk_cap_stack_fraction": 0.455,
            "near_nut_floor": 0.654,
        }
        inherited = SafetyGates.from_mapping(frozen)
        self.assertEqual(
            inherited.risk_cap_on_effective_stack,
            DEFAULT_SAFETY_GATES.risk_cap_on_effective_stack,
        )
        self.assertEqual(
            inherited.call_gates_on_effective_stack,
            DEFAULT_SAFETY_GATES.call_gates_on_effective_stack,
        )
        self.assertEqual(
            inherited.reveal_expense_equity_slope,
            DEFAULT_SAFETY_GATES.reveal_expense_equity_slope,
        )

    def test_an_explicit_manifest_pin_survives_a_default_change(self) -> None:
        """Pinning is what makes a served configuration auditable."""

        pinned = SafetyGates.from_mapping(
            {
                "risk_cap_on_effective_stack": True,
                "call_gates_on_effective_stack": True,
                "reveal_expense_equity_slope": 0.12,
            }
        )
        self.assertTrue(pinned.risk_cap_on_effective_stack)
        self.assertTrue(pinned.call_gates_on_effective_stack)
        self.assertEqual(pinned.reveal_expense_equity_slope, 0.12)

    def test_a_short_hero_cannot_tell_the_two_denominators_apart(self) -> None:
        # Impossible by construction: min(1_133, 11_842) == 1_133.
        table = _stacked_table(
            hero_stack=1_133,
            opponent_stacks=[11_842],
            pot=1_000,
            call_chips=600,
            raise_min=100,
            raise_max=1_133,
        )
        live = _policy(EFFECTIVE_STACK_GATES)
        reverted = _policy(HERO_PURSE_GATES)
        for effective in (True, False):
            self.assertEqual(live._gate_stack(table, effective=effective), 1_133)
            self.assertEqual(reverted._gate_stack(table, effective=effective), 1_133)
        self.assertEqual(
            self._sized(live, table, 0.10), self._sized(reverted, table, 0.10)
        )

    def test_a_covering_hero_is_where_the_dial_binds(self) -> None:
        table = _stacked_table(
            hero_stack=11_842,
            opponent_stacks=[1_133],
            pot=1_000,
            call_chips=600,
            raise_min=1_200,
            raise_max=11_842,
        )
        policy = _policy(EFFECTIVE_STACK_GATES)
        self.assertEqual(policy._gate_stack(table, effective=True), 1_133)
        self.assertEqual(policy._gate_stack(table, effective=False), 11_842)

    def test_the_reverted_default_restores_the_5388_cap(self) -> None:
        """The pre-2026-08-15 arithmetic, now the shipped one.

        On the effective stack the cap is 516, below the 1,200 minimum
        legal raise, so sizing declines entirely. On hero's purse it is
        round(0.455 * 11_842) = 5,388.
        """

        table = _stacked_table(
            hero_stack=11_842,
            opponent_stacks=[1_133],
            pot=1_000,
            call_chips=600,
            raise_min=1_200,
            raise_max=11_842,
        )
        self.assertIsNone(self._sized(_policy(EFFECTIVE_STACK_GATES), table, 0.10))

        reverted = self._sized(_policy(HERO_PURSE_GATES), table, 0.10)
        self.assertIsNotNone(reverted)
        action, amount = reverted
        self.assertEqual(action, "raise")
        self.assertLessEqual(amount, 5_388)
        self.assertGreaterEqual(amount, 1_200)

    def test_the_reverted_default_allows_the_worst_live_hands_call(self) -> None:
        """2,207 into a 2,207 effective stack at 0.53 -- the -3,768 hand.

        The revert re-admits this call. That is the cost side of the
        decision and it is deliberate: the edit that refuses it measured
        +16.02 BB/100 against, and on the live journal it refuses at most
        8 of 363 calls.
        """

        table = _stacked_table(
            hero_stack=9_143,
            opponent_stacks=[2_207],
            pot=2_326,
            call_chips=2_207,
            raise_min=4_414,
            raise_max=9_143,
            street="flop",
        )
        self.assertFalse(self._calls(_policy(EFFECTIVE_STACK_GATES), table, 0.53))
        self.assertTrue(self._calls(_policy(HERO_PURSE_GATES), table, 0.53))

    def test_the_two_edits_stay_independently_switchable(self) -> None:
        cap_only = dataclasses.replace(
            HERO_PURSE_GATES, risk_cap_on_effective_stack=True
        )
        self.assertTrue(cap_only.risk_cap_on_effective_stack)
        self.assertFalse(cap_only.call_gates_on_effective_stack)

        calls_only = dataclasses.replace(
            HERO_PURSE_GATES,
            call_gates_on_effective_stack=True,
            reveal_expense_equity_slope=0.12,
        )
        self.assertFalse(calls_only.risk_cap_on_effective_stack)
        self.assertTrue(calls_only.call_gates_on_effective_stack)

    def test_the_all_in_denominator_collapse_cuts_both_ways(self) -> None:
        """``PENDING_EDITS``, 2026-08-26 -- the defect that is also a shield.

        ``effective_stack_chips`` counts chips *behind* and an all-in seat
        stays active with 0, so the effective denominator collapses to
        ``max(1, 0)`` and trips every call gate at any price. Hero's own
        purse cannot.

        Both consequences are real and they point opposite ways. It folds
        an 84-chip call into a 2,328 pot at 0.69 equity, which is
        indefensible. It ALSO folds a 602-chip call by a hand that turned
        out to be drawing dead, which on 2026-08-26 was the difference
        between busting the bankroll and not. The gate fires whenever the
        opponent is all-in -- i.e. exactly where the most chips are at
        stake -- so it is crude protection rather than no protection, and
        that is why the fix is to denominate it correctly rather than to
        pick a side of the flag.
        """

        table = _stacked_table(
            hero_stack=9_143,
            opponent_stacks=[0],
            pot=2_328,
            call_chips=84,
            raise_min=168,
            raise_max=9_143,
            street="flop",
        )
        # candidate-v7-0001c's OWN manifest gates, which is what shipped.
        # The effect depends on the floor: at the aggressive preset's 0.584
        # the reveal penalty of 0.08 does not quite reach 0.69, so the fold
        # below only appears under the incumbent's 0.626 first gate.
        served = dataclasses.replace(
            EFFECTIVE_STACK_GATES,
            call_stack_gates=((0.78, 0.626), (0.455, 0.584)),
        )
        effective = _policy(served)
        purse = _policy(dataclasses.replace(served, **{
            "risk_cap_on_effective_stack": False,
            "call_gates_on_effective_stack": False,
            "reveal_expense_equity_slope": 0.0,
        }))
        self.assertEqual(effective._gate_stack(table, effective=True), 1)
        self.assertEqual(purse._gate_stack(table, effective=False), 9_143)
        # 84 into 2,328 at 0.69 equity is a 27:1 price.
        self.assertFalse(self._calls(effective, table, 0.69))
        self.assertTrue(self._calls(purse, table, 0.69))


class ContestedStackTests(unittest.TestCase):
    """Fix A: counting an opponent's committed chips in the denominator.

    ``effective_stack_chips`` counts chips BEHIND, and an all-in seat stays
    active with 0, so the denominator collapses to 0 -> clamped to 1 the
    moment the last live opponent shoves. Every call stack gate then trips
    at any positive price and ``card_reveal_expense`` saturates on top.
    ``PENDING_EDITS.md``, 2026-08-26.
    """

    @staticmethod
    def _table(
        *,
        street="flop",
        board=("Kc", "Jd", "3c"),
        pot=2_328,
        to_call=84,
        hero_stack=9_143,
        hero_bet=0,
        opponents=((0, 2_244),),
    ) -> dict:
        seats = [
            {
                "seatNumber": 4,
                "status": "Active",
                "stackChips": hero_stack,
                "currentBetChips": hero_bet,
                "holeCards": ["Tc", "As"],
            }
        ]
        for index, (behind, committed) in enumerate(opponents, start=1):
            seats.append(
                {
                    "seatNumber": index,
                    "status": "AllIn" if behind == 0 else "Active",
                    "stackChips": behind,
                    "currentBetChips": committed,
                }
            )
        return {
            "id": "t",
            "tableId": "t",
            "street": street,
            "potChips": pot,
            "currentBet": hero_bet + to_call,
            "boardCards": list(board),
            "smallBlindChips": 2,
            "bigBlindChips": 4,
            "selfSeatNumber": 4,
            "seats": seats,
            "allowedActions": {
                "canFold": True,
                "canCheck": False,
                "canCall": True,
                "canBet": False,
                "canRaise": False,
                "canAllIn": False,
                "callAmount": to_call,
                "callChips": to_call,
                "callToAmount": hero_bet + to_call,
                "minBet": None,
                "minRaiseTo": None,
                "betRange": None,
                "raiseRange": None,
                "allInToAmount": None,
                "availableActions": ["fold", "call"],
                "amountSemantics": "toAmount",
                "reasoningRequired": False,
            },
            "recentEvents": [],
        }

    def _calls(self, policy, table, equity) -> bool:
        return policy._call_clears_margin(table, table["allowedActions"], equity)

    def test_the_dial_ships_off(self) -> None:
        """A loosening change to a live safety gate does not ride a default."""

        for gates in (DEFAULT_SAFETY_GATES, AGGRESSIVE_SAFETY_GATES):
            self.assertFalse(gates.gate_stack_counts_committed_chips)

    def test_contested_is_never_below_effective(self) -> None:
        """Impossible by construction: it adds a non-negative before the max."""

        for opponents in (
            ((0, 2_244),),
            ((500, 0),),
            ((500, 300),),
            ((0, 1),),
            ((0, 0),),
            ((900, 100), (0, 4_000), (50, 25)),
        ):
            table = self._table(opponents=opponents)
            with self.subTest(opponents=opponents):
                self.assertGreaterEqual(
                    contested_stack_chips(table), effective_stack_chips(table)
                )

    def test_a_positive_price_can_never_exceed_the_contested_stack(self) -> None:
        """The invariant the behind-only form lacks, which IS the defect.

        The seat holding the high bet is an active opponent whose
        ``currentBetChips`` is at least hero's contribution plus the price,
        so neither arm of the ``min`` can fall below ``callChips``.
        """

        for to_call, opponents in (
            (84, ((0, 2_244),)),
            (602, ((0, 823),)),
            (1_170, ((0, 2_958),)),
            (1, ((0, 1),)),
        ):
            table = self._table(to_call=to_call, opponents=opponents)
            with self.subTest(to_call=to_call):
                self.assertGreaterEqual(contested_stack_chips(table), to_call)
                # And the form it repairs does NOT hold the same floor.
                self.assertLess(effective_stack_chips(table), to_call)

    def test_an_all_in_opponent_collapses_only_the_behind_count(self) -> None:
        table = self._table(opponents=((0, 2_244),))
        self.assertEqual(effective_stack_chips(table), 0)
        self.assertEqual(contested_stack_chips(table), 2_244)

    def test_the_repair_reproduces_the_documented_denominator(self) -> None:
        """823 on the -1,043 turn call, the figure PENDING_EDITS cites."""

        table = self._table(
            street="turn",
            board=("6h", "8c", "Td", "Qd"),
            pot=1_486,
            to_call=602,
            hero_stack=1_709,
            hero_bet=221,
            opponents=((0, 823),),
        )
        self.assertEqual(contested_stack_chips(table), 823)
        off = _policy(EFFECTIVE_STACK_GATES)
        on = _policy(
            dataclasses.replace(
                EFFECTIVE_STACK_GATES, gate_stack_counts_committed_chips=True
            )
        )
        # Both refuse -- but only one of them evaluated the spot.
        self.assertEqual(off._gate_stack(table, effective=True), 1)
        self.assertEqual(on._gate_stack(table, effective=True), 823)
        self.assertFalse(self._calls(off, table, 0.3675))
        self.assertFalse(self._calls(on, table, 0.3675))

    def test_the_repair_removes_the_indefensible_fold(self) -> None:
        """FAILS ON THE UNFIXED CODE. This is the discriminating case.

        84 into a 2,328 pot at 0.69 equity is a 27:1 price, declined by a
        gate that exists to prevent overcommitment, in a state where no
        opponent can bet again. Off, the clamped denominator folds it; on,
        the real denominator of 2,244 never trips a gate.
        """

        served = dataclasses.replace(
            EFFECTIVE_STACK_GATES,
            call_stack_gates=((0.78, 0.626), (0.455, 0.584)),
        )
        table = self._table(opponents=((0, 2_244),))
        off = _policy(served)
        on = _policy(
            dataclasses.replace(served, gate_stack_counts_committed_chips=True)
        )
        self.assertEqual(off._gate_stack(table, effective=True), 1)
        self.assertEqual(on._gate_stack(table, effective=True), 2_244)
        self.assertFalse(self._calls(off, table, 0.69))
        self.assertTrue(self._calls(on, table, 0.69))

    def test_the_dial_is_inert_when_no_opponent_has_committed(self) -> None:
        """No committed chips anywhere means the two denominators agree."""

        table = self._table(opponents=((2_244, 0),))
        self.assertEqual(
            contested_stack_chips(table), effective_stack_chips(table)
        )
        off = _policy(EFFECTIVE_STACK_GATES)
        on = _policy(
            dataclasses.replace(
                EFFECTIVE_STACK_GATES, gate_stack_counts_committed_chips=True
            )
        )
        self.assertEqual(
            off._gate_stack(table, effective=True),
            on._gate_stack(table, effective=True),
        )

    def test_the_dial_never_reaches_the_hero_purse_branch(self) -> None:
        """It selects within the effective branch and nowhere else."""

        table = self._table(opponents=((0, 2_244),))
        on = _policy(
            dataclasses.replace(
                EFFECTIVE_STACK_GATES, gate_stack_counts_committed_chips=True
            )
        )
        self.assertEqual(on._gate_stack(table, effective=False), 9_143)


class UncallableOverhangTests(unittest.TestCase):
    """Pricing a call on chips hero cannot win.

    An opponent who bets more than hero can match gets the excess back,
    so it never joins a pot hero is contesting -- but ``potChips`` counts
    it, and the price reads low exactly where the most is at risk.

    Live 2026-08-26, table ``cmtael1m86iff11453wifw45v``: a 2,958 shove
    into hero's 1,181 purse left 1,777 uncallable and the price read
    0.2823 where it was really 0.4943. Hero called off the stack with
    Ah 4s at an estimated 0.515 and lost 1,181 -- half that day's bust.
    """

    @staticmethod
    def _shove_table(*, hero_stack=1_170, hero_bet=11, opp_bet=2_958, pot=2_974):
        to_call = hero_stack
        return {
            "id": "t",
            "tableId": "t",
            "street": "preflop",
            "potChips": pot,
            "currentBet": opp_bet,
            "boardCards": [],
            "smallBlindChips": 1,
            "bigBlindChips": 2,
            "selfSeatNumber": 5,
            "seats": [
                {
                    "seatNumber": 5,
                    "status": "Active",
                    "stackChips": hero_stack,
                    "currentBetChips": hero_bet,
                    "holeCards": ["4s", "Ah"],
                },
                {
                    "seatNumber": 6,
                    "status": "AllIn",
                    "stackChips": 0,
                    "currentBetChips": opp_bet,
                },
            ],
            "allowedActions": {
                "canFold": True,
                "canCheck": False,
                "canCall": True,
                "canBet": False,
                "canRaise": False,
                "canAllIn": True,
                "callAmount": to_call,
                "callChips": to_call,
                "callToAmount": hero_bet + to_call,
                "minBet": None,
                "minRaiseTo": None,
                "betRange": None,
                "raiseRange": None,
                "allInToAmount": hero_bet + to_call,
                "availableActions": ["all-in", "call", "fold"],
                "amountSemantics": "toAmount",
                "reasoningRequired": False,
            },
            "recentEvents": [],
        }

    def test_the_dial_ships_off(self) -> None:
        for gates in (DEFAULT_SAFETY_GATES, AGGRESSIVE_SAFETY_GATES):
            self.assertFalse(gates.pot_odds_exclude_uncallable)

    def test_the_overhang_is_what_the_forensics_recorded(self) -> None:
        """FAILS ON THE UNFIXED CODE -- 1,777 is the measured figure."""

        table = self._shove_table()
        self.assertEqual(_policy()._uncallable_chips(table, 1_170), 1_777)

    def test_the_price_moves_from_the_recorded_wrong_one_to_the_right_one(
        self,
    ) -> None:
        """FAILS ON THE UNFIXED CODE. 0.2823 was served; 0.4943 is real."""

        table = self._shove_table()
        allowed = table["allowedActions"]
        off = _policy(dataclasses.replace(AGGRESSIVE_SAFETY_GATES))
        on = _policy(
            dataclasses.replace(
                AGGRESSIVE_SAFETY_GATES, pot_odds_exclude_uncallable=True
            )
        )
        self.assertAlmostEqual(off._pot_odds(table, allowed), 0.2823, places=4)
        self.assertAlmostEqual(on._pot_odds(table, allowed), 0.4943, places=4)

    def test_it_is_inert_when_nobody_overshoves(self) -> None:
        """An opponent hero can fully match leaves the price untouched."""

        table = self._shove_table(opp_bet=900, pot=920)
        allowed = table["allowedActions"]
        self.assertEqual(_policy()._uncallable_chips(table, 1_170), 0)
        off = _policy()
        on = _policy(
            dataclasses.replace(
                AGGRESSIVE_SAFETY_GATES, pot_odds_exclude_uncallable=True
            )
        )
        self.assertEqual(off._pot_odds(table, allowed), on._pot_odds(table, allowed))

    def test_the_correction_can_only_raise_the_price(self) -> None:
        """Monotone by construction: it only ever subtracts from the pot."""

        off = _policy()
        on = _policy(
            dataclasses.replace(
                AGGRESSIVE_SAFETY_GATES, pot_odds_exclude_uncallable=True
            )
        )
        for opp_bet in (0, 100, 900, 1_181, 1_182, 2_958, 50_000):
            table = self._shove_table(opp_bet=opp_bet, pot=opp_bet + 16)
            allowed = table["allowedActions"]
            with self.subTest(opp_bet=opp_bet):
                self.assertGreaterEqual(
                    on._pot_odds(table, allowed), off._pot_odds(table, allowed)
                )

    def test_an_unreadable_snapshot_reproduces_the_old_arithmetic(self) -> None:
        """The correction must never turn a bad snapshot into an exception."""

        table = self._shove_table()
        table["seats"][1]["currentBetChips"] = None
        allowed = table["allowedActions"]
        on = _policy(
            dataclasses.replace(
                AGGRESSIVE_SAFETY_GATES, pot_odds_exclude_uncallable=True
            )
        )
        self.assertEqual(on._uncallable_chips(table, 1_170), 0)
        self.assertAlmostEqual(on._pot_odds(table, allowed), 0.2823, places=4)

    def test_a_free_check_is_still_priced_at_zero(self) -> None:
        table = self._shove_table()
        allowed = dict(table["allowedActions"], callChips=0)
        on = _policy(
            dataclasses.replace(
                AGGRESSIVE_SAFETY_GATES, pot_odds_exclude_uncallable=True
            )
        )
        self.assertEqual(on._pot_odds(table, allowed), 0.0)


class UnpricedRangeConditioningTests(unittest.TestCase):
    """Whether observed aggression survives hero acting first.

    ``_call_top_fraction`` returns 1.0 -- uniformly random -- the moment
    ``callChips <= 0``, before it consults aggression at all. Hero was
    the small blind in the 2026-08-26 hand, so hero opened the flop and
    the turn, and both leads were priced against a random hand while the
    opponent had already raised. Deterministic corpus-wide: 1,369 of
    4,795 logged decisions have no price and every one carries width
    exactly 1.0.

    The dial ships OFF. Feature 138 is 1.0 whenever feature 134 is 0 in
    all 1,730,110 training rows, so enabling it serves the network a
    joint it has never seen.
    """

    @staticmethod
    def _lead_table(*, raises=2, pot=442, to_call=0):
        events = [
            {
                "type": "PlayerAction",
                "summary": {"seatNumber": 1, "action": "raise", "amount": 100 * n},
            }
            for n in range(1, raises + 1)
        ]
        return {
            "id": "t",
            "tableId": "lead-test",
            "street": "turn",
            "potChips": pot,
            "currentBet": to_call,
            "boardCards": ["6h", "8c", "Td", "Qd"],
            "smallBlindChips": 1,
            "bigBlindChips": 2,
            "selfSeatNumber": 4,
            "seats": [
                {
                    "seatNumber": 4,
                    "status": "Active",
                    "stackChips": 1_930,
                    "currentBetChips": 0,
                    "holeCards": ["Tc", "As"],
                },
                {
                    "seatNumber": 1,
                    "status": "Active",
                    "stackChips": 823,
                    "currentBetChips": 0,
                },
            ],
            "allowedActions": {
                "canFold": True,
                "canCheck": to_call == 0,
                "canCall": to_call > 0,
                "canBet": to_call == 0,
                "canRaise": False,
                "canAllIn": True,
                "callAmount": to_call,
                "callChips": to_call,
                "callToAmount": to_call,
                "minBet": 2,
                "minRaiseTo": None,
                "betRange": {"min": 2, "max": 823},
                "raiseRange": None,
                "allInToAmount": 823,
                "availableActions": ["all-in", "bet", "check", "fold"],
                "amountSemantics": "toAmount",
                "reasoningRequired": False,
            },
            "recentEvents": events,
        }

    @staticmethod
    def _on():
        return dataclasses.replace(
            AGGRESSIVE_SAFETY_GATES, condition_range_without_price=True
        )

    def test_the_dial_ships_off(self) -> None:
        for gates in (DEFAULT_SAFETY_GATES, AGGRESSIVE_SAFETY_GATES):
            self.assertFalse(gates.condition_range_without_price)

    def test_the_shipped_guard_discards_every_observed_raise(self) -> None:
        table = self._lead_table(raises=4)
        allowed = table["allowedActions"]
        self.assertEqual(_policy()._call_top_fraction(table, allowed), 1.0)

    def test_the_dial_keeps_the_escalation_exponent(self) -> None:
        """FAILS ON THE UNFIXED CODE. 0.75 * 0.8**(raises-1)."""

        for raises, expected in ((1, 0.75), (2, 0.60), (3, 0.48), (4, 0.384)):
            table = self._lead_table(raises=raises)
            allowed = table["allowedActions"]
            with self.subTest(raises=raises):
                self.assertAlmostEqual(
                    _policy(self._on())._call_top_fraction(table, allowed),
                    expected,
                    places=6,
                )

    def test_no_aggression_still_means_no_conditioning(self) -> None:
        """The second guard is untouched: nothing observed, nothing assumed."""

        table = self._lead_table(raises=0)
        allowed = table["allowedActions"]
        self.assertEqual(_policy(self._on())._call_top_fraction(table, allowed), 1.0)

    def test_the_size_multiplier_self_neutralises_without_a_price(self) -> None:
        """Why the guard was defensible, and why removing it is bounded.

        ``bet_fraction`` is ``to_call / max(pot - to_call, 1)``, which is
        exactly 0 with nothing to call, so neither size branch can fire.
        The conditioned value is therefore the pure escalation term.
        """

        table = self._lead_table(raises=3, pot=10_000)
        allowed = table["allowedActions"]
        cheap = self._lead_table(raises=3, pot=12)
        # Pot size cannot matter when there is no price.
        self.assertAlmostEqual(
            _policy(self._on())._call_top_fraction(table, allowed),
            _policy(self._on())._call_top_fraction(cheap, cheap["allowedActions"]),
            places=9,
        )

    def test_the_dial_is_inert_whenever_there_is_a_price(self) -> None:
        """It selects the unpriced branch only; priced rows must not move."""

        for raises in (0, 1, 3):
            table = self._lead_table(raises=raises, to_call=150)
            allowed = table["allowedActions"]
            with self.subTest(raises=raises):
                self.assertEqual(
                    _policy()._call_top_fraction(table, allowed),
                    _policy(self._on())._call_top_fraction(table, allowed),
                )

    def test_the_conditioned_value_respects_the_hard_clamp(self) -> None:
        table = self._lead_table(raises=20)
        allowed = table["allowedActions"]
        self.assertGreaterEqual(
            _policy(self._on())._call_top_fraction(table, allowed), 0.20
        )


if __name__ == "__main__":
    unittest.main()
