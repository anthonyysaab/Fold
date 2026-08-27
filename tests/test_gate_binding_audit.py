"""Checks for the live-journal gate binding audit.

The audit restates the gates as arithmetic over stored journal fields
instead of driving the engine, which buys it the ability to price a
counterfactual on real hands and costs it the engine's own guarantee of
correctness. These tests buy that guarantee back: every constant and
every formula it restates is pinned against the module it copied from,
so a drift in either is a test failure rather than a silent skew in a
report someone will quote.
"""

from __future__ import annotations

import unittest

from devfun_poker_playground import game_state
from devfun_poker_playground.decision_engine import SafetyGates
from tools.gate_binding_audit import (
    REVEALS_REMAINING,
    Decision,
    call_gate_refuses,
    call_gate_triggers,
    cap_applies,
    risk_cap,
    stage_instrument,
)


def _decision(**overrides) -> Decision:
    values = dict(
        table_id="t1",
        street="flop",
        big_blind=100,
        hero_stack=9_143,
        effective_stack=2_207,
        contribution=0,
        to_call=2_207,
        raise_min=4_414,
        raise_max=9_143,
        equity=0.53,
        action="call",
        amount_to=None,
        hyper=False,
    )
    values.update(overrides)
    return Decision(**values)


def _gates(**overrides) -> dict:
    gates = SafetyGates()
    values = {
        "risk_cap_stack_fraction": gates.risk_cap_stack_fraction,
        "near_nut_floor": gates.near_nut_floor,
        "call_stack_gates": [list(pair) for pair in gates.call_stack_gates],
        "reveal_expense_equity_slope": gates.reveal_expense_equity_slope,
    }
    values.update(overrides)
    return values


class PinnedToTheEngineTests(unittest.TestCase):
    """Anything the audit restates must equal what it restated."""

    def test_reveals_remaining_matches_game_state(self) -> None:
        self.assertEqual(REVEALS_REMAINING, game_state._REVEALS_REMAINING)

    def test_reveal_expense_matches_card_reveal_expense(self) -> None:
        """The audit's formula against the engine's, over real snapshots."""

        for street, board in (
            ("preflop", []),
            ("flop", ["Kc", "Jd", "3c"]),
            ("turn", ["Kc", "Jd", "3c", "Qc"]),
            ("river", ["Kc", "Jd", "3c", "Qc", "2h"]),
        ):
            for price in (0, 1, 700, 2_207, 99_999):
                table = {
                    "street": street,
                    "boardCards": board,
                    "selfSeatNumber": 1,
                    "smallBlindChips": 50,
                    "bigBlindChips": 100,
                    "seats": [
                        {
                            "seatNumber": 1,
                            "status": "Active",
                            "stackChips": 9_143,
                            "currentBetChips": 0,
                        },
                        {
                            "seatNumber": 2,
                            "status": "Active",
                            "stackChips": 2_207,
                            "currentBetChips": 0,
                        },
                    ],
                }
                decision = _decision(street=street, effective_stack=2_207)
                with self.subTest(street=street, price=price):
                    self.assertAlmostEqual(
                        decision.reveal_expense(price),
                        game_state.card_reveal_expense(table, price),
                        places=12,
                    )

    def test_risk_cap_matches_the_engine_formula(self) -> None:
        gates = _gates()
        decision = _decision(contribution=300, hero_stack=11_842, effective_stack=1_133)
        # contribution + max(bb, round(fraction * denominator))
        self.assertEqual(
            risk_cap(decision, gates, effective=False),
            300 + max(100, round(0.455 * 11_842)),
        )
        self.assertEqual(
            risk_cap(decision, gates, effective=True),
            300 + max(100, round(0.455 * 1_133)),
        )

    def test_the_cap_releases_at_the_near_nut_floor(self) -> None:
        gates = _gates()
        self.assertTrue(cap_applies(_decision(equity=0.10), gates))
        self.assertTrue(cap_applies(_decision(equity=None), gates))
        self.assertFalse(cap_applies(_decision(equity=0.90), gates))

    def test_call_gate_triggers_key_on_the_named_denominator(self) -> None:
        gates = _gates(call_stack_gates=[[0.65, 0.584]])
        decision = _decision(hero_stack=9_143, effective_stack=2_207, to_call=2_207)
        # 2_207 >= 0.65 * 2_207, but 2_207 < 0.65 * 9_143.
        self.assertEqual(call_gate_triggers(decision, gates, effective=True), [0.65])
        self.assertEqual(call_gate_triggers(decision, gates, effective=False), [])


class InstrumentTests(unittest.TestCase):
    """The audit's own known-answer checks must actually be able to fail."""

    def test_a_clean_journal_passes_every_blocking_check(self) -> None:
        decisions = [_decision(), _decision(hero_stack=1_000, effective_stack=1_000)]
        report = stage_instrument(decisions, _gates())
        self.assertTrue(report["all_passed"])

    def test_an_effective_stack_over_heros_purse_fails(self) -> None:
        """Impossible by construction: effective is min(hero, opponents)."""

        planted = _decision(hero_stack=1_000, effective_stack=5_000)
        report = stage_instrument([planted], _gates())
        self.assertFalse(report["all_passed"])
        self.assertEqual(report["effective_never_exceeds_hero"]["verdict"], "FAIL")

    def test_equal_denominators_can_never_differ(self) -> None:
        equal = [
            _decision(hero_stack=n, effective_stack=n, to_call=n // 2)
            for n in (500, 1_000, 9_143)
        ]
        report = stage_instrument(equal, _gates())
        check = report["equal_denominators_give_equal_verdicts"]
        self.assertEqual(check["records"], 3)
        self.assertEqual(check["differences"], 0)
        self.assertEqual(check["verdict"], "PASS")

    def test_engine_parity_flags_a_bet_over_the_hero_purse_cap(self) -> None:
        """A bet no hero-purse-capped engine could have produced.

        This is the check that would catch the audit mis-modelling the
        cap -- and equally, a supervisor that restarted under the changed
        gates mid-journal.
        """

        over = _decision(
            action="raise",
            amount_to=10_000,
            hero_stack=1_000,
            effective_stack=1_000,
            equity=0.10,
        )
        report = stage_instrument([over], _gates())
        parity = report["engine_parity_on_the_hero_purse_cap"]
        self.assertEqual(parity["over_the_hero_purse_cap"], 1)
        self.assertEqual(parity["verdict"], "REVIEW")
        # REVIEW is interpretive, not a wiring fault, so it must not block.
        self.assertTrue(report["all_passed"])

    def test_river_reveal_expense_is_zero_by_definition(self) -> None:
        # Paying on the river buys a showdown, not a card.
        self.assertEqual(_decision(street="river").reveal_expense(2_207), 0.0)


class ArrivalIsNotRefusalTests(unittest.TestCase):
    """The distinction that a published figure once got wrong.

    The first version of this audit counted stack-fraction *arrivals* and
    labelled the column "changes the verdict". It overstated the
    call-gate edit by 72% on the live journal and wrongly attributed the
    deployment-ending -5,000 hand to it. The engine refuses only when
    ``equity < required``; the trigger merely decides whether the floor
    is consulted at all.
    """

    def test_a_gate_can_be_reached_without_refusing(self) -> None:
        gates = _gates(call_stack_gates=[[0.65, 0.584]], reveal_expense_equity_slope=0.0)
        # Price clears the trigger, equity clears the floor.
        strong = _decision(to_call=2_207, effective_stack=2_207, equity=0.90)
        self.assertEqual(call_gate_triggers(strong, gates, effective=True), [0.65])
        self.assertFalse(call_gate_refuses(strong, gates, effective=True))

    def test_a_gate_reached_with_weak_equity_refuses(self) -> None:
        gates = _gates(call_stack_gates=[[0.65, 0.584]], reveal_expense_equity_slope=0.0)
        weak = _decision(to_call=2_207, effective_stack=2_207, equity=0.10)
        self.assertEqual(call_gate_triggers(weak, gates, effective=True), [0.65])
        self.assertTrue(call_gate_refuses(weak, gates, effective=True))

    def test_refusal_requires_the_trigger_too(self) -> None:
        """Weak equity alone is not a refusal -- the price must reach a gate."""

        gates = _gates(call_stack_gates=[[0.65, 0.584]], reveal_expense_equity_slope=0.0)
        cheap = _decision(to_call=10, effective_stack=2_207, equity=0.10)
        self.assertEqual(call_gate_triggers(cheap, gates, effective=True), [])
        self.assertFalse(call_gate_refuses(cheap, gates, effective=True))

    def test_the_reveal_slope_can_turn_a_call_into_a_refusal(self) -> None:
        """Equity between the bare floor and the floor plus penalty."""

        priced = _gates(
            call_stack_gates=[[0.65, 0.584]], reveal_expense_equity_slope=0.12
        )
        neutral = _gates(
            call_stack_gates=[[0.65, 0.584]], reveal_expense_equity_slope=0.0
        )
        # Whole effective stack on the flop -> expense 2/3, penalty 0.08.
        row = _decision(
            street="flop", to_call=2_207, effective_stack=2_207, equity=0.62
        )
        self.assertAlmostEqual(row.reveal_expense(row.to_call), 2 / 3, places=9)
        self.assertFalse(call_gate_refuses(row, neutral, effective=True))
        self.assertTrue(call_gate_refuses(row, priced, effective=True))

    def test_a_missing_equity_cannot_refuse(self) -> None:
        gates = _gates(call_stack_gates=[[0.65, 0.584]])
        blind = _decision(to_call=2_207, effective_stack=2_207, equity=None)
        self.assertTrue(call_gate_triggers(blind, gates, effective=True))
        self.assertFalse(call_gate_refuses(blind, gates, effective=True))

    def test_refusals_are_a_subset_of_arrivals_on_every_row(self) -> None:
        """Structural invariant: you cannot be refused by a gate you never reach."""

        gates = _gates()
        for to_call in (1, 50, 700, 2_207, 9_143):
            for equity in (None, 0.05, 0.3, 0.584, 0.7, 0.99):
                for effective in (True, False):
                    row = _decision(to_call=to_call, equity=equity)
                    with self.subTest(to_call=to_call, equity=equity, eff=effective):
                        if call_gate_refuses(row, gates, effective=effective):
                            self.assertTrue(
                                call_gate_triggers(row, gates, effective=effective)
                            )


class AllInDenominatorCollapseTests(unittest.TestCase):
    """The shipped defect this audit surfaced (PENDING_EDITS, 2026-08-26).

    ``effective_stack_chips`` counts chips *behind* and ``_active_seats``
    excludes only folded/settled seats, so an all-in opponent leaves the
    denominator at 0. The engine clamps that to 1, which trips every
    stack gate at any price and simultaneously maxes the reveal expense.
    """

    def test_an_effective_stack_of_one_trips_any_positive_price(self) -> None:
        gates = _gates(call_stack_gates=[[0.78, 0.626], [0.455, 0.584]])
        collapsed = _decision(effective_stack=1, to_call=84, equity=0.69)
        self.assertEqual(
            call_gate_triggers(collapsed, gates, effective=True), [0.78, 0.455]
        )

    def test_the_collapse_also_maxes_the_reveal_expense(self) -> None:
        collapsed = _decision(street="flop", effective_stack=1, to_call=84)
        # min(1, 84/1) * (2/3) -- the share saturates at 1.0.
        self.assertAlmostEqual(collapsed.reveal_expense(84), 2 / 3, places=9)

    def test_the_hero_purse_denominator_does_not_collapse(self) -> None:
        # The incumbent's live configuration, named explicitly: the slope
        # defaulted to 0.12 when this defect was found and was reverted to
        # 0.0 on 2026-08-26, so inheriting the default here would quietly
        # stop exercising the case.
        gates = _gates(
            call_stack_gates=[[0.78, 0.626], [0.455, 0.584]],
            reveal_expense_equity_slope=0.12,
        )
        collapsed = _decision(
            effective_stack=1, hero_stack=9_143, to_call=84, equity=0.69
        )
        self.assertEqual(call_gate_triggers(collapsed, gates, effective=False), [])
        self.assertTrue(call_gate_refuses(collapsed, gates, effective=True))
        self.assertFalse(call_gate_refuses(collapsed, gates, effective=False))


if __name__ == "__main__":
    unittest.main()
