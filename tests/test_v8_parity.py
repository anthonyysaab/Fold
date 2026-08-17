"""Parity pins: the v8 extractor's mirrors against the engine originals (F5).

``feature_extract_v8`` promises it is an assembler, not a second
implementation — ``_lead_position_unit`` mirrors
``decision_engine.DecisionEngine._lead_position`` (with the engine's own
0.5 fallback when the reading raises ``ValueError``), and
``hero_aggression_count`` mirrors ``DecisionEngine._aggressive_events``
whole-hand (``street=None``), the schema-2 "kept" semantics. These tests
hold the two sides against each other on shared snapshots, so a drift in
either breaks a test instead of silently forking the definition.

Importing private functions is deliberate: parity between privates is
exactly what is being pinned. ``_lead_position`` never touches ``self``,
so it is exercised unbound with a ``None`` receiver — if the engine ever
grows a ``self`` dependency there, this file fails loudly and the mirror
must be revisited anyway.
"""

from __future__ import annotations

import unittest

from devfun_poker_playground.decision_engine import DecisionEngine
from devfun_poker_playground.feature_extract_v8 import (
    _lead_position_unit,
    extract_features_v8,
)
from devfun_poker_playground.game_state import _hero_and_seats
from devfun_poker_playground.schema3 import FEATURE_NAMES_V8

# Reduced from the extractor's 400 default: parity tests pin wiring, not
# Monte-Carlo precision, and the compared quantities ignore the trials.
_TRIALS = 16


def _seat(
    number: int,
    stack: int,
    *,
    status: str = "Active",
    bet: int = 0,
    committed: int | None = None,
    hole: tuple[str, str] | None = None,
) -> dict:
    seat: dict = {
        "seatNumber": number,
        "status": status,
        "stackChips": stack,
        "currentBetChips": bet,
        "holeCards": list(hole) if hole else None,
    }
    if committed is not None:
        seat["totalCommittedChips"] = committed
    return seat


def _event(street: str, seat: int, action: str, **amounts: int) -> dict:
    summary: dict = {"seatNumber": seat, "action": action}
    summary.update(amounts)
    return {"type": "ActionTaken", "street": street, "summary": summary}


def _blind(seat: int, amount: int) -> dict:
    return {
        "type": "BlindPosted",
        "street": "preflop",
        "summary": {"seatNumber": seat, "amount": amount},
    }


def _table(
    *,
    hero_seat: int,
    seats: list[dict],
    street: str,
    board: tuple[str, ...],
    pot: int,
    to_call: int,
    events: list[dict],
    available: tuple[str, ...],
    bet_range: tuple[int, int] | None = None,
    raise_range: tuple[int, int] | None = None,
) -> dict:
    hero = next(seat for seat in seats if seat["seatNumber"] == hero_seat)
    return {
        "id": "v8-parity-test",
        "tableId": "v8-parity-test",
        "street": street,
        "potChips": pot,
        "currentBet": hero["currentBetChips"] + to_call,
        "boardCards": list(board),
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": hero_seat,
        "seats": seats,
        "allowedActions": {
            "canFold": "fold" in available,
            "canCheck": "check" in available,
            "canCall": "call" in available,
            "canBet": "bet" in available,
            "canRaise": "raise" in available,
            "canAllIn": "all-in" in available,
            "callAmount": to_call,
            "callChips": to_call,
            "callToAmount": (
                hero["currentBetChips"] + to_call if to_call else None
            ),
            "betRange": (
                {"min": bet_range[0], "max": bet_range[1]} if bet_range else None
            ),
            "raiseRange": (
                {"min": raise_range[0], "max": raise_range[1]}
                if raise_range
                else None
            ),
            "allInToAmount": None,
            "availableActions": list(available),
            "amountSemantics": "toAmount",
        },
        "recentEvents": events,
    }


def _dense_turn_table() -> dict:
    """Hero bet the flop, faces a turn bet: whole-hand hero aggression 1."""

    return _table(
        hero_seat=1,
        seats=[
            _seat(1, 2600, hole=("Ah", "Kd"), committed=500),
            _seat(2, 1800, bet=200, committed=700),
            _seat(3, 900, committed=100),
        ],
        street="turn",
        board=("2c", "7d", "9h", "Qs"),
        pot=1300,
        to_call=200,
        events=[
            _blind(2, 50),
            _blind(3, 100),
            _event("preflop", 1, "call", amount=100),
            _event("preflop", 2, "call", amount=50),
            _event("preflop", 3, "check"),
            _event("flop", 1, "bet", toAmount=200),
            _event("flop", 2, "call", amount=200),
            _event("flop", 3, "fold"),
            _event("turn", 2, "bet", toAmount=200),
        ],
        available=("fold", "call", "raise"),
        raise_range=(400, 900),
    )


def _sparse_preflop_table() -> dict:
    """Sparse seats {1, 4, 5}, hero 4, a raising war back on hero."""

    return _table(
        hero_seat=4,
        seats=[
            _seat(1, 3100, bet=800, committed=800),
            _seat(4, 1400, bet=900, committed=900, hole=("Qs", "Qd")),
            _seat(5, 700, bet=2000, committed=2000),
        ],
        street="preflop",
        board=(),
        pot=3700,
        to_call=1100,
        events=[
            _blind(5, 50),
            _blind(1, 100),
            _event("preflop", 5, "raise", toAmount=300),
            _event("preflop", 4, "raise", toAmount=900),
            _event("preflop", 1, "call", amount=800),
            _event("preflop", 5, "all-in", toAmount=2000),
        ],
        available=("fold", "call"),
    )


def _folded_field_table() -> dict:
    """Every opponent folded: the lead reading raises, the unit degrades."""

    return _table(
        hero_seat=1,
        seats=[
            _seat(1, 2000, hole=("Ah", "Kd"), committed=300),
            _seat(2, 1800, status="Folded", committed=400),
            _seat(3, 500, status="Folded"),
        ],
        street="turn",
        board=("2c", "7d", "9h", "Qs"),
        pot=700,
        to_call=0,
        events=[
            _blind(2, 50),
            _blind(3, 100),
            _event("preflop", 1, "raise", toAmount=300),
            _event("preflop", 2, "call", amount=250),
            _event("flop", 2, "fold"),
        ],
        available=("check", "bet"),
        bet_range=(100, 1500),
    )


_SNAPSHOTS = {
    "dense_turn": _dense_turn_table,
    "sparse_preflop": _sparse_preflop_table,
    "folded_field": _folded_field_table,
}


def _by_name(vector: tuple[float, ...]) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES_V8, vector))


class LeadPositionParityTest(unittest.TestCase):
    def test_unit_matches_the_engine_on_every_snapshot(self) -> None:
        for name, build in _SNAPSHOTS.items():
            table = build()
            engine_lead = DecisionEngine._lead_position(None, table)
            expected = (
                0.5 if engine_lead is None else (engine_lead + 100.0) / 200.0
            )
            hero, seats = _hero_and_seats(table)
            with self.subTest(snapshot=name):
                self.assertEqual(
                    _lead_position_unit(table, hero, seats), expected
                )
                values = _by_name(
                    extract_features_v8(table, potential_trials=_TRIALS)
                )
                self.assertEqual(values["lead_position_unit"], expected)

    def test_the_raising_snapshot_actually_raises(self) -> None:
        # Guard against a vacuous probe: the fallback parity above proves
        # nothing unless one snapshot genuinely takes the ValueError path.
        self.assertIsNone(
            DecisionEngine._lead_position(None, _folded_field_table())
        )
        self.assertIsNotNone(
            DecisionEngine._lead_position(None, _dense_turn_table())
        )


class HeroAggressionParityTest(unittest.TestCase):
    def test_count_matches_the_engine_whole_hand(self) -> None:
        for name, build in _SNAPSHOTS.items():
            table = build()
            expected = float(DecisionEngine._aggressive_events(table, hero=True))
            values = _by_name(
                extract_features_v8(table, potential_trials=_TRIALS)
            )
            with self.subTest(snapshot=name):
                self.assertEqual(values["hero_aggression_count"], expected)

    def test_the_whole_hand_scope_is_actually_exercised(self) -> None:
        # Non-vacuous guard: the dense snapshot's only hero aggression is
        # on the flop while the decision street is the turn — a
        # current-street count would read 0 and the parity above would
        # never notice a scope regression on all-zero snapshots.
        self.assertEqual(
            DecisionEngine._aggressive_events(_dense_turn_table(), hero=True), 1
        )


if __name__ == "__main__":
    unittest.main()
