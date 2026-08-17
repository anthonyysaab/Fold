"""Phase-A dataset generator: label semantics, invariants, and the build.

Three hand-built miniature event streams exercise the label definitions
against outcomes known by construction (house rule — the instrument is
measured before its results are believed):

- ``mini-ft`` — a raise that folds out the whole table. The one aggress
  decision must read ``fold_through`` 1.0 (the task's mandated
  impossible-by-construction case), sized into the large branch, with no
  continuing opponent to bucket; the folds that follow must read the
  remaining opponent's AA as octile 7.
- ``mini-sd`` — a river bet with the mortal nuts, called by a full house.
  Exact river arithmetic: the bettor's ``equity_called`` is exactly 1.0,
  the caller's exactly 0.0, and the caller sees the royal flush in
  octile 7.
- ``mini-dd`` — a half-pot turn bet by quads against a drawing-dead
  caller. Exercises the exact one-card enumeration (1.0 vs 0.0 with no
  sampling noise) and the small-branch size assignment.

``_showdown_equity`` additionally gets direct known-answer checks (a
three-way board-play tie is exactly 1/3; AA over KK on a dry turn is
exactly 42/44), one real replay from the s13 archive is parsed end to
end, and the full build path (gzip dataset + sidecar) is run twice over
the miniatures to prove byte-identical determinism.
"""

from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path

from devfun_poker_playground import schema3
from devfun_poker_playground.strength_metric import strength_percentile
from tools.build_phase_a_dataset import (
    _LABEL_NAMES,
    _showdown_equity,
    build_dataset,
    replay_rows,
)
from tools.collect_foreign_play_data import _read_json, _unwrap_rpc

import random

_REAL_REPLAY = (
    Path("foreign play data")
    / "20260812T082057Z_poker-playground_s13_top15"
    / "raw"
    / "tables"
    / "cmspqme3qt6ct14cawl3trl69.json"
)

# Small trial counts keep the suite fast; every exact assertion below is
# independent of both numbers by construction (enumeration paths and
# masks/buckets do not sample).
_FAST = {"potential_trials": 32, "equity_trials": 64}


def _allowed(
    *,
    available: list[str],
    call_chips: int = 0,
    call_to: int | None = None,
    min_raise_to: int | None = None,
    raise_range: tuple[int, int] | None = None,
    bet_range: tuple[int, int] | None = None,
    all_in_to: int | None = None,
) -> dict:
    return {
        "canFold": "fold" in available,
        "canCheck": "check" in available,
        "canCall": "call" in available,
        "canBet": "bet" in available,
        "canRaise": "raise" in available,
        "canAllIn": "all-in" in available,
        "callChips": call_chips,
        "callAmount": call_chips,
        "callToAmount": call_to,
        "minBet": bet_range[0] if bet_range else None,
        "minRaiseTo": min_raise_to,
        "betRange": (
            {"min": bet_range[0], "max": bet_range[1]} if bet_range else None
        ),
        "raiseRange": (
            {"min": raise_range[0], "max": raise_range[1]}
            if raise_range
            else None
        ),
        "allInToAmount": all_in_to,
        "maxCommit": all_in_to,
        "amountSemantics": "toAmount",
        "availableActions": list(available),
    }


def _seat(
    number: int,
    agent: str,
    name: str,
    hole: list[str],
    stack: int,
    bet: int,
    committed: int,
    status: str = "Active",
) -> dict:
    return {
        "seatNumber": number,
        "agentId": agent,
        "agentName": name,
        "holeCards": list(hole),
        "stackChips": stack,
        "currentBetChips": bet,
        "totalCommittedChips": committed,
        "status": status,
    }


def _snapshot(table: dict, seats: list[dict], board: list[str], pot: int) -> dict:
    return {
        **{k: v for k, v in table.items() if k != "seats"},
        "seats": seats,
        "boardCards": list(board),
        "potChips": pot,
    }


def _action(
    seq: int,
    street: str,
    agent: str,
    payload: dict,
    snapshot: dict,
    *,
    event_type: str = "ActionTaken",
) -> dict:
    return {
        "type": event_type,
        "street": street,
        "sequence": seq,
        "agentId": agent,
        "payload": payload,
        "snapshot": snapshot,
    }


def _table(table_id: str, seats: list[dict], winners: list[str]) -> dict:
    return {
        "id": table_id,
        "tableId": table_id,
        "status": "Completed",
        "street": "Preflop",
        "potChips": 0,
        "currentBet": 0,
        "minRaiseTo": 2,
        "boardCards": [],
        "smallBlindChips": 1,
        "bigBlindChips": 2,
        "buyInChips": 2,
        "tableNumber": 1,
        "competitionId": "mini-competition",
        "startedAt": 0,
        "endedAt": 1,
        "seats": seats,
        "winners": [{"agentId": agent} for agent in winners],
    }


def _fold_through_replay() -> dict:
    """4 seats preflop: a timeout fold, a raise to 6, and two live folds."""

    holes = {1: ["As", "Ad"], 2: ["7c", "2d"], 3: ["8h", "3s"], 4: ["Kc", "Kd"]}
    agents = {1: "a1", 2: "b2", 3: "c3", 4: "t4"}
    names = {1: "Alice", 2: "Bob", 3: "Carol", 4: "Tia"}

    def seats(states: dict[int, tuple[int, int, int, str]]) -> list[dict]:
        return [
            _seat(n, agents[n], names[n], holes[n], *states[n])
            for n in sorted(states)
        ]

    table = _table(
        "mini-ft",
        seats(
            {
                1: (194, 0, 6, "Active"),
                2: (199, 0, 1, "Folded"),
                3: (198, 0, 2, "Folded"),
                4: (200, 0, 0, "Folded"),
            }
        ),
        winners=["a1"],
    )
    events = [
        {"type": "TableStarted", "sequence": 1, "payload": {"dealerSeatNumber": 1}},
        {
            "type": "HoleCardsDealt",
            "sequence": 2,
            "payload": {
                "seats": [
                    {
                        "seatNumber": n,
                        "agentId": agents[n],
                        "agentName": names[n],
                        "holeCards": holes[n],
                    }
                    for n in sorted(holes)
                ]
            },
        },
        {
            "type": "BlindPosted",
            "sequence": 3,
            "street": "Preflop",
            "payload": {"blind": "small", "amount": 1, "seatNumber": 2},
        },
        {
            "type": "BlindPosted",
            "sequence": 4,
            "street": "Preflop",
            "payload": {"blind": "big", "amount": 2, "seatNumber": 3},
        },
        _action(
            5,
            "Preflop",
            agents[4],
            {
                "pot": 3,
                "action": "fold",
                "amount": None,
                "toAmount": None,
                "callAmount": 2,
                "agentName": names[4],
                "seatNumber": 4,
                "stackBefore": 200,
                "allowedActions": _allowed(
                    available=["fold", "call", "raise", "all-in"],
                    call_chips=2,
                    call_to=2,
                    min_raise_to=4,
                    raise_range=(4, 200),
                    all_in_to=200,
                ),
                "currentBetBefore": 2,
                "minRaiseToBefore": 4,
                "actorCurrentBetBefore": 0,
                "dealerSeatNumber": 1,
            },
            _snapshot(
                _table("mini-ft", [], ["a1"]),
                seats(
                    {
                        1: (200, 0, 0, "Active"),
                        2: (199, 1, 1, "Active"),
                        3: (198, 2, 2, "Active"),
                        4: (200, 0, 0, "Folded"),
                    }
                ),
                [],
                3,
            ),
            event_type="TimeoutAction",
        ),
        _action(
            6,
            "Preflop",
            agents[1],
            {
                "pot": 3,
                "action": "raise",
                "amount": None,
                "toAmount": 6,
                "callAmount": 2,
                "agentName": names[1],
                "seatNumber": 1,
                "stackBefore": 200,
                "allowedActions": _allowed(
                    available=["fold", "call", "raise", "all-in"],
                    call_chips=2,
                    call_to=2,
                    min_raise_to=4,
                    raise_range=(4, 200),
                    all_in_to=200,
                ),
                "currentBetBefore": 2,
                "minRaiseToBefore": 4,
                "actorCurrentBetBefore": 0,
                "dealerSeatNumber": 1,
            },
            _snapshot(
                _table("mini-ft", [], ["a1"]),
                seats(
                    {
                        1: (194, 6, 6, "Active"),
                        2: (199, 1, 1, "Active"),
                        3: (198, 2, 2, "Active"),
                        4: (200, 0, 0, "Folded"),
                    }
                ),
                [],
                9,
            ),
        ),
        _action(
            7,
            "Preflop",
            agents[2],
            {
                "pot": 9,
                "action": "fold",
                "amount": None,
                "toAmount": None,
                "callAmount": 5,
                "agentName": names[2],
                "seatNumber": 2,
                "stackBefore": 199,
                "allowedActions": _allowed(
                    available=["fold", "call", "raise", "all-in"],
                    call_chips=5,
                    call_to=6,
                    min_raise_to=10,
                    raise_range=(10, 199),
                    all_in_to=199,
                ),
                "currentBetBefore": 6,
                "minRaiseToBefore": 10,
                "actorCurrentBetBefore": 1,
                "dealerSeatNumber": 1,
            },
            _snapshot(
                _table("mini-ft", [], ["a1"]),
                seats(
                    {
                        1: (194, 6, 6, "Active"),
                        2: (199, 1, 1, "Folded"),
                        3: (198, 2, 2, "Active"),
                        4: (200, 0, 0, "Folded"),
                    }
                ),
                [],
                9,
            ),
        ),
        _action(
            8,
            "Preflop",
            agents[3],
            {
                "pot": 9,
                "action": "fold",
                "amount": None,
                "toAmount": None,
                "callAmount": 4,
                "agentName": names[3],
                "seatNumber": 3,
                "stackBefore": 198,
                "allowedActions": _allowed(
                    available=["fold", "call", "raise", "all-in"],
                    call_chips=4,
                    call_to=6,
                    min_raise_to=10,
                    raise_range=(10, 198),
                    all_in_to=198,
                ),
                "currentBetBefore": 6,
                "minRaiseToBefore": 10,
                "actorCurrentBetBefore": 2,
                "dealerSeatNumber": 1,
            },
            _snapshot(
                _table("mini-ft", [], ["a1"]),
                seats(
                    {
                        1: (194, 6, 6, "Active"),
                        2: (199, 1, 1, "Folded"),
                        3: (198, 2, 2, "Folded"),
                        4: (200, 0, 0, "Folded"),
                    }
                ),
                [],
                9,
            ),
        ),
        {"type": "Payout", "sequence": 9, "payload": {"payouts": []}},
    ]
    return {"table": table, "events": events}


def _two_seat_replay(
    table_id: str,
    holes: dict[int, list[str]],
    flop: list[str],
    turn: str,
    river: str,
    *,
    turn_bet_to: int | None = None,
    river_bet_to: int | None = None,
    winners: list[str] = (),
) -> dict:
    """Heads-up limped pot, optional turn/river bet-and-call, showdown.

    Seat 1 posts the small blind and acts second postflop; seat 2 posts
    the big blind and opens each postflop street. Exactly one of
    ``turn_bet_to`` / ``river_bet_to`` may be set: seat 2 bets that
    to-amount and seat 1 calls; every other action is a check.
    """

    agents = {1: "d1", 2: "e2"}
    names = {1: "Dana", 2: "Eve"}

    def seats(states: dict[int, tuple[int, int, int]]) -> list[dict]:
        return [
            _seat(n, agents[n], names[n], holes[n], *states[n])
            for n in sorted(states)
        ]

    def snap(states: dict[int, tuple[int, int, int]], board: list[str], pot: int) -> dict:
        return _snapshot(_table(table_id, [], list(winners)), seats(states), board, pot)

    check_allowed = _allowed(
        available=["check", "bet", "all-in"],
        bet_range=(2, 96),
        all_in_to=96,
    )

    def check(seq: int, street: str, seat: int, board: list[str], pot: int, states) -> dict:
        return _action(
            seq,
            street,
            agents[seat],
            {
                "pot": pot,
                "action": "check",
                "amount": None,
                "toAmount": None,
                "callAmount": 0,
                "agentName": names[seat],
                "seatNumber": seat,
                "stackBefore": states[seat][0],
                "allowedActions": check_allowed,
                "currentBetBefore": 0,
                "minRaiseToBefore": 2,
                "actorCurrentBetBefore": 0,
                "dealerSeatNumber": 1,
            },
            snap(states, board, pot),
        )

    def street_dealt(seq: int, street: str, cards: list[str], board: list[str]) -> dict:
        return {
            "type": "StreetDealt",
            "sequence": seq,
            "street": street,
            "payload": {"cards": cards, "street": street, "boardCards": board},
        }

    events: list[dict] = [
        {
            "type": "HoleCardsDealt",
            "sequence": 1,
            "payload": {
                "seats": [
                    {
                        "seatNumber": n,
                        "agentId": agents[n],
                        "agentName": names[n],
                        "holeCards": holes[n],
                    }
                    for n in sorted(holes)
                ]
            },
        },
        {
            "type": "BlindPosted",
            "sequence": 2,
            "street": "Preflop",
            "payload": {"blind": "small", "amount": 1, "seatNumber": 1},
        },
        {
            "type": "BlindPosted",
            "sequence": 3,
            "street": "Preflop",
            "payload": {"blind": "big", "amount": 2, "seatNumber": 2},
        },
        _action(
            4,
            "Preflop",
            agents[1],
            {
                "pot": 3,
                "action": "call",
                "amount": 1,
                "toAmount": 2,
                "callAmount": 1,
                "agentName": names[1],
                "seatNumber": 1,
                "stackBefore": 99,
                "allowedActions": _allowed(
                    available=["fold", "call", "raise", "all-in"],
                    call_chips=1,
                    call_to=2,
                    min_raise_to=4,
                    raise_range=(4, 100),
                    all_in_to=100,
                ),
                "currentBetBefore": 2,
                "minRaiseToBefore": 4,
                "actorCurrentBetBefore": 1,
                "dealerSeatNumber": 1,
            },
            snap({1: (98, 2, 2), 2: (98, 2, 2)}, [], 4),
        ),
        _action(
            5,
            "Preflop",
            agents[2],
            {
                "pot": 4,
                "action": "check",
                "amount": None,
                "toAmount": None,
                "callAmount": 0,
                "agentName": names[2],
                "seatNumber": 2,
                "stackBefore": 98,
                "allowedActions": _allowed(
                    available=["check", "raise", "all-in"],
                    min_raise_to=4,
                    raise_range=(4, 100),
                    all_in_to=100,
                ),
                "currentBetBefore": 2,
                "minRaiseToBefore": 4,
                "actorCurrentBetBefore": 2,
                "dealerSeatNumber": 1,
            },
            snap({1: (98, 2, 2), 2: (98, 2, 2)}, [], 4),
        ),
        street_dealt(6, "Flop", flop, flop),
        check(7, "Flop", 2, flop, 4, {1: (98, 0, 2), 2: (98, 0, 2)}),
        check(8, "Flop", 1, flop, 4, {1: (98, 0, 2), 2: (98, 0, 2)}),
        street_dealt(9, "Turn", [turn], flop + [turn]),
    ]
    seq = 10
    turn_board = flop + [turn]
    states_flat = {1: (98, 0, 2), 2: (98, 0, 2)}
    if turn_bet_to is None:
        events.append(check(seq, "Turn", 2, turn_board, 4, states_flat))
        events.append(check(seq + 1, "Turn", 1, turn_board, 4, states_flat))
        pot_after_turn = 4
    else:
        bet = turn_bet_to
        events.append(
            _action(
                seq,
                "Turn",
                agents[2],
                {
                    "pot": 4,
                    "action": "bet",
                    "amount": None,
                    "toAmount": bet,
                    "callAmount": 0,
                    "agentName": names[2],
                    "seatNumber": 2,
                    "stackBefore": 98,
                    "allowedActions": check_allowed,
                    "currentBetBefore": 0,
                    "minRaiseToBefore": 2,
                    "actorCurrentBetBefore": 0,
                    "dealerSeatNumber": 1,
                },
                snap({1: (98, 0, 2), 2: (98 - bet, bet, 2 + bet)}, turn_board, 4 + bet),
            )
        )
        events.append(
            _action(
                seq + 1,
                "Turn",
                agents[1],
                {
                    "pot": 4 + bet,
                    "action": "call",
                    "amount": bet,
                    "toAmount": bet,
                    "callAmount": bet,
                    "agentName": names[1],
                    "seatNumber": 1,
                    "stackBefore": 98,
                    "allowedActions": _allowed(
                        available=["fold", "call", "raise", "all-in"],
                        call_chips=bet,
                        call_to=bet,
                        min_raise_to=2 * bet,
                        raise_range=(2 * bet, 98),
                        all_in_to=98,
                    ),
                    "currentBetBefore": bet,
                    "minRaiseToBefore": 2 * bet,
                    "actorCurrentBetBefore": 0,
                    "dealerSeatNumber": 1,
                },
                snap(
                    {1: (98 - bet, bet, 2 + bet), 2: (98 - bet, bet, 2 + bet)},
                    turn_board,
                    4 + 2 * bet,
                ),
            )
        )
        pot_after_turn = 4 + 2 * bet
    seq += 2
    river_board = turn_board + [river]
    events.append(street_dealt(seq, "River", [river], river_board))
    seq += 1
    stack = 98 - (turn_bet_to or 0)
    committed = 2 + (turn_bet_to or 0)
    states_river = {1: (stack, 0, committed), 2: (stack, 0, committed)}
    if river_bet_to is None:
        events.append(check(seq, "River", 2, river_board, pot_after_turn, states_river))
        events.append(
            check(seq + 1, "River", 1, river_board, pot_after_turn, states_river)
        )
    else:
        bet = river_bet_to
        events.append(
            _action(
                seq,
                "River",
                agents[2],
                {
                    "pot": pot_after_turn,
                    "action": "bet",
                    "amount": None,
                    "toAmount": bet,
                    "callAmount": 0,
                    "agentName": names[2],
                    "seatNumber": 2,
                    "stackBefore": stack,
                    "allowedActions": check_allowed,
                    "currentBetBefore": 0,
                    "minRaiseToBefore": 2,
                    "actorCurrentBetBefore": 0,
                    "dealerSeatNumber": 1,
                },
                snap(
                    {1: states_river[1], 2: (stack - bet, bet, committed + bet)},
                    river_board,
                    pot_after_turn + bet,
                ),
            )
        )
        events.append(
            _action(
                seq + 1,
                "River",
                agents[1],
                {
                    "pot": pot_after_turn + bet,
                    "action": "call",
                    "amount": bet,
                    "toAmount": bet,
                    "callAmount": bet,
                    "agentName": names[1],
                    "seatNumber": 1,
                    "stackBefore": stack,
                    "allowedActions": _allowed(
                        available=["fold", "call", "raise", "all-in"],
                        call_chips=bet,
                        call_to=bet,
                        min_raise_to=2 * bet,
                        raise_range=(2 * bet, stack),
                        all_in_to=stack,
                    ),
                    "currentBetBefore": bet,
                    "minRaiseToBefore": 2 * bet,
                    "actorCurrentBetBefore": 0,
                    "dealerSeatNumber": 1,
                },
                snap(
                    {
                        1: (stack - bet, bet, committed + bet),
                        2: (stack - bet, bet, committed + bet),
                    },
                    river_board,
                    pot_after_turn + 2 * bet,
                ),
            )
        )
    seq += 2
    events.append({"type": "Showdown", "sequence": seq, "payload": {"seats": []}})
    events.append({"type": "Payout", "sequence": seq + 1, "payload": {"payouts": []}})
    table = _table(
        table_id,
        seats({1: (90, 0, 0), 2: (90, 0, 0)}),
        winners=list(winners),
    )
    return {"table": table, "events": events}


def _rows_by_seq(rows: list[dict]) -> dict[int, dict]:
    return {row["sequence"]: row for row in rows}


def _assert_row_contract(test: unittest.TestCase, row: dict) -> None:
    schema3.require_vector(row["features"])
    for name in _LABEL_NAMES:
        test.assertIn(row["masks"][name], (0, 1))
    test.assertGreaterEqual(row["labels"]["equity_called"], 0.0)
    test.assertLessEqual(row["labels"]["equity_called"], 1.0)
    test.assertIn(row["labels"]["fold_through_small"], (0.0, 1.0))
    test.assertIn(row["labels"]["fold_through_large"], (0.0, 1.0))
    test.assertIn(row["labels"]["range_bucket"], range(schema3.BELIEF_BUCKETS))
    test.assertIn(row["street"], ("preflop", "flop", "turn", "river"))


class FoldThroughHandTests(unittest.TestCase):
    """mini-ft: a preflop raise folds out the table."""

    rows: list[dict]

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.stats = replay_rows(_fold_through_replay(), **_FAST)

    def test_row_count_and_timeout_handling(self) -> None:
        # 3 ActionTaken decisions; the TimeoutAction fold yields no row.
        self.assertEqual(len(self.rows), 3)
        self.assertEqual(self.stats["timeout_actions"], 1)
        self.assertEqual(self.stats["skipped_decisions"], 0)

    def test_everyone_folded_to_one_bet_reads_fold_through_one(self) -> None:
        row = _rows_by_seq(self.rows)[6]
        self.assertEqual(row["labels"]["fold_through_small"], 1.0)
        self.assertEqual(row["labels"]["fold_through_large"], 1.0)
        # Realized wager 6 sits above the E6 midpoint (targets 4.5 and 7
        # on pot 3, call 2, effective stack 199): the large branch mask.
        self.assertEqual(row["masks"]["fold_through_large"], 1)
        self.assertEqual(row["masks"]["fold_through_small"], 0)
        # No opponent continued: range and equity are undefined.
        self.assertEqual(row["masks"]["range_bucket"], 0)
        self.assertEqual(row["masks"]["equity_called"], 0)

    def test_fold_decisions_never_define_fold_through(self) -> None:
        for seq in (7, 8):
            row = _rows_by_seq(self.rows)[seq]
            self.assertEqual(row["masks"]["fold_through_small"], 0)
            self.assertEqual(row["masks"]["fold_through_large"], 0)

    def test_folds_see_the_remaining_aces_in_the_top_octile(self) -> None:
        # After each fold the only continuing opponent holds AA — the top
        # preflop class, octile 7 by the canonical preflop table.
        for seq in (7, 8):
            row = _rows_by_seq(self.rows)[seq]
            self.assertEqual(row["masks"]["range_bucket"], 1)
            self.assertEqual(row["labels"]["range_bucket"], 7)
            self.assertEqual(row["masks"]["equity_called"], 1)
            self.assertLess(row["labels"]["equity_called"], 0.35)

    def test_row_contract_and_determinism(self) -> None:
        for row in self.rows:
            _assert_row_contract(self, row)
        rerun, _ = replay_rows(_fold_through_replay(), **_FAST)
        self.assertEqual(self.rows, rerun)


class ShowdownHandTests(unittest.TestCase):
    """mini-sd: a river bet with the royal flush, called by a full house."""

    rows: list[dict]

    @classmethod
    def setUpClass(cls) -> None:
        cls.replay = _two_seat_replay(
            "mini-sd",
            {1: ["As", "2s"], 2: ["Jh", "Th"]},
            flop=["Ah", "Kh", "Qh"],
            turn="2c",
            river="2d",
            river_bet_to=10,
            winners=["e2"],
        )
        cls.rows, cls.stats = replay_rows(cls.replay, **_FAST)

    def test_row_count(self) -> None:
        self.assertEqual(len(self.rows), 8)
        self.assertEqual(self.stats["skipped_decisions"], 0)

    def test_river_bet_labels_are_exact(self) -> None:
        rows = _rows_by_seq(self.rows)
        bet = next(
            row
            for row in self.rows
            if row["street"] == "river" and row["seat"] == 2
        )
        # Called, not folded through.
        self.assertEqual(bet["labels"]["fold_through_small"], 0.0)
        self.assertEqual(
            bet["masks"]["fold_through_small"]
            + bet["masks"]["fold_through_large"],
            1,
        )
        # The royal flush against the caller's actual full house on the
        # complete board: exactly 1.0.
        self.assertEqual(bet["masks"]["equity_called"], 1)
        self.assertEqual(bet["labels"]["equity_called"], 1.0)
        # The caller's boat, bucketed on the canonical metric.
        expected = strength_percentile(
            ["As", "2s"], ["Ah", "Kh", "Qh", "2c", "2d"]
        )
        self.assertEqual(
            bet["labels"]["range_bucket"],
            min(7, int(expected * schema3.BELIEF_BUCKETS)),
        )

    def test_river_call_is_drawing_dead_against_the_royal(self) -> None:
        call = next(
            row
            for row in self.rows
            if row["street"] == "river" and row["seat"] == 1
        )
        self.assertEqual(call["masks"]["equity_called"], 1)
        self.assertEqual(call["labels"]["equity_called"], 0.0)
        self.assertEqual(call["labels"]["range_bucket"], 7)
        self.assertEqual(call["masks"]["fold_through_small"], 0)
        self.assertEqual(call["masks"]["fold_through_large"], 0)

    def test_every_row_survives_the_contract(self) -> None:
        for row in self.rows:
            _assert_row_contract(self, row)


class TurnEnumerationHandTests(unittest.TestCase):
    """mini-dd: quads bet the turn small; the caller is drawing dead."""

    rows: list[dict]

    @classmethod
    def setUpClass(cls) -> None:
        cls.replay = _two_seat_replay(
            "mini-dd",
            {1: ["7h", "8h"], 2: ["2s", "9c"]},
            flop=["2c", "2d", "2h"],
            turn="3s",
            river="Kd",
            turn_bet_to=2,
            winners=["e2"],
        )
        cls.rows, cls.stats = replay_rows(cls.replay, **_FAST)

    def test_half_pot_turn_bet_takes_the_small_branch(self) -> None:
        bet = next(
            row
            for row in self.rows
            if row["street"] == "turn" and row["seat"] == 2
        )
        # Pot 4, call 0: small target 2, large target 4, midpoint 3;
        # the realized wager of 2 is the small branch.
        self.assertEqual(bet["masks"]["fold_through_small"], 1)
        self.assertEqual(bet["masks"]["fold_through_large"], 0)
        self.assertEqual(bet["labels"]["fold_through_small"], 0.0)

    def test_turn_equities_are_exact_by_enumeration(self) -> None:
        bet = next(
            row
            for row in self.rows
            if row["street"] == "turn" and row["seat"] == 2
        )
        call = next(
            row
            for row in self.rows
            if row["street"] == "turn" and row["seat"] == 1
        )
        # Quad deuces cannot be outdrawn by 7h8h on any river; exact
        # enumeration must return 1.0 and 0.0 with no sampling noise.
        self.assertEqual(bet["labels"]["equity_called"], 1.0)
        self.assertEqual(call["labels"]["equity_called"], 0.0)
        # The quads read above 0.99 on the canonical metric (its own
        # contract), so the caller sees octile 7.
        self.assertEqual(call["labels"]["range_bucket"], 7)

    def test_every_row_survives_the_contract(self) -> None:
        for row in self.rows:
            _assert_row_contract(self, row)


class ShowdownEquityTests(unittest.TestCase):
    """Known-answer checks for the equity computation itself."""

    def test_three_way_board_play_is_exactly_one_third(self) -> None:
        equity = _showdown_equity(
            ["Kd", "8h"],
            [["Qs", "9d"], ["Jc", "9h"]],
            ["2c", "3c", "4d", "5h", "6s"],
            [["Kd", "8h"], ["Qs", "9d"], ["Jc", "9h"]],
            random.Random(0),
            equity_trials=10,
        )
        self.assertAlmostEqual(equity, 1.0 / 3.0, places=12)

    def test_turn_enumeration_counts_the_two_king_outs(self) -> None:
        # AA vs KK on 2c7d9hQs: 44 unseen rivers, KK wins on exactly the
        # two remaining kings — 42/44, exactly.
        equity = _showdown_equity(
            ["As", "Ad"],
            [["Ks", "Kd"]],
            ["2c", "7d", "9h", "Qs"],
            [["As", "Ad"], ["Ks", "Kd"]],
            random.Random(0),
            equity_trials=10,
        )
        self.assertEqual(equity, 42.0 / 44.0)


class RealReplayTests(unittest.TestCase):
    """One real s13 replay end to end (4 decisions, one timeout)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _REAL_REPLAY.is_file():
            raise unittest.SkipTest(f"real replay not on disk: {_REAL_REPLAY}")
        cls.replay = _unwrap_rpc(_read_json(_REAL_REPLAY))
        cls.rows, cls.stats = replay_rows(cls.replay, **_FAST)

    def test_every_action_taken_yields_a_row(self) -> None:
        expected = sum(
            1
            for event in self.replay["events"]
            if event.get("type") == "ActionTaken"
        )
        self.assertEqual(len(self.rows), expected)
        self.assertEqual(len(self.rows), 4)
        self.assertEqual(self.stats["timeout_actions"], 1)
        self.assertEqual(self.stats["skipped_decisions"], 0)

    def test_rows_survive_the_contract(self) -> None:
        for row in self.rows:
            _assert_row_contract(self, row)
            self.assertTrue(row["table_id"])
            self.assertTrue(row["actor_agent"])
            # All four real decisions are folds: fold_through undefined.
            self.assertEqual(row["masks"]["fold_through_small"], 0)
            self.assertEqual(row["masks"]["fold_through_large"], 0)

    def test_determinism(self) -> None:
        rerun, _ = replay_rows(self.replay, **_FAST)
        self.assertEqual(self.rows, rerun)


class BuildDatasetTests(unittest.TestCase):
    """The full build over the three miniatures: files, sidecar, bytes."""

    def _write_collection(self, root: Path) -> None:
        tables = root / "raw" / "tables"
        tables.mkdir(parents=True)
        replays = {
            "mini-ft": _fold_through_replay(),
            "mini-sd": _two_seat_replay(
                "mini-sd",
                {1: ["As", "2s"], 2: ["Jh", "Th"]},
                flop=["Ah", "Kh", "Qh"],
                turn="2c",
                river="2d",
                river_bet_to=10,
                winners=["e2"],
            ),
            "mini-dd": _two_seat_replay(
                "mini-dd",
                {1: ["7h", "8h"], 2: ["2s", "9c"]},
                flop=["2c", "2d", "2h"],
                turn="3s",
                river="Kd",
                turn_bet_to=2,
                winners=["e2"],
            ),
        }
        for name, replay in replays.items():
            (tables / f"{name}.json").write_text(
                json.dumps({"result": {"data": {"json": replay}}}),
                encoding="utf-8",
            )

    def test_build_writes_dataset_and_sidecar_deterministically(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "mini-collection"
            self._write_collection(root)
            first = Path(scratch) / "out-a" / "phase-a-dataset.jsonl.gz"
            second = Path(scratch) / "out-b" / "phase-a-dataset.jsonl.gz"
            sidecar_a = build_dataset([root], first, workers=1, **_FAST)
            sidecar_b = build_dataset([root], second, workers=1, **_FAST)
            self.assertEqual(sidecar_a, sidecar_b)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            with gzip.open(first, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(len(rows), 19)  # 3 + 8 + 8
            # Sorted by (table_id, sequence) for worker-independence.
            keys = [(row["table_id"], row["sequence"]) for row in rows]
            self.assertEqual(keys, sorted(keys))
            for row in rows:
                _assert_row_contract(self, row)
                self.assertEqual(len(row["features"]), schema3.INPUT_SIZE_V8)

            counts = sidecar_a["counts"]
            self.assertEqual(counts["rows"], 19)
            self.assertEqual(counts["tables_with_rows"], 3)
            coverage = counts["label_coverage"]
            self.assertEqual(coverage["fold_through_small"], 1)
            self.assertEqual(coverage["fold_through_large"], 2)
            self.assertEqual(coverage["range_bucket"], 18)
            self.assertEqual(coverage["equity_called"], 18)
            per_street = sidecar_a["per_street"]
            self.assertEqual(per_street["preflop"]["rows"], 7)
            self.assertEqual(per_street["flop"]["rows"], 4)
            self.assertEqual(per_street["turn"]["rows"], 4)
            self.assertEqual(per_street["river"]["rows"], 4)
            # Masks recomputed from the rows must match the sidecar.
            for name in _LABEL_NAMES:
                self.assertEqual(
                    coverage[name], sum(row["masks"][name] for row in rows)
                )


if __name__ == "__main__":
    unittest.main()
