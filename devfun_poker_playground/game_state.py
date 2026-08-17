"""Validate Arena game state and extract policy features.

Everything in this module is torch-free so deployment builds can reuse the
same validation and feature pipeline without the training-time dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from devfun_poker_playground.policy_features import FEATURE_NAMES

_STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_AGGRESSIVE_ACTIONS = ("bet", "raise", "all-in")


class ArenaSnapshotError(ValueError):
    """Raised when an untrusted Arena snapshot is incomplete or inconsistent."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArenaSnapshotError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ArenaSnapshotError(f"{name} must be an array")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArenaSnapshotError(f"{name} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ArenaSnapshotError(f"{name} must be finite")
    integer = int(value)
    if integer != value or integer < minimum:
        raise ArenaSnapshotError(f"{name} must be an integer >= {minimum}")
    return integer


def _normalize_card(value: object) -> str:
    if not isinstance(value, str):
        raise ArenaSnapshotError("cards must be strings")
    token = value.strip()
    if len(token) == 3 and token[:2] == "10":
        token = f"T{token[2]}"
    if len(token) != 2:
        raise ArenaSnapshotError(f"invalid card {value!r}")
    card = f"{token[0].upper()}{token[1].lower()}"
    if card[0] not in "23456789TJQKA" or card[1] not in "cdhs":
        raise ArenaSnapshotError(f"invalid card {value!r}")
    return card


def _cards(values: object, name: str, expected: int | None = None) -> tuple[str, ...]:
    cards = tuple(_normalize_card(value) for value in _sequence(values, name))
    if expected is not None and len(cards) != expected:
        raise ArenaSnapshotError(f"{name} must contain exactly {expected} cards")
    if len(set(cards)) != len(cards):
        raise ArenaSnapshotError(f"{name} contains duplicate cards")
    return cards


def _hero_and_seats(
    table: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    self_seat_number = _integer(
        table.get("selfSeatNumber"), "selfSeatNumber", minimum=1
    )
    seats = [
        _mapping(value, f"seats[{index}]")
        for index, value in enumerate(_sequence(table.get("seats"), "seats"))
    ]
    hero = next(
        (
            seat
            for seat in seats
            if seat.get("seatNumber") is not None
            and _integer(seat.get("seatNumber"), "seatNumber", minimum=1)
            == self_seat_number
        ),
        None,
    )
    if hero is None:
        raise ArenaSnapshotError("selfSeatNumber does not match any seat")
    return hero, seats


def _active_seats(
    seats: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        seat
        for seat in seats
        if str(seat.get("status") or "").casefold() not in {"folded", "settled"}
    ]


def active_opponent_count(table: Mapping[str, Any]) -> int:
    """Number of opponents still eligible to win the current hand."""

    hero, seats = _hero_and_seats(table)
    return sum(seat is not hero for seat in _active_seats(seats))


def effective_stack_chips(table: Mapping[str, Any]) -> int:
    """Maximum chips hero can put at risk against one active opponent."""

    hero, seats = _hero_and_seats(table)
    stack = _integer(hero.get("stackChips"), "hero stackChips")
    opponent_stacks = [
        _integer(seat.get("stackChips"), "opponent stackChips")
        for seat in _active_seats(seats)
        if seat is not hero
    ]
    return min(stack, max(opponent_stacks, default=stack))


# Board cards still to be dealt on each street. The river is 0 by
# definition: paying there buys a showdown, not a card.
_REVEALS_REMAINING = {"preflop": 3, "flop": 2, "turn": 1, "river": 0}


def card_reveal_expense(table: Mapping[str, Any], price_chips: int) -> float:
    """What it costs to reach the next card, as a share of what can be lost.

    Chips committed before a reveal are chips staked on cards nobody has
    seen. The engine already knows how far it is from the river (the risk
    temperature's ``distance_from_river`` factor) but nowhere prices that
    distance in chips, so a decision cannot tell a cheap look at the turn
    from staking the stack on it.

    Measured on live play 2026-08-15: the deployment's worst hand paid
    100% of the effective stack on the turn holding ``QsJc`` into a
    three-club ``KcJd3cQc`` board, then folded the river for -3,768 chips.
    Nine such committed-then-folded hands cost -5,858 in total, every one a
    complete loss of the commitment.

    Public data only -- price, stacks, and the street -- so this is equally
    usable as a gate input and as a learned feature. Returns ``[0, 1]``.
    """

    at_risk = max(1, effective_stack_chips(table))
    share = min(1.0, max(0, price_chips) / at_risk)
    remaining = _REVEALS_REMAINING.get(str(table.get("street") or "").casefold(), 0)
    return share * (remaining / 3.0)


def _blind_seats(table: Mapping[str, Any]) -> tuple[int | None, int | None]:
    small_blind = _integer(table.get("smallBlindChips"), "smallBlindChips", minimum=1)
    big_blind = _integer(table.get("bigBlindChips"), "bigBlindChips", minimum=1)
    small_seat: int | None = None
    big_seat: int | None = None
    for raw_event in _sequence(table.get("recentEvents") or [], "recentEvents"):
        event = _mapping(raw_event, "recentEvent")
        if event.get("type") != "BlindPosted":
            continue
        summary_value = event.get("summary")
        if summary_value is None:
            continue
        summary = _mapping(summary_value, "recentEvent.summary")
        if summary.get("seatNumber") is None or summary.get("amount") is None:
            continue
        seat_number = _integer(summary["seatNumber"], "blind seat", minimum=1)
        amount = _integer(summary["amount"], "blind amount", minimum=0)
        if amount == small_blind:
            small_seat = seat_number
        if amount == big_blind:
            big_seat = seat_number
    return small_seat, big_seat


def _position(
    table: Mapping[str, Any], seats: list[Mapping[str, Any]], hero: Mapping[str, Any]
) -> float:
    seat_numbers = sorted(
        _integer(seat["seatNumber"], "seatNumber", minimum=1)
        for seat in seats
        if seat.get("seatNumber") is not None
    )
    hero_number = _integer(hero.get("seatNumber"), "hero seatNumber", minimum=1)
    if hero_number not in seat_numbers:
        raise ArenaSnapshotError("hero seat is not in the seated player list")
    if len(seat_numbers) <= 1:
        return 0.0

    small_seat, big_seat = _blind_seats(table)
    if big_seat in seat_numbers:
        if len(seat_numbers) == 2:
            other = next(number for number in seat_numbers if number != big_seat)
            ordered = [big_seat, other]
        elif small_seat in seat_numbers and small_seat != big_seat:
            pivot = seat_numbers.index(big_seat)
            clockwise = seat_numbers[pivot + 1 :] + seat_numbers[:pivot]
            ordered = [small_seat, big_seat, *(n for n in clockwise if n != small_seat)]
        else:
            ordered = seat_numbers
    else:
        ordered = seat_numbers
    return ordered.index(hero_number) / (len(ordered) - 1)


def _log_bb(amount: float, big_blind: int) -> float:
    return math.log1p(max(0.0, amount) / big_blind)


def _aggression_count(table: Mapping[str, Any], street: str) -> int:
    count = 0
    for raw_event in _sequence(table.get("recentEvents") or [], "recentEvents"):
        event = _mapping(raw_event, "recentEvent")
        if str(event.get("street") or "").casefold() != street:
            continue
        summary_value = event.get("summary")
        if summary_value is None:
            continue
        action = str(_mapping(summary_value, "recentEvent.summary").get("action") or "")
        if action.casefold() in _AGGRESSIVE_ACTIONS:
            count += 1
    return count


def features_from_table(
    table: Mapping[str, Any],
    *,
    position: float | None = None,
) -> tuple[float, ...]:
    """Convert one fresh Arena table snapshot to the training feature contract."""

    hero, seats = _hero_and_seats(table)
    allowed = _mapping(table.get("allowedActions"), "allowedActions")
    available = {
        str(value)
        for value in _sequence(allowed.get("availableActions"), "availableActions")
    }
    if not available:
        raise ArenaSnapshotError("no actions are available")

    street = str(table.get("street") or "").casefold()
    if street not in _STREET_INDEX:
        raise ArenaSnapshotError(f"unsupported decision street {table.get('street')!r}")
    hole_cards = _cards(hero.get("holeCards"), "hero holeCards", expected=2)
    board_cards = _cards(table.get("boardCards"), "boardCards")
    if len(board_cards) > 5 or len({*hole_cards, *board_cards}) != len(
        hole_cards
    ) + len(board_cards):
        raise ArenaSnapshotError(
            "hole and board cards must be unique and board length <= 5"
        )

    big_blind = _integer(table.get("bigBlindChips"), "bigBlindChips", minimum=1)
    pot = _integer(table.get("potChips"), "potChips")
    stack = _integer(hero.get("stackChips"), "hero stackChips")
    contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
    current_bet = _integer(table.get("currentBet"), "currentBet")
    to_call = _integer(allowed.get("callChips", 0), "allowedActions.callChips")
    min_raise_to = allowed.get("minRaiseTo")
    if min_raise_to is None:
        min_raise_to = allowed.get("minBet") or 0
    min_raise_to = _integer(min_raise_to, "minimum aggressive amount")

    active_seats = _active_seats(seats)
    effective_stack = effective_stack_chips(table)

    legal_fold = bool(allowed.get("canFold")) and "fold" in available
    legal_check_call = (bool(allowed.get("canCheck")) and "check" in available) or (
        bool(allowed.get("canCall")) and "call" in available
    )
    legal_aggress = any(action in available for action in _AGGRESSIVE_ACTIONS)
    if not (legal_fold or legal_check_call or legal_aggress):
        raise ArenaSnapshotError("allowedActions contains no supported legal action")

    values = {name: 0.0 for name in FEATURE_NAMES}
    for card in hole_cards:
        values[f"hole_{card}"] = 1.0
    for card in board_cards:
        values[f"board_{card}"] = 1.0
    for index in range(4):
        values[f"street_{('preflop', 'flop', 'turn', 'river')[index]}"] = float(
            index == _STREET_INDEX[street]
        )
    if position is None:
        position = _position(table, seats, hero)
    if not math.isfinite(position) or not 0.0 <= position <= 1.0:
        raise ArenaSnapshotError("position must be between 0 and 1")

    values.update(
        {
            "player_count": float(len(seats)),
            "active_player_count": float(len(active_seats)),
            "position": position,
            "log_pot_bb": _log_bb(pot, big_blind),
            "log_stack_bb": _log_bb(stack, big_blind),
            "log_effective_stack_bb": _log_bb(effective_stack, big_blind),
            "log_to_call_bb": _log_bb(to_call, big_blind),
            "log_street_contribution_bb": _log_bb(contribution, big_blind),
            "log_current_bet_bb": _log_bb(current_bet, big_blind),
            "log_min_raise_to_bb": _log_bb(min_raise_to, big_blind),
            "pot_odds": 0.0 if to_call <= 0 else to_call / max(pot + to_call, 1),
            "spr": 0.0 if pot <= 0 else effective_stack / pot,
            "raises_current_street": float(_aggression_count(table, street)),
            "legal_fold": float(legal_fold),
            "legal_check_call": float(legal_check_call),
            "legal_aggress": float(legal_aggress),
            "hole_known_fraction": 1.0,
        }
    )
    features = tuple(values[name] for name in FEATURE_NAMES)
    if not all(math.isfinite(value) for value in features):
        raise ArenaSnapshotError("snapshot produced non-finite model features")
    return features


__all__ = [
    "active_opponent_count",
    "ArenaSnapshotError",
    "effective_stack_chips",
    "features_from_table",
]
