"""Schema-3 action-history encoder: telemetry records to the history block.

``encode_action_history`` consumes the ``recent_actions`` list carried by a
decision record — the trimmed ``recentEvents`` copy that
``training_telemetry`` writes (``street`` casefolded, ``seat_number`` int or
None, ``action`` casefolded, ``amount`` an int only when the event carried
one) — and produces the ``schema3.HISTORY_FEATURE_NAMES`` block: six
right-aligned slots per street plus a per-street overflow flag.

Encoding constraints, and why:

- Right alignment (the most recent action is always the last slot) keeps
  each slot's meaning fixed no matter how many actions a street saw;
  padding lives in the low slots as all-zero rows, indistinguishable from
  "street not reached", which is the intended prior.
- Only records whose action maps to a family via
  ``training_telemetry.action_family`` occupy a slot; deals, blinds and
  show events are table bookkeeping, not decisions, and must not displace
  real actions from the slot window. Records for streets outside
  ``schema3.HISTORY_STREETS`` are ignored for the same reason.
- A street that saw more than ``schema3.HISTORY_SLOTS`` family actions
  keeps the last ones and raises its overflow flag — truncation is
  recorded, never silent (schema3's contract).
- ``size`` is ``log1p(amount / big_blind)`` — the documented deviation
  from Deep CFR's pot-fraction sizes (see the schema3 docstring): the
  telemetry record does not carry the pot at the moment of a past action.
  Which wager lands in ``amount`` is the record producer's decision — the
  live path (``feature_extract_v8._hand_events``) normalises it to the
  raise-to amount for aggressive actions and the call wager for calls,
  the price each raw event actually carries. A missing or malformed
  amount reads 0.0, matching the telemetry writer's skip-don't-fail
  posture.
- ``actor`` is the actor's clockwise seat distance from hero, scaled to
  [0, 1]; hero's own actions read 0.0. An unknown seat reads the
  midpoint, maximally non-committal between hero and the furthest seat.
  With ``seat_numbers`` the distance is ordinal in the sorted seat set,
  so sparse seat numbers cannot alias (seats {1, 4, 5} with hero 4 read
  1.0 / 0.5 / 0.0, never hero's own 0.0 for seat 1); without it the
  documented fallback is modular arithmetic on raw seat numbers, which
  is correct only when seats are dense and consecutive.

Layout positions are resolved against ``schema3.HISTORY_FEATURE_NAMES`` at
import time; any drift between this encoder's naming and the schema fails
the import instead of silently misplacing values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from engine import schema3
from engine.training_telemetry import action_family

__all__ = ["encode_action_history"]

# Ablatable: midpoint of the [0, 1] actor unit for a record whose seat is
# unknown — equidistant from hero (0.0) and the furthest seat (1.0), so the
# network cannot mistake "unknown" for either extreme.
_UNKNOWN_ACTOR = 0.5


def _positions() -> tuple[dict[tuple[str, int, str], int], dict[str, int]]:
    """Resolve every slot channel and overflow flag to its block position.

    Names are rebuilt with the same format ``schema3._history_names`` uses
    and looked up in ``schema3.HISTORY_FEATURE_NAMES``; a renamed or
    reordered schema raises here (KeyError or the count check) at import
    time, which is the failure mode we want — loud, not misaligned.
    """

    by_name = {
        name: position
        for position, name in enumerate(schema3.HISTORY_FEATURE_NAMES)
    }
    slot_positions: dict[tuple[str, int, str], int] = {}
    overflow_positions: dict[str, int] = {}
    for street in schema3.HISTORY_STREETS:
        for slot in range(schema3.HISTORY_SLOTS):
            for channel in schema3.HISTORY_CHANNELS:
                slot_positions[street, slot, channel] = by_name[
                    f"hist_{street}{slot}_{channel}"
                ]
        overflow_positions[street] = by_name[f"hist_{street}_overflow"]
    if len(slot_positions) + len(overflow_positions) != len(by_name):
        raise schema3.Schema3Error(
            "schema-3 history block contains features this encoder "
            "does not produce"
        )
    return slot_positions, overflow_positions


_SLOT_POSITION, _OVERFLOW_POSITION = _positions()


def _bet_size(amount: object, big_blind_chips: int) -> float:
    """``log1p`` of the wager in big blinds; 0.0 when absent or malformed.

    Negative or non-int amounts cannot come from ``_recent_actions`` but
    are absorbed as 0.0 anyway: a corrupt journal entry must degrade one
    channel, not raise mid-encode (and ``log1p`` of a deep negative would).
    """

    if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
        return 0.0
    return math.log1p(amount / big_blind_chips)


def _seat_ordinals(seat_numbers: Sequence[int]) -> dict[int, int]:
    """Ordinal index per seat number over ``sorted(set(seat_numbers))``.

    Malformed entries (non-int, bool) are dropped rather than raising:
    a snapshot seat without a usable number simply cannot be resolved,
    and its actions read the unknown midpoint downstream.
    """

    ordered = sorted(
        {
            seat
            for seat in seat_numbers
            if isinstance(seat, int) and not isinstance(seat, bool)
        }
    )
    return {seat: index for index, seat in enumerate(ordered)}


def _actor_unit(
    seat_number: object,
    hero_seat_number: int,
    player_count: int,
    seat_ordinals: Mapping[int, int] | None = None,
) -> float:
    """Clockwise seat distance from hero scaled to [0, 1]; hero reads 0.0.

    With ``seat_ordinals`` (from ``_seat_ordinals``) the distance is
    ordinal in the sorted seat set — sparse seat numbers cannot alias —
    and a seat absent from the set (including a hero somehow absent) reads
    the unknown midpoint. Without it, the documented fallback is modular
    arithmetic on raw seat numbers over ``player_count``, correct only for
    dense consecutive seats.
    """

    if not isinstance(seat_number, int) or isinstance(seat_number, bool):
        return _UNKNOWN_ACTOR
    if seat_ordinals is not None:
        actor_index = seat_ordinals.get(seat_number)
        hero_index = seat_ordinals.get(hero_seat_number)
        if actor_index is None or hero_index is None:
            return _UNKNOWN_ACTOR
        count = len(seat_ordinals)
        distance = (actor_index - hero_index) % count
        return distance / max(1, count - 1)
    distance = (seat_number - hero_seat_number) % player_count
    # max(1, ...) only guards the degenerate one-seat table, where the
    # sole possible distance is already 0.
    return distance / max(1, player_count - 1)


def encode_action_history(
    recent_actions: Sequence[Mapping[str, object]],
    *,
    hero_seat_number: int,
    player_count: int,
    big_blind_chips: int,
    seat_numbers: Sequence[int] | None = None,
) -> tuple[float, ...]:
    """Encode telemetry action records as the schema-3 history block.

    Returns exactly ``len(schema3.HISTORY_FEATURE_NAMES)`` floats in
    schema-3 order. Records are consumed in sequence order (oldest first),
    per street: the last ``schema3.HISTORY_SLOTS`` family actions fill the
    slots right-aligned, and any earlier ones set the street's overflow
    flag. Records on unknown streets, or whose action has no family, are
    skipped without consuming a slot. Empty input yields all zeros.

    ``seat_numbers`` is the snapshot's seat-number set. When given, the
    actor channel is the ordinal distance in ``sorted(set(seat_numbers))``
    — ``(idx(seat) - idx(hero)) mod n`` over ``max(1, n - 1)`` — so sparse
    seat numbers cannot alias onto hero's own 0.0; an unknown seat, or one
    absent from the set, reads the 0.5 midpoint. ``None`` keeps the
    documented modular fallback on raw seat numbers (dense seats only).

    Raises ``schema3.Schema3Error`` when ``big_blind_chips`` is below 1
    (chip amounts are positive integers, and the size channel divides by
    it) or ``player_count`` is below 1 (the actor unit reduces seat
    distance modulo the table size).
    """

    if big_blind_chips < 1:
        raise schema3.Schema3Error(
            f"big_blind_chips must be >= 1, got {big_blind_chips!r}"
        )
    if player_count < 1:
        raise schema3.Schema3Error(
            f"player_count must be >= 1, got {player_count!r}"
        )

    seat_ordinals = (
        _seat_ordinals(seat_numbers) if seat_numbers is not None else None
    )
    per_street: dict[str, list[tuple[str, Mapping[str, object]]]] = {
        street: [] for street in schema3.HISTORY_STREETS
    }
    for record in recent_actions:
        street = record.get("street")
        if not isinstance(street, str) or street not in per_street:
            continue
        action = record.get("action")
        family = action_family(action) if isinstance(action, str) else None
        if family is None:
            continue
        per_street[street].append((family, record))

    values = [0.0] * len(schema3.HISTORY_FEATURE_NAMES)
    for street, entries in per_street.items():
        if len(entries) > schema3.HISTORY_SLOTS:
            values[_OVERFLOW_POSITION[street]] = 1.0
            entries = entries[-schema3.HISTORY_SLOTS:]
        first_slot = schema3.HISTORY_SLOTS - len(entries)
        for offset, (family, record) in enumerate(entries):
            slot = first_slot + offset
            values[_SLOT_POSITION[street, slot, "occurred"]] = 1.0
            # action_family's non-None returns are exactly the three
            # family channel names; a divergence raises KeyError here
            # rather than silently mis-setting a channel.
            values[_SLOT_POSITION[street, slot, family]] = 1.0
            values[_SLOT_POSITION[street, slot, "size"]] = _bet_size(
                record.get("amount"), big_blind_chips
            )
            values[_SLOT_POSITION[street, slot, "actor"]] = _actor_unit(
                record.get("seat_number"),
                hero_seat_number,
                player_count,
                seat_ordinals,
            )
    return tuple(values)
