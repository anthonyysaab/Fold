"""Session-scoped opponent aggression tracking for range conditioning.

The decision engine estimates equity against the top fraction of an
aggressor's range. That assumption is only as good as its belief about how
often the aggressor attacks: a player who shoves every hand cannot hold a
premium range, and treating each shove as strength is exactly the leak a
permanent all-in attacker milks. This module supplies the evidence.

:class:`AggressionTracker` counts, per opponent, how many recorded actions
were aggressive (bet, raise, all-in) out of how many actions they took, and
exposes an evidence-gated range floor::

    range_floor = aggressions / (opportunities + prior_opportunities)

With no evidence the floor is 0.0 and the engine's prior conditioning stands
untouched. As evidence accumulates the floor converges to the observed
aggression frequency, because a range cannot be narrower than how often its
owner plays it fast. Within one hand, though, the floor decays with each
further aggressive action by the same opponent (``_ESCALATION_DECAY``):
raising 42% of hands is not the same evidence as raising five times on one
board, and conflating them caused the 2026-08-13 live bust. There is no
hardcoded cap in either direction; the two :class:`TrackerSettings` fields
(evidence damping and memory decay) are future learned parameters like
every other tuning surface in this project.

Opponents are keyed by Arena ``agentId`` when seats carry one, so evidence
follows a player across tables; otherwise keys fall back to table and seat.
Everything is deterministic given the observed snapshots. Because the
tracker remembers earlier hands, a decision is reproducible from the session
history rather than from one snapshot alone.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from engine.game_state import (
    _active_seats,
    _AGGRESSIVE_ACTIONS,
    _hero_and_seats,
    _integer,
    _mapping,
    _sequence,
)

# Tables whose per-hand event bookkeeping is retained; older tables roll off.
_MAX_TRACKED_TABLES = 32

# Per-hand escalation decay of the range floor: each aggressive action past
# an opponent's first in the current hand multiplies their floor by this
# factor. The floor is a GLOBAL aggression frequency, and applying it flat
# caused the 2026-08-13 bust (V5_BUST_POSTMORTEM.md): a 0.4237-frequency
# villain four raises deep on one flop was still modelled at his global top
# 42%, so hero shipped 756 BB holding trip tens with a weak kicker at a
# believed 87.5% equity. Calibration anchors at frequency 0.42: one raise
# this hand leaves the floor unchanged (0.42), three raises cut it by more
# than half (0.49x -> 0.206), five raises put it at 0.24x -> 0.101, below
# 0.15. The engine's 0.20 conditioning clamp is applied after this decay
# and stays the hard minimum.
_ESCALATION_DECAY = 0.70


@dataclass(frozen=True, slots=True)
class TrackerSettings:
    """Evidence handling for the opponent aggression model.

    ``prior_opportunities`` damps early evidence: the range floor is
    ``aggressions / (opportunities + prior_opportunities)``, so it needs
    several observed actions before it moves anything. ``decay`` is the
    per-observation exponential memory (0.98 keeps roughly the last fifty
    actions relevant), letting the belief follow opponents who change gear.
    Both are future learned parameters.
    """

    prior_opportunities: float = 4.0
    decay: float = 0.98

    def __post_init__(self) -> None:
        if not 0.5 <= float(self.prior_opportunities) <= 100.0:
            raise ValueError("prior_opportunities must be between 0.5 and 100")
        if not 0.80 <= float(self.decay) <= 1.0:
            raise ValueError("decay must be between 0.8 and 1")

    def to_mapping(self) -> dict[str, object]:
        """JSON-ready form for a future learned-artifact manifest."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "TrackerSettings":
        """Rebuild validated settings from artifact JSON; unknown keys fail."""

        return cls(**dict(mapping))


DEFAULT_TRACKER_SETTINGS = TrackerSettings()


def _event_actions(table: Mapping[str, Any]) -> list[tuple[int, str, str]]:
    """(seat, action, dedup key) for each recorded decision event this hand.

    Forced blind posts are not decisions and must not dilute the aggression
    frequency; a permanent shover's one blind and one shove per hand would
    otherwise cap the measured frequency near one half.
    """

    actions: list[tuple[int, str, str]] = []
    for raw_event in _sequence(table.get("recentEvents") or [], "recentEvents"):
        event = _mapping(raw_event, "recentEvent")
        if str(event.get("type") or "") not in ("ActionTaken", "TimeoutAction"):
            continue
        summary_value = event.get("summary")
        if summary_value is None:
            continue
        summary = _mapping(summary_value, "recentEvent.summary")
        action = str(summary.get("action") or "").casefold()
        seat_number = summary.get("seatNumber")
        if (
            not action
            or isinstance(seat_number, bool)
            or not isinstance(seat_number, int)
        ):
            continue
        street = str(event.get("street") or "").casefold()
        amount = summary.get("amount")
        key = f"{street}|{seat_number}|{action}|{amount}"
        actions.append((seat_number, action, key))
    return actions


class AggressionTracker:
    """Deterministic per-opponent aggression frequencies for one session."""

    def __init__(self, settings: TrackerSettings | None = None) -> None:
        self.settings = settings or DEFAULT_TRACKER_SETTINGS
        # key -> [decayed opportunities, decayed aggressions, decayed folds]
        self._stats: dict[str, list[float]] = {}
        # tableId -> (hand marker, event keys already counted this hand)
        self._hands: OrderedDict[str, tuple[str, set[str]]] = OrderedDict()

    @staticmethod
    def _table_id(table: Mapping[str, Any]) -> str:
        return str(table.get("tableId") or table.get("id") or "")

    @staticmethod
    def _seat_key(
        table: Mapping[str, Any], seats: list[Mapping[str, Any]], seat_number: int
    ) -> str:
        for seat in seats:
            if seat.get("seatNumber") == seat_number:
                for field in ("agentId", "agent_id", "name"):
                    identity = seat.get(field)
                    if isinstance(identity, str) and identity:
                        return identity
                break
        return f"{AggressionTracker._table_id(table)}:{seat_number}"

    @staticmethod
    def _hand_marker(table: Mapping[str, Any], hero: Mapping[str, Any]) -> str:
        """A value that changes hand to hand so per-hand bookkeeping resets."""

        hand_id = table.get("handId") or table.get("handNumber")
        if hand_id is not None:
            return str(hand_id)
        hole = hero.get("holeCards") or ()
        return "|".join(str(card) for card in _sequence(hole, "holeCards"))

    def observe(self, table: Mapping[str, Any]) -> None:
        """Fold one pending-action snapshot into the per-opponent evidence.

        Safe to call on every poll of the same hand: events already counted
        under the current hand marker are skipped, and a marker change
        (new hole cards or hand id) starts the hand's bookkeeping afresh.
        """

        hero, seats = _hero_and_seats(table)
        hero_seat = _integer(hero.get("seatNumber"), "seatNumber", minimum=1)
        table_id = self._table_id(table)
        marker = self._hand_marker(table, hero)

        stored = self._hands.get(table_id)
        if stored is None or stored[0] != marker:
            counted: set[str] = set()
        else:
            counted = stored[1]

        decay = self.settings.decay
        for seat_number, action, key in _event_actions(table):
            if seat_number == hero_seat or key in counted:
                continue
            counted.add(key)
            stats = self._stats.setdefault(
                self._seat_key(table, seats, seat_number), [0.0, 0.0, 0.0]
            )
            stats[0] = stats[0] * decay + 1.0
            stats[1] = stats[1] * decay + (
                1.0 if action in _AGGRESSIVE_ACTIONS else 0.0
            )
            stats[2] = stats[2] * decay + (1.0 if action == "fold" else 0.0)

        self._hands[table_id] = (marker, counted)
        self._hands.move_to_end(table_id)
        while len(self._hands) > _MAX_TRACKED_TABLES:
            self._hands.popitem(last=False)

    def clone(self) -> "AggressionTracker":
        """A copy that shares nothing mutable with ``self``.

        The counterfactual replay deep-copies seat lists, and each seat
        carries its agent, so a generic copy walks every tracker too.
        ``observe`` mutates the per-opponent stats lists and the per-hand
        bookkeeping sets IN PLACE, so both containers are copied; the
        frozen ``settings`` dataclass is shared. A clone and its original
        then record evidence independently, exactly as a ``deepcopy``
        would have.
        """

        clone = AggressionTracker.__new__(AggressionTracker)
        clone.settings = self.settings
        clone._stats = {
            key: list(values) for key, values in self._stats.items()
        }
        clone._hands = OrderedDict(
            (table_id, (marker, set(counted)))
            for table_id, (marker, counted) in self._hands.items()
        )
        return clone

    def opportunities(self, key: str) -> float:
        return self._stats.get(key, (0.0, 0.0, 0.0))[0]

    def range_floor(self, key: str) -> float:
        """Narrowest credible range width for this opponent, 0 without data."""

        opportunities, aggressions, _ = self._stats.get(key, (0.0, 0.0, 0.0))
        return aggressions / (opportunities + self.settings.prior_opportunities)

    def stickiness(self, key: str) -> float:
        """Evidence-gated continue tendency: how rarely this opponent folds.

        A station (calls everything) converges toward 1.0 and a nit toward
        its low continue rate; with no evidence the value is 0.0, meaning
        unknown rather than sticky. This is the discriminator aggression
        frequency cannot provide, since stations and nits are both passive.
        """

        opportunities, _, folds = self._stats.get(key, (0.0, 0.0, 0.0))
        return max(0.0, opportunities - folds) / (
            opportunities + self.settings.prior_opportunities
        )

    def confidence(self, key: str) -> float:
        """How much observed evidence supports one opponent profile."""

        opportunities = self.opportunities(key)
        return opportunities / (opportunities + self.settings.prior_opportunities)

    def pressure_actor_profile(
        self, table: Mapping[str, Any]
    ) -> tuple[float, float, float]:
        """Wildness, stickiness, and confidence for the latest aggressor.

        When nobody has applied pressure this hand, use the active opponent
        with the most evidence rather than mixing identities with independent
        maxima.
        """

        hero, seats = _hero_and_seats(table)
        hero_seat = _integer(hero.get("seatNumber"), "seatNumber", minimum=1)
        for seat_number, action, _ in reversed(_event_actions(table)):
            if seat_number == hero_seat or action not in _AGGRESSIVE_ACTIONS:
                continue
            key = self._seat_key(table, seats, seat_number)
            return self.range_floor(key), self.stickiness(key), self.confidence(key)

        keys = [
            self._seat_key(
                table,
                seats,
                _integer(seat.get("seatNumber"), "seatNumber", minimum=1),
            )
            for seat in _active_seats(seats)
            if seat is not hero and seat.get("seatNumber") is not None
        ]
        if not keys:
            return 0.0, 0.0, 0.0
        key = max(keys, key=self.confidence)
        return self.range_floor(key), self.stickiness(key), self.confidence(key)

    def range_floor_for_table(self, table: Mapping[str, Any]) -> float:
        """Widest escalation-decayed range floor among this hand's attackers.

        Each attacker's global floor is decayed by ``_ESCALATION_DECAY`` per
        aggressive action past their first in the current hand, counted per
        opponent so one player's five-bet never discounts another's single
        raise. A maniac still starts every hand floored at their observed
        frequency; repeated raising on one board earns back its credit.
        """

        hero, seats = _hero_and_seats(table)
        hero_seat = _integer(hero.get("seatNumber"), "seatNumber", minimum=1)
        counts: dict[str, int] = {}
        for seat_number, action, _ in _event_actions(table):
            if seat_number == hero_seat or action not in _AGGRESSIVE_ACTIONS:
                continue
            key = self._seat_key(table, seats, seat_number)
            counts[key] = counts.get(key, 0) + 1
        floor = 0.0
        for key, count in counts.items():
            decayed = self.range_floor(key) * _ESCALATION_DECAY ** (count - 1)
            floor = max(floor, decayed)
        return floor

    def _max_over_active(self, table: Mapping[str, Any], reader) -> float:
        hero, seats = _hero_and_seats(table)
        best = 0.0
        for seat in _active_seats(seats):
            if seat is hero or seat.get("seatNumber") is None:
                continue
            seat_number = _integer(seat.get("seatNumber"), "seatNumber", minimum=1)
            best = max(best, reader(self._seat_key(table, seats, seat_number)))
        return best

    def max_active_wildness(self, table: Mapping[str, Any]) -> float:
        """Highest observed aggression frequency among live opponents."""

        return self._max_over_active(table, self.range_floor)

    def max_active_stickiness(self, table: Mapping[str, Any]) -> float:
        """Highest observed continue tendency among live opponents.

        The stickiest player still in the hand bounds a bluff's fold
        equity, because everyone must fold for a bluff to take the pot.
        """

        return self._max_over_active(table, self.stickiness)


__all__ = [
    "AggressionTracker",
    "DEFAULT_TRACKER_SETTINGS",
    "TrackerSettings",
]
