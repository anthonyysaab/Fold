"""Deterministic Arena decision rules shared by every policy backend.

:class:`DecisionRules` holds the safety rails that were originally methods of
``PlaygroundPolicy``: legal-action mapping, bounded raise sizing, equity
fallbacks for short-handed tables, and the sub-two-second deadline path. A
backend only supplies :meth:`DecisionRules._family`, which maps a feature
vector to one of the proposal families (``fold`` / ``check_call`` /
``aggress``). The torch checkpoint adapter and the pure-Python deployment
build both inherit from this class, so their behavior can differ only in the
family proposal itself.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from devfun_poker_playground.equity import board_improvement, estimate_equity
from devfun_poker_playground.snapshots import (
    _AGGRESSIVE_ACTIONS,
    ArenaSnapshotError,
    _cards,
    _hero_and_seats,
    _integer,
    _mapping,
    _sequence,
    features_from_table,
)


@dataclass(frozen=True, slots=True)
class ArenaAction:
    action: str
    amount: int | None
    message: str
    reasoning: str | None = None

    def to_payload(self) -> dict[str, str | int]:
        payload: dict[str, str | int] = {
            "action": self.action,
            "message": self.message,
        }
        if self.amount is not None:
            payload["amount"] = self.amount
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        return payload


# Extra equity demanded over raw pot odds to continue against a bet, by
# street. Replaces the old flat -0.03 loosening that made almost any pair a
# call against any bet size.
_CALL_MARGINS = {"preflop": 0.0, "flop": 0.02, "turn": 0.05, "river": 0.08}

# Board-contribution discount (v3), from three rated-match stack-offs where
# the five-card hand was mostly the board's: trips-plus-kicker and hollow
# two pair on paired boards kept calling into ranges stuffed with boats.
# Discounted tiers (equity.board_improvement) condition harder on the
# aggressor's range, demand a bigger margin, stop stacking off, and stop
# barreling; ``fresh`` hands are untouched.
_BOARD_DISCOUNT_RANGE_TIGHTEN = {"kicker": 0.60, "thin": 0.75}
_BOARD_DISCOUNT_MARGINS = {"kicker": 0.12, "thin": 0.10}
_BOARD_DISCOUNT_STACK_GATES = {"kicker": (0.18, 0.75), "thin": (0.30, 0.80)}
_BOARD_DISCOUNT_AGGRESSION_FLOORS = {"kicker": 0.82, "thin": 0.78}


class DecisionRules:
    """Family proposals plus deterministic Arena safety rails."""

    def __init__(self, *, equity_trials: int = 100, seed: int = 7) -> None:
        if equity_trials < 0:
            raise ValueError("equity_trials cannot be negative")
        self.equity_trials = equity_trials
        self.seed = seed

    def _family(self, features: tuple[float, ...]) -> str:
        raise NotImplementedError("policy backends must implement _family")

    def _equity(self, table: Mapping[str, Any], top_fraction: float = 1.0) -> float | None:
        if self.equity_trials == 0:
            return None
        hero, seats = _hero_and_seats(table)
        hole_cards = _cards(hero.get("holeCards"), "holeCards", 2)
        hole = (hole_cards[0], hole_cards[1])
        board = _cards(table.get("boardCards"), "boardCards")
        opponents = sum(
            1
            for seat in seats
            if seat is not hero
            and str(seat.get("status") or "").casefold() not in {"folded", "settled"}
        )
        table_id = str(table.get("tableId") or table.get("id") or "")
        digest = hashlib.sha256(f"{self.seed}:{table_id}".encode()).digest()
        trial_seed = int.from_bytes(digest[:8], "big")
        return estimate_equity(
            hole,
            board,
            opponents,
            trials=self.equity_trials,
            seed=trial_seed,
            top_fraction=top_fraction,
        )

    @staticmethod
    def _street(table: Mapping[str, Any]) -> str:
        return str(table.get("street") or "").casefold()

    @staticmethod
    def _board_tier(table: Mapping[str, Any]) -> str:
        """Board-contribution tier of the hero holding (see board_improvement)."""

        hero, _ = _hero_and_seats(table)
        hole_cards = _cards(hero.get("holeCards"), "holeCards", 2)
        board = _cards(table.get("boardCards"), "boardCards")
        return board_improvement((hole_cards[0], hole_cards[1]), board)

    @staticmethod
    def _hero_seat_number(table: Mapping[str, Any]) -> int:
        return _integer(table.get("selfSeatNumber"), "selfSeatNumber", minimum=1)

    @classmethod
    def _aggressive_events(
        cls, table: Mapping[str, Any], *, hero: bool, street: str | None = None
    ) -> int:
        """Count aggressive actions by hero (or by opponents) in recentEvents."""

        hero_seat = cls._hero_seat_number(table)
        count = 0
        for raw_event in _sequence(table.get("recentEvents") or [], "recentEvents"):
            event = _mapping(raw_event, "recentEvent")
            if street is not None and str(event.get("street") or "").casefold() != street:
                continue
            summary_value = event.get("summary")
            if summary_value is None:
                continue
            summary = _mapping(summary_value, "recentEvent.summary")
            action = str(summary.get("action") or "").casefold()
            if action not in _AGGRESSIVE_ACTIONS:
                continue
            seat_number = summary.get("seatNumber")
            is_hero = isinstance(seat_number, int) and seat_number == hero_seat
            if is_hero == hero:
                count += 1
        return count

    def _call_top_fraction(
        self, table: Mapping[str, Any], allowed: Mapping[str, Any]
    ) -> float:
        """How much of the opponent's range to consider when facing a bet.

        The more they have raised this hand — and the bigger the bet in
        front of us — the more their range is weighted toward strong made
        hands, so the smaller the fraction of holdings we simulate against.
        No aggression means no conditioning (1.0 = uniformly random).
        """

        to_call = _integer(allowed.get("callChips", 0), "callChips")
        if to_call <= 0:
            return 1.0
        opponent_raises = self._aggressive_events(table, hero=False)
        if opponent_raises == 0:
            return 1.0
        fraction = 0.75 * (0.8 ** (opponent_raises - 1))
        pot = _integer(table.get("potChips"), "potChips")
        bet_fraction = to_call / max(pot - to_call, 1)
        if bet_fraction > 1.0:
            fraction *= 0.6
        elif bet_fraction > 0.6:
            fraction *= 0.8
        # A bet into a hand that barely improves the board is aimed at the
        # board itself: weight the aggressor even further toward hands that
        # beat it.
        fraction *= _BOARD_DISCOUNT_RANGE_TIGHTEN.get(self._board_tier(table), 1.0)
        return min(1.0, max(0.20, fraction))

    def _call_clears_margin(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float | None,
    ) -> bool:
        """Whether calling is justified: pot odds + street margin + stack gate."""

        if equity is None:
            return True
        tier = self._board_tier(table)
        margin = _CALL_MARGINS.get(self._street(table), 0.08)
        margin += _BOARD_DISCOUNT_MARGINS.get(tier, 0.0)
        if equity < self._pot_odds(table, allowed) + margin:
            return False
        hero, _ = _hero_and_seats(table)
        stack = _integer(hero.get("stackChips"), "hero stackChips")
        to_call = _integer(allowed.get("callChips", 0), "callChips")
        if to_call >= 0.6 * stack and equity < 0.68:
            return False
        if to_call >= 0.35 * stack and equity < 0.62:
            return False
        gate = _BOARD_DISCOUNT_STACK_GATES.get(tier)
        if gate is not None:
            stack_fraction, floor = gate
            if to_call >= stack_fraction * stack and equity < floor:
                return False
        return True

    def decide(
        self,
        table: Mapping[str, Any],
        deadline_s: float = 10.0,
        research_context: Mapping[str, Any] | None = None,
    ) -> dict[str, str | int]:
        context = research_context or {}
        cached_position = context.get("position")
        if cached_position is not None:
            try:
                cached_position = float(cached_position)
            except (TypeError, ValueError) as exc:
                raise ArenaSnapshotError("research_context.position must be a number") from exc
        features = features_from_table(table, position=cached_position)
        allowed = _mapping(table.get("allowedActions"), "allowedActions")
        available = {
            str(value)
            for value in _sequence(allowed.get("availableActions"), "availableActions")
        }
        if deadline_s < 2.0:
            action = self._deadline_action(table, allowed, available)
            return self._render(action, table, allowed, equity=None).to_payload()

        # Facing aggression, estimate equity against the strong part of the
        # opponent's range instead of a uniformly random hand.
        equity = self._equity(table, top_fraction=self._call_top_fraction(table, allowed))
        _, seats = _hero_and_seats(table)
        family = (
            self._short_handed_family(table, allowed, available, equity, features=features)
            if len(seats) < 6 and equity is not None
            else self._family(features)
        )
        if family == "aggress":
            action = self._aggressive_action(table, allowed, available, equity)
        elif family == "check_call":
            action = self._passive_action(table, allowed, available, equity)
        else:
            action = self._fold_action(table, allowed, available, equity)
        return self._render(action, table, allowed, equity).to_payload()

    def _short_handed_family(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float,
        features: tuple[float, ...] | None = None,
    ) -> str:
        del features  # Equity thresholds decide here; model backends may use them.
        _, seats = _hero_and_seats(table)
        opponent_count = max(1, len(seats) - 1)
        aggression_floor = min(0.72, 0.52 + 0.05 * max(0, opponent_count - 1))
        if self._street(table) == "preflop":
            # Keep junk like K4o/J9o from min-raising into strength preflop.
            aggression_floor += 0.04
        if any(action in available for action in ("bet", "raise")) and equity >= aggression_floor:
            return "aggress"
        if "check" in available:
            return "check_call"
        if "call" in available and self._call_clears_margin(table, allowed, equity):
            return "check_call"
        return "fold"

    def _fold_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float | None,
    ) -> tuple[str, int | None]:
        if "check" in available:
            return "check", None
        pot_odds = self._pot_odds(table, allowed)
        if (
            "call" in available
            and equity is not None
            and equity >= max(0.60, pot_odds + 0.15)
            and self._call_clears_margin(table, allowed, equity)
        ):
            return "call", None
        if "fold" in available:
            return "fold", None
        return self._passive_action(table, allowed, available, equity)

    def _passive_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float | None,
    ) -> tuple[str, int | None]:
        if "check" in available:
            return "check", None
        if "call" in available and self._call_clears_margin(table, allowed, equity):
            return "call", None
        if "fold" in available:
            return "fold", None
        if "call" in available:
            return "call", None
        return self._first_legal_aggression(table, allowed, available)

    def _aggressive_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float | None,
    ) -> tuple[str, int | None]:
        hero, seats = _hero_and_seats(table)
        opponent_count = sum(
            1
            for seat in seats
            if seat is not hero
            and str(seat.get("status") or "").casefold() not in {"folded", "settled"}
        )
        aggression_floor = min(0.74, 0.50 + 0.04 * max(0, opponent_count - 1))
        street = self._street(table)
        to_call = _integer(allowed.get("callChips", 0), "callChips")
        if to_call > 0 and self._aggressive_events(table, hero=True, street=street) > 0:
            # We already raised this street and got raised back: continuing
            # the war needs a near-nut hand, not a marginal edge vs random.
            aggression_floor = max(aggression_floor, 0.72)
        if equity is not None:
            # Betting the board's own hand only ever levers out worse board
            # play; hands that beat it never fold. Near-nuts may still bet.
            tier_floor = _BOARD_DISCOUNT_AGGRESSION_FLOORS.get(self._board_tier(table))
            if tier_floor is not None:
                aggression_floor = max(aggression_floor, tier_floor)
        if equity is not None and equity < aggression_floor:
            return self._passive_action(table, allowed, available, equity)

        if "bet" in available:
            sized = self._sized_action("bet", table, allowed, equity)
            if sized is not None:
                return sized
        if "raise" in available:
            sized = self._sized_action("raise", table, allowed, equity)
            if sized is not None:
                return sized
        # The warm-start model never controls an optional all-in in v0.
        return self._passive_action(table, allowed, available, equity)

    def _sized_action(
        self,
        action: str,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float | None,
    ) -> tuple[str, int] | None:
        hero, _ = _hero_and_seats(table)
        pot = _integer(table.get("potChips"), "potChips")
        big_blind = _integer(table.get("bigBlindChips"), "bigBlindChips", minimum=1)
        stack = _integer(hero.get("stackChips"), "hero stackChips")
        contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
        call_chips = _integer(allowed.get("callChips", 0), "callChips")

        range_name = "betRange" if action == "bet" else "raiseRange"
        amount_range = _mapping(allowed.get(range_name), range_name)
        minimum = _integer(amount_range.get("min"), f"{range_name}.min")
        maximum = _integer(amount_range.get("max"), f"{range_name}.max")
        base = contribution if action == "bet" else _integer(
            allowed.get("callToAmount"), "callToAmount"
        )
        desired = base + max(big_blind, round(0.5 * (pot + call_chips)))

        if equity is None or equity < 0.72:
            risk_cap = contribution + max(big_blind, round(0.35 * stack))
            maximum = min(maximum, risk_cap)
        if maximum < minimum:
            return None
        return action, min(max(desired, minimum), maximum)

    def _first_legal_aggression(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
    ) -> tuple[str, int | None]:
        for action in ("bet", "raise"):
            if action in available:
                sized = self._sized_action(action, table, allowed, equity=None)
                if sized is not None:
                    return sized
        if "all-in" in available:
            return "all-in", _integer(allowed.get("allInToAmount"), "allInToAmount")
        raise ArenaSnapshotError("no legal fallback action")

    def _deadline_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
    ) -> tuple[str, int | None]:
        if "check" in available:
            return "check", None
        hero, _ = _hero_and_seats(table)
        stack = _integer(hero.get("stackChips"), "hero stackChips")
        call_chips = _integer(allowed.get("callChips", 0), "callChips")
        if "call" in available and call_chips <= max(1, round(0.05 * stack)):
            return "call", None
        if "fold" in available:
            return "fold", None
        if "call" in available:
            return "call", None
        return self._first_legal_aggression(table, allowed, available)

    @staticmethod
    def _pot_odds(table: Mapping[str, Any], allowed: Mapping[str, Any]) -> float:
        pot = _integer(table.get("potChips"), "potChips")
        call_chips = _integer(allowed.get("callChips", 0), "callChips")
        return 0.0 if call_chips == 0 else call_chips / max(pot + call_chips, 1)

    def _render(
        self,
        action: tuple[str, int | None],
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float | None,
    ) -> ArenaAction:
        del equity  # Keep private-card estimates out of public table chat.
        action_name, amount = action
        templates = {
            "fold": (
                "the price and line make this a clean release",
                "this branch is too expensive to continue",
            ),
            "check": (
                "keeping the pot controlled on this texture",
                "taking the free card and preserving flexibility",
            ),
            "call": (
                "the price leaves enough room to continue",
                "continuing without inflating the pot",
            ),
            "bet": (
                "a measured size pressures the weaker range",
                "using a controlled size to deny cheap realization",
            ),
            "raise": (
                "applying pressure while keeping stack risk bounded",
                "this line supports a measured pressure raise",
            ),
            "all-in": (
                "stack geometry makes full commitment cleaner than a partial size",
                "the remaining stack works better as one decision",
            ),
        }
        table_id = str(table.get("tableId") or table.get("id") or "")
        digest = hashlib.sha256(f"{table_id}:{action_name}".encode()).digest()
        choices = templates[action_name]
        message = choices[digest[0] % len(choices)]

        reasoning: str | None = None
        if bool(allowed.get("reasoningRequired")):
            pot_odds = round(100 * self._pot_odds(table, allowed))
            fields = [f'ke: "pot odds {pot_odds}%"', 'pp: "risk-controlled line"']
            if action_name in _AGGRESSIVE_ACTIONS:
                fields.append('sr: "bounded pot pressure"')
            reasoning = "{" + ", ".join(fields) + "}"
        return ArenaAction(action_name, amount, message, reasoning)


__all__ = [
    "ArenaAction",
    "DecisionRules",
]
