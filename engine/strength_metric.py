"""The canonical hand-strength metric — one definition, three consumers.

V8_DESIGN.md §2. This session (2026-08-16) found two hand-strength scales in
circulation disagreeing 3x on the same decisions: feature 125's multiway
Monte-Carlo equity (player-count- and street-dependent) and the frozen field
comparison's percentile-style scale. This module is the fix: features,
training labels, and evaluation reports all import strength from here, and
nowhere else.

Definition:

- **Postflop** — the combo-weighted share of possible opponent two-card
  holdings the hero's current made hand beats on this board, ties counted
  half, by exact enumeration. Deterministic, seedless, street-comparable,
  player-count invariant.
- **Preflop** — the fixed percentile of the holding's canonical class under
  a heads-up all-in-equity ordering (``preflop_percentiles.py``, generated
  once by ``tools/build_preflop_percentiles.py``). A dict lookup at serve.

Stdlib-only (vendored ``treys``), like every live-inference module.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from engine._vendor.treys import Card, Evaluator
from engine.preflop_percentiles import PREFLOP_PERCENTILES
from engine.schema3 import CARD_CODES

# The card universe is the schema's, never restated (F8): the deck and all
# membership validation derive from schema3.CARD_CODES. _RANKS survives
# only as the canonical_class ordering (low to high).
_CARD_CODE_SET = frozenset(CARD_CODES)
_RANKS = "23456789TJQKA"

# One-line consistency check: the ordering string and the schema's card
# universe must agree, or class ordering and validation would drift apart.
assert set(_RANKS) == {code[0] for code in CARD_CODES}, "rank/CARD_CODES drift"


class StrengthMetricError(ValueError):
    """Raised when cards passed to the metric are malformed."""


_EVALUATOR: Evaluator | None = None


def _evaluator() -> Evaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = Evaluator()
    return _EVALUATOR


@lru_cache(maxsize=1)
def _full_deck() -> tuple[int, ...]:
    return tuple(Card.new(code) for code in CARD_CODES)


def _parse(cards: Sequence[str], name: str) -> list[int]:
    parsed = []
    for card in cards:
        if not isinstance(card, str) or card not in _CARD_CODE_SET:
            raise StrengthMetricError(f"{name} card {card!r} is not a card code")
        parsed.append(Card.new(card))
    if len(set(parsed)) != len(parsed):
        raise StrengthMetricError(f"{name} contains duplicate cards")
    return parsed


def canonical_class(hole: Sequence[str]) -> str:
    """``["Ah","Kh"]`` -> ``"AKs"``; ``["Ks","Ah"]`` -> ``"AKo"``; pairs ``"AA"``."""

    if len(hole) != 2:
        raise StrengthMetricError("a holding is exactly two cards")
    _parse(hole, "hole")
    first, second = hole
    if _RANKS.index(first[0]) < _RANKS.index(second[0]):
        first, second = second, first
    if first[0] == second[0]:
        return first[0] * 2
    suffix = "s" if first[1] == second[1] else "o"
    return f"{first[0]}{second[0]}{suffix}"


def strength_percentile(hole: Sequence[str], board: Sequence[str]) -> float:
    """The canonical strength of ``hole`` on ``board``, in [0, 1].

    Postflop this is exact: every unseen opponent holding is enumerated and
    scored with the vendored evaluator (~1k seven-card evaluations). Preflop
    it is the committed table lookup. Both are deterministic.
    """

    hole_cards = _parse(hole, "hole")
    if len(hole_cards) != 2:
        raise StrengthMetricError("a holding is exactly two cards")
    board_cards = _parse(board, "board")
    if board_cards and len(board_cards) not in (3, 4, 5):
        raise StrengthMetricError("a board has 0, 3, 4 or 5 cards")
    if set(hole_cards) & set(board_cards):
        raise StrengthMetricError("hole and board share a card")

    if not board_cards:
        return PREFLOP_PERCENTILES[canonical_class(hole)]

    evaluator = _evaluator()
    mine = evaluator.evaluate(board_cards, hole_cards)
    dead = set(hole_cards) | set(board_cards)
    live = [card for card in _full_deck() if card not in dead]
    beaten = tied = total = 0
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            theirs = evaluator.evaluate(board_cards, [live[i], live[j]])
            total += 1
            if mine < theirs:  # treys: lower rank is stronger
                beaten += 1
            elif mine == theirs:
                tied += 1
    return (beaten + 0.5 * tied) / total


__all__ = [
    "StrengthMetricError",
    "canonical_class",
    "strength_percentile",
]
