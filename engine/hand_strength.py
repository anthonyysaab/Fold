"""Estimate poker hand strength with small Monte Carlo simulations.

Uses the vendored, MIT-licensed pure-Python ``treys`` evaluator (see
``_vendor/treys/LICENSE``) so deployment images need no third-party wheels.
The evaluator's lookup tables are built once per process and reused.
"""

from __future__ import annotations

from collections import Counter
from random import Random

from engine._vendor.treys import Card, Deck, Evaluator

_EVALUATOR: Evaluator | None = None
_FULL_DECK: tuple[int, ...] | None = None


def _shared_evaluator() -> Evaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = Evaluator()
    return _EVALUATOR


def _full_deck() -> tuple[int, ...]:
    global _FULL_DECK
    if _FULL_DECK is None:
        _FULL_DECK = tuple(Deck.GetFullDeck())
    return _FULL_DECK


def prewarm() -> None:
    """Build the card lookup tables ahead of the first timed decision."""

    _shared_evaluator()
    _full_deck()


def _treys_card(value: str) -> int:
    return Card.new(f"{value[0].upper()}{value[1].lower()}")


# treys hand-class integers (lower is stronger).
_CLASS_QUADS = 2
_CLASS_FULL_HOUSE = 3
_CLASS_TRIPS = 6
_CLASS_TWO_PAIR = 7
_CLASS_PAIR = 8


def _pair_structure_class(counts: Counter) -> int:
    """Hand class a paired board makes on its own (pair structure only)."""

    shape = sorted(counts.values(), reverse=True)
    if shape[0] >= 4:
        return _CLASS_QUADS
    if shape[0] == 3:
        return _CLASS_FULL_HOUSE if len(shape) > 1 and shape[1] >= 2 else _CLASS_TRIPS
    if len(shape) > 1 and shape[1] >= 2:
        return _CLASS_TWO_PAIR
    return _CLASS_PAIR


def _structural_ranks(
    hand_class: int, combined: Counter
) -> list[tuple[int, int]] | None:
    """(rank, copies) pairs forming the class structure of the best hand."""

    if hand_class == _CLASS_QUADS:
        return [(max(r for r, c in combined.items() if c >= 4), 4)]
    if hand_class == _CLASS_FULL_HOUSE:
        trips = max(r for r, c in combined.items() if c >= 3)
        pair = max(r for r, c in combined.items() if c >= 2 and r != trips)
        return [(trips, 3), (pair, 2)]
    if hand_class == _CLASS_TRIPS:
        return [(max(r for r, c in combined.items() if c >= 3), 3)]
    if hand_class == _CLASS_TWO_PAIR:
        pairs = sorted((r for r, c in combined.items() if c >= 2), reverse=True)
        return [(pairs[0], 2), (pairs[1], 2)]
    if hand_class == _CLASS_PAIR:
        return [(max(r for r, c in combined.items() if c >= 2), 2)]
    return None


def board_improvement(
    hole_cards: tuple[str, str],
    board_cards: tuple[str, ...],
) -> str:
    """Classify how much the hole cards improve on the board's own made hand.

    Returns ``"fresh"`` when the holding genuinely upgrades the board (or the
    board is unpaired / still preflop), ``"thin"`` when it keeps the board's
    own hand class and only upgrades within it (pairing the lone side card of
    a double-paired board, an overpair on a boat board), and ``"kicker"``
    when the made hand is the board's own structure and the hole cards
    contribute kickers at most. The discounted tiers lose to any holding
    that connects with a paired board, so bets there are heavily
    value-weighted against us no matter how strong our five cards look.
    """

    if len(board_cards) < 3:
        return "fresh"
    hole = [_treys_card(card) for card in hole_cards]
    board = [_treys_card(card) for card in board_cards]
    evaluator = _shared_evaluator()
    hero_rank = evaluator.evaluate(hole, board)
    if len(board) == 5 and hero_rank == evaluator.evaluate([], board):
        return "kicker"  # the hole cards add nothing at all

    board_counts = Counter(Card.get_rank_int(card) for card in board)
    if max(board_counts.values()) < 2:
        return "fresh"  # unpaired board: any improvement is real
    board_class = _pair_structure_class(board_counts)
    hero_class = evaluator.get_rank_class(hero_rank)
    if hero_class < board_class:  # lower treys class = stronger hand
        return "fresh"

    combined = board_counts.copy()
    for card in hole:
        combined[Card.get_rank_int(card)] += 1
    structural = _structural_ranks(hero_class, combined)
    if structural is None:  # straight/flush classes never tie a paired board
        return "fresh"
    if all(board_counts.get(rank, 0) >= copies for rank, copies in structural):
        return "kicker"
    return "thin"


def _chen_score(combo: tuple[int, int]) -> float:
    """Chen-formula preflop strength; higher is stronger."""

    rank_a = Card.get_rank_int(combo[0])
    rank_b = Card.get_rank_int(combo[1])
    high, low = (rank_a, rank_b) if rank_a >= rank_b else (rank_b, rank_a)
    points = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(high, (high + 2) / 2.0)
    if rank_a == rank_b:
        return max(5.0, points * 2.0)
    if Card.get_suit_int(combo[0]) == Card.get_suit_int(combo[1]):
        points += 2.0
    gap = high - low - 1
    points -= (0.0, 1.0, 2.0, 4.0)[gap] if gap <= 3 else 5.0
    if gap <= 1 and high <= 9:  # connected low cards can still make straights
        points += 1.0
    return points


def _restricted_combos(
    deck: list[int],
    board: tuple[int, ...],
    top_fraction: float,
) -> list[tuple[int, int]]:
    """Opponent hole combos ranked by strength, strongest ``top_fraction`` kept.

    Postflop, strength is the made-hand rank on the current board, a crude
    but effective proxy for the hands an opponent bets and raises with.
    Preflop it falls back to the Chen formula. This deliberately ignores
    draws; big aggression weighted toward made hands is the point.
    """

    combos = [
        (deck[i], deck[j]) for i in range(len(deck)) for j in range(i + 1, len(deck))
    ]
    if board:
        evaluator = _shared_evaluator()
        board_list = list(board)
        combos.sort(key=lambda combo: evaluator.evaluate(board_list, list(combo)))
    else:
        combos.sort(key=_chen_score, reverse=True)
    keep = max(1, round(top_fraction * len(combos)))
    return combos[:keep]


def estimate_equity(
    hero_hole_cards: tuple[str, str],
    board_cards: tuple[str, ...],
    opponent_count: int,
    *,
    trials: int,
    seed: int,
    top_fraction: float = 1.0,
) -> float:
    """Monte Carlo equity for the hero hand.

    ``top_fraction`` conditions ONE opponent (the aggressor) on the strongest
    fraction of possible holdings instead of a uniformly random hand; any
    remaining opponents stay random. 1.0 reproduces the unconditioned
    estimate exactly.
    """

    if len(hero_hole_cards) != 2:
        raise ValueError("hero_hole_cards must contain two cards")
    if len(board_cards) > 5:
        raise ValueError("board_cards cannot contain more than five cards")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    if opponent_count < 1:
        return 1.0
    if trials < 1:
        raise ValueError("trials must be positive")

    hero = tuple(_treys_card(card) for card in hero_hole_cards)
    board = tuple(_treys_card(card) for card in board_cards)
    known = {*hero, *board}
    if len(known) != len(hero) + len(board):
        raise ValueError("known cards must be unique")

    deck = [card for card in _full_deck() if card not in known]
    missing_board = 5 - len(board)
    cards_needed = 2 * opponent_count + missing_board
    if cards_needed > len(deck):
        raise ValueError("not enough cards remain to simulate the table")

    restricted = (
        _restricted_combos(deck, board, top_fraction) if top_fraction < 1.0 else None
    )

    evaluator = _shared_evaluator()
    rng = Random(seed)
    equity = 0.0
    for _ in range(trials):
        opponents: list[tuple[int, int]] = []
        if restricted is None:
            sample = rng.sample(deck, cards_needed)
            cursor = 0
            for _ in range(opponent_count):
                opponents.append((sample[cursor], sample[cursor + 1]))
                cursor += 2
            runout = [*board, *sample[cursor:]]
        else:
            aggressor = restricted[rng.randrange(len(restricted))]
            opponents.append(aggressor)
            rest = [card for card in deck if card not in aggressor]
            sample = rng.sample(rest, cards_needed - 2)
            cursor = 0
            for _ in range(opponent_count - 1):
                opponents.append((sample[cursor], sample[cursor + 1]))
                cursor += 2
            runout = [*board, *sample[cursor:]]

        hero_score = evaluator.evaluate(runout, list(hero))
        opponent_scores = [evaluator.evaluate(runout, list(hole)) for hole in opponents]
        best_score = min(hero_score, *opponent_scores)
        if hero_score == best_score:
            equity += 1.0 / (1 + sum(score == best_score for score in opponent_scores))
    return equity / trials


__all__ = ["board_improvement", "estimate_equity", "prewarm"]
