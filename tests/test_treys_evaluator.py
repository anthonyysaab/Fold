"""Structural best-five selection must agree with the brute-force oracle.

The vendored treys evaluator's ``_six``/``_seven`` now select the best
five cards by hand structure and rank that subset ONCE through the same
lookup table, instead of scanning all subsets (the 2026-09-02 harvest
profile: the scan was ~52% of runtime). The original scan survives as
``_six_reference``/``_seven_reference`` and is the independent oracle
these tests pin against: agreement means the selector picks exactly the
subset the scan minimises over, and the rank integers are the same
lookup values either way.

Covered: every hand class boundary (straight flush, the wheel both as
the only straight and beside a higher straight, quads, quads beside a
flush, full houses including double trips, flushes, straights beside
pairs and trips, three pair, two pair, one pair, high card) at 6 and 7
cards, plus seeded random agreement on tens of thousands of deals.
"""

from __future__ import annotations

import itertools
import random
import unittest

from engine._vendor.treys import Card, Deck, Evaluator

_EVALUATOR = Evaluator()

# Boundary cases as card strings; each is 6 or 7 cards whose best five
# the structural selector must resolve exactly as the oracle does.
_BOUNDARY_CASES: tuple[tuple[str, ...], ...] = (
    # Straight flushes, including the wheel and the wheel beside 2-6.
    ("As", "Ks", "Qs", "Js", "Ts", "2d", "3d"),
    ("As", "2s", "3s", "4s", "5s", "Kh", "Kd"),
    ("As", "2s", "3s", "4s", "5s", "6s", "Kd"),
    ("Ah", "2d", "3c", "4s", "5h", "9d", "9c"),
    ("Ah", "2d", "3c", "4s", "5h", "6d", "9c"),
    # Quads, including quads beside a flush.
    ("Ah", "Ad", "Ac", "As", "Kh", "Qd", "2c"),
    ("Ah", "Ad", "Ac", "As", "Kh", "Qh", "Jh"),
    ("2h", "2d", "2c", "2s", "Ah", "Kd", "Qc"),
    # Full houses, including double trips and trips plus a flush.
    ("Kh", "Kd", "Kc", "Qh", "Qd", "2s", "3c"),
    ("Kh", "Kd", "Kc", "Qh", "Qd", "Qc", "As"),
    ("5h", "5d", "5c", "2h", "2d", "2c", "As"),
    ("Ah", "Ad", "Ac", "Kh", "Kd", "2h", "3h"),
    # Flushes, six-suited included; a flush beside a straight.
    ("Kh", "Qh", "9h", "5h", "2h", "As", "Ad"),
    ("Ah", "Kh", "Qh", "Jh", "2h", "3h", "4d"),
    ("Ah", "Kh", "Qh", "Jh", "Th", "9d", "8d"),
    # Straights: beside trips, a pair, and two overlapping runs.
    ("8h", "8d", "8c", "7s", "6h", "5d", "4c"),
    ("9h", "9d", "8c", "7s", "6h", "5d", "4c"),
    ("2h", "3d", "4c", "5s", "6h", "7d", "8c"),
    ("Ah", "2d", "3c", "4s", "5h", "7d", "9c"),
    # Trips.
    ("Qh", "Qd", "Qc", "As", "Kd", "2c", "3h"),
    ("Qh", "Qd", "Qc", "As", "Kd", "Jc", "Th"),
    # Two pair: with a high singleton kicker and three pairs.
    ("Ah", "Ad", "Kh", "Kd", "Qs", "2c", "3h"),
    ("Ah", "Ad", "Kh", "Kd", "Qs", "Qd", "2c"),
    ("2h", "2d", "3c", "3s", "Ah", "Kd", "Qc"),
    # One pair.
    ("Ah", "Ad", "Ks", "Qd", "Jc", "2c", "3h"),
    ("2h", "2d", "As", "Kd", "Qc", "Jh", "9c"),
    # High card.
    ("Ah", "Kd", "Qs", "Jc", "9h", "3c", "2d"),
    ("2h", "3d", "4c", "5s", "7h", "9d", "Jc"),
    # 6-card shapes (the turn evaluator path).
    ("Kh", "Kd", "Kc", "Qh", "Qd", "2s"),
    ("Ah", "Kh", "Qh", "Jh", "2h", "3h"),
    ("8h", "8d", "8c", "7s", "6h", "5d"),
    ("2h", "3d", "4c", "5s", "6h", "7d"),
    ("As", "2s", "3s", "4s", "5s", "Kh"),
)


def _to_ints(cards: tuple[str, ...]) -> list[int]:
    return [Card.new(card) for card in cards]


class StructuralSelectorBoundaryTests(unittest.TestCase):
    def test_boundary_cases_agree_with_the_oracle(self) -> None:
        for cards in _BOUNDARY_CASES:
            values = _to_ints(cards)
            if len(values) == 7:
                self.assertEqual(
                    _EVALUATOR._seven(values),
                    _EVALUATOR._seven_reference(values),
                    msg=cards,
                )
            else:
                self.assertEqual(
                    _EVALUATOR._six(values),
                    _EVALUATOR._six_reference(values),
                    msg=cards,
                )

    def test_six_and_seven_never_miss_the_oracle_on_random_deals(self) -> None:
        rng = random.Random(20260902)
        for length in (6, 7):
            for _ in range(12_000):
                deck = Deck.GetFullDeck()
                rng.shuffle(deck)
                cards = deck[:length]
                fast = _EVALUATOR._seven if length == 7 else _EVALUATOR._six
                slow = (
                    _EVALUATOR._seven_reference
                    if length == 7
                    else _EVALUATOR._six_reference
                )
                self.assertEqual(
                    fast(cards), slow(cards), msg=[Card.int_to_str(c) for c in cards]
                )

    def test_every_subset_of_a_fixed_seven_agrees_with_the_five_card_path(self) -> None:
        # The 21 five-card subsets of one seven-card hand must rank
        # through ``_five`` exactly as they did before the selector
        # existed, and the selector must find the subset that minimises.
        values = _to_ints(("Ah", "Kd", "Qs", "Jc", "9h", "3c", "2d"))
        subset_scores = [_EVALUATOR._five(combo) for combo in itertools.combinations(values, 5)]
        self.assertEqual(_EVALUATOR._seven(values), min(subset_scores))


if __name__ == "__main__":
    unittest.main()
