"""Deterministic boundary tests for the Monte Carlo equity estimator.

SCRIPT_REVIEW precondition for touching the range model: these pin the
``estimate_equity`` contract — seeded reproducibility, [0, 1] bounds, the
monotone response to range narrowing and opponent count, and the
preflop/postflop board-size rules — before any conditioning change ships.
"""

from __future__ import annotations

import unittest

from engine.hand_strength import estimate_equity

# A medium-strength holding: top pair, good kicker on a dry board. Strong
# against a random hand, behind most of a tight aggressor's range.
_MEDIUM_HOLE = ("Qh", "Jh")
_DRY_BOARD = ("Qc", "7d", "2s")


class EstimateEquityDeterminismTests(unittest.TestCase):
    def test_same_seed_reproduces_the_estimate_exactly(self) -> None:
        first = estimate_equity(_MEDIUM_HOLE, _DRY_BOARD, 1, trials=200, seed=11)
        second = estimate_equity(_MEDIUM_HOLE, _DRY_BOARD, 1, trials=200, seed=11)
        self.assertEqual(first, second)

    def test_a_different_seed_draws_a_different_sample(self) -> None:
        first = estimate_equity(_MEDIUM_HOLE, _DRY_BOARD, 1, trials=200, seed=11)
        other = estimate_equity(_MEDIUM_HOLE, _DRY_BOARD, 1, trials=200, seed=12)
        self.assertNotEqual(first, other)


class EstimateEquityBoundsTests(unittest.TestCase):
    def test_equity_stays_within_the_unit_interval(self) -> None:
        cases = (
            (_MEDIUM_HOLE, _DRY_BOARD, 2, 0.3),
            (("2c", "7d"), (), 3, 1.0),
            (("Ah", "Kh"), ("Qh", "Jh", "Th"), 1, 0.5),
            (("2h", "3d"), ("Ac", "Ad", "Ks", "Kd", "7c"), 4, 1.0),
        )
        for hole, board, opponents, top_fraction in cases:
            equity = estimate_equity(
                hole, board, opponents, trials=100, seed=4, top_fraction=top_fraction
            )
            self.assertGreaterEqual(equity, 0.0, msg=f"{hole} on {board}")
            self.assertLessEqual(equity, 1.0, msg=f"{hole} on {board}")

    def test_a_premium_pair_dominates_one_random_hand(self) -> None:
        equity = estimate_equity(("Ah", "Ad"), (), 1, trials=300, seed=3)
        self.assertGreater(equity, 0.7)

    def test_no_opponents_means_certain_equity(self) -> None:
        self.assertEqual(estimate_equity(("2h", "3d"), (), 0, trials=100, seed=1), 1.0)


class EstimateEquityConditioningTests(unittest.TestCase):
    def test_narrowing_top_fraction_never_helps_a_medium_hand(self) -> None:
        # Same seed and trials throughout: only the conditioning moves.
        uniform = estimate_equity(
            _MEDIUM_HOLE, _DRY_BOARD, 1, trials=300, seed=5, top_fraction=1.0
        )
        halved = estimate_equity(
            _MEDIUM_HOLE, _DRY_BOARD, 1, trials=300, seed=5, top_fraction=0.5
        )
        tight = estimate_equity(
            _MEDIUM_HOLE, _DRY_BOARD, 1, trials=300, seed=5, top_fraction=0.2
        )
        self.assertLessEqual(tight, halved)
        self.assertLessEqual(halved, uniform)
        # The gap is structural, not sampling noise at these settings.
        self.assertLess(tight, uniform)

    def test_more_opponents_never_raise_equity(self) -> None:
        estimates = [
            estimate_equity(("Ah", "Ad"), (), count, trials=300, seed=9)
            for count in (1, 2, 3, 4)
        ]
        self.assertEqual(estimates, sorted(estimates, reverse=True))


class EstimateEquityContractTests(unittest.TestCase):
    def test_preflop_and_postflop_board_sizes_are_accepted(self) -> None:
        for board in (
            (),
            _DRY_BOARD,
            (*_DRY_BOARD, "9h"),
            (*_DRY_BOARD, "9h", "3c"),
        ):
            equity = estimate_equity(_MEDIUM_HOLE, board, 1, trials=50, seed=4)
            self.assertGreaterEqual(equity, 0.0, msg=f"board size {len(board)}")
            self.assertLessEqual(equity, 1.0, msg=f"board size {len(board)}")

    def test_a_six_card_board_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_equity(
                _MEDIUM_HOLE,
                (*_DRY_BOARD, "9h", "3c", "4d"),
                1,
                trials=50,
                seed=4,
            )

    def test_hero_must_hold_exactly_two_cards(self) -> None:
        for hole in ((), ("Ah",), ("Ah", "Kh", "Qh")):
            with self.assertRaises(ValueError, msg=f"hole size {len(hole)}"):
                estimate_equity(hole, (), 1, trials=50, seed=4)

    def test_top_fraction_must_sit_in_the_half_open_unit_interval(self) -> None:
        for top_fraction in (0.0, -0.2, 1.5):
            with self.assertRaises(ValueError, msg=f"top_fraction {top_fraction}"):
                estimate_equity(
                    _MEDIUM_HOLE,
                    _DRY_BOARD,
                    1,
                    trials=50,
                    seed=4,
                    top_fraction=top_fraction,
                )

    def test_duplicate_known_cards_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_equity(("Qh", "Qh"), (), 1, trials=50, seed=4)
        with self.assertRaises(ValueError):
            estimate_equity(("Qh", "Jh"), ("Qh", "7d", "2s"), 1, trials=50, seed=4)

    def test_trials_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            estimate_equity(_MEDIUM_HOLE, _DRY_BOARD, 1, trials=0, seed=4)

    def test_a_table_larger_than_the_deck_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_equity(_MEDIUM_HOLE, (), 24, trials=50, seed=4)


if __name__ == "__main__":
    unittest.main()
