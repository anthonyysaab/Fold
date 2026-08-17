"""Contract tests for the Loki/Poki ``hand_potential`` decomposition.

Pins the ``(ppot, npot)`` contract before anything consumes it: the
impossible-by-construction bounds invariant (each component is a conditional
relative frequency, so [0, 1] cannot fail unless the estimator is wrong),
the preflop/river zero rule, seeded reproducibility, two boards whose
potential is known by construction, and the card-validation surface.

Trial counts are reduced from the 1000 default throughout: these tests pin
structure, not precision, and must keep the file's runtime in seconds.
"""

from __future__ import annotations

import random
import unittest

from devfun_poker_playground._vendor.treys import Card, Evaluator
from devfun_poker_playground.hand_potential import HandPotentialError, hand_potential
from devfun_poker_playground.schema3 import CARD_CODES

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_DECK = tuple(r + s for r in _RANKS for s in _SUITS)


class HandPotentialInvariantTests(unittest.TestCase):
    def test_components_are_probabilities_on_fuzzed_boards(self) -> None:
        # Impossible-by-construction: ppot and npot are each a ratio of a
        # [0, den]-bounded numerator to its own conditioning count (0.0 when
        # the condition is empty). No holding, board, seed, or trial count
        # can legitimately push either outside [0, 1] — a violation can only
        # mean the accounting is wrong.
        draw = random.Random(2026)  # fuzz seed: arbitrary, every value must pass
        for case in range(24):
            cards = draw.sample(_DECK, 6)
            board_len = 3 if case % 2 == 0 else 4  # alternate flop and turn
            hole, board = cards[:2], cards[2 : 2 + board_len]
            ppot, npot = hand_potential(hole, board, trials=40, seed=case)
            for name, value in (("ppot", ppot), ("npot", npot)):
                self.assertGreaterEqual(value, 0.0, msg=f"{name} {hole} on {board}")
                self.assertLessEqual(value, 1.0, msg=f"{name} {hole} on {board}")


class HandPotentialStreetRuleTests(unittest.TestCase):
    def test_preflop_has_zero_potential_by_definition(self) -> None:
        self.assertEqual(hand_potential(("Ah", "Ad"), ()), (0.0, 0.0))

    def test_river_has_zero_potential_by_definition(self) -> None:
        board = ("Qc", "7d", "2s", "9h", "3c")
        self.assertEqual(hand_potential(("Qh", "Jh"), board), (0.0, 0.0))


class HandPotentialKnownBoardTests(unittest.TestCase):
    def test_a_flopped_royal_flush_has_no_negative_potential(self) -> None:
        # The absolute nuts on every runout: npot is exactly 0.0, and ppot is
        # 0.0 via the empty-conditioning rule (the hero is never behind-or-tied,
        # so there is nothing to improve from).
        ppot, npot = hand_potential(("As", "Ks"), ("Qs", "Js", "Ts"), trials=120, seed=3)
        self.assertEqual(npot, 0.0)
        self.assertEqual(ppot, 0.0)

    def test_a_nut_flush_draw_has_real_positive_potential(self) -> None:
        # Nine flush outs twice plus over-card outs, conditional on being
        # behind: well above the 0.15 floor at any seed worth its salt.
        ppot, _ = hand_potential(("Ah", "Kh"), ("Qh", "7h", "2s"), trials=300, seed=7)
        self.assertGreater(ppot, 0.15)


class ShowdownTieAccountingTests(unittest.TestCase):
    """F7: the documented probability statement must match the code.

    The docstring states that ties on the current board condition *both*
    components and that a showdown tie credits 0.5 to the favorable
    outcome. The board here — quad treys with hero holding the two
    remaining deuces — makes that pinnable: hero's kicker (max of a deuce
    and the river) can never beat an opponent's (max of their cards and
    the river), so hero never *wins* a showdown and is behind-or-tied now
    in every sample. Every favorable ppot outcome is therefore a showdown
    tie, and ppot must equal exactly half the tie rate.
    """

    HOLE = ("2c", "2d")
    BOARD = ("3c", "3d", "3h", "3s")
    TRIALS = 400
    SEED = 11  # arbitrary but fixed, like every seed in this file

    def _tie_rate_oracle(self) -> float:
        """Showdown tie rate over the exact draws the estimator makes.

        Replays the estimator's sampling — same deck order (the schema's
        ``CARD_CODES``, which ``hand_potential`` now builds from), same
        ``random.Random(seed)``, one 3-card ``sample`` per trial — and
        scores ties independently with the vendored evaluator.
        """

        evaluator = Evaluator()
        hole = [Card.new(code) for code in self.HOLE]
        board = [Card.new(code) for code in self.BOARD]
        dead = set(hole) | set(board)
        unseen = [
            Card.new(code)
            for code in CARD_CODES
            if Card.new(code) not in dead
        ]
        rng = random.Random(self.SEED)
        ties = 0
        for _ in range(self.TRIALS):
            drawn = rng.sample(unseen, 3)
            runout = board + drawn[2:]
            if evaluator.evaluate(runout, hole) == evaluator.evaluate(
                runout, drawn[:2]
            ):
                ties += 1
        self.assertGreater(ties, 0)  # non-vacuous: ties must occur here
        self.assertLess(ties, self.TRIALS)  # and losses must occur too
        return ties / self.TRIALS

    def test_ppot_is_exactly_half_the_showdown_tie_rate(self) -> None:
        ppot, _ = hand_potential(
            self.HOLE, self.BOARD, trials=self.TRIALS, seed=self.SEED
        )
        self.assertEqual(ppot, 0.5 * self._tie_rate_oracle())

    def test_npot_charges_half_a_loss_on_a_tie(self) -> None:
        # Mirror pin: hero conditions into npot only through ties-now
        # (never ahead), and a showdown tie then charges exactly 0.5.
        # Hero ties now only against the case-deuces opponent, whose
        # showdown is always another tie -> npot is 0.5 when that combo
        # is sampled, 0.0 (empty conditioning) when it is not.
        _, npot = hand_potential(
            self.HOLE, self.BOARD, trials=self.TRIALS, seed=self.SEED
        )
        self.assertIn(npot, (0.0, 0.5))


class HandPotentialDeterminismTests(unittest.TestCase):
    def test_same_seed_reproduces_the_pair_exactly(self) -> None:
        first = hand_potential(("Qh", "Jh"), ("Qc", "7d", "2s"), trials=150, seed=11)
        second = hand_potential(("Qh", "Jh"), ("Qc", "7d", "2s"), trials=150, seed=11)
        self.assertEqual(first, second)

    def test_a_different_seed_draws_a_different_sample(self) -> None:
        first = hand_potential(("Qh", "Jh"), ("Qc", "7d", "2s"), trials=150, seed=11)
        other = hand_potential(("Qh", "Jh"), ("Qc", "7d", "2s"), trials=150, seed=12)
        self.assertNotEqual(first, other)


class HandPotentialValidationTests(unittest.TestCase):
    def test_the_error_type_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(HandPotentialError, ValueError))

    def test_malformed_and_duplicate_cards_are_rejected(self) -> None:
        bad_calls = (
            (("Xs", "Kd"), ("Qc", "7d", "2s")),  # bad rank
            (("As", "Kx"), ("Qc", "7d", "2s")),  # bad suit
            (("As", "As"), ("Qc", "7d", "2s")),  # duplicate within hole
            (("As", "Kd"), ("Qc", "Qc", "2s")),  # duplicate within board
            (("As", "Kd"), ("As", "7d", "2s")),  # hole/board overlap
            (("As",), ("Qc", "7d", "2s")),  # one-card holding
            (("As", "Kd", "Qd"), ("Qc", "7d", "2s")),  # three-card holding
            (("As", "Kd"), ("Qc", "7d")),  # two-card board
            (("As", "Kd"), ("Qc", "7d", "2s", "9h", "3c", "4c")),  # six-card board
        )
        for hole, board in bad_calls:
            with self.assertRaises(HandPotentialError, msg=f"{hole} on {board}"):
                hand_potential(hole, board, trials=10, seed=1)

    def test_a_non_positive_trial_count_is_rejected(self) -> None:
        with self.assertRaises(HandPotentialError):
            hand_potential(("Qh", "Jh"), ("Qc", "7d", "2s"), trials=0, seed=1)


if __name__ == "__main__":
    unittest.main()
