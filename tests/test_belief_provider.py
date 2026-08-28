"""Tests for the Tier-2 belief-bucket plumbing.

The impossible-by-construction invariant: the neutral provider's output is
fed straight back through ``require_buckets`` and checked bucket-by-bucket
that each entry's complement equals the mass of the other buckets — a
property no valid distribution can violate, so a failure here means the
code (not the example) is wrong. Everything else is boundary coverage for
the validator: wrong length, negatives, NaN/inf, off-sum, non-numeric.

No randomness is used or needed — the module under test is constant by
construction, and the tests assert that repeat calls are identical.
"""

from __future__ import annotations

import math
import unittest

from engine.belief_provider import (
    BeliefProvider,
    BeliefProviderError,
    NeutralBeliefProvider,
    require_buckets,
)
from engine.schema3 import BELIEF_BUCKETS

# Derived: half a bucket of mass, unambiguously beyond the module's 1e-6
# sum tolerance, used to build off-sum and negative counterexamples.
_GROSS_ERROR = 0.5 / BELIEF_BUCKETS


def _uniform() -> tuple[float, ...]:
    # Derived from the schema, never hand-written as 0.125s.
    return tuple(1.0 / BELIEF_BUCKETS for _ in range(BELIEF_BUCKETS))


class NeutralProviderTest(unittest.TestCase):
    def test_uniform_output_validates_and_sums_to_one(self) -> None:
        provider = NeutralBeliefProvider()
        buckets = provider.continuing_range_buckets({}, [])

        # Impossible-by-construction invariant: the output re-validates,
        # every entry is a probability, and each entry's complement equals
        # the mass of the remaining buckets. 1/8 is a binary power, so
        # both sums are exact — any tolerance needed here is a bug.
        self.assertEqual(require_buckets(buckets), buckets)
        self.assertEqual(len(buckets), BELIEF_BUCKETS)
        self.assertEqual(math.fsum(buckets), 1.0)
        for index, value in enumerate(buckets):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
            others = math.fsum(
                other for position, other in enumerate(buckets) if position != index
            )
            self.assertEqual(1.0 - value, others)

    def test_output_is_uniform_and_input_independent(self) -> None:
        provider = NeutralBeliefProvider()
        baseline = provider.continuing_range_buckets({}, [])
        self.assertEqual(baseline, _uniform())
        # Neutrality: conditioning inputs must not change the answer, and
        # repeat calls are identical (no hidden state, no RNG).
        loaded = provider.continuing_range_buckets(
            {"pot_bb": 40.0, "players": 5},
            [{"action": "raise", "amount_bb": 12.0}],
        )
        self.assertEqual(loaded, baseline)
        self.assertEqual(provider.continuing_range_buckets({}, []), baseline)

    def test_protocol_is_runtime_checkable(self) -> None:
        self.assertIsInstance(NeutralBeliefProvider(), BeliefProvider)

        class NotAProvider:
            pass

        self.assertNotIsInstance(NotAProvider(), BeliefProvider)


class RequireBucketsTest(unittest.TestCase):
    def test_returns_input_unchanged(self) -> None:
        buckets = _uniform()
        result = require_buckets(buckets)
        self.assertIs(result, buckets)  # tuple in, same tuple out
        as_list = list(buckets)
        self.assertEqual(require_buckets(as_list), buckets)

    def test_accepts_non_uniform_distribution_within_tolerance(self) -> None:
        skewed = [0.0] * BELIEF_BUCKETS
        skewed[0] = 0.5
        skewed[-1] = 0.5
        self.assertEqual(require_buckets(skewed), tuple(skewed))

    def test_rejects_wrong_length(self) -> None:
        for length in (0, BELIEF_BUCKETS - 1, BELIEF_BUCKETS + 1):
            with self.assertRaises(BeliefProviderError):
                require_buckets([1.0 / max(length, 1)] * length)

    def test_rejects_negative_entry(self) -> None:
        buckets = list(_uniform())
        buckets[3] -= _GROSS_ERROR + 1.0 / BELIEF_BUCKETS  # now negative
        buckets[4] += _GROSS_ERROR + 1.0 / BELIEF_BUCKETS  # sum still 1.0
        with self.assertRaises(BeliefProviderError):
            require_buckets(buckets)

    def test_rejects_nan_and_inf(self) -> None:
        for poison in (math.nan, math.inf, -math.inf):
            buckets = list(_uniform())
            buckets[0] = poison
            with self.assertRaises(BeliefProviderError):
                require_buckets(buckets)

    def test_rejects_off_sum(self) -> None:
        over = list(_uniform())
        over[0] += _GROSS_ERROR
        under = list(_uniform())
        under[0] -= _GROSS_ERROR
        for bad in (over, under):
            with self.assertRaises(BeliefProviderError):
                require_buckets(bad)

    def test_rejects_non_numeric_entry(self) -> None:
        for poison in (None, "an-eighth", object()):
            buckets: list[object] = list(_uniform())
            buckets[2] = poison
            with self.assertRaises(BeliefProviderError):
                require_buckets(buckets)  # type: ignore[arg-type]

    def test_error_is_a_value_error(self) -> None:
        # Callers that already catch ValueError keep working.
        self.assertTrue(issubclass(BeliefProviderError, ValueError))


if __name__ == "__main__":
    unittest.main()
