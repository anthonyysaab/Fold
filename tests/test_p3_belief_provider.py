"""The fitted P3 belief provider (block 8 wired, owner 2026-08-30).

Directional expectations lean on the fit's own CHECKED sign invariant
(better hands fold less, worse prices fold more — enforced at load, so a
sign-broken artifact cannot reach these tests): after a priced continue,
mass must move toward the top octiles, and a bigger price must move more.
Everything else is independent construction: the uniform no-evidence
case is the mathematical prior, not a mirrored constant.
"""

from __future__ import annotations

import math
import unittest

from engine.belief_provider import NeutralBeliefProvider, require_buckets
from engine.p3_belief_provider import P3BeliefProvider
from engine.schema3 import BELIEF_BUCKETS


def _table(**overrides) -> dict:
    base = {
        "selfSeatNumber": 1,
        "boardCards": ["Qs", "7d", "2c"],
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "seats": [
            {"seatNumber": 1, "status": "Active"},
            {"seatNumber": 2, "status": "Active"},
            {"seatNumber": 3, "status": "Folded"},
        ],
    }
    base.update(overrides)
    return base


def _bet_call(amount: int, caller: int = 2) -> list[dict]:
    return [
        {"street": "flop", "seat_number": 1, "action": "bet", "amount": amount},
        {"street": "flop", "seat_number": caller, "action": "call", "amount": amount},
    ]


class P3BeliefProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provider = P3BeliefProvider.from_artifact()

    def test_no_evidence_is_exactly_the_neutral_prior(self) -> None:
        buckets = self.provider.continuing_range_buckets(_table(), [])
        neutral = NeutralBeliefProvider().continuing_range_buckets(_table(), [])
        self.assertEqual(tuple(buckets), tuple(neutral))
        self.assertIsNone(self.provider.last_degrade_reason)

    def test_output_is_a_valid_distribution(self) -> None:
        buckets = self.provider.continuing_range_buckets(_table(), _bet_call(400))
        require_buckets(buckets)  # the extractor's own gate
        self.assertEqual(len(buckets), BELIEF_BUCKETS)
        self.assertAlmostEqual(sum(buckets), 1.0)
        self.assertTrue(all(value >= 0.0 for value in buckets))

    def test_a_priced_continue_tilts_toward_strength(self) -> None:
        buckets = self.provider.continuing_range_buckets(_table(), _bet_call(400))
        self.assertGreater(buckets[-1], 1.0 / BELIEF_BUCKETS)
        self.assertLess(buckets[0], 1.0 / BELIEF_BUCKETS)
        # Monotone across the octiles — the fit's sign invariant, surfaced.
        for lower, upper in zip(buckets, buckets[1:]):
            self.assertLessEqual(lower, upper)

    def test_bigger_price_tilts_harder(self) -> None:
        big = self.provider.continuing_range_buckets(_table(), _bet_call(400))
        small = self.provider.continuing_range_buckets(_table(), _bet_call(50))
        self.assertGreater(big[-1], small[-1])
        self.assertLess(big[0], small[0])

    def test_two_continues_tilt_harder_than_one(self) -> None:
        one = self.provider.continuing_range_buckets(_table(), _bet_call(200))
        two_events = _bet_call(200) + [
            {"street": "turn", "seat_number": 1, "action": "bet", "amount": 300},
            {"street": "turn", "seat_number": 2, "action": "call", "amount": 300},
        ]
        table = _table(boardCards=["Qs", "7d", "2c", "3h"])
        two = self.provider.continuing_range_buckets(table, two_events)
        self.assertGreater(two[-1], one[-1])

    def test_folded_and_hero_seats_carry_no_evidence(self) -> None:
        # Seat 3 is Folded and seat 1 is hero: neither's call conditions.
        uniform = 1.0 / BELIEF_BUCKETS
        for caller in (1, 3):
            records = [
                {"street": "flop", "seat_number": 2, "action": "bet", "amount": 400},
                {"street": "flop", "seat_number": caller, "action": "call",
                 "amount": 400},
            ]
            buckets = self.provider.continuing_range_buckets(_table(), records)
            self.assertTrue(
                all(math.isclose(value, uniform) for value in buckets),
                f"caller {caller} must not condition the range",
            )

    def test_free_actions_carry_no_evidence(self) -> None:
        records = [
            {"street": "flop", "seat_number": 2, "action": "check"},
            {"street": "flop", "seat_number": 1, "action": "check"},
        ]
        buckets = self.provider.continuing_range_buckets(_table(), records)
        self.assertTrue(all(math.isclose(b, 1.0 / BELIEF_BUCKETS) for b in buckets))

    def test_deterministic(self) -> None:
        first = self.provider.continuing_range_buckets(_table(), _bet_call(400))
        second = self.provider.continuing_range_buckets(_table(), _bet_call(400))
        self.assertEqual(tuple(first), tuple(second))

    def test_malformed_state_degrades_to_uniform_with_a_reason(self) -> None:
        buckets = self.provider.continuing_range_buckets(
            {"seats": object()}, [{"street": "flop"}]
        )
        self.assertTrue(all(math.isclose(b, 1.0 / BELIEF_BUCKETS) for b in buckets))

    def test_v9_policy_defaults_to_the_fitted_provider(self) -> None:
        import tempfile
        from pathlib import Path

        from test_learned_policy_v9 import _write_artifact
        from engine.learned_policy_v9 import load_policy_v9

        with tempfile.TemporaryDirectory() as raw:
            policy = load_policy_v9(_write_artifact(Path(raw)))
        self.assertIsInstance(policy._belief_provider, P3BeliefProvider)


if __name__ == "__main__":
    unittest.main()
