"""Contract tests for schema 3 — the v8 feature layout must not drift.

``engine/schema3.py`` is the single coordination point for
the v8 feature vector; every producer and consumer imports the layout from
it, so the only way to catch accidental drift is a test that pins the frozen
contract from outside the module. A failure here is a schema event, not a
bug to patch around.

Two kinds of assertion, deliberately mixed:

- **Restated pins** — block sizes, card-code order, plane order. These are
  restated on purpose (the one place in the repo allowed to): a test that
  merely compared two modules would stay green through a coordinated edit
  of both.
- **Impossible-by-construction invariants** — name uniqueness (set size
  equals tuple length), the card/context index partition of
  ``range(INPUT_SIZE_V8)``, and full ``feature_index`` round-trips. These
  cannot fail unless the module is wrong, per the house rule that a probe
  must be verified against such an invariant before its answer is believed.
"""

from __future__ import annotations

import unittest

from engine import policy_features, schema3
from engine.schema3 import (
    BELIEF_BUCKETS,
    BELIEF_FEATURE_NAMES,
    CARD_BLOCK_SIZE,
    CARD_CODES,
    CARD_FEATURE_NAMES,
    CARD_INDICES,
    CARD_PLANES,
    CONTEXT_BLOCK_SIZE,
    CONTEXT_FEATURE_NAMES,
    CONTEXT_INDICES,
    FEATURE_NAMES_V8,
    HISTORY_CHANNELS,
    HISTORY_FEATURE_NAMES,
    HISTORY_SLOTS,
    HISTORY_STREETS,
    INPUT_SIZE_V8,
    Schema3Error,
    feature_index,
    history_slot_index,
    require_vector,
)

# Frozen schema-3 contract sizes (V8_DESIGN section 3). Not ablatable:
# changing any of them is a schema event that invalidates every v8 artifact.
EXPECTED_INPUT_SIZE = 413
EXPECTED_CARD_BLOCK = 208
EXPECTED_CONTEXT_BLOCK = 205


class TestFrozenSizes(unittest.TestCase):
    def test_block_sizes_are_the_frozen_contract(self) -> None:
        self.assertEqual(INPUT_SIZE_V8, EXPECTED_INPUT_SIZE)
        self.assertEqual(CARD_BLOCK_SIZE, EXPECTED_CARD_BLOCK)
        self.assertEqual(CONTEXT_BLOCK_SIZE, EXPECTED_CONTEXT_BLOCK)

    def test_sizes_agree_with_the_name_tuples(self) -> None:
        # Impossible-by-construction unless a size constant was hand-edited
        # away from the tuple it is defined from.
        self.assertEqual(len(FEATURE_NAMES_V8), INPUT_SIZE_V8)
        self.assertEqual(len(CARD_FEATURE_NAMES), CARD_BLOCK_SIZE)
        self.assertEqual(len(CONTEXT_FEATURE_NAMES), CONTEXT_BLOCK_SIZE)
        self.assertEqual(CARD_BLOCK_SIZE + CONTEXT_BLOCK_SIZE, INPUT_SIZE_V8)


class TestNameUniqueness(unittest.TestCase):
    def test_feature_names_are_unique(self) -> None:
        # Impossible-by-construction: a set can only be smaller than the
        # tuple it was built from if the tuple repeats a name.
        self.assertEqual(len(set(FEATURE_NAMES_V8)), len(FEATURE_NAMES_V8))


class TestCardBlock(unittest.TestCase):
    def test_card_codes_are_rank_major_cdhs(self) -> None:
        # Restated pin: rank-major, suits cdhs within each rank. This is the
        # v6/v7 card order the entire telemetry corpus was encoded with.
        expected = tuple(
            f"{rank}{suit}" for rank in "23456789TJQKA" for suit in "cdhs"
        )
        self.assertEqual(CARD_CODES, expected)

    def test_card_codes_match_policy_features_order(self) -> None:
        # Cross-module check against the public v6/v7 feature contract: its
        # vector opens with one ``hole_<code>`` per card in card order.
        hole_names = policy_features.FEATURE_NAMES[: len(CARD_CODES)]
        for name in hole_names:
            self.assertTrue(name.startswith("hole_"), name)
        legacy_codes = tuple(name.removeprefix("hole_") for name in hole_names)
        self.assertEqual(CARD_CODES, legacy_codes)

    def test_card_block_is_planes_by_codes(self) -> None:
        self.assertEqual(CARD_PLANES, ("hole", "flop", "turn", "river"))
        expected = tuple(
            f"{plane}_{code}" for plane in CARD_PLANES for code in CARD_CODES
        )
        self.assertEqual(CARD_FEATURE_NAMES, expected)

    def test_card_block_leads_the_vector(self) -> None:
        self.assertEqual(FEATURE_NAMES_V8[:CARD_BLOCK_SIZE], CARD_FEATURE_NAMES)
        self.assertEqual(FEATURE_NAMES_V8[CARD_BLOCK_SIZE:], CONTEXT_FEATURE_NAMES)


class TestIndexPartition(unittest.TestCase):
    def test_card_and_context_indices_partition_the_vector(self) -> None:
        # Impossible-by-construction: two blocks partition range(N) iff they
        # are disjoint, duplicate-free, and their union is exactly range(N).
        card = set(CARD_INDICES)
        context = set(CONTEXT_INDICES)
        self.assertEqual(len(card), len(CARD_INDICES))
        self.assertEqual(len(context), len(CONTEXT_INDICES))
        self.assertTrue(card.isdisjoint(context))
        self.assertEqual(card | context, set(range(INPUT_SIZE_V8)))


class TestFeatureIndex(unittest.TestCase):
    def test_round_trips_every_name(self) -> None:
        # Impossible-by-construction: enumerate() cannot disagree with a
        # correct name->index map.
        for index, name in enumerate(FEATURE_NAMES_V8):
            self.assertEqual(feature_index(name), index)

    def test_unknown_name_is_a_contract_error(self) -> None:
        with self.assertRaises(Schema3Error):
            feature_index("no_such_feature")
        with self.assertRaises(Schema3Error):
            feature_index("hole_XX")


class TestHistoryLayout(unittest.TestCase):
    def test_slot_index_matches_feature_index(self) -> None:
        self.assertEqual(
            history_slot_index("flop", 5, "size"),
            feature_index("hist_flop5_size"),
        )

    def test_history_size_is_derived_from_its_axes(self) -> None:
        # Derived, never restated: streets x (slots x channels + overflow).
        per_street = HISTORY_SLOTS * len(HISTORY_CHANNELS) + 1
        self.assertEqual(
            len(HISTORY_FEATURE_NAMES), len(HISTORY_STREETS) * per_street
        )

    def test_each_street_block_ends_with_its_overflow_flag(self) -> None:
        for street in HISTORY_STREETS:
            last_slot = history_slot_index(
                street, HISTORY_SLOTS - 1, HISTORY_CHANNELS[-1]
            )
            self.assertEqual(
                feature_index(f"hist_{street}_overflow"), last_slot + 1
            )

    def test_invalid_coordinates_are_contract_errors(self) -> None:
        with self.assertRaises(Schema3Error):
            history_slot_index("showdown", 0, "size")
        with self.assertRaises(Schema3Error):
            history_slot_index("flop", -1, "size")
        with self.assertRaises(Schema3Error):
            history_slot_index("flop", HISTORY_SLOTS, "size")
        with self.assertRaises(Schema3Error):
            history_slot_index("flop", 0, "wager")


class TestRequireVector(unittest.TestCase):
    def test_correct_length_passes(self) -> None:
        self.assertIsNone(require_vector([0.0] * INPUT_SIZE_V8))
        self.assertIsNone(require_vector(tuple([0.0] * INPUT_SIZE_V8)))

    def test_wrong_lengths_are_contract_errors(self) -> None:
        # Off-by-one in both directions plus empty, derived from the size.
        for length in (0, INPUT_SIZE_V8 - 1, INPUT_SIZE_V8 + 1):
            with self.assertRaises(Schema3Error):
                require_vector([0.0] * length)


class TestBeliefTail(unittest.TestCase):
    def test_belief_buckets_close_the_vector(self) -> None:
        self.assertEqual(len(BELIEF_FEATURE_NAMES), BELIEF_BUCKETS)
        # Restated pin: 8 price-conditioned range buckets (V8_DESIGN, Tier 2).
        self.assertEqual(BELIEF_BUCKETS, 8)
        self.assertEqual(
            FEATURE_NAMES_V8[-len(BELIEF_FEATURE_NAMES):], BELIEF_FEATURE_NAMES
        )


class TestDormantBlock(unittest.TestCase):
    """The nine format-gated features reserved for heads-up / tournament play.

    They are dormant because the Playground reseats every hand, not because
    they carry nothing. These tests pin the property that makes reactivation
    safe: appending cannot disturb anything that already exists.
    """

    def test_the_nine_dormant_names_are_exactly_the_schema_2_dropouts(self) -> None:
        self.assertEqual(
            set(schema3.DORMANT_FEATURE_NAMES),
            {
                "opponent_range_width",
                "opponent_max_wildness",
                "opponent_max_stickiness",
                "opponent_aggression_count",
                "risk_temperature",
                "risk_weakness",
                "risk_bet_pressure",
                "risk_distance_from_river",
                "risk_player_pressure",
            },
        )
        self.assertEqual(len(schema3.DORMANT_FEATURE_NAMES), 9)

    def test_every_dormant_name_still_exists_in_schema_2(self) -> None:
        # The whole point of parking rather than deleting: the engine still
        # computes these, so reactivation needs no code recovered.
        from engine.learning_contract import LEARNING_FEATURE_NAMES

        for name in schema3.DORMANT_FEATURE_NAMES:
            self.assertIn(name, LEARNING_FEATURE_NAMES, name)

    def test_dormant_names_are_not_active(self) -> None:
        for name in schema3.DORMANT_FEATURE_NAMES:
            self.assertNotIn(name, FEATURE_NAMES_V8, name)

    def test_enabling_cannot_shift_an_existing_index(self) -> None:
        # The impossible-by-construction guarantee: the extended contract is
        # the active one with names APPENDED, so every active index is
        # preserved. If this fails, enabling the block would silently
        # invalidate every artifact and corpus built against schema 3.
        extended = schema3.FEATURE_NAMES_V8_EXTENDED
        self.assertEqual(extended[: len(FEATURE_NAMES_V8)], FEATURE_NAMES_V8)
        for index, name in enumerate(FEATURE_NAMES_V8):
            self.assertEqual(extended.index(name), index, name)
        self.assertEqual(
            schema3.INPUT_SIZE_V8_EXTENDED, INPUT_SIZE_V8 + 9
        )

    def test_extended_names_are_unique(self) -> None:
        extended = schema3.FEATURE_NAMES_V8_EXTENDED
        self.assertEqual(len(set(extended)), len(extended))

    def test_feature_names_selector(self) -> None:
        self.assertEqual(schema3.feature_names(), FEATURE_NAMES_V8)
        self.assertEqual(
            schema3.feature_names(include_dormant=True),
            schema3.FEATURE_NAMES_V8_EXTENDED,
        )

    def test_dormant_lookup_explains_itself(self) -> None:
        # A dormant name must not read as a typo -- the error has to say it is
        # reserved and how to enable it.
        with self.assertRaises(Schema3Error) as caught:
            schema3.feature_index("opponent_range_width")
        message = str(caught.exception)
        self.assertIn("DORMANT", message)
        self.assertIn("include_dormant", message)
        # An actual typo still reads as unknown, not dormant.
        with self.assertRaises(Schema3Error) as caught:
            schema3.feature_index("opponent_range_widht")
        self.assertIn("unknown", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
