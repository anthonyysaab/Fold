"""Behavioural and invariant checks for the schema-3 history encoder.

Positions are derived here from the public schema3 index API, independently
of the encoder's own name-map, so a placement bug on either side breaks
these tests rather than cancelling out. The fuzz test carries the
impossible-by-construction invariants: within each street the occurred
flags are monotone non-decreasing across slots (right alignment forbids
gaps), every occupied slot has exactly one family flag, and every empty
slot is all-zero.
"""

from __future__ import annotations

import math
import random
import unittest

from devfun_poker_playground import schema3
from devfun_poker_playground.action_history import encode_action_history
from devfun_poker_playground.training_telemetry import action_family

_FAMILY_CHANNELS = ("fold", "check_call", "aggress")
_HISTORY_BASE = schema3.feature_index(schema3.HISTORY_FEATURE_NAMES[0])


def _pos(street: str, slot: int, channel: str) -> int:
    """Block position of one slot channel, via the schema3 index API."""

    return schema3.history_slot_index(street, slot, channel) - _HISTORY_BASE


def _overflow_pos(street: str) -> int:
    return schema3.HISTORY_FEATURE_NAMES.index(f"hist_{street}_overflow")


def _record(
    street: str = "flop",
    seat: int | None = 3,
    action: str = "raise",
    amount: int | None = 40,
    *,
    omit_amount: bool = False,
) -> dict[str, object]:
    """A record shaped like training_telemetry._recent_actions output."""

    record: dict[str, object] = {
        "street": street,
        "seat_number": seat,
        "action": action,
    }
    if not omit_amount:
        record["amount"] = amount
    return record


def _encode(
    records: list[dict[str, object]],
    *,
    hero: int = 1,
    players: int = 6,
    big_blind: int = 20,
    seat_numbers: tuple[int, ...] | None = None,
) -> tuple[float, ...]:
    return encode_action_history(
        records,
        hero_seat_number=hero,
        player_count=players,
        big_blind_chips=big_blind,
        seat_numbers=seat_numbers,
    )


class LayoutDerivationTest(unittest.TestCase):
    def test_history_block_is_contiguous_in_the_full_vector(self) -> None:
        # _pos above subtracts the block base from full-vector indices;
        # that is only sound if the history features are contiguous.
        indices = [
            schema3.feature_index(name)
            for name in schema3.HISTORY_FEATURE_NAMES
        ]
        self.assertEqual(
            indices,
            list(
                range(
                    _HISTORY_BASE,
                    _HISTORY_BASE + len(schema3.HISTORY_FEATURE_NAMES),
                )
            ),
        )


class EncodeActionHistoryTest(unittest.TestCase):
    def test_length_and_finiteness(self) -> None:
        vector = _encode([_record(), _record(street="preflop", action="call")])
        self.assertEqual(len(vector), len(schema3.HISTORY_FEATURE_NAMES))
        self.assertEqual(len(vector), 148)
        for value in vector:
            self.assertTrue(math.isfinite(value))

    def test_empty_input_is_all_zeros(self) -> None:
        vector = _encode([])
        self.assertEqual(
            vector, (0.0,) * len(schema3.HISTORY_FEATURE_NAMES)
        )

    def test_single_action_right_aligns_to_last_slot(self) -> None:
        vector = _encode([_record(action="bet")])
        last = schema3.HISTORY_SLOTS - 1
        self.assertEqual(vector[_pos("flop", last, "occurred")], 1.0)
        self.assertEqual(vector[_pos("flop", last, "aggress")], 1.0)
        for slot in range(last):
            self.assertEqual(vector[_pos("flop", slot, "occurred")], 0.0)
        self.assertEqual(vector[_overflow_pos("flop")], 0.0)

    def test_two_actions_keep_order_most_recent_last(self) -> None:
        vector = _encode(
            [
                _record(action="bet", seat=2),
                _record(action="call", seat=3),
            ]
        )
        last = schema3.HISTORY_SLOTS - 1
        self.assertEqual(vector[_pos("flop", last - 1, "aggress")], 1.0)
        self.assertEqual(vector[_pos("flop", last, "check_call")], 1.0)

    def test_overflow_fires_only_above_slot_count(self) -> None:
        exactly_full = [
            _record(action="bet", amount=10 * (index + 1))
            for index in range(schema3.HISTORY_SLOTS)
        ]
        vector = _encode(exactly_full)
        self.assertEqual(vector[_overflow_pos("flop")], 0.0)
        self.assertEqual(vector[_pos("flop", 0, "occurred")], 1.0)

        one_over = [
            _record(action="bet", amount=10 * (index + 1))
            for index in range(schema3.HISTORY_SLOTS + 1)
        ]
        vector = _encode(one_over, big_blind=1)
        self.assertEqual(vector[_overflow_pos("flop")], 1.0)
        # The last HISTORY_SLOTS records survive: slot 0 now holds the
        # second record (amount 20), not the truncated first (amount 10).
        self.assertEqual(
            vector[_pos("flop", 0, "size")], math.log1p(20.0)
        )
        # Only flop overflowed.
        for street in schema3.HISTORY_STREETS:
            if street != "flop":
                self.assertEqual(vector[_overflow_pos(street)], 0.0)

    def test_hero_actor_channel_is_zero(self) -> None:
        vector = _encode([_record(seat=4)], hero=4)
        last = schema3.HISTORY_SLOTS - 1
        self.assertEqual(vector[_pos("flop", last, "actor")], 0.0)

    def test_actor_unit_scales_and_wraps(self) -> None:
        last = schema3.HISTORY_SLOTS - 1
        vector = _encode([_record(seat=3)], hero=2, players=6)
        self.assertEqual(vector[_pos("flop", last, "actor")], 1.0 / 5.0)
        # Wrap: seat 1 is two seats clockwise of hero 5 at a 6-max table.
        vector = _encode([_record(seat=1)], hero=5, players=6)
        self.assertEqual(vector[_pos("flop", last, "actor")], 2.0 / 5.0)

    def test_unknown_seat_reads_midpoint(self) -> None:
        vector = _encode([_record(seat=None)])
        last = schema3.HISTORY_SLOTS - 1
        self.assertEqual(vector[_pos("flop", last, "actor")], 0.5)

    def test_size_is_log1p_of_big_blinds(self) -> None:
        last = schema3.HISTORY_SLOTS - 1
        vector = _encode([_record(amount=40)], big_blind=20)
        self.assertEqual(
            vector[_pos("flop", last, "size")], math.log1p(2.0)
        )
        for absent in (
            _record(omit_amount=True),
            _record(amount=None),
        ):
            vector = _encode([absent])
            self.assertEqual(vector[_pos("flop", last, "size")], 0.0)

    def test_familyless_action_skipped_without_consuming_a_slot(self) -> None:
        # A record whose action has no family must not right-shift real
        # actions: "post" is skipped entirely, so "bet" still lands last.
        self.assertIsNone(action_family("post"))
        vector = _encode(
            [_record(action="post"), _record(action="bet")]
        )
        last = schema3.HISTORY_SLOTS - 1
        self.assertEqual(vector[_pos("flop", last, "aggress")], 1.0)
        self.assertEqual(vector[_pos("flop", last - 1, "occurred")], 0.0)

    def test_unknown_street_ignored(self) -> None:
        vector = _encode(
            [_record(street="showdown"), _record(street="")]
        )
        self.assertEqual(
            vector, (0.0,) * len(schema3.HISTORY_FEATURE_NAMES)
        )

    def test_big_blind_below_one_rejected(self) -> None:
        for bad in (0, -20):
            with self.assertRaises(schema3.Schema3Error):
                _encode([_record()], big_blind=bad)

    def test_player_count_below_one_rejected(self) -> None:
        with self.assertRaises(schema3.Schema3Error):
            _encode([_record()], players=0)


class SparseSeatActorUnitTest(unittest.TestCase):
    """F1: sparse seat numbers must not alias onto hero's own 0.0.

    The adversarial review's exact probe: seats {1, 4, 5} with hero in
    seat 4. Under the old modular arithmetic, seat 1 read
    ``(1 - 4) % 3 == 0`` — hero's own value. With ``seat_numbers`` the
    unit is the ordinal distance in the sorted seat set.
    """

    def test_review_probe_seats_1_4_5_hero_4(self) -> None:
        vector = _encode(
            [
                _record(seat=1, action="bet"),
                _record(seat=5, action="call"),
                _record(seat=4, action="raise"),
            ],
            hero=4,
            players=3,
            seat_numbers=(1, 4, 5),
        )
        last = schema3.HISTORY_SLOTS - 1
        # sorted set [1, 4, 5]: idx(1)=0, idx(4)=1, idx(5)=2, n=3.
        self.assertEqual(vector[_pos("flop", last - 2, "actor")], 1.0)  # seat 1
        self.assertEqual(vector[_pos("flop", last - 1, "actor")], 0.5)  # seat 5
        self.assertEqual(vector[_pos("flop", last, "actor")], 0.0)  # hero

    def test_none_seat_numbers_keeps_the_documented_modular_fallback(self) -> None:
        # Without seat_numbers the aliasing behavior is the documented
        # fallback: seat 1 reads hero's 0.0 at a sparse 3-seat table.
        vector = _encode([_record(seat=1)], hero=4, players=3)
        last = schema3.HISTORY_SLOTS - 1
        self.assertEqual(vector[_pos("flop", last, "actor")], 0.0)

    def test_seat_outside_the_set_reads_the_midpoint(self) -> None:
        last = schema3.HISTORY_SLOTS - 1
        vector = _encode(
            [_record(seat=9)], hero=4, players=3, seat_numbers=(1, 4, 5)
        )
        self.assertEqual(vector[_pos("flop", last, "actor")], 0.5)
        vector = _encode(
            [_record(seat=None)], hero=4, players=3, seat_numbers=(1, 4, 5)
        )
        self.assertEqual(vector[_pos("flop", last, "actor")], 0.5)

    def test_dense_seats_agree_with_the_modular_fallback(self) -> None:
        # On dense consecutive seats the two definitions coincide, so
        # passing seat_numbers can never change an already-correct table.
        records = [_record(seat=seat, action="bet") for seat in (1, 2, 3)]
        with_set = _encode(
            records, hero=2, players=3, seat_numbers=(1, 2, 3)
        )
        without = _encode(records, hero=2, players=3)
        self.assertEqual(with_set, without)


class FuzzInvariantTest(unittest.TestCase):
    """Impossible-by-construction invariants over seeded random inputs."""

    # Arbitrary fixed seed: every seed must pass; fixed for reproducibility.
    SEED = 0x5EED_AC71
    # Ablatable: trial count trades coverage against runtime; 200 keeps the
    # test well under a second while exercising every street/overflow mix.
    TRIALS = 200

    def _random_records(
        self, rng: random.Random
    ) -> list[dict[str, object]]:
        streets = (*schema3.HISTORY_STREETS, "showdown", "")
        actions = (
            "fold",
            "check",
            "call",
            "bet",
            "raise",
            "all-in",
            "post",
            "show",
            "deal",
        )
        records: list[dict[str, object]] = []
        for _ in range(rng.randint(0, 40)):
            record = _record(
                street=rng.choice(streets),
                seat=rng.choice((None, rng.randint(1, 9))),
                action=rng.choice(actions),
                amount=rng.randint(0, 5_000),
                omit_amount=rng.random() < 0.3,
            )
            records.append(record)
        return records

    def test_fuzzed_inputs_keep_structural_invariants(self) -> None:
        rng = random.Random(self.SEED)
        for _ in range(self.TRIALS):
            players = rng.randint(1, 9)
            hero = rng.randint(1, 9)
            big_blind = rng.choice((1, 20, 100))
            records = self._random_records(rng)
            vector = encode_action_history(
                records,
                hero_seat_number=hero,
                player_count=players,
                big_blind_chips=big_blind,
            )

            self.assertEqual(
                len(vector), len(schema3.HISTORY_FEATURE_NAMES)
            )
            for value in vector:
                self.assertTrue(math.isfinite(value))

            for street in schema3.HISTORY_STREETS:
                occurred = [
                    vector[_pos(street, slot, "occurred")]
                    for slot in range(schema3.HISTORY_SLOTS)
                ]
                # Right alignment makes gaps impossible: once a slot is
                # occupied, every later slot must be occupied too.
                for earlier, later in zip(occurred, occurred[1:]):
                    self.assertLessEqual(earlier, later)

                # Independent oracle for occupancy and overflow.
                family_count = sum(
                    1
                    for record in records
                    if record["street"] == street
                    and action_family(str(record["action"])) is not None
                )
                self.assertEqual(
                    sum(occurred),
                    float(min(family_count, schema3.HISTORY_SLOTS)),
                )
                self.assertEqual(
                    vector[_overflow_pos(street)],
                    float(family_count > schema3.HISTORY_SLOTS),
                )

                for slot in range(schema3.HISTORY_SLOTS):
                    self.assertIn(occurred[slot], (0.0, 1.0))
                    families = [
                        vector[_pos(street, slot, channel)]
                        for channel in _FAMILY_CHANNELS
                    ]
                    if occurred[slot] == 1.0:
                        # Exactly one family flag, and each is binary.
                        for flag in families:
                            self.assertIn(flag, (0.0, 1.0))
                        self.assertEqual(sum(families), 1.0)
                        actor = vector[_pos(street, slot, "actor")]
                        self.assertGreaterEqual(actor, 0.0)
                        self.assertLessEqual(actor, 1.0)
                        self.assertGreaterEqual(
                            vector[_pos(street, slot, "size")], 0.0
                        )
                    else:
                        for channel in schema3.HISTORY_CHANNELS:
                            self.assertEqual(
                                vector[_pos(street, slot, channel)], 0.0
                            )

    def test_determinism(self) -> None:
        def batch() -> list[tuple[float, ...]]:
            rng = random.Random(self.SEED)
            vectors = []
            for _ in range(50):
                records = self._random_records(rng)
                vectors.append(
                    encode_action_history(
                        records,
                        hero_seat_number=rng.randint(1, 9),
                        player_count=rng.randint(1, 9),
                        big_blind_chips=rng.choice((1, 20, 100)),
                    )
                )
            return vectors

        self.assertEqual(batch(), batch())
        # Same literal input twice through one call path.
        records = [_record(), _record(action="call", seat=None)]
        self.assertEqual(_encode(records), _encode(records))


if __name__ == "__main__":
    unittest.main()
