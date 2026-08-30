"""Invariant checks for the schema-4 assembler (engine/feature_extract_v9).

The load-bearing property is the wrapper construction: every feature the
two schemas share must be BIT-IDENTICAL to the v8 assembler's output —
fuzzed, not asserted. The seven-name delta is checked against its
definitions: the legality quartet against the contract, the cost
conventions per lane, and the dials-off costs against bare g recomputed
independently.
"""

from __future__ import annotations

import unittest

from engine import schema3, schema4
from engine.aggression_sizing import (
    active_bet_wager,
    aggressive_target,
    table_boldness,
)
from engine.branch_contract_v9 import BRANCH_LABELS_V9, legal_branches
from engine.feature_extract_v8 import _EQUITY_TRIALS, extract_features_v8
from engine.feature_extract_v9 import extract_features_v9
from engine.game_state import effective_stack_chips
from engine.hand_strength import estimate_equity


def _snapshot(
    *,
    street: str = "turn",
    board=("Qs", "7d", "2c", "3h"),
    hole=("Qh", "Qd"),
    pot: int = 500,
    to_call: int = 200,
    hero_stack: int = 700,
    opp_stack: int = 600,
    available=("fold", "call", "raise"),
    bet_range=None,
    raise_range=(400, 700),
) -> dict:
    return {
        "id": "extract-v9-test",
        "tableId": "extract-v9-test",
        "street": street,
        "potChips": pot,
        "currentBet": to_call,
        "boardCards": list(board),
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": 1,
        "seats": [
            {
                "seatNumber": 1,
                "status": "Active",
                "stackChips": hero_stack,
                "currentBetChips": 0,
                "holeCards": list(hole),
            },
            {
                "seatNumber": 2,
                "status": "Active",
                "stackChips": opp_stack,
                "currentBetChips": to_call,
                "holeCards": None,
            },
        ],
        "allowedActions": {
            "canFold": "fold" in available,
            "canCheck": "check" in available,
            "canCall": "call" in available,
            "canBet": "bet" in available,
            "canRaise": "raise" in available,
            "canAllIn": False,
            "callAmount": to_call,
            "callChips": to_call,
            "callToAmount": to_call if to_call else None,
            "minBet": bet_range[0] if bet_range else None,
            "minRaiseTo": raise_range[0] if raise_range else None,
            "betRange": (
                {"min": bet_range[0], "max": bet_range[1]} if bet_range else None
            ),
            "raiseRange": (
                {"min": raise_range[0], "max": raise_range[1]}
                if raise_range
                else None
            ),
            "allInToAmount": None,
            "availableActions": list(available),
            "amountSemantics": "toAmount",
            "reasoningRequired": False,
        },
        "recentEvents": [
            {
                "type": "BlindPosted",
                "street": "preflop",
                "summary": {"seatNumber": 1, "amount": 50},
            },
            {
                "type": "BlindPosted",
                "street": "preflop",
                "summary": {"seatNumber": 2, "amount": 100},
            },
        ],
    }


def _read_equity(table: dict, seed: int) -> float:
    """g's own read-equity for a snapshot, recomputed independently."""

    from engine.game_state import active_opponent_count

    hero = table["seats"][0]
    return estimate_equity(
        (hero["holeCards"][0], hero["holeCards"][1]),
        tuple(table["boardCards"]),
        active_opponent_count(table),
        trials=_EQUITY_TRIALS,
        seed=seed,
    )


_FREE_SPOT = dict(
    to_call=0,
    available=("check", "bet"),
    bet_range=(100, 700),
    raise_range=None,
)

_CASES = {
    "priced-turn": {},
    "free-turn": _FREE_SPOT,
    "priced-flop": {"street": "flop", "board": ("Qs", "7d", "2c")},
    "free-preflop": {**_FREE_SPOT, "street": "preflop", "board": ()},
    "priced-river-weak": {
        "street": "river",
        "board": ("Qs", "7d", "2c", "3h", "9s"),
        "hole": ("6h", "4d"),
    },
}


class SharedNameEqualityTests(unittest.TestCase):
    """Every shared feature is the v8 assembler's value, bit for bit."""

    def test_shared_names_bit_identical(self) -> None:
        for label, overrides in _CASES.items():
            with self.subTest(case=label):
                table = _snapshot(**overrides)
                v8 = dict(
                    zip(schema3.FEATURE_NAMES_V8, extract_features_v8(table, seed=11))
                )
                v9 = dict(
                    zip(schema4.FEATURE_NAMES_V9, extract_features_v9(table, seed=11))
                )
                for name in schema4.FEATURE_NAMES_V9:
                    if name in v8:
                        self.assertEqual(v9[name], v8[name], name)

    def test_deterministic(self) -> None:
        table = _snapshot()
        self.assertEqual(
            extract_features_v9(table, seed=5), extract_features_v9(table, seed=5)
        )


class DeltaDefinitionTests(unittest.TestCase):
    def test_vector_shape(self) -> None:
        features = extract_features_v9(_snapshot())
        self.assertEqual(len(features), schema4.INPUT_SIZE_V9)

    def test_legality_quartet_matches_hand_derived_expectations(self) -> None:
        # Expectations derived BY HAND from the v9 taxonomy, deliberately
        # not by calling legal_branches: a test that mirrors the
        # extractor's own call would pass any consistent bug on both
        # sides (this caught exactly one: string-vs-index membership).
        expectations = {
            "priced-turn": {"fatal": 1, "passive": 0, "active": 1, "aggressive": 1},
            "free-turn": {"fatal": 0, "passive": 1, "active": 1, "aggressive": 0},
            "priced-flop": {"fatal": 1, "passive": 0, "active": 1, "aggressive": 1},
            "free-preflop": {"fatal": 0, "passive": 1, "active": 1, "aggressive": 0},
            "priced-river-weak": {
                "fatal": 1, "passive": 0, "active": 1, "aggressive": 1,
            },
        }
        for label, overrides in _CASES.items():
            with self.subTest(case=label):
                values = dict(
                    zip(
                        schema4.FEATURE_NAMES_V9,
                        extract_features_v9(_snapshot(**overrides)),
                    )
                )
                for branch, expected in expectations[label].items():
                    self.assertEqual(
                        values[f"legal_{branch}"],
                        float(expected),
                        f"{label}: legal_{branch}",
                    )
        # And the contract agrees with the hand derivation (index form).
        priced = legal_branches({"fold", "call", "raise"}, 200)
        self.assertEqual(
            [BRANCH_LABELS_V9[i] for i in priced], ["fatal", "active", "aggressive"]
        )

    def test_priced_cost_conventions(self) -> None:
        table = _snapshot()
        values = dict(zip(schema4.FEATURE_NAMES_V9, extract_features_v9(table)))
        eff = max(1, effective_stack_chips(table))
        self.assertEqual(values["cost_active_eff"], min(1.0, 200 / eff))
        self.assertEqual(values["legal_aggressive"], 1.0)
        self.assertIn(values["branch_aggressive_executable"], (0.0, 1.0))

    def test_free_spot_masks_the_aggressive_lane(self) -> None:
        values = dict(
            zip(
                schema4.FEATURE_NAMES_V9,
                extract_features_v9(_snapshot(**_FREE_SPOT)),
            )
        )
        self.assertEqual(values["legal_aggressive"], 0.0)
        self.assertEqual(values["cost_aggressive_eff"], 0.0)
        self.assertEqual(values["branch_aggressive_executable"], 0.0)
        self.assertEqual(values["legal_fatal"], 0.0)  # free check: fold dominated
        self.assertGreater(values["cost_active_eff"], 0.0)

    def test_free_spot_raise_shape_uses_the_raise_range(self) -> None:
        """27 live rows offer 'raise' at to_call == 0 with betRange null.

        Without the raiseRange fallback the cost skips the clamped path
        entirely, so this asserts the CLAMPED value — which is what the
        engine would actually wager — not merely that it is non-zero.
        """

        table = _snapshot(
            street="preflop",
            board=(),
            to_call=0,
            available=("check", "fold", "raise", "all-in"),
            bet_range=None,
            # Minimum deliberately ABOVE g's unclamped wager, so the clamp
            # decides the cost: the un-fixed path found no range at all and
            # reported the raw target, which differs here.
            raise_range=(500, 640),
        )
        values = dict(
            zip(schema4.FEATURE_NAMES_V9, extract_features_v9(table, seed=7))
        )
        self.assertEqual(values["legal_active"], 1.0)
        self.assertEqual(values["legal_aggressive"], 0.0)  # still escalation-only

        eff = max(1, effective_stack_chips(table))
        raw = active_bet_wager(
            table["potChips"],
            table_boldness(table, table["allowedActions"], _read_equity(table, 7)),
        )
        self.assertLess(raw, 500)                       # the clamp must bind
        self.assertEqual(values["cost_active_eff"], min(1.0, 500 / eff))
        self.assertNotEqual(values["cost_active_eff"], min(1.0, raw / eff))

    def test_equity_multiway_is_the_read_equity(self) -> None:
        """Owner queue item 1: the multiway strength input IS g's read
        equity — one computation, two consumers, proven by recomputing
        the read independently for several player counts."""

        for extra_seats in (0, 2):
            table = _snapshot()
            for offset in range(extra_seats):
                table["seats"].append(
                    {
                        "seatNumber": 3 + offset,
                        "status": "Active",
                        "stackChips": 500,
                        "currentBetChips": 0,
                        "holeCards": None,
                    }
                )
            values = dict(
                zip(schema4.FEATURE_NAMES_V9, extract_features_v9(table, seed=11))
            )
            self.assertEqual(
                values["equity_multiway"], _read_equity(table, 11)
            )
        # More opponents must not read as MORE equity for the same hand.
        lone = _snapshot()
        crowd = _snapshot()
        crowd["seats"].extend(
            {
                "seatNumber": 3 + offset,
                "status": "Active",
                "stackChips": 500,
                "currentBetChips": 0,
                "holeCards": None,
            }
            for offset in range(3)
        )
        one = dict(zip(schema4.FEATURE_NAMES_V9, extract_features_v9(lone, seed=11)))
        many = dict(zip(schema4.FEATURE_NAMES_V9, extract_features_v9(crowd, seed=11)))
        self.assertLess(many["equity_multiway"], one["equity_multiway"])

    def test_dials_off_aggressive_cost_is_bare_g(self) -> None:
        """The composed cost with every dial off equals g recomputed here."""

        table = _snapshot()
        seed = 7
        values = dict(
            zip(schema4.FEATURE_NAMES_V9, extract_features_v9(table, seed=seed))
        )
        eff = max(1, effective_stack_chips(table))
        equity_read = estimate_equity(
            ("Qh", "Qd"),
            ("Qs", "7d", "2c", "3h"),
            1,
            trials=_EQUITY_TRIALS,
            seed=seed,
        )
        boldness = table_boldness(table, table["allowedActions"], equity_read)
        target = aggressive_target(
            pot=500, to_call=200, effective_stack=eff, boldness=boldness
        )
        to_amount = min(700, max(400, 0 + target))
        self.assertEqual(values["cost_aggressive_eff"], min(1.0, to_amount / eff))


if __name__ == "__main__":
    unittest.main()
