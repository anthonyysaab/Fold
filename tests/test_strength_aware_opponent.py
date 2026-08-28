"""Contract tests for the P3 strength-aware opponent, on the shipped artifact.

The fit agent's original task list named this file and it was never written
(a failed subagent is not a clean result — the absence was only noticed by
the audit). Everything here asserts either a property the module's own
docstrings promise (sign invariants, RNG discipline, ``reads_cards``, the
degrade path) or the 2026-08-16 price-support clamp repair (fold probability
flat in bet size above the fitted-support cap).

Complements, not duplicates: ``tests/test_p3_audit.py`` characterises the
dataset and the defect history; ``tests/test_p3_gate.py`` covers the strict
battery seat and the gate instrument. This file is the direct contract of
``engine.strength_aware_opponent`` itself.
"""

from __future__ import annotations

import json
import math
import random
import unittest
from pathlib import Path

from engine.strength_aware_opponent import (
    DEFAULT_FIT_PATH,
    FIT_FEATURE_NAMES,
    PRICE_INDEX,
    STREETS,
    STRENGTH_INDEX,
    LogisticModel,
    P3Decision,
    P3FitError,
    StrengthAwareAgent,
    load_fit,
    strength_aware_lineup,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIT_PATH = REPO_ROOT / DEFAULT_FIT_PATH


def _require_fit() -> Path:
    if not FIT_PATH.exists():  # pragma: no cover - artifact present in repo
        raise unittest.SkipTest(f"missing {FIT_PATH}")
    return FIT_PATH


def _flop_table(*, pot_before: int, bet: int, hole: tuple[str, ...],
                board: tuple[str, ...] = ("2c", "7d", "9h")) -> dict:
    """A format-exact snapshot: villain (seat 2) faces ``bet`` into ``pot_before``."""

    pot = pot_before + bet
    return {
        "id": "contract", "tableId": "contract", "handId": "contract-h0",
        "street": {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(board)],
        "potChips": pot,
        "currentBet": bet,
        "boardCards": list(board),
        "smallBlindChips": 50, "bigBlindChips": 100,
        "selfSeatNumber": 2,
        "seats": [
            {"seatNumber": 1, "agentId": "hero", "status": "Active",
             "stackChips": 1_000_000, "currentBetChips": bet,
             "totalCommittedChips": pot_before // 2 + bet, "holeCards": None},
            {"seatNumber": 2, "agentId": "villain", "status": "Active",
             "stackChips": 1_000_000, "currentBetChips": 0,
             "totalCommittedChips": pot_before // 2, "holeCards": list(hole)},
        ],
        "allowedActions": {
            "canFold": True, "canCheck": False, "canCall": True, "canBet": False,
            "canRaise": True, "canAllIn": True, "callAmount": bet,
            "callChips": bet, "callToAmount": bet, "minBet": None,
            "minRaiseTo": 2 * bet, "betRange": None,
            "raiseRange": {"min": 2 * bet, "max": 1_000_000},
            "allInToAmount": 1_000_000,
            "availableActions": ["fold", "call", "raise", "all-in"],
            "amountSemantics": "toAmount", "reasoningRequired": False,
        },
        "recentEvents": [
            {"type": "BlindPosted", "street": "preflop",
             "summary": {"seatNumber": 1, "action": "smallBlind", "amount": 50}},
            {"type": "BlindPosted", "street": "preflop",
             "summary": {"seatNumber": 2, "action": "bigBlind", "amount": 100}},
        ],
        "simulationRollout": None,
    }


def _agent(**kwargs) -> StrengthAwareAgent:
    return StrengthAwareAgent(
        "contract", 0.226, 0.5, 0.0, 5, fit=load_fit(str(_require_fit())), **kwargs
    )


def _decision(street: str, strength: float, price: float) -> P3Decision:
    return P3Decision(
        street=street,
        strength_percentile=strength,
        pot_odds=price,
        bet_to_pot=price / max(1e-9, 1.0 - price),
        texture=0.0 if street == "preflop" else 0.3,
        active_players=2,
        position_unit=0.5,
        to_call=100,
        pot=200,
    )


class ShippedSignInvariantTests(unittest.TestCase):
    """Better hands fold less; worse prices fold more — on the shipped JSON.

    Re-asserted from the raw document, not through ``load_fit``, so a broken
    artifact is caught even if the loader's own check were ever weakened.
    """

    def test_every_shipped_model_has_the_right_signs(self) -> None:
        document = json.loads(_require_fit().read_text(encoding="utf-8"))
        self.assertEqual(tuple(document["features"]), FIT_FEATURE_NAMES)
        models = document["models"]
        self.assertGreaterEqual(len(models), 1)
        for key, model in sorted(models.items()):
            with self.subTest(model=key):
                self.assertLess(model["coefficients"][STRENGTH_INDEX], 0.0)
                self.assertGreater(model["coefficients"][PRICE_INDEX], 0.0)

    def test_the_loader_enforces_the_same_invariant(self) -> None:
        fit = load_fit(str(_require_fit()))
        fit.check_sign_invariant()  # must not raise
        for key, model in sorted(fit.models.items()):
            with self.subTest(model=key):
                self.assertLess(model.strength_coefficient, 0.0)
                self.assertGreater(model.price_coefficient, 0.0)


class MonotoneSweepTests(unittest.TestCase):
    """The two response properties, swept through the clamped serve function."""

    def setUp(self) -> None:
        self.agent = _agent()

    def test_fold_probability_decreases_in_strength_on_every_street(self) -> None:
        grid = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
        for street in STREETS:
            with self.subTest(street=street):
                values = [
                    self.agent.fold_probability_for(_decision(street, s, 1 / 3))
                    for s in grid
                ]
                for weaker, stronger in zip(values, values[1:]):
                    self.assertLessEqual(stronger, weaker + 1e-12)
                self.assertGreater(values[0], values[-1])

    def test_fold_probability_increases_in_price_up_to_the_cap(self) -> None:
        grid = [0.15, 0.2, 0.25, 0.3, 1 / 3, 0.36, 0.4, 0.45, 0.5]
        for street in STREETS:
            with self.subTest(street=street):
                values = [
                    self.agent.fold_probability_for(_decision(street, 0.5, p))
                    for p in grid
                ]
                for cheaper, dearer in zip(values, values[1:]):
                    self.assertGreaterEqual(dearer, cheaper - 1e-12)
                self.assertGreater(values[-1], values[0])


class PriceSupportClampTests(unittest.TestCase):
    """The 2026-08-16 repair: above the fitted support, size buys nothing."""

    def setUp(self) -> None:
        self.agent = _agent()

    def test_shipped_caps_are_data_recorded_and_self_consistent(self) -> None:
        document = json.loads(_require_fit().read_text(encoding="utf-8"))
        support = document["price_support"]
        self.assertEqual(support["clamped_feature"], "pot_odds")
        self.assertTrue(support["quantile_ablatable"])
        self.assertGreater(support["quantile"], 0.0)
        fit = load_fit(str(FIT_PATH))
        for key, model in sorted(fit.models.items()):
            with self.subTest(model=key):
                recorded = support["per_model"][key]
                self.assertEqual(model.price_cap, recorded["price_cap"])
                # A cap outside the observed data would be a hand-authored
                # constant wearing a derivation; it must sit inside support.
                self.assertGreater(recorded["price_cap"],
                                   recorded["train_price_median"])
                self.assertLessEqual(recorded["price_cap"],
                                     recorded["train_price_max"])

    def test_fold_probability_is_flat_between_2x_and_4x_pot(self) -> None:
        """Identical fold probability at 2x and 4x pot on the same state.

        Both prices (0.400 and 0.444) sit above the flop cap, so the clamp
        must make them indistinguishable — a bigger overbet buys no extra
        fold equity. Strictly higher than at half pot (price 0.250, inside
        support), where the fitted response is genuine.
        """

        pot = 1_000
        hole = ("Qs", "8d")
        tables = {
            multiplier: _flop_table(pot_before=pot, bet=int(pot * multiplier),
                                    hole=hole)
            for multiplier in (0.5, 2, 4)
        }
        values = {
            multiplier: self.agent._fold_probability(
                table, table["allowedActions"]
            )
            for multiplier, table in tables.items()
        }
        self.assertEqual(values[2], values[4])
        self.assertGreater(values[2], values[0.5])
        self.assertEqual(self.agent.fallback_count, 0)

    def test_the_clamp_is_the_models_not_the_bands(self) -> None:
        """The flat region comes from ``price_cap``, not the output band."""

        fit = load_fit(str(FIT_PATH))
        model = fit.model_for("flop")
        self.assertIsNotNone(model.price_cap)
        at_cap = model.predict(
            _decision("flop", 0.5, model.price_cap).vector
        )
        above = model.predict(
            _decision("flop", 0.5, min(0.99, model.price_cap + 0.1)).vector
        )
        self.assertEqual(at_cap, above)
        low, high = fit.band
        self.assertLess(at_cap, high, "flatness must not be band saturation")

    def test_below_support_prices_are_not_clamped(self) -> None:
        fit = load_fit(str(FIT_PATH))
        model = fit.model_for("flop")
        cheap = model.predict(_decision("flop", 0.5, 0.15).vector)
        cheaper = model.predict(_decision("flop", 0.5, 0.10).vector)
        self.assertGreater(cheap, cheaper)


class LogisticModelClampUnitTests(unittest.TestCase):
    """The clamp mechanism itself, off the artifact, on a hand-built model."""

    @staticmethod
    def _model(price_cap: float | None) -> LogisticModel:
        return LogisticModel(
            intercept=0.0,
            coefficients=(-1.0, 2.0, 0.0, 0.0, 0.0),
            mean=(0.5, 0.3, 0.0, 2.0, 0.5),
            std=(0.25, 0.06, 1.0, 1.0, 0.3),
            price_cap=price_cap,
        )

    def test_prices_above_the_cap_evaluate_exactly_at_the_cap(self) -> None:
        capped = self._model(price_cap=0.36)
        free = self._model(price_cap=None)
        base = [0.5, 0.36, 0.0, 2.0, 0.5]
        overbet = [0.5, 0.48, 0.0, 2.0, 0.5]
        self.assertEqual(capped.logit(overbet), free.logit(base))
        self.assertEqual(capped.predict(overbet), capped.predict(base))
        below = [0.5, 0.2, 0.0, 2.0, 0.5]
        self.assertEqual(capped.logit(below), free.logit(below))

    def test_the_clamp_does_not_mutate_the_callers_vector(self) -> None:
        capped = self._model(price_cap=0.36)
        values = [0.5, 0.48, 0.0, 2.0, 0.5]
        capped.predict(values)
        self.assertEqual(values, [0.5, 0.48, 0.0, 2.0, 0.5])

    def test_a_missing_cap_means_no_clamp(self) -> None:
        free = self._model(price_cap=None)
        self.assertGreater(
            free.predict([0.5, 0.48, 0.0, 2.0, 0.5]),
            free.predict([0.5, 0.36, 0.0, 2.0, 0.5]),
        )

    def test_an_invalid_cap_is_refused(self) -> None:
        for bad in (0.0, 1.0, -0.2, math.inf, math.nan):
            with self.subTest(cap=bad):
                with self.assertRaises(P3FitError):
                    self._model(price_cap=bad)

    def test_a_cap_on_the_wrong_feature_order_is_refused(self) -> None:
        with self.assertRaises(P3FitError):
            LogisticModel(
                intercept=0.0,
                coefficients=(-1.0, 2.0, 0.0, 0.0, 0.0),
                mean=(0.5, 0.3, 0.0, 2.0, 0.5),
                std=(0.25, 0.06, 1.0, 1.0, 0.3),
                feature_names=("pot_odds", "strength_percentile", "texture",
                               "active_players", "position_unit"),
                price_cap=0.36,
            )

    def test_round_trip_through_the_document_keeps_the_cap(self) -> None:
        capped = self._model(price_cap=0.36)
        document = capped.to_document()
        self.assertEqual(document["price_cap"], 0.36)
        reloaded = LogisticModel.from_document(document, FIT_FEATURE_NAMES)
        self.assertEqual(reloaded.price_cap, 0.36)
        free_document = self._model(price_cap=None).to_document()
        self.assertNotIn("price_cap", free_document)
        self.assertIsNone(
            LogisticModel.from_document(free_document, FIT_FEATURE_NAMES).price_cap
        )


class DeterminismAndRngDisciplineTests(unittest.TestCase):
    """No randomness in, no randomness out, bit-identical forever."""

    def setUp(self) -> None:
        self.agent = _agent()
        self.table = _flop_table(pot_before=1_000, bet=700, hole=("Qs", "8d"))
        self.allowed = self.table["allowedActions"]

    def test_repeated_calls_are_bit_identical(self) -> None:
        values = {
            self.agent._fold_probability(self.table, self.allowed)
            for _ in range(25)
        }
        self.assertEqual(len(values), 1)

    def test_a_fresh_instance_agrees_exactly(self) -> None:
        fresh = _agent()
        self.assertEqual(
            fresh._fold_probability(self.table, self.allowed),
            self.agent._fold_probability(self.table, self.allowed),
        )

    def test_fold_probability_consumes_no_module_randomness(self) -> None:
        random.seed(20260816)
        before = random.getstate()
        for _ in range(10):
            self.agent._fold_probability(self.table, self.allowed)
        self.assertEqual(random.getstate(), before)

    def test_fold_probability_does_not_advance_a_callers_stream(self) -> None:
        reference = random.Random(4242)
        expected = [reference.random() for _ in range(2)]
        stream = random.Random(4242)
        observed = [stream.random()]
        for _ in range(10):
            self.agent._fold_probability(self.table, self.allowed)
        observed.append(stream.random())
        self.assertEqual(observed, expected)


class ReadsCardsContractTests(unittest.TestCase):
    """``table_simulator`` keys the chance-salt exclusion on this flag."""

    def test_the_class_and_every_lineup_seat_read_cards(self) -> None:
        self.assertIs(StrengthAwareAgent.reads_cards, True)
        fit = load_fit(str(_require_fit()))
        for name, agent in strength_aware_lineup(fit):
            with self.subTest(seat=name):
                self.assertIs(agent.reads_cards, True)


class FallbackPathTests(unittest.TestCase):
    """Degrading to the card-blind constant is counted, never silent."""

    def setUp(self) -> None:
        self.agent = _agent()

    def test_hidden_hole_cards_degrade_to_the_constant_and_count(self) -> None:
        table = _flop_table(pot_before=1_000, bet=500, hole=("Ah", "Kd"))
        table["seats"][1]["holeCards"] = None
        value = self.agent._fold_probability(table, table["allowedActions"])
        self.assertEqual(value, self.agent.fold_vs_bet)
        self.assertEqual(self.agent.fallback_count, 1)
        self.assertIn("hole cards", self.agent.last_fallback_reason)

    def test_an_unreadable_snapshot_degrades_and_records_why(self) -> None:
        table = _flop_table(pot_before=1_000, bet=500, hole=("Ah", "Kd"))
        table["potChips"] = "not-a-number"
        value = self.agent._fold_probability(table, table["allowedActions"])
        self.assertEqual(value, self.agent.fold_vs_bet)
        self.assertEqual(self.agent.fallback_count, 1)
        self.assertNotEqual(self.agent.last_fallback_reason, "")

    def test_a_clean_snapshot_never_touches_the_fallback(self) -> None:
        table = _flop_table(pot_before=1_000, bet=500, hole=("Ah", "Kd"))
        self.agent._fold_probability(table, table["allowedActions"])
        self.assertEqual(self.agent.fallback_count, 0)
        self.assertEqual(self.agent.last_fallback_reason, "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
