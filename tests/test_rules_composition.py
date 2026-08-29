"""Invariant checks for ``engine/rules`` (spec: engine/rules/README.md).

The three composition invariants are fuzzed, not asserted in prose:

- **Zero-diff** — every dial off, the composed pipeline is bit-identical
  to bare g on whatever state the fuzzer produces. Integration without
  enablement must be a no-op.
- **Damper supremacy** — with C5 active, no emitted size exceeds the same
  state's size with C5 disabled (the min-with-undamped construction).
- **Attribution** — exactly one setter per composed wager, drawn from the
  closed vocabulary.

Plus per-module checks: C1's derived threshold identity, C4's counter
excluding hero and its measured default, C3's largest-in-band choice, and
the same bool/NaN parameter hardening g carries.
"""

from __future__ import annotations

import math
import random
import unittest

from engine.aggression_sizing import (
    active_bet_wager,
    aggressive_target,
)
from engine.rules import (
    CommitmentGateParams,
    EscalationMarginParams,
    GeometricSizingParams,
    RuinDamperParams,
    SnapToCoverParams,
    compose_active_wager,
    compose_aggressive_target,
    escalation_margin,
    forward_commitment,
    geometric_fraction,
    opponent_raises_this_street,
    snap_to_cover,
)
from engine.rules.escalation_margin import KAPPA_E_ESTIMATED

_FUZZ_SEED = 0xC0FFEE
_FUZZ_CASES = 300
_STREETS = ("preflop", "flop", "turn", "river")


def _random_state(rng: random.Random) -> dict:
    return {
        "boldness": rng.uniform(-1.0, 1.0),
        "pot": rng.randint(1, 5_000),
        "to_call": rng.randint(1, 3_000),
        "effective_stack": rng.randint(0, 20_000),
        "contribution": rng.randint(0, 2_000),
        "street": rng.choice(_STREETS),
        "bankroll": rng.randint(0, 50_000),
        "exposure": rng.randint(0, 10_000),
    }


class ZeroDiffTests(unittest.TestCase):
    """Every dial off => bit-identical to bare g."""

    def test_aggressive_lane_matches_g_exactly(self) -> None:
        rng = random.Random(_FUZZ_SEED)
        for _ in range(_FUZZ_CASES):
            state = _random_state(rng)
            composed = compose_aggressive_target(
                boldness=state["boldness"],
                pot=state["pot"],
                to_call=state["to_call"],
                effective_stack=state["effective_stack"],
                contribution=state["contribution"],
                street=state["street"],
                bankroll=state["bankroll"],
                exposure=state["exposure"],
            )
            bare = aggressive_target(
                pot=state["pot"],
                to_call=state["to_call"],
                effective_stack=state["effective_stack"],
                boldness=state["boldness"],
            )
            self.assertEqual(composed.target, bare)
            self.assertEqual(composed.to_amount, state["contribution"] + bare)
            self.assertEqual(composed.boldness_used, state["boldness"])

    def test_active_lane_matches_g_exactly(self) -> None:
        rng = random.Random(_FUZZ_SEED + 1)
        for _ in range(_FUZZ_CASES):
            state = _random_state(rng)
            composed = compose_active_wager(
                boldness=state["boldness"],
                pot=state["pot"],
                effective_stack=state["effective_stack"],
                contribution=state["contribution"],
                street=state["street"],
                bankroll=state["bankroll"],
                exposure=state["exposure"],
            )
            bare = active_bet_wager(state["pot"], state["boldness"])
            self.assertEqual(composed.target, bare)

    def test_aggressive_lane_keeps_the_contract_refusal(self) -> None:
        with self.assertRaises(ValueError):
            compose_aggressive_target(
                boldness=0.0, pot=100, to_call=0, effective_stack=1_000,
                contribution=0, street="flop", bankroll=1_000, exposure=100,
            )


class DamperSupremacyTests(unittest.TestCase):
    """C5 active can only shrink or hold the emitted size, never grow it."""

    def test_no_size_exceeds_the_undamped_size(self) -> None:
        rng = random.Random(_FUZZ_SEED + 2)
        tight = RuinDamperParams(enabled=True, kappa_r=8.0)
        geo = GeometricSizingParams(enabled=True)
        for _ in range(_FUZZ_CASES):
            state = _random_state(rng)
            kwargs = dict(
                boldness=state["boldness"],
                pot=state["pot"],
                to_call=state["to_call"],
                effective_stack=state["effective_stack"],
                contribution=state["contribution"],
                street=state["street"],
                bankroll=state["bankroll"],
                exposure=state["exposure"],
                geometric=geo,
            )
            damped = compose_aggressive_target(damper=tight, **kwargs)
            undamped = compose_aggressive_target(**kwargs)
            self.assertLessEqual(damped.target, undamped.target)

    def test_attribution_is_exactly_one_of_the_vocabulary(self) -> None:
        rng = random.Random(_FUZZ_SEED + 3)
        geo = GeometricSizingParams(enabled=True)
        snap = SnapToCoverParams(enabled=True)
        tight = RuinDamperParams(enabled=True, kappa_r=8.0)
        for _ in range(_FUZZ_CASES):
            state = _random_state(rng)
            composed = compose_aggressive_target(
                boldness=state["boldness"],
                pot=state["pot"],
                to_call=state["to_call"],
                effective_stack=state["effective_stack"],
                contribution=state["contribution"],
                street=state["street"],
                bankroll=state["bankroll"],
                exposure=state["exposure"],
                covered_allin_to_amounts=(rng.randint(1, 6_000),),
                geometric=geo,
                snap=snap,
                damper=tight,
            )
            self.assertIn(composed.set_by, {"g", "C2", "stack-cap", "C3A"})


class CommitmentGateTests(unittest.TestCase):
    def test_derived_threshold_matches_the_price_identity(self) -> None:
        # At SPR' exactly 1, a next-street shove of E' into P' prices 1/3.
        gate_stack, to_call, pot = 700, 100, 400
        spr_post = (gate_stack - to_call) / (pot + 2 * to_call)
        self.assertEqual(spr_post, 1.0)
        remaining = gate_stack - to_call
        next_pot = pot + 2 * to_call
        price = remaining / (next_pot + 2 * remaining)
        self.assertAlmostEqual(price, 1.0 / 3.0)
        verdict = forward_commitment(
            CommitmentGateParams(enabled=True),
            gate_stack=gate_stack, to_call=to_call, pot=pot,
        )
        self.assertTrue(verdict.fired)

    def test_uncommitted_geometry_passes(self) -> None:
        verdict = forward_commitment(
            CommitmentGateParams(enabled=True),
            gate_stack=10_000, to_call=100, pot=400,
        )
        self.assertFalse(verdict.fired)
        self.assertGreater(verdict.spr_post, 1.0)

    def test_disabled_is_inert_and_collapse_fires(self) -> None:
        self.assertFalse(
            forward_commitment(
                CommitmentGateParams(), gate_stack=0, to_call=100, pot=400
            ).fired
        )
        self.assertTrue(
            forward_commitment(
                CommitmentGateParams(enabled=True),
                gate_stack=1, to_call=100, pot=400,
            ).fired
        )


class GeometricTests(unittest.TestCase):
    def test_textbook_values(self) -> None:
        self.assertAlmostEqual(geometric_fraction(13.0, 3), 1.0)
        self.assertAlmostEqual(geometric_fraction(1.0, 1), 1.0)
        self.assertAlmostEqual(geometric_fraction(4.0, 3), 0.5400, places=3)


class EscalationTests(unittest.TestCase):
    def test_default_is_the_measured_value(self) -> None:
        self.assertEqual(
            EscalationMarginParams().kappa_e, KAPPA_E_ESTIMATED
        )

    def test_margin_shape(self) -> None:
        params = EscalationMarginParams(enabled=True)
        self.assertEqual(escalation_margin(params, 0).margin_added, 0.0)
        self.assertEqual(escalation_margin(params, 1).margin_added, 0.0)
        self.assertAlmostEqual(
            escalation_margin(params, 3).margin_added, 2 * params.kappa_e
        )

    def test_counter_excludes_hero(self) -> None:
        table = {
            "recentEvents": [
                {"street": "flop", "summary": {"seatNumber": 1, "action": "bet"}},
                {"street": "flop", "summary": {"seatNumber": 2, "action": "raise"}},
                {"street": "flop", "summary": {"seatNumber": 3, "action": "call"}},
                {"street": "turn", "summary": {"seatNumber": 2, "action": "bet"}},
            ]
        }
        self.assertEqual(opponent_raises_this_street(table, "flop", 1), 1)
        self.assertEqual(opponent_raises_this_street(table, "flop", 9), 2)


class SnapTests(unittest.TestCase):
    def test_largest_in_band_wins_and_disabled_is_inert(self) -> None:
        params = SnapToCoverParams(enabled=True, band=0.15)
        verdict = snap_to_cover(
            params, to_amount=900.0, covered_allin_to_amounts=(800, 1_000, 5_000)
        )
        self.assertTrue(verdict.fired)
        self.assertEqual(verdict.snapped_to, 1_000.0)
        self.assertFalse(
            snap_to_cover(
                SnapToCoverParams(), to_amount=900.0,
                covered_allin_to_amounts=(1_000,),
            ).fired
        )


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
    """A minimal valid Arena snapshot (the feature-extract fixture, trimmed)."""

    return {
        "id": "rules-verdict-test",
        "tableId": "rules-verdict-test",
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


class VerdictTelemetryTests(unittest.TestCase):
    """Fired verdicts reach DecisionResult; dial-off decisions carry None."""

    @staticmethod
    def _engine(family: str, **kwargs):
        from engine.decision_engine import DecisionEngine

        return type(
            "Probe", (DecisionEngine,), {"_family": lambda self, f: family}
        )(seed=3, **kwargs)

    def test_c1_attribution_reaches_the_result(self) -> None:
        from engine.rules.composition import RuleLayerParams

        engine = self._engine(
            "check_call",
            rule_layer=RuleLayerParams(
                commitment=CommitmentGateParams(enabled=True)
            ),
        )
        # SPR' = (600 - 200) / (500 + 400) = 0.44 <= 1: C1 must fire, and
        # hero's set of queens clears every floor, so the call survives
        # WITH the attribution recorded. No raise offered: with one the
        # heuristic ladder aggresses on this equity and never prices the
        # call.
        result = engine.decide_with_diagnostics(
            _snapshot(available=("fold", "call"), raise_range=None)
        )
        self.assertIsNotNone(result.rule_verdicts)
        rules = [record["rule"] for record in result.rule_verdicts]
        self.assertIn("C1-forward-commitment", rules)
        fired = next(
            record for record in result.rule_verdicts
            if record["rule"] == "C1-forward-commitment"
        )
        self.assertTrue(fired["fired"])
        self.assertLessEqual(fired["spr_post"], 1.0)

    def test_c5_attribution_reaches_the_result(self) -> None:
        from engine.rules.composition import RuleLayerParams

        engine = self._engine(
            "aggress",
            rule_layer=RuleLayerParams(
                damper=RuinDamperParams(enabled=True, kappa_r=8.0)
            ),
        )
        # Free spot, hero bets; bankroll 700 vs exposure 600 at kr-8 damps
        # (d ~ 0.146), so the sizing verdict must be recorded.
        result = engine.decide_with_diagnostics(
            _snapshot(
                to_call=0,
                available=("check", "bet"),
                bet_range=(100, 700),
                raise_range=None,
            )
        )
        self.assertIsNotNone(result.rule_verdicts)
        rules = [record["rule"] for record in result.rule_verdicts]
        self.assertIn("C5-ruin-damper", rules)

    def test_dials_off_yield_none(self) -> None:
        result = self._engine("check_call").decide_with_diagnostics(_snapshot())
        self.assertIsNone(result.rule_verdicts)

    def test_journal_record_carries_the_field(self) -> None:
        from engine.rules.composition import RuleLayerParams
        from engine.training_telemetry import make_decision_record

        engine = self._engine(
            "check_call",
            rule_layer=RuleLayerParams(
                commitment=CommitmentGateParams(enabled=True)
            ),
        )
        table = _snapshot(available=("fold", "call"), raise_range=None)
        decision = engine.decide_with_diagnostics(table)

        def record_for(result):
            return make_decision_record(
                competition_id="c",
                policy_version="rules-verdict-test",
                table=table,
                payload=result.to_payload(),
                decision=result,
                deadline_budget_s=5.0,
                fallback_reason=None,
                action_status=200,
                identity_verified=True,
                recorded_at_ms=1,
            )

        record = record_for(decision)
        self.assertEqual(record["telemetry_schema_version"], 3)
        self.assertIsInstance(record["rule_verdicts"], list)
        self.assertEqual(record["rule_verdicts"], list(decision.rule_verdicts))
        off = self._engine("check_call").decide_with_diagnostics(table)
        self.assertIsNone(record_for(off)["rule_verdicts"])


class ParameterHardeningTests(unittest.TestCase):
    def test_bool_and_nan_refused_everywhere(self) -> None:
        for build in (
            lambda: CommitmentGateParams(spr_threshold=True),
            lambda: CommitmentGateParams(spr_threshold=float("nan")),
            lambda: SnapToCoverParams(band=True),
            lambda: SnapToCoverParams(band=0.0),
            lambda: EscalationMarginParams(kappa_e=True),
            lambda: EscalationMarginParams(kappa_e=float("nan")),
            lambda: RuinDamperParams(kappa_r=True),
            lambda: RuinDamperParams(kappa_r=0.0),
        ):
            with self.assertRaises(ValueError):
                build()

    def test_verdict_mappings_are_json_shaped(self) -> None:
        composed = compose_aggressive_target(
            boldness=0.4, pot=300, to_call=100, effective_stack=2_000,
            contribution=0, street="turn", bankroll=5_000, exposure=1_000,
        )
        for record in composed.verdicts():
            self.assertIsInstance(record["rule"], str)
            self.assertIsInstance(record["fired"], bool)
            for value in record.values():
                self.assertTrue(
                    value is None or isinstance(value, (str, bool, int, float))
                )
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
