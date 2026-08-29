"""Invariant checks for ``engine/rules`` (spec: engine/rules/README.md).

The three composition invariants are fuzzed, not asserted in prose:

- **Zero-diff** — every dial off, the composed pipeline is bit-identical
  to bare g on whatever state the fuzzer produces. Integration without
  enablement must be a no-op.
- **Damper supremacy** — with C5 active, no emitted size exceeds the same
  state's size with C5 disabled (the min-with-undamped construction).
- **Attribution** — exactly one setter per composed wager, drawn from the
  closed vocabulary.

Plus per-module checks: C1's derived threshold identity, C4's street
ordinal and its measured step table, C3's largest-in-band choice, and the
same bool/NaN parameter hardening g carries. Every regression test here
derives its expectation from an INDEPENDENT source (the pot convention,
the live journal, a textbook value, a hand-built state) — twice now a
test written from the same premise as the code has passed the code's
bug.
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
    street_aggressions,
    snap_to_cover,
)

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
    def test_pot_convention_is_the_engine_s_own(self) -> None:
        """potChips already contains the bet faced — C1's denominator rests
        on this, so it is pinned against the engine's pot-odds definition
        rather than restated (the first C1 draft double-counted it)."""

        from engine.decision_engine import DecisionEngine

        engine = type(
            "Probe", (DecisionEngine,), {"_family": lambda self, f: "check_call"}
        )()
        pot, to_call = 300, 100
        # pot odds = price / (pot AFTER the call). If potChips excluded the
        # outstanding bet this would have to read to_call/(pot+2*to_call).
        self.assertAlmostEqual(
            engine._pot_odds({"potChips": pot}, {"callChips": to_call}),
            to_call / (pot + to_call),
        )

    def test_threshold_is_the_shove_price_boundary(self) -> None:
        # Built forward from the convention, NOT from the gate's formula:
        # a state where calling leaves exactly as many chips behind as the
        # pot the call creates (SPR' = 1).
        to_call, pot_before = 100, 400
        pot_after_call = pot_before + to_call          # the convention
        gate_stack = pot_after_call + to_call          # E' == P' after calling
        remaining = gate_stack - to_call
        self.assertEqual(remaining, pot_after_call)    # SPR' == 1 by construction

        # At that geometry a next-street shove prices at exactly 1/3.
        price = remaining / (pot_after_call + 2 * remaining)
        self.assertAlmostEqual(price, 1.0 / 3.0)

        verdict = forward_commitment(
            CommitmentGateParams(enabled=True),
            gate_stack=gate_stack, to_call=to_call, pot=pot_before,
        )
        self.assertTrue(verdict.fired)
        self.assertAlmostEqual(verdict.spr_post, 1.0)

    def test_just_above_the_boundary_does_not_fire(self) -> None:
        # One chip deeper than the SPR' == 1 state must not fire.
        to_call, pot_before = 100, 400
        gate_stack = pot_before + 2 * to_call + 1
        verdict = forward_commitment(
            CommitmentGateParams(enabled=True),
            gate_stack=gate_stack, to_call=to_call, pot=pot_before,
        )
        self.assertFalse(verdict.fired)
        self.assertGreater(verdict.spr_post, 1.0)

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

    def test_self_deactivates_above_the_lane_top(self) -> None:
        """Regression: the first draft clamped the blend at lane_top, so a
        mildly value-leaning read at live SPR jumped to the band MAXIMUM
        instead of leaving the lane fraction alone."""

        from engine.rules import blended_fraction

        # SPR 120 — the live median. f_geo is far above any lane top.
        verdict = blended_fraction(
            GeometricSizingParams(enabled=True),
            lane_fraction=0.775,
            boldness=0.1,
            pot=100,
            effective_stack=12_000,
            street="turn",
            lane_top=1.0,
        )
        self.assertFalse(verdict.fired)
        self.assertEqual(verdict.f_out, 0.775)
        self.assertGreater(verdict.f_geo, 1.0)

    def test_still_blends_inside_the_band(self) -> None:
        from engine.rules import blended_fraction

        # SPR 4 on the flop: f_geo ~ 0.54, inside the aggressive band.
        verdict = blended_fraction(
            GeometricSizingParams(enabled=True),
            lane_fraction=1.0,
            boldness=1.0,
            pot=100,
            effective_stack=400,
            street="flop",
            lane_top=1.0,
        )
        self.assertTrue(verdict.fired)
        # Independent expectation: SPR 4 over 3 remaining streets is the
        # textbook ~0.54 pot, and at w = 1 the blend IS that target.
        self.assertAlmostEqual(verdict.f_out, 0.5400, places=3)
        self.assertLess(verdict.f_out, 1.0)


class AttributionTests(unittest.TestCase):
    """Regression: the emitted target must be explained by its own run."""

    def test_attribution_follows_the_winning_pipeline(self) -> None:
        # The state the sweep found: damped run is stack-bound at 1012.5,
        # undamped run's pot arm is smaller (956.5) and wins the min.
        composed = compose_aggressive_target(
            boldness=1.0,
            pot=10_000,
            to_call=100,
            effective_stack=3_000,
            contribution=0,
            street="flop",
            bankroll=800,
            exposure=1_000,
            geometric=GeometricSizingParams(enabled=True),
            damper=RuinDamperParams(enabled=True, kappa_r=8.0),
        )
        undamped = compose_aggressive_target(
            boldness=1.0,
            pot=10_000,
            to_call=100,
            effective_stack=3_000,
            contribution=0,
            street="flop",
            bankroll=10**9,  # damper inert
            exposure=1_000,
            geometric=GeometricSizingParams(enabled=True),
        )
        # Same emitted number in both cases...
        self.assertAlmostEqual(composed.target, undamped.target)
        # ...so it must carry the same explanation, not the discarded one.
        self.assertEqual(composed.set_by, undamped.set_by)
        self.assertEqual(composed.boldness_used, undamped.boldness_used)
        self.assertAlmostEqual(
            composed.geometric.f_out, undamped.geometric.f_out
        )

    def test_recorded_verdict_reproduces_the_emitted_target(self) -> None:
        """Whatever run is attributed, its f_out must rebuild the target."""

        rng = random.Random(_FUZZ_SEED + 9)
        geo = GeometricSizingParams(enabled=True)
        damp = RuinDamperParams(enabled=True, kappa_r=8.0)
        checked = 0
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
                geometric=geo,
                damper=damp,
            )
            if composed.set_by != "C2":
                continue
            checked += 1
            expected = state["to_call"] + composed.geometric.f_out * (
                state["pot"] + state["to_call"]
            )
            self.assertAlmostEqual(composed.target, expected, places=6)
        self.assertGreater(checked, 0, "no C2-attributed cases were exercised")


class EscalationTests(unittest.TestCase):
    def test_default_is_the_measured_step_table(self) -> None:
        from engine.rules.escalation_margin import ESCALATION_STEPS

        self.assertEqual(EscalationMarginParams().steps, ESCALATION_STEPS)
        # The steps ARE the measured per-k means minus the k=1 mean, read
        # off the artifact rather than restated here as a slope.
        self.assertAlmostEqual(ESCALATION_STEPS[2], 0.7235 - 0.6436, places=4)
        self.assertAlmostEqual(ESCALATION_STEPS[3], 0.7626 - 0.6436, places=4)

    def test_margin_saturates_at_the_measured_support(self) -> None:
        """Regression: the slope form multiplied an UNBOUNDED count, so a
        long street demanded more than the validator's own ceiling."""

        params = EscalationMarginParams(enabled=True)
        top = escalation_margin(params, 3).margin_applied
        for count in (4, 8, 50):
            self.assertEqual(escalation_margin(params, count).margin_applied, top)
            self.assertLessEqual(escalation_margin(params, count).margin_applied, 0.5)
        self.assertIn("measured support ends", escalation_margin(params, 9).reason)

    def test_steps_must_be_non_decreasing_and_bounded(self) -> None:
        for bad in ((0.0, 0.0, 0.2, 0.1), (0.0, 0.1), (0.0, 0.0, 0.9), (0.0,)):
            with self.assertRaises(ValueError):
                EscalationMarginParams(steps=bad)

    def test_margin_shape(self) -> None:
        params = EscalationMarginParams(enabled=True)
        self.assertEqual(escalation_margin(params, 0).margin_applied, 0.0)
        self.assertEqual(escalation_margin(params, 1).margin_applied, 0.0)
        self.assertAlmostEqual(
            escalation_margin(params, 2).margin_applied, params.steps[2]
        )
        # The journaled number is the one that moved the gate: already
        # scaled by (1 - wildness), with the raw value kept beside it.
        damped = escalation_margin(params, 3, wildness=0.25)
        self.assertAlmostEqual(damped.margin_raw, params.steps[3])
        self.assertAlmostEqual(damped.margin_applied, params.steps[3] * 0.75)

    def test_counter_is_the_street_ordinal(self) -> None:
        table = {
            "recentEvents": [
                {"street": "flop", "summary": {"seatNumber": 1, "action": "bet"}},
                {"street": "flop", "summary": {"seatNumber": 2, "action": "raise"}},
                {"street": "flop", "summary": {"seatNumber": 3, "action": "call"}},
                {"street": "turn", "summary": {"seatNumber": 2, "action": "bet"}},
            ]
        }
        # The street ordinal INCLUDES hero's own bet: that is the
        # quantity kappa_e was measured against (module docstring).
        self.assertEqual(street_aggressions(table, "flop"), 2)
        self.assertEqual(street_aggressions(table, "turn"), 1)


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


class SweepRegressionTests(unittest.TestCase):
    """One test per bug the 2026-08-29 adversarial sweep confirmed."""

    def test_engine_damper_never_grows_a_wager(self) -> None:
        """BLOCKER: the sizer arm is monotone INCREASING in boldness, so a
        damped negative (hot) read raised the pot fraction — the damper
        grew bets in exactly the states it exists to cool."""

        from engine.decision_engine import DecisionEngine
        from engine.rules.composition import RuleLayerParams

        def engine(**kwargs):
            return type(
                "Probe", (DecisionEngine,), {"_family": lambda self, f: "aggress"}
            )(seed=3, **kwargs)

        # Preflop free spot with a weak read: distance-from-river and low
        # equity push the temperature past the setpoint, so boldness is
        # NEGATIVE — the regime where damping used to enlarge the bet.
        table = _snapshot(
            street="preflop",
            board=(),
            to_call=0,
            available=("check", "bet"),
            bet_range=(100, 700),
            raise_range=None,
        )
        allowed = table["allowedActions"]
        weak_equity = 0.05
        off = engine()
        on = engine(
            rule_layer=RuleLayerParams(
                damper=RuinDamperParams(enabled=True, kappa_r=8.0)
            )
        )
        self.assertLess(off._boldness(table, allowed, weak_equity), 0.0)
        sized_off = off._sized_action("bet", table, allowed, weak_equity)
        sized_on = on._sized_action("bet", table, allowed, weak_equity)
        self.assertIsNotNone(sized_off)
        self.assertLessEqual(sized_on[1], sized_off[1])

    def test_snap_band_is_two_sided(self) -> None:
        """A large wager must not snap DOWN onto a tiny covered all-in."""

        params = SnapToCoverParams(enabled=True, band=0.15)
        far_below = snap_to_cover(
            params, to_amount=5_000.0, covered_allin_to_amounts=(100,)
        )
        self.assertFalse(far_below.fired)
        self.assertEqual(far_below.to_amount, 5_000.0)
        # ...while a target inside the band below an all-in still snaps up.
        in_band = snap_to_cover(
            params, to_amount=900.0, covered_allin_to_amounts=(1_000,)
        )
        self.assertTrue(in_band.fired)
        self.assertEqual(in_band.snapped_to, 1_000.0)

    def test_composed_record_cannot_be_half_loaded(self) -> None:
        """The bare-g loader must refuse a composed record rather than
        silently dropping its dial states."""

        from engine.aggression_sizing import (
            DEFAULT_SIZING_PARAMETERS,
            parameters_from_record,
            sizing_record,
        )
        from engine.rules.composition import (
            RuleLayerParams,
            composed_sizing_record,
            parameters_and_rules_from_record,
        )

        rules = RuleLayerParams(
            damper=RuinDamperParams(enabled=True, kappa_r=8.0),
            geometric=GeometricSizingParams(enabled=True),
        )
        record = composed_sizing_record(DEFAULT_SIZING_PARAMETERS, rules)
        with self.assertRaises(ValueError):
            parameters_from_record(record)
        sizing_out, rules_out = parameters_and_rules_from_record(record)
        self.assertEqual(sizing_out, DEFAULT_SIZING_PARAMETERS)
        self.assertEqual(rules_out, rules)
        # ...and the composed loader refuses a bare record symmetrically.
        with self.assertRaises(ValueError):
            parameters_and_rules_from_record(sizing_record())

    def test_free_spot_raise_is_an_active_wager(self) -> None:
        """27 live rows offer 'raise' at to_call == 0 with betRange null."""

        from engine.branch_contract_v9 import legal_branch_labels

        # NO 'all-in' and NO 'bet': only 'raise' can make the active lane
        # legal here. The pre-fix contract passed the with-all-in form
        # trivially, which is why the first version of this test was
        # vacuous.
        labels = legal_branch_labels({"check", "fold", "raise"}, 0)
        self.assertIn("active", labels)
        self.assertIn("passive", labels)
        self.assertNotIn("aggressive", labels)   # escalation-only
        self.assertNotIn("fatal", labels)        # dominated at a free check

    def test_engine_refuses_dials_its_sizer_cannot_honour(self) -> None:
        """C2/C3A act through the composition, which only the v9 serve path
        consults — but feature_extract_v9 DOES apply them. Enabling them on
        this engine would teach a corpus sizes it never plays."""

        from engine.decision_engine import DecisionEngine
        from engine.rules.composition import RuleLayerParams

        def build(**dials):
            return type(
                "Probe", (DecisionEngine,), {"_family": lambda self, f: "check_call"}
            )(rule_layer=RuleLayerParams(**dials))

        for dials in (
            {"geometric": GeometricSizingParams(enabled=True)},
            {"snap": SnapToCoverParams(enabled=True)},
        ):
            with self.assertRaises(ValueError):
                build(**dials)
        # The three the sizer and ladder DO honour construct fine.
        build(
            damper=RuinDamperParams(enabled=True),
            commitment=CommitmentGateParams(enabled=True),
            escalation=EscalationMarginParams(enabled=True),
        )

    def test_damper_verdict_only_journals_when_it_wins(self) -> None:
        """In the hot regime the damped fraction loses the min, so the size
        is byte-identical to dial-off; journaling "sizes cooled" there
        would assert an effect that did not happen."""

        from engine.decision_engine import DecisionEngine
        from engine.rules.composition import RuleLayerParams

        engine = type(
            "Probe", (DecisionEngine,), {"_family": lambda self, f: "aggress"}
        )(
            seed=3,
            rule_layer=RuleLayerParams(
                damper=RuinDamperParams(enabled=True, kappa_r=8.0)
            ),
        )
        table = _snapshot(
            street="preflop", board=(), to_call=0,
            available=("check", "bet"), bet_range=(100, 700), raise_range=None,
        )
        engine._rule_verdicts = []
        engine._sized_action("bet", table, table["allowedActions"], 0.05)
        self.assertEqual(
            [v for v in engine._rule_verdicts if v["rule"] == "C5-ruin-damper"], []
        )

    def test_verdict_accumulator_closes_after_a_decision(self) -> None:
        from engine.decision_engine import DecisionEngine

        engine = type(
            "Probe", (DecisionEngine,), {"_family": lambda self, f: "check_call"}
        )(seed=3)
        engine.decide_with_diagnostics(_snapshot())
        self.assertIsNone(engine._rule_verdicts)


class ParameterHardeningTests(unittest.TestCase):
    def test_bool_and_nan_refused_everywhere(self) -> None:
        for build in (
            lambda: CommitmentGateParams(spr_threshold=True),
            lambda: CommitmentGateParams(spr_threshold=float("nan")),
            lambda: SnapToCoverParams(band=True),
            lambda: SnapToCoverParams(band=0.0),
            lambda: EscalationMarginParams(steps=(0.0, 0.0, True)),
            lambda: EscalationMarginParams(steps=(0.0, 0.0, float("nan"))),
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
