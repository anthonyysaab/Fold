"""L5 engine-coupling tests: hardened catch-alls and the gated shove lane.

Expectations are hand-derived from the spec, never from the code under
test — and every fixture here was rebuilt after an adversarial sweep
showed the first version's premises were hollow (a "never bluff-converts"
test whose hero held trip queens, so the advisor declined on hand
strength and the test passed under the OPPOSITE implementation; an
all-in amount that equalled three different fixture quantities at once;
a shove test that pinned only the big-blind floor). Each test below
names the single rule it kills.

The load-bearing test is `test_unsizeable_escalation_obeys_the_call
_ladder`: it pins the fix for the sweep's blocker, where a v9 flow
answered the risk cap's refusal with an ungated call — 19,000 of a
20,000 stack at 0.05 equity.
"""

from __future__ import annotations

import unittest

from engine.decision_engine import ArenaSnapshotError, DecisionEngine


def _snapshot(**overrides) -> dict:
    from test_rules_composition import _snapshot as base

    return base(**overrides)


def _engine(*, composed: bool = False, **overrides) -> DecisionEngine:
    kwargs = dict(equity_trials=20, seed=7, hyper_aggression_chance=0.0)
    kwargs.update(overrides)
    probe = type(
        "Probe",
        (DecisionEngine,),
        {
            "_family": lambda self, features: "check_call",
            "serves_composed_sizing": composed,
        },
    )
    return probe(**kwargs)


def _unsizeable_escalation(*, opp_stack: int = 8_000, **overrides) -> dict:
    """A priced state whose raise range the risk cap cannot reach.

    Hand-derived: cap = contribution + max(bb, round(0.455 x gate_stack)).
    With contribution 0, bb 100 and gate_stack = effective = opp_stack
    8,000 that is 3,640, below the 4,000 raise minimum — so
    `_sized_action` returns None and `_aggressive_action` falls through.
    """

    table = _snapshot(
        pot=3_000,
        to_call=2_000,
        hero_stack=10_000,
        opp_stack=opp_stack,
        available=("fold", "call", "raise"),
        raise_range=(4_000, 10_000),
        **overrides,
    )
    return table


def _allin_only(*, hero_stack: int = 5_800, contribution: int = 200) -> dict:
    """A priced state whose only wager action is all-in.

    Hero carries a live street contribution so the all-in TO-amount
    (6,000), hero's remaining stack (5,800) and the contribution are
    three DISTINCT numbers — the sweep found the first fixture made all
    three equal, so a mutant shoving the wrong one passed.
    """

    table = _snapshot(
        pot=500,
        to_call=500,
        hero_stack=hero_stack,
        opp_stack=6_000,
        available=("fold", "call", "all-in"),
        raise_range=None,
    )
    hero = table["seats"][0]
    hero["currentBetChips"] = contribution
    allowed = table["allowedActions"]
    allowed["canAllIn"] = True
    allowed["allInToAmount"] = hero_stack + contribution
    return table


def _act(engine: DecisionEngine, table: dict, equity: float):
    allowed = table["allowedActions"]
    available = {str(name) for name in allowed["availableActions"]}
    return engine._aggressive_action(table, allowed, available, equity)


class HardenedCatchAllTests(unittest.TestCase):
    """Both silent-corruption channels for version skew now raise."""

    def test_family_available_refuses_unknown_names(self) -> None:
        # A v9 BRANCH name is not a family: projection belongs to the
        # contract. Before L5 this read as aggress-availability and then
        # fell through to the fold default, silently.
        for bogus in ("aggressive", "fatal", "passive", "garbage"):
            with self.assertRaises(ArenaSnapshotError):
                DecisionEngine._family_available(bogus, {"bet", "raise"})

    def test_family_available_keeps_the_frozen_trio(self) -> None:
        available = {"fold", "check", "call", "bet", "raise"}
        self.assertTrue(DecisionEngine._family_available("fold", available))
        self.assertTrue(DecisionEngine._family_available("check_call", available))
        self.assertTrue(DecisionEngine._family_available("aggress", available))
        self.assertFalse(
            DecisionEngine._family_available("check_call", {"fold", "bet"})
        )
        self.assertFalse(DecisionEngine._family_available("aggress", {"fold", "call"}))

    def test_dispatch_refuses_an_unknown_policy_family(self) -> None:
        # equity_trials=0 gives no equity read, so `_family` is live and
        # its return reaches the dispatch verbatim — the old code
        # silently FOLDED this.
        bogus = type(
            "Bogus", (DecisionEngine,), {"_family": lambda self, features: "bogus"}
        )(equity_trials=0, hyper_aggression_chance=0.0)
        with self.assertRaises(ArenaSnapshotError):
            bogus.decide(_snapshot())

    def test_forced_pin_of_an_unknown_family_raises(self) -> None:
        engine = _engine()
        with self.assertRaises(ArenaSnapshotError):
            engine.decide_forced(_snapshot(), family="aggressive")


class DemotionLadderTests(unittest.TestCase):
    """The sweep's blocker: an unsizeable escalation must obey the gates."""

    def test_unsizeable_escalation_obeys_the_call_ladder(self) -> None:
        # Sanity FIRST — the state must genuinely reach the fallthrough.
        table = _unsizeable_escalation()
        allowed = table["allowedActions"]
        self.assertIsNone(_engine()._sized_action("raise", table, allowed, 0.30))

        # Hand-derived: pot odds 2000/5000 = 0.40, turn margin 0.05, so a
        # call needs ~0.45+ equity even before the temperature shave.
        # 0.30 is below it -> fold, on BOTH engines. Before the fix the
        # composed engine returned an ungated ('call', None) here.
        for composed in (False, True):
            with self.subTest(composed=composed):
                self.assertEqual(
                    _act(_engine(composed=composed), table, 0.30), ("fold", None)
                )

    def test_the_two_engines_agree_wherever_no_shove_is_gated_in(self) -> None:
        # The demotion is `_passive_action`, shared by both engines, so
        # sub-near-nut states must produce IDENTICAL actions.
        table = _unsizeable_escalation()
        for equity in (0.05, 0.20, 0.30, 0.40, 0.60):
            with self.subTest(equity=equity):
                self.assertEqual(
                    _act(_engine(composed=True), table, equity),
                    _act(_engine(composed=False), table, equity),
                )

    def test_a_justified_price_still_calls(self) -> None:
        """The ladder is a gate, not a veto.

        Note the window this has to live in: "unsizeable" is itself
        equity-dependent, because the risk cap RELEASES at
        `near_nut_floor` (0.654) — at 0.90 this same state sizes a
        raise to 5,336 and never reaches the fallthrough. So the test
        needs equity above the call's price (~0.45 = pot odds 0.40 +
        turn margin 0.05) and below the cap's release.
        """

        table = _unsizeable_escalation()
        allowed = table["allowedActions"]
        self.assertIsNone(_engine()._sized_action("raise", table, allowed, 0.60))
        self.assertEqual(_act(_engine(composed=True), table, 0.60), ("call", None))


class GatedShoveLaneTests(unittest.TestCase):
    """One release only: near-nut equity."""

    def test_near_nut_escalation_shoves_the_all_in_to_amount(self) -> None:
        table = _allin_only()
        # 6,000 is allInToAmount and is distinct from hero's stack
        # (5,800) and contribution (200) — so this pins the right field.
        self.assertEqual(_act(_engine(composed=True), table, 0.90), ("all-in", 6_000))

    def test_the_near_nut_floor_boundary_is_pinned(self) -> None:
        # near_nut_floor is 0.654: the boundary pair kills a mutant that
        # moves the floor, which the first version of this test did not.
        table = _allin_only()
        engine = _engine(composed=True)
        floor = engine.safety_gates.near_nut_floor
        self.assertEqual(_act(engine, table, floor), ("all-in", 6_000))
        below = _act(engine, table, floor - 0.001)
        self.assertNotEqual(below, ("all-in", 6_000))

    def test_sub_near_nut_never_shoves_even_on_a_collapsed_stack(self) -> None:
        # The deferred second arm: an all-in opponent collapses
        # effective_stack_chips to 0, which the first draft read as
        # "the cap already commits the stack" and shoved on at ANY
        # equity. It must not fire.
        table = _allin_only()
        table["seats"][1]["stackChips"] = 0
        for equity in (0.05, 0.30, 0.50):
            with self.subTest(equity=equity):
                self.assertNotEqual(
                    _act(_engine(composed=True), table, equity)[0], "all-in"
                )

    def test_base_engine_never_shoves(self) -> None:
        # The v0 rule: the warm-start model never controls an optional
        # all-in. This is the guard that keeps L5 off every legacy path.
        table = _allin_only()
        for equity in (0.30, 0.90):
            with self.subTest(equity=equity):
                self.assertNotEqual(_act(_engine(), table, equity)[0], "all-in")

    def test_shove_lane_needs_an_equity_read(self) -> None:
        table = _allin_only()
        allowed = table["allowedActions"]
        available = {str(name) for name in allowed["availableActions"]}
        self.assertIsNone(
            _engine(composed=True)._gated_shove(table, allowed, available, None)
        )

    def test_a_malformed_all_in_amount_raises(self) -> None:
        # A malformed snapshot is not a policy choice — the same posture
        # `_first_legal_aggression` takes on the identical field. The
        # first draft returned None and silently degraded a near-nut
        # shove into whatever followed.
        table = _allin_only()
        table["allowedActions"]["allInToAmount"] = None
        allowed = table["allowedActions"]
        available = {str(name) for name in allowed["availableActions"]}
        with self.assertRaises(ArenaSnapshotError):
            _engine(composed=True)._gated_shove(table, allowed, available, 0.95)


class SplitWagerFloorTests(unittest.TestCase):
    """The active/aggressive floor split, v9-scoped and default-inert."""

    def test_default_premium_reproduces_the_unsplit_floor(self) -> None:
        # Ships 0.0: both lanes must return the base floor unchanged, so
        # the split is behaviourally inert until someone measures a value.
        engine = _engine(composed=True)
        table = _snapshot()
        self.assertEqual(engine.safety_gates.escalation_floor_premium, 0.0)
        for to_call in (0, 200):
            with self.subTest(to_call=to_call):
                self.assertAlmostEqual(
                    engine._composed_wager_floor(table, 0.60, to_call), 0.60
                )

    def test_the_premium_lifts_only_the_escalation_lane(self) -> None:
        from engine.decision_engine import SafetyGates

        engine = _engine(
            composed=True,
            safety_gates=SafetyGates(escalation_floor_premium=0.08),
        )
        table = _snapshot()
        # Hand-derived: the continuation floor is untouched at 0.60; the
        # escalation floor is 0.60 + 0.08.
        self.assertAlmostEqual(engine._composed_wager_floor(table, 0.60, 0), 0.60)
        self.assertAlmostEqual(engine._composed_wager_floor(table, 0.60, 200), 0.68)

    def test_escalation_is_never_below_continuation_at_any_premium(self) -> None:
        # The invariant is enforced AFTER the clamp, so it must hold even
        # where the clamp binds. Swept rather than argued.
        from engine.decision_engine import SafetyGates

        table = _snapshot()
        for premium in (0.0, 0.01, 0.10, 0.30):
            engine = _engine(
                composed=True,
                safety_gates=SafetyGates(escalation_floor_premium=premium),
            )
            for base in (0.0, 0.05, 0.30, 0.60, 0.90, 0.95, 1.0):
                with self.subTest(premium=premium, base=base):
                    active = engine._composed_wager_floor(table, base, 0)
                    escalation = engine._composed_wager_floor(table, base, 200)
                    self.assertGreaterEqual(escalation, active)
                    self.assertGreaterEqual(active, 0.05)
                    self.assertLessEqual(escalation, 0.95)

    def test_the_premium_is_validated(self) -> None:
        from engine.decision_engine import SafetyGates

        for bad in (-0.01, 0.31):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    SafetyGates(escalation_floor_premium=bad)

    def test_legacy_engines_never_reach_the_split(self) -> None:
        # `serves_composed_sizing` is the whole guard: a legacy engine's
        # chosen wager must be judged by the unsplit floor, so the two
        # engines agree wherever the composed path adds nothing.
        table = _snapshot(
            pot=400, to_call=0, available=("check", "bet"), bet_range=(100, 6_000)
        )
        for equity in (0.20, 0.50, 0.80):
            with self.subTest(equity=equity):
                self.assertEqual(
                    _act(_engine(), table, equity),
                    _act(_engine(composed=True), table, equity),
                )


class WildnessBlendRestorationTests(unittest.TestCase):
    """The v9 projection had silently dropped the tracked-wildness blend."""

    def _tracked(self, engine: DecisionEngine, table: dict, wildness: float):
        engine.opponent_tracker.max_active_wildness = (  # type: ignore[method-assign]
            lambda _table, _w=wildness: _w
        )
        return engine

    def test_wildness_raises_the_bar_toward_the_war_floor(self) -> None:
        # Hand-derived from the blend's own formula:
        # (1 - w) * base + w * near_nut_floor, with base 0.30, w 0.5 and
        # near_nut_floor 0.654 -> 0.477.
        engine = self._tracked(_engine(composed=True), _snapshot(), 0.5)
        self.assertAlmostEqual(
            engine._composed_wager_floor(_snapshot(), 0.30, 200), 0.477
        )

    def test_no_wildness_leaves_the_floor_alone(self) -> None:
        engine = self._tracked(_engine(composed=True), _snapshot(), 0.0)
        self.assertAlmostEqual(
            engine._composed_wager_floor(_snapshot(), 0.30, 200), 0.30
        )

    def test_a_blocked_escalation_demotes_through_the_gated_call(self) -> None:
        """The demotion is `_passive_action`, not a bare call — the
        distinction the earlier L5 draft got wrong at a cost of 19,000
        chips at 0.05 equity."""

        table = _snapshot(
            pot=500,
            to_call=690,
            hole=("2c", "7d"),
            board=("Qs", "Jh", "9s", "3d"),
            available=("fold", "call", "raise"),
            raise_range=(1_380, 6_000),
            hero_stack=6_000,
            opp_stack=6_000,
        )
        engine = self._tracked(_engine(composed=True), table, 1.0)
        # Wildness 1.0 pins the floor at near_nut_floor, so a 0.30-equity
        # escalation is blocked; the price is bad, so the gated call
        # refuses too and the demotion lands on a fold.
        self.assertEqual(_act(engine, table, 0.30), ("fold", None))


class LegacyByteIdentityTests(unittest.TestCase):
    def test_legacy_forced_families_still_route_through_the_ladder(self) -> None:
        # A forced check_call at a terrible price must still FOLD via
        # `_call_clears_margin` — the v7/v8 semantics, unchanged. Hand-
        # derived: pot odds 690/1190 = 0.58, turn margin 0.05, against a
        # weak holding whose equity is far below.
        table = _snapshot(
            pot=500,
            to_call=690,
            hole=("2c", "7d"),
            board=("Qs", "Jh", "9s", "3d"),
            available=("fold", "call", "raise"),
        )
        payload = _engine().decide_forced(table, family="check_call")
        self.assertEqual(payload["action"], "fold")

    def test_legacy_families_stay_in_the_frozen_label_set(self) -> None:
        from engine.policy_features import LABELS

        # Every family the engine can report must be indexable in LABELS:
        # the sweep found a widened vocabulary produced an all-zero
        # one-hot that the telemetry contract refuses.
        table = _snapshot(pot=400, to_call=0, available=("check", "bet"),
                          bet_range=(100, 6_000))
        result = _engine().decide_with_diagnostics(table)
        self.assertIn(result.family, LABELS)
        self.assertAlmostEqual(sum(result.behavior_probabilities), 1.0)


if __name__ == "__main__":
    unittest.main()
