"""L5 engine-coupling tests: forced vocabulary, catch-alls, shove lane.

Expectations are hand-derived from the specs, never from the code under
test. The two byte-identity contrasts are load-bearing: the SAME state
that a v9 flow demotes to a call must still fold through the legacy
ladder on a base engine, and the legacy ``check_call`` forced family
must still be gate-laddered on the exact state where the v9 ``call``
family executes literally.
"""

from __future__ import annotations

import unittest

from engine.decision_engine import ArenaSnapshotError, DecisionEngine


def _snapshot(**overrides) -> dict:
    from test_rules_composition import _snapshot as base

    return base(**overrides)


def _base_engine(**overrides) -> DecisionEngine:
    kwargs = dict(equity_trials=20, seed=7, hyper_aggression_chance=0.0)
    kwargs.update(overrides)
    probe = type(
        "Probe", (DecisionEngine,), {"_family": lambda self, features: "fold"}
    )
    return probe(**kwargs)


def _v9_engine(**overrides) -> DecisionEngine:
    kwargs = dict(equity_trials=20, seed=7, hyper_aggression_chance=0.0)
    kwargs.update(overrides)
    probe = type(
        "V9Probe",
        (DecisionEngine,),
        {
            "_family": lambda self, features: "check_call",
            "serves_composed_sizing": True,
        },
    )
    return probe(**kwargs)


def _allin_only_priced(*, opp_stack: int = 6_000, to_call: int = 500) -> dict:
    """A priced state whose only wager action is all-in."""

    table = _snapshot(
        pot=500,
        to_call=to_call,
        hero_stack=6_000,
        opp_stack=opp_stack,
        available=("fold", "call", "all-in"),
        raise_range=None,
    )
    table["allowedActions"]["canAllIn"] = True
    table["allowedActions"]["allInToAmount"] = 6_000
    return table


class ForcedVocabularyTests(unittest.TestCase):
    """The v9-grown forced families are literal; the legacy trio is not."""

    def test_forced_call_is_literal_where_the_ladder_would_fold(self) -> None:
        # A terrible price with a weak holding: the call-margin ladder
        # refuses this call (the legacy contrast below proves it), but
        # the v9 'call' family executes the contract action literally.
        table = _snapshot(
            pot=500,
            to_call=690,
            hole=("2c", "7d"),
            board=("Qs", "Jh", "9s", "3d"),
            available=("fold", "call", "raise"),
        )
        engine = _base_engine()
        payload = engine.decide_forced(table, family="call")
        self.assertEqual(payload["action"], "call")

    def test_legacy_check_call_is_still_gate_laddered(self) -> None:
        # The SAME state through the legacy family: no check exists, the
        # call fails the margin, and the ladder folds — the v8 semantics,
        # bit for bit.
        table = _snapshot(
            pot=500,
            to_call=690,
            hole=("2c", "7d"),
            board=("Qs", "Jh", "9s", "3d"),
            available=("fold", "call", "raise"),
        )
        engine = _base_engine()
        payload = engine.decide_forced(table, family="check_call")
        self.assertEqual(payload["action"], "fold")

    def test_forced_check_is_literal_and_never_bluff_converted(self) -> None:
        # The literal arm skips the bluff mixer structurally; across
        # several salted table ids not one forced check may convert.
        for index in range(8):
            table = _snapshot(
                pot=400,
                to_call=0,
                available=("check", "bet"),
                bet_range=(100, 6_000),
            )
            table["tableId"] = table["id"] = f"forced-check-{index}"
            engine = _base_engine()
            payload = engine.decide_forced(table, family="check")
            self.assertEqual(payload["action"], "check")

    def test_unavailable_pin_dissolves_into_the_policy_choice(self) -> None:
        # 'check' cannot execute at a price; the pin dissolves and the
        # stub policy's own fold family decides (weak hand: no rescue).
        table = _snapshot(
            pot=500,
            to_call=690,
            hole=("2c", "7d"),
            board=("Qs", "Jh", "9s", "3d"),
            available=("fold", "call", "raise"),
        )
        engine = _base_engine()
        payload = engine.decide_forced(table, family="check")
        self.assertEqual(payload["action"], "fold")

    def test_unknown_forced_family_raises(self) -> None:
        table = _snapshot()
        engine = _base_engine()
        # A v9 BRANCH name is not a family: projection belongs to the
        # contract, and this raising is the version-skew tripwire.
        for bogus in ("aggressive", "fatal", "passive", "garbage"):
            with self.assertRaises(ArenaSnapshotError):
                engine.decide_forced(table, family=bogus)


class HardenedCatchAllTests(unittest.TestCase):
    def test_family_available_refuses_unknown_names(self) -> None:
        with self.assertRaises(ArenaSnapshotError):
            DecisionEngine._family_available("aggressive", {"bet", "raise"})

    def test_dispatch_refuses_an_unknown_policy_family(self) -> None:
        # equity_trials=0 gives no equity read, so the family comes from
        # the backend verbatim — the old code silently FOLDED this.
        bogus = type(
            "Bogus", (DecisionEngine,), {"_family": lambda self, features: "bogus"}
        )(equity_trials=0, hyper_aggression_chance=0.0)
        with self.assertRaises(ArenaSnapshotError):
            bogus.decide(_snapshot())

    def test_legacy_families_still_available_check(self) -> None:
        available = {"fold", "check", "call", "bet", "raise"}
        self.assertTrue(DecisionEngine._family_available("fold", available))
        self.assertTrue(DecisionEngine._family_available("check_call", available))
        self.assertTrue(DecisionEngine._family_available("aggress", available))
        self.assertTrue(DecisionEngine._family_available("check", available))
        self.assertTrue(DecisionEngine._family_available("call", available))
        self.assertFalse(DecisionEngine._family_available("check", {"fold", "call"}))
        self.assertFalse(DecisionEngine._family_available("call", {"fold", "check"}))


class GatedShoveLaneTests(unittest.TestCase):
    """Direct units on _aggressive_action with hand-pinned equities."""

    def _act(self, engine: DecisionEngine, table: dict, equity: float):
        allowed = table["allowedActions"]
        available = {str(name) for name in allowed["availableActions"]}
        return engine._aggressive_action(table, allowed, available, equity)

    def test_near_nut_escalation_shoves_on_the_v9_flow(self) -> None:
        table = _allin_only_priced()
        action = self._act(_v9_engine(), table, equity=0.9)
        # 0.9 >= near_nut_floor (0.654): the gated lane fires.
        self.assertEqual(action, ("all-in", 6_000))

    def test_sub_near_nut_deep_stack_demotes_to_a_call_never_a_fold(self) -> None:
        table = _allin_only_priced()
        action = self._act(_v9_engine(), table, equity=0.30)
        # Deep effective stack: the risk cap (0.455 x 6,000 = 2,730 as a
        # to-amount) does not commit the 6,000 effective stack, so no
        # shove — and the demotion rule returns the active branch's call
        # instead of cascading into the fold ladder.
        self.assertEqual(action, ("call", None))

    def test_effective_stack_collapse_releases_the_shove(self) -> None:
        # The opponent is all-in with nothing behind: effective stack
        # clamps to 1, the capped maximum (>= one big blind) already
        # commits it, and the shove's real risk is the price of a call
        # (the Arena refunds the uncalled excess — the measured benign
        # collapse). Fires even at weak equity.
        table = _allin_only_priced(opp_stack=0)
        action = self._act(_v9_engine(), table, equity=0.30)
        self.assertEqual(action, ("all-in", 6_000))

    def test_base_engine_behaviour_is_untouched(self) -> None:
        # The identical states through a non-v9 engine: no shove lane,
        # no demotion — the v0 rule stands (never an optional all-in)
        # and the passive fallback folds the weak hand at this price.
        table = _allin_only_priced()
        self.assertEqual(
            self._act(_base_engine(), table, equity=0.9),
            ("call", None),
        )
        # equity 0.9 clears the call margin, so the passive fallback
        # calls — but via the LADDER, not the shove lane.
        self.assertEqual(
            self._act(_base_engine(), table, equity=0.30),
            ("fold", None),
        )

    def test_shove_lane_needs_an_equity_read(self) -> None:
        table = _allin_only_priced()
        engine = _v9_engine()
        allowed = table["allowedActions"]
        available = {str(name) for name in allowed["availableActions"]}
        self.assertIsNone(engine._gated_shove(table, allowed, available, None))


if __name__ == "__main__":
    unittest.main()
