"""Tests for the v9 Phase-B harvester.

The load-bearing test is organic and end to end: a tiny harvest with a
REAL v9 policy (the zero-weight format-4 artifact, loaded through
``load_policy_v9``) writes a corpus that is then loaded through the
TRAINER's own ``load_phase_b_corpus_v9`` — the schema-2 gate, the
frozen-g sizing re-derivation, and emission-equals-legality all run on
organically harvested rows, and the loader's derived constants are then
checked against the serve composition on random head outputs. Candidate
sets and sizing fields are asserted against HAND-COMPUTED expectations
from the g spec (b = 0: aggressive f = 0.75, s = 0.325; active f = 0.5),
never against the composition functions themselves.

The P3 swap physics, arrangement parity, and selection machinery are the
v8 harvester's own inherited code and stay covered by
``test_build_phase_b_corpus.py``; here one P3-seat smoke asserts the v9
wiring composes with them.
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from engine.branch_contract_v9 import BRANCH_LABELS_V9
from engine.learned_policy_v9 import compose_branch_values_v9
from engine.p3_belief_provider import P3BeliefProvider
from engine.v9_trainer_phase_b import (
    compose_from_constants_v9,
    load_phase_b_corpus_v9,
)
from tools.build_phase_b_corpus_v9 import (
    ContractForcingRecorder,
    PhaseBHarvestSimulatorV9,
    corpus_header_v9,
    corpus_statistics,
    expected_executions,
    write_phase_b_corpus_v9,
)


def _provider() -> P3BeliefProvider:
    return P3BeliefProvider.from_artifact()


def _capture(**overrides) -> dict:
    from test_rules_composition import _snapshot

    # Deep stacks by default so the hand-derived expectations exercise
    # the pot arm; a test that wants the stack arm passes shallow ones.
    overrides.setdefault("hero_stack", 6_000)
    overrides.setdefault("opp_stack", 6_000)
    return _snapshot(**overrides)


def _bare_simulator(**overrides) -> PhaseBHarvestSimulatorV9:
    """A harvester instance for unit-level calls (no match is played)."""

    kwargs = dict(
        small_blind=50,
        big_blind=100,
        starting_stack=6_000,
        seed=3,
        collect_counterfactuals=True,
        counterfactual_rollouts=1,
        hero_id="hero",
        hero_recorder=None,
        leg_name="unit",
        potential_trials=40,
        belief_provider=_provider(),
    )
    kwargs.update(overrides)
    return PhaseBHarvestSimulatorV9(**kwargs)


class ExpectedExecutionTests(unittest.TestCase):
    """The contract's execution table, hand-written (never derived)."""

    def test_the_acceptance_table(self) -> None:
        self.assertEqual(expected_executions("fatal", False), {"fold"})
        self.assertEqual(expected_executions("passive", True), {"check"})
        self.assertEqual(expected_executions("active", False), {"call"})
        self.assertEqual(
            expected_executions("active", True), {"bet", "raise", "all-in"}
        )
        self.assertEqual(
            expected_executions("aggressive", False), {"raise", "all-in"}
        )

    def test_acceptance_sets_are_disjoint_within_each_state(self) -> None:
        """Why acceptance alone implies distinct executed actions."""

        priced = [
            expected_executions(branch, False)
            for branch in ("fatal", "active", "aggressive")
        ]
        free = [
            expected_executions(branch, True) for branch in ("passive", "active")
        ]
        for state in (priced, free):
            for index, left in enumerate(state):
                for right in state[index + 1 :]:
                    self.assertFalse(left & right)


class CandidateSetTests(unittest.TestCase):
    """Contract candidates and sizing fields, hand-derived at b = 0."""

    def test_priced_state_candidates(self) -> None:
        simulator = _bare_simulator()
        capture = _capture(
            pot=300,
            to_call=100,
            available=("fold", "call", "raise", "all-in"),
            raise_range=(200, 6_000),
        )
        context = simulator._context_v9(capture)
        self.assertEqual(
            context["legal_labels"], ["fatal", "active", "aggressive"]
        )
        self.assertEqual(context["raise_range"], (200, 6_000))
        candidates, sizing_fields = simulator._candidates_v9(context, 0.0)
        self.assertEqual(
            [(label, family) for label, family, _ in candidates],
            [("fatal", "fold"), ("active", "check_call"), ("aggressive", "aggress")],
        )
        # Hand-derived at b = 0, dials off: f = 0.75, s = 0.325 ->
        # target = min(100 + 0.75*400, 0.325*eff) = 400; the forced
        # fraction is (400-100)/400 = 0.75; to_amount = 400 in range.
        fractions = {label: fraction for label, _, fraction in candidates}
        self.assertIsNone(fractions["fatal"])
        self.assertIsNone(fractions["active"])
        self.assertAlmostEqual(fractions["aggressive"], 0.75)
        self.assertEqual(sizing_fields, {"aggressive": (400.0, 400.0)})

    def test_free_state_candidates(self) -> None:
        simulator = _bare_simulator()
        capture = _capture(
            pot=500,
            to_call=0,
            available=("check", "bet"),
            bet_range=(100, 6_000),
            raise_range=None,
        )
        context = simulator._context_v9(capture)
        self.assertEqual(context["legal_labels"], ["passive", "active"])
        candidates, sizing_fields = simulator._candidates_v9(context, 0.0)
        self.assertEqual(
            [(label, family) for label, family, _ in candidates],
            [("passive", "check_call"), ("active", "aggress")],
        )
        # b = 0 -> f = 0.5 -> wager 250 of pot 500; fraction 0.5.
        fractions = {label: fraction for label, _, fraction in candidates}
        self.assertAlmostEqual(fractions["active"], 0.5)
        self.assertEqual(sizing_fields, {"active": (250.0, 250.0)})

    def test_blind_option_free_spot_uses_the_raise_range_fallback(self) -> None:
        """The Arena names the unprovoked wager 'raise' at blind-option
        spots; the bet lane resolves through raiseRange exactly as the
        serve path does."""

        simulator = _bare_simulator()
        capture = _capture(
            pot=150,
            to_call=0,
            available=("check", "raise", "all-in", "fold"),
            bet_range=None,
            raise_range=(200, 6_000),
        )
        context = simulator._context_v9(capture)
        self.assertEqual(context["legal_labels"], ["passive", "active"])
        self.assertEqual(context["bet_range"], (200, 6_000))
        candidates, sizing_fields = simulator._candidates_v9(context, 0.0)
        # b = 0 -> wager 75 of pot 150, to_amount 75 clamped UP to the
        # range minimum 200.
        self.assertEqual(sizing_fields, {"active": (75.0, 200.0)})
        self.assertEqual(len(candidates), 2)

    def test_allin_only_escalation_keeps_the_unclamped_amount(self) -> None:
        simulator = _bare_simulator()
        capture = _capture(
            pot=300,
            to_call=100,
            available=("fold", "call", "all-in"),
            raise_range=None,
        )
        context = simulator._context_v9(capture)
        self.assertEqual(
            context["legal_labels"], ["fatal", "active", "aggressive"]
        )
        self.assertIsNone(context["raise_range"])
        _, sizing_fields = simulator._candidates_v9(context, 0.0)
        target, to_amount = sizing_fields["aggressive"]
        self.assertAlmostEqual(target, 400.0)
        self.assertAlmostEqual(to_amount, 400.0)  # no range: unclamped


class PurityVerdictTests(unittest.TestCase):
    def test_clean_probe_passes_and_counts_nothing(self) -> None:
        simulator = _bare_simulator()
        candidates = [
            ("fatal", "fold", None),
            ("active", "check_call", None),
            ("aggressive", "aggress", 0.75),
        ]
        executed = {
            "fatal": ("fold", None),
            "active": ("call", None),
            "aggressive": ("raise", 400),
        }
        self.assertIsNone(
            simulator._purity_verdict(candidates, executed, to_call_zero=False)
        )
        self.assertEqual(simulator.probe_action_mismatches, Counter())
        self.assertEqual(simulator.probe_collisions, 0)

    def test_rail_retargeted_branches_are_dropped_and_classified(self) -> None:
        simulator = _bare_simulator()
        candidates = [("fatal", "fold", None), ("active", "check_call", None)]
        cases = {
            # The rescue rail calling a forced fold.
            "fatal->call": {"fatal": ("call", None), "active": ("call", None)},
            # A gate folding a forced call.
            "active->fold": {"fatal": ("fold", None), "active": ("fold", None)},
        }
        for expected_key, executed in cases.items():
            verdict = simulator._purity_verdict(
                candidates, executed, to_call_zero=False
            )
            self.assertIsNotNone(verdict)
            self.assertEqual(simulator.probe_action_mismatches[expected_key], 1)

    def test_pre_l5_allin_only_demotion_is_classified(self) -> None:
        """Until the L5 shove lane lands, a forced escalation at an
        all-in-only state demotes to a call; the counter names it."""

        simulator = _bare_simulator()
        candidates = [
            ("fatal", "fold", None),
            ("active", "check_call", None),
            ("aggressive", "aggress", 0.75),
        ]
        executed = {
            "fatal": ("fold", None),
            "active": ("call", None),
            "aggressive": ("call", None),
        }
        verdict = simulator._purity_verdict(
            candidates, executed, to_call_zero=False
        )
        self.assertIsNotNone(verdict)
        self.assertEqual(
            simulator.probe_action_mismatches["aggressive->call"], 1
        )


class _StubForcedPolicy:
    """Records forced calls; stands in for a real engine policy."""

    policy_version = "stub"

    def __init__(self) -> None:
        self.forced_calls: list[tuple[str, float | None]] = []

    def decide_forced(self, table, *, family, pot_fraction=None) -> dict:
        self.forced_calls.append((family, pot_fraction))
        return {"action": "raise", "amount": 400, "message": "engine"}


class ContractForcingTests(unittest.TestCase):
    """The forcing table: literal contract actions, engine wagers."""

    def test_fold_and_check_call_are_literal(self) -> None:
        stub = _StubForcedPolicy()
        recorder = ContractForcingRecorder(stub)
        priced = _capture(to_call=100, available=("fold", "call", "raise"))
        free = _capture(
            to_call=0, available=("check", "bet"), bet_range=(100, 6_000)
        )
        self.assertEqual(
            recorder.decide_forced(priced, family="fold")["action"], "fold"
        )
        self.assertEqual(
            recorder.decide_forced(free, family="check_call")["action"], "check"
        )
        self.assertEqual(
            recorder.decide_forced(priced, family="check_call")["action"], "call"
        )
        # The rails never saw any of it: the policy was not consulted.
        self.assertEqual(stub.forced_calls, [])

    def test_sized_wagers_still_route_through_the_engine(self) -> None:
        stub = _StubForcedPolicy()
        recorder = ContractForcingRecorder(stub)
        priced = _capture(
            to_call=100,
            available=("fold", "call", "raise"),
            raise_range=(200, 6_000),
        )
        payload = recorder.decide_forced(
            priced, family="aggress", pot_fraction=0.75
        )
        self.assertEqual(payload["action"], "raise")
        self.assertEqual(stub.forced_calls, [("aggress", 0.75)])

    def test_allin_only_escalation_is_the_literal_shove(self) -> None:
        stub = _StubForcedPolicy()
        recorder = ContractForcingRecorder(stub)
        table = {
            "allowedActions": {
                "availableActions": ["fold", "call", "all-in"],
                "betRange": None,
                "raiseRange": None,
                "allInToAmount": 6_000,
            }
        }
        payload = recorder.decide_forced(table, family="aggress", pot_fraction=0.6)
        self.assertEqual(payload["action"], "all-in")
        self.assertEqual(payload["amount"], 6_000)
        self.assertEqual(stub.forced_calls, [])


# ---------------------------------------------------------------------------
# Organic end-to-end harvests (cached: each plays once per test session)
# ---------------------------------------------------------------------------

_ARTIFACT_DIR: tempfile.TemporaryDirectory | None = None
_V9_HARVEST: PhaseBHarvestSimulatorV9 | None = None
_P3_V9_HARVEST: PhaseBHarvestSimulatorV9 | None = None


def _hero_recorder():
    from test_learned_policy_v9 import _write_artifact

    from engine.learned_policy_v9 import load_policy_v9

    global _ARTIFACT_DIR
    if _ARTIFACT_DIR is None:
        _ARTIFACT_DIR = tempfile.TemporaryDirectory()
        _write_artifact(Path(_ARTIFACT_DIR.name))
    manifest = Path(_ARTIFACT_DIR.name) / "candidate-v9-test.manifest.json"
    policy = load_policy_v9(
        manifest,
        equity_trials=20,
        belief_provider=_provider(),
        potential_trials=40,
    )
    return ContractForcingRecorder(policy)


def _play_harvest(
    *, seed: int, with_p3: bool, hands: int
) -> PhaseBHarvestSimulatorV9:
    from engine.strength_aware_opponent import StrengthAwareAgent, load_fit
    from engine.table_simulator import ScriptedAgent
    from tools.build_phase_b_corpus import P3SeatWrapper

    recorder = _hero_recorder()
    simulator = PhaseBHarvestSimulatorV9(
        small_blind=50,
        big_blind=100,
        starting_stack=6_000,
        seed=seed,
        collect_counterfactuals=True,
        counterfactual_rollouts=2,
        hero_id="hero",
        hero_recorder=recorder,
        leg_name="v9-mini",
        potential_trials=40,
        belief_provider=_provider(),
    )
    agents = [("hero", recorder)]
    if with_p3:
        agents.append(
            (
                "p3",
                P3SeatWrapper(
                    StrengthAwareAgent("p3", 0.226, 0.5, 0.0, 11, fit=load_fit())
                ),
            )
        )
    agents.append(("median", ScriptedAgent("median", 0.226, 0.5, 0.0, seed=6)))
    simulator.play_match(agents, hands=hands)
    return simulator


def _v9_harvest() -> PhaseBHarvestSimulatorV9:
    global _V9_HARVEST
    if _V9_HARVEST is None:
        _V9_HARVEST = _play_harvest(seed=8, with_p3=False, hands=10)
    return _V9_HARVEST


def _p3_v9_harvest() -> PhaseBHarvestSimulatorV9:
    global _P3_V9_HARVEST
    if _P3_V9_HARVEST is None:
        _P3_V9_HARVEST = _play_harvest(seed=9, with_p3=True, hands=8)
    return _P3_V9_HARVEST


def _write_and_load(simulator: PhaseBHarvestSimulatorV9, directory: Path):
    from engine.rules.composition import composed_sizing_record

    header = corpus_header_v9(
        sizing_record=composed_sizing_record(),
        belief_fit_source=_provider().fit_source,
        equity_trials=20,
        starting_stack=6_000,
        big_blind=100,
        seeds=[simulator.seed],
    )
    path = directory / "mini.phase-b.jsonl.gz"
    write_phase_b_corpus_v9(path, header, simulator.phase_b_rows)
    return path, load_phase_b_corpus_v9(path)


class EndToEndHarvestTests(unittest.TestCase):
    """Organic rows through the REAL trainer loader — the L4 proof."""

    def test_harvest_emits_rows_and_accounting_balances(self) -> None:
        simulator = _v9_harvest()
        self.assertGreater(simulator.decisions_emitted, 0)
        dropped = (
            sum(simulator.probe_action_mismatches.values())
            + simulator.probe_collisions
        )
        self.assertEqual(
            simulator.decisions_selected,
            simulator.decisions_emitted
            + simulator.single_branch_groups
            + dropped,
        )

    def test_trainer_loader_accepts_the_organic_corpus(self) -> None:
        simulator = _v9_harvest()
        with tempfile.TemporaryDirectory() as raw:
            _, corpus = _write_and_load(simulator, Path(raw))
        self.assertEqual(len(corpus.decisions), simulator.decisions_emitted)
        self.assertEqual(corpus.equity_trials, 20)
        for decision in corpus.decisions:
            self.assertGreaterEqual(len(decision.emitted), 2)
            for label in decision.emitted:
                self.assertIn(label, BRANCH_LABELS_V9)
            # Wager constants exist exactly where wagers executed.
            if decision.to_call_zero:
                self.assertIsNotNone(decision.wager_unit[0])
                self.assertIsNone(decision.wager_unit[1])
            elif "aggressive" in decision.emitted:
                self.assertIsNone(decision.wager_unit[0])
                self.assertIsNotNone(decision.wager_unit[1])

    def test_organic_constants_reproduce_the_serve_composition(self) -> None:
        """Loader-derived constants vs the live serve composition on the
        recorded context — over ORGANIC rows, not synthetic fixtures."""

        from engine.aggression_sizing import context_int_to_temperature

        simulator = _v9_harvest()
        with tempfile.TemporaryDirectory() as raw:
            _, corpus = _write_and_load(simulator, Path(raw))
        rng = random.Random(23)
        for decision in corpus.decisions:
            for _ in range(5):
                outputs = {
                    "fold_through": [rng.uniform(-4, 4) for _ in range(2)],
                    "range": [rng.uniform(-1, 1) for _ in range(8)],
                    "equity_called": [rng.uniform(-0.3, 1.3) for _ in range(3)],
                    "residual": [rng.uniform(-0.5, 0.5) for _ in range(4)],
                }
                mine = compose_from_constants_v9(outputs, decision)
                context = decision.context
                boldness = corpus.sizing.boldness(
                    context_int_to_temperature(context["read_temperature_x10"])
                )
                serve, _ = compose_branch_values_v9(
                    outputs,
                    pot=context["pot"],
                    to_call=context["to_call"],
                    contribution=context["contribution"],
                    effective_stack=context["effective_stack"],
                    purse=context["purse"],
                    boldness=boldness,
                    street=context["street"],
                    bankroll=context["bankroll"],
                    exposure=context["exposure"],
                    covered_allin_to_amounts=tuple(
                        context["covered_allin_to_amounts"]
                    ),
                    legal_labels=frozenset(decision.emitted),
                    bet_range=context["bet_range"],
                    raise_range=context["raise_range"],
                    sizing=corpus.sizing,
                    rules=corpus.rules,
                )
                self.assertEqual(set(mine), set(serve))
                for label in decision.emitted:
                    self.assertLess(abs(mine[label] - serve[label]), 1e-12)

    def test_harvest_is_deterministic(self) -> None:
        simulator = _v9_harvest()
        again = _play_harvest(seed=8, with_p3=False, hands=10)
        self.assertEqual(
            json.dumps(simulator.phase_b_rows, sort_keys=True),
            json.dumps(again.phase_b_rows, sort_keys=True),
        )

    def test_corpus_write_is_byte_deterministic(self) -> None:
        simulator = _v9_harvest()
        with tempfile.TemporaryDirectory() as raw:
            first, _ = _write_and_load(simulator, Path(raw))
            first_bytes = first.read_bytes()
        with tempfile.TemporaryDirectory() as raw:
            second, _ = _write_and_load(simulator, Path(raw))
            self.assertEqual(first_bytes, second.read_bytes())

    def test_statistics_report_the_operator_view(self) -> None:
        simulator = _v9_harvest()
        with tempfile.TemporaryDirectory() as raw:
            _, corpus = _write_and_load(simulator, Path(raw))
        report = corpus_statistics(corpus)
        self.assertEqual(report["decisions"], len(corpus.decisions))
        self.assertEqual(
            report["branch_rows"],
            sum(len(decision.emitted) for decision in corpus.decisions),
        )
        self.assertEqual(report["equity_trials"], 20)
        self.assertEqual(
            report["instrument"],
            {"starting_stack": 6_000, "big_blind": 100, "seeds": [8]},
        )

    def test_p3_seat_composes_with_the_v9_branch_layer(self) -> None:
        simulator = _p3_v9_harvest()
        self.assertGreater(simulator.decisions_emitted, 0)
        with tempfile.TemporaryDirectory() as raw:
            _, corpus = _write_and_load(simulator, Path(raw))
        self.assertEqual(len(corpus.decisions), simulator.decisions_emitted)
        # The conditional resample ran (P3 seats were swapped per rollout).
        self.assertGreater(simulator.p3_stats.swaps_applied, 0)
        for row in simulator.phase_b_rows:
            self.assertIn("p3", row)


class RowShapeTests(unittest.TestCase):
    def test_rows_carry_the_pinned_context_and_read(self) -> None:
        simulator = _v9_harvest()
        for row in simulator.phase_b_rows:
            context = row["context"]
            for key in (
                "pot",
                "to_call",
                "contribution",
                "effective_stack",
                "purse",
                "read_temperature_x10",
                "street",
                "bankroll",
                "exposure",
                "covered_allin_to_amounts",
                "legal_labels",
                "bet_range",
                "raise_range",
            ):
                self.assertIn(key, context)
            self.assertEqual(
                [entry["branch"] for entry in row["branches"]],
                context["legal_labels"],
            )
            self.assertTrue(0 <= context["read_temperature_x10"] <= 1000)
            for entry in row["branches"]:
                is_wager = entry["branch"] == "aggressive" or (
                    entry["branch"] == "active" and context["to_call"] == 0
                )
                self.assertEqual("sizing_target" in entry, is_wager)
                self.assertEqual("sizing_to_amount" in entry, is_wager)
                self.assertNotIn("e6_target", entry)

if __name__ == "__main__":
    unittest.main()
