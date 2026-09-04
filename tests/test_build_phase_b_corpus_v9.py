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
from engine.table_simulator import _CounterfactualPoint
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
    merge_corpora_v9,
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


class PostflopSelectionTests(unittest.TestCase):
    """Street-targeted selection for supplemental postflop harvests."""

    @staticmethod
    def _point(ordinal: int, street: str, agent: str = "hero") -> _CounterfactualPoint:
        return _CounterfactualPoint(
            agent_id=agent,
            decision_ordinal=ordinal,
            example=None,
            legal_families=("fold",),
            proposed_risk_fraction=0.5,
            street=street,
        )

    def test_uniform_mode_is_the_frozen_one_per_agent_rule(self) -> None:
        simulator = _bare_simulator()
        points = [
            self._point(index, street)
            for index, street in enumerate(("preflop", "flop", "turn", "river"))
        ]
        selected = simulator._select_points(0, points)
        expected = random.Random(
            f"{simulator.seed}:0:hero:counterfactual"
        ).choice(points)
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0], expected)

    def test_postflop_mode_picks_one_point_per_reached_street(self) -> None:
        simulator = _bare_simulator(
            postflop_selection=True,
            street_quotas={"flop": 5, "turn": 5, "river": 5},
        )
        points = [
            self._point(0, "preflop"),
            self._point(1, "preflop"),
            self._point(2, "flop"),
            self._point(3, "flop"),
            self._point(4, "turn"),
            self._point(5, "river"),
        ]
        selected = simulator._select_points(0, points)
        self.assertEqual(sorted(point.street for point in selected), ["flop", "river", "turn"])
        flop_choices = [point for point in points if point.street == "flop"]
        expected_flop = random.Random(
            f"{simulator.seed}:0:hero:postflop:flop"
        ).choice(flop_choices)
        self.assertIn(expected_flop, selected)

    def test_postflop_quotas_cap_each_street(self) -> None:
        simulator = _bare_simulator(
            postflop_selection=True,
            street_quotas={"flop": 2, "turn": 1, "river": 3},
        )
        flop_points = [self._point(index, "flop") for index in range(5)]
        selected: list[_CounterfactualPoint] = []
        for hand in range(4):
            selected.extend(simulator._select_points(hand, flop_points))
        self.assertEqual(len(selected), 2)

    def test_streets_without_points_are_skipped(self) -> None:
        simulator = _bare_simulator(
            postflop_selection=True,
            street_quotas={"flop": 5, "turn": 5, "river": 5},
        )
        selected = simulator._select_points(0, [self._point(0, "preflop")])
        self.assertEqual(selected, [])


class MergeCorporaTests(unittest.TestCase):
    """The supplemental-harvest merge: compatible corpora combine, incompatibles refuse."""

    def test_merge_combines_compatible_corpora(self) -> None:
        from test_v9_trainer_phase_b import _header, _priced_row, _write_corpus

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            base = directory / "base.jsonl.gz"
            extra = directory / "extra.jsonl.gz"
            output = directory / "merged.jsonl.gz"
            _write_corpus(
                base, [_priced_row("t1", 0)], _header(seeds=[71])
            )
            _write_corpus(
                extra,
                [_priced_row("t2", 0)],
                {
                    **_header(seeds=[72]),
                    "selection": {
                        "mode": "postflop",
                        "street_targets": {"flop": 15_000, "turn": 10_000, "river": 6_000},
                    },
                },
            )
            report = merge_corpora_v9(base, extra, output)
        self.assertEqual(report["base_rows"], 1)
        self.assertEqual(report["extra_rows"], 1)
        self.assertEqual(report["merged_rows"], 2)
        self.assertEqual(report["decisions"], 2)
        self.assertEqual(report["seeds"], [71, 72])

    def test_merge_records_both_selection_provenances(self) -> None:
        import gzip as _gzip

        from test_v9_trainer_phase_b import _header, _priced_row, _write_corpus

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            base = directory / "base.jsonl.gz"
            extra = directory / "extra.jsonl.gz"
            output = directory / "merged.jsonl.gz"
            _write_corpus(
                base,
                [_priced_row("t1", 0)],
                {**_header(seeds=[71]), "selection": {"mode": "uniform"}},
            )
            _write_corpus(
                extra,
                [_priced_row("t2", 0)],
                {**_header(seeds=[72]), "selection": {"mode": "postflop"}},
            )
            merge_corpora_v9(base, extra, output)
            with _gzip.open(output, "rt", encoding="utf-8") as stream:
                merged_header = json.loads(stream.readline())
        self.assertEqual(
            merged_header["selection"],
            [{"mode": "uniform"}, {"mode": "postflop"}],
        )

    def test_merge_refuses_incompatible_corpora(self) -> None:
        from tools.build_phase_b_corpus import PhaseBError
        from test_v9_trainer_phase_b import _header, _priced_row, _write_corpus

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            base = directory / "base.jsonl.gz"
            extra = directory / "extra.jsonl.gz"
            output = directory / "merged.jsonl.gz"
            _write_corpus(base, [_priced_row("t1", 0)], _header(equity_trials=1000))
            _write_corpus(extra, [_priced_row("t2", 0)], _header(equity_trials=500))
            with self.assertRaises(PhaseBError):
                merge_corpora_v9(base, extra, output)

    def test_merge_refuses_overlapping_leg_seeds(self) -> None:
        # decision ids are sim-<seed>-<hand>:..., so shared leg seeds
        # collide probabilistically — the refusal must be structural.
        from tools.build_phase_b_corpus import PhaseBError
        from test_v9_trainer_phase_b import _header, _priced_row, _write_corpus

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            base = directory / "base.jsonl.gz"
            extra = directory / "extra.jsonl.gz"
            output = directory / "merged.jsonl.gz"
            _write_corpus(base, [_priced_row("t1", 0)], _header(seeds=[71, 112]))
            _write_corpus(extra, [_priced_row("t2", 0)], _header(seeds=[112, 117]))
            with self.assertRaises(PhaseBError):
                merge_corpora_v9(base, extra, output)


class CallEventShapeTests(unittest.TestCase):
    """Pre-harvest decision 3: the replay's event shape follows the flag.

    A mixed flag — main play in one shape, replays in the other — is the
    skew the decision exists to remove, rebuilt on the replay side.
    """

    def test_replay_class_follows_the_flag(self) -> None:
        from tools.build_phase_b_corpus import PhaseBReplaySimulator
        from tools.build_phase_b_corpus_v9 import _ArenaShapedReplaySimulator

        on = _bare_simulator()
        self.assertTrue(on.arena_shaped_call_amounts)
        self.assertIs(on.replay_class, _ArenaShapedReplaySimulator)
        off = _bare_simulator(arena_shaped_call_amounts=False)
        self.assertFalse(off.arena_shaped_call_amounts)
        self.assertIs(off.replay_class, PhaseBReplaySimulator)


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
            simulator._purity_verdict(
                candidates,
                executed,
                to_call_zero=False,
                sizing_fields={"aggressive": (400.0, 400.0)},
            )
        )
        self.assertEqual(simulator.probe_action_mismatches, Counter())
        self.assertEqual(simulator.probe_size_mismatches, Counter())
        self.assertEqual(simulator.probe_collisions, 0)

    def test_a_wager_executed_at_another_size_is_dropped(self) -> None:
        """The sweep's finding: the action NAME can be admissible while
        the amount is not the one the row records. A near-nut escalation
        whose raise the cap refused now shoves — {raise, all-in} admits
        it — while sizing_to_amount still describes the small raise, so
        the value formula would price a 6,000 shove's reward at 400."""

        simulator = _bare_simulator()
        candidates = [
            ("fatal", "fold", None),
            ("active", "check_call", None),
            ("aggressive", "aggress", 0.75),
        ]
        executed = {
            "fatal": ("fold", None),
            "active": ("call", None),
            "aggressive": ("all-in", 6_000),
        }
        verdict = simulator._purity_verdict(
            candidates,
            executed,
            to_call_zero=False,
            sizing_fields={"aggressive": (400.0, 400.0)},
        )
        self.assertIsNotNone(verdict)
        self.assertEqual(simulator.probe_size_mismatches["aggressive->all-in"], 1)
        # The action name alone would have ADMITTED this row.
        self.assertIn("all-in", expected_executions("aggressive", False))

    def test_legalization_sized_differences_are_tolerated(self) -> None:
        """The engine's big-blind floor and integer rounding are the
        approximation the value formula already makes at BOTH train and
        serve time, so a sub-big-blind difference must NOT drop the row.
        Measured: a tighter tolerance threw away 4 of 140 decisions on a
        production-settings leg for agreeing with the design."""

        simulator = _bare_simulator()  # big_blind 100
        candidates = [("passive", "check_call", None), ("active", "aggress", 0.5)]
        executed = {"passive": ("check", None), "active": ("bet", 250)}
        # Composed float 202.5 -> engine floors/rounds to 250: inside one
        # big blind of the recorded amount, so the row survives.
        self.assertIsNone(
            simulator._purity_verdict(
                candidates,
                executed,
                to_call_zero=True,
                sizing_fields={"active": (202.5, 202.5)},
            )
        )
        self.assertEqual(simulator.probe_size_mismatches, Counter())
        # A category error is still caught at the same tolerance.
        self.assertIsNotNone(
            simulator._purity_verdict(
                candidates,
                {"passive": ("check", None), "active": ("bet", 6_000)},
                to_call_zero=True,
                sizing_fields={"active": (202.5, 202.5)},
            )
        )

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
                candidates, executed, to_call_zero=False, sizing_fields={}
            )
            self.assertIsNotNone(verdict)
            self.assertEqual(simulator.probe_action_mismatches[expected_key], 1)

    def test_a_demoted_escalation_is_classified(self) -> None:
        """An escalation the engine could not size executes as a call
        (the risk cap refused the raise and the ladder allowed the
        price); the counter names the class. Renamed after L5: the
        all-in-only state no longer produces this — it now comes from a
        RANGED spot whose raise the cap emptied."""

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
            candidates,
            executed,
            to_call_zero=False,
            sizing_fields={"aggressive": (400.0, 400.0)},
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


def _hero_recorder(provider: P3BeliefProvider):
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
        belief_provider=provider,
        potential_trials=40,
    )
    return ContractForcingRecorder(policy)


def _play_harvest(
    *, seed: int, with_p3: bool, hands: int
) -> PhaseBHarvestSimulatorV9:
    from engine.strength_aware_opponent import StrengthAwareAgent, load_fit
    from engine.table_simulator import ScriptedAgent
    from tools.build_phase_b_corpus import P3SeatWrapper

    # ONE provider for both the hero policy and the harvester's own
    # extraction, exactly as run_leg_v9 wires it. The sweep found the
    # test building two, which made the belief-degrade assertion
    # unfalsifiable: it watched a provider the hero never touched.
    provider = _provider()
    recorder = _hero_recorder(provider)
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
        belief_provider=provider,
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
        # Defect 18h: the emitted-set-size histogram must be counted at
        # the emission — one entry per emitted decision, sizes >= 2, and
        # the histogram must equal the branch rows actually recorded.
        self.assertEqual(
            sum(simulator.emitted_branch_counts.values()),
            simulator.decisions_emitted,
        )
        self.assertTrue(all(size >= 2 for size in simulator.emitted_branch_counts))
        self.assertEqual(
            sum(len(row["branches"]) for row in simulator.phase_b_rows),
            sum(
                size * count
                for size, count in simulator.emitted_branch_counts.items()
            ),
        )
        # Well-formed sim snapshots never degrade the belief provider;
        # the counter exists so a real harvest surfaces any that do.
        self.assertEqual(simulator.belief_degrades, 0)

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


class LegDiagnosticsTests(unittest.TestCase):
    """Defect 18h: the leg result carries the emitted-size histogram and
    every drop counter, so the leg summary prints a complete picture."""

    def test_run_leg_v9_serialises_emitted_branch_counts(self) -> None:
        from test_learned_policy_v9 import _write_artifact
        from tools.build_phase_b_corpus import LegSpec
        from tools.build_phase_b_corpus_v9 import run_leg_v9

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            candidate = str(_write_artifact(directory))
            spec = LegSpec(
                name="mini-leg",
                opponents=("median-bot", "tight-bot"),
                hands=6,
                seed=7,
                session_hands=6,
                candidate=candidate,
                equity_trials=20,
                potential_trials=40,
                feature_seed=7,
                counterfactual_rollouts=1,
                accept_threshold=0.35,
                resample_tries=40,
            )
            result = run_leg_v9(spec)
        self.assertGreater(result["decisions_emitted"], 0)
        counts = result["emitted_branch_counts"]
        self.assertIsInstance(counts, dict)
        self.assertEqual(sum(counts.values()), result["decisions_emitted"])
        self.assertTrue(all(int(size) >= 2 for size in counts))
        for name in (
            "single_branch_groups",
            "purity_dropped_decisions",
            "probe_action_mismatches",
            "probe_size_mismatches",
            "probe_collisions",
            "belief_degrades",
        ):
            self.assertIn(name, result)


class OwnerDecisionPinTests(unittest.TestCase):
    def test_harvest_equity_trials_default_is_the_settled_1000(self) -> None:
        """Owner decision 2026-08-30 (L4 close): harvest == serve at
        1,000 trials. The default IS what ships when nobody remembers
        the flag, so the decision is pinned here — and the v7/v8 serve
        default stays 200, because the frozen instruments bake it."""

        from engine.learned_policy import DEFAULT_SERVE_EQUITY_TRIALS
        from tools.build_phase_b_corpus_v9 import _parser

        self.assertEqual(_parser().get_default("equity_trials"), 1_000)
        self.assertEqual(DEFAULT_SERVE_EQUITY_TRIALS, 200)


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


# NOTE (2026-09-02): LearnedPanelTests lived here and was removed with the
# feature it covered. Phase B's counterfactual replay resamples every
# card-reading opponent's holes conditional on that opponent's own prefix
# decisions, and only P3SeatWrapper records that prefix -- so a learned
# seat cannot be harvested honestly without a conditional hole sampler of
# its own. See the reasoning kept in build_phase_b_corpus._OPPONENT_KINDS.


if __name__ == "__main__":
    unittest.main()
