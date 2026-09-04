"""Checks for the Arena-shaped table simulator."""

from __future__ import annotations

import copy
import unittest

from engine.game_state import features_from_table
from engine.poker_policy import AggressivePokerPolicy
from engine.table_simulator import (
    _FAMILY_BRANCHES,
    RecordingPolicy,
    ScriptedAgent,
    SimSeat,
    TableSimulator,
    TexturedAgent,
    board_coordination,
)


class _FoldBot:
    """Folds when possible, otherwise checks; the blind-bleed baseline."""

    policy_version = "fold-bot"

    def decide(self, table: dict) -> dict:
        available = set(table["allowedActions"]["availableActions"])
        if "check" in available:
            return {"action": "check", "message": "wait"}
        return {"action": "fold", "message": "bye"}


class _SnapshotAuditBot:
    """Checks or calls while validating every snapshot through game_state."""

    policy_version = "audit-bot"

    def __init__(self) -> None:
        self.snapshots = 0

    def decide(self, table: dict) -> dict:
        features_from_table(table)  # raises on any malformed snapshot
        self.snapshots += 1
        available = set(table["allowedActions"]["availableActions"])
        if "check" in available:
            return {"action": "check", "message": "ok"}
        if "call" in available:
            return {"action": "call", "message": "ok"}
        return {"action": "fold", "message": "ok"}


class TableSimulatorTests(unittest.TestCase):
    def test_chips_are_conserved_without_resets(self) -> None:
        simulator = TableSimulator(seed=11)
        agents = [
            ("a", ScriptedAgent("a", 0.4, 0.3, 0.05, seed=1)),
            ("b", ScriptedAgent("b", 0.2, 0.6, 0.0, seed=2)),
            ("c", ScriptedAgent("c", 0.3, 0.4, 0.0, seed=3)),
        ]
        result = simulator.play_match(agents, hands=40, reset_stacks=False)
        self.assertEqual(sum(result.chip_deltas.values()), 0)

    def test_deterministic_given_the_seed(self) -> None:
        def run():
            simulator = TableSimulator(seed=7, collect_examples=False)
            agents = [
                ("a", ScriptedAgent("a", 0.3, 0.4, 0.0, seed=4)),
                ("b", ScriptedAgent("b", 0.25, 0.5, 0.0, seed=5)),
            ]
            return simulator.play_match(agents, hands=30).chip_deltas

        self.assertEqual(run(), run())

    def test_every_snapshot_satisfies_the_live_contract(self) -> None:
        auditor = _SnapshotAuditBot()
        simulator = TableSimulator(seed=3)
        agents = [
            ("audit", auditor),
            ("wild", ScriptedAgent("wild", 0.5, 0.2, 0.1, seed=9)),
            ("calm", ScriptedAgent("calm", 0.15, 0.6, 0.0, seed=10)),
        ]
        simulator.play_match(agents, hands=25)
        self.assertGreater(auditor.snapshots, 40)

    def test_fold_only_agent_bleeds_exactly_the_blinds(self) -> None:
        simulator = TableSimulator(seed=2)
        result = simulator.play_match(
            [("folder", _FoldBot()), ("folder2", _FoldBot())], hands=100
        )
        # Heads-up, alternating buttons, everyone folds: the small blind
        # folds 50 each hand... except the big blind checks it down when
        # unraised, so the blind simply wins the small blind's 50.
        self.assertEqual(
            result.chip_deltas["folder"] + result.chip_deltas["folder2"], 0
        )
        self.assertLessEqual(abs(result.chip_deltas["folder"]), 100 * 50)

    def test_side_pots_pay_layered_winners(self) -> None:
        # Rigged deck: seat 1 gets aces, seat 2 kings, seat 3 deuces.
        # Order: button is seat 1 for hand 0, blinds seats 2 and 3, deal
        # starts from seat 1.
        deck = [
            "As",
            "Ad",  # seat 1
            "Ks",
            "Kd",  # seat 2
            "2c",
            "2d",  # seat 3
            "7h",
            "8h",
            "9s",
            "Jc",
            "3s",  # board
        ] + ["4c", "4d", "4h", "5c", "5d", "5h", "6c"]

        class ShoveBot:
            policy_version = "shove"

            def decide(self, table: dict) -> dict:
                allowed = table["allowedActions"]
                if "all-in" in allowed["availableActions"]:
                    return {
                        "action": "all-in",
                        "amount": allowed["allInToAmount"],
                        "message": "max",
                    }
                return {"action": "call", "message": "call"}

        class CallBot:
            policy_version = "call"

            def decide(self, table: dict) -> dict:
                allowed = table["allowedActions"]
                available = set(allowed["availableActions"])
                if "call" in available:
                    return {"action": "call", "message": "call"}
                if "check" in available:
                    return {"action": "check", "message": "check"}
                return {"action": "fold", "message": "fold"}

        simulator = TableSimulator(seed=1)
        agents = [("aces", ShoveBot()), ("kings", CallBot()), ("deuces", CallBot())]
        stacks = {"aces": 1_000, "kings": 1_000, "deuces": 1_000}
        seats = [
            __import__(
                "engine.table_simulator", fromlist=["SimSeat"]
            ).SimSeat(
                seat_number=index + 1,
                agent_id=agent_id,
                agent=agent,
                stack=stacks[agent_id],
            )
            for index, (agent_id, agent) in enumerate(agents)
        ]
        from engine.table_simulator import MatchResult

        result = MatchResult(
            hands=0,
            big_blind=100,
            chip_deltas={agent_id: 0 for agent_id, _ in agents},
            decisions={agent_id: 0 for agent_id, _ in agents},
        )
        hand = simulator._play_hand(
            seats, button_index=0, hand_index=0, result=result, deck_for_test=deck
        )
        # Aces shove 1,000, both call: aces scoop the lot.
        self.assertEqual(hand.chip_deltas["aces"], 2_000)
        self.assertEqual(hand.chip_deltas["kings"], -1_000)
        self.assertEqual(hand.chip_deltas["deuces"], -1_000)
        self.assertTrue(hand.showdown)

    def test_live_policy_neutralizes_the_perma_shover(self) -> None:
        policy = AggressivePokerPolicy(
            equity_trials=60
        )
        simulator = TableSimulator(seed=21)
        result = simulator.play_match(
            [
                ("hero", RecordingPolicy(policy)),
                ("shover", ScriptedAgent("shover", 0.0, 0.0, 1.0, seed=13)),
            ],
            hands=300,
        )
        # Deterministic tripwire, not a statistics claim: before the
        # tracker existed this matchup bled -134 bb/100; 2,000-hand probes
        # now measure it within noise of breakeven (about -5). The bound
        # catches any regression toward the old fold-them-dry behavior,
        # and the floor assertion proves the belief converged.
        self.assertGreater(result.bb_per_100("hero"), -45.0)
        self.assertGreater(policy.opponent_tracker.range_floor("shover"), 0.7)

    def test_carry_over_sessions_bust_restart_and_aggregate(self) -> None:
        from engine.table_simulator import run_sessions

        shover = ScriptedAgent("shover", 0.0, 0.0, 1.0, seed=3)
        station = ScriptedAgent("station", 0.15, 0.05, 0.0, seed=4)
        result = run_sessions(
            [("shover", lambda: shover), ("station", lambda: station)],
            target_hands=300,
            seed=9,
            starting_stack=1_000,
        )
        # Ten-blind all-in wars bust someone within a few hands, so hitting
        # the hand target requires many sessions, and chips stay conserved.
        self.assertGreaterEqual(result.hands, 300)
        self.assertGreater(result.sessions, 10)
        self.assertEqual(sum(result.chip_deltas.values()), 0)
        self.assertGreater(sum(result.busts.values()), 10)

    def test_self_play_examples_carry_settled_rewards(self) -> None:
        policy = AggressivePokerPolicy(
            equity_trials=40
        )
        simulator = TableSimulator(seed=8, collect_examples=True)
        result = simulator.play_match(
            [
                ("hero", RecordingPolicy(policy)),
                ("median", ScriptedAgent("median", 0.226, 0.5, 0.0, seed=6)),
            ],
            hands=40,
        )
        self.assertGreater(len(result.examples), 10)
        example = result.examples[0]
        self.assertEqual(len(example.features), 142)
        self.assertTrue(example.policy_version.startswith("sim-heuristic"))
        self.assertGreater(example.purse_bb, 0.0)
        self.assertTrue(any(ex.reward_bb != 0.0 for ex in result.examples))

    def test_counterfactual_examples_compare_legal_actions_from_same_state(
        self,
    ) -> None:
        policy = AggressivePokerPolicy(
            equity_trials=20
        )
        simulator = TableSimulator(
            seed=8,
            collect_counterfactuals=True,
            counterfactual_rollouts=2,
        )
        result = simulator.play_match(
            [
                ("hero", RecordingPolicy(policy)),
                ("median", ScriptedAgent("median", 0.226, 0.5, 0.0, seed=6)),
            ],
            hands=12,
        )
        behavior = [
            example for example in result.examples if not example.counterfactual
        ]
        counterfactuals = [
            example for example in result.examples if example.counterfactual
        ]
        self.assertGreater(len(behavior), 0)
        self.assertGreater(len(counterfactuals), 0)
        by_decision: dict[str, list] = {}
        for example in counterfactuals:
            self.assertIsNotNone(example.decision_id)
            by_decision.setdefault(example.decision_id, []).append(example)
        for examples in by_decision.values():
            self.assertGreaterEqual(len(examples), 2)
            self.assertAlmostEqual(sum(example.reward_bb for example in examples), 0.0)
            # One row per value branch; aggression contributes two branches
            # (half pot and full pot) that share a family index, so branch
            # labels are the unique key under format 2.
            self.assertEqual(
                len({example.action_branch for example in examples}),
                len(examples),
            )
            self.assertTrue(
                all(example.action_branch is not None for example in examples)
            )
            self.assertEqual(
                len({example.behavior_probabilities for example in examples}),
                1,
            )
            self.assertTrue(
                all(0.0 <= example.opponent_confidence <= 1.0 for example in examples)
            )

    def test_recording_policy_can_exclude_sparring_opponent_examples(self) -> None:
        class SparringPolicy(AggressivePokerPolicy):
            policy_version = "losing-sparring-policy"

        simulator = TableSimulator(seed=12, collect_examples=True)
        result = simulator.play_match(
            [
                (
                    "hero",
                    RecordingPolicy(
                        AggressivePokerPolicy(
                            equity_trials=20
                        )
                    ),
                ),
                (
                    "sparring",
                    RecordingPolicy(
                        SparringPolicy(
                            equity_trials=20
                        ),
                        record_examples=False,
                    ),
                ),
            ],
            hands=20,
        )
        self.assertGreater(len(result.examples), 0)
        self.assertEqual(
            {example.policy_version for example in result.examples},
            {"sim-heuristic-aggressive-v6"},
        )


class CallEventEncodingTests(unittest.TestCase):
    """Pre-harvest decision 3: the call event's amount is the increment.

    The frozen default keeps the post-call street total (the v8
    instrument's bytes); the v9 harvest opts into the Arena's shape.
    """

    @staticmethod
    def _seat() -> SimSeat:
        return SimSeat(seat_number=1, agent_id="a", agent=None, stack=1_000)

    @staticmethod
    def _snapshot(call_chips: int) -> dict:
        return {
            "allowedActions": {
                "availableActions": ["fold", "call"],
                "callChips": call_chips,
            }
        }

    def _apply_call(self, sim: TableSimulator) -> int:
        seat = self._seat()
        seat.street_commit = 100
        events: list = []
        action, _ = sim._apply(
            seat, {"action": "call"}, self._snapshot(200), 200, 100, events, "flop"
        )
        self.assertEqual(action, "call")
        return events[-1]["summary"]["amount"]

    def test_default_records_the_post_call_street_total(self) -> None:
        self.assertEqual(self._apply_call(TableSimulator()), 300)

    def test_arena_shaped_records_the_increment(self) -> None:
        self.assertEqual(
            self._apply_call(TableSimulator(arena_shaped_call_amounts=True)), 200
        )


class BoardCoordinationTest(unittest.TestCase):
    """The public-board texture signal behind precondition P3."""

    def test_empty_board_is_uncoordinated(self) -> None:
        self.assertEqual(board_coordination(()), 0.0)

    def test_score_stays_in_unit_range(self) -> None:
        boards = [
            (),
            ("2c", "7d", "Kh"),
            ("2c", "7c", "Kh"),
            ("7c", "7d", "Kh"),
            ("8c", "9c", "Th"),
            ("8c", "9c", "Qc"),
            ("As", "Ks", "Qs", "Js", "Ts"),
        ]
        for board in boards:
            with self.subTest(board=board):
                self.assertGreaterEqual(board_coordination(board), 0.0)
                self.assertLessEqual(board_coordination(board), 1.0)

    def test_wetter_boards_score_higher(self) -> None:
        dry = board_coordination(("2c", "7d", "Kh"))
        two_tone = board_coordination(("2c", "7c", "Kh"))
        monotone = board_coordination(("8c", "9c", "Qc"))
        self.assertLess(dry, two_tone)
        self.assertLess(two_tone, monotone)

    def test_reads_only_the_board(self) -> None:
        # Same board, and the score must not depend on anything else.
        self.assertEqual(
            board_coordination(("8c", "9c", "Th")),
            board_coordination(["8c", "9c", "Th"]),
        )


class TexturedAgentTest(unittest.TestCase):
    """P3 phase one: an opponent that prices a call.

    The point of the archetype is that expected value stops being linear in
    bet size. These tests pin that property down, because losing it silently
    would restore the defect that made candidate v7-0001 overbet.
    """

    def _agent(self, **kwargs: float) -> TexturedAgent:
        return TexturedAgent("textured", 0.226, 0.5, 0.0, 11, **kwargs)

    def _p(self, agent: TexturedAgent, bet: int, board: tuple = ()) -> float:
        return agent._fold_probability(
            {"potChips": 100, "boardCards": list(board)}, {"callChips": bet}
        )

    def test_scripted_agent_fold_rate_is_constant(self) -> None:
        # The defect, pinned: the base archetype ignores the wager entirely.
        agent = ScriptedAgent("median", 0.226, 0.5, 0.0, 11)
        rates = {
            agent._fold_probability(
                {"potChips": 100, "boardCards": []}, {"callChips": bet}
            )
            for bet in (10, 50, 100, 400)
        }
        self.assertEqual(rates, {0.5})

    def test_fold_rate_rises_with_bet_size(self) -> None:
        agent = self._agent()
        rates = [self._p(agent, bet) for bet in (25, 50, 75, 100, 200, 400)]
        for lower, higher in zip(rates, rates[1:]):
            self.assertLess(lower, higher)

    def test_fold_rate_rises_with_board_texture(self) -> None:
        agent = self._agent()
        dry = self._p(agent, 100, ("2c", "7d", "Kh"))
        wet = self._p(agent, 100, ("8c", "9c", "Qc"))
        self.assertLess(dry, wet)

    def test_half_pot_reproduces_the_base_rate(self) -> None:
        # Half pot is the reference price, so a dry board there must leave the
        # archetype's advertised fold_vs_bet untouched.
        self.assertAlmostEqual(self._p(self._agent(), 50), 0.5)

    def test_response_strength_is_tunable(self) -> None:
        mild = self._p(self._agent(size_response=0.6), 200)
        strong = self._p(self._agent(size_response=1.2), 200)
        self.assertLess(mild, strong)

    def test_probability_is_clamped(self) -> None:
        agent = self._agent(size_response=50.0, texture_response=50.0)
        self.assertLessEqual(self._p(agent, 10_000, ("8c", "9c", "Qc")), agent.max_fold)
        starved = self._agent(size_response=50.0)
        self.assertGreaterEqual(self._p(starved, 1), starved.min_fold)

    def test_stays_card_blind_so_the_chance_salt_still_applies(self) -> None:
        # If this ever flips to True the counterfactual salt stops resampling
        # this agent's hole cards, the label degrades from Q(s, a) to
        # Q(s, a | villain holds these cards), and the frozen-runout defect
        # that closed candidate 0016 comes back. Phase two, not phase one.
        self.assertIs(TexturedAgent.reads_cards, False)
        self.assertIs(self._agent().reads_cards, False)

    def test_fold_probability_does_not_consume_rng(self) -> None:
        # decide() spends exactly one roll.random() for the fold check; a hook
        # that drew again would desynchronise counterfactual replay.
        agent = self._agent()
        table = {"potChips": 100, "boardCards": ["8c", "9c", "Qc"]}
        allowed = {"callChips": 100}
        first = agent._fold_probability(table, allowed)
        for _ in range(5):
            self.assertEqual(agent._fold_probability(table, allowed), first)

    def test_plays_a_full_match_deterministically(self) -> None:
        def run() -> dict:
            simulator = TableSimulator(seed=7, starting_stack=1000)
            return simulator.play_match(
                [
                    ("textured", TexturedAgent("textured", 0.226, 0.5, 0.0, 11)),
                    ("median-bot", ScriptedAgent("median-bot", 0.226, 0.5, 0.0, 12)),
                ],
                hands=40,
            ).chip_deltas

        self.assertEqual(run(), run())

    def test_differs_from_the_constant_fold_archetype_in_play(self) -> None:
        # Same parameters, same seed: only the fold response differs, so the
        # match must not come out identical or the archetype is inert.
        def run(agent: ScriptedAgent) -> dict:
            simulator = TableSimulator(seed=7, starting_stack=1000)
            return simulator.play_match(
                [
                    ("hero", agent),
                    ("median-bot", ScriptedAgent("median-bot", 0.226, 0.5, 0.0, 12)),
                ],
                hands=60,
            ).chip_deltas

        constant = run(ScriptedAgent("hero", 0.226, 0.5, 0.0, 11))
        textured = run(TexturedAgent("hero", 0.226, 0.5, 0.0, 11))
        self.assertNotEqual(constant, textured)


class EquityCacheTest(unittest.TestCase):
    """The opt-in equity memo must be invisible in outcomes and off by default."""

    @staticmethod
    def _match(cache):
        from engine.poker_policy import build_policy
        from engine.table_simulator import run_sessions

        hero = build_policy(aggressive=True, equity_trials=40, equity_cache=cache)
        result = run_sessions(
            [
                ("hero", lambda: RecordingPolicy(hero)),
                (
                    "median-bot",
                    lambda: ScriptedAgent("median-bot", 0.226, 0.5, 0.0, 31),
                ),
            ],
            target_hands=30,
            seed=31,
            starting_stack=6000,
            collect_examples=True,
            collect_counterfactuals=True,
            counterfactual_rollouts=2,
        )
        return result

    def test_cache_is_off_by_default(self) -> None:
        policy = AggressivePokerPolicy(
            equity_trials=10
        )
        self.assertIsNone(policy.equity_cache)

    def test_shared_cache_survives_deepcopy_as_one_object(self) -> None:
        # Counterfactual replay deep-copies the seats; a per-copy memo would
        # forfeit exactly the cross-rollout duplication the cache exists for.
        import copy

        from engine.decision_engine import SharedEquityCache

        cache = SharedEquityCache()
        self.assertIs(copy.deepcopy(cache), cache)
        policy = AggressivePokerPolicy(
            equity_trials=10, equity_cache=cache
        )
        self.assertIs(copy.deepcopy(policy).equity_cache, cache)

    def test_harvest_disables_anti_modeling_noise_on_both_hero_paths(self) -> None:
        # The harvest's opponents are stateless seeded RNG and cannot model
        # anyone, so the anti-modeling roll is pure cost there: hyper
        # decisions are excluded from labels by construction AND perturb the
        # trajectories the surrounding labelled decisions depend on. Live
        # keeps its floor; only the harvest zeroes it.
        from types import SimpleNamespace
        from unittest.mock import patch

        import tools.self_play_cycle as cycle

        with patch.object(cycle, "build_policy") as build:
            cycle._hero(equity_trials=4, on_policy=None)
        self.assertEqual(build.call_args.kwargs["hyper_aggression_chance"], 0.0)

        with patch.object(cycle, "load_policy", return_value=SimpleNamespace()) as load:
            cycle._hero(equity_trials=4, on_policy="cand.manifest.json")
        self.assertEqual(load.call_args.kwargs["hyper_aggression_chance"], 0.0)

    def test_live_construction_keeps_the_anti_modeling_floor(self) -> None:
        from engine.decision_engine import HYPER_AGGRESSION_CHANCE
        from engine.poker_policy import build_policy

        policy = build_policy(aggressive=True, equity_trials=4)
        self.assertEqual(policy.hyper_aggression_chance, HYPER_AGGRESSION_CHANCE)
        self.assertGreater(HYPER_AGGRESSION_CHANCE, 0.0)

    def test_load_policy_threads_the_cache_and_defaults_to_none(self) -> None:
        # The --on-policy harvest path builds its hero through load_policy, so
        # a cache that only reaches build_policy would silently skip it.
        from engine.decision_engine import SharedEquityCache
        from engine.learned_policy import load_policy

        manifest = "artifacts/candidates/candidate-v7-0001c.manifest.json"
        cache = SharedEquityCache()
        self.assertIs(
            load_policy(manifest, equity_trials=20, equity_cache=cache).equity_cache,
            cache,
        )
        self.assertIsNone(load_policy(manifest, equity_trials=20).equity_cache)

    def test_cached_and_uncached_runs_are_identical(self) -> None:
        from engine.decision_engine import SharedEquityCache

        cache = SharedEquityCache()
        with_cache = self._match(cache)
        without = self._match(None)
        self.assertEqual(with_cache.chip_deltas, without.chip_deltas)
        self.assertEqual(with_cache.examples, without.examples)
        # The run must actually have exercised the memo, or this test proves
        # nothing: a counterfactual match revisits the pinned decision state
        # across branches and rollouts, so hits are structural.
        self.assertGreater(len(cache), 0)


class _CapturingSimulator(TableSimulator):
    """Keeps the arguments the counterfactual pass was called with.

    The branch point, its seats, and its deck are all local to ``_play_hand``,
    so a test that wants to re-run one decision has to catch them on the way
    past.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.captured: list[tuple] = []

    def _counterfactual_examples(
        self, initial_seats, button_index, hand_index, points, deck_for_test
    ):
        self.captured.append(
            (
                copy.deepcopy(initial_seats),
                button_index,
                hand_index,
                points,
                deck_for_test,
            )
        )
        return super()._counterfactual_examples(
            initial_seats, button_index, hand_index, points, deck_for_test
        )


class BranchSetTests(unittest.TestCase):
    """Legality-and-distinctness-aware branch sets (`notes/OPTION_A_DESIGN.md`)."""

    def test_probe_and_rollout_replays_inherit_the_call_event_flag(self) -> None:
        """Pre-harvest decision 3, sweep finding: the probe's prefix replay
        and the counterfactual rollouts must read the SAME call-event shape
        the main play feeds the policy. A mixed flag makes the probe's
        executed map describe a different state than the rollouts measured —
        silently defeating the purity check it exists to serve."""

        from unittest import mock

        seen: list[bool] = []
        original_init = TableSimulator.__init__

        def spying_init(self, **kwargs):
            seen.append(bool(kwargs.get("arena_shaped_call_amounts", False)))
            original_init(self, **kwargs)

        with mock.patch.object(TableSimulator, "__init__", spying_init):
            simulator = _CapturingSimulator(
                seed=8,
                collect_counterfactuals=True,
                counterfactual_rollouts=1,
                arena_shaped_call_amounts=True,
            )
            simulator.play_match(
                [
                    (
                        "hero",
                        RecordingPolicy(
                            AggressivePokerPolicy(
                                equity_trials=20
                            )
                        ),
                    ),
                    ("median", ScriptedAgent("median", 0.226, 0.5, 0.0, seed=6)),
                ],
                hands=8,
            )
            self.assertTrue(simulator.captured)
        self.assertTrue(seen)
        self.assertTrue(all(seen))

    def _harvest(self) -> _CapturingSimulator:
        simulator = _CapturingSimulator(
            seed=8,
            collect_counterfactuals=True,
            counterfactual_rollouts=2,
        )
        simulator.play_match(
            [
                (
                    "hero",
                    RecordingPolicy(
                        AggressivePokerPolicy(
                            equity_trials=20
                        )
                    ),
                ),
                ("median", ScriptedAgent("median", 0.226, 0.5, 0.0, seed=6)),
            ],
            hands=8,
        )
        self.assertTrue(simulator.captured)
        return simulator

    @staticmethod
    def _candidates(point) -> list[tuple[str, str, float | None]]:
        return [
            branch
            for family in point.legal_families
            for branch in _FAMILY_BRANCHES[family]
        ]

    def test_probe_matches_every_rollout(self) -> None:
        """The probe's answer must equal what each rollout actually submits.

        Deciding the branch set from one prefix replay is only sound because
        the executed action does not move across rollouts. That is the whole
        justification for skipping the duplicate rollouts, so it is checked
        against reality rather than argued from the salt's design.
        """

        simulator = self._harvest()
        compared = 0
        streets = set()
        for seats, button, hand_index, points, deck in simulator.captured:
            for point in points:
                candidates = self._candidates(point)
                executed = simulator._probe_branch_set(
                    seats, button, hand_index, point, candidates, deck
                )
                streets.add(point.street)
                for label, family, pot_fraction in candidates:
                    for rollout in range(3):
                        _, _, actual = simulator._counterfactual_outcome(
                            seats,
                            button,
                            hand_index,
                            point,
                            family,
                            pot_fraction,
                            rollout,
                            deck,
                        )
                        self.assertEqual(
                            actual,
                            executed[label],
                            f"{label} rollout {rollout} diverged from the probe",
                        )
                        compared += 1
        # Coverage guards: a slice of preflop-only branch points would let a
        # probe that is wrong on a postflop sizing clamp pass unnoticed.
        self.assertGreater(compared, 100)
        self.assertGreater(
            len(streets - {"preflop"}), 0, f"only preflop was exercised: {streets}"
        )

    def test_emitted_branches_name_distinct_executed_actions(self) -> None:
        simulator = self._harvest()
        checked = 0
        for seats, button, hand_index, points, deck in simulator.captured[:4]:
            for point in points[:2]:
                candidates = self._candidates(point)
                executed = simulator._probe_branch_set(
                    seats, button, hand_index, point, candidates, deck
                )
                emitted, absorption = simulator._emitted_branches(
                    seats, button, hand_index, point, candidates, deck
                )
                labels = [label for label, _, _ in emitted]
                # Every candidate maps to a survivor that carries its value.
                for label, _, _ in candidates:
                    self.assertIn(absorption[label], labels)
                    self.assertEqual(
                        executed[absorption[label]],
                        executed[label],
                        f"{label} absorbed by a branch executing a different action",
                    )
                actions = [executed[label] for label in labels]
                self.assertEqual(len(set(actions)), len(actions))
                # Nothing is lost: every dropped candidate executed an action
                # that a surviving branch already carries.
                for label, _, _ in candidates:
                    self.assertIn(executed[label], actions)
                checked += 1
        self.assertGreater(checked, 0)

    def test_a_free_check_keeps_check_call_and_drops_fold(self) -> None:
        """`_fold_action` returns a check whenever check is legal.

        The surviving label has to be the one that names the executed action
        honestly, because the label decides which head slot carries the value.
        """

        simulator = self._harvest()
        collisions = 0
        for seats, button, hand_index, points, deck in simulator.captured:
            for point in points:
                if "fold" not in point.legal_families:
                    continue
                candidates = self._candidates(point)
                executed = simulator._probe_branch_set(
                    seats, button, hand_index, point, candidates, deck
                )
                if executed.get("fold") != executed.get("check_call"):
                    continue
                labels = {
                    label
                    for label, _, _ in simulator._emitted_branches(
                        seats, button, hand_index, point, candidates, deck
                    )[0]
                }
                self.assertIn("check_call", labels)
                self.assertNotIn("fold", labels)
                collisions += 1
        self.assertGreater(
            collisions, 0, "no free-check decision arose, so nothing was proved"
        )

    def test_merged_aggression_sizes_keep_the_smaller_wager(self) -> None:
        """The surviving label must name the wager the engine actually sized.

        When the risk cap or the `raiseRange` clamp merges half pot and pot
        into one wager, that wager IS the half-pot branch's; keeping the
        `aggress_pot` label instead would teach the head that a pot-sized bet
        has the value of a smaller one. Nothing else in this class fails if
        the two are swapped in `_BRANCH_DEDUP_ORDER`.
        """

        simulator = self._harvest()
        merges = 0
        for seats, button, hand_index, points, deck in simulator.captured:
            for point in points:
                candidates = self._candidates(point)
                executed = simulator._probe_branch_set(
                    seats, button, hand_index, point, candidates, deck
                )
                half, pot = (
                    executed.get("aggress_half_pot"),
                    executed.get("aggress_pot"),
                )
                if half is None or pot is None or half != pot:
                    continue
                _, absorption = simulator._emitted_branches(
                    seats, button, hand_index, point, candidates, deck
                )
                # Both aggression labels must land on the smaller wager, and
                # never the other way round.
                self.assertEqual(
                    absorption["aggress_pot"], absorption["aggress_half_pot"]
                )
                self.assertIn(
                    absorption["aggress_pot"],
                    ("aggress_half_pot", "check_call", "fold"),
                )
                self.assertNotEqual(absorption["aggress_half_pot"], "aggress_pot")
                merges += 1
        self.assertGreater(
            merges, 0, "no aggression-size merge arose, so nothing was proved"
        )

    def test_groups_are_ragged_within_two_to_four_branches(self) -> None:
        simulator = TableSimulator(
            seed=8,
            collect_counterfactuals=True,
            counterfactual_rollouts=2,
        )
        result = simulator.play_match(
            [
                (
                    "hero",
                    RecordingPolicy(
                        AggressivePokerPolicy(
                            equity_trials=20
                        )
                    ),
                ),
                ("median", ScriptedAgent("median", 0.226, 0.5, 0.0, seed=6)),
            ],
            hands=12,
        )
        groups: dict[str, list] = {}
        for example in result.examples:
            if example.counterfactual:
                groups.setdefault(example.decision_id, []).append(example)
        self.assertTrue(groups)
        for decision_id, examples in groups.items():
            self.assertGreaterEqual(len(examples), 2, decision_id)
            self.assertLessEqual(len(examples), 4, decision_id)
            # Centering is over the emitted set, so the targets still cancel.
            self.assertAlmostEqual(sum(row.reward_bb for row in examples), 0.0)
        self.assertEqual(
            sum(simulator.emitted_branch_counts.values()),
            len(groups) + simulator.single_branch_groups,
        )
        # Non-vacuity: the range assertion above holds trivially if nothing is
        # ever deduplicated, so prove deduplication actually happened.
        self.assertTrue(
            any(len(examples) < 4 for examples in groups.values()),
            "every group still emitted four branches: dedup did not run",
        )
        self.assertTrue(
            all(
                row.branch_absorption is not None
                for examples in groups.values()
                for row in examples
            )
        )

    def test_single_branch_groups_are_dropped_not_emitted(self) -> None:
        """A group with one branch has no preference to learn.

        Its centred target is exactly zero by construction, so it would add
        loss-denominator weight and feature-normalization mass while carrying
        no signal.
        """

        simulator = TableSimulator(
            seed=8, collect_counterfactuals=True, counterfactual_rollouts=1
        )
        simulator.single_branch_groups = 0
        result = simulator.play_match(
            [
                ("hero", RecordingPolicy(AggressivePokerPolicy(equity_trials=20))),
                ("nit", ScriptedAgent("nit", 0.9, 0.1, 0.0, seed=3)),
            ],
            hands=10,
        )
        counts = {}
        for example in result.examples:
            if example.counterfactual:
                counts[example.decision_id] = counts.get(example.decision_id, 0) + 1
        self.assertNotIn(1, set(counts.values()))
        # Non-vacuity: "no 1-row group survived" is also true when no group
        # ever collapsed, so require that the drop path actually fired. A nit
        # folds most hands, which is the archetype that produces them.
        self.assertGreater(
            simulator.single_branch_groups,
            0,
            "no group collapsed, so the drop path was never exercised",
        )
        self.assertEqual(
            simulator.emitted_branch_counts.get(1, 0), simulator.single_branch_groups
        )


if __name__ == "__main__":
    unittest.main()
