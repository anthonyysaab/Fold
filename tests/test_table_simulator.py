"""Checks for the Arena-shaped table simulator."""

from __future__ import annotations

import unittest

from devfun_poker_playground.game_state import features_from_table
from devfun_poker_playground.poker_policy import AggressivePokerPolicy
from devfun_poker_playground.policy_features import FEATURE_NAMES
from devfun_poker_playground.table_simulator import (
    RecordingPolicy,
    ScriptedAgent,
    TableSimulator,
)


def _weights_favoring_fold() -> dict:
    return {
        "input_size": len(FEATURE_NAMES),
        "hidden_size": 1,
        "feature_names": list(FEATURE_NAMES),
        "labels": ["fold", "check_call", "aggress"],
        "w1": [[0.0] * len(FEATURE_NAMES)],
        "b1": [0.0],
        "w2": [[0.0], [0.0], [0.0]],
        "b2": [5.0, 0.0, 0.0],
    }


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
                "devfun_poker_playground.table_simulator", fromlist=["SimSeat"]
            ).SimSeat(
                seat_number=index + 1,
                agent_id=agent_id,
                agent=agent,
                stack=stacks[agent_id],
            )
            for index, (agent_id, agent) in enumerate(agents)
        ]
        from devfun_poker_playground.table_simulator import MatchResult

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
            weights=_weights_favoring_fold(), equity_trials=60
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
        from devfun_poker_playground.table_simulator import run_sessions

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
            weights=_weights_favoring_fold(), equity_trials=40
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
            weights=_weights_favoring_fold(), equity_trials=20
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
                            weights=_weights_favoring_fold(), equity_trials=20
                        )
                    ),
                ),
                (
                    "sparring",
                    RecordingPolicy(
                        SparringPolicy(
                            weights=_weights_favoring_fold(), equity_trials=20
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


if __name__ == "__main__":
    unittest.main()
