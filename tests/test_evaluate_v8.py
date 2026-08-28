"""Checks for the v8 gauntlet wrapper: trivial baselines, plan structure,
strength separation math, and the format-3 policy spec.

Simulation checks run tiny seeded matches (tens of hands against scripted
archetypes) so the suite stays fast; nothing here loads torch or touches
the network for play except one loader round-trip on the real candidate
artifact, which is exactly what the wrapper's workers do.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from engine.learned_policy_v8 import LearnedPokerPolicyV8
from engine.poker_policy import build_policy
from engine.table_simulator import ScriptedAgent, TableSimulator
from tools import evaluate_v8

_REPO = Path(__file__).resolve().parent.parent
_CANDIDATE = _REPO / "artifacts" / "candidates" / "candidate-v8-0001.manifest.json"


def _tiny_match(agent, *, hands: int = 40, seed: int = 11, opponents=None):
    """Play a small seeded match with the agent in seat 0."""

    simulator = TableSimulator(seed=seed, starting_stack=2_000)
    lineup = [("hero", agent)]
    if opponents is None:
        opponents = [
            ("median", ScriptedAgent("median", 0.226, 0.5, 0.0, 13)),
            ("shover", ScriptedAgent("shover", 0.0, 0.0, 1.0, 14)),
        ]
    lineup.extend(opponents)
    return simulator.play_match(lineup, hands=hands)


class _ActionSpy:
    """Wrap a trivial agent and record every submitted action name."""

    reads_cards = False

    def __init__(self, agent) -> None:
        self.agent = agent
        self.actions: list[str] = []

    def decide(self, table):
        payload = self.agent.decide(table)
        self.actions.append(str(payload.get("action")))
        return payload


class TrivialAgentTests(unittest.TestCase):
    def test_every_mode_plays_legally_and_deterministically(self) -> None:
        for mode in evaluate_v8.TRIVIAL_MODES:
            with self.subTest(mode=mode):
                first = _tiny_match(evaluate_v8.TrivialAgent(mode))
                second = _tiny_match(evaluate_v8.TrivialAgent(mode))
                # No SimulationError means every action was legal; identical
                # seeded reruns must settle identical chips.
                self.assertEqual(first.chip_deltas, second.chip_deltas)
                self.assertEqual(first.hands, second.hands)

    def test_always_fold_never_volunteers_chips(self) -> None:
        spy = _ActionSpy(evaluate_v8.TrivialAgent("always_fold"))
        _tiny_match(spy)
        self.assertTrue(spy.actions)
        self.assertLessEqual(set(spy.actions), {"check", "fold"})

    def test_always_check_call_never_folds_or_bets(self) -> None:
        spy = _ActionSpy(evaluate_v8.TrivialAgent("always_check_call"))
        _tiny_match(spy)
        self.assertTrue(spy.actions)
        self.assertLessEqual(set(spy.actions), {"check", "call"})

    def test_aggress_modes_size_at_the_e6_branch_targets(self) -> None:
        # Hand-built snapshot: pot 600, 200 to call, hero already has 100
        # in, effective stack 4000 (hero 4000 vs opponent 4200+200).
        table = {
            "tableId": "t1",
            "street": "flop",
            "potChips": 600,
            "selfSeatNumber": 1,
            "boardCards": ["2c", "7d", "Js"],
            "seats": [
                {
                    "seatNumber": 1,
                    "status": "Active",
                    "stackChips": 3_900,
                    "currentBetChips": 100,
                    "totalCommittedChips": 100,
                    "holeCards": ["Ah", "Kh"],
                },
                {
                    "seatNumber": 2,
                    "status": "Active",
                    "stackChips": 3_900,
                    "currentBetChips": 300,
                    "totalCommittedChips": 300,
                    "holeCards": None,
                },
            ],
            "allowedActions": {
                "availableActions": ["fold", "call", "raise", "all-in"],
                "callChips": 200,
                "raiseRange": {"min": 500, "max": 4_000},
                "allInToAmount": 4_000,
            },
        }
        # E6 arithmetic (effective stack = 4000 for both seats):
        #   large: min(200 + 1.0*(600+200), 0.45*4000) = min(1000, 1800) = 1000
        #          to-amount = 100 + 1000 = 1100
        #   small: min(200 + 0.5*(600+200), 0.20*4000) = min(600, 800) = 600
        #          to-amount = 100 + 600 = 700
        large = evaluate_v8.TrivialAgent("always_aggress_large").decide(table)
        self.assertEqual(large, {"action": "raise", "amount": 1_100, "message": "trivial"})
        small = evaluate_v8.TrivialAgent("always_aggress_small").decide(table)
        self.assertEqual(small, {"action": "raise", "amount": 700, "message": "trivial"})

    def test_always_fold_takes_the_free_check(self) -> None:
        table = {
            "tableId": "t2",
            "street": "flop",
            "potChips": 200,
            "selfSeatNumber": 1,
            "boardCards": ["2c", "7d", "Js"],
            "seats": [
                {
                    "seatNumber": 1,
                    "status": "Active",
                    "stackChips": 1_000,
                    "currentBetChips": 0,
                    "totalCommittedChips": 100,
                    "holeCards": ["Ah", "Kh"],
                }
            ],
            "allowedActions": {
                "availableActions": ["fold", "check", "bet", "all-in"],
                "callChips": 0,
                "betRange": {"min": 100, "max": 1_000},
                "allInToAmount": 1_000,
            },
        }
        payload = evaluate_v8.TrivialAgent("always_fold").decide(table)
        self.assertEqual(payload["action"], "check")

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_v8.TrivialAgent("always_win")


class TrivialPlanTests(unittest.TestCase):
    def test_plan_mirrors_the_instrument_seed_construction(self) -> None:
        plan = evaluate_v8.trivial_battery_plan(
            "always_fold", seeds=(0, 1), scale=1.0, stack=6_000
        )
        # 5 archetype channels + five-max, two seeds each.
        self.assertEqual(len(plan), 12)
        by_channel: dict[str, list] = {}
        for battery, hands, task in plan:
            by_channel.setdefault(battery, []).append((hands, task))
        self.assertEqual(len(by_channel), 6)
        for battery, entries in by_channel.items():
            for index, (hands, task) in enumerate(entries):
                if battery == "five-max-lineup":
                    self.assertEqual(task.seed, 200 + index)
                    self.assertEqual(task.opponent_seed, 23 + index)
                else:
                    self.assertEqual(task.seed, 100 + index)
                    self.assertEqual(task.opponent_seed, 13 + index)
                self.assertEqual(task.hands, hands)
        shover = [hands for hands, _ in by_channel["vs-shover"]]
        self.assertEqual(shover, [1_500, 1_500])

    def test_scale_floor_is_fifty_hands(self) -> None:
        plan = evaluate_v8.trivial_battery_plan(
            "always_fold", seeds=(0,), scale=0.001, stack=6_000
        )
        self.assertTrue(all(hands == 50 for _, hands, _ in plan))


class SeparationReportTests(unittest.TestCase):
    def test_hand_computed_separation(self) -> None:
        records = [
            ("flop", "aggress", 0.8),
            ("flop", "aggress", 0.6),
            ("flop", "fold", 0.2),
            ("flop", "fold", 0.4),
            ("turn", "check_call", 0.5),
        ]
        report = evaluate_v8.separation_report(records)
        flop = report["flop"]
        # mean(aggress)=0.7, mean(fold)=0.3 -> separation 0.4;
        # var_a = var_f = 0.02 -> se = sqrt(0.01 + 0.01) = 0.1414
        self.assertEqual(flop["separation_aggress_minus_fold"], 0.4)
        self.assertEqual(flop["se"], 0.1414)
        self.assertEqual(flop["ci95"], [0.1228, 0.6772])
        self.assertEqual(flop["families"]["aggress"]["count"], 2)
        self.assertEqual(flop["families"]["fold"]["mean_strength"], 0.3)
        # Turn has no aggress/fold pair, so no separation is claimed.
        self.assertIsNone(report["turn"]["separation_aggress_minus_fold"])
        self.assertEqual(report["all_streets"]["decisions"], 5)
        self.assertEqual(report["river"]["decisions"], 0)

    def test_single_sample_families_report_no_separation(self) -> None:
        records = [("river", "aggress", 0.9), ("river", "fold", 0.1)]
        report = evaluate_v8.separation_report(records)
        self.assertIsNone(report["river"]["separation_aggress_minus_fold"])


class StrengthRecorderTests(unittest.TestCase):
    def test_records_streets_families_and_bounded_strengths(self) -> None:
        records: list[tuple[str, str, float]] = []
        recorder = evaluate_v8.StrengthRecorder(
            build_policy(aggressive=True, equity_trials=10), records
        )
        _tiny_match(
            recorder,
            hands=12,
            opponents=[("median", ScriptedAgent("median", 0.226, 0.5, 0.0, 13))],
        )
        self.assertTrue(records)
        streets = {"preflop", "flop", "turn", "river"}
        families = {"fold", "check_call", "aggress"}
        for street, family, strength in records:
            self.assertIn(street, streets)
            self.assertIn(family, families)
            self.assertGreaterEqual(strength, 0.0)
            self.assertLessEqual(strength, 1.0)


class BatteryComparisonTests(unittest.TestCase):
    def test_pairs_seeds_against_champion_and_floor(self) -> None:
        battery = {
            "seed_count": 2,
            "channels": {
                "vs-nit": {"bb_per_100": 50.0, "seeds": [40.0, 60.0]}
            },
        }
        trivial = {
            "floors": {
                "always_fold": {
                    "vs-nit": {"bb_per_100": -70.0, "seeds": [-72.0, -68.0]}
                }
            }
        }
        noise = {
            "battery_channels": {
                "vs-nit": {
                    "mean_bb_per_100": 79.72,
                    "mde_bb_per_100": {"2_seeds": 6.44},
                    "per_seed_bb_per_100": [76.14, 75.84, 85.39],
                }
            }
        }
        result = evaluate_v8.battery_comparisons(battery, trivial, noise)
        channel = result["channels"]["vs-nit"]
        self.assertEqual(channel["published_mde_bb_per_100"], 6.44)
        self.assertEqual(channel["champion_mean_bb_per_100"], 79.72)
        # Diffs pair seed-for-seed: 40-76.14, 60-75.84.
        self.assertEqual(
            channel["paired_vs_champion"]["diffs"], [-36.14, -15.84]
        )
        floor = channel["trivial_floors"]["always_fold"]
        self.assertTrue(floor["beats_floor_mean"])
        self.assertEqual(floor["paired_vs_floor"]["diffs"], [112.0, 128.0])

    def test_missing_references_degrade_gracefully(self) -> None:
        battery = {
            "seed_count": 2,
            "channels": {"vs-nit": {"bb_per_100": 50.0, "seeds": [40.0, 60.0]}},
        }
        result = evaluate_v8.battery_comparisons(battery, None, None)
        channel = result["channels"]["vs-nit"]
        self.assertNotIn("paired_vs_champion", channel)
        self.assertNotIn("trivial_floors", channel)
        self.assertNotIn("paired_vs_fresh_champion", channel)


class FreshChampionArmTests(unittest.TestCase):
    """The freshly rebuilt champion arm is additive, not a replacement."""

    battery = {
        "seed_count": 2,
        "channels": {"vs-nit": {"bb_per_100": 50.0, "seeds": [40.0, 60.0]}},
    }
    noise = {
        "battery_channels": {
            "vs-nit": {
                "mean_bb_per_100": 79.72,
                "mde_bb_per_100": {"2_seeds": 6.44},
                "per_seed_bb_per_100": [76.14, 75.84, 85.39],
            }
        }
    }
    champion = {
        "noise_floor_champion_label": "heuristic-aggressive-v6",
        "incumbent_label": "candidate-v7-0001c",
        "seed_count": 2,
        "arms": {
            "heuristic-aggressive-v6": {
                "vs-nit": {"bb_per_100": 70.0, "seeds": [66.0, 74.0]}
            },
            "candidate-v7-0001c": {
                "vs-nit": {"bb_per_100": 30.0, "seeds": [28.0, 32.0]}
            },
        },
    }

    def test_fresh_arm_pairs_seedwise_and_keeps_published_columns(self) -> None:
        result = evaluate_v8.battery_comparisons(
            self.battery, None, self.noise, self.champion
        )
        channel = result["channels"]["vs-nit"]
        # Fresh champion arm: seed-for-seed, 40-66 and 60-74.
        self.assertEqual(channel["fresh_champion_label"], "heuristic-aggressive-v6")
        self.assertEqual(channel["fresh_champion_bb_per_100"], 70.0)
        self.assertEqual(
            channel["paired_vs_fresh_champion"]["diffs"], [-26.0, -14.0]
        )
        # The duel incumbent is a SEPARATE arm, not the same thing.
        self.assertEqual(channel["fresh_incumbent_label"], "candidate-v7-0001c")
        self.assertEqual(channel["fresh_incumbent_bb_per_100"], 30.0)
        self.assertEqual(
            channel["paired_vs_fresh_incumbent"]["diffs"], [12.0, 28.0]
        )
        # The stale published columns survive alongside them, unmodified.
        self.assertEqual(channel["champion_mean_bb_per_100"], 79.72)
        self.assertEqual(channel["published_mde_bb_per_100"], 6.44)
        self.assertEqual(
            channel["paired_vs_champion"]["diffs"], [-36.14, -15.84]
        )

    def test_empirical_mde_is_two_standard_errors_of_the_paired_diffs(self) -> None:
        result = evaluate_v8.battery_comparisons(
            self.battery, None, self.noise, self.champion
        )
        channel = result["channels"]["vs-nit"]
        paired = channel["paired_vs_fresh_champion"]
        expected = round(2.0 * paired["sd"] / math.sqrt(2), 2)
        self.assertEqual(
            channel["fresh_champion_empirical_mde_bb_per_100"], expected
        )

    def test_identical_arms_give_exactly_zero_paired_difference(self) -> None:
        """Impossible-by-construction: an arm paired against itself is 0.0."""

        same = {
            "noise_floor_champion_label": "self",
            "incumbent_label": None,
            "seed_count": 2,
            "arms": {"self": {"vs-nit": dict(self.battery["channels"]["vs-nit"])}},
        }
        result = evaluate_v8.battery_comparisons(self.battery, None, None, same)
        paired = result["channels"]["vs-nit"]["paired_vs_fresh_champion"]
        self.assertEqual(paired["diffs"], [0.0, 0.0])
        self.assertEqual(paired["mean"], 0.0)
        self.assertEqual(
            result["channels"]["vs-nit"][
                "fresh_champion_empirical_mde_bb_per_100"
            ],
            0.0,
        )

    def test_champion_stage_is_registered(self) -> None:
        self.assertIn("champion", evaluate_v8._STAGES)
        self.assertTrue(hasattr(evaluate_v8, "stage_champion"))

    def test_noise_floor_champion_is_not_the_duel_incumbent(self) -> None:
        """The published means belong to heuristic-aggressive-v6.

        Guards the exact confusion that produced a fake 100 BB/100
        "staleness" number: the noise floor's own ``champion`` field names
        the heuristic policy, not the candidate the duel is fought against.
        """

        self.assertEqual(
            evaluate_v8.NOISE_FLOOR_CHAMPION_LABEL, "heuristic-aggressive-v6"
        )
        self.assertNotEqual(
            evaluate_v8.NOISE_FLOOR_CHAMPION_LABEL, evaluate_v8.DEFAULT_INCUMBENT
        )
        noise_path = Path(evaluate_v8.DEFAULT_NOISE_FLOOR)
        if noise_path.exists():
            published = json.loads(noise_path.read_text(encoding="utf-8"))
            self.assertEqual(
                published.get("champion"), evaluate_v8.NOISE_FLOOR_CHAMPION_LABEL
            )


class PublishedReproductionTests(unittest.TestCase):
    """The like-for-like instrument gate: same policy, fresh vs published."""

    noise = {
        "champion": "heuristic-aggressive-v6",
        "battery_channels": {
            "vs-nit": {
                "mean_bb_per_100": 79.72,
                "mde_bb_per_100": {"2_seeds": 6.44},
                "per_seed_bb_per_100": [76.14, 75.84, 85.39],
            }
        },
    }

    def _champion(self, seeds):
        return {
            "noise_floor_champion_label": "heuristic-aggressive-v6",
            "incumbent_label": "candidate-v7-0001c",
            "arms": {
                "heuristic-aggressive-v6": {
                    "vs-nit": {
                        "bb_per_100": sum(seeds) / len(seeds),
                        "seeds": seeds,
                    }
                }
            },
        }

    def test_exact_reproduction_is_flagged(self) -> None:
        result = evaluate_v8.published_reproduction(
            self._champion([76.14, 75.84]), self.noise
        )
        channel = result["channels"]["vs-nit"]
        self.assertEqual(channel["per_seed_diffs"], [0.0, 0.0])
        self.assertEqual(channel["max_abs_per_seed_diff"], 0.0)
        self.assertTrue(channel["reproduces_published"])
        self.assertTrue(result["all_channels_reproduce"])

    def test_drift_is_reported_as_a_number_not_a_claim(self) -> None:
        result = evaluate_v8.published_reproduction(
            self._champion([70.14, 71.84]), self.noise
        )
        channel = result["channels"]["vs-nit"]
        self.assertEqual(channel["per_seed_diffs"], [-6.0, -4.0])
        self.assertEqual(channel["max_abs_per_seed_diff"], 6.0)
        self.assertFalse(channel["reproduces_published"])
        self.assertFalse(result["all_channels_reproduce"])

    def test_missing_inputs_degrade_to_empty(self) -> None:
        self.assertEqual(evaluate_v8.published_reproduction(None, self.noise), {})
        self.assertEqual(
            evaluate_v8.published_reproduction(self._champion([1.0]), None), {}
        )


@unittest.skipUnless(_CANDIDATE.exists(), "candidate-v8-0001 artifact not present")
class V8PolicySpecTests(unittest.TestCase):
    def test_build_loads_the_format3_policy(self) -> None:
        spec = evaluate_v8.V8PolicySpec(
            label="candidate-v8-0001", manifest=str(_CANDIDATE), equity_trials=80
        )
        policy = spec.build()
        self.assertIsInstance(policy, LearnedPokerPolicyV8)
        self.assertEqual(policy.policy_version, "candidate-v8-0001")


if __name__ == "__main__":
    unittest.main()
