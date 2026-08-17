"""Checks for the evaluation gauntlet's statistics and report structure."""

from __future__ import annotations

import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devfun_poker_playground.poker_policy import AggressivePokerPolicy, build_policy
from tools import evaluate_policies


class _FakeResult:
    """MatchResult look-alike carrying only the fields ruin_stats reads."""

    def __init__(
        self,
        hands: int,
        sessions: int,
        busts: dict,
        chip_delta: int,
        big_blind: int = 100,
    ) -> None:
        self.hands = hands
        self.sessions = sessions
        self.busts = busts
        self.chip_deltas = {"hero": chip_delta}
        self.big_blind = big_blind


class PairedStatsTest(unittest.TestCase):
    def test_two_seeds(self) -> None:
        stats = evaluate_policies.paired_stats([1.0, 3.0])
        self.assertEqual(stats["diffs"], [1.0, 3.0])
        self.assertEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["sd"], 1.41, places=2)
        self.assertEqual(stats["se"], 1.0)
        self.assertEqual(stats["t"], 2.0)

    def test_four_seeds_match_manual_computation(self) -> None:
        stats = evaluate_policies.paired_stats([10.0, -10.0, 30.0, 10.0])
        self.assertEqual(stats["mean"], 10.0)
        self.assertAlmostEqual(stats["sd"], 16.33, places=2)
        self.assertAlmostEqual(stats["se"], 8.16, places=2)
        self.assertAlmostEqual(stats["t"], 1.22, places=2)

    def test_zero_variance_reports_null_t(self) -> None:
        stats = evaluate_policies.paired_stats([5.0, 5.0])
        self.assertEqual(stats["sd"], 0.0)
        self.assertIsNone(stats["t"])

    def test_single_seed_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_policies.paired_stats([1.0])


class RuinStatsTest(unittest.TestCase):
    def test_aggregates_across_seeds(self) -> None:
        results = [
            _FakeResult(hands=100, sessions=3, busts={"hero": 1}, chip_delta=-3_000),
            _FakeResult(hands=100, sessions=2, busts={"hero": 2}, chip_delta=1_000),
        ]
        stats = evaluate_policies.ruin_stats(results, "hero")
        self.assertEqual(stats["sessions"], 5)
        self.assertEqual(stats["busts"], 3)
        self.assertEqual(stats["busts_per_100_hands"], 1.5)
        self.assertEqual(stats["bust_rate_per_session"], 0.6)
        self.assertEqual(stats["mean_session_hands"], 40.0)
        self.assertEqual(stats["bb_per_100"], -10.0)

    def test_reset_mode_results_without_bust_tracking(self) -> None:
        results = [_FakeResult(hands=50, sessions=1, busts={}, chip_delta=500)]
        stats = evaluate_policies.ruin_stats(results, "hero")
        self.assertEqual(stats["busts"], 0)
        self.assertEqual(stats["bust_rate_per_session"], 0.0)
        self.assertEqual(stats["bb_per_100"], 10.0)


class PolicyLabelTest(unittest.TestCase):
    def test_reads_policy_version(self) -> None:
        class Versioned:
            policy_version = "test-v9"

        self.assertEqual(
            evaluate_policies._policy_label(Versioned(), "fallback"), "test-v9"
        )

    def test_falls_back_when_absent(self) -> None:
        self.assertEqual(
            evaluate_policies._policy_label(object(), "heuristic-v5"), "heuristic-v5"
        )


class DuelSeatSwapTest(unittest.TestCase):
    def test_reports_both_orientations_and_seat_means(self) -> None:
        def factory() -> object:
            return build_policy(aggressive=True, equity_trials=4)

        data = evaluate_policies.duel(
            "alpha",
            factory,
            "beta",
            factory,
            seeds=(0, 1),
            scale=0.01,
            stack=6_000,
            reset_stacks=False,
        )
        self.assertEqual(len(data["orientations"]["a_first"]), 2)
        self.assertEqual(len(data["orientations"]["b_first"]), 2)
        self.assertEqual(len(data["seeds"]), 2)
        for mean, first, second in zip(
            data["seeds"],
            data["orientations"]["a_first"],
            data["orientations"]["b_first"],
        ):
            self.assertAlmostEqual(mean, (first + second) / 2, delta=0.02)
        self.assertIn("alpha_bb_per_100", data)
        self.assertIn("paired", data)
        self.assertEqual(set(data["ruin"]), {"alpha", "beta"})
        for stats in data["ruin"].values():
            self.assertIn("busts_per_100_hands", stats)
            self.assertIn("bust_rate_per_session", stats)
            self.assertIn("sessions", stats)
            self.assertIn("bb_per_100", stats)


class GauntletReportSmokeTest(unittest.TestCase):
    def test_tiny_scale_report_has_new_fields(self) -> None:
        def fake_load(path: str, **kwargs: object) -> object:
            policy = build_policy(aggressive=True, equity_trials=4)
            policy.policy_version = "candidate-test-0001"
            return policy

        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            with mock.patch.object(evaluate_policies, "load_policy", fake_load):
                with contextlib.redirect_stdout(stdout):
                    exit_code = evaluate_policies.main(
                        [
                            "--include-heuristic",
                            "--candidate",
                            "fake-manifest.json",
                            "--scale",
                            "0.01",
                            "--seeds",
                            "2",
                            "--duel-seeds",
                            "3",
                            "--equity-trials",
                            "4",
                            "--json",
                            "--output",
                            str(output),
                            # Policies are rebuilt inside worker processes,
                            # where this mock does not exist; a mocked
                            # loader requires the sequential path.
                            "--workers",
                            "1",
                        ]
                    )
            raw = output.read_bytes()
        self.assertEqual(exit_code, 0)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        report = json.loads(raw.decode("utf-8"))
        self.assertEqual(json.loads(stdout.getvalue()), report)

        heuristic_label = AggressivePokerPolicy.policy_version
        self.assertEqual(report["baseline"], heuristic_label)
        self.assertEqual(report["battery_seed_count"], 2)
        self.assertEqual(report["duel_seed_count"], 3)
        self.assertEqual(
            set(report["batteries"]),
            {heuristic_label, "candidate-test-0001"},
        )
        for name, batteries in report["batteries"].items():
            for data in batteries.values():
                self.assertEqual(len(data["seeds"]), 2)
                self.assertIn("bb_per_100", data)
                ruin = data["ruin"]
                self.assertIn("busts_per_100_hands", ruin)
                self.assertIn("bust_rate_per_session", ruin)
                self.assertIn("sessions", ruin)
                self.assertIn("bb_per_100", ruin)
                if name == heuristic_label:
                    self.assertNotIn("paired", data)
                else:
                    paired = data["paired"]
                    self.assertEqual(paired["baseline"], heuristic_label)
                    self.assertEqual(len(paired["diffs"]), 2)
                    self.assertIn("mean", paired)
                    self.assertIn("sd", paired)
                    self.assertIn("se", paired)
                    self.assertIn("t", paired)

        duel_key = f"{heuristic_label} vs candidate-test-0001"
        self.assertEqual(list(report["duels"]), [duel_key])
        duel_data = report["duels"][duel_key]
        self.assertIn(f"{heuristic_label}_bb_per_100", duel_data)
        self.assertEqual(len(duel_data["seeds"]), 3)
        self.assertEqual(len(duel_data["orientations"]["a_first"]), 3)
        self.assertEqual(len(duel_data["orientations"]["b_first"]), 3)
        self.assertIn("paired", duel_data)
        self.assertEqual(
            set(duel_data["ruin"]),
            {heuristic_label, "candidate-test-0001"},
        )


class GauntletWorkerCountTest(unittest.TestCase):
    def test_one_worker_stays_sequential(self) -> None:
        self.assertEqual(evaluate_policies._gauntlet_workers(1, 128), 1)

    def test_single_task_never_spawns_a_pool(self) -> None:
        self.assertEqual(evaluate_policies._gauntlet_workers(0, 1), 1)

    def test_explicit_count_never_exceeds_the_task_count(self) -> None:
        self.assertEqual(evaluate_policies._gauntlet_workers(32, 6), 6)

    def test_auto_stays_within_the_task_count(self) -> None:
        self.assertLessEqual(evaluate_policies._gauntlet_workers(0, 4), 4)
        self.assertGreaterEqual(evaluate_policies._gauntlet_workers(0, 4), 1)


class TaskSpecTest(unittest.TestCase):
    def test_policy_specs_are_picklable_by_value(self) -> None:
        # Matches cross a process boundary; factories are closures and would
        # not survive, which is why policies are described rather than passed.
        spec = evaluate_policies._PolicySpec(
            label="heuristic", kind="heuristic", equity_trials=10
        )
        self.assertEqual(pickle.loads(pickle.dumps(spec)), spec)

    def test_battery_task_rebuilds_its_opponents(self) -> None:
        spec = evaluate_policies._PolicySpec(
            label="heuristic", kind="heuristic", equity_trials=10
        )
        task = evaluate_policies._BatteryTask(
            policy=spec,
            battery="vs-station",
            opponent_seed=13,
            hands=50,
            seed=100,
            stack=600,
            reset_stacks=False,
        )

        opponents = task.opponents()

        self.assertEqual(len(opponents), 1)
        self.assertEqual(opponents[0][0], "vs-station")
        self.assertEqual(opponents[0][1].seed, 13)
        self.assertEqual(pickle.loads(pickle.dumps(task)), task)

    def test_five_max_task_rebuilds_the_calibrated_lineup(self) -> None:
        spec = evaluate_policies._PolicySpec(
            label="heuristic", kind="heuristic", equity_trials=10
        )
        task = evaluate_policies._BatteryTask(
            policy=spec,
            battery="five-max-lineup",
            opponent_seed=23,
            hands=50,
            seed=200,
            stack=600,
            reset_stacks=False,
        )

        self.assertEqual(len(task.opponents()), 4)

    def test_duel_tasks_cover_both_orientations_per_seed(self) -> None:
        first = evaluate_policies._PolicySpec(
            label="a", kind="heuristic", equity_trials=10
        )
        second = evaluate_policies._PolicySpec(
            label="b", kind="heuristic", equity_trials=10
        )

        tasks = evaluate_policies.duel_tasks(
            first, second, seeds=(0, 1), scale=0.05, stack=600, reset_stacks=False
        )

        self.assertEqual(len(tasks), 4)
        self.assertEqual([task.first.label for task in tasks], ["a", "b", "a", "b"])
        # Both orientations of one seed share the simulator seed, which is
        # what makes the seat-mean a paired comparison.
        self.assertEqual(tasks[0].seed, tasks[1].seed)
        self.assertNotEqual(tasks[0].seed, tasks[2].seed)


class GauntletParallelEquivalenceTest(unittest.TestCase):
    def test_parallel_report_matches_sequential_exactly(self) -> None:
        reports = []
        for workers in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "report.json"
                argv = [
                    "--include-heuristic",
                    "--seeds",
                    "2",
                    "--duel-seeds",
                    "1",
                    "--scale",
                    "0.01",
                    "--equity-trials",
                    "4",
                    "--json",
                    "--output",
                    str(out),
                    "--workers",
                    str(workers),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(evaluate_policies.main(argv), 0)
                reports.append(json.loads(out.read_text(encoding="utf-8")))

        self.assertEqual(reports[0], reports[1])
        self.assertTrue(reports[0]["batteries"])


if __name__ == "__main__":
    unittest.main()
