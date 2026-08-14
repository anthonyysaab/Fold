"""Checks for the evaluation gauntlet's statistics and report structure."""

from __future__ import annotations

import contextlib
import io
import json
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


if __name__ == "__main__":
    unittest.main()
