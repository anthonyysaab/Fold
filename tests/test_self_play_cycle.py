"""Checks for safe self-play session preparation."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from devfun_poker_playground.offline_trainer import TRAINING_OBJECTIVE, TrainingConfig
from devfun_poker_playground.training_telemetry import (
    save_training_corpus,
    TrainingExample,
)
from tools.self_play_cycle import _LegSpec, _run_leg, main


class SelfPlayCycleTests(unittest.TestCase):
    def test_dry_run_does_not_harvest_or_train(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with (
                patch("tools.self_play_cycle.harvest") as harvest,
                patch("tools.self_play_cycle.train_candidate") as train,
                redirect_stdout(output),
            ):
                status = main(
                    [
                        "--model-version",
                        "candidate-dry-run",
                        "--output-dir",
                        directory,
                        "--dry-run",
                    ]
                )

            plan = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(plan["objective"], TRAINING_OBJECTIVE)
            self.assertEqual(plan["training"]["reinforcement_multiplier"], 1.5)
            self.assertEqual(plan["training"]["gradient_clip"], 5.0)
            self.assertEqual(plan["training"]["return_scale_pct"], 20.0)
            self.assertEqual(plan["training"]["device"], "cpu")
            self.assertEqual(plan["training"]["device_name"], "cpu")
            self.assertEqual(plan["training"]["batch_size"], 1024)
            self.assertEqual(plan["training"]["counterfactual_rollouts"], 1)
            self.assertFalse(plan["training"]["train_risk_head"])
            self.assertEqual(plan["state_policy"], "heuristic-aggressive-v6")
            self.assertEqual(plan["foreign_row_total"], 0)
            self.assertIn("no simulation or training", plan["mode"])
            harvest.assert_not_called()
            train.assert_not_called()
            self.assertFalse(
                (Path(directory) / "candidate-dry-run.manifest.json").exists()
            )
            self.assertFalse(
                (Path(directory) / "candidate-dry-run.weights.json").exists()
            )

    def test_obsolete_baseline_warmup_fails_before_harvest(self) -> None:
        with (
            patch("tools.self_play_cycle.harvest") as harvest,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            main(
                [
                    "--model-version",
                    "candidate-invalid",
                    "--baseline-warmup-epochs",
                    "1",
                ]
            )
        harvest.assert_not_called()


def _corpus_example(index: int) -> TrainingExample:
    return TrainingExample(
        table_id=f"sim-1-{index}",
        policy_version="heuristic-aggressive-v6",
        features=(0.5, -1.0, float(index)),
        action_family_index=index % 3,
        behavior_probabilities=(1.0, 0.0, 0.0),
        submitted_risk_fraction=0.1,
        purse_bb=60.0,
        reward_bb=1.0,
        counterfactual=True,
        opponent_confidence=0.5,
        decision_id=f"sim-1-{index}:hero:0",
    )


def _leg_result(examples: list, hands: int) -> SimpleNamespace:
    return SimpleNamespace(
        examples=examples,
        hands=hands,
        sessions=1,
        busts={},
        bb_per_100=lambda agent_id: 0.0,
    )


class LegTaggingTests(unittest.TestCase):
    def test_run_leg_stamps_the_leg_name_on_every_example(self) -> None:
        spec = _LegSpec(
            name="heads-up vs shover",
            hands=4,
            seed=91,
            equity_trials=4,
            starting_stack=600,
            on_policy=None,
            counterfactual_rollouts=1,
            archetype=("shover", 0.0, 0.0, 1.0, 81),
        )
        result = _leg_result([_corpus_example(0), _corpus_example(1)], hands=4)
        with patch("tools.self_play_cycle.run_sessions", return_value=result):
            summary, examples = _run_leg(spec)
        self.assertTrue(summary.startswith("heads-up vs shover:"))
        self.assertEqual(
            [example.harvest_leg for example in examples],
            ["heads-up vs shover", "heads-up vs shover"],
        )


class SparringSeatTests(unittest.TestCase):
    def _sparring_factory(self, spec: _LegSpec):
        captured = {}

        def fake_run_sessions(agents, **kwargs):
            captured["agents"] = agents
            return _leg_result([], hands=spec.hands)

        with patch("tools.self_play_cycle.run_sessions", side_effect=fake_run_sessions):
            _run_leg(spec)
        return dict(captured["agents"])["sparring"]

    def _spec(self, **overrides) -> _LegSpec:
        values = {
            "name": "sparring",
            "hands": 2,
            "seed": 101,
            "equity_trials": 4,
            "starting_stack": 600,
            "on_policy": None,
            "counterfactual_rollouts": 1,
            "sparring": "candidate.manifest.json",
        }
        values.update(overrides)
        return _LegSpec(**values)

    def test_default_sparring_partner_stays_unrecorded(self) -> None:
        sentinel = object()
        with patch("tools.self_play_cycle.load_policy", return_value=sentinel) as load:
            partner = self._sparring_factory(self._spec())()
        load.assert_called_once_with("candidate.manifest.json", equity_trials=4)
        self.assertIs(partner.policy, sentinel)
        self.assertFalse(partner.record_examples)

    def test_record_both_wraps_the_champion_with_recording_on(self) -> None:
        sentinel = object()
        with (
            patch("tools.self_play_cycle.build_policy", return_value=sentinel) as build,
            patch("tools.self_play_cycle.load_policy") as load,
        ):
            partner = self._sparring_factory(
                self._spec(sparring="champion", sparring_record_both=True)
            )()
        build.assert_called_once_with(aggressive=True, equity_trials=4)
        load.assert_not_called()
        self.assertIs(partner.policy, sentinel)
        self.assertTrue(partner.record_examples)

    def test_record_both_keeps_manifest_partners_recorded(self) -> None:
        sentinel = object()
        with patch("tools.self_play_cycle.load_policy", return_value=sentinel):
            partner = self._sparring_factory(self._spec(sparring_record_both=True))()
        self.assertIs(partner.policy, sentinel)
        self.assertTrue(partner.record_examples)


class CorpusCycleTests(unittest.TestCase):
    def test_examples_in_dry_run_reports_loaded_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.gz"
            save_training_corpus(corpus, [_corpus_example(0), _corpus_example(1)])
            output = io.StringIO()
            with (
                patch("tools.self_play_cycle.harvest") as harvest,
                patch("tools.self_play_cycle.train_candidate") as train,
                redirect_stdout(output),
            ):
                status = main(
                    [
                        "--model-version",
                        "candidate-corpus-dry",
                        "--output-dir",
                        directory,
                        "--dry-run",
                        "--examples-in",
                        str(corpus),
                    ]
                )
            plan = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(plan["examples_in_rows"], {str(corpus): 2})
        self.assertEqual(plan["examples_in_total"], 2)
        self.assertEqual(plan["harvest_workers"], 0)
        self.assertEqual(
            plan["harvest_hands"],
            {
                "five_max": 0,
                "shover": 0,
                "station": 0,
                "nit": 0,
                "champion_only_sparring": 0,
            },
        )
        harvest.assert_not_called()
        train.assert_not_called()

    def test_examples_in_skips_harvest_and_trains_from_the_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.gz"
            second = Path(directory) / "second.gz"
            save_training_corpus(first, [_corpus_example(0)])
            save_training_corpus(second, [_corpus_example(1), _corpus_example(2)])
            summary = SimpleNamespace(
                examples=3,
                train_loss=0.1,
                validation_loss=0.2,
                validation_best_action_accuracy=0.5,
                validation_mean_regret_pct=1.0,
                validation_action_value_mae_pct=2.0,
                manifest_path=Path(directory) / "manifest.json",
            )
            with (
                patch("tools.self_play_cycle.harvest") as harvest,
                patch(
                    "tools.self_play_cycle.train_candidate", return_value=summary
                ) as train,
                redirect_stdout(io.StringIO()),
            ):
                status = main(
                    [
                        "--model-version",
                        "candidate-corpus-train",
                        "--output-dir",
                        directory,
                        "--examples-in",
                        str(first),
                        "--examples-in",
                        str(second),
                        "--init-seed",
                        "23",
                        "--learning-rate",
                        "0.005",
                    ]
                )
        self.assertEqual(status, 0)
        harvest.assert_not_called()
        (trained, _, config), _ = train.call_args
        self.assertEqual(
            [example.table_id for example in trained],
            ["sim-1-0", "sim-1-1", "sim-1-2"],
        )
        self.assertEqual(
            [example.harvest_leg for example in trained], [None, None, None]
        )
        config_fields = {field.name for field in fields(TrainingConfig)}
        if {"split_seed", "init_seed"} <= config_fields:
            self.assertEqual(config.split_seed, 17)
            self.assertEqual(config.init_seed, 23)
        else:
            self.assertEqual(config.seed, 23)
        self.assertEqual(config.learning_rate, 0.005)

    def test_examples_in_rejects_foreign_csv(self) -> None:
        with (
            patch("tools.self_play_cycle.harvest") as harvest,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            main(
                [
                    "--model-version",
                    "candidate-conflicting-inputs",
                    "--examples-in",
                    "corpus.gz",
                    "--foreign-csv",
                    "rows.csv",
                ]
            )
        harvest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
