"""Checks for the offline candidate trainer."""

from __future__ import annotations

import copy
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from engine.learning_contract import (
    LEARNING_INPUT_SIZE,
    validate_artifact_manifest,
)
from training.offline_trainer import (
    TRAINING_OBJECTIVE,
    TrainingConfig,
    _action_value_target,
    _action_value_weight,
    _hybrid_calibration,
    _init_behavior_head,
    _init_weights,
    _log_purse_return,
    _normalized,
    _policy_signal,
    _split,
    _step,
    train_candidate,
)
from engine.forward_kernel import _forward
from engine.training_telemetry import TrainingExample


def _example(
    table_id: str,
    action_family_index: int,
    risk_fraction: float,
    reward_bb: float,
    features: tuple[float, ...] | None = None,
    purse_bb: float = 100.0,
    counterfactual: bool = True,
    opponent_confidence: float = 1.0,
    decision_id: str | None = None,
    behavior_probabilities: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> TrainingExample:
    return TrainingExample(
        table_id=table_id,
        policy_version="heuristic-test-v1",
        features=features or (0.1,) * LEARNING_INPUT_SIZE,
        action_family_index=action_family_index,
        behavior_probabilities=behavior_probabilities,
        submitted_risk_fraction=risk_fraction,
        purse_bb=purse_bb,
        reward_bb=reward_bb,
        counterfactual=counterfactual,
        opponent_confidence=opponent_confidence,
        decision_id=decision_id,
    )


class OfflineTrainerTests(unittest.TestCase):
    def test_trainer_writes_contract_valid_candidate_artifact(self) -> None:
        examples = (
            _example("table-1", 0, 0.0, -1.0),
            _example("table-2", 1, 0.0, 0.5),
            _example("table-3", 2, 0.2, 2.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = train_candidate(
                examples,
                directory,
                TrainingConfig(
                    epochs=1,
                    learning_rate=0.001,
                    validation_fraction=1 / 3,
                    seed=1,
                    model_version="candidate-test",
                ),
            )
            manifest = json.loads(summary.manifest_path.read_text())
            weights = json.loads(summary.weights_path.read_text())

        validate_artifact_manifest(manifest)
        self.assertEqual(summary.examples, 3)
        self.assertEqual(summary.validation_examples, 1)
        self.assertEqual(manifest["state"], "candidate")
        self.assertEqual(manifest["training_window"]["hand_count"], 3)
        self.assertEqual(manifest["weights_sha256"], summary.weights_sha256)
        self.assertEqual(manifest["training"]["objective"], TRAINING_OBJECTIVE)
        self.assertEqual(
            manifest["training"]["validation_split"], "whole hands by table_id"
        )
        self.assertEqual(
            manifest["training"]["targets"]["reward"],
            "sign((action_chip_delta - mean_legal_action_chip_delta) / "
            "starting_purse) * log1p(abs((action_chip_delta - "
            "mean_legal_action_chip_delta) / starting_purse))",
        )
        self.assertEqual(manifest["training"]["baseline_warmup_epochs"], 0)
        self.assertEqual(manifest["training"]["behavior_warmup_epochs"], 1)
        self.assertEqual(manifest["training"]["reinforcement_multiplier"], 1.5)
        self.assertEqual(manifest["training"]["gradient_clip"], 5.0)
        self.assertEqual(
            manifest["training"]["behavior_warmup_target"],
            "behavior-only examples train a transient classifier and the shared "
            "trunk; the persisted action-value and sizing heads are untouched",
        )
        self.assertEqual(manifest["training"]["counterfactual_rollouts_per_family"], 1)
        self.assertEqual(manifest["training_window"]["counterfactual_example_count"], 3)
        self.assertEqual(manifest["training_window"]["behavior_example_count"], 0)
        self.assertEqual(
            manifest["training"]["reward_examples"],
            "counterfactual simulator rollouts only",
        )
        self.assertEqual(
            manifest["training"]["action_head_semantics"],
            "counterfactual_value",
        )
        self.assertFalse(manifest["training"]["train_risk_head"])
        self.assertIn("validation_best_action_accuracy", manifest["evaluation"])
        self.assertIn("validation_mean_regret_pct", manifest["evaluation"])
        self.assertIn("validation_action_value_mae_pct", manifest["evaluation"])
        self.assertIn("hybrid_confidence_calibration", manifest["evaluation"])
        self.assertEqual(manifest["training"]["return_scale_fraction"], 0.2)
        self.assertEqual(weights["model_version"], "candidate-test")
        self.assertIn("risk_fraction_w", weights["weights"])
        self.assertNotIn("behavior_w", weights["weights"])
        self.assertEqual(
            sorted(manifest["engine_parameters"]),
            [
                "bluff_settings",
                "safety_gates",
                "temperature_shaping",
                "tracker_settings",
            ],
        )
        self.assertEqual(
            manifest["engine_parameters"]["safety_gates"]["near_nut_floor"], 0.654
        )

    def test_empty_training_set_fails_before_writing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                train_candidate((), Path(directory))

    def test_existing_candidate_artifact_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "candidate-test.weights.json"
            existing.write_text("keep me", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                train_candidate(
                    (_example("hand-1", 0, 0.0, 0.0),),
                    directory,
                    TrainingConfig(model_version="candidate-test"),
                )
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep me")

    def test_validation_split_keeps_each_hand_together(self) -> None:
        examples = tuple(
            _example(hand_id, action, 0.0, 0.0)
            for hand_id in ("hand-1", "hand-2", "hand-3", "hand-4")
            for action in (0, 1, 2)
        )
        train, validation = _split(
            examples, TrainingConfig(validation_fraction=0.5, seed=3)
        )
        train_hands = {example.table_id for example in train}
        validation_hands = {example.table_id for example in validation}

        self.assertFalse(train_hands & validation_hands)
        self.assertEqual(
            train_hands | validation_hands, {f"hand-{n}" for n in range(1, 5)}
        )
        self.assertTrue(
            all(sum(ex.table_id == hand for ex in train) == 3 for hand in train_hands)
        )
        self.assertTrue(
            all(
                sum(ex.table_id == hand for ex in validation) == 3
                for hand in validation_hands
            )
        )

    def test_signed_return_reinforces_wins_and_suppresses_losses(self) -> None:
        config = TrainingConfig(learning_rate=0.01)
        winning = _example("win", 2, 0.4, 100.0)
        losing = _example("loss", 2, 0.4, -100.0)
        initial = _init_weights(random.Random(5))
        winner_weights = copy.deepcopy(initial)
        loser_weights = copy.deepcopy(initial)
        before = _forward(initial, winning.features)["action_probabilities"][2]

        _step(winner_weights, winning, config, policy_objective="reward")
        _step(loser_weights, losing, config, policy_objective="reward")

        winner_after = _forward(winner_weights, winning.features)[
            "action_probabilities"
        ][2]
        loser_after = _forward(loser_weights, losing.features)["action_probabilities"][
            2
        ]
        self.assertGreater(winner_after, before)
        self.assertLess(loser_after, before)
        reinforcement = winner_weights["action_b"][2] - initial["action_b"][2]
        punishment = initial["action_b"][2] - loser_weights["action_b"][2]
        self.assertGreater(reinforcement, punishment)

    def test_reward_is_percentage_based_and_logarithmic(self) -> None:
        ten_percent = _example("ten", 2, 0.0, 10.0, purse_bb=100.0)
        one_percent = _example("one", 2, 0.0, 10.0, purse_bb=1_000.0)
        hundred_percent = _example("double", 2, 0.0, 100.0, purse_bb=100.0)
        three_hundred_percent = _example("quadruple", 2, 0.0, 300.0, purse_bb=100.0)

        self.assertAlmostEqual(_log_purse_return(ten_percent), math.log1p(0.10))
        self.assertAlmostEqual(_log_purse_return(one_percent), math.log1p(0.01))
        self.assertGreater(
            _log_purse_return(ten_percent), _log_purse_return(one_percent)
        )
        self.assertAlmostEqual(_log_purse_return(hundred_percent), math.log(2.0))
        self.assertAlmostEqual(_log_purse_return(three_hundred_percent), math.log(4.0))

        config = TrainingConfig(return_scale_fraction=0.2)
        self.assertGreater(_policy_signal(one_percent, config)[1], 0.0)
        self.assertLess(_policy_signal(_example("loss", 2, 0.0, -0.01), config)[1], 0.0)

    def test_uncertain_opponent_context_is_discounted(self) -> None:
        config = TrainingConfig(return_scale_fraction=0.2)
        confident = _example("known", 2, 0.0, 10.0, opponent_confidence=1.0)
        uncertain = _example("unknown", 2, 0.0, 10.0, opponent_confidence=0.0)

        self.assertAlmostEqual(
            _policy_signal(uncertain, config)[1],
            _policy_signal(confident, config)[1] * 0.25,
        )

    def test_behavior_only_data_cannot_supply_reward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no counterfactual"):
                train_candidate(
                    (
                        _example(
                            "direct",
                            2,
                            0.5,
                            100.0,
                            counterfactual=False,
                        ),
                    ),
                    directory,
                )

    def test_behavior_corpus_does_not_control_value_normalization(self) -> None:
        counterfactual_features = (2.0,) + (0.1,) * (LEARNING_INPUT_SIZE - 1)
        behavior_features = (10_000.0,) + (0.1,) * (LEARNING_INPUT_SIZE - 1)
        examples = (
            _example(
                "counterfactual",
                2,
                0.2,
                1.0,
                features=counterfactual_features,
            ),
            _example(
                "behavior",
                0,
                0.0,
                0.0,
                features=behavior_features,
                counterfactual=False,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = train_candidate(
                examples,
                directory,
                TrainingConfig(
                    epochs=1,
                    validation_fraction=0.0,
                    model_version="normalization-source-test",
                ),
            )
            weights = json.loads(summary.weights_path.read_text())
            manifest = json.loads(summary.manifest_path.read_text())

        self.assertEqual(weights["feature_normalization"]["means"][0], 2.0)
        self.assertEqual(manifest["training_window"]["behavior_example_count"], 1)
        self.assertIn(
            "counterfactual rows",
            manifest["training"]["input_normalization"],
        )

    def test_action_value_target_is_bounded_and_positive_weight_is_larger(self) -> None:
        config = TrainingConfig(return_scale_fraction=0.2)
        win = _example("win", 2, 0.0, 100.0)
        loss = _example("loss", 2, 0.0, -100.0)

        self.assertEqual(_action_value_target(win, config), 1.0)
        self.assertEqual(_action_value_target(loss, config), -1.0)
        self.assertAlmostEqual(
            _action_value_weight(win, config),
            1.5 * _action_value_weight(loss, config),
        )

    def test_sizing_head_only_learns_from_profitable_aggression(self) -> None:
        config = TrainingConfig(learning_rate=0.01, train_risk_head=True)
        initial = _init_weights(random.Random(7))
        losing_weights = copy.deepcopy(initial)
        winning_weights = copy.deepcopy(initial)

        _step(
            losing_weights,
            _example("loss", 2, 1.0, -100.0),
            config,
            policy_objective="reward",
        )
        _step(
            winning_weights,
            _example("win", 2, 1.0, 100.0),
            config,
            policy_objective="reward",
        )

        self.assertEqual(losing_weights["risk_fraction_w"], initial["risk_fraction_w"])
        self.assertEqual(losing_weights["risk_fraction_b"], initial["risk_fraction_b"])
        self.assertNotEqual(
            winning_weights["risk_fraction_b"], initial["risk_fraction_b"]
        )

    def test_imitation_warmup_does_not_teach_sizing(self) -> None:
        config = TrainingConfig(learning_rate=0.01)
        rng = random.Random(9)
        weights = _init_weights(rng)
        behavior_head = _init_behavior_head(rng)
        action_weights_before = copy.deepcopy(weights["action_w"])
        trunk_before = copy.deepcopy(weights["w2"])
        risk_weights_before = copy.deepcopy(weights["risk_fraction_w"])
        risk_bias_before = weights["risk_fraction_b"]

        _step(
            weights,
            _example("hand", 2, 1.0, 100.0),
            config,
            policy_objective="imitation",
            behavior_head=behavior_head,
        )

        self.assertEqual(weights["action_w"], action_weights_before)
        self.assertNotEqual(weights["w2"], trunk_before)
        self.assertEqual(weights["risk_fraction_w"], risk_weights_before)
        self.assertEqual(weights["risk_fraction_b"], risk_bias_before)

    def test_hybrid_threshold_is_calibrated_against_recorded_behavior(self) -> None:
        examples = (
            _example(
                "hand",
                0,
                0.0,
                0.0,
                decision_id="decision",
                behavior_probabilities=(1.0, 0.0, 0.0),
            ),
            _example(
                "hand",
                2,
                0.2,
                10.0,
                decision_id="decision",
                behavior_probabilities=(1.0, 0.0, 0.0),
            ),
        )
        calibration = _hybrid_calibration(
            examples,
            ([0.0, 0.0, 0.2], [0.0, 0.0, 0.2]),
        )

        self.assertIsNotNone(calibration)
        assert calibration is not None
        self.assertEqual(calibration["behavior_mean_regret_pct"], 10.0)
        self.assertEqual(calibration["recommended_min_value_advantage"], 0.0)

    def test_wide_scale_features_are_learnable_after_normalization(self) -> None:
        # A four-digit feature (spr in deep games) used to saturate the
        # softmax and kill the ReLU trunk, collapsing every candidate to a
        # constant predictor. Normalization must keep such signals usable.
        examples = []
        for index in range(120):
            aggress = index % 2 == 0
            features = [0.1] * LEARNING_INPUT_SIZE
            features[30] = 5_000.0 if aggress else -5_000.0
            decision_id = f"decision-{index}"
            for action in (0, 2):
                examples.append(
                    _example(
                        f"table-{index}",
                        action,
                        0.2 if action == 2 else 0.0,
                        1.0 if (action == 2) == aggress else -1.0,
                        features=tuple(features),
                        decision_id=decision_id,
                    )
                )
        with tempfile.TemporaryDirectory() as directory:
            summary = train_candidate(
                tuple(examples),
                directory,
                TrainingConfig(
                    epochs=1,
                    behavior_warmup_epochs=5,
                    validation_fraction=0.25,
                    model_version="scale-test",
                ),
            )
            weights = json.loads(summary.weights_path.read_text())

        self.assertGreaterEqual(summary.validation_best_action_accuracy, 0.9)
        self.assertLess(summary.validation_mean_regret_pct, 1.0)
        normalization = weights["feature_normalization"]
        self.assertEqual(len(normalization["means"]), LEARNING_INPUT_SIZE)
        self.assertEqual(len(normalization["stds"]), LEARNING_INPUT_SIZE)

    def test_reward_weighting_tilts_toward_profitable_actions(self) -> None:
        # Same situation, conflicting labels: half the teachers folded and
        # lost, half aggressed and won. Reward weighting must break the tie
        # toward the winning action.
        examples = []
        for index in range(60):
            aggress = index % 2 == 0
            examples.append(
                _example(
                    f"table-{index}",
                    2 if aggress else 0,
                    0.2 if aggress else 0.0,
                    8.0 if aggress else -2.0,
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            summary = train_candidate(
                tuple(examples),
                directory,
                TrainingConfig(
                    epochs=4,
                    validation_fraction=0.2,
                    model_version="advantage-test",
                    return_scale_fraction=0.2,
                ),
            )
            weights = json.loads(summary.weights_path.read_text())

        normalization = weights["feature_normalization"]
        probe = _normalized(examples[0], normalization["means"], normalization["stds"])
        probabilities = _forward(weights["weights"], probe.features)[
            "action_probabilities"
        ]
        self.assertEqual(max(range(3), key=probabilities.__getitem__), 2)

    def test_class_weight_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                train_candidate(
                    (_example("t", 0, 0.0, 0.0),),
                    directory,
                    TrainingConfig(class_weights=(1.0, -1.0, 1.0)),
                )

    def test_warmup_epoch_validation(self) -> None:
        examples = (_example("t", 0, 0.0, 0.0),)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                train_candidate(
                    examples,
                    directory,
                    TrainingConfig(baseline_warmup_epochs=1),
                )
            with self.assertRaises(ValueError):
                train_candidate(
                    examples,
                    directory,
                    TrainingConfig(behavior_warmup_epochs=-1),
                )
            # 1.0 disables the positive tilt and is legal since the
            # 2026-08-14 amendment; below 1.0 stays rejected.
            with self.assertRaises(ValueError):
                train_candidate(
                    examples,
                    directory,
                    TrainingConfig(reinforcement_multiplier=0.99),
                )
            with self.assertRaises(ValueError):
                train_candidate(
                    examples,
                    directory,
                    TrainingConfig(gradient_clip=0.0),
                )
            with self.assertRaises(ValueError):
                train_candidate(
                    examples,
                    directory,
                    TrainingConfig(return_scale_fraction=0.0),
                )
            with self.assertRaises(ValueError):
                train_candidate(
                    examples,
                    directory,
                    TrainingConfig(device="quantum"),
                )
            with self.assertRaises(ValueError):
                train_candidate(
                    examples,
                    directory,
                    TrainingConfig(batch_size=0),
                )


if __name__ == "__main__":
    unittest.main()
