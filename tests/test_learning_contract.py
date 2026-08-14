"""Checks for the versioned multi-head learning contract."""

from __future__ import annotations

import unittest

from devfun_poker_playground.learning_contract import (
    build_learning_features,
    FEATURE_SCHEMA_VERSION,
    HEAD_SIZES,
    HIDDEN_SIZES,
    LEARNING_FEATURE_NAMES,
    LEARNING_INPUT_SIZE,
    LearningContractError,
    MODEL_FORMAT,
    MODEL_FORMAT_VERSION,
    PARAMETER_COUNT,
    validate_artifact_manifest,
)
from devfun_poker_playground.policy_features import FEATURE_NAMES, LABELS


def _manifest() -> dict:
    return {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "model_version": "candidate-0001",
        "state": "candidate",
        "parent_version": None,
        "created_at": "2026-08-12T12:00:00Z",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_size": LEARNING_INPUT_SIZE,
        "feature_names": list(LEARNING_FEATURE_NAMES),
        "action_labels": list(LABELS),
        "architecture": {
            "hidden_sizes": list(HIDDEN_SIZES),
            "heads": HEAD_SIZES.copy(),
        },
        "weights_file": "weights.json",
        "weights_sha256": "a" * 64,
        "training_window": {"hand_count": 0},
        "evaluation": {},
        "promotion": None,
    }


class LearningContractTests(unittest.TestCase):
    def test_feature_builder_has_the_declared_142_value_order(self) -> None:
        features = build_learning_features(
            [0.0] * len(FEATURE_NAMES),
            hand_strength=0.7,
            board_tier="thin",
            risk_temperature=0.4,
            risk_factors={
                "hand_strength": 0.3,
                "bet_pressure": 0.2,
                "distance_from_river": 1 / 3,
                "players": 0.5,
            },
            call_effective_stack_fraction=0.1,
            min_aggressive_effective_stack_fraction=0.25,
            hero_aggression_count=1,
            opponent_aggression_count=2,
            opponent_range_width=0.45,
            opponent_max_wildness=0.7,
            opponent_max_stickiness=0.9,
            lead_position_unit=0.35,
        )

        self.assertEqual(len(features), 142)
        self.assertEqual(LEARNING_INPUT_SIZE, 142)
        self.assertEqual(
            features[-17:],
            (
                0.7, 0.0, 1.0, 0.0, 0.4, 0.3, 0.2, 1 / 3, 0.5, 0.1, 0.25, 1.0, 2.0,
                0.45, 0.7, 0.9, 0.35,
            ),
        )

    def test_parameter_count_matches_the_three_head_shape(self) -> None:
        calculated = (
            (LEARNING_INPUT_SIZE + 1) * HIDDEN_SIZES[0]
            + (HIDDEN_SIZES[0] + 1) * HIDDEN_SIZES[1]
            + (HIDDEN_SIZES[1] + 1) * sum(HEAD_SIZES.values())
        )
        self.assertEqual(calculated, PARAMETER_COUNT)

    def test_manifest_binds_weights_to_the_exact_contract(self) -> None:
        validate_artifact_manifest(_manifest())
        invalid = _manifest()
        invalid["weights_file"] = "../weights.json"
        with self.assertRaises(LearningContractError):
            validate_artifact_manifest(invalid)

    def test_manifest_validates_optional_engine_parameters(self) -> None:
        from bluff import DEFAULT_BLUFF_SETTINGS

        from devfun_poker_playground.decision_engine import (
            DEFAULT_SAFETY_GATES,
            DEFAULT_TEMPERATURE_SHAPING,
        )
        from devfun_poker_playground.opponent_model import DEFAULT_TRACKER_SETTINGS

        manifest = _manifest()
        manifest["engine_parameters"] = {
            "safety_gates": DEFAULT_SAFETY_GATES.to_mapping(),
            "temperature_shaping": DEFAULT_TEMPERATURE_SHAPING.to_mapping(),
            "tracker_settings": DEFAULT_TRACKER_SETTINGS.to_mapping(),
            "bluff_settings": DEFAULT_BLUFF_SETTINGS.to_mapping(),
        }
        validate_artifact_manifest(manifest)

        for bad_parameters in (
            {"unknown_block": {}},
            {"safety_gates": {"near_nut_floor": 0.2}},  # below break-even
            {"temperature_shaping": {"unknown_field": 1.0}},
            {"bluff_settings": {"bluff_density": 9.0}},
            {"tracker_settings": "not-an-object"},
        ):
            invalid = _manifest()
            invalid["engine_parameters"] = bad_parameters
            with self.assertRaises(
                LearningContractError, msg=f"no error for {bad_parameters}"
            ):
                validate_artifact_manifest(invalid)


if __name__ == "__main__":
    unittest.main()
