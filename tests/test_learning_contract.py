"""Checks for the versioned multi-head learning contract."""

from __future__ import annotations

import unittest

from engine.learning_contract import (
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
from engine.policy_features import FEATURE_NAMES, LABELS


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

        from engine.decision_engine import (
            DEFAULT_SAFETY_GATES,
            DEFAULT_TEMPERATURE_SHAPING,
        )
        from engine.opponent_model import DEFAULT_TRACKER_SETTINGS

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


def _manifest_v9() -> dict:
    """A format-4 manifest as the v9 Phase-B trainer writes one.

    Built from the live constants rather than a checked-in artifact:
    ``artifacts/candidates/`` is gitignored, so a test that read one
    would pass here and skip on every fresh clone.
    """

    from engine import schema4
    from engine.branch_contract_v9 import BRANCH_LABELS_V9, MODEL_FORMAT_VERSION_V9
    from engine.v8_trainer import default_v9_architecture

    return {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION_V9,
        "model_version": "candidate-v9-0000",
        "state": "candidate",
        "parent_version": None,
        "created_at": "2026-09-02T12:00:00Z",
        "feature_schema_version": schema4.SCHEMA_VERSION_V9,
        "input_size": schema4.INPUT_SIZE_V9,
        "feature_names": list(schema4.FEATURE_NAMES_V9),
        "action_labels": list(BRANCH_LABELS_V9),
        "architecture": default_v9_architecture(),
        "serve": {"deployable": True},
        "weights_file": "weights.json",
        "weights_sha256": "b" * 64,
        "training_window": {"hand_count": 12, "phase_b_decisions": 56043},
        "evaluation": {},
        "promotion": None,
    }


class FormatFourPromotionTests(unittest.TestCase):
    """The contract gate that makes a v9 artifact promotable, and only a v9
    Phase-B one. Format 4 is schema 4 (414 inputs, four v9 branches); none
    of the 142-input constants describe it, so the risk this class guards
    is a manifest of one schema passing under the other's checks."""

    def test_a_phase_b_v9_manifest_validates(self) -> None:
        validate_artifact_manifest(_manifest_v9())

    def test_the_142_input_schema_is_not_accepted_under_format_four(self) -> None:
        for field, wrong in (
            ("input_size", LEARNING_INPUT_SIZE),
            ("feature_schema_version", FEATURE_SCHEMA_VERSION),
            ("feature_names", list(LEARNING_FEATURE_NAMES)),
            ("action_labels", list(LABELS)),
        ):
            invalid = _manifest_v9()
            invalid[field] = wrong
            with self.assertRaises(
                LearningContractError, msg=f"format 4 accepted a v1 {field}"
            ):
                validate_artifact_manifest(invalid)

    def test_a_format_two_body_relabelled_format_four_is_refused(self) -> None:
        # The cross-schema case that matters: an artifact of the old shape
        # wearing the new version number must not reach the v9 branch and
        # be served under v9 slot meanings.
        relabelled = _manifest()
        relabelled["format_version"] = 4
        with self.assertRaises(LearningContractError):
            validate_artifact_manifest(relabelled)

    def test_a_phase_a_artifact_is_refused(self) -> None:
        # Phase A trains the component heads only and its own trainer has
        # always stamped a prose note saying it must not be deployed. The
        # two phases are identical in format, architecture and labels, so
        # the refusal has to come from the machine-readable marker.
        phase_a = _manifest_v9()
        phase_a["serve"] = {"deployable": False}
        with self.assertRaises(LearningContractError):
            validate_artifact_manifest(phase_a)

    def test_the_marker_is_fail_closed(self) -> None:
        # Every manifest written before the marker existed lacks it, and
        # must stay out of live play rather than inherit a default.
        missing_key = _manifest_v9()
        missing_key["serve"] = {}
        no_block = _manifest_v9()
        del no_block["serve"]
        for invalid in (missing_key, no_block):
            with self.assertRaises(LearningContractError):
                validate_artifact_manifest(invalid)

    def test_the_marker_must_be_a_real_boolean(self) -> None:
        # A JSON round-trip that turned the flag into a truthy string, or
        # an int, must not read as consent to deploy.
        for truthy in ("true", 1, [1], {"yes": True}):
            invalid = _manifest_v9()
            invalid["serve"] = {"deployable": truthy}
            with self.assertRaises(
                LearningContractError, msg=f"{truthy!r} read as deployable"
            ):
                validate_artifact_manifest(invalid)

    def test_deployability_also_requires_phase_b_provenance(self) -> None:
        # The marker alone is a declaration, and a declaration can be
        # hand-edited onto a Phase-A manifest. Claiming deployability has
        # to mean forging a corpus record as well: Phase A writes
        # ``dataset``/``row_count`` and has no phase_b_decisions.
        phase_a_window = _manifest_v9()
        phase_a_window["training_window"] = {
            "hand_count": 1195,
            "dataset": "phase-a-dataset-v9.jsonl.gz",
            "row_count": 9084,
        }
        with self.assertRaises(LearningContractError):
            validate_artifact_manifest(phase_a_window)

        for bogus in (0, -1, True, "56043", None):
            invalid = _manifest_v9()
            invalid["training_window"] = {
                "hand_count": 12,
                "phase_b_decisions": bogus,
            }
            with self.assertRaises(
                LearningContractError, msg=f"{bogus!r} read as a corpus"
            ):
                validate_artifact_manifest(invalid)

    def test_a_v8_architecture_cannot_ride_a_format_four_manifest(self) -> None:
        invalid = _manifest_v9()
        invalid["architecture"] = dict(invalid["architecture"])
        invalid["architecture"]["family"] = "v8-composed-value"
        with self.assertRaises(LearningContractError):
            validate_artifact_manifest(invalid)


if __name__ == "__main__":
    unittest.main()
