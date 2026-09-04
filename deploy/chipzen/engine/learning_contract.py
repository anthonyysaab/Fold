"""Versioned feature and artifact contract for the learned poker policy."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from engine.branch_contract_v9 import MODEL_FORMAT_VERSION_V9
from engine.policy_features import FEATURE_NAMES, LABELS

# Schema 2 (2026-08-12) adds the session opponent model and table standing
# to the inputs: three cycles proved diet balancing alone cannot teach
# opponent-conditional aggression, because "who am I facing" was not
# visible to the network. Schema-1 artifacts (138 inputs) no longer load.
FEATURE_SCHEMA_VERSION = 2
MODEL_FORMAT = "fold-multihead-policy"
MODEL_FORMAT_VERSION = 1
HIDDEN_SIZES = (128, 64)
HEAD_SIZES = {"action_logits": 3, "playability": 1, "risk_fraction": 1}
PARAMETER_COUNT = 26_885

# Format 2 (2026-08-14, "v7"): manifest-driven two-branch architecture with a
# size-conditioned action-value head. The action head's outputs are branches,
# not families: aggression is valued at the size the policy would actually
# play, which is the repair for the counterfactual sizing defect. Format-1
# artifacts keep loading unchanged.
MODEL_FORMAT_VERSION_V7 = 2
BRANCH_LABELS = ("fold", "check_call", "aggress_half_pot", "aggress_pot")
BRANCH_FAMILIES = ("fold", "check_call", "aggress", "aggress")
BRANCH_POT_FRACTIONS = (None, None, 0.5, 1.0)
V7_HEAD_SIZES = {
    "action_value": len(BRANCH_LABELS),
    "behavior_prior": 3,
    "state_value": 1,
    "residual_scale": len(BRANCH_LABELS),
}
# Heads the pure-Python serve path evaluates; the rest are diagnostics.
V7_SERVE_HEADS = ("action_value",)

EXTRA_FEATURE_NAMES = (
    "hand_strength",
    "board_tier_fresh",
    "board_tier_thin",
    "board_tier_kicker",
    "risk_temperature",
    "risk_weakness",
    "risk_bet_pressure",
    "risk_distance_from_river",
    "risk_player_pressure",
    "call_effective_stack_fraction",
    "min_aggressive_effective_stack_fraction",
    "hero_aggression_count",
    "opponent_aggression_count",
    # Schema 2: session opponent evidence and standing.
    "opponent_range_width",
    "opponent_max_wildness",
    "opponent_max_stickiness",
    "lead_position_unit",
)
LEARNING_FEATURE_NAMES = (*FEATURE_NAMES, *EXTRA_FEATURE_NAMES)
LEARNING_INPUT_SIZE = len(LEARNING_FEATURE_NAMES)


class LearningContractError(ValueError):
    """Raised when features or a learned-policy artifact violate the contract."""


def card_feature_indices() -> tuple[int, ...]:
    """Indices of the raw card one-hots within the learning vector.

    ``hole_known_fraction`` shares the prefix but is a scalar, not a card
    identity, so it stays in the context block. The complement of this set
    is the 38-input context block; the OOD guard uses the complement because
    one-hot card features z-score to 5.0-6.1 whenever set, which made the
    old whole-vector guard detect card identity instead of state novelty.
    """

    return tuple(
        index
        for index, name in enumerate(LEARNING_FEATURE_NAMES)
        if name.startswith(("hole_", "board_"))
        and name != "hole_known_fraction"
        and not name.startswith("board_tier_")
    )


def context_feature_indices() -> tuple[int, ...]:
    """Indices of every non-card learning input (the OOD-guard block)."""

    cards = set(card_feature_indices())
    return tuple(index for index in range(LEARNING_INPUT_SIZE) if index not in cards)


def default_v7_architecture() -> dict[str, object]:
    """The build-now v7 instantiation from the 2026-08-14 design review.

    Two encoders whose input partition is recorded by index so a schema
    change cannot silently re-split the vector, a three-layer shared trunk,
    and per-head towers. Widening to the ~1M instantiation is editing the
    four width numbers; the validator only checks internal consistency.
    """

    return {
        "family": "v7-two-branch",
        "card_indices": list(card_feature_indices()),
        "context_indices": list(context_feature_indices()),
        "encoder_width": 96,
        "trunk_widths": [256, 256, 128],
        "head_towers": {
            "action_value": 64,
            "behavior_prior": 32,
            "state_value": 32,
            "residual_scale": 32,
        },
        "heads": dict(V7_HEAD_SIZES),
        "layer_norm": True,
        "dropout": 0.2,
    }


def validate_v7_architecture(architecture: object) -> None:
    """Structural checks for a format-2 architecture descriptor."""

    if not isinstance(architecture, Mapping):
        raise LearningContractError("architecture must be an object")
    if architecture.get("family") != "v7-two-branch":
        raise LearningContractError("unsupported architecture family")
    cards = architecture.get("card_indices")
    context = architecture.get("context_indices")
    if not isinstance(cards, Sequence) or not isinstance(context, Sequence):
        raise LearningContractError("architecture must list both input partitions")
    combined = sorted([*cards, *context])
    if combined != list(range(LEARNING_INPUT_SIZE)):
        raise LearningContractError(
            "card and context indices must partition the learning inputs"
        )
    width = architecture.get("encoder_width")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise LearningContractError("encoder_width must be a positive integer")
    trunk = architecture.get("trunk_widths")
    if (
        not isinstance(trunk, Sequence)
        or not trunk
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in trunk
        )
    ):
        raise LearningContractError("trunk_widths must be positive integers")
    heads = architecture.get("heads")
    if heads != V7_HEAD_SIZES:
        raise LearningContractError("architecture heads do not match the contract")
    towers = architecture.get("head_towers")
    if not isinstance(towers, Mapping) or set(towers) != set(V7_HEAD_SIZES):
        raise LearningContractError("head_towers must cover exactly the heads")
    for name, tower in towers.items():
        if isinstance(tower, bool) or not isinstance(tower, int) or tower < 1:
            raise LearningContractError(f"head tower {name} must be positive")
    dropout = architecture.get("dropout")
    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise LearningContractError("dropout must be in [0, 1)")
    if not isinstance(architecture.get("layer_norm"), bool):
        raise LearningContractError("layer_norm must be a boolean")


def _unit_interval(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise LearningContractError(f"{name} must be finite and between 0 and 1")
    return number


def build_learning_features(
    base_features: Sequence[float],
    *,
    hand_strength: float,
    board_tier: str,
    risk_temperature: float,
    risk_factors: Mapping[str, float],
    call_effective_stack_fraction: float,
    min_aggressive_effective_stack_fraction: float,
    hero_aggression_count: int,
    opponent_aggression_count: int,
    opponent_range_width: float,
    opponent_max_wildness: float,
    opponent_max_stickiness: float,
    lead_position_unit: float,
) -> tuple[float, ...]:
    """Combine validated live features with the 17 learning-only inputs."""

    if len(base_features) != len(FEATURE_NAMES):
        raise LearningContractError(
            f"expected {len(FEATURE_NAMES)} base features, found {len(base_features)}"
        )
    base = tuple(float(value) for value in base_features)
    if not all(math.isfinite(value) for value in base):
        raise LearningContractError("base features must be finite")
    if board_tier not in {"fresh", "thin", "kicker"}:
        raise LearningContractError("board_tier must be fresh, thin, or kicker")
    expected_factors = {
        "hand_strength",
        "bet_pressure",
        "distance_from_river",
        "players",
    }
    if set(risk_factors) != expected_factors:
        raise LearningContractError(
            "risk_factors do not match the temperature contract"
        )
    for name in ("hero_aggression_count", "opponent_aggression_count"):
        value = locals()[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LearningContractError(f"{name} must be a non-negative integer")

    extras = (
        _unit_interval("hand_strength", hand_strength),
        float(board_tier == "fresh"),
        float(board_tier == "thin"),
        float(board_tier == "kicker"),
        _unit_interval("risk_temperature", risk_temperature),
        _unit_interval("risk_weakness", risk_factors["hand_strength"]),
        _unit_interval("risk_bet_pressure", risk_factors["bet_pressure"]),
        _unit_interval("risk_distance_from_river", risk_factors["distance_from_river"]),
        _unit_interval("risk_player_pressure", risk_factors["players"]),
        _unit_interval("call_effective_stack_fraction", call_effective_stack_fraction),
        _unit_interval(
            "min_aggressive_effective_stack_fraction",
            min_aggressive_effective_stack_fraction,
        ),
        float(hero_aggression_count),
        float(opponent_aggression_count),
        _unit_interval("opponent_range_width", opponent_range_width),
        _unit_interval("opponent_max_wildness", opponent_max_wildness),
        _unit_interval("opponent_max_stickiness", opponent_max_stickiness),
        _unit_interval("lead_position_unit", lead_position_unit),
    )
    features = (*base, *extras)
    if len(features) != LEARNING_INPUT_SIZE:
        raise AssertionError("learning feature contract has the wrong size")
    return features


def _timestamp(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise LearningContractError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LearningContractError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise LearningContractError(f"{name} must include a timezone")


def _validate_engine_parameters(parameters: object) -> None:
    """Check an artifact's optional engine-parameter blocks.

    Each present block must rebuild through its dataclass ``from_mapping``,
    so an artifact can never carry gate, shaping, tracker, or bluff values
    outside the validated learning ranges. Imports happen lazily because the
    decision engine imports this module at load time.
    """

    from bluff import BluffSettings

    from engine.decision_engine import (
        SafetyGates,
        TemperatureShaping,
    )
    from engine.opponent_model import TrackerSettings

    if not isinstance(parameters, Mapping):
        raise LearningContractError("engine_parameters must be an object")
    validators = {
        "safety_gates": SafetyGates.from_mapping,
        "temperature_shaping": TemperatureShaping.from_mapping,
        "tracker_settings": TrackerSettings.from_mapping,
        "bluff_settings": BluffSettings.from_mapping,
    }
    unknown = sorted(set(parameters) - set(validators))
    if unknown:
        raise LearningContractError(
            f"unknown engine_parameters blocks: {', '.join(unknown)}"
        )
    for name, rebuild in validators.items():
        block = parameters.get(name)
        if block is None:
            continue
        if not isinstance(block, Mapping):
            raise LearningContractError(f"engine_parameters.{name} must be an object")
        try:
            rebuild(block)
        except (TypeError, ValueError) as exc:
            raise LearningContractError(
                f"engine_parameters.{name} is invalid: {exc}"
            ) from exc


def _validate_v9_manifest_schema(
    manifest: Mapping[str, object], architecture: Mapping[str, object]
) -> None:
    """Contract-side schema checks for a format-4 (v9) artifact.

    Format 4 is schema 4: 414 inputs, the four v9 branch labels, and the
    composed-value architecture. None of the 142-input constants in this
    module describe it, so this is a separate path rather than a widening
    of them -- the reasoning ``validate_v9_architecture`` records, that a
    check accepting both shapes is how an artifact of one schema reaches
    live play under the other's meanings unnoticed.

    ``validate_v9_architecture`` lives in :mod:`engine.v8_trainer`, which
    imports this module, so that import is deferred to call time.
    """

    from engine import schema4
    from engine.branch_contract_v9 import BRANCH_LABELS_V9
    from engine.v8_trainer import validate_v9_architecture

    if manifest.get("feature_schema_version") != schema4.SCHEMA_VERSION_V9:
        raise LearningContractError("unsupported feature schema version")
    if manifest.get("input_size") != schema4.INPUT_SIZE_V9:
        raise LearningContractError("artifact input size does not match the contract")
    if tuple(manifest.get("feature_names", ())) != tuple(schema4.FEATURE_NAMES_V9):
        raise LearningContractError("artifact feature names do not match the contract")
    if tuple(manifest.get("action_labels", ())) != tuple(BRANCH_LABELS_V9):
        raise LearningContractError(
            "format-4 artifacts must label the four v9 branches"
        )
    try:
        validate_v9_architecture(architecture)
    except ValueError as error:
        raise LearningContractError(f"invalid v9 architecture: {error}") from error

    # Phase-A and Phase-B v9 artifacts are indistinguishable by format,
    # architecture, and action labels -- a Phase-A artifact trains the
    # component heads only, and its own trainer has always stamped a prose
    # note saying it must not be deployed. Prose cannot stop a promotion,
    # so the trainers now emit ``serve.deployable`` and this reads it.
    # Fail-closed on absence: an artifact that does not claim to be
    # servable is not servable, which keeps every manifest written before
    # the marker existed out of live play until it is re-stamped.
    serve = manifest.get("serve")
    if not isinstance(serve, Mapping):
        raise LearningContractError("format-4 artifacts must carry a serve block")
    if serve.get("deployable") is not True:
        raise LearningContractError(
            "artifact does not declare serve.deployable: only a Phase-B "
            "composed-value v9 artifact may be promoted"
        )
    # The marker on its own is a declaration, and a declaration can be
    # hand-edited onto a Phase-A manifest. Require the Phase-B trainer's
    # own provenance alongside it, so claiming deployability also means
    # forging a corpus record: Phase A writes ``dataset``/``row_count``
    # and has no ``phase_b_decisions`` to offer.
    window = manifest.get("training_window")
    decisions = window.get("phase_b_decisions") if isinstance(window, Mapping) else None
    if isinstance(decisions, bool) or not isinstance(decisions, int) or decisions < 1:
        raise LearningContractError(
            "a deployable format-4 artifact must record "
            "training_window.phase_b_decisions: Phase-A component heads "
            "are not a serve path"
        )


def validate_artifact_manifest(manifest: Mapping[str, object]) -> None:
    """Validate metadata for an immutable, retrievable policy artifact."""

    if manifest.get("format") != MODEL_FORMAT:
        raise LearningContractError("unsupported artifact format")
    format_version = manifest.get("format_version")
    if format_version not in {
        MODEL_FORMAT_VERSION,
        MODEL_FORMAT_VERSION_V7,
        MODEL_FORMAT_VERSION_V9,
    }:
        raise LearningContractError("unsupported artifact format version")
    architecture = manifest.get("architecture")
    if not isinstance(architecture, Mapping):
        raise LearningContractError("artifact architecture must be an object")
    if format_version == MODEL_FORMAT_VERSION_V9:
        # Format 4 carries schema 4 (414 inputs, four v9 branches), so the
        # 142-input checks below cannot apply to it. Kept as a separate
        # branch rather than widened constants for the reason
        # ``validate_v9_architecture`` records: a check that accepts both
        # schemas is how an artifact of one shape reaches live play under
        # the other's meanings without anyone noticing.
        _validate_v9_manifest_schema(manifest, architecture)
    else:
        if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise LearningContractError("unsupported feature schema version")
        if manifest.get("input_size") != LEARNING_INPUT_SIZE:
            raise LearningContractError(
                "artifact input size does not match the contract"
            )
        if tuple(manifest.get("feature_names", ())) != LEARNING_FEATURE_NAMES:
            raise LearningContractError(
                "artifact feature names do not match the contract"
            )
        if format_version == MODEL_FORMAT_VERSION_V7:
            if tuple(manifest.get("action_labels", ())) != BRANCH_LABELS:
                raise LearningContractError(
                    "format-2 artifacts must label the four value branches"
                )
            validate_v7_architecture(architecture)
        else:
            if tuple(manifest.get("action_labels", ())) != LABELS:
                raise LearningContractError(
                    "artifact action labels do not match the contract"
                )
            if tuple(architecture.get("hidden_sizes", ())) != HIDDEN_SIZES:
                raise LearningContractError(
                    "artifact hidden sizes do not match the contract"
                )
            if architecture.get("heads") != HEAD_SIZES:
                raise LearningContractError(
                    "artifact heads do not match the contract"
                )

    model_version = manifest.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        raise LearningContractError("model_version must be a non-empty string")
    state = manifest.get("state")
    if state not in {"candidate", "approved", "retired"}:
        raise LearningContractError("artifact state is invalid")
    parent = manifest.get("parent_version")
    if parent is not None and (not isinstance(parent, str) or not parent.strip()):
        raise LearningContractError("parent_version must be null or a non-empty string")
    _timestamp("created_at", manifest.get("created_at"))

    weights_file = manifest.get("weights_file")
    if (
        not isinstance(weights_file, str)
        or not weights_file
        or Path(weights_file).name != weights_file
        or not weights_file.endswith(".json")
    ):
        raise LearningContractError("weights_file must be a local JSON filename")
    checksum = manifest.get("weights_sha256")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise LearningContractError("weights_sha256 must be lowercase hexadecimal")

    parameters = manifest.get("engine_parameters")
    if parameters is not None:
        _validate_engine_parameters(parameters)

    training_window = manifest.get("training_window")
    if not isinstance(training_window, Mapping):
        raise LearningContractError("training_window must be an object")
    hand_count = training_window.get("hand_count")
    if (
        isinstance(hand_count, bool)
        or not isinstance(hand_count, int)
        or hand_count < 0
    ):
        raise LearningContractError("training_window.hand_count must be non-negative")
    if not isinstance(manifest.get("evaluation"), Mapping):
        raise LearningContractError("evaluation must be an object")
    promotion = manifest.get("promotion")
    if state == "approved":
        if not isinstance(promotion, Mapping):
            raise LearningContractError("approved artifacts require promotion metadata")
        _timestamp("promotion.approved_at", promotion.get("approved_at"))
        if (
            not isinstance(promotion.get("reason"), str)
            or not promotion["reason"].strip()
        ):
            raise LearningContractError("approved artifacts require a promotion reason")
    elif promotion is not None:
        raise LearningContractError(
            "only approved artifacts may have promotion metadata"
        )


__all__ = [
    "BRANCH_FAMILIES",
    "BRANCH_LABELS",
    "BRANCH_POT_FRACTIONS",
    "build_learning_features",
    "card_feature_indices",
    "context_feature_indices",
    "default_v7_architecture",
    "EXTRA_FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "HEAD_SIZES",
    "HIDDEN_SIZES",
    "LEARNING_FEATURE_NAMES",
    "LEARNING_INPUT_SIZE",
    "LearningContractError",
    "MODEL_FORMAT",
    "MODEL_FORMAT_VERSION",
    "MODEL_FORMAT_VERSION_V7",
    "PARAMETER_COUNT",
    "V7_HEAD_SIZES",
    "V7_SERVE_HEADS",
    "validate_artifact_manifest",
    "validate_v7_architecture",
]
