"""Offline counterfactual-value trainer for the versioned poker model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from bluff import DEFAULT_BLUFF_SETTINGS

from engine.decision_engine import (
    DEFAULT_SAFETY_GATES,
    DEFAULT_TEMPERATURE_SHAPING,
)
from engine.learning_contract import (
    BRANCH_FAMILIES,
    BRANCH_LABELS,
    context_feature_indices,
    default_v7_architecture,
    FEATURE_SCHEMA_VERSION,
    HEAD_SIZES,
    HIDDEN_SIZES,
    LEARNING_FEATURE_NAMES,
    LEARNING_INPUT_SIZE,
    MODEL_FORMAT,
    MODEL_FORMAT_VERSION,
    MODEL_FORMAT_VERSION_V7,
    V7_HEAD_SIZES,
    validate_artifact_manifest,
    validate_v7_architecture,
)
from engine.foreign_data import load_foreign_training_examples
from engine.opponent_model import DEFAULT_TRACKER_SETTINGS
from engine.policy_features import LABELS
from engine.training_telemetry import (
    TrainingExample,
    load_training_examples,
)

TRAINING_OBJECTIVE = "counterfactual_action_value_v6"
TRAINING_OBJECTIVE_V7 = "centered_counterfactual_branch_value_v7"
DEGENERATE_GROUP_FILTERS = ("off", "zero_weight", "drop", "random")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 8
    learning_rate: float = 0.01
    # A 20% purse move maps to one unit before clipping.
    return_scale_fraction: float = 0.2
    validation_fraction: float = 0.2
    seed: int = 17
    model_version: str | None = None
    baseline_warmup_epochs: int = 0
    behavior_warmup_epochs: int = 1
    # Per-family weights on the action objective (fold, check_call, aggress).
    # The foreign audit found 51.4% folds, so inverse-frequency weights keep
    # minority-family advantages visible.
    class_weights: tuple[float, float, float] | None = None
    # An equal purse gain should matter more than an equal purse loss.
    # 1.0 disables the tilt; the 2026-08-14 ablation measured it costing
    # regret, so v7 recipes pass 1.0 while v6 keeps its recorded 1.5.
    reinforcement_multiplier: float = 1.5
    # Per-parameter gradient bound for the small plain-SGD implementation.
    gradient_clip: float = 5.0
    device: str = "cpu"
    batch_size: int = 1024
    counterfactual_rollouts: int = 1
    # Keep sizing on the proven heuristic until action-value evaluation passes.
    train_risk_head: bool = False
    # The one seed used to double for both purposes; splitting them makes
    # "same corpus split, different initialization" experiments expressible.
    # None means fall back to `seed`, which preserves every old recipe.
    split_seed: int | None = None
    init_seed: int | None = None
    # "v6" is the frozen three-output path above; "v7" is the format-2
    # two-branch architecture with the size-conditioned value head.
    architecture: str = "v6"
    # v7 optimization recipe (ignored by the v6 path).
    weight_decay: float = 0.01
    dropout: float = 0.2
    warmup_steps: int = 200
    early_stop_patience: int = 10
    behavior_prior_weight: float = 0.2
    state_value_weight: float = 0.1
    residual_scale_weight: float = 0.05
    serve_temperature: float | None = None
    # Zero-signal decision groups -- every branch carrying the same weighted
    # value target -- are exactly minimised by a CONSTANT action head, and off
    # the constant they pull a discriminating head back toward one. They are
    # 29.66% of the candidate-v7-0001 corpus and 27.76% of the action-loss
    # gradient at a discriminating head. See
    # artifacts/evaluations/dead-head-objective-2026-08-27.md.
    #
    #   "off"          every group trains the action head. Reproduces every
    #                  artifact built before this option existed.
    #   "zero_weight"  zero-signal groups stay in their batch but carry no
    #                  action weight. Optimizer steps, the LR schedule, the
    #                  behavior:reward update ratio, and the state_value and
    #                  residual_scale supervision are all unchanged; only the
    #                  action objective's row set moves.
    #   "drop"         zero-signal groups leave the reward batches. That also
    #                  removes their state_value and residual_scale rows and
    #                  cuts reward steps per epoch (-10.1% on the v7-0001
    #                  corpus at batch_size 256).
    #
    #   "random"       the ATTRIBUTION CONTROL. Mutes a uniformly random set
    #                  of groups of exactly the same SIZE as the zero-signal
    #                  set, using `degenerate_group_filter_seed`. Without
    #                  this arm the experiment cannot separate "removing the
    #                  attractor helped" from "removing 29.66% of the groups
    #                  helped"; a treated arm that only matches this one has
    #                  demonstrated nothing about degeneracy. It mutes like
    #                  "zero_weight" so the two are comparable.
    #
    # v7 only: the v6 objective is not centered and has no such attractor.
    degenerate_group_filter: str = "off"
    #: Draw seed for the "random" arm. Recorded in the manifest so the mask
    #: is reproducible; irrelevant in every other mode.
    degenerate_group_filter_seed: int = 0

    @property
    def resolved_split_seed(self) -> int:
        return self.seed if self.split_seed is None else self.split_seed

    @property
    def resolved_init_seed(self) -> int:
        return self.seed if self.init_seed is None else self.init_seed


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    examples: int
    train_examples: int
    validation_examples: int
    train_loss: float
    validation_loss: float | None
    validation_best_action_accuracy: float | None
    validation_mean_regret_pct: float | None
    validation_action_value_mae_pct: float | None
    weights_sha256: str
    manifest_path: Path
    weights_path: Path
    # Format 2 only: the share of validation decision groups whose branches
    # all tie, on which accuracy and regret are undefined. None under
    # format 1, which has no branch-degeneracy concept.
    validation_degenerate_group_fraction: float | None = None


def _check_config(config: TrainingConfig) -> None:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.baseline_warmup_epochs != 0:
        raise ValueError(
            "baseline_warmup_epochs must be zero for action-value training"
        )
    if config.behavior_warmup_epochs < 0:
        raise ValueError("behavior_warmup_epochs cannot be negative")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if (
        not math.isfinite(config.return_scale_fraction)
        or config.return_scale_fraction <= 0.0
    ):
        raise ValueError("return_scale_fraction must be positive and finite")
    if not 0.0 <= config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if config.class_weights is not None:
        if len(config.class_weights) != len(LABELS):
            raise ValueError(f"class_weights must have {len(LABELS)} entries")
        for weight in config.class_weights:
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError("class_weights must be positive and finite")
    if (
        not math.isfinite(config.reinforcement_multiplier)
        or config.reinforcement_multiplier < 1.0
    ):
        raise ValueError("reinforcement_multiplier must be finite and at least 1")
    if config.architecture not in {"v6", "v7"}:
        raise ValueError("architecture must be v6 or v7")
    if config.degenerate_group_filter not in DEGENERATE_GROUP_FILTERS:
        raise ValueError(
            "degenerate_group_filter must be one of "
            + ", ".join(DEGENERATE_GROUP_FILTERS)
        )
    if config.degenerate_group_filter != "off" and config.architecture != "v7":
        raise ValueError(
            "degenerate_group_filter applies to the v7 centered objective only"
        )
    if config.architecture == "v7":
        if config.device != "cuda":
            raise ValueError("the v7 architecture trains on CUDA only")
        if not math.isfinite(config.weight_decay) or config.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative and finite")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if config.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if config.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be positive")
        if config.serve_temperature is not None and (
            not math.isfinite(config.serve_temperature)
            or config.serve_temperature <= 0.0
        ):
            raise ValueError("serve_temperature must be positive when set")
    if not math.isfinite(config.gradient_clip) or config.gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be positive and finite")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.counterfactual_rollouts < 1:
        raise ValueError("counterfactual_rollouts must be positive")


def validate_training_device(device: str) -> str:
    """Fail before harvesting if the requested accelerator is unavailable."""

    if device == "cpu":
        return "cpu"
    if device != "cuda":
        raise ValueError("device must be cpu or cuda")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA training requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but PyTorch cannot access the GPU")
    return str(torch.cuda.get_device_name(0))


def _class_weight(example: TrainingExample, config: TrainingConfig) -> float:
    if config.class_weights is None:
        return 1.0
    return config.class_weights[example.action_family_index]


def balanced_class_weights(
    examples: Sequence[TrainingExample],
) -> tuple[float, float, float]:
    """Inverse-frequency weights so each action family pulls equally."""

    counts = [0] * len(LABELS)
    for example in examples:
        counts[example.action_family_index] += 1
    total = max(1, len(examples))
    weights = tuple(total / (len(LABELS) * max(1, count)) for count in counts)
    return weights  # type: ignore[return-value]


def _feature_normalization(
    examples: Sequence[TrainingExample],
) -> tuple[list[float], list[float]]:
    """Per-feature mean and std over the supplied training examples.

    The contract carries unbounded raw features (``spr`` reaches four
    digits in deep games), which saturate the softmax and kill the ReLU
    trunk under plain SGD. Candidates therefore train on z-scored inputs
    and ship their normalization in the weights file, so any future
    runtime applies the identical transform.
    """

    count = max(1, len(examples))
    means = [0.0] * LEARNING_INPUT_SIZE
    for example in examples:
        for index, value in enumerate(example.features):
            means[index] += value
    means = [total / count for total in means]
    variances = [0.0] * LEARNING_INPUT_SIZE
    for example in examples:
        for index, value in enumerate(example.features):
            variances[index] += (value - means[index]) ** 2
    # Floor raised 1e-3 -> 0.05 (2026-08-14): near-constant inputs such as
    # the legality flags z-scored to +-1000 under the old floor, injecting
    # three-orders-of-magnitude spikes into a trunk whose other inputs are
    # order one.
    stds = [max(0.05, math.sqrt(total / count)) for total in variances]
    return means, stds


def _normalized(
    example: TrainingExample, means: Sequence[float], stds: Sequence[float]
) -> TrainingExample:
    return replace(
        example,
        features=tuple(
            (value - mean) / std
            for value, mean, std in zip(example.features, means, stds)
        ),
    )


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _init_weights(rng: random.Random) -> dict[str, object]:
    def matrix(rows: int, cols: int) -> list[list[float]]:
        scale = math.sqrt(2.0 / max(1, cols))
        return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    return {
        "w1": matrix(HIDDEN_SIZES[0], LEARNING_INPUT_SIZE),
        "b1": [0.0] * HIDDEN_SIZES[0],
        "w2": matrix(HIDDEN_SIZES[1], HIDDEN_SIZES[0]),
        "b2": [0.0] * HIDDEN_SIZES[1],
        "action_w": matrix(len(LABELS), HIDDEN_SIZES[1]),
        "action_b": [0.0] * len(LABELS),
        "playability_w": matrix(1, HIDDEN_SIZES[1])[0],
        "playability_b": 0.0,
        "risk_fraction_w": matrix(1, HIDDEN_SIZES[1])[0],
        "risk_fraction_b": 0.0,
    }


def _init_behavior_head(rng: random.Random) -> dict[str, object]:
    """Transient classifier used to pretrain the shared trunk only."""

    scale = math.sqrt(2.0 / HIDDEN_SIZES[1])
    return {
        "w": [
            [rng.uniform(-scale, scale) for _ in range(HIDDEN_SIZES[1])] for _ in LABELS
        ],
        "b": [0.0] * len(LABELS),
    }


def _softmax(logits: Sequence[float]) -> list[float]:
    offset = max(logits)
    values = [math.exp(value - offset) for value in logits]
    total = sum(values)
    return [value / total for value in values]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _expected_log_return(playability: float, return_scale_fraction: float) -> float:
    """Invert the value head's sigmoid encoding into signed log purse return."""

    bounded = min(1.0 - 1e-6, max(1e-6, playability))
    return math.log1p(return_scale_fraction) * math.log(bounded / (1.0 - bounded))


def _dot(row: Sequence[float], values: Sequence[float]) -> float:
    # sum(map(mul, ...)) measured 2.09x the generator form with bit-identical
    # accumulation order; this is the live serve path's inner loop.
    return sum(map(operator.mul, row, values))


def _forward(
    weights: dict[str, object], features: Sequence[float]
) -> dict[str, object]:
    w1 = weights["w1"]
    b1 = weights["b1"]
    w2 = weights["w2"]
    b2 = weights["b2"]
    action_w = weights["action_w"]
    action_b = weights["action_b"]
    assert isinstance(w1, list) and isinstance(b1, list)
    assert isinstance(w2, list) and isinstance(b2, list)
    assert isinstance(action_w, list) and isinstance(action_b, list)
    h1_pre = [_dot(row, features) + bias for row, bias in zip(w1, b1)]
    h1 = [value if value > 0.0 else 0.0 for value in h1_pre]
    h2_pre = [_dot(row, h1) + bias for row, bias in zip(w2, b2)]
    h2 = [value if value > 0.0 else 0.0 for value in h2_pre]
    action_logits = [_dot(row, h2) + bias for row, bias in zip(action_w, action_b)]
    playability = _sigmoid(
        _dot(weights["playability_w"], h2) + weights["playability_b"]
    )  # type: ignore[arg-type,operator]
    risk_fraction = _sigmoid(
        _dot(weights["risk_fraction_w"], h2) + weights["risk_fraction_b"]
    )  # type: ignore[arg-type,operator]
    return {
        "h1_pre": h1_pre,
        "h1": h1,
        "h2_pre": h2_pre,
        "h2": h2,
        "action_logits": action_logits,
        "action_probabilities": _softmax(action_logits),
        "playability": playability,
        "risk_fraction": risk_fraction,
    }


def _purse_return_fraction(example: TrainingExample) -> float:
    return example.reward_bb / example.purse_bb


def _log_purse_return(example: TrainingExample) -> float:
    fraction = _purse_return_fraction(example)
    return math.copysign(math.log1p(abs(fraction)), fraction)


def _fraction_from_log_return(value: float) -> float:
    return math.copysign(math.expm1(abs(value)), value)


def _targets(example: TrainingExample, config: TrainingConfig) -> tuple[float, float]:
    playability = _sigmoid(
        _log_purse_return(example) / math.log1p(config.return_scale_fraction)
    )
    return playability, example.submitted_risk_fraction


def _policy_signal(
    example: TrainingExample, config: TrainingConfig
) -> tuple[float, float]:
    """Return signed log-percentage advantage and its policy multiplier."""

    raw_log_return = _log_purse_return(example)
    scaled = min(
        1.0,
        max(-1.0, raw_log_return / math.log1p(config.return_scale_fraction)),
    )
    scaled *= _evidence_weight(example)
    if scaled > 0.0:
        scaled *= config.reinforcement_multiplier
    return raw_log_return, scaled


def _action_value_target(example: TrainingExample, config: TrainingConfig) -> float:
    """Bounded signed-log advantage for one legal action."""

    return min(
        1.0,
        max(
            -1.0,
            _log_purse_return(example) / math.log1p(config.return_scale_fraction),
        ),
    )


def _action_value_weight(example: TrainingExample, config: TrainingConfig) -> float:
    """Confidence and owner-required positive reinforcement affect weight only."""

    weight = _class_weight(example, config) * _evidence_weight(example)
    if _action_value_target(example, config) > 0.0:
        weight *= config.reinforcement_multiplier
    return weight


def _evidence_weight(example: TrainingExample) -> float:
    """Discount uncertain opponent reads without discarding generic lessons."""

    confidence = min(1.0, max(0.0, example.opponent_confidence))
    return 0.25 + 0.75 * confidence


def _risk_weight(example: TrainingExample, scaled_signal: float) -> float:
    """Only profitable aggressive decisions are valid sizing teachers."""

    return max(0.0, scaled_signal) if example.action_family_index == 2 else 0.0


def _policy_loss(
    probabilities: Sequence[float], chosen: int, advantage: float, weight: float
) -> float:
    probability = min(1.0 - 1e-12, max(1e-12, probabilities[chosen]))
    if advantage >= 0.0:
        return -weight * advantage * math.log(probability)
    return -weight * -advantage * math.log1p(-probability)


def _policy_gradient(
    probabilities: Sequence[float], chosen: int, advantage: float, weight: float
) -> list[float]:
    if advantage >= 0.0:
        gradient = list(probabilities)
        gradient[chosen] -= 1.0
        return [delta * weight * advantage for delta in gradient]

    # Penalize a losing action with -log(1 - p). Unlike negative cross-entropy,
    # this has a finite minimum and its gradient fades as p approaches zero.
    magnitude = weight * -advantage
    chosen_probability = probabilities[chosen]
    remaining_probability = max(1e-12, 1.0 - chosen_probability)
    return [
        magnitude * chosen_probability
        if index == chosen
        else -magnitude * chosen_probability * probability / remaining_probability
        for index, probability in enumerate(probabilities)
    ]


def _updated(value: float, gradient: float, config: TrainingConfig) -> float:
    if not math.isfinite(gradient):
        raise FloatingPointError("non-finite gradient during training")
    gradient = min(config.gradient_clip, max(-config.gradient_clip, gradient))
    result = value - config.learning_rate * gradient
    if not math.isfinite(result):
        raise FloatingPointError("non-finite parameter update during training")
    return result


def _assert_finite_weights(weights: dict[str, object], stage: str) -> None:
    def visit(value: object) -> bool:
        if isinstance(value, list):
            return all(visit(item) for item in value)
        if isinstance(value, dict):
            # Format-2 weights nest encoder/trunk/head blocks as objects.
            return all(visit(item) for item in value.values())
        return isinstance(value, (int, float)) and math.isfinite(value)

    if not visit(list(weights.values())):
        raise FloatingPointError(f"non-finite model parameter after {stage}")


def _decision_key(example: TrainingExample) -> object:
    return example.decision_id or (example.table_id, example.features)


def _action_metrics(
    examples: Sequence[TrainingExample],
    score_rows: Sequence[Sequence[float]],
    config: TrainingConfig,
) -> tuple[float, float, float]:
    """Best-family accuracy, regret, and value error over decision groups."""

    grouped: dict[object, list[tuple[TrainingExample, Sequence[float]]]] = {}
    for example, scores in zip(examples, score_rows):
        grouped.setdefault(_decision_key(example), []).append((example, scores))
    correct = 0
    total_regret = 0.0
    total_value_error = 0.0
    value_count = 0
    for rows in grouped.values():
        action_returns = {
            example.action_family_index: _purse_return_fraction(example)
            for example, _ in rows
        }
        scores = rows[0][1]
        predicted = max(action_returns, key=lambda action: scores[action])
        best_return = max(action_returns.values())
        correct += int(
            math.isclose(action_returns[predicted], best_return, abs_tol=1e-12)
        )
        total_regret += 100.0 * (best_return - action_returns[predicted])
        for example, row_scores in rows:
            predicted_scaled = min(
                1.0,
                max(-1.0, float(row_scores[example.action_family_index])),
            )
            predicted_log = predicted_scaled * math.log1p(config.return_scale_fraction)
            predicted_return = _fraction_from_log_return(predicted_log)
            total_value_error += 100.0 * abs(
                _purse_return_fraction(example) - predicted_return
            )
            value_count += 1
    group_count = max(1, len(grouped))
    return (
        correct / group_count,
        total_regret / group_count,
        total_value_error / max(1, value_count),
    )


def _hybrid_calibration(
    examples: Sequence[TrainingExample],
    score_rows: Sequence[Sequence[float]],
) -> dict[str, object] | None:
    """Measure held-out regret when value margins gate teacher overrides."""

    grouped: dict[object, list[tuple[TrainingExample, Sequence[float]]]] = {}
    for example, scores in zip(examples, score_rows):
        grouped.setdefault(_decision_key(example), []).append((example, scores))
    decisions = []
    for rows in grouped.values():
        action_returns = {
            example.action_family_index: _purse_return_fraction(example)
            for example, _ in rows
        }
        scores = rows[0][1]
        behavior = max(
            range(len(LABELS)),
            key=lambda action: rows[0][0].behavior_probabilities[action],
        )
        if behavior not in action_returns:
            continue
        predicted = max(action_returns, key=lambda action: scores[action])
        decisions.append(
            (
                action_returns,
                behavior,
                predicted,
                float(scores[predicted]) - float(scores[behavior]),
            )
        )
    if not decisions:
        return None

    behavior_regret = (
        100.0
        * sum(
            max(returns.values()) - returns[behavior]
            for returns, behavior, _, _ in decisions
        )
        / len(decisions)
    )
    curves = []
    for threshold in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        regret = 0.0
        overrides = 0
        for returns, behavior, predicted, margin in decisions:
            chosen = behavior
            if predicted != behavior and margin >= threshold:
                chosen = predicted
                overrides += 1
            regret += max(returns.values()) - returns[chosen]
        mean_regret = 100.0 * regret / len(decisions)
        curves.append(
            {
                "min_value_advantage": threshold,
                "override_rate": overrides / len(decisions),
                "mean_regret_pct": mean_regret,
                "regret_delta_vs_behavior_pct": mean_regret - behavior_regret,
            }
        )
    helpful = [
        row
        for row in curves
        if row["override_rate"] > 0.0 and row["mean_regret_pct"] < behavior_regret
    ]
    recommended = (
        min(
            helpful,
            key=lambda row: (row["mean_regret_pct"], row["min_value_advantage"]),
        )["min_value_advantage"]
        if helpful
        else None
    )
    return {
        "decision_groups": len(decisions),
        "behavior_mean_regret_pct": behavior_regret,
        "recommended_min_value_advantage": recommended,
        "curve": curves,
    }


def _loss(
    weights: dict[str, object],
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> tuple[float, float, float, float]:
    if not examples:
        return 0.0, 0.0, 0.0, 0.0
    total_loss = 0.0
    score_rows: list[Sequence[float]] = []
    for example in examples:
        output = _forward(weights, example.features)
        scores = output["action_logits"]
        assert isinstance(scores, list)
        score_rows.append(scores)
        chosen = example.action_family_index
        target = _action_value_target(example, config)
        total_loss += (
            _action_value_weight(example, config) * (scores[chosen] - target) ** 2
        )
        if config.train_risk_head:
            risk_fraction = float(output["risk_fraction"])
            _, scaled_signal = _policy_signal(example, config)
            total_loss += (
                _risk_weight(example, scaled_signal)
                * (risk_fraction - example.submitted_risk_fraction) ** 2
            )
    accuracy, regret, value_error = _action_metrics(examples, score_rows, config)
    return total_loss / len(examples), accuracy, regret, value_error


def _step(
    weights: dict[str, object],
    example: TrainingExample,
    config: TrainingConfig,
    *,
    policy_objective: str = "reward",
    behavior_head: dict[str, object] | None = None,
) -> None:
    if policy_objective not in {"none", "imitation", "reward"}:
        raise ValueError("unsupported policy objective")
    output = _forward(weights, example.features)
    action_values = output["action_logits"]
    h1_pre = output["h1_pre"]
    h1 = output["h1"]
    h2_pre = output["h2_pre"]
    h2 = output["h2"]
    assert isinstance(action_values, list)
    assert isinstance(h1_pre, list) and isinstance(h1, list)
    assert isinstance(h2_pre, list) and isinstance(h2, list)

    risk_fraction = float(output["risk_fraction"])
    _, scaled_signal = _policy_signal(example, config)
    action_w = weights["action_w"]
    action_b = weights["action_b"]
    assert isinstance(action_w, list) and isinstance(action_b, list)
    output_w = action_w
    output_b = action_b
    probabilities = output["action_probabilities"]
    assert isinstance(probabilities, list)
    if policy_objective == "imitation":
        if behavior_head is None:
            raise ValueError("imitation requires a transient behavior head")
        output_w = behavior_head["w"]
        output_b = behavior_head["b"]
        assert isinstance(output_w, list) and isinstance(output_b, list)
        probabilities = _softmax(
            [_dot(row, h2) + bias for row, bias in zip(output_w, output_b)]
        )
    d_action = [0.0] * len(probabilities)
    if policy_objective == "imitation":
        d_action = _policy_gradient(
            probabilities,
            example.action_family_index,
            1.0,
            _class_weight(example, config),
        )
    elif policy_objective == "reward":
        chosen = example.action_family_index
        d_action[chosen] = (
            2.0
            * _action_value_weight(example, config)
            * (action_values[chosen] - _action_value_target(example, config))
        )
    d_playability = 0.0
    d_risk = 0.0
    if policy_objective == "reward" and config.train_risk_head:
        d_risk = (
            2.0
            * _risk_weight(example, scaled_signal)
            * (risk_fraction - example.submitted_risk_fraction)
            * risk_fraction
            * (1.0 - risk_fraction)
        )

    playability_w = weights["playability_w"]
    risk_w = weights["risk_fraction_w"]
    assert isinstance(playability_w, list) and isinstance(risk_w, list)

    d_h2 = [0.0] * HIDDEN_SIZES[1]
    for row_index, delta in enumerate(d_action):
        for col_index, weight in enumerate(output_w[row_index]):
            d_h2[col_index] += delta * weight
    for index, weight in enumerate(playability_w):
        d_h2[index] += d_playability * weight
    for index, weight in enumerate(risk_w):
        d_h2[index] += d_risk * weight

    for row_index, delta in enumerate(d_action):
        for col_index, value in enumerate(h2):
            output_w[row_index][col_index] = _updated(
                output_w[row_index][col_index], delta * value, config
            )
        output_b[row_index] = _updated(output_b[row_index], delta, config)
    for index, value in enumerate(h2):
        playability_w[index] = _updated(
            playability_w[index], d_playability * value, config
        )
        risk_w[index] = _updated(risk_w[index], d_risk * value, config)
    weights["playability_b"] = _updated(
        float(weights["playability_b"]), d_playability, config
    )
    weights["risk_fraction_b"] = _updated(
        float(weights["risk_fraction_b"]), d_risk, config
    )

    d_h2_pre = [delta if pre > 0.0 else 0.0 for delta, pre in zip(d_h2, h2_pre)]
    w2 = weights["w2"]
    b2 = weights["b2"]
    assert isinstance(w2, list) and isinstance(b2, list)
    d_h1 = [0.0] * HIDDEN_SIZES[0]
    for row_index, delta in enumerate(d_h2_pre):
        for col_index, weight in enumerate(w2[row_index]):
            d_h1[col_index] += delta * weight
            w2[row_index][col_index] = _updated(
                w2[row_index][col_index], delta * h1[col_index], config
            )
        b2[row_index] = _updated(b2[row_index], delta, config)

    d_h1_pre = [delta if pre > 0.0 else 0.0 for delta, pre in zip(d_h1, h1_pre)]
    w1 = weights["w1"]
    b1 = weights["b1"]
    assert isinstance(w1, list) and isinstance(b1, list)
    for row_index, delta in enumerate(d_h1_pre):
        for col_index, value in enumerate(example.features):
            w1[row_index][col_index] = _updated(
                w1[row_index][col_index], delta * value, config
            )
        b1[row_index] = _updated(b1[row_index], delta, config)


def _split(
    examples: Sequence[TrainingExample], config: TrainingConfig
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    by_hand: dict[str, list[TrainingExample]] = {}
    for example in examples:
        by_hand.setdefault(example.table_id, []).append(example)
    hand_ids = list(by_hand)
    random.Random(config.resolved_split_seed).shuffle(hand_ids)
    validation_count = int(round(len(hand_ids) * config.validation_fraction))
    if validation_count >= len(hand_ids):
        validation_count = max(0, len(hand_ids) - 1)
    validation_ids = set(hand_ids[:validation_count])
    train = [example for example in examples if example.table_id not in validation_ids]
    validation = [example for example in examples if example.table_id in validation_ids]
    return train, validation


def _train_cuda(
    train_behavior_examples: Sequence[TrainingExample],
    train_reward_examples: Sequence[TrainingExample],
    validation_reward_examples: Sequence[TrainingExample],
    config: TrainingConfig,
) -> tuple[
    dict[str, object],
    tuple[float, float, float, float],
    tuple[float, float, float, float] | None,
    list[list[float]],
    str,
]:
    """Fit the existing artifact shape in vectorized CUDA batches."""

    import torch
    from torch import nn

    device_name = validate_training_device("cuda")
    device = torch.device("cuda")
    torch.manual_seed(config.resolved_init_seed)
    torch.cuda.manual_seed_all(config.resolved_init_seed)

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = nn.Linear(LEARNING_INPUT_SIZE, HIDDEN_SIZES[0])
            self.w2 = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])
            self.action = nn.Linear(HIDDEN_SIZES[1], len(LABELS))
            self.behavior = nn.Linear(HIDDEN_SIZES[1], len(LABELS))
            self.playability = nn.Linear(HIDDEN_SIZES[1], 1)
            self.risk = nn.Linear(HIDDEN_SIZES[1], 1)

        def forward(self, features: object) -> tuple[object, object, object, object]:
            hidden = torch.relu(self.w1(features))
            hidden = torch.relu(self.w2(hidden))
            return (
                self.action(hidden),
                self.behavior(hidden),
                torch.sigmoid(self.playability(hidden).squeeze(1)),
                torch.sigmoid(self.risk(hidden).squeeze(1)),
            )

    model = Network()
    initial = _init_weights(random.Random(config.resolved_init_seed))
    with torch.no_grad():
        model.w1.weight.copy_(torch.tensor(initial["w1"], dtype=torch.float32))
        model.w1.bias.copy_(torch.tensor(initial["b1"], dtype=torch.float32))
        model.w2.weight.copy_(torch.tensor(initial["w2"], dtype=torch.float32))
        model.w2.bias.copy_(torch.tensor(initial["b2"], dtype=torch.float32))
        model.action.weight.copy_(
            torch.tensor(initial["action_w"], dtype=torch.float32)
        )
        model.action.bias.copy_(torch.tensor(initial["action_b"], dtype=torch.float32))
        model.playability.weight.copy_(
            torch.tensor([initial["playability_w"]], dtype=torch.float32)
        )
        model.playability.bias.fill_(float(initial["playability_b"]))
        model.risk.weight.copy_(
            torch.tensor([initial["risk_fraction_w"]], dtype=torch.float32)
        )
        model.risk.bias.fill_(float(initial["risk_fraction_b"]))
    model.to(device)

    def dataset(examples: Sequence[TrainingExample]) -> dict[str, object]:
        return {
            "features": torch.tensor(
                [example.features for example in examples],
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.tensor(
                [example.action_family_index for example in examples],
                dtype=torch.long,
                device=device,
            ),
            "risks": torch.tensor(
                [example.submitted_risk_fraction for example in examples],
                dtype=torch.float32,
                device=device,
            ),
            "signals": torch.tensor(
                [_policy_signal(example, config)[1] for example in examples],
                dtype=torch.float32,
                device=device,
            ),
            "action_targets": torch.tensor(
                [_action_value_target(example, config) for example in examples],
                dtype=torch.float32,
                device=device,
            ),
            "action_weights": torch.tensor(
                [_action_value_weight(example, config) for example in examples],
                dtype=torch.float32,
                device=device,
            ),
            "class_weights": torch.tensor(
                [_class_weight(example, config) for example in examples],
                dtype=torch.float32,
                device=device,
            ),
        }

    train_behavior_data = dataset(train_behavior_examples)
    train_reward_data = dataset(train_reward_examples)
    validation_reward_data = (
        dataset(validation_reward_examples) if validation_reward_examples else None
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    def losses(data: dict[str, object], indexes: object, objective: str) -> object:
        features = data["features"][indexes]
        actions = data["actions"][indexes]
        targets = data["action_targets"][indexes]
        action_weights = data["action_weights"][indexes]
        signals = data["signals"][indexes]
        class_weights = data["class_weights"][indexes]
        action_values, behavior_logits, _, risk = model(features)
        log_probabilities = torch.log_softmax(behavior_logits, dim=1)
        rows = torch.arange(actions.shape[0], device=device)
        chosen_log_probability = log_probabilities[rows, actions]
        chosen_value = action_values[rows, actions]
        value_loss = torch.zeros_like(chosen_value)
        if objective == "none":
            policy_loss = torch.zeros_like(value_loss)
            risk_loss = torch.zeros_like(value_loss)
        elif objective == "imitation":
            policy_loss = -class_weights * chosen_log_probability
            value_loss = torch.zeros_like(value_loss)
            risk_loss = torch.zeros_like(value_loss)
        else:
            policy_loss = action_weights * (chosen_value - targets).square()
            risk_weights = (
                torch.where(actions == 2, torch.clamp_min(signals, 0.0), 0.0)
                if config.train_risk_head
                else torch.zeros_like(signals)
            )
            risk_loss = risk_weights * (risk - data["risks"][indexes]).square()
        return policy_loss, value_loss, risk_loss, action_values

    def epoch(
        data: dict[str, object],
        examples: Sequence[TrainingExample],
        objective: str,
    ) -> None:
        model.train()
        count = len(examples)
        if count == 0:
            return
        order = torch.randperm(count, device=device)
        for start in range(0, count, config.batch_size):
            indexes = order[start : start + config.batch_size]
            policy_loss, value_loss, risk_loss, _ = losses(data, indexes, objective)
            optimizer.zero_grad(set_to_none=True)
            (policy_loss + value_loss + risk_loss).mean().backward()
            for parameter in model.parameters():
                if parameter.grad is None:
                    continue
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("non-finite CUDA gradient during training")
                parameter.grad.clamp_(-config.gradient_clip, config.gradient_clip)
            optimizer.step()
            if not all(torch.isfinite(value).all() for value in model.parameters()):
                raise FloatingPointError("non-finite CUDA parameter during training")

    for _ in range(config.behavior_warmup_epochs):
        epoch(train_behavior_data, train_behavior_examples, "imitation")
    for _ in range(config.epochs):
        epoch(train_reward_data, train_reward_examples, "reward")

    def metrics(
        examples: Sequence[TrainingExample], data: dict[str, object]
    ) -> tuple[tuple[float, float, float, float], list[list[float]]]:
        model.eval()
        total_loss = 0.0
        total_weight = 0.0
        score_rows: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(examples), config.batch_size):
                indexes = torch.arange(
                    start,
                    min(start + config.batch_size, len(examples)),
                    device=device,
                )
                policy_loss, value_loss, risk_loss, action_values = losses(
                    data, indexes, "reward"
                )
                total_loss += float((policy_loss + value_loss + risk_loss).sum())
                score_rows.extend(action_values.detach().cpu().tolist())
                # Normalize by the weight mass, not the row count: weights span
                # 0.25-1.5, so a per-row mean shifts with the split's weight mix
                # at byte-identical predictions and cannot be compared across
                # configurations.
                total_weight += float(data["action_weights"][indexes].sum())
        accuracy, regret, value_error = _action_metrics(examples, score_rows, config)
        return (
            total_loss / max(1e-9, total_weight),
            accuracy,
            regret,
            value_error,
        ), score_rows

    train_metrics, _ = metrics(train_reward_examples, train_reward_data)
    validation_result = (
        metrics(validation_reward_examples, validation_reward_data)
        if validation_reward_data is not None
        else None
    )
    validation_metrics = validation_result[0] if validation_result else None
    validation_score_rows = validation_result[1] if validation_result else []
    weights = {
        "w1": model.w1.weight.detach().cpu().tolist(),
        "b1": model.w1.bias.detach().cpu().tolist(),
        "w2": model.w2.weight.detach().cpu().tolist(),
        "b2": model.w2.bias.detach().cpu().tolist(),
        "action_w": model.action.weight.detach().cpu().tolist(),
        "action_b": model.action.bias.detach().cpu().tolist(),
        "playability_w": model.playability.weight.detach().cpu().tolist()[0],
        "playability_b": float(model.playability.bias.detach().cpu().item()),
        "risk_fraction_w": model.risk.weight.detach().cpu().tolist()[0],
        "risk_fraction_b": float(model.risk.bias.detach().cpu().item()),
    }
    _assert_finite_weights(weights, "CUDA training")
    return (
        weights,
        train_metrics,
        validation_metrics,
        validation_score_rows,
        device_name,
    )


# ---------------------------------------------------------------------------
# Format 2 ("v7"): two-branch encoder, shared trunk, size-conditioned values
# ---------------------------------------------------------------------------


def _round9(value: object) -> object:
    """Recursively shorten floats to 9 significant digits for export.

    Nine digits is below the float32 quantum the trained values carry, so
    the loaded network is bit-identical while the JSON artifact shrinks by
    roughly a third (measured 20.94 -> ~13 bytes per parameter).
    """

    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, list):
        return [_round9(item) for item in value]
    if isinstance(value, dict):
        return {key: _round9(item) for key, item in value.items()}
    return value


def _branch_index(example: TrainingExample) -> int:
    """Resolve a counterfactual row's value branch under format 2."""

    branch = getattr(example, "action_branch", None)
    if branch is not None:
        return BRANCH_LABELS.index(branch)
    # Legacy rows carry only the family; map fold/check_call directly and
    # treat un-sized aggression as the half-pot branch, which matches the
    # deployed policy's ~half-pot sizing.
    family_to_branch = {0: 0, 1: 1, 2: 2}
    return family_to_branch[example.action_family_index]


def _intent_branch(
    label: str, absorption: Mapping[str, str], returns: Mapping[int, float]
) -> int:
    """Resolve a trivial policy's intended branch to one the group emitted."""

    survivor = absorption.get(label, label)
    index = BRANCH_LABELS.index(survivor)
    if index in returns:
        return index
    # Legacy corpus with no absorption map, or an intent whose branch was
    # never a candidate here: fall back to the lowest emitted branch.
    return min(returns)


def _absorbing_family(
    example: TrainingExample, by_family: Mapping[int, Sequence[float]]
) -> dict[int, int]:
    """Map each family onto the family whose branch carries its value.

    A branch dropped for naming an action that another branch already
    executes still has a value -- the survivor's. A family stands for itself
    whenever it kept any branch; otherwise it follows the absorption map
    recorded at harvest. Legacy corpora carry no map, and there every family
    that is present stands for itself, which is the pre-Option-A behaviour.
    """

    absorption = dict(example.branch_absorption or ())
    mapping: dict[int, int] = {}
    for family_index, family_name in enumerate(LABELS):
        if family_index in by_family:
            mapping[family_index] = family_index
            continue
        for branch_index, label in enumerate(BRANCH_LABELS):
            if BRANCH_FAMILIES[branch_index] != family_name:
                continue
            survivor = absorption.get(label)
            if survivor is None:
                continue
            target = LABELS.index(BRANCH_FAMILIES[BRANCH_LABELS.index(survivor)])
            if target in by_family:
                mapping[family_index] = target
                break
    return mapping


def _branch_class_weight(example: TrainingExample, config: TrainingConfig) -> float:
    if config.class_weights is None:
        return 1.0
    return config.class_weights[example.action_family_index]


def _branch_value_weight(example: TrainingExample, config: TrainingConfig) -> float:
    """Row weight for the centered branch objective.

    The reinforcement multiplier stays expressible (config >= 1.0) but the
    v7 recipe passes 1.0: the 2026-08-14 ablation measured the 1.5x tilt
    costing regret (off 13.42 vs per-row 13.57).
    """

    weight = _branch_class_weight(example, config) * _evidence_weight(example)
    if (
        config.reinforcement_multiplier > 1.0
        and _action_value_target(example, config) > 0.0
    ):
        weight *= config.reinforcement_multiplier
    return weight


def _layer_norm(
    values: list[float], gamma: Sequence[float], beta: Sequence[float]
) -> list[float]:
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    scale = 1.0 / math.sqrt(variance + 1e-5)
    return [(value - mean) * scale * g + b for value, g, b in zip(values, gamma, beta)]


def _forward_v2(
    architecture: Mapping[str, object],
    weights: Mapping[str, object],
    features: Sequence[float],
    *,
    heads: Sequence[str] = ("action_value",),
) -> dict[str, list[float]]:
    """Pure-Python inference for a format-2 artifact (evaluation mode).

    Must stay numerically aligned with ``_NetworkV7.forward`` in eval mode:
    linear -> ReLU -> LayerNorm on the encoders and every trunk layer except
    the last, which is linear -> ReLU; head towers are linear -> ReLU ->
    linear. Dropout is train-only and does not appear here.
    """

    def linear(block: Mapping[str, object], values: Sequence[float]) -> list[float]:
        return [
            _dot(row, values) + bias
            for row, bias in zip(block["w"], block["b"])  # type: ignore[index]
        ]

    def encode(name: str, indices: Sequence[int]) -> list[float]:
        block = weights[name]
        taken = [features[index] for index in indices]
        hidden = [value if value > 0.0 else 0.0 for value in linear(block, taken)]
        return _layer_norm(hidden, block["ln_g"], block["ln_b"])  # type: ignore[index]

    card = encode("card_encoder", architecture["card_indices"])  # type: ignore[arg-type]
    context = encode("context_encoder", architecture["context_indices"])  # type: ignore[arg-type]
    hidden = [*card, *context]
    trunk_blocks = weights["trunk"]
    assert isinstance(trunk_blocks, list)
    for position, block in enumerate(trunk_blocks):
        hidden = [value if value > 0.0 else 0.0 for value in linear(block, hidden)]
        if position < len(trunk_blocks) - 1:
            hidden = _layer_norm(hidden, block["ln_g"], block["ln_b"])
    outputs: dict[str, list[float]] = {}
    head_blocks = weights["heads"]
    assert isinstance(head_blocks, dict)
    for name in heads:
        block = head_blocks[name]
        tower = [
            value if value > 0.0 else 0.0
            for value in linear({"w": block["tower_w"], "b": block["tower_b"]}, hidden)
        ]
        outputs[name] = linear({"w": block["out_w"], "b": block["out_b"]}, tower)
    return outputs


@dataclass(frozen=True, slots=True)
class BranchMetricsV7:
    """Branch metrics split by whether the group can discriminate at all.

    A *degenerate* group is one whose every branch ties for best. On such a
    group best-action accuracy and regret are constants -- 1 and 0 -- for
    every possible predictor, including the trivial baselines, so including
    them adds a fixed bonus to each contender and compresses the differences
    the metric exists to resolve. The primary figures therefore score only
    the groups where the metric is defined; ``all_groups`` keeps the old
    tie-credited numbers visible, and the degenerate counts travel with
    every figure so the inflation can never go unnoticed again.
    """

    total_groups: int
    degenerate_groups: int
    scored_groups: int
    best_action_accuracy: float | None
    mean_regret_pct: float | None
    chance_accuracy: float | None
    best_action_accuracy_all_groups: float
    mean_regret_pct_all_groups: float
    action_value_mae_pct: float
    # Groups that emitted a single branch. Also unscorable, but for a
    # different reason -- no choice existed rather than every choice tying --
    # and they are counted apart because the degeneracy rate is the headline
    # diagnostic for the branch-set work, and mixing "nothing to choose" into
    # "everything ties" would make that measurement unreadable.
    single_branch_groups: int = 0

    @property
    def degenerate_group_fraction(self) -> float:
        return self.degenerate_groups / max(1, self.total_groups)

    def to_mapping(self) -> dict[str, object]:
        return {
            "scoring": (
                "primary figures exclude degenerate groups (every branch ties "
                "for best), where accuracy and regret are 1 and 0 for every "
                "predictor; all_groups repeats them with ties credited"
            ),
            "total_groups": self.total_groups,
            "degenerate_groups": self.degenerate_groups,
            "single_branch_groups": self.single_branch_groups,
            "scored_groups": self.scored_groups,
            "degenerate_group_fraction": round(self.degenerate_group_fraction, 6),
            "best_action_accuracy": _round_or_none(self.best_action_accuracy),
            "mean_regret_pct": _round_or_none(self.mean_regret_pct),
            "chance_accuracy": _round_or_none(self.chance_accuracy),
            "action_value_mae_pct": round(self.action_value_mae_pct, 6),
            "all_groups": {
                "best_action_accuracy": round(self.best_action_accuracy_all_groups, 6),
                "mean_regret_pct": round(self.mean_regret_pct_all_groups, 6),
            },
        }


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def print_branch_summary(summary: "TrainingSummary") -> None:
    """Print the validation branch metrics with their degenerate context.

    The degenerate share always prints beside accuracy and regret: those two
    figures are computed on the complementary subset, and quoting either
    without it is how the inflated 60.28% headline survived review.
    """

    def show(name: str, value: float | None) -> None:
        print(f"{name}: {'n/a' if value is None else format(value, '.6f')}")

    fraction = summary.validation_degenerate_group_fraction
    if fraction is not None:
        print(f"validation_degenerate_group_fraction: {fraction:.6f}")
        print(
            "  (accuracy and regret below exclude those groups; every branch "
            "ties there, so no predictor can be wrong)"
        )
    show("validation_best_action_accuracy", summary.validation_best_action_accuracy)
    show("validation_mean_regret_pct", summary.validation_mean_regret_pct)
    show("validation_action_value_mae_pct", summary.validation_action_value_mae_pct)


def _optimal_branches(returns: Mapping[int, float]) -> set[int]:
    """Branches tied for best, using the metric's own equality tolerance."""

    best = max(returns.values())
    return {
        branch
        for branch, value in returns.items()
        if math.isclose(value, best, abs_tol=1e-12)
    }


def _zero_signal_group(
    examples: Sequence[TrainingExample],
    rows: Sequence[int],
    config: TrainingConfig,
) -> bool:
    """True when a CONSTANT action head exactly minimises this group's loss.

    The v7 action objective centers each prediction inside its decision
    group::

        centered = predicted - mean(predicted over the group)
        loss     = sum_i w_i (centered_i - t_i)^2 / sum_i w_i

    ``centered`` sums to zero by construction, so the derivative with respect
    to ``predicted_i`` at a constant head is ``-2 (w_i t_i - mean_j(w_j
    t_j))``. It vanishes for every row exactly when ``w_i * t_i`` is the same
    on every row, and the per-group loss is convex, so that condition is
    equivalent to "a constant head is this group's global minimiser". Such a
    group supervises nothing, and away from the constant it pulls the head
    back toward one.

    Three consequences, each of which the obvious definition -- "every target
    is zero" -- gets wrong:

    * **Identical NON-zero targets qualify too**, whenever the row weights are
      equal. The candidate-v7-0001 corpus happens to contain none, but a group
      whose every branch clips to the same bound would be one, and
      ``return_scale_fraction`` clips at +/-1.
    * **A single-branch group always qualifies.** With one row ``centered`` is
      identically zero, so no prediction can move its loss at all; it is a
      permanent additive constant in the numerator and a permanent term in the
      denominator. The harvester can emit 2-4 branches per group
      (``.handoff/PENDING_EDITS.md`` 18h), so this is reachable.
    * **A weight-zero row does not qualify its group on its own.** It still
      shifts the group mean, so it still changes what the other rows are
      centered against. Its product ``w_i * t_i`` is zero and has to match
      every other row's product like any other row's does.

    Equality is exact and deliberately untoleranced: a threshold would
    reclassify a group that carries a little signal as one that carries none,
    at an arbitrary cut. Measured on candidate-v7-0001, no group has a
    weighted-target spread anywhere in ``(0, 1e-6)`` -- 14,842 sit at exactly
    zero and the next 26 at 1e-6 or above -- so the cut is inert there and the
    exact rule is the honest one.

    Related but NOT the same as :attr:`BranchMetricsV7.degenerate_groups`,
    which asks whether every branch ties for *best* on the raw purse return.
    That is a metrics question (can any predictor be wrong here?); this is an
    optimisation question (is a constant the minimiser here?). They coincide
    on candidate-v7-0001 and need not in general.
    """

    products = [
        _branch_value_weight(examples[row], config)
        * _action_value_target(examples[row], config)
        for row in rows
    ]
    return max(products) == min(products)


def _branch_metrics_v7(
    examples: Sequence[TrainingExample],
    score_rows: Sequence[Sequence[float]],
) -> BranchMetricsV7:
    """Best-branch accuracy, mean regret pct, and value MAE over groups.

    Degenerate groups -- every branch tied for best -- are excluded from the
    headline accuracy and regret because no predictor can be wrong on them;
    see :class:`BranchMetricsV7`.
    """

    grouped: dict[object, list[tuple[TrainingExample, Sequence[float]]]] = {}
    for example, scores in zip(examples, score_rows):
        grouped.setdefault(_decision_key(example), []).append((example, scores))
    correct = 0
    total_regret = 0.0
    correct_all = 0
    total_regret_all = 0.0
    chance = 0.0
    degenerate = 0
    single_branch = 0
    total_value_error = 0.0
    value_count = 0
    for rows in grouped.values():
        returns = {
            _branch_index(example): _purse_return_fraction(example)
            for example, _ in rows
        }
        scores = rows[0][1]
        predicted = max(returns, key=lambda branch: scores[branch])
        best_return = max(returns.values())
        optimal = _optimal_branches(returns)
        hit = int(predicted in optimal)
        regret = 100.0 * (best_return - returns[predicted])
        correct_all += hit
        total_regret_all += regret
        if len(returns) < 2:
            single_branch += 1
        elif len(optimal) == len(returns):
            degenerate += 1
        else:
            correct += hit
            total_regret += regret
            chance += len(optimal) / len(returns)
        centered = {
            branch: scores[branch] - sum(scores[b] for b in returns) / len(returns)
            for branch in returns
        }
        target_mean = sum(returns.values()) / len(returns)
        for example, _ in rows:
            branch = _branch_index(example)
            total_value_error += 100.0 * abs(
                (returns[branch] - target_mean) - centered[branch]
            )
            value_count += 1
    total_groups = len(grouped)
    scored = total_groups - degenerate - single_branch
    return BranchMetricsV7(
        total_groups=total_groups,
        degenerate_groups=degenerate,
        single_branch_groups=single_branch,
        scored_groups=scored,
        best_action_accuracy=None if scored == 0 else correct / scored,
        mean_regret_pct=None if scored == 0 else total_regret / scored,
        chance_accuracy=None if scored == 0 else chance / scored,
        best_action_accuracy_all_groups=correct_all / max(1, total_groups),
        mean_regret_pct_all_groups=total_regret_all / max(1, total_groups),
        action_value_mae_pct=total_value_error / max(1, value_count),
    )


def _baseline_floor_v7(
    examples: Sequence[TrainingExample],
    seed: int,
) -> dict[str, object]:
    """Trivial predictors every candidate must beat before a gauntlet runs.

    Candidate 0016 lost to `always aggress` on both offline metrics without
    anyone noticing; this table makes that failure blocking and visible.
    Scored on the same footing as :func:`_branch_metrics_v7`: degenerate
    groups are excluded from the headline figures, because they hand every
    contender the same free point and shrink the gap the floor must resolve.
    """

    grouped: dict[object, dict[int, float]] = {}
    absorption_by_group: dict[object, dict[str, str]] = {}
    for example in examples:
        key = _decision_key(example)
        grouped.setdefault(key, {})[_branch_index(example)] = _purse_return_fraction(
            example
        )
        if key not in absorption_by_group:
            absorption_by_group[key] = dict(example.branch_absorption or ())
    optimal_by_group = {
        key: _optimal_branches(returns) for key, returns in grouped.items()
    }
    # Same footing as :func:`_branch_metrics_v7`: unscorable groups are the
    # ones where every branch ties *or* where only one branch was emitted.
    single_branch = {key for key, returns in grouped.items() if len(returns) < 2}
    degenerate = {
        key
        for key, optimal in optimal_by_group.items()
        if key not in single_branch and len(optimal) == len(grouped[key])
    }
    rng = random.Random(seed)
    # A trivial policy names an intent, not a branch index. When the branch
    # it wants was dropped for naming an action a surviving branch already
    # executes, the honest score is the survivor's -- that *is* what the
    # engine plays for that intent. Guessing a fallback chain instead
    # mis-scores the floor, and this table is the blocking promotion gate.
    policies: dict[str, str | None] = {
        "always_fold": "fold",
        "always_check_call": "check_call",
        "always_aggress_half_pot": "aggress_half_pot",
        "always_aggress_pot": "aggress_pot",
        "uniform_random_legal": None,
    }
    total = len(grouped)
    scored = total - len(degenerate) - len(single_branch)
    table: dict[str, dict[str, float | None]] = {}
    for name, intent in policies.items():
        correct = 0
        regret = 0.0
        correct_all = 0
        regret_all = 0.0
        for key, returns in grouped.items():
            absorption = absorption_by_group.get(key) or {}
            if intent is not None:
                chosen = _intent_branch(intent, absorption, returns)
            elif absorption:
                # Uniform over the LEGAL actions, which is what the contract
                # name promises: the absorption map's keys are exactly the
                # candidate branches, so sample those and follow each to the
                # branch that carries its value. Sampling survivors directly
                # would silently re-weight toward whichever survivor absorbed
                # the most candidates.
                chosen = _intent_branch(
                    rng.choice(sorted(absorption)), absorption, returns
                )
            else:
                chosen = rng.choice(sorted(returns))
            best = max(returns.values())
            hit = int(chosen in optimal_by_group[key])
            gap = 100.0 * (best - returns[chosen])
            correct_all += hit
            regret_all += gap
            if key not in degenerate and key not in single_branch:
                correct += hit
                regret += gap
        table[name] = {
            "best_action_accuracy": _round_or_none(
                None if scored == 0 else correct / scored
            ),
            "mean_regret_pct": _round_or_none(None if scored == 0 else regret / scored),
            "best_action_accuracy_all_groups": round(correct_all / max(1, total), 6),
            "mean_regret_pct_all_groups": round(regret_all / max(1, total), 6),
        }
    return {
        "scoring": (
            "primary figures exclude degenerate groups, matching "
            "evaluation.branch_metrics"
        ),
        "total_groups": total,
        "degenerate_groups": len(degenerate),
        "single_branch_groups": len(single_branch),
        "scored_groups": scored,
        "degenerate_group_fraction": round(len(degenerate) / max(1, total), 6),
        "policies": table,
    }


def _margin_quantiles_v7(
    examples: Sequence[TrainingExample],
    score_rows: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Top-2 branch margin distribution, the unit of the hybrid gate.

    A raw threshold such as 0.2 means a different override rate on every
    candidate; the serve gate thresholds a quantile of this distribution
    and maps it back through this table.
    """

    grouped: dict[object, tuple[Sequence[float], set[int]]] = {}
    for example, scores in zip(examples, score_rows):
        key = _decision_key(example)
        if key not in grouped:
            grouped[key] = (scores, set())
        grouped[key][1].add(_branch_index(example))
    margins: list[float] = []
    for scores, branches in grouped.values():
        if len(branches) < 2:
            continue
        ordered = sorted((scores[branch] for branch in branches), reverse=True)
        margins.append(ordered[0] - ordered[1])
    margins.sort()
    if not margins:
        return {}

    def quantile(fraction: float) -> float:
        index = min(len(margins) - 1, max(0, round(fraction * (len(margins) - 1))))
        return round(margins[index], 6)

    return {
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
    }


def _hybrid_calibration_v7(
    examples: Sequence[TrainingExample],
    score_rows: Sequence[Sequence[float]],
) -> dict[str, object] | None:
    """Held-out override curve in branch space.

    The recorded behavior action carries a family, not a branch; aggressive
    behavior maps to the half-pot branch, which matches the deployed
    policy's ~half-pot sizing. This is an approximation and is labelled as
    such in the output.
    """

    grouped: dict[object, list[tuple[TrainingExample, Sequence[float]]]] = {}
    for example, scores in zip(examples, score_rows):
        grouped.setdefault(_decision_key(example), []).append((example, scores))
    family_to_branch = {0: 0, 1: 1, 2: 2}
    decisions = []
    for rows in grouped.values():
        returns = {
            _branch_index(example): _purse_return_fraction(example)
            for example, _ in rows
        }
        scores = rows[0][1]
        behavior_family = max(
            range(3), key=lambda action: rows[0][0].behavior_probabilities[action]
        )
        behavior = family_to_branch[behavior_family]
        if behavior not in returns:
            continue
        predicted = max(returns, key=lambda branch: scores[branch])
        decisions.append(
            (
                returns,
                behavior,
                predicted,
                float(scores[predicted]) - float(scores[behavior]),
            )
        )
    if not decisions:
        return None
    behavior_regret = (
        100.0
        * sum(
            max(returns.values()) - returns[behavior]
            for returns, behavior, _, _ in decisions
        )
        / len(decisions)
    )
    curves = []
    for threshold in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        regret = 0.0
        overrides = 0
        for returns, behavior, predicted, margin in decisions:
            chosen = behavior
            if predicted != behavior and margin >= threshold:
                chosen = predicted
                overrides += 1
            regret += max(returns.values()) - returns[chosen]
        mean_regret = 100.0 * regret / len(decisions)
        curves.append(
            {
                "min_value_advantage": threshold,
                "override_rate": overrides / len(decisions),
                "mean_regret_pct": mean_regret,
                "regret_delta_vs_behavior_pct": mean_regret - behavior_regret,
            }
        )
    helpful = [
        row
        for row in curves
        if row["override_rate"] > 0.0 and row["mean_regret_pct"] < behavior_regret
    ]
    recommended = (
        min(
            helpful,
            key=lambda row: (row["mean_regret_pct"], row["min_value_advantage"]),
        )["min_value_advantage"]
        if helpful
        else None
    )
    return {
        "decision_groups": len(decisions),
        "behavior_mean_regret_pct": behavior_regret,
        "recommended_min_value_advantage": recommended,
        "behavior_branch_mapping": "aggressive behavior maps to aggress_half_pot",
        "curve": curves,
    }


def _train_cuda_v7(
    train_behavior_examples: Sequence[TrainingExample],
    train_reward_examples: Sequence[TrainingExample],
    validation_reward_examples: Sequence[TrainingExample],
    architecture: Mapping[str, object],
    config: TrainingConfig,
) -> tuple[
    dict[str, object],
    tuple[float, BranchMetricsV7],
    tuple[float, BranchMetricsV7] | None,
    list[list[float]],
    str,
    dict[str, object],
]:
    """Fit the format-2 architecture with the measured 2026-08-14 recipe.

    AdamW with decoupled weight decay (biases and LayerNorm excluded),
    linear warmup then cosine decay, global-norm gradient clipping, He
    trunk with zero-initialized output layers, dropout, group-coherent
    batches, and best-epoch checkpointing on weight-normalized validation
    loss. Every element here was A/B measured against the v6 recipe; see
    reviews/V7_TRAINING_REVIEW.md section 4.
    """

    import torch
    from torch import nn

    device_name = validate_training_device("cuda")
    device = torch.device("cuda")
    torch.manual_seed(config.resolved_init_seed)
    torch.cuda.manual_seed_all(config.resolved_init_seed)

    card_indices = list(architecture["card_indices"])  # type: ignore[arg-type]
    context_indices = list(architecture["context_indices"])  # type: ignore[arg-type]
    encoder_width = int(architecture["encoder_width"])  # type: ignore[arg-type]
    trunk_widths = [int(v) for v in architecture["trunk_widths"]]  # type: ignore[union-attr]
    towers = architecture["head_towers"]

    class _NetworkV7(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.card_enc = nn.Linear(len(card_indices), encoder_width)
            self.ctx_enc = nn.Linear(len(context_indices), encoder_width)
            self.card_ln = nn.LayerNorm(encoder_width)
            self.ctx_ln = nn.LayerNorm(encoder_width)
            dims = [2 * encoder_width, *trunk_widths]
            self.trunk = nn.ModuleList(
                nn.Linear(dims[i], dims[i + 1]) for i in range(len(trunk_widths))
            )
            self.trunk_ln = nn.ModuleList(
                nn.LayerNorm(dims[i + 1]) for i in range(len(trunk_widths) - 1)
            )
            self.drop = nn.Dropout(float(config.dropout))
            width = trunk_widths[-1]
            self.towers = nn.ModuleDict(
                {
                    name: nn.Linear(width, int(towers[name]))  # type: ignore[index]
                    for name in V7_HEAD_SIZES
                }
            )
            self.outs = nn.ModuleDict(
                {
                    name: nn.Linear(int(towers[name]), size)  # type: ignore[index]
                    for name, size in V7_HEAD_SIZES.items()
                }
            )

        def forward(self, card, ctx):
            left = self.card_ln(torch.relu(self.card_enc(card)))
            right = self.ctx_ln(torch.relu(self.ctx_enc(ctx)))
            hidden = torch.cat([left, right], dim=1)
            for index, layer in enumerate(self.trunk):
                hidden = torch.relu(layer(hidden))
                if index < len(self.trunk_ln):
                    hidden = self.drop(self.trunk_ln[index](hidden))
            return {
                name: self.outs[name](torch.relu(self.towers[name](hidden)))
                for name in V7_HEAD_SIZES
            }

    model = _NetworkV7()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=math.sqrt(2.0 / module.in_features))
                nn.init.zeros_(module.bias)
        # Zero output layers: the model starts at the constant predictor,
        # which is the correct prior at this signal-to-noise ratio and was
        # the largest single init effect in the recipe sweep.
        for name in V7_HEAD_SIZES:
            nn.init.zeros_(model.outs[name].weight)
            nn.init.zeros_(model.outs[name].bias)
    model.to(device)

    def tensors(examples: Sequence[TrainingExample]) -> dict[str, object]:
        features = torch.tensor(
            [example.features for example in examples],
            dtype=torch.float32,
            device=device,
        )
        return {
            "card": features[:, card_indices],
            "ctx": features[:, context_indices],
            "branch": torch.tensor(
                [_branch_index(example) for example in examples],
                dtype=torch.long,
                device=device,
            ),
            "family": torch.tensor(
                [example.action_family_index for example in examples],
                dtype=torch.long,
                device=device,
            ),
            "target": torch.tensor(
                [_action_value_target(example, config) for example in examples],
                dtype=torch.float32,
                device=device,
            ),
            "weight": torch.tensor(
                [_branch_value_weight(example, config) for example in examples],
                dtype=torch.float32,
                device=device,
            ),
            "class_weight": torch.tensor(
                [_branch_class_weight(example, config) for example in examples],
                dtype=torch.float32,
                device=device,
            ),
        }

    def group_rows(examples: Sequence[TrainingExample]) -> list[list[int]]:
        groups: dict[object, list[int]] = {}
        for row, example in enumerate(examples):
            groups.setdefault(_decision_key(example), []).append(row)
        return list(groups.values())

    def state_value_targets(
        examples: Sequence[TrainingExample], groups: list[list[int]]
    ) -> "torch.Tensor":
        values = [0.0] * len(examples)
        for rows in groups:
            by_family: dict[int, list[float]] = {}
            for row in rows:
                by_family.setdefault(examples[row].action_family_index, []).append(
                    _action_value_target(examples[row], config)
                )
            probabilities = examples[rows[0]].behavior_probabilities
            # V(s) = E_{a~pi}[Q(s, a)], and `behavior_probabilities` is
            # one-hot by construction (decision_engine builds it as
            # `float(label == family)`), so this is exactly the value of the
            # branch the actor really played.
            #
            # Under legality-and-distinctness-aware branch sets that branch
            # may carry no label of its own: it was dropped as a duplicate of
            # a branch the engine executes identically. Follow the absorption
            # map to the survivor that holds its value. Renormalizing the
            # probabilities instead is a no-op on a one-hot distribution --
            # it silently writes V(s) = 0 for exactly the decisions whose own
            # action was absorbed, which is most of them.
            absorbed = _absorbing_family(examples[rows[0]], by_family)
            expected = sum(
                probabilities[family]
                * (sum(by_family[target]) / len(by_family[target]))
                for family, target in absorbed.items()
                if target in by_family
            )
            for row in rows:
                values[row] = expected
        return torch.tensor(values, dtype=torch.float32, device=device)

    train_data = tensors(train_reward_examples)
    all_train_groups = group_rows(train_reward_examples)
    # state_value must see EVERY group. Its target is a property of the
    # decision, not of the action objective, and the tensor it returns is
    # indexed by row: handing it a filtered group list would silently leave
    # 0.0 in every row the filter removed, which is a wrong label rather than
    # an absent one.
    train_data["state_value"] = state_value_targets(
        train_reward_examples, all_train_groups
    )
    behavior_data = (
        tensors(train_behavior_examples) if train_behavior_examples else None
    )
    validation_data = (
        tensors(validation_reward_examples) if validation_reward_examples else None
    )
    validation_groups = (
        group_rows(validation_reward_examples) if validation_reward_examples else []
    )

    # --- the zero-signal group filter -------------------------------------
    #
    # `action_weight` is a SEPARATE tensor from `weight`. The train_loss and
    # validation_loss this function reports through `summarize()` are computed
    # from `weight`, and they have to stay on the same footing as every
    # artifact built before this option existed.
    filter_mode = config.degenerate_group_filter
    train_zero_signal = [
        _zero_signal_group(train_reward_examples, rows, config)
        for rows in all_train_groups
    ]
    validation_zero_signal = [
        _zero_signal_group(validation_reward_examples, rows, config)
        for rows in validation_groups
    ]

    #: How many groups the random arm actually caught that were degenerate.
    #: Reported so an inert control is visible: a "random" mask that happens
    #: to remove the whole zero-signal set, or none of it, is not a control.
    random_arm_trace: dict[str, int] = {}

    def _mask_flags(
        groups: Sequence[Sequence[int]],
        zero_signal: Sequence[bool],
        label: str,
    ) -> list[bool]:
        """Which groups this mode mutes."""

        if filter_mode != "random":
            return list(zero_signal)
        # Same COUNT as the zero-signal set, drawn uniformly over all
        # groups, so the arm differs from the treated one in WHICH groups
        # it mutes and in nothing else.
        wanted = sum(1 for flat in zero_signal if flat)
        # Deterministic and split-specific: train and validation must not
        # receive the same draw, and the seed must be recoverable from the
        # manifest. `hash()` is not used -- it is PYTHONHASHSEED-randomised
        # and would make the mask irreproducible across processes.
        chooser = random.Random(
            config.degenerate_group_filter_seed * 1_000_003
            + len(groups) * 7
            + wanted
            + (0 if label == "train" else 1)
        )
        picked = set(chooser.sample(range(len(groups)), wanted)) if wanted else set()
        random_arm_trace[f"{label}_muted"] = len(picked)
        random_arm_trace[f"{label}_muted_that_were_degenerate"] = sum(
            1 for index in picked if zero_signal[index]
        )
        random_arm_trace[f"{label}_zero_signal"] = wanted
        return [index in picked for index in range(len(groups))]

    def action_weights(
        data: dict[str, object],
        groups: Sequence[Sequence[int]],
        zero_signal: Sequence[bool],
        label: str = "train",
    ) -> "torch.Tensor":
        weight = data["weight"].clone()
        if filter_mode != "off":
            flags = _mask_flags(groups, zero_signal, label)
            muted = [
                row
                for rows, flat in zip(groups, flags)
                if flat
                for row in rows
            ]
            if muted:
                weight[torch.tensor(muted, dtype=torch.long, device=device)] = 0.0
        return weight

    train_data["action_weight"] = action_weights(
        train_data, all_train_groups, train_zero_signal, "train"
    )
    if validation_data is not None:
        validation_data["action_weight"] = action_weights(
            validation_data, validation_groups, validation_zero_signal, "validation"
        )

    # Zero-weighting and dropping give a group's batch the SAME action loss:
    # a muted group contributes nothing to the numerator and nothing to the
    # denominator either way. The two modes differ only in whether the rows
    # stay in the batch -- and therefore in the step count, the
    # behavior:reward update ratio, and whether state_value and
    # residual_scale keep learning from them.
    if filter_mode == "drop":
        train_groups = [
            rows for rows, flat in zip(all_train_groups, train_zero_signal) if not flat
        ]
    else:
        train_groups = list(all_train_groups)
    if filter_mode != "off" and not train_groups:
        raise ValueError(
            "degenerate_group_filter removed every training decision group"
        )
    if filter_mode != "off" and float(train_data["action_weight"].sum()) <= 0.0:
        raise ValueError(
            "degenerate_group_filter muted every training row's action weight"
        )

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith("bias") or "_ln" in name else decay).append(
            parameter
        )
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config.weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    # Batches are group-coherent, so the configured example batch size has to
    # be converted into a group count. This used to divide by a hardcoded 4;
    # with legality-and-distinctness-aware branch sets the mean group is
    # smaller than that, and the constant would quietly shrink the effective
    # batch and raise gradient noise without any recipe change being recorded.
    #
    # It counts the rows actually batched, not len(train_reward_examples).
    # Under "drop" the example list still holds the filtered rows, so the old
    # expression would read 5.69 rows per group instead of 4.00 and quietly
    # cut group_batches from 64 to 45 -- the same silent-shrink failure the
    # paragraph above is about.
    batched_rows = sum(len(rows) for rows in train_groups)
    mean_group_rows = batched_rows / max(1, len(train_groups))
    group_batches = max(1, int(config.batch_size / max(1.0, mean_group_rows)))
    reward_steps_per_epoch = max(1, math.ceil(len(train_groups) / group_batches))
    steps_per_epoch = reward_steps_per_epoch
    if behavior_data is not None:
        steps_per_epoch += max(
            1, math.ceil(len(train_behavior_examples) / config.batch_size)
        )
    total_steps = max(1, steps_per_epoch * config.epochs)
    step = 0

    def set_learning_rate() -> None:
        if step < config.warmup_steps:
            factor = (step + 1) / max(1, config.warmup_steps)
        else:
            progress = (step - config.warmup_steps) / max(
                1, total_steps - config.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            factor = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * factor

    def centered_prediction(outputs, data, rows) -> "torch.Tensor":
        branch = data["branch"][rows]
        values = outputs["action_value"][
            torch.arange(rows.shape[0], device=device), branch
        ]
        return values

    def reward_batch_loss(rows_list: list[list[int]]) -> "torch.Tensor":
        rows = torch.tensor(
            [row for rows in rows_list for row in rows],
            dtype=torch.long,
            device=device,
        )
        group_ids = torch.tensor(
            [
                index
                for index, rows_in_group in enumerate(rows_list)
                for _ in rows_in_group
            ],
            dtype=torch.long,
            device=device,
        )
        outputs = model(train_data["card"][rows], train_data["ctx"][rows])
        predicted = centered_prediction(outputs, train_data, rows)
        counts = torch.zeros(len(rows_list), device=device).index_add_(
            0, group_ids, torch.ones_like(predicted)
        )
        sums = torch.zeros(len(rows_list), device=device).index_add_(
            0, group_ids, predicted
        )
        centered = predicted - (sums / counts)[group_ids]
        target = train_data["target"][rows]
        weight = train_data["action_weight"][rows]
        # Normalize on the UNFILTERED weight, deliberately.
        #
        # Dividing by the muted sum would renormalize the surviving rows and
        # make the action term ~1.38x larger relative to `state_loss` and
        # `residual_loss`, which are weight-blind `.mean()`s at fixed
        # weights. The filtered arm would then differ from the control in
        # TWO ways -- which rows the action objective sees, AND the balance
        # between the three heads -- and no result could be attributed to
        # the intervention. With the filter off the two sums are equal, so
        # this is bit-identical to the unfiltered trainer.
        #
        # The clamp stays for the "zero_weight" 0/0 case: a whole batch can
        # be muted (measured 1.6e-34 at batch_size 256, but 7.7e-3 at 16 and
        # 0.30 at 4). The numerator is zero there too, so a muted batch
        # contributes no action gradient rather than a NaN that
        # clip_grad_norm_ would raise on.
        scale = train_data["weight"][rows]
        action_loss = (weight * (centered - target).square()).sum() / torch.clamp(
            scale.sum(), min=1e-12
        )
        state = outputs["state_value"].squeeze(1)
        state_loss = (state - train_data["state_value"][rows]).square().mean()
        residual = outputs["residual_scale"][
            torch.arange(rows.shape[0], device=device), train_data["branch"][rows]
        ]
        residual_loss = (residual - (centered - target).abs().detach()).square().mean()
        return (
            action_loss
            + config.state_value_weight * state_loss
            + config.residual_scale_weight * residual_loss
        )

    def behavior_batch_loss(rows: "torch.Tensor") -> "torch.Tensor":
        outputs = model(behavior_data["card"][rows], behavior_data["ctx"][rows])
        logits = outputs["behavior_prior"]
        log_probabilities = torch.log_softmax(logits, dim=1)
        chosen = log_probabilities[
            torch.arange(rows.shape[0], device=device),
            behavior_data["family"][rows],
        ]
        weight = behavior_data["class_weight"][rows]
        return config.behavior_prior_weight * (-(weight * chosen).sum() / weight.sum())

    def optimize(loss: "torch.Tensor") -> None:
        nonlocal step
        set_learning_rate()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite CUDA gradient during v7 training")
        optimizer.step()
        step += 1

    def validation_loss_weighted(*, filtered: bool) -> float | None:
        """Weighted centered validation loss -- the best-epoch criterion.

        ``filtered`` scores with the same action weights training uses.
        Leaving the zero-signal groups in makes this criterion strictly more
        tolerant of a constant head: they add nothing to the numerator that a
        constant head incurs, while adding 27.8% to the denominator, so the
        unfiltered criterion accepts sqrt(W_all/W_live) = 1.177x more
        off-target spread (38.4% more spread variance) before it prefers the
        constant. A training-side filter judged by an unfiltered criterion can
        therefore still early-stop onto the dead checkpoint, which is the
        failure the filter exists to remove.
        """

        if validation_data is None:
            return None
        model.eval()
        with torch.no_grad():
            outputs = model(validation_data["card"], validation_data["ctx"])
            rows = torch.arange(validation_data["branch"].shape[0], device=device)
            predicted = outputs["action_value"][rows, validation_data["branch"]]
            group_ids = torch.zeros_like(rows)
            for index, rows_in_group in enumerate(validation_groups):
                for row in rows_in_group:
                    group_ids[row] = index
            counts = torch.zeros(len(validation_groups), device=device).index_add_(
                0, group_ids, torch.ones_like(predicted)
            )
            sums = torch.zeros(len(validation_groups), device=device).index_add_(
                0, group_ids, predicted
            )
            centered = predicted - (sums / counts)[group_ids]
            weight = validation_data["action_weight" if filtered else "weight"]
            # Same reasoning as the training term: the denominator is the
            # unfiltered weight, so a filtered run's `best_validation_loss`
            # stays on the same scale as the control's and the early-stopping
            # criterion is not silently redefined under an unchanged key.
            scale = validation_data["weight"]
            loss = (
                weight * (centered - validation_data["target"]).square()
            ).sum() / torch.clamp(scale.sum(), min=1e-12)
        model.train()
        return float(loss)

    generator = random.Random(config.resolved_init_seed + 1)
    best_loss = math.inf
    best_state: dict[str, object] | None = None
    best_epoch = 0
    stale_epochs = 0
    epochs_run = 0
    model.train()
    for epoch in range(config.epochs):
        epochs_run = epoch + 1
        generator.shuffle(train_groups)
        for start in range(0, len(train_groups), group_batches):
            optimize(reward_batch_loss(train_groups[start : start + group_batches]))
        if behavior_data is not None:
            order = list(range(len(train_behavior_examples)))
            generator.shuffle(order)
            for start in range(0, len(order), config.batch_size):
                rows = torch.tensor(
                    order[start : start + config.batch_size],
                    dtype=torch.long,
                    device=device,
                )
                optimize(behavior_batch_loss(rows))
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("non-finite CUDA parameter during v7 training")
        epoch_loss = validation_loss_weighted(filtered=filter_mode != "off")
        if epoch_loss is None:
            continue
        if epoch_loss < best_loss - 1e-9:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stop_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    # Both criteria on the selected checkpoint, so an artifact trained with
    # the filter on stays comparable with every artifact trained without it.
    final_validation_unfiltered = validation_loss_weighted(filtered=False)
    final_validation_filtered = validation_loss_weighted(filtered=True)
    model.eval()

    def scores_for(data: dict[str, object]) -> list[list[float]]:
        with torch.no_grad():
            outputs = model(data["card"], data["ctx"])
            return outputs["action_value"].detach().cpu().tolist()

    def summarize(
        examples: Sequence[TrainingExample],
        data: dict[str, object] | None,
    ) -> tuple[tuple[float, BranchMetricsV7], list[list[float]]] | None:
        if data is None or not examples:
            return None
        rows = scores_for(data)
        metrics = _branch_metrics_v7(examples, rows)
        with torch.no_grad():
            outputs = model(data["card"], data["ctx"])
            indexes = torch.arange(data["branch"].shape[0], device=device)
            predicted = outputs["action_value"][indexes, data["branch"]]
            weight = data["weight"]
            loss = (weight * (predicted - data["target"]).square()).sum() / weight.sum()
        return (float(loss), metrics), rows

    train_summary = summarize(train_reward_examples, train_data)
    assert train_summary is not None
    validation_summary = summarize(validation_reward_examples, validation_data)

    def block(module: "nn.Linear") -> dict[str, object]:
        return {
            "w": module.weight.detach().cpu().tolist(),
            "b": module.bias.detach().cpu().tolist(),
        }

    def norm_block(module: "nn.LayerNorm") -> dict[str, object]:
        return {
            "ln_g": module.weight.detach().cpu().tolist(),
            "ln_b": module.bias.detach().cpu().tolist(),
        }

    trunk_blocks = []
    for index, layer in enumerate(model.trunk):
        entry = block(layer)
        if index < len(model.trunk_ln):
            entry.update(norm_block(model.trunk_ln[index]))
        trunk_blocks.append(entry)
    weights = {
        "card_encoder": {**block(model.card_enc), **norm_block(model.card_ln)},
        "context_encoder": {**block(model.ctx_enc), **norm_block(model.ctx_ln)},
        "trunk": trunk_blocks,
        "heads": {
            name: {
                "tower_w": model.towers[name].weight.detach().cpu().tolist(),
                "tower_b": model.towers[name].bias.detach().cpu().tolist(),
                "out_w": model.outs[name].weight.detach().cpu().tolist(),
                "out_b": model.outs[name].bias.detach().cpu().tolist(),
            }
            for name in V7_HEAD_SIZES
        },
    }
    _assert_finite_weights(weights, "v7 CUDA training")
    training_trace = {
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "best_validation_loss_weighted": (
            None if best_loss is math.inf else round(best_loss, 6)
        ),
        "optimizer_steps": step,
        "degenerate_group_filter": filter_mode,
        "degenerate_group_filter_seed": (
            config.degenerate_group_filter_seed if filter_mode == "random" else None
        ),
        "degenerate_group_filter_trace": dict(random_arm_trace) or None,
        "degenerate_group_predicate": (
            "weighted target (branch value weight x action value target) "
            "identical across the group, so a constant head is its exact "
            "minimiser; single-branch groups always qualify"
        ),
        "train_groups_total": len(all_train_groups),
        "train_groups_zero_signal": sum(train_zero_signal),
        "train_groups_batched": len(train_groups),
        "validation_groups_total": len(validation_groups),
        "validation_groups_zero_signal": sum(validation_zero_signal),
        "group_batches": group_batches,
        "reward_steps_per_epoch": reward_steps_per_epoch,
        "best_epoch_criterion": (
            "filtered centered validation loss"
            if filter_mode != "off"
            else "centered validation loss over every group"
        ),
        "validation_loss_weighted_unfiltered": _round_or_none(
            final_validation_unfiltered
        ),
        "validation_loss_weighted_filtered": _round_or_none(
            final_validation_filtered
        ),
        # Invariant, not a metric: state_value is supervised on every group
        # regardless of the filter, so this figure must be identical across
        # filter modes on the same split. If it moves, a filtered group list
        # reached state_value_targets and 0.0 was written as a label.
        "state_value_target_abs_sum": round(
            float(train_data["state_value"].abs().sum()), 6
        ),
    }
    return (
        weights,
        train_summary[0],
        validation_summary[0] if validation_summary else None,
        validation_summary[1] if validation_summary else [],
        device_name,
        training_trace,
    )


def _finish_v7(
    examples: Sequence[TrainingExample],
    train_examples: Sequence[TrainingExample],
    validation_examples: Sequence[TrainingExample],
    train_behavior_examples: Sequence[TrainingExample],
    train_reward_examples: Sequence[TrainingExample],
    validation_reward_examples: Sequence[TrainingExample],
    means: Sequence[float],
    stds: Sequence[float],
    model_version: str,
    weights_path: Path,
    manifest_path: Path,
    output_path: Path,
    config: TrainingConfig,
) -> TrainingSummary:
    """Train, export, and summarize a format-2 candidate."""

    architecture = default_v7_architecture()
    architecture["dropout"] = float(config.dropout)
    validate_v7_architecture(architecture)
    (
        weights,
        train_metrics,
        validation_metrics,
        validation_score_rows,
        device_name,
        training_trace,
    ) = _train_cuda_v7(
        train_behavior_examples,
        train_reward_examples,
        validation_reward_examples,
        architecture,
        config,
    )
    train_loss = train_metrics[0]
    validation_loss = validation_metrics[0] if validation_metrics else None
    validation_branch_metrics = validation_metrics[1] if validation_metrics else None
    validation_accuracy = (
        validation_branch_metrics.best_action_accuracy
        if validation_branch_metrics
        else None
    )
    validation_regret_pct = (
        validation_branch_metrics.mean_regret_pct if validation_branch_metrics else None
    )
    validation_action_value_mae_pct = (
        validation_branch_metrics.action_value_mae_pct
        if validation_branch_metrics
        else None
    )
    validation_degenerate_fraction = (
        validation_branch_metrics.degenerate_group_fraction
        if validation_branch_metrics
        else None
    )
    validation_calibration = _hybrid_calibration_v7(
        validation_reward_examples, validation_score_rows
    )
    baselines = _baseline_floor_v7(
        validation_reward_examples or train_reward_examples,
        config.resolved_split_seed,
    )
    margin_quantiles = _margin_quantiles_v7(
        validation_reward_examples, validation_score_rows
    )

    output_path.mkdir(parents=True, exist_ok=True)
    weights_document = _round9(
        {
            "format": MODEL_FORMAT,
            "format_version": MODEL_FORMAT_VERSION_V7,
            "model_version": model_version,
            "feature_normalization": {"means": list(means), "stds": list(stds)},
            "weights": weights,
        }
    )
    encoded = json.dumps(
        weights_document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    weights_sha256 = hashlib.sha256(encoded).hexdigest()
    weights_path.write_bytes(encoded + b"\n")

    manifest = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION_V7,
        "model_version": model_version,
        "state": "candidate",
        "parent_version": None,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_size": LEARNING_INPUT_SIZE,
        "feature_names": list(LEARNING_FEATURE_NAMES),
        "action_labels": list(BRANCH_LABELS),
        "architecture": architecture,
        "weights_file": weights_path.name,
        "weights_sha256": weights_sha256,
        "training_window": {
            "hand_count": len({example.table_id for example in examples}),
            "example_count": len(examples),
            "counterfactual_example_count": sum(
                example.counterfactual for example in examples
            ),
            "behavior_example_count": sum(
                not example.counterfactual for example in examples
            ),
            "source_policy_versions": sorted(
                {example.policy_version for example in examples}
            ),
        },
        "engine_parameters": {
            "safety_gates": DEFAULT_SAFETY_GATES.to_mapping(),
            "temperature_shaping": DEFAULT_TEMPERATURE_SHAPING.to_mapping(),
            "tracker_settings": DEFAULT_TRACKER_SETTINGS.to_mapping(),
            "bluff_settings": DEFAULT_BLUFF_SETTINGS.to_mapping(),
        },
        "serve": {
            # The OOD guard watches only the context block: card one-hots
            # z-score to 5.0-6.1 whenever set, so a whole-vector guard
            # detects card identity, which is how both archived hybrid
            # gauntlets ended up measuring a ~99% heuristic policy.
            "ood_guard_indices": list(context_feature_indices()),
            "margin_quantiles": margin_quantiles,
            "temperature": config.serve_temperature,
        },
        "training": {
            "objective": TRAINING_OBJECTIVE_V7,
            "epochs": config.epochs,
            "epochs_run": training_trace["epochs_run"],
            "best_epoch": training_trace["best_epoch"],
            "best_validation_loss_weighted": training_trace[
                "best_validation_loss_weighted"
            ],
            "optimizer": "adamw",
            "optimizer_steps": training_trace["optimizer_steps"],
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "warmup_steps": config.warmup_steps,
            "early_stop_patience": config.early_stop_patience,
            "gradient_clip": "global-norm 1.0",
            "return_scale_fraction": config.return_scale_fraction,
            "validation_fraction": config.validation_fraction,
            "split_seed": config.resolved_split_seed,
            "init_seed": config.resolved_init_seed,
            "class_weights": (
                list(config.class_weights) if config.class_weights else None
            ),
            "reinforcement_multiplier": config.reinforcement_multiplier,
            "behavior_prior_weight": config.behavior_prior_weight,
            "state_value_weight": config.state_value_weight,
            "residual_scale_weight": config.residual_scale_weight,
            "backend": "pytorch",
            "device": config.device,
            "device_name": device_name,
            "batch_size": config.batch_size,
            "validation_split": "whole hands by table_id",
            "reward_examples": "counterfactual simulator rollouts only",
            "action_head_semantics": "counterfactual_value",
            "counterfactual_rollouts_per_family": config.counterfactual_rollouts,
            "degenerate_group_filter": training_trace["degenerate_group_filter"],
            "degenerate_group_predicate": training_trace[
                "degenerate_group_predicate"
            ],
            "degenerate_group_counts": {
                "train_total": training_trace["train_groups_total"],
                "train_zero_signal": training_trace["train_groups_zero_signal"],
                "train_batched": training_trace["train_groups_batched"],
                "validation_total": training_trace["validation_groups_total"],
                "validation_zero_signal": training_trace[
                    "validation_groups_zero_signal"
                ],
            },
            "group_batches": training_trace["group_batches"],
            "reward_steps_per_epoch": training_trace["reward_steps_per_epoch"],
            "state_value_target_abs_sum": training_trace[
                "state_value_target_abs_sum"
            ],
            "best_epoch_criterion": training_trace["best_epoch_criterion"],
            "validation_loss_weighted_unfiltered": training_trace[
                "validation_loss_weighted_unfiltered"
            ],
            "validation_loss_weighted_filtered": training_trace[
                "validation_loss_weighted_filtered"
            ],
            "targets": {
                "action": (
                    "four centered signed-log branch values; aggression is "
                    "valued at half-pot and full-pot sizes executed through "
                    "the acting policy's own decision path"
                ),
                "behavior_prior": (
                    "three-family cross-entropy on behavior rows; diagnostic "
                    "head, never a serve-path fallback"
                ),
                "state_value": "behavior-probability-weighted branch value",
                "residual_scale": "absolute centered residual; logged only",
            },
            "input_normalization": (
                "per-feature z-score from counterfactual rows in the training "
                "split, std floor 0.05; scales stored in the weights file"
            ),
        },
        "evaluation": {
            "train_loss": round(train_loss, 6),
            "validation_loss": None
            if validation_loss is None
            else round(validation_loss, 6),
            # Accuracy and regret exclude degenerate groups; the tie-credited
            # figures they replace live under branch_metrics.all_groups.
            "validation_best_action_accuracy": _round_or_none(validation_accuracy),
            "validation_mean_regret_pct": _round_or_none(validation_regret_pct),
            "validation_action_value_mae_pct": _round_or_none(
                validation_action_value_mae_pct
            ),
            "validation_degenerate_group_fraction": _round_or_none(
                validation_degenerate_fraction
            ),
            "branch_metrics": {
                "train": train_metrics[1].to_mapping(),
                "validation": None
                if validation_branch_metrics is None
                else validation_branch_metrics.to_mapping(),
            },
            "trivial_baselines": baselines,
            "hybrid_confidence_calibration": validation_calibration,
        },
        "promotion": None,
    }
    validate_artifact_manifest(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return TrainingSummary(
        examples=len(examples),
        train_examples=len(train_examples),
        validation_examples=len(validation_examples),
        train_loss=train_loss,
        validation_loss=validation_loss,
        validation_best_action_accuracy=validation_accuracy,
        validation_mean_regret_pct=validation_regret_pct,
        validation_action_value_mae_pct=validation_action_value_mae_pct,
        validation_degenerate_group_fraction=validation_degenerate_fraction,
        weights_sha256=weights_sha256,
        manifest_path=manifest_path,
        weights_path=weights_path,
    )


def train_candidate(
    examples: Sequence[TrainingExample],
    output_dir: str | Path,
    config: TrainingConfig = TrainingConfig(),
) -> TrainingSummary:
    """Train a candidate artifact from settled eligible telemetry examples."""

    _check_config(config)
    if not examples:
        raise ValueError("no settled eligible training examples found")
    for example in examples:
        if len(example.features) != LEARNING_INPUT_SIZE:
            raise ValueError("example features do not match the learning contract")
        if not math.isfinite(example.purse_bb) or example.purse_bb <= 0.0:
            raise ValueError("example purse_bb must be positive and finite")
        if (
            not math.isfinite(example.opponent_confidence)
            or not 0.0 <= example.opponent_confidence <= 1.0
        ):
            raise ValueError("example opponent_confidence must be in [0, 1]")
    if not any(example.counterfactual for example in examples):
        raise ValueError(
            "no counterfactual training examples found; settled telemetry is "
            "behavior warm-up data only"
        )
    output_path = Path(output_dir).expanduser().resolve()
    model_version = config.model_version or "candidate-" + datetime.now(UTC).strftime(
        "%Y%m%d%H%M%S"
    )
    weights_name = f"{model_version}.weights.json"
    weights_path = output_path / weights_name
    manifest_path = output_path / f"{model_version}.manifest.json"
    if weights_path.exists() or manifest_path.exists():
        raise FileExistsError(f"candidate artifact already exists for {model_version}")
    train_examples, validation_examples = _split(examples, config)
    train_reward_examples = [
        example for example in train_examples if example.counterfactual
    ]
    if not train_reward_examples:
        raise ValueError("training split has no counterfactual examples")
    # Behavior corpora may be much larger and lack live-only schema-2 context.
    # Keep normalization anchored to the counterfactual state distribution that
    # supplies the value targets, then apply those scales to every example.
    means, stds = _feature_normalization(train_reward_examples)
    train_examples = [_normalized(example, means, stds) for example in train_examples]
    validation_examples = [
        _normalized(example, means, stds) for example in validation_examples
    ]
    train_reward_examples = [
        example for example in train_examples if example.counterfactual
    ]
    validation_reward_examples = [
        example for example in validation_examples if example.counterfactual
    ]
    train_behavior_examples = [
        example for example in train_examples if not example.counterfactual
    ]
    if config.architecture == "v7":
        return _finish_v7(
            examples,
            train_examples,
            validation_examples,
            train_behavior_examples,
            train_reward_examples,
            validation_reward_examples,
            means,
            stds,
            model_version,
            weights_path,
            manifest_path,
            output_path,
            config,
        )
    device_name = "cpu"
    if config.device == "cuda":
        (
            weights,
            train_metrics,
            validation_metrics,
            validation_score_rows,
            device_name,
        ) = _train_cuda(
            train_behavior_examples,
            train_reward_examples,
            validation_reward_examples,
            config,
        )
        train_loss = train_metrics[0]
        validation_loss = validation_metrics[0] if validation_metrics else None
        validation_accuracy = validation_metrics[1] if validation_metrics else None
        validation_regret_pct = validation_metrics[2] if validation_metrics else None
        validation_action_value_mae_pct = (
            validation_metrics[3] if validation_metrics else None
        )
        validation_calibration = _hybrid_calibration(
            validation_reward_examples, validation_score_rows
        )
    else:
        rng = random.Random(config.resolved_init_seed)
        weights = _init_weights(rng)
        behavior_head = _init_behavior_head(rng)
        for _ in range(config.behavior_warmup_epochs):
            rng.shuffle(train_behavior_examples)
            for example in train_behavior_examples:
                _step(
                    weights,
                    example,
                    config,
                    policy_objective="imitation",
                    behavior_head=behavior_head,
                )
            _assert_finite_weights(weights, "behavior warm-up")
        for epoch in range(config.epochs):
            rng.shuffle(train_reward_examples)
            for example in train_reward_examples:
                _step(weights, example, config)
            _assert_finite_weights(weights, f"reward epoch {epoch + 1}")

        train_loss, _, _, _ = _loss(weights, train_reward_examples, config)
        validation_loss = None
        validation_accuracy = None
        validation_regret_pct = None
        validation_action_value_mae_pct = None
        validation_calibration = None
        if validation_reward_examples:
            (
                validation_loss,
                validation_accuracy,
                validation_regret_pct,
                validation_action_value_mae_pct,
            ) = _loss(weights, validation_reward_examples, config)
            validation_calibration = _hybrid_calibration(
                validation_reward_examples,
                [
                    _forward(weights, example.features)["action_logits"]
                    for example in validation_reward_examples
                ],
            )

    output_path.mkdir(parents=True, exist_ok=True)
    weights_document = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "model_version": model_version,
        "feature_normalization": {"means": means, "stds": stds},
        "weights": weights,
    }
    encoded = json.dumps(
        weights_document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    weights_sha256 = hashlib.sha256(encoded).hexdigest()
    weights_path.write_bytes(encoded + b"\n")

    manifest = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "model_version": model_version,
        "state": "candidate",
        "parent_version": None,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_size": LEARNING_INPUT_SIZE,
        "feature_names": list(LEARNING_FEATURE_NAMES),
        "action_labels": list(LABELS),
        "architecture": {
            "hidden_sizes": list(HIDDEN_SIZES),
            "heads": HEAD_SIZES.copy(),
        },
        "weights_file": weights_name,
        "weights_sha256": weights_sha256,
        "training_window": {
            "hand_count": len({example.table_id for example in examples}),
            "example_count": len(examples),
            "counterfactual_example_count": sum(
                example.counterfactual for example in examples
            ),
            "behavior_example_count": sum(
                not example.counterfactual for example in examples
            ),
            "source_policy_versions": sorted(
                {example.policy_version for example in examples}
            ),
        },
        # The engine parameters this candidate assumes. Written as the
        # current defaults today; evaluation-driven search may replace them
        # per candidate, and the contract validates every block.
        "engine_parameters": {
            "safety_gates": DEFAULT_SAFETY_GATES.to_mapping(),
            "temperature_shaping": DEFAULT_TEMPERATURE_SHAPING.to_mapping(),
            "tracker_settings": DEFAULT_TRACKER_SETTINGS.to_mapping(),
            "bluff_settings": DEFAULT_BLUFF_SETTINGS.to_mapping(),
        },
        "training": {
            "objective": TRAINING_OBJECTIVE,
            "epochs": config.epochs,
            "baseline_warmup_epochs": 0,
            "behavior_warmup_epochs": config.behavior_warmup_epochs,
            "behavior_warmup_target": (
                "behavior-only examples train a transient classifier and the shared "
                "trunk; the persisted action-value and sizing heads are untouched"
            ),
            "learning_rate": config.learning_rate,
            "return_scale_fraction": config.return_scale_fraction,
            "validation_fraction": config.validation_fraction,
            "seed": config.seed,
            "class_weights": (
                list(config.class_weights) if config.class_weights else None
            ),
            "reinforcement_multiplier": config.reinforcement_multiplier,
            "gradient_clip": config.gradient_clip,
            "backend": "pytorch" if config.device == "cuda" else "plain-python",
            "device": config.device,
            "device_name": device_name,
            "batch_size": config.batch_size if config.device == "cuda" else 1,
            "validation_split": "whole hands by table_id",
            "reward_examples": "counterfactual simulator rollouts only",
            "action_head_semantics": "counterfactual_value",
            "counterfactual_sampling": (
                "one eligible decision per recorded actor per hand; each legal action "
                "family is replayed from the identical seeded state and stochastic "
                "continuations are averaged"
            ),
            "counterfactual_rollouts_per_family": config.counterfactual_rollouts,
            "opponent_context_weight": ("0.25 + 0.75 * opponent evidence confidence"),
            "input_normalization": (
                "per-feature z-score from counterfactual rows in the training split; "
                "scales stored in the weights file"
            ),
            "targets": {
                "reward": (
                    "sign((action_chip_delta - mean_legal_action_chip_delta) / "
                    "starting_purse) * log1p(abs((action_chip_delta - "
                    "mean_legal_action_chip_delta) / starting_purse))"
                ),
                "action": (
                    "three bounded signed-log counterfactual action values; "
                    "positive targets receive the reinforcement multiplier as "
                    "loss weight, never as target magnitude"
                ),
                "playability": (
                    "retained for artifact compatibility; not trained by v6"
                ),
                "risk_fraction": (
                    "submitted new chips divided by effective stack for profitable "
                    "aggressive decisions when explicitly enabled; disabled by "
                    "default until action-value evaluation passes"
                ),
            },
            "train_risk_head": config.train_risk_head,
        },
        "evaluation": {
            "train_loss": round(train_loss, 6),
            "validation_loss": None
            if validation_loss is None
            else round(validation_loss, 6),
            "validation_best_action_accuracy": None
            if validation_accuracy is None
            else round(validation_accuracy, 6),
            "validation_mean_regret_pct": None
            if validation_regret_pct is None
            else round(validation_regret_pct, 6),
            "validation_action_value_mae_pct": None
            if validation_action_value_mae_pct is None
            else round(validation_action_value_mae_pct, 6),
            "hybrid_confidence_calibration": validation_calibration,
        },
        "promotion": None,
    }
    validate_artifact_manifest(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return TrainingSummary(
        examples=len(examples),
        train_examples=len(train_examples),
        validation_examples=len(validation_examples),
        train_loss=train_loss,
        validation_loss=validation_loss,
        validation_best_action_accuracy=validation_accuracy,
        validation_mean_regret_pct=validation_regret_pct,
        validation_action_value_mae_pct=validation_action_value_mae_pct,
        weights_sha256=weights_sha256,
        manifest_path=manifest_path,
        weights_path=weights_path,
    )


def train_from_telemetry(
    telemetry_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig = TrainingConfig(),
) -> TrainingSummary:
    return train_candidate(load_training_examples(telemetry_path), output_dir, config)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _examples(paths: Iterable[str]) -> tuple[TrainingExample, ...]:
    examples: list[TrainingExample] = []
    for path in paths:
        examples.extend(load_training_examples(path))
    return tuple(examples)


def main(argv: Sequence[str] | None = None) -> int:
    # With slots=True the class attributes are member descriptors, so CLI
    # defaults must come from a constructed instance.
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", nargs="*", help="telemetry JSONL file(s)")
    parser.add_argument(
        "--foreign-csv",
        action="append",
        default=[],
        help="collector decision CSV of teacher-eligible foreign rows",
    )
    parser.add_argument("--output-dir", default="artifacts/candidates")
    parser.add_argument("--epochs", type=_positive_int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--return-scale-pct",
        type=float,
        default=100.0 * defaults.return_scale_fraction,
        help="purse-change percentage that maps to one policy-reward unit",
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=defaults.validation_fraction
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--model-version")
    parser.add_argument(
        "--baseline-warmup-epochs",
        type=int,
        default=defaults.baseline_warmup_epochs,
    )
    parser.add_argument(
        "--behavior-warmup-epochs",
        type=int,
        default=defaults.behavior_warmup_epochs,
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="weight the action loss by inverse family frequency",
    )
    parser.add_argument(
        "--reinforcement-multiplier",
        type=float,
        default=defaults.reinforcement_multiplier,
        help="positive-return weight; must be greater than the loss weight of 1",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=defaults.gradient_clip,
        help="per-parameter gradient bound for SGD",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=defaults.device)
    parser.add_argument("--batch-size", type=_positive_int, default=defaults.batch_size)
    parser.add_argument(
        "--counterfactual-rollouts",
        type=_positive_int,
        default=defaults.counterfactual_rollouts,
        help="rollouts already averaged into each counterfactual target",
    )
    parser.add_argument(
        "--train-risk-head",
        action="store_true",
        help="train learned sizing; off by default until action values pass",
    )
    args = parser.parse_args(argv)
    if not args.telemetry and not args.foreign_csv:
        parser.error("provide telemetry JSONL file(s) and/or --foreign-csv")
    examples = list(_examples(args.telemetry))
    for csv_path in args.foreign_csv:
        examples.extend(load_foreign_training_examples(csv_path))
    class_weights = balanced_class_weights(examples) if args.balance_classes else None
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        return_scale_fraction=args.return_scale_pct / 100.0,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        model_version=args.model_version,
        baseline_warmup_epochs=args.baseline_warmup_epochs,
        behavior_warmup_epochs=args.behavior_warmup_epochs,
        class_weights=class_weights,
        reinforcement_multiplier=args.reinforcement_multiplier,
        gradient_clip=args.gradient_clip,
        device=args.device,
        batch_size=args.batch_size,
        counterfactual_rollouts=args.counterfactual_rollouts,
        train_risk_head=args.train_risk_head,
    )
    if class_weights is not None:
        rounded = ", ".join(f"{weight:.3f}" for weight in class_weights)
        print(f"class_weights: {rounded}")
    summary = train_candidate(tuple(examples), args.output_dir, config)
    print(f"examples: {summary.examples}")
    print(f"train_loss: {summary.train_loss:.6f}")
    if summary.validation_loss is not None:
        print(f"validation_loss: {summary.validation_loss:.6f}")
        print_branch_summary(summary)
    print(f"manifest: {summary.manifest_path}")
    print(f"weights: {summary.weights_path}")
    print(f"weights_sha256: {summary.weights_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BranchMetricsV7",
    "print_branch_summary",
    "TRAINING_OBJECTIVE",
    "TrainingConfig",
    "TrainingSummary",
    "train_candidate",
    "train_from_telemetry",
    "validate_training_device",
]
