"""Live runtime for learned multi-head candidates.

:func:`load_policy` turns an artifact manifest into a playing policy: it
re-validates the manifest against the learning contract, verifies the
weights file's SHA-256, rebuilds any `engine_parameters` blocks through
their validated dataclasses, and returns a :class:`LearnedPokerPolicy`.

The learned network only ever proposes: its masked action head picks a
family among the legal ones and its risk head proposes an effective-stack
fraction for sizing. Every hard safety gate, the temperature shaping, the
opponent tracker, and the bluff path stay exactly as in the heuristic
engine -- the model steers inside the same rails.

Approved deployment goes through ``artifacts/approved.json`` (written
atomically by ``tools/promote_candidate.py``); :func:`load_approved`
follows that pointer and :func:`approved_fingerprint` lets the runner
detect a new approval between hands.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bluff import BluffSettings

from engine.decision_engine import (
    DecisionEngine,
    SafetyGates,
    SharedEquityCache,
    TemperatureShaping,
)
from engine.game_state import _integer, effective_stack_chips
from engine.hand_strength import prewarm
from engine.learning_contract import (
    BRANCH_FAMILIES,
    BRANCH_POT_FRACTIONS,
    LEARNING_INPUT_SIZE,
    MODEL_FORMAT_VERSION,
    MODEL_FORMAT_VERSION_V7,
    MODEL_FORMAT_VERSION_V9,
    LearningContractError,
    validate_artifact_manifest,
)
from engine.forward_kernel import _forward, _forward_v2
from engine.opponent_model import AggressionTracker, TrackerSettings
from engine.policy_features import LABELS

APPROVED_POINTER = "approved.json"

_FAMILY_ACTIONS = {
    "fold": ("fold",),
    "check_call": ("check", "call"),
    "aggress": ("bet", "raise"),
}


class LearnedPolicyError(ValueError):
    """Raised when an artifact cannot be loaded safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LearnedPolicyError(message)


def _matrix(value: object, name: str, rows: int, cols: int) -> list[list[float]]:
    _require(
        isinstance(value, list) and len(value) == rows, f"{name} must have {rows} rows"
    )
    matrix = []
    for row in value:  # type: ignore[union-attr]
        _require(
            isinstance(row, list) and len(row) == cols,
            f"{name} rows need {cols} values",
        )
        matrix.append([float(item) for item in row])
    return matrix


def _vector(value: object, name: str, size: int) -> list[float]:
    _require(
        isinstance(value, list) and len(value) == size,
        f"{name} must have {size} values",
    )
    return [float(item) for item in value]  # type: ignore[union-attr]


class LearnedPokerPolicy(DecisionEngine):
    """Multi-head proposals inside the shared deterministic safety rails."""

    def __init__(
        self,
        *,
        model_version: str,
        weights: Mapping[str, Any],
        normalization: Mapping[str, Sequence[float]],
        equity_trials: int = 200,
        seed: int = 7,
        temperature_shaping: TemperatureShaping | None = None,
        safety_gates: SafetyGates | None = None,
        opponent_tracker: AggressionTracker | None = None,
        bluff_settings: BluffSettings | None = None,
        equity_cache: SharedEquityCache | None = None,
        hyper_aggression_chance: float | None = None,
        action_head_semantics: str = "policy_logits",
        use_learned_sizing: bool = True,
        hybrid_min_value_advantage: float | None = None,
        hybrid_max_abs_z: float = 5.0,
    ) -> None:
        super().__init__(
            equity_trials=equity_trials,
            seed=seed,
            temperature_shaping=temperature_shaping,
            safety_gates=safety_gates,
            opponent_tracker=opponent_tracker,
            bluff_settings=bluff_settings,
            equity_cache=equity_cache,
            hyper_aggression_chance=hyper_aggression_chance,
        )
        self.policy_version = str(model_version)
        self._weights = {
            "w1": _matrix(weights.get("w1"), "w1", 128, LEARNING_INPUT_SIZE),
            "b1": _vector(weights.get("b1"), "b1", 128),
            "w2": _matrix(weights.get("w2"), "w2", 64, 128),
            "b2": _vector(weights.get("b2"), "b2", 64),
            "action_w": _matrix(weights.get("action_w"), "action_w", len(LABELS), 64),
            "action_b": _vector(weights.get("action_b"), "action_b", len(LABELS)),
            "playability_w": _vector(weights.get("playability_w"), "playability_w", 64),
            "playability_b": float(weights.get("playability_b", 0.0)),
            "risk_fraction_w": _vector(
                weights.get("risk_fraction_w"), "risk_fraction_w", 64
            ),
            "risk_fraction_b": float(weights.get("risk_fraction_b", 0.0)),
        }
        self._means = [float(v) for v in normalization["means"]]
        self._stds = [max(1e-6, float(v)) for v in normalization["stds"]]
        _require(
            len(self._means) == LEARNING_INPUT_SIZE
            and len(self._stds) == LEARNING_INPUT_SIZE,
            "feature normalization must cover all learning inputs",
        )
        self._proposed_risk_fraction: float | None = None
        _require(
            action_head_semantics in {"policy_logits", "counterfactual_value"},
            "unsupported action-head semantics",
        )
        _require(
            hybrid_min_value_advantage is None
            or (
                action_head_semantics == "counterfactual_value"
                and math.isfinite(float(hybrid_min_value_advantage))
                and float(hybrid_min_value_advantage) >= 0.0
            ),
            "hybrid mode requires non-negative counterfactual value advantage",
        )
        _require(
            math.isfinite(float(hybrid_max_abs_z)) and float(hybrid_max_abs_z) > 0.0,
            "hybrid_max_abs_z must be positive and finite",
        )
        self._action_head_semantics = action_head_semantics
        self._use_learned_sizing = bool(use_learned_sizing)
        self._hybrid_min_value_advantage = hybrid_min_value_advantage
        self._hybrid_max_abs_z = float(hybrid_max_abs_z)

    # The network proposes; the engine's gates and clamps dispose.

    def _family(self, features: tuple[float, ...]) -> str:
        # Without an equity read there is no validated learning-feature
        # vector, so the safe passive family stands in.
        del features
        return "check_call"

    def _equity_family(
        self,
        table,
        allowed,
        available,
        equity,
        features=None,
    ) -> str:
        self._proposed_risk_fraction = None
        if features is None or equity is None:
            return super()._equity_family(table, allowed, available, equity)
        temperature = self._situation_temperature(table, allowed, equity)
        learning = self._learning_features(
            table, allowed, features, equity, temperature
        )
        normalized = tuple(
            (value - mean) / std
            for value, mean, std in zip(learning, self._means, self._stds)
        )
        output = _forward(self._weights, normalized)
        probabilities = output["action_probabilities"]
        action_values = output["action_logits"]
        assert isinstance(probabilities, list) and isinstance(action_values, list)
        scores = (
            action_values
            if self._action_head_semantics == "counterfactual_value"
            else probabilities
        )
        legal_families = {
            family
            for family, actions in _FAMILY_ACTIONS.items()
            if any(action in available for action in actions)
        }
        if "fold" in available:
            legal_families.add("fold")
        if not legal_families:
            return super()._equity_family(table, allowed, available, equity)
        best = max(
            (family for family in LABELS if family in legal_families),
            key=lambda family: scores[LABELS.index(family)],
        )
        if self._hybrid_min_value_advantage is not None:
            heuristic = super()._equity_family(
                table, allowed, available, equity, features
            )
            if (
                max(abs(value) for value in normalized) > self._hybrid_max_abs_z
                or best == heuristic
                or scores[LABELS.index(best)] - scores[LABELS.index(heuristic)]
                < self._hybrid_min_value_advantage
            ):
                return heuristic
        if self._use_learned_sizing:
            self._proposed_risk_fraction = min(
                1.0, max(0.0, float(output["risk_fraction"]))
            )
        return best

    def _sized_action(
        self,
        action: str,
        table,
        allowed,
        equity,
        pot_fraction: float | None = None,
    ):
        if (
            self._use_learned_sizing
            and pot_fraction is None
            and self._proposed_risk_fraction is not None
        ):
            effective = max(1, effective_stack_chips(table))
            pot = _integer(table.get("potChips"), "potChips")
            call_chips = _integer(allowed.get("callChips", 0), "callChips")
            desired_new = self._proposed_risk_fraction * effective
            pot_fraction = min(2.0, max(0.1, desired_new / max(1, pot + call_chips)))
        return super()._sized_action(
            action, table, allowed, equity, pot_fraction=pot_fraction
        )


class LearnedPokerPolicyV7(DecisionEngine):
    """Format-2 runtime: size-conditioned branch values inside the rails.

    The action head's four outputs are branches, not families: fold,
    check/call, aggress at half pot, aggress at full pot. Choosing an
    aggress branch pins the engine's pot fraction to that branch's size,
    so the value served is the value that was measured. The behavior,
    state-value, and residual heads are diagnostics and are never read on
    the serve path; the heuristic remains the only fallback.
    """

    def __init__(
        self,
        *,
        model_version: str,
        architecture: Mapping[str, Any],
        weights: Mapping[str, Any],
        normalization: Mapping[str, Sequence[float]],
        serve: Mapping[str, Any],
        equity_trials: int = 200,
        seed: int = 7,
        temperature_shaping: TemperatureShaping | None = None,
        safety_gates: SafetyGates | None = None,
        opponent_tracker: AggressionTracker | None = None,
        bluff_settings: BluffSettings | None = None,
        equity_cache: SharedEquityCache | None = None,
        hyper_aggression_chance: float | None = None,
        serve_mode: str = "argmax",
        hybrid_min_margin_quantile: str | None = None,
        hybrid_max_abs_z: float = 5.0,
    ) -> None:
        super().__init__(
            equity_trials=equity_trials,
            seed=seed,
            temperature_shaping=temperature_shaping,
            safety_gates=safety_gates,
            opponent_tracker=opponent_tracker,
            bluff_settings=bluff_settings,
            equity_cache=equity_cache,
            hyper_aggression_chance=hyper_aggression_chance,
        )
        self.policy_version = str(model_version)
        self._architecture = dict(architecture)
        self._weights = _validate_v7_weights(weights, architecture)
        self._means = [float(v) for v in normalization["means"]]
        self._stds = [max(1e-6, float(v)) for v in normalization["stds"]]
        _require(
            len(self._means) == LEARNING_INPUT_SIZE
            and len(self._stds) == LEARNING_INPUT_SIZE,
            "feature normalization must cover all learning inputs",
        )
        guard = serve.get("ood_guard_indices") or ()
        _require(
            all(
                isinstance(index, int) and 0 <= index < LEARNING_INPUT_SIZE
                for index in guard
            ),
            "ood_guard_indices must be valid feature indices",
        )
        self._guard_indices = tuple(guard)
        quantiles = serve.get("margin_quantiles") or {}
        _require(isinstance(quantiles, Mapping), "margin_quantiles must be a mapping")
        self._margin_quantiles = {
            str(key): float(value) for key, value in quantiles.items()
        }
        _require(serve_mode in {"argmax", "sample"}, "unsupported serve mode")
        temperature = serve.get("temperature")
        _require(
            temperature is None
            or (math.isfinite(float(temperature)) and float(temperature) > 0.0),
            "serve temperature must be positive when set",
        )
        if serve_mode == "sample":
            _require(
                temperature is not None,
                "sampled serving requires a calibrated temperature",
            )
        self._serve_mode = serve_mode
        self._serve_temperature = None if temperature is None else float(temperature)
        if hybrid_min_margin_quantile is not None:
            _require(
                hybrid_min_margin_quantile in self._margin_quantiles,
                "hybrid_min_margin_quantile must name a persisted quantile",
            )
        self._hybrid_quantile = hybrid_min_margin_quantile
        _require(
            math.isfinite(float(hybrid_max_abs_z)) and float(hybrid_max_abs_z) > 0.0,
            "hybrid_max_abs_z must be positive and finite",
        )
        self._hybrid_max_abs_z = float(hybrid_max_abs_z)
        self._branch_pot_fraction: float | None = None

    def _family(self, features: tuple[float, ...]) -> str:
        del features
        return "check_call"

    def _legal_branches(self, available: set[str]) -> list[int]:
        branches = []
        for index, family in enumerate(BRANCH_FAMILIES):
            if family == "fold" and "fold" in available:
                branches.append(index)
            elif family == "check_call" and (
                "check" in available or "call" in available
            ):
                branches.append(index)
            elif family == "aggress" and ("bet" in available or "raise" in available):
                branches.append(index)
        return branches

    def _choose_branch(
        self, values: Sequence[float], branches: Sequence[int], table
    ) -> int:
        if self._serve_mode == "argmax" or self._serve_temperature is None:
            return max(branches, key=lambda branch: values[branch])
        # Deterministic per-state mixing, same shape as the scripted agents'
        # mix key: an argmax policy is exploitable by any adapting opponent,
        # while a calibrated-temperature softmax stays unpredictable at
        # equal expected value.
        centered = [values[branch] for branch in branches]
        offset = max(centered)
        weights = [
            math.exp((value - offset) / self._serve_temperature) for value in centered
        ]
        total = sum(weights)
        key = (
            f"{self.seed}:{self.policy_version}:{table.get('tableId')}"
            f":{table.get('street')}:{table.get('potChips')}"
        )
        digest = hashlib.sha256(key.encode()).digest()
        roll = int.from_bytes(digest[:8], "big") / float(1 << 64) * total
        for branch, weight in zip(branches, weights):
            roll -= weight
            if roll <= 0.0:
                return branch
        return branches[-1]

    def _equity_family(
        self,
        table,
        allowed,
        available,
        equity,
        features=None,
    ) -> str:
        self._branch_pot_fraction = None
        if features is None or equity is None:
            return super()._equity_family(table, allowed, available, equity)
        temperature = self._situation_temperature(table, allowed, equity)
        learning = self._learning_features(
            table, allowed, features, equity, temperature
        )
        normalized = tuple(
            (value - mean) / std
            for value, mean, std in zip(learning, self._means, self._stds)
        )
        values = _forward_v2(self._architecture, self._weights, normalized)[
            "action_value"
        ]
        branches = self._legal_branches(available)
        if not branches:
            return super()._equity_family(table, allowed, available, equity)
        best = self._choose_branch(values, branches, table)
        if self._hybrid_quantile is not None:
            heuristic = super()._equity_family(
                table, allowed, available, equity, features
            )
            heuristic_branches = [
                branch for branch in branches if BRANCH_FAMILIES[branch] == heuristic
            ]
            heuristic_value = (
                max(values[branch] for branch in heuristic_branches)
                if heuristic_branches
                else min(values[branch] for branch in branches)
            )
            threshold = self._margin_quantiles[self._hybrid_quantile]
            guard_values = (
                [abs(normalized[index]) for index in self._guard_indices]
                if self._guard_indices
                else [abs(value) for value in normalized]
            )
            if (
                max(guard_values) > self._hybrid_max_abs_z
                or BRANCH_FAMILIES[best] == heuristic
                or values[best] - heuristic_value < threshold
            ):
                return heuristic
        self._branch_pot_fraction = BRANCH_POT_FRACTIONS[best]
        return BRANCH_FAMILIES[best]

    def _sized_action(
        self,
        action: str,
        table,
        allowed,
        equity,
        pot_fraction: float | None = None,
    ):
        if pot_fraction is None and self._branch_pot_fraction is not None:
            pot_fraction = self._branch_pot_fraction
        return super()._sized_action(
            action, table, allowed, equity, pot_fraction=pot_fraction
        )


def _validate_v7_weights(
    weights: Mapping[str, Any], architecture: Mapping[str, Any]
) -> dict[str, Any]:
    """Structural validation of a format-2 weights block."""

    encoder_width = int(architecture["encoder_width"])
    trunk_widths = [int(v) for v in architecture["trunk_widths"]]
    towers = architecture["head_towers"]
    heads = architecture["heads"]

    def linear(block: object, name: str, rows: int, cols: int) -> dict[str, Any]:
        _require(isinstance(block, Mapping), f"{name} must be an object")
        return {
            "w": _matrix(block.get("w"), f"{name}.w", rows, cols),
            "b": _vector(block.get("b"), f"{name}.b", rows),
        }

    def with_norm(block: object, name: str, rows: int, cols: int) -> dict[str, Any]:
        built = linear(block, name, rows, cols)
        built["ln_g"] = _vector(block.get("ln_g"), f"{name}.ln_g", rows)
        built["ln_b"] = _vector(block.get("ln_b"), f"{name}.ln_b", rows)
        return built

    card_count = len(architecture["card_indices"])
    context_count = len(architecture["context_indices"])
    built: dict[str, Any] = {
        "card_encoder": with_norm(
            weights.get("card_encoder"), "card_encoder", encoder_width, card_count
        ),
        "context_encoder": with_norm(
            weights.get("context_encoder"),
            "context_encoder",
            encoder_width,
            context_count,
        ),
    }
    trunk_blocks = weights.get("trunk")
    _require(
        isinstance(trunk_blocks, list) and len(trunk_blocks) == len(trunk_widths),
        "trunk must list one block per trunk layer",
    )
    previous = 2 * encoder_width
    built_trunk = []
    for index, (block, width) in enumerate(zip(trunk_blocks, trunk_widths)):
        name = f"trunk[{index}]"
        if index < len(trunk_widths) - 1:
            built_trunk.append(with_norm(block, name, width, previous))
        else:
            built_trunk.append(linear(block, name, width, previous))
        previous = width
    built["trunk"] = built_trunk
    head_blocks = weights.get("heads")
    _require(isinstance(head_blocks, Mapping), "heads must be an object")
    built_heads: dict[str, Any] = {}
    for name, size in heads.items():
        block = head_blocks.get(name)
        _require(isinstance(block, Mapping), f"heads.{name} must be an object")
        tower = int(towers[name])
        built_heads[name] = {
            "tower_w": _matrix(
                block.get("tower_w"), f"heads.{name}.tower_w", tower, trunk_widths[-1]
            ),
            "tower_b": _vector(block.get("tower_b"), f"heads.{name}.tower_b", tower),
            "out_w": _matrix(block.get("out_w"), f"heads.{name}.out_w", size, tower),
            "out_b": _vector(block.get("out_b"), f"heads.{name}.out_b", size),
        }
    built["heads"] = built_heads
    return built


def _load_engine_parameters(manifest: Mapping[str, Any]) -> dict[str, Any]:
    parameters = manifest.get("engine_parameters") or {}
    built: dict[str, Any] = {}
    if parameters.get("safety_gates") is not None:
        built["safety_gates"] = SafetyGates.from_mapping(parameters["safety_gates"])
    if parameters.get("temperature_shaping") is not None:
        built["temperature_shaping"] = TemperatureShaping.from_mapping(
            parameters["temperature_shaping"]
        )
    if parameters.get("tracker_settings") is not None:
        built["opponent_tracker"] = AggressionTracker(
            TrackerSettings.from_mapping(parameters["tracker_settings"])
        )
    if parameters.get("bluff_settings") is not None:
        built["bluff_settings"] = BluffSettings.from_mapping(
            parameters["bluff_settings"]
        )
    return built


#: Monte-Carlo trials per equity estimate when nothing pins the value.
#:
#: `equity_trials` governs the precision of the number every safety gate
#: compares against, and until 2026-08-27 no approved artifact could
#: record it -- it lived only as this Python default, exactly the
#: unpinned-serve-parameter hole that let three unmeasured gate changes
#: ship under an approved manifest on 2026-08-17.
#:
#: A manifest may now pin `serve.equity_trials`, which wins over this
#: default and loses to an explicit caller argument. **The value itself
#: is unchanged at 200 and is NOT settled**: on the 2026-08-26 hand a
#: 521 BB call cleared its threshold by 0.0323 against an estimator
#: standard deviation of 0.0327, and the same spot calls on 267 of 400
#: seeds. Raising it is queued in `NEXT.md`; a proposed 2,048 was
#: rejected because it was derived from the recorded 200-trial draw
#: rather than from a high-precision equity.
DEFAULT_SERVE_EQUITY_TRIALS = 200


def load_policy(
    manifest_path: str | Path,
    *,
    equity_trials: int | None = None,
    use_learned_sizing: bool | None = None,
    hybrid_min_value_advantage: float | None = None,
    hybrid_max_abs_z: float = 5.0,
    serve_mode: str = "argmax",
    hybrid_min_margin_quantile: str | None = None,
    equity_cache: SharedEquityCache | None = None,
    hyper_aggression_chance: float | None = None,
) -> LearnedPokerPolicy | LearnedPokerPolicyV7:
    """Load, verify, and assemble a playing policy from one manifest.

    Format-1 artifacts return the three-output :class:`LearnedPokerPolicy`;
    format-2 artifacts return :class:`LearnedPokerPolicyV7`. The format-2
    hybrid gate is quantile-based (``hybrid_min_margin_quantile`` naming a
    persisted quantile such as ``"p90"``); ``hybrid_min_value_advantage``
    keeps its raw-threshold meaning for format-1 artifacts only.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LearnedPolicyError(f"cannot read manifest: {error}") from error

    # Explicit argument wins; then the artifact's own pin; then the
    # module default. An artifact that records the value is one whose
    # served precision can be audited from the artifact alone.
    if equity_trials is None:
        pinned = (manifest.get("serve") or {}).get("equity_trials")
        if pinned is None:
            equity_trials = DEFAULT_SERVE_EQUITY_TRIALS
        else:
            _require(
                isinstance(pinned, int)
                and not isinstance(pinned, bool)
                and pinned > 0,
                "serve.equity_trials must be a positive integer",
            )
            equity_trials = int(pinned)
    validate_artifact_manifest(manifest)
    weights_file = manifest_file.parent / str(manifest["weights_file"])
    try:
        raw = weights_file.read_bytes()
    except OSError as error:
        raise LearnedPolicyError(f"cannot read weights: {error}") from error
    digest = hashlib.sha256(raw.rstrip(b"\n")).hexdigest()
    _require(
        digest == manifest["weights_sha256"],
        "weights checksum does not match the manifest",
    )
    document = json.loads(raw.decode("utf-8"))
    _require(
        document.get("model_version") == manifest["model_version"],
        "weights and manifest disagree on model_version",
    )
    normalization = document.get("feature_normalization")
    _require(
        isinstance(normalization, Mapping)
        and "means" in normalization
        and "stds" in normalization,
        "weights file lacks feature normalization",
    )
    prewarm()
    if manifest.get("format_version") == MODEL_FORMAT_VERSION_V7:
        _require(
            hybrid_min_value_advantage is None,
            "format-2 artifacts gate on margin quantiles, not raw advantages",
        )
        serve = manifest.get("serve") or {}
        _require(isinstance(serve, Mapping), "manifest serve block must be an object")
        return LearnedPokerPolicyV7(
            model_version=str(manifest["model_version"]),
            architecture=manifest["architecture"],
            weights=document["weights"],
            normalization=normalization,
            serve=serve,
            equity_trials=equity_trials,
            serve_mode=serve_mode,
            hybrid_min_margin_quantile=hybrid_min_margin_quantile,
            hybrid_max_abs_z=hybrid_max_abs_z,
            equity_cache=equity_cache,
            hyper_aggression_chance=hyper_aggression_chance,
            **_load_engine_parameters(manifest),
        )
    training = manifest.get("training") or {}
    _require(
        isinstance(training, Mapping), "manifest training metadata must be an object"
    )
    action_head_semantics = str(
        training.get("action_head_semantics") or "policy_logits"
    )
    if use_learned_sizing is None:
        use_learned_sizing = action_head_semantics != "counterfactual_value"
    return LearnedPokerPolicy(
        model_version=str(manifest["model_version"]),
        weights=document["weights"],
        normalization=normalization,
        equity_trials=equity_trials,
        action_head_semantics=action_head_semantics,
        use_learned_sizing=use_learned_sizing,
        hybrid_min_value_advantage=hybrid_min_value_advantage,
        hybrid_max_abs_z=hybrid_max_abs_z,
        equity_cache=equity_cache,
        hyper_aggression_chance=hyper_aggression_chance,
        **_load_engine_parameters(manifest),
    )


def approved_fingerprint(artifacts_dir: str | Path) -> str | None:
    """Cheap change token for the approved pointer; None when absent."""

    pointer = Path(artifacts_dir).expanduser() / APPROVED_POINTER
    try:
        raw = pointer.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(raw).hexdigest()


def _load_by_format(
    manifest_file: Path, *, equity_trials: int | None
) -> DecisionEngine:
    """Route an approved manifest to the loader its format requires.

    Each promotable format has its own loader, its own input schema and
    its own branch meanings. A format-4 artifact read by the format-1/2
    loader would meet 414 schema-4 inputs against the 142-input
    contract, so the declared version is dispatched on rather than
    assumed. A version with no loader is refused outright -- falling
    back to the v7 loader is how an artifact of one shape gets served
    under another shape's meanings.
    """

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LearnedPolicyError(
            f"cannot read approved manifest {manifest_file}: {error}"
        ) from error
    if not isinstance(manifest, Mapping):
        raise LearnedPolicyError("approved manifest must be an object")
    version = manifest.get("format_version")
    if version == MODEL_FORMAT_VERSION_V9:
        # ``load_policy`` runs the contract itself on the format-1/2 path,
        # but ``load_policy_v9`` checks only the v9 loader's own
        # invariants -- it has no view of ``serve.deployable``. Run the
        # contract here so the v9 path gets the same serve-time refusal:
        # otherwise a hand-edited pointer naming a Phase-A artifact, whose
        # own trainer stamps it undeployable, would load and play.
        #
        # Re-raised as LearnedPolicyError on purpose: LearningContractError
        # is a sibling of it, not a subclass, and run_agent's between-hands
        # refresh catches only LearnedPolicyError -- an uncaught contract
        # error there would take the live session down instead of keeping
        # the current policy.
        try:
            validate_artifact_manifest(manifest)
        except LearningContractError as error:
            raise LearnedPolicyError(
                f"approved artifact fails the learning contract: {error}"
            ) from error
        # Deferred: the v9 loader pulls in the P3 belief fit, which a
        # format-1/2 serve has no reason to touch, and importing it at
        # module scope would tie this module to the whole v9 stack.
        from engine.learned_policy_v9 import load_policy_v9

        return load_policy_v9(manifest_file, equity_trials=equity_trials)
    if version in {MODEL_FORMAT_VERSION, MODEL_FORMAT_VERSION_V7}:
        return load_policy(manifest_file, equity_trials=equity_trials)
    raise LearnedPolicyError(
        f"approved artifact declares format_version {version!r}, which has "
        "no serve path"
    )


def load_approved(
    artifacts_dir: str | Path, *, equity_trials: int | None = None
) -> DecisionEngine:
    """Follow the approved pointer to a verified playing policy."""

    root = Path(artifacts_dir).expanduser()
    pointer_file = root / APPROVED_POINTER
    try:
        pointer = json.loads(pointer_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LearnedPolicyError(
            f"no readable approved pointer at {pointer_file}: {error}"
        ) from error
    # Pointers written on Windows before 2026-08-14 carry backslashes, which
    # are a filename character on POSIX rather than a separator; normalize so
    # a pointer promoted on one platform loads on the other.
    relative = str(pointer.get("manifest_file") or "").replace("\\", "/")
    manifest_file = root.joinpath(*[part for part in relative.split("/") if part])
    policy = _load_by_format(manifest_file, equity_trials=equity_trials)
    expected = pointer.get("weights_sha256")
    _require(
        expected is None
        or expected
        == json.loads(manifest_file.read_text(encoding="utf-8")).get("weights_sha256"),
        "approved pointer checksum does not match the manifest",
    )
    return policy


__all__ = [
    "APPROVED_POINTER",
    "approved_fingerprint",
    "LearnedPokerPolicy",
    "LearnedPokerPolicyV7",
    "LearnedPolicyError",
    "load_approved",
    "load_policy",
]
