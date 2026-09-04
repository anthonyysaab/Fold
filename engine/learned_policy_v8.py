"""Stdlib-only loading and serving of a format-3 (v8 composed-value) artifact.

:func:`load_policy_v8` turns a format-3 candidate manifest into a playing
policy: it re-validates the manifest against ``schema3`` and the v8
architecture contract, verifies the weights file's SHA-256, rebuilds any
``engine_parameters`` blocks through their validated dataclasses, and
returns a :class:`LearnedPokerPolicyV8`.

The v8 network never emits an action value directly. Its four component
heads — ``fold_through``, ``range``, ``equity_called``, ``residual`` — are
composed by fixed arithmetic at serve (V8_DESIGN §4):

    V(fold)        = 0                                  (sunk chips are sunk)
    V(check_call)  = Eq_cc · pot' − cost_call
    V(aggress_k)   = P_ft_k · pot_now
                   + (1 − P_ft_k) · (Eq_k · pot'_k − cost_k)
                   [+ residual_k, capped at ±0.05 · pot_now, ablatable]

with every quantity purse-normalized exactly as v7 targets were
(``reward_bb / purse_bb``; the serve-time purse is hero's stack plus
hero's whole-hand commitment). ``pot'`` is the pot after the priced
continuation: ``pot + to_call`` for check/call, and for an aggress branch
``pot + 2·wager_k − to_call`` — hero's wager plus the facing bettor's
matching call (the facing bettor already owns ``to_call`` of hero's price,
so its call price is ``wager_k − to_call``). Wagers and costs use the same
snapshot arithmetic as ``feature_extract_v8._branch_block``: the E6 branch
target ``min(to_call + pot_fraction·(pot + to_call), stack_fraction·
effective_stack)``, its to-amount clamped into the stated legal bet/raise
range, with the same two documented deviations (no big-blind minimum
increment, no integer rounding — the engine reapplies its own clamps when
the action executes). The ``range`` head is not read by the composition;
it exists for calibration audits.

Branch emission mirrors the branch-executability features (the E6 rule "a
branch is emitted only where the engine can execute it and where it is
distinct"): no stated bet/raise range emits no aggress branch, and the
large branch is absorbed whenever its clamped to-amount collides with the
small branch's. The residual head's four outputs align with
``BRANCH_LABELS_V8``; per the §4 composition only the two aggress entries
are ever read on the serve path.

Choosing an aggress branch pins the engine's pot fraction to the branch's
**state-dependent** E6 conversion (``(target − call) / (pot + call)``, the
v7 pinning mechanism made stack-aware): when the pot arm binds this is
byte-identical to v7's half-pot/full-pot, and when the stack arm binds the
fraction shrinks. The pinned fraction is computed from the unclamped
target — ``DecisionEngine._sized_action`` applies the legal min/max, the
big-blind minimum, and the risk cap exactly as today, and every other hard
gate, the temperature shaping, and the bluff path stay untouched: the
composed argmax only proposes a family inside legality; the engine
disposes.

Failure posture: a malformed artifact fails loud at load
(:class:`LearnedPolicyV8Error`); a failure on the serve path — feature
extraction, forward pass, or composition raising, or any non-finite
composed value — falls back closed to the heuristic
``DecisionEngine._equity_family``, exactly as the v7 runtime does.

Determinism: the feature extractor runs on a fixed seed, the forward pass
and composition are pure arithmetic, and the branch argmax breaks ties
toward the earlier branch in ``BRANCH_LABELS_V8`` order. Identical
snapshots give identical decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bluff import BluffSettings

from engine import schema3
from engine.belief_provider import BeliefProvider
from engine.decision_engine import (
    DecisionEngine,
    SafetyGates,
    SharedEquityCache,
    TemperatureShaping,
)
from engine.feature_extract_v8 import (
    _aggressive_range,
    _BRANCH_LARGE,
    _BRANCH_SMALL,
    extract_features_v8,
)
from engine.game_state import (
    _hero_and_seats,
    _integer,
    effective_stack_chips,
)
from engine.hand_strength import prewarm
from engine.learned_policy import (
    LearnedPolicyError,
    _load_engine_parameters,
    _matrix,
    _vector,
)
from engine.learning_contract import MODEL_FORMAT
from engine.forward_kernel import _dot, _layer_norm, _sigmoid
from engine.opponent_model import AggressionTracker
from engine.architecture_v8 import (
    BRANCH_LABELS_V8,
    EQUITY_SLOTS,
    FOLD_THROUGH_BRANCHES,
    MODEL_FORMAT_VERSION_V8,
    V8_HEAD_SIZES,
    validate_v8_architecture,
)

# The §4 residual cap: the correction may move a branch value by at most
# this fraction of the (purse-normalized) current pot. Ablatable (§6) via
# the constructor, never silently.
RESIDUAL_CAP_POT_FRACTION = 0.05

_AGGRESS_BRANCHES = ("aggress_small", "aggress_large")
_BRANCH_SPECS = {
    "aggress_small": _BRANCH_SMALL,
    "aggress_large": _BRANCH_LARGE,
}


class LearnedPolicyV8Error(LearnedPolicyError):
    """Raised when a format-3 artifact cannot be loaded safely.

    Subclasses :class:`LearnedPolicyError` so existing fail-closed callers
    that catch the v7 error keep catching this one.
    """


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LearnedPolicyV8Error(message)


def _forward_v3(
    architecture: Mapping[str, object],
    weights: Mapping[str, object],
    features: Sequence[float],
) -> dict[str, list[float]]:
    """Pure-Python inference for a format-3 artifact (evaluation mode).

    Mirrors ``offline_trainer._forward_v2``'s structure for the v8 shapes
    and must stay numerically aligned with the v8 trainer's
    ``_NetworkV8.forward`` in eval mode: the card and context encoders
    (over ``architecture['card_indices']`` / ``['context_indices']``) are
    linear -> ReLU -> LayerNorm; every trunk layer is linear -> ReLU with
    LayerNorm after all but the last; each head tower is linear -> ReLU ->
    linear. Dropout is train-only and does not appear here.

    Returns the raw head outputs: ``fold_through`` logits, ``range``
    logits, ``equity_called`` values, ``residual`` values. Probability
    transforms (sigmoid/softmax) belong to the consumer.
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
    for name in V8_HEAD_SIZES:
        block = head_blocks[name]
        tower = [
            value if value > 0.0 else 0.0
            for value in linear({"w": block["tower_w"], "b": block["tower_b"]}, hidden)
        ]
        outputs[name] = linear({"w": block["out_w"], "b": block["out_b"]}, tower)
    return outputs


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def compose_branch_values(
    head_outputs: Mapping[str, Sequence[float]],
    *,
    pot: int,
    to_call: int,
    contribution: int,
    effective_stack: int,
    purse: int,
    legal_range: tuple[int, int] | None,
    use_residual: bool = True,
    residual_cap_pot_fraction: float = RESIDUAL_CAP_POT_FRACTION,
) -> tuple[dict[str, float], dict[str, float]]:
    """The §4 fixed-arithmetic composition over the emitted branch set.

    Returns ``(values, pot_fractions)``: composed (uncentered) values per
    emitted branch in ``BRANCH_LABELS_V8`` order, and the state-dependent
    E6 pot fraction per emitted aggress branch (from the *unclamped*
    branch target, so the engine's own sizing path realizes it with every
    clamp intact). ``fold`` and ``check_call`` are always emitted;
    legality is the caller's mask, not this function's.

    All quantities are purse-normalized (chips / purse). Equities are
    clipped into [0, 1] before use — they are probabilities and the
    regression head's raw output may stray a hair outside. Fold-through
    logits go through the sigmoid. The residual correction (aggress
    entries of the 4-wide head, ``BRANCH_LABELS_V8``-aligned) is capped at
    ``±residual_cap_pot_fraction · pot`` in the same purse units and can
    be ablated off entirely.
    """

    eff = max(1, int(effective_stack))
    purse_chips = max(1, int(purse))
    fold_through = head_outputs["fold_through"]
    equity_called = head_outputs["equity_called"]
    residual = head_outputs["residual"]

    pot_unit = pot / purse_chips
    equity_cc = _clip01(float(equity_called[EQUITY_SLOTS.index("check_call")]))
    values: dict[str, float] = {
        "fold": 0.0,
        "check_call": (
            equity_cc * (pot + to_call) / purse_chips - to_call / purse_chips
        ),
    }
    pot_fractions: dict[str, float] = {}
    if legal_range is None:
        return values, pot_fractions

    low, high = legal_range
    cap = abs(float(residual_cap_pot_fraction)) * pot_unit
    targets: dict[str, float] = {}
    to_amounts: dict[str, float] = {}
    for branch in _AGGRESS_BRANCHES:
        pot_fraction, stack_fraction = _BRANCH_SPECS[branch]
        target = min(
            to_call + pot_fraction * (pot + to_call),
            stack_fraction * eff,
        )
        targets[branch] = target
        to_amounts[branch] = min(high, max(low, contribution + target))

    emitted = ["aggress_small"]
    if to_amounts["aggress_large"] != to_amounts["aggress_small"]:
        emitted.append("aggress_large")
    for branch in emitted:
        p_fold_through = _sigmoid(
            float(fold_through[FOLD_THROUGH_BRANCHES.index(branch)])
        )
        equity_k = _clip01(float(equity_called[EQUITY_SLOTS.index(branch)]))
        wager = max(0.0, to_amounts[branch] - contribution)
        pot_if_called = pot + 2.0 * wager - to_call
        value = p_fold_through * pot_unit + (1.0 - p_fold_through) * (
            equity_k * pot_if_called / purse_chips - wager / purse_chips
        )
        if use_residual:
            correction = float(residual[BRANCH_LABELS_V8.index(branch)])
            value += min(cap, max(-cap, correction))
        values[branch] = value
        pot_fractions[branch] = (targets[branch] - to_call) / max(
            1, pot + to_call
        )
    return values, pot_fractions


def _validate_v8_weights(
    weights: Mapping[str, Any], architecture: Mapping[str, Any]
) -> dict[str, Any]:
    """Structural validation of a format-3 weights block, fail-closed."""

    card_count = len(architecture["card_indices"])
    context_count = len(architecture["context_indices"])
    card_width = int(architecture["card_encoder_width"])
    context_width = int(architecture["context_encoder_width"])
    trunk_widths = [int(value) for value in architecture["trunk_widths"]]
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

    built: dict[str, Any] = {
        "card_encoder": with_norm(
            weights.get("card_encoder"), "card_encoder", card_width, card_count
        ),
        "context_encoder": with_norm(
            weights.get("context_encoder"),
            "context_encoder",
            context_width,
            context_count,
        ),
    }
    trunk_blocks = weights.get("trunk")
    _require(
        isinstance(trunk_blocks, list) and len(trunk_blocks) == len(trunk_widths),
        "trunk must list one block per trunk layer",
    )
    previous = card_width + context_width
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
    _require(
        set(head_blocks) == set(V8_HEAD_SIZES),
        f"heads must be exactly {sorted(V8_HEAD_SIZES)}",
    )
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
            "out_w": _matrix(
                block.get("out_w"), f"heads.{name}.out_w", int(size), tower
            ),
            "out_b": _vector(block.get("out_b"), f"heads.{name}.out_b", int(size)),
        }
    built["heads"] = built_heads
    return built


class LearnedPokerPolicyV8(DecisionEngine):
    """Format-3 runtime: composed component values inside the rails.

    The network's heads are components, never action values: the fixed §4
    arithmetic composes them per branch, the argmax runs inside legality
    and branch emission, and an aggress choice pins the engine's pot
    fraction to the branch's state-dependent E6 size. Every hard safety
    gate, the temperature shaping, the opponent tracker, and the bluff
    path stay exactly as in the heuristic engine, which also remains the
    only fallback.
    """

    def __init__(
        self,
        *,
        model_version: str,
        architecture: Mapping[str, Any],
        weights: Mapping[str, Any],
        normalization: Mapping[str, Sequence[float]],
        serve: Mapping[str, Any] | None = None,
        equity_trials: int = 200,
        seed: int = 7,
        temperature_shaping: TemperatureShaping | None = None,
        safety_gates: SafetyGates | None = None,
        opponent_tracker: AggressionTracker | None = None,
        bluff_settings: BluffSettings | None = None,
        equity_cache: SharedEquityCache | None = None,
        hyper_aggression_chance: float | None = None,
        belief_provider: BeliefProvider | None = None,
        potential_trials: int = 400,
        feature_seed: int = 7,
        use_residual: bool = True,
        residual_cap_pot_fraction: float = RESIDUAL_CAP_POT_FRACTION,
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
        try:
            validate_v8_architecture(architecture)
        except ValueError as error:
            raise LearnedPolicyV8Error(f"invalid v8 architecture: {error}") from error
        self._architecture = dict(architecture)
        self._weights = _validate_v8_weights(weights, architecture)
        self._means = [float(value) for value in normalization["means"]]
        self._stds = [max(1e-6, float(value)) for value in normalization["stds"]]
        _require(
            len(self._means) == schema3.INPUT_SIZE_V8
            and len(self._stds) == schema3.INPUT_SIZE_V8,
            "feature normalization must cover the full schema-3 vector",
        )
        guard = (serve or {}).get("ood_guard_indices") or ()
        _require(
            all(
                isinstance(index, int) and 0 <= index < schema3.INPUT_SIZE_V8
                for index in guard
            ),
            "ood_guard_indices must be valid schema-3 feature indices",
        )
        self._guard_indices = tuple(guard)
        _require(
            math.isfinite(float(residual_cap_pot_fraction))
            and float(residual_cap_pot_fraction) >= 0.0,
            "residual_cap_pot_fraction must be finite and non-negative",
        )
        _require(potential_trials >= 1, "potential_trials must be positive")
        self._belief_provider = belief_provider
        self._potential_trials = int(potential_trials)
        self._feature_seed = int(feature_seed)
        self._use_residual = bool(use_residual)
        self._residual_cap_pot_fraction = float(residual_cap_pot_fraction)
        self._branch_pot_fraction: float | None = None

    def _family(self, features: tuple[float, ...]) -> str:
        # Without an equity read there is no feature vector to serve from,
        # so the safe passive family stands in (the v7 rule, unchanged).
        del features
        return "check_call"

    def _composed_decision(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Extract, forward, compose. Raises on any malformed input."""

        vector = extract_features_v8(
            table,
            equity=equity,
            belief_provider=self._belief_provider,
            potential_trials=self._potential_trials,
            seed=self._feature_seed,
        )
        normalized = tuple(
            (value - mean) / std
            for value, mean, std in zip(vector, self._means, self._stds)
        )
        outputs = _forward_v3(self._architecture, self._weights, normalized)
        hero, _ = _hero_and_seats(table)
        pot = _integer(table.get("potChips"), "potChips")
        to_call = _integer(allowed.get("callChips", 0), "callChips")
        contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
        stack = _integer(hero.get("stackChips"), "hero stackChips")
        total_committed = _integer(
            hero.get("totalCommittedChips", contribution), "hero totalCommittedChips"
        )
        values, pot_fractions = compose_branch_values(
            outputs,
            pot=pot,
            to_call=to_call,
            contribution=contribution,
            effective_stack=effective_stack_chips(table),
            purse=stack + total_committed,
            legal_range=_aggressive_range(allowed, available),
            use_residual=self._use_residual,
            residual_cap_pot_fraction=self._residual_cap_pot_fraction,
        )
        if not all(math.isfinite(value) for value in values.values()):
            raise LearnedPolicyV8Error("composed branch values are not finite")
        return values, pot_fractions

    def _legal_branches(self, values: Mapping[str, float], available: set[str]):
        branches = []
        for branch in BRANCH_LABELS_V8:
            if branch not in values:
                continue  # not emitted for this state
            if branch == "fold" and "fold" in available:
                branches.append(branch)
            elif branch == "check_call" and (
                "check" in available or "call" in available
            ):
                branches.append(branch)
            elif branch in _AGGRESS_BRANCHES and (
                "bet" in available or "raise" in available
            ):
                branches.append(branch)
        return branches

    def _equity_family(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float,
        features: tuple[float, ...] | None = None,
    ) -> str:
        self._branch_pot_fraction = None
        if features is None or equity is None:
            return super()._equity_family(table, allowed, available, equity)
        try:
            values, pot_fractions = self._composed_decision(
                table, allowed, available, equity
            )
        except Exception:
            # Fail closed: any malformed snapshot, feature, or artifact
            # behavior degrades to the heuristic engine, never to a guess.
            return super()._equity_family(table, allowed, available, equity)
        branches = self._legal_branches(values, available)
        if not branches:
            return super()._equity_family(table, allowed, available, equity)
        # Centered within the decision (the v7 rule survives): centering
        # never moves an argmax, but the values diagnostics read should be
        # the values the objective defines.
        offset = sum(values[branch] for branch in branches) / len(branches)
        centered = {branch: values[branch] - offset for branch in branches}
        best = max(branches, key=lambda branch: centered[branch])
        if best in _AGGRESS_BRANCHES:
            self._branch_pot_fraction = pot_fractions[best]
            return "aggress"
        return best

    def _sized_action(
        self,
        action: str,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float | None,
        pot_fraction: float | None = None,
    ):
        if pot_fraction is None and self._branch_pot_fraction is not None:
            pot_fraction = self._branch_pot_fraction
        return super()._sized_action(
            action, table, allowed, equity, pot_fraction=pot_fraction
        )


_REQUIRED_MANIFEST_KEYS = (
    "format",
    "format_version",
    "model_version",
    "feature_schema_version",
    "input_size",
    "feature_names",
    "action_labels",
    "architecture",
    "weights_file",
    "weights_sha256",
)


def load_policy_v8(
    manifest_path: str | Path,
    *,
    equity_trials: int = 200,
    equity_cache: SharedEquityCache | None = None,
    hyper_aggression_chance: float | None = None,
    belief_provider: BeliefProvider | None = None,
    potential_trials: int = 400,
    feature_seed: int = 7,
    use_residual: bool = True,
    residual_cap_pot_fraction: float = RESIDUAL_CAP_POT_FRACTION,
) -> LearnedPokerPolicyV8:
    """Load, verify, and assemble a v8 playing policy from one manifest.

    Fail-loud validation before anything is served: format and
    format_version 3, feature schema 3 with the exact ``schema3`` name
    list and input size, the v8 action labels, the architecture contract,
    and the weights file's SHA-256 against the manifest. Engine parameter
    blocks are rebuilt through their validated dataclasses exactly as the
    v7 loader rebuilds them. Loading neither promotes nor deploys
    anything: it constructs an object, and ``artifacts/approved.json`` is
    never read or written here.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LearnedPolicyV8Error(f"cannot read manifest: {error}") from error
    _require(isinstance(manifest, Mapping), "manifest must be an object")
    for key in _REQUIRED_MANIFEST_KEYS:
        _require(key in manifest, f"manifest is missing {key!r}")
    _require(
        manifest["format"] == MODEL_FORMAT,
        f"manifest format must be {MODEL_FORMAT!r}",
    )
    _require(
        manifest["format_version"] == MODEL_FORMAT_VERSION_V8,
        f"format_version must be {MODEL_FORMAT_VERSION_V8}",
    )
    _require(
        manifest["feature_schema_version"] == schema3.SCHEMA_VERSION,
        f"feature_schema_version must be {schema3.SCHEMA_VERSION}",
    )
    _require(
        manifest["input_size"] == schema3.INPUT_SIZE_V8,
        f"input_size must be {schema3.INPUT_SIZE_V8}",
    )
    _require(
        list(manifest["feature_names"]) == list(schema3.FEATURE_NAMES_V8),
        "feature_names must match schema3.FEATURE_NAMES_V8",
    )
    _require(
        list(manifest["action_labels"]) == list(BRANCH_LABELS_V8),
        f"action_labels must be {list(BRANCH_LABELS_V8)}",
    )
    try:
        validate_v8_architecture(manifest["architecture"])
    except ValueError as error:
        raise LearnedPolicyV8Error(f"invalid v8 architecture: {error}") from error

    weights_file = manifest_file.parent / str(manifest["weights_file"])
    try:
        raw = weights_file.read_bytes()
    except OSError as error:
        raise LearnedPolicyV8Error(f"cannot read weights: {error}") from error
    digest = hashlib.sha256(raw.rstrip(b"\n")).hexdigest()
    _require(
        digest == manifest["weights_sha256"],
        "weights checksum does not match the manifest",
    )
    try:
        document = json.loads(raw.decode("utf-8"))
    except ValueError as error:
        raise LearnedPolicyV8Error(f"cannot parse weights: {error}") from error
    _require(
        document.get("format_version") == MODEL_FORMAT_VERSION_V8,
        "weights file format_version must match the manifest",
    )
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
    serve = manifest.get("serve") or {}
    _require(isinstance(serve, Mapping), "manifest serve block must be an object")
    prewarm()
    return LearnedPokerPolicyV8(
        model_version=str(manifest["model_version"]),
        architecture=manifest["architecture"],
        weights=document["weights"],
        normalization=normalization,
        serve=serve,
        equity_trials=equity_trials,
        equity_cache=equity_cache,
        hyper_aggression_chance=hyper_aggression_chance,
        belief_provider=belief_provider,
        potential_trials=potential_trials,
        feature_seed=feature_seed,
        use_residual=use_residual,
        residual_cap_pot_fraction=residual_cap_pot_fraction,
        **_load_engine_parameters(manifest),
    )


__all__ = [
    "LearnedPokerPolicyV8",
    "LearnedPolicyV8Error",
    "RESIDUAL_CAP_POT_FRACTION",
    "compose_branch_values",
    "load_policy_v8",
]
