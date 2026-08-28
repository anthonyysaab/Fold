"""The live poker policy and its fixed-weight action proposal network.

The live agent uses this module as its single policy entry point. It can load a
small JSON proposal network and delegates final legal actions and sizing to the
decision engine. With the shipped artifact, deterministic equity rules select
the action family. Live play needs no PyTorch.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from engine.policy_features import (
    FEATURE_NAMES,
    LABELS,
    LEGALITY_FEATURE_INDEXES,
)
from engine.decision_engine import (
    BluffSettings,
    DecisionEngine,
    DEFAULT_SAFETY_GATES,
    SafetyGates,
    SharedEquityCache,
    TemperatureShaping,
)
from engine.hand_strength import prewarm
from engine.opponent_model import AggressionTracker

_WEIGHTS_ENV = "POKER_POLICY_WEIGHTS"
_WEIGHTS_FILENAME = "tiny-policy-pure.json"


class PolicyWeightsError(ValueError):
    """Raised when an exported weights file violates the policy contract."""


def _matrix(value: object, name: str, rows: int | None, cols: int) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise PolicyWeightsError(f"{name} must be a non-empty list of rows")
    if rows is not None and len(value) != rows:
        raise PolicyWeightsError(f"{name} must have {rows} rows, found {len(value)}")
    matrix: list[list[float]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != cols:
            raise PolicyWeightsError(
                f"{name}[{index}] must be a list of {cols} numbers"
            )
        matrix.append([float(item) for item in row])
    return matrix


def _vector(value: object, name: str, size: int) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise PolicyWeightsError(f"{name} must be a list of {size} numbers")
    return [float(item) for item in value]


def validate_policy_weights(document: dict[str, Any]) -> None:
    """Reject weight exports whose contract differs from this package's."""

    if int(document.get("input_size", -1)) != len(FEATURE_NAMES):
        raise PolicyWeightsError(
            "weights input size does not match the Playground feature contract"
        )
    if tuple(document.get("feature_names", ())) != FEATURE_NAMES:
        raise PolicyWeightsError(
            "weights feature names do not match the Playground feature contract"
        )
    if tuple(document.get("labels", ())) != LABELS:
        raise PolicyWeightsError(
            "weights labels do not match the Playground action contract"
        )


class FixedPolicyNetwork:
    """Pure-Python forward pass for the exported one-hidden-layer network."""

    def __init__(self, document: dict[str, Any]) -> None:
        validate_policy_weights(document)
        hidden_size = int(document.get("hidden_size", 0))
        if hidden_size < 1:
            raise PolicyWeightsError("hidden_size must be a positive integer")
        self.hidden_size = hidden_size
        self.w1 = _matrix(document.get("w1"), "w1", hidden_size, len(FEATURE_NAMES))
        self.b1 = _vector(document.get("b1"), "b1", hidden_size)
        self.w2 = _matrix(document.get("w2"), "w2", len(LABELS), hidden_size)
        self.b2 = _vector(document.get("b2"), "b2", len(LABELS))

    def __deepcopy__(self, memo: dict) -> FixedPolicyNetwork:
        """Share the weights instead of copying them.

        Counterfactual replay deep-copies the seat list, and each seat
        carries its agent, so a generic copy clones every weight for every
        rollout: this network is ~8,300 floats and dominated harvest time.
        Sharing is exact rather than approximate because the weights are
        written once in ``__init__`` and ``logits`` only reads them, so no
        replay can observe or mutate another's copy.
        """

        memo[id(self)] = self
        return self

    def logits(self, features: Sequence[float]) -> list[float]:
        if len(features) != len(FEATURE_NAMES):
            raise PolicyWeightsError(
                f"expected {len(FEATURE_NAMES)} features, received {len(features)}"
            )
        hidden = [
            max(0.0, sum(w * x for w, x in zip(row, features)) + bias)
            for row, bias in zip(self.w1, self.b1)
        ]
        return [
            sum(w * h for w, h in zip(row, hidden)) + bias
            for row, bias in zip(self.w2, self.b2)
        ]


def masked_family(logits: Sequence[float], features: Sequence[float]) -> str:
    """Pick the highest-scoring family among the legal ones.

    Mirrors ``torch_network.mask_illegal_logits`` plus argmax: illegal families
    are excluded outright, and at least one family must be legal.
    """

    legal = [features[index] > 0.5 for index in LEGALITY_FEATURE_INDEXES]
    if not any(legal):
        raise ValueError("each decision must have at least one legal action")
    best_index: int | None = None
    for index, is_legal in enumerate(legal):
        if not is_legal:
            continue
        if best_index is None or logits[index] > logits[best_index]:
            best_index = index
    assert best_index is not None
    return LABELS[best_index]


class PokerPolicy(DecisionEngine):
    """Torch-free standard policy with optional exported-network proposals."""

    policy_version = "heuristic-standard-v5"

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        weights: dict[str, Any] | None = None,
        equity_trials: int = 100,
        seed: int = 7,
        temperature_shaping: TemperatureShaping | None = None,
        safety_gates: SafetyGates | None = None,
        opponent_tracker: AggressionTracker | None = None,
        bluff_settings: BluffSettings | None = None,
        hyper_aggression_chance: float | None = None,
        equity_cache: SharedEquityCache | None = None,
    ) -> None:
        super().__init__(
            equity_trials=equity_trials,
            seed=seed,
            temperature_shaping=temperature_shaping,
            safety_gates=safety_gates,
            opponent_tracker=opponent_tracker,
            bluff_settings=bluff_settings,
            hyper_aggression_chance=hyper_aggression_chance,
            equity_cache=equity_cache,
        )
        if weights is None:
            path = self._weights_path(weights_path)
            with open(path, encoding="utf-8") as handle:
                weights = json.load(handle)
        self.forward = FixedPolicyNetwork(weights)
        # Table sizes whose short-handed decisions the network itself drives
        # (self-play-trained checkpoints declare these). Absent or empty
        # means the deterministic equity thresholds keep short-handed play.
        raw_sizes = weights.get("table_sizes") or ()
        self.table_sizes = frozenset(
            int(size) for size in raw_sizes if isinstance(size, (int, float))
        )

    @staticmethod
    def _weights_path(value: str | Path | None) -> Path:
        configured = value or os.environ.get(_WEIGHTS_ENV)
        if configured:
            path = Path(configured).expanduser().resolve()
            if path.is_file():
                return path
            raise FileNotFoundError(f"policy weights not found at {path}")

        module_path = Path(__file__).resolve()
        candidates = (
            module_path.parents[2] / "artifacts" / _WEIGHTS_FILENAME,
            module_path.parents[1] / "artifacts" / _WEIGHTS_FILENAME,
        )
        for path in candidates:
            if path.is_file():
                return path
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"policy weights not found; set {_WEIGHTS_ENV} "
            f"or place the export at one of: {checked}"
        )

    def _family(self, features: tuple[float, ...]) -> str:
        return masked_family(self.forward.logits(features), features)

    def _equity_family(
        self,
        table,
        allowed,
        available,
        equity,
        features=None,
    ) -> str:
        if features is not None and len(table["seats"]) in self.table_sizes:
            return self._family(features)
        return super()._equity_family(
            table, allowed, available, equity, features=features
        )


AGGRESSIVE_SETTINGS = {
    "aggression_base": 0.42,
    "aggression_per_opponent": 0.03,
    "aggression_cap": 0.66,
    "call_margin": {"preflop": -0.02, "flop": 0.0, "turn": 0.02, "river": 0.04},
}

# The aggressive policy keeps a single, later-triggering stack-off gate.
# Softened 30% on 2026-08-12 (trigger x1.30, floor keeps 70% of its excess
# over 0.50); was (0.5, 0.62). Like every SafetyGates field, a future
# learned artifact may replace it.
AGGRESSIVE_SAFETY_GATES = replace(
    DEFAULT_SAFETY_GATES, call_stack_gates=((0.65, 0.584),)
)


class AggressivePokerPolicy(PokerPolicy):
    """Current higher-volume policy option for two-to-six-player tables."""

    # v6 = the escalation-decayed opponent range floor (the 2026-08-13 bust
    # fix). Shipping the change under a new version string, never by
    # mutating v5 in place, is what keeps heuristic-aggressive-v5 the
    # frozen, comparable baseline in the archived gauntlet reports
    # (DECISIONS.md).
    policy_version = "heuristic-aggressive-v6"
    default_safety_gates = AGGRESSIVE_SAFETY_GATES

    def _aggression_floor(self, table, opponent_count):
        settings = AGGRESSIVE_SETTINGS
        return min(
            settings["aggression_cap"],
            settings["aggression_base"]
            + settings["aggression_per_opponent"] * max(0, opponent_count - 1),
        )

    def _call_margin(self, table):
        return AGGRESSIVE_SETTINGS["call_margin"].get(self._street(table), 0.04)


def build_policy(
    *,
    aggressive: bool = False,
    weights_path: str | Path | None = None,
    equity_trials: int = 200,
    equity_cache: SharedEquityCache | None = None,
    hyper_aggression_chance: float | None = None,
) -> PokerPolicy:
    """Prepare hand evaluation and build the selected live policy.

    ``equity_cache`` opts into memoized equity estimates (see
    :class:`DecisionEngine`); the harvest passes a fresh dict per policy,
    live construction leaves it ``None``.

    ``hyper_aggression_chance`` overrides the anti-modeling floor; ``None``
    keeps ``HYPER_AGGRESSION_CHANCE``. The harvest passes ``0.0`` because its
    opponents cannot model anyone, live construction leaves it ``None``.
    """

    prewarm()
    policy_type = AggressivePokerPolicy if aggressive else PokerPolicy
    return policy_type(
        weights_path=weights_path,
        equity_trials=equity_trials,
        equity_cache=equity_cache,
        hyper_aggression_chance=hyper_aggression_chance,
    )


__all__ = [
    "AGGRESSIVE_SAFETY_GATES",
    "AGGRESSIVE_SETTINGS",
    "AggressivePokerPolicy",
    "build_policy",
    "PokerPolicy",
    "PolicyWeightsError",
    "FixedPolicyNetwork",
    "masked_family",
    "validate_policy_weights",
]
