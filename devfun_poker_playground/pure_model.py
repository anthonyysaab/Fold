"""Dependency-free inference over exported tiny-policy weights.

``tools/export_pure_weights.py`` converts the trained PyTorch checkpoint to a
plain JSON document (weights as nested lists plus the full contract
metadata). This module re-validates that contract at load time — the same
gate ``validate_checkpoint_contract`` applies to the torch checkpoint — and
runs the one-hidden-layer forward pass in pure Python. The network is tiny
(125 -> 64 -> 3), so a decision costs well under a millisecond.

The sandbox note: no torch, no numpy, no pickle — JSON in, floats out.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from devfun_poker_playground.contract import (
    FEATURE_NAMES,
    LABELS,
    LEGALITY_FEATURE_INDEXES,
)
from devfun_poker_playground.rules import DecisionRules

_WEIGHTS_ENV = "POKER_PURE_WEIGHTS"
_WEIGHTS_FILENAME = "tiny-policy-pure.json"


class PureWeightsError(ValueError):
    """Raised when an exported weights file violates the policy contract."""


def _matrix(value: object, name: str, rows: int | None, cols: int) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise PureWeightsError(f"{name} must be a non-empty list of rows")
    if rows is not None and len(value) != rows:
        raise PureWeightsError(f"{name} must have {rows} rows, found {len(value)}")
    matrix: list[list[float]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != cols:
            raise PureWeightsError(f"{name}[{index}] must be a list of {cols} numbers")
        matrix.append([float(item) for item in row])
    return matrix


def _vector(value: object, name: str, size: int) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise PureWeightsError(f"{name} must be a list of {size} numbers")
    return [float(item) for item in value]


def validate_pure_weights(document: dict[str, Any]) -> None:
    """Reject weight exports whose contract differs from this package's."""

    if int(document.get("input_size", -1)) != len(FEATURE_NAMES):
        raise PureWeightsError(
            "weights input size does not match the Playground feature contract"
        )
    if tuple(document.get("feature_names", ())) != FEATURE_NAMES:
        raise PureWeightsError(
            "weights feature names do not match the Playground feature contract"
        )
    if tuple(document.get("labels", ())) != LABELS:
        raise PureWeightsError(
            "weights labels do not match the Playground action contract"
        )


class TinyPolicyForward:
    """Pure-Python forward pass for the exported one-hidden-layer network."""

    def __init__(self, document: dict[str, Any]) -> None:
        validate_pure_weights(document)
        hidden_size = int(document.get("hidden_size", 0))
        if hidden_size < 1:
            raise PureWeightsError("hidden_size must be a positive integer")
        self.hidden_size = hidden_size
        self.w1 = _matrix(document.get("w1"), "w1", hidden_size, len(FEATURE_NAMES))
        self.b1 = _vector(document.get("b1"), "b1", hidden_size)
        self.w2 = _matrix(document.get("w2"), "w2", len(LABELS), hidden_size)
        self.b2 = _vector(document.get("b2"), "b2", len(LABELS))

    def logits(self, features: Sequence[float]) -> list[float]:
        if len(features) != len(FEATURE_NAMES):
            raise PureWeightsError(
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

    Mirrors ``model_contract.mask_illegal_logits`` + argmax: illegal families
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


class PurePolicy(DecisionRules):
    """Torch-free policy: exported tiny-policy proposals with the shared rails."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        weights: dict[str, Any] | None = None,
        equity_trials: int = 100,
        seed: int = 7,
    ) -> None:
        super().__init__(equity_trials=equity_trials, seed=seed)
        if weights is None:
            path = self._weights_path(weights_path)
            with open(path, encoding="utf-8") as handle:
                weights = json.load(handle)
        self.forward = TinyPolicyForward(weights)
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
            raise FileNotFoundError(f"pure policy weights not found at {path}")

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
            f"pure policy weights not found; set {_WEIGHTS_ENV} "
            f"or place the export at one of: {checked}"
        )

    def _family(self, features: tuple[float, ...]) -> str:
        return masked_family(self.forward.logits(features), features)

    def _short_handed_family(
        self,
        table,
        allowed,
        available,
        equity,
        features=None,
    ) -> str:
        if features is not None and len(table["seats"]) in self.table_sizes:
            return self._family(features)
        return super()._short_handed_family(
            table, allowed, available, equity, features=features
        )


__all__ = [
    "PurePolicy",
    "PureWeightsError",
    "TinyPolicyForward",
    "masked_family",
    "validate_pure_weights",
]
