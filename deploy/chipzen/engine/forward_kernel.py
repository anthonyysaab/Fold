"""Pure-Python forward-pass primitives shared by serving and training.

These six functions are the numeric kernel of the torch-free serve path. They
lived in `offline_trainer.py` until 2026-09-04, which made `engine/` look
inseparable from the trainers: `learned_policy.py`, `learned_policy_v8.py` and
`learned_policy_v9.py` all imported them, so the serving path formally depended
on a module named "trainer" for arithmetic that has nothing to do with
training. Moving them here is what let the training package leave `engine/`.

Nothing here imports from `engine` -- only `math` and `operator`. That is the
property to preserve: this module sits below every other engine module in the
import graph, so it can never participate in a cycle.

`_forward` and `_forward_v2` must stay numerically aligned with their torch
counterparts in the training package; `tests/test_learned_policy_v8.py` pins
`_forward_v3` against the real network on the CUDA interpreter, and the v7/v8
equivalents are pinned against hand-computed fixtures. Changing the
accumulation order here is a serving change, not a cleanup.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence


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


__all__ = [
    "_dot",
    "_forward",
    "_forward_v2",
    "_layer_norm",
    "_sigmoid",
    "_softmax",
]
