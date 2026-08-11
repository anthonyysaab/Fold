"""Inference-only contract shared with the separately trained checkpoint.

The feature/label constants live in the torch-free
:mod:`devfun_poker_playground.contract`; this module keeps the PyTorch
surface (network definition, logits masking, checkpoint validation) and
re-exports the constants for backward compatibility.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from devfun_poker_playground.contract import (
    FEATURE_NAMES,
    LABELS,
    LEGALITY_FEATURE_INDEXES,
)

__all__ = [
    "FEATURE_NAMES",
    "LABELS",
    "TinyPolicy",
    "mask_illegal_logits",
    "validate_checkpoint_contract",
]


class TinyPolicy(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(LABELS)),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


def mask_illegal_logits(logits: Tensor, features: Tensor) -> Tensor:
    legal = features[..., list(LEGALITY_FEATURE_INDEXES)] > 0.5
    if not torch.all(legal.any(dim=-1)):
        raise ValueError("each decision must have at least one legal action")
    return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)


def validate_checkpoint_contract(checkpoint: dict[str, object]) -> None:
    if int(checkpoint.get("input_size", -1)) != len(FEATURE_NAMES):
        raise ValueError("checkpoint input size does not match the Playground feature contract")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature names do not match the Playground feature contract")
    if tuple(checkpoint.get("labels", ())) != LABELS:
        raise ValueError("checkpoint labels do not match the Playground action contract")
