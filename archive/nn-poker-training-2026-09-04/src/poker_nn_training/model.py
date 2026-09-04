from __future__ import annotations

import torch
from torch import Tensor, nn

from poker_nn_training.data import FEATURE_NAMES, LABELS


class TinyPolicy(nn.Module):
    """One hidden layer is enough for the first behavior-cloning baseline."""

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
    """Prevent the policy from selecting an action family the replay state forbids."""

    legal_feature_names = ("legal_fold", "legal_check_call", "legal_aggress")
    legal_columns = [FEATURE_NAMES.index(name) for name in legal_feature_names]
    legal = features[..., legal_columns] > 0.5
    if not torch.all(legal.any(dim=-1)):
        raise ValueError("each decision must have at least one legal action")
    return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
