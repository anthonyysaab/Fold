"""Optional PyTorch-backed policy used to check training checkpoints.

This module never performs network I/O. A runner may pass each fresh Arena
table snapshot to :func:`decide`, then submit the returned payload itself.

Game-state validation lives in :mod:`devfun_poker_playground.game_state` and
the safety rails in :mod:`devfun_poker_playground.decision_engine`; this
module adds the PyTorch checkpoint backend. Deployment builds that cannot
ship torch use :class:`devfun_poker_playground.poker_policy.PokerPolicy`, which
shares the same rails over exported weights.

With the current decision engine, a positive ``equity_trials`` value routes to
equity rules before this adapter's logits. Set it to zero only when testing the
legacy checkpoint itself.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from devfun_poker_playground.torch_network import (
    LABELS,
    TinyPolicy,
    mask_illegal_logits,
    validate_checkpoint_contract,
)
from devfun_poker_playground.decision_engine import ArenaAction, DecisionEngine
from devfun_poker_playground.game_state import ArenaSnapshotError, features_from_table

__all__ = [
    "ArenaAction",
    "ArenaSnapshotError",
    "TorchPokerPolicy",
    "decide",
    "features_from_table",
]


class TorchPokerPolicy(DecisionEngine):
    """Tiny neural proposal policy with deterministic Arena safety rails."""

    policy_version = "legacy-torch-v1"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        model: nn.Module | None = None,
        equity_trials: int = 100,
        seed: int = 7,
    ) -> None:
        super().__init__(equity_trials=equity_trials, seed=seed)
        if model is None:
            path = self._checkpoint_path(checkpoint_path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            validate_checkpoint_contract(checkpoint)
            model = TinyPolicy(hidden_size=int(checkpoint["hidden_size"]))
            model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.eval()

    @staticmethod
    def _checkpoint_path(value: str | Path | None) -> Path:
        configured = value or os.environ.get("POKER_POLICY_CHECKPOINT")
        if configured:
            path = Path(configured).expanduser().resolve()
            if path.is_file():
                return path
            raise FileNotFoundError(f"policy checkpoint not found at {path}")

        project_root = Path(__file__).resolve().parents[2]
        candidates = (
            project_root / "artifacts" / "tiny-policy.pt",
            project_root.parent / "poker-nn-training" / "artifacts" / "tiny-policy.pt",
        )
        for path in candidates:
            if path.is_file():
                return path
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "policy checkpoint not found; set POKER_POLICY_CHECKPOINT "
            f"or place it at one of: {checked}"
        )

    def _family(self, features: tuple[float, ...]) -> str:
        batch = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            logits: Tensor = self.model(batch)
            logits = mask_illegal_logits(logits, batch)
        return LABELS[int(logits.argmax(dim=1).item())]


_DEFAULT_POLICY: TorchPokerPolicy | None = None


def decide(
    table: Mapping[str, Any],
    deadline_s: float = 10.0,
    research_context: Mapping[str, Any] | None = None,
) -> dict[str, str | int]:
    """Drop-in strategy function; network registration/submission lives elsewhere."""

    global _DEFAULT_POLICY
    if _DEFAULT_POLICY is None:
        _DEFAULT_POLICY = TorchPokerPolicy()
    return _DEFAULT_POLICY.decide(table, deadline_s, research_context)
