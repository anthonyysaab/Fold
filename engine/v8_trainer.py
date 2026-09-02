"""Phase-A supervised trainer for the v8 composed-value network.

V8_DESIGN.md §4 (amended) / §5 Phase A: train the three independently
supervisable component heads — ``fold_through``, ``range``, and
``equity_called`` — on the complete-information Phase-A dataset built by
``tools.build_phase_a_dataset``. The ``residual`` head is exported
zero-initialized and receives **zero gradient** in this phase: it belongs
to the composed-value objective (Phase B), not to supervised component
fitting.

Architecture (schema 3 is normative for the input partition):

- card encoder 208 -> 64, context encoder 205 -> 48, each
  linear -> ReLU -> LayerNorm; the card block is **not** z-scored
  (identity scales are stored), the context block is z-scored from the
  training split only with the retained 0.05 std floor.
- trunk 112 -> 128 -> 128, LayerNorm + dropout after every trunk layer
  except the last (linear -> ReLU only), exactly the v7 discipline.
- four towers, 32 wide: fold_through -> 2, range -> 8 (softmax),
  equity_called -> 3, residual -> 4 (zero-initialized output).

Optimization copies the measured v7 recipe verbatim: AdamW with
decoupled weight decay excluding biases and LayerNorm, He init with
zero-initialized output heads, dropout train-only, global-norm gradient
clip 1.0, linear warmup then cosine decay, early stopping with
best-checkpoint restore on total validation loss, and non-finite
gradients/parameters failing closed.

Phase-A losses are masked by the dataset's per-row label masks:

- ``fold_through``: BCE per branch; a row supervises only the branch its
  realized size matched (mask from the dataset).
- ``range``: cross-entropy over the 8 strength-percentile octiles.
- ``equity_called``: MSE against the exact/MC equity label, routed to
  the slot of the realized action — the taken aggress branch for sized
  aggressions, the ``check_call`` slot for every non-aggress row (the
  continuing set was observed at the existing price without hero
  aggression; folds share that conditional).

Torch imports are function-local (``offline_trainer.py``'s pattern): the
module imports cleanly on the stdlib-only interpreter, and training runs
in the CUDA venv. This trainer writes immutable candidate artifacts only
— it never promotes, never touches ``artifacts/approved.json``, and its
manifest validator accepts no state but ``"candidate"``.

Usage (CUDA venv):
    python -m engine.v8_trainer \
        --model-version candidate-v8-0001 --init-seeds 101 202 303
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bluff import DEFAULT_BLUFF_SETTINGS

from engine import schema3
from engine.decision_engine import (
    DEFAULT_SAFETY_GATES,
    DEFAULT_TEMPERATURE_SHAPING,
)
from engine.feature_extract_v8 import _BRANCH_LARGE, _BRANCH_SMALL
from engine.learning_contract import MODEL_FORMAT
from engine.offline_trainer import (
    _assert_finite_weights,
    _round9,
    validate_training_device,
)
from engine.opponent_model import DEFAULT_TRACKER_SETTINGS

MODEL_FORMAT_VERSION_V8 = 3
MODEL_FAMILY_V8 = "v8-composed-value"
TRAINING_OBJECTIVE_V8_PHASE_A = "phase_a_supervised_component_heads_v8"

CARD_ENCODER_WIDTH = 64
CONTEXT_ENCODER_WIDTH = 48
TRUNK_WIDTHS = (128, 128)
HEAD_TOWER_WIDTH = 32
V8_HEAD_SIZES: dict[str, int] = {
    "fold_through": 2,
    "range": schema3.BELIEF_BUCKETS,
    "equity_called": 3,
    "residual": 4,
}
SUPERVISED_HEADS = ("fold_through", "range", "equity_called")
BRANCH_LABELS_V8 = ("fold", "check_call", "aggress_small", "aggress_large")
FOLD_THROUGH_BRANCHES = ("aggress_small", "aggress_large")
EQUITY_SLOTS = ("aggress_small", "aggress_large", "check_call")
CONTEXT_STD_FLOOR = 0.05

_FT_LABEL_NAMES = ("fold_through_small", "fold_through_large")


@dataclass(frozen=True, slots=True)
class PhaseARow:
    """One validated Phase-A supervision row."""

    table_id: str
    street: str
    features: tuple[float, ...]
    fold_through_label: float
    fold_through_mask: tuple[int, int]  # (aggress_small, aggress_large)
    range_bucket: int
    range_mask: int
    equity_called: float
    equity_mask: int
    equity_slot: int  # index into EQUITY_SLOTS


@dataclass(frozen=True, slots=True)
class V8TrainingConfig:
    epochs: int = 150
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    dropout: float = 0.2
    warmup_steps: int = 200
    early_stop_patience: int = 10
    batch_size: int = 256
    validation_fraction: float = 0.1
    split_seed: int = 17
    init_seed: int = 101
    device: str = "cuda"
    model_version: str | None = None


@dataclass(frozen=True, slots=True)
class V8TrainingSummary:
    rows: int
    train_rows: int
    validation_rows: int
    selected_init_seed: int
    train_losses: Mapping[str, float | None]
    validation_losses: Mapping[str, float | None]
    seed_runs: tuple[Mapping[str, object], ...]
    calibration: Mapping[str, object]
    weights_sha256: str
    weights_path: Path
    manifest_path: Path
    wall_time_seconds: float


def check_v8_config(config: V8TrainingConfig) -> None:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative and finite")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if config.early_stop_patience < 1:
        raise ValueError("early_stop_patience must be positive")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")


def default_v8_architecture() -> dict[str, object]:
    """The fixed Phase-A architecture; schema 3 supplies the partition."""

    return {
        "family": MODEL_FAMILY_V8,
        "card_indices": list(schema3.CARD_INDICES),
        "context_indices": list(schema3.CONTEXT_INDICES),
        "card_encoder_width": CARD_ENCODER_WIDTH,
        "context_encoder_width": CONTEXT_ENCODER_WIDTH,
        "trunk_widths": list(TRUNK_WIDTHS),
        "head_towers": {name: HEAD_TOWER_WIDTH for name in V8_HEAD_SIZES},
        "heads": dict(V8_HEAD_SIZES),
        "fold_through_branches": list(FOLD_THROUGH_BRANCHES),
        "equity_slots": list(EQUITY_SLOTS),
        "layer_norm": True,
        "dropout": None,
    }


def default_v9_architecture() -> dict[str, object]:
    """The v9 architecture block; schema 4 supplies the partition.

    Same encoder/trunk/tower widths as v8 — the v9 change is the branch
    CONTRACT (slot meanings and the ``schema4`` partition), never
    the network shape. Head sizes come from ``branch_contract_v9``, the
    single definition site.
    """

    from engine import schema4
    from engine.branch_contract_v9 import (
        EQUITY_SLOTS_V9,
        FOLD_THROUGH_BRANCHES_V9,
        MODEL_FAMILY_V9,
        V9_HEAD_SIZES,
    )

    return {
        "family": MODEL_FAMILY_V9,
        "card_indices": list(schema4.CARD_INDICES_V9),
        "context_indices": list(schema4.CONTEXT_INDICES_V9),
        "card_encoder_width": CARD_ENCODER_WIDTH,
        "context_encoder_width": CONTEXT_ENCODER_WIDTH,
        "trunk_widths": list(TRUNK_WIDTHS),
        "head_towers": {name: HEAD_TOWER_WIDTH for name in V9_HEAD_SIZES},
        "heads": dict(V9_HEAD_SIZES),
        "fold_through_branches": list(FOLD_THROUGH_BRANCHES_V9),
        "equity_slots": list(EQUITY_SLOTS_V9),
        "layer_norm": True,
        "dropout": None,
    }


def validate_v9_architecture(architecture: Mapping[str, object]) -> None:
    """Fail-closed validation of a format-4 architecture block.

    A separate validator, never a widening of the v8 one: a validator
    that accepts both shapes is how a v8 artifact serves under v9 slot
    meanings without anyone noticing.
    """

    from engine import schema4
    from engine.branch_contract_v9 import (
        EQUITY_SLOTS_V9,
        FOLD_THROUGH_BRANCHES_V9,
        MODEL_FAMILY_V9,
        V9_HEAD_SIZES,
    )

    if architecture.get("family") != MODEL_FAMILY_V9:
        raise ValueError(f"architecture family must be {MODEL_FAMILY_V9!r}")
    if list(architecture.get("card_indices") or []) != list(
        schema4.CARD_INDICES_V9
    ):
        raise ValueError("card_indices must match schema4.CARD_INDICES_V9")
    if list(architecture.get("context_indices") or []) != list(
        schema4.CONTEXT_INDICES_V9
    ):
        raise ValueError("context_indices must match schema4.CONTEXT_INDICES_V9")
    if architecture.get("card_encoder_width") != CARD_ENCODER_WIDTH:
        raise ValueError(f"card_encoder_width must be {CARD_ENCODER_WIDTH}")
    if architecture.get("context_encoder_width") != CONTEXT_ENCODER_WIDTH:
        raise ValueError(f"context_encoder_width must be {CONTEXT_ENCODER_WIDTH}")
    if list(architecture.get("trunk_widths") or []) != list(TRUNK_WIDTHS):
        raise ValueError(f"trunk_widths must be {list(TRUNK_WIDTHS)}")
    if dict(architecture.get("heads") or {}) != V9_HEAD_SIZES:
        raise ValueError(f"heads must be {V9_HEAD_SIZES}")
    towers = dict(architecture.get("head_towers") or {})
    if towers != {name: HEAD_TOWER_WIDTH for name in V9_HEAD_SIZES}:
        raise ValueError(f"head_towers must all be {HEAD_TOWER_WIDTH}")
    if list(architecture.get("fold_through_branches") or []) != list(
        FOLD_THROUGH_BRANCHES_V9
    ):
        raise ValueError(
            f"fold_through_branches must be {list(FOLD_THROUGH_BRANCHES_V9)}"
        )
    if list(architecture.get("equity_slots") or []) != list(EQUITY_SLOTS_V9):
        raise ValueError(f"equity_slots must be {list(EQUITY_SLOTS_V9)}")
    dropout = architecture.get("dropout")
    if dropout is not None and not (
        isinstance(dropout, float) and 0.0 <= dropout < 1.0
    ):
        raise ValueError("dropout must be None or a float in [0, 1)")


def validate_v8_architecture(architecture: Mapping[str, object]) -> None:
    if architecture.get("family") != MODEL_FAMILY_V8:
        raise ValueError(f"architecture family must be {MODEL_FAMILY_V8!r}")
    if list(architecture.get("card_indices") or []) != list(schema3.CARD_INDICES):
        raise ValueError("card_indices must match schema3.CARD_INDICES")
    if list(architecture.get("context_indices") or []) != list(
        schema3.CONTEXT_INDICES
    ):
        raise ValueError("context_indices must match schema3.CONTEXT_INDICES")
    if architecture.get("card_encoder_width") != CARD_ENCODER_WIDTH:
        raise ValueError(f"card_encoder_width must be {CARD_ENCODER_WIDTH}")
    if architecture.get("context_encoder_width") != CONTEXT_ENCODER_WIDTH:
        raise ValueError(f"context_encoder_width must be {CONTEXT_ENCODER_WIDTH}")
    if list(architecture.get("trunk_widths") or []) != list(TRUNK_WIDTHS):
        raise ValueError(f"trunk_widths must be {list(TRUNK_WIDTHS)}")
    if dict(architecture.get("heads") or {}) != V8_HEAD_SIZES:
        raise ValueError(f"heads must be {V8_HEAD_SIZES}")
    towers = dict(architecture.get("head_towers") or {})
    if towers != {name: HEAD_TOWER_WIDTH for name in V8_HEAD_SIZES}:
        raise ValueError(f"head_towers must all be {HEAD_TOWER_WIDTH}")
    dropout = architecture.get("dropout")
    if dropout is not None and not (
        isinstance(dropout, float) and 0.0 <= dropout < 1.0
    ):
        raise ValueError("dropout must be None or a float in [0, 1)")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _require_finite_unit(value: object, name: str, line: int) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"line {line}: {name} is not a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"line {line}: {name} {value!r} is not in [0, 1]")
    return number


def _parse_row(document: Mapping[str, Any], line: int) -> PhaseARow:
    features = document.get("features")
    if not isinstance(features, list):
        raise ValueError(f"line {line}: features is not a list")
    if len(features) != schema3.INPUT_SIZE_V8:
        raise ValueError(
            f"line {line}: features must have {schema3.INPUT_SIZE_V8} entries, "
            f"found {len(features)}"
        )
    values: list[float] = []
    for value in features:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"line {line}: non-numeric feature {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"line {line}: non-finite feature {value!r}")
        values.append(number)

    labels = document.get("labels")
    masks = document.get("masks")
    if not isinstance(labels, Mapping) or not isinstance(masks, Mapping):
        raise ValueError(f"line {line}: labels/masks are not objects")
    mask_values: dict[str, int] = {}
    for name in (*_FT_LABEL_NAMES, "range_bucket", "equity_called"):
        raw = masks.get(name)
        if raw not in (0, 1) or isinstance(raw, bool):
            raise ValueError(f"line {line}: mask {name} must be 0 or 1")
        mask_values[name] = int(raw)
    ft_small = mask_values["fold_through_small"]
    ft_large = mask_values["fold_through_large"]
    if ft_small and ft_large:
        raise ValueError(f"line {line}: both fold_through branch masks are set")

    ft_label = 0.0
    if ft_small or ft_large:
        ft_label = _require_finite_unit(
            labels.get("fold_through_small"), "fold_through_small", line
        )
        if ft_label not in (0.0, 1.0):
            raise ValueError(
                f"line {line}: fold_through label {ft_label!r} is not binary"
            )

    bucket = labels.get("range_bucket")
    if not isinstance(bucket, int) or isinstance(bucket, bool):
        raise ValueError(f"line {line}: range_bucket is not an integer")
    if mask_values["range_bucket"] and not 0 <= bucket < schema3.BELIEF_BUCKETS:
        raise ValueError(f"line {line}: range_bucket {bucket} out of range")

    equity = 0.0
    if mask_values["equity_called"]:
        equity = _require_finite_unit(
            labels.get("equity_called"), "equity_called", line
        )

    table_id = document.get("table_id")
    if not isinstance(table_id, str) or not table_id:
        raise ValueError(f"line {line}: missing table_id")
    street = document.get("street")
    if street not in ("preflop", "flop", "turn", "river"):
        raise ValueError(f"line {line}: unsupported street {street!r}")

    # Equity slot: the taken aggress branch for sized aggressions, the
    # check_call slot for every other row (see the module docstring).
    equity_slot = 0 if ft_small else 1 if ft_large else 2
    return PhaseARow(
        table_id=table_id,
        street=str(street),
        features=tuple(values),
        fold_through_label=ft_label,
        fold_through_mask=(ft_small, ft_large),
        range_bucket=max(0, min(schema3.BELIEF_BUCKETS - 1, bucket)),
        range_mask=mask_values["range_bucket"],
        equity_called=equity,
        equity_mask=mask_values["equity_called"],
        equity_slot=equity_slot,
    )


def load_phase_a_dataset(path: str | Path) -> tuple[PhaseARow, ...]:
    """Load and validate a Phase-A JSONL(.gz) dataset, fail-closed."""

    resolved = Path(path)
    opener = gzip.open if resolved.name.endswith(".gz") else open
    rows: list[PhaseARow] = []
    with opener(resolved, "rt", encoding="utf-8") as stream:  # type: ignore[operator]
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            document = json.loads(text)
            if not isinstance(document, Mapping):
                raise ValueError(f"line {line_number}: row is not an object")
            rows.append(_parse_row(document, line_number))
    if not rows:
        raise ValueError(f"no rows in {resolved}")
    return tuple(rows)


# ---------------------------------------------------------------------------
# Split and normalization
# ---------------------------------------------------------------------------


def table_split_value(split_seed: int, table_id: str) -> float:
    """Deterministic per-table hash in [0, 1), independent of row order."""

    digest = hashlib.sha256(f"{split_seed}:{table_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def split_rows(
    rows: Sequence[PhaseARow], config: V8TrainingConfig
) -> tuple[list[PhaseARow], list[PhaseARow]]:
    """90/10-style split by table_id hash: whole tables, seed recorded."""

    validation_tables = {
        table_id
        for table_id in {row.table_id for row in rows}
        if table_split_value(config.split_seed, table_id)
        < config.validation_fraction
    }
    train = [row for row in rows if row.table_id not in validation_tables]
    validation = [row for row in rows if row.table_id in validation_tables]
    return train, validation


def context_normalization(
    rows: Sequence[PhaseARow],
) -> tuple[list[float], list[float]]:
    """Full-vector scales: context z-scored (floor 0.05), card identity.

    Means/stds are computed from the supplied rows only — the caller
    passes the training split, never the validation split. The card
    block's binary one-hots pass through raw (V8_DESIGN §3: z-scoring
    them to 5-6 sigma is what broke the whole-vector OOD guard), so card
    indices carry the identity transform (mean 0, std 1).
    """

    if not rows:
        raise ValueError("cannot compute normalization from zero rows")
    means = [0.0] * schema3.INPUT_SIZE_V8
    stds = [1.0] * schema3.INPUT_SIZE_V8
    count = len(rows)
    for index in schema3.CONTEXT_INDICES:
        total = 0.0
        for row in rows:
            total += row.features[index]
        mean = total / count
        variance = 0.0
        for row in rows:
            variance += (row.features[index] - mean) ** 2
        means[index] = mean
        stds[index] = max(CONTEXT_STD_FLOOR, math.sqrt(variance / count))
    return means, stds


def _mask_counts(rows: Sequence[PhaseARow]) -> dict[str, int]:
    return {
        "rows": len(rows),
        "fold_through_small": sum(row.fold_through_mask[0] for row in rows),
        "fold_through_large": sum(row.fold_through_mask[1] for row in rows),
        "range_bucket": sum(row.range_mask for row in rows),
        "equity_called": sum(row.equity_mask for row in rows),
    }


# ---------------------------------------------------------------------------
# Torch fitting (function-local torch, offline_trainer.py's pattern)
# ---------------------------------------------------------------------------


def fit_phase_a(
    rows: Sequence[PhaseARow], config: V8TrainingConfig
) -> dict[str, object]:
    """Fit one init seed; return weights, per-head losses, calibration."""

    check_v8_config(config)
    train_rows, validation_rows = split_rows(rows, config)
    if not train_rows:
        raise ValueError("training split is empty")
    if not validation_rows:
        raise ValueError(
            "validation split is empty; adjust validation_fraction or split_seed"
        )
    means, stds = context_normalization(train_rows)

    import torch
    from torch import nn

    if config.device == "cuda":
        device_name = validate_training_device("cuda")
    else:
        device_name = "cpu"
    device = torch.device(config.device)
    torch.manual_seed(config.init_seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.init_seed)

    card_indices = list(schema3.CARD_INDICES)
    context_indices = list(schema3.CONTEXT_INDICES)

    class _NetworkV8(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.card_enc = nn.Linear(len(card_indices), CARD_ENCODER_WIDTH)
            self.ctx_enc = nn.Linear(len(context_indices), CONTEXT_ENCODER_WIDTH)
            self.card_ln = nn.LayerNorm(CARD_ENCODER_WIDTH)
            self.ctx_ln = nn.LayerNorm(CONTEXT_ENCODER_WIDTH)
            dims = [CARD_ENCODER_WIDTH + CONTEXT_ENCODER_WIDTH, *TRUNK_WIDTHS]
            self.trunk = nn.ModuleList(
                nn.Linear(dims[i], dims[i + 1]) for i in range(len(TRUNK_WIDTHS))
            )
            self.trunk_ln = nn.ModuleList(
                nn.LayerNorm(dims[i + 1]) for i in range(len(TRUNK_WIDTHS) - 1)
            )
            self.drop = nn.Dropout(float(config.dropout))
            self.towers = nn.ModuleDict(
                {
                    name: nn.Linear(TRUNK_WIDTHS[-1], HEAD_TOWER_WIDTH)
                    for name in V8_HEAD_SIZES
                }
            )
            self.outs = nn.ModuleDict(
                {
                    name: nn.Linear(HEAD_TOWER_WIDTH, size)
                    for name, size in V8_HEAD_SIZES.items()
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
                for name in V8_HEAD_SIZES
            }

    model = _NetworkV8()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=math.sqrt(2.0 / module.in_features))
                nn.init.zeros_(module.bias)
        # Zero output layers (all four heads, residual included): the model
        # starts at the constant predictor — the v7 recipe's largest single
        # init effect, kept verbatim.
        for name in V8_HEAD_SIZES:
            nn.init.zeros_(model.outs[name].weight)
            nn.init.zeros_(model.outs[name].bias)
    model.to(device)

    mean_tensor = torch.tensor(means, dtype=torch.float32)
    std_tensor = torch.tensor(stds, dtype=torch.float32)

    def tensors(examples: Sequence[PhaseARow]) -> dict[str, object]:
        features = torch.tensor(
            [row.features for row in examples], dtype=torch.float32
        )
        features = ((features - mean_tensor) / std_tensor).to(device)
        return {
            "card": features[:, card_indices],
            "ctx": features[:, context_indices],
            "ft_target": torch.tensor(
                [[row.fold_through_label] * 2 for row in examples],
                dtype=torch.float32,
                device=device,
            ),
            "ft_mask": torch.tensor(
                [
                    [float(row.fold_through_mask[0]), float(row.fold_through_mask[1])]
                    for row in examples
                ],
                dtype=torch.float32,
                device=device,
            ),
            "range_target": torch.tensor(
                [row.range_bucket for row in examples],
                dtype=torch.long,
                device=device,
            ),
            "range_mask": torch.tensor(
                [float(row.range_mask) for row in examples],
                dtype=torch.float32,
                device=device,
            ),
            "eq_target": torch.tensor(
                [row.equity_called for row in examples],
                dtype=torch.float32,
                device=device,
            ),
            "eq_slot": torch.tensor(
                [row.equity_slot for row in examples],
                dtype=torch.long,
                device=device,
            ),
            "eq_mask": torch.tensor(
                [float(row.equity_mask) for row in examples],
                dtype=torch.float32,
                device=device,
            ),
        }

    train_data = tensors(train_rows)
    validation_data = tensors(validation_rows)

    def head_losses(data: dict[str, object], indexes) -> tuple:
        """Masked per-head losses over the indexed rows.

        The residual head is deliberately absent: Phase A gives it zero
        gradient (its parameters never enter the loss graph).
        """

        outputs = model(data["card"][indexes], data["ctx"][indexes])
        arange = torch.arange(indexes.shape[0], device=device)
        ft_elementwise = nn.functional.binary_cross_entropy_with_logits(
            outputs["fold_through"], data["ft_target"][indexes], reduction="none"
        )
        ft_mask = data["ft_mask"][indexes]
        ft_loss = (ft_elementwise * ft_mask).sum() / ft_mask.sum().clamp(min=1.0)
        log_probabilities = torch.log_softmax(outputs["range"], dim=1)
        nll = -log_probabilities[arange, data["range_target"][indexes]]
        range_mask = data["range_mask"][indexes]
        range_loss = (nll * range_mask).sum() / range_mask.sum().clamp(min=1.0)
        eq_predicted = outputs["equity_called"][arange, data["eq_slot"][indexes]]
        eq_mask = data["eq_mask"][indexes]
        eq_loss = (
            (eq_predicted - data["eq_target"][indexes]).square() * eq_mask
        ).sum() / eq_mask.sum().clamp(min=1.0)
        return ft_loss, range_loss, eq_loss

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
    steps_per_epoch = max(1, math.ceil(len(train_rows) / config.batch_size))
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

    def optimize(loss) -> None:
        nonlocal step
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss during v8 training")
        set_learning_rate()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite gradient during v8 training")
        optimizer.step()
        step += 1

    def evaluated_losses(
        data: dict[str, object], counts: Mapping[str, int]
    ) -> dict[str, float | None]:
        """Full-pass masked losses in eval mode; None when a head is unmasked."""

        was_training = model.training
        model.eval()
        with torch.no_grad():
            indexes = torch.arange(data["range_mask"].shape[0], device=device)
            ft_loss, range_loss, eq_loss = head_losses(data, indexes)
        if was_training:
            model.train()
        ft_defined = counts["fold_through_small"] + counts["fold_through_large"] > 0
        losses: dict[str, float | None] = {
            "fold_through": float(ft_loss) if ft_defined else None,
            "range": float(range_loss) if counts["range_bucket"] > 0 else None,
            "equity_called": (
                float(eq_loss) if counts["equity_called"] > 0 else None
            ),
        }
        losses["total"] = sum(value for value in losses.values() if value is not None)
        return losses

    train_counts = _mask_counts(train_rows)
    validation_counts = _mask_counts(validation_rows)

    generator = random.Random(config.init_seed + 1)
    best_loss = math.inf
    best_state: dict[str, object] | None = None
    best_epoch = 0
    stale_epochs = 0
    epochs_run = 0
    model.train()
    for epoch in range(config.epochs):
        epochs_run = epoch + 1
        order = list(range(len(train_rows)))
        generator.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            indexes = torch.tensor(
                order[start : start + config.batch_size],
                dtype=torch.long,
                device=device,
            )
            ft_loss, range_loss, eq_loss = head_losses(train_data, indexes)
            optimize(ft_loss + range_loss + eq_loss)
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("non-finite parameter during v8 training")
        epoch_losses = evaluated_losses(validation_data, validation_counts)
        epoch_total = float(epoch_losses["total"])
        if epoch_total < best_loss - 1e-9:
            best_loss = epoch_total
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
    model.eval()

    train_losses = evaluated_losses(train_data, train_counts)
    validation_losses = evaluated_losses(validation_data, validation_counts)

    def calibration() -> dict[str, object]:
        with torch.no_grad():
            outputs = model(validation_data["card"], validation_data["ctx"])
            ft_probabilities = torch.sigmoid(outputs["fold_through"]).cpu().tolist()
            range_probabilities = (
                torch.softmax(outputs["range"], dim=1).cpu().tolist()
            )
            arange = torch.arange(validation_data["eq_slot"].shape[0], device=device)
            eq_predicted = (
                outputs["equity_called"][arange, validation_data["eq_slot"]]
                .cpu()
                .tolist()
            )

        pairs: list[tuple[float, float]] = []
        for row, probabilities in zip(validation_rows, ft_probabilities):
            for branch in range(2):
                if row.fold_through_mask[branch]:
                    pairs.append((probabilities[branch], row.fold_through_label))
        pairs.sort()
        deciles: list[dict[str, float | int]] = []
        for decile in range(10):
            low = decile * len(pairs) // 10
            high = (decile + 1) * len(pairs) // 10
            chunk = pairs[low:high]
            if not chunk:
                continue
            deciles.append(
                {
                    "decile": decile,
                    "count": len(chunk),
                    "mean_predicted": round(
                        sum(pred for pred, _ in chunk) / len(chunk), 6
                    ),
                    "observed_rate": round(
                        sum(label for _, label in chunk) / len(chunk), 6
                    ),
                }
            )

        bucket_predicted = [0.0] * schema3.BELIEF_BUCKETS
        bucket_observed = [0] * schema3.BELIEF_BUCKETS
        range_count = 0
        for row, probabilities in zip(validation_rows, range_probabilities):
            if not row.range_mask:
                continue
            range_count += 1
            bucket_observed[row.range_bucket] += 1
            for bucket in range(schema3.BELIEF_BUCKETS):
                bucket_predicted[bucket] += probabilities[bucket]
        range_buckets = [
            {
                "bucket": bucket,
                "mean_predicted": round(
                    bucket_predicted[bucket] / max(1, range_count), 6
                ),
                "empirical": round(
                    bucket_observed[bucket] / max(1, range_count), 6
                ),
            }
            for bucket in range(schema3.BELIEF_BUCKETS)
        ]

        eq_error = 0.0
        eq_count = 0
        slot_counts = [0] * len(EQUITY_SLOTS)
        for row, predicted in zip(validation_rows, eq_predicted):
            if not row.equity_mask:
                continue
            eq_error += abs(predicted - row.equity_called)
            eq_count += 1
            slot_counts[row.equity_slot] += 1
        return {
            "fold_through_deciles": {
                "count": len(pairs),
                "observed_rate": round(
                    sum(label for _, label in pairs) / max(1, len(pairs)), 6
                ),
                "mean_predicted": round(
                    sum(pred for pred, _ in pairs) / max(1, len(pairs)), 6
                ),
                "deciles": deciles,
            },
            "range_buckets": {"count": range_count, "buckets": range_buckets},
            "equity_called": {
                "count": eq_count,
                "mae": round(eq_error / max(1, eq_count), 6),
                "supervised_slot_counts": {
                    name: slot_counts[index]
                    for index, name in enumerate(EQUITY_SLOTS)
                },
            },
        }

    def block(module) -> dict[str, object]:
        return {
            "w": module.weight.detach().cpu().tolist(),
            "b": module.bias.detach().cpu().tolist(),
        }

    def norm_block(module) -> dict[str, object]:
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
            for name in V8_HEAD_SIZES
        },
    }
    _assert_finite_weights(weights, "v8 phase-A training")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "weights": weights,
        "means": means,
        "stds": stds,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_tables": len({row.table_id for row in train_rows}),
        "validation_tables": len({row.table_id for row in validation_rows}),
        "train_counts": train_counts,
        "validation_counts": validation_counts,
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "calibration": calibration(),
        "trace": {
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "optimizer_steps": step,
            "best_validation_loss_total": (
                None if best_loss is math.inf else round(best_loss, 6)
            ),
        },
        "device_name": device_name,
        "parameter_count": parameter_count,
        "init_seed": config.init_seed,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _require_matrix(name: str, value: object, rows: int, cols: int) -> None:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} must have {rows} rows")
    for row in value:
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(f"{name} rows must have {cols} entries")


def _require_vector(name: str, value: object, size: int) -> None:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must have {size} entries")


def validate_v8_weight_shapes(weights: Mapping[str, object]) -> None:
    """Fail-closed structural check before an artifact is written."""

    card = weights["card_encoder"]
    _require_matrix(
        "card_encoder.w", card["w"], CARD_ENCODER_WIDTH, schema3.CARD_BLOCK_SIZE
    )
    _require_vector("card_encoder.b", card["b"], CARD_ENCODER_WIDTH)
    _require_vector("card_encoder.ln_g", card["ln_g"], CARD_ENCODER_WIDTH)
    _require_vector("card_encoder.ln_b", card["ln_b"], CARD_ENCODER_WIDTH)
    context = weights["context_encoder"]
    _require_matrix(
        "context_encoder.w",
        context["w"],
        CONTEXT_ENCODER_WIDTH,
        schema3.CONTEXT_BLOCK_SIZE,
    )
    _require_vector("context_encoder.b", context["b"], CONTEXT_ENCODER_WIDTH)
    _require_vector("context_encoder.ln_g", context["ln_g"], CONTEXT_ENCODER_WIDTH)
    _require_vector("context_encoder.ln_b", context["ln_b"], CONTEXT_ENCODER_WIDTH)
    trunk = weights["trunk"]
    if not isinstance(trunk, list) or len(trunk) != len(TRUNK_WIDTHS):
        raise ValueError(f"trunk must have {len(TRUNK_WIDTHS)} blocks")
    dims = [CARD_ENCODER_WIDTH + CONTEXT_ENCODER_WIDTH, *TRUNK_WIDTHS]
    for index, entry in enumerate(trunk):
        _require_matrix(f"trunk[{index}].w", entry["w"], dims[index + 1], dims[index])
        _require_vector(f"trunk[{index}].b", entry["b"], dims[index + 1])
        if index < len(TRUNK_WIDTHS) - 1:
            _require_vector(f"trunk[{index}].ln_g", entry["ln_g"], dims[index + 1])
            _require_vector(f"trunk[{index}].ln_b", entry["ln_b"], dims[index + 1])
        elif "ln_g" in entry or "ln_b" in entry:
            raise ValueError("the final trunk block must not carry LayerNorm")
    heads = weights["heads"]
    if set(heads) != set(V8_HEAD_SIZES):
        raise ValueError(f"heads must be exactly {sorted(V8_HEAD_SIZES)}")
    for name, size in V8_HEAD_SIZES.items():
        head = heads[name]
        _require_matrix(
            f"heads.{name}.tower_w", head["tower_w"], HEAD_TOWER_WIDTH, TRUNK_WIDTHS[-1]
        )
        _require_vector(f"heads.{name}.tower_b", head["tower_b"], HEAD_TOWER_WIDTH)
        _require_matrix(f"heads.{name}.out_w", head["out_w"], size, HEAD_TOWER_WIDTH)
        _require_vector(f"heads.{name}.out_b", head["out_b"], size)


_REQUIRED_MANIFEST_KEYS = (
    "format",
    "format_version",
    "model_version",
    "state",
    "parent_version",
    "created_at",
    "feature_schema_version",
    "input_size",
    "feature_names",
    "action_labels",
    "architecture",
    "weights_file",
    "weights_sha256",
    "training_window",
    "engine_parameters",
    "serve",
    "training",
    "evaluation",
    "promotion",
)


def validate_v8_manifest(manifest: Mapping[str, object]) -> None:
    """Structural contract for a format-3 candidate manifest.

    Deliberately accepts no state but ``"candidate"``: promotion is a
    separate, explicit, human-authorised act and never this module's.
    """

    for key in _REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            raise ValueError(f"manifest is missing {key!r}")
    if manifest["format"] != MODEL_FORMAT:
        raise ValueError(f"format must be {MODEL_FORMAT!r}")
    if manifest["format_version"] != MODEL_FORMAT_VERSION_V8:
        raise ValueError(f"format_version must be {MODEL_FORMAT_VERSION_V8}")
    if manifest["state"] != "candidate":
        raise ValueError("a v8 trainer manifest state must be 'candidate'")
    if manifest["promotion"] is not None:
        raise ValueError("a v8 trainer manifest cannot carry a promotion record")
    if manifest["feature_schema_version"] != schema3.SCHEMA_VERSION:
        raise ValueError(f"feature_schema_version must be {schema3.SCHEMA_VERSION}")
    if manifest["input_size"] != schema3.INPUT_SIZE_V8:
        raise ValueError(f"input_size must be {schema3.INPUT_SIZE_V8}")
    if list(manifest["feature_names"]) != list(schema3.FEATURE_NAMES_V8):
        raise ValueError("feature_names must match schema3.FEATURE_NAMES_V8")
    if list(manifest["action_labels"]) != list(BRANCH_LABELS_V8):
        raise ValueError(f"action_labels must be {list(BRANCH_LABELS_V8)}")
    validate_v8_architecture(manifest["architecture"])  # type: ignore[arg-type]
    sha = manifest["weights_sha256"]
    if not (isinstance(sha, str) and len(sha) == 64):
        raise ValueError("weights_sha256 must be a 64-character digest")


def _rounded_losses(losses: Mapping[str, float | None]) -> dict[str, float | None]:
    return {
        name: (None if value is None else round(value, 6))
        for name, value in losses.items()
    }


def _seed_run_record(result: Mapping[str, object]) -> dict[str, object]:
    trace = result["trace"]
    assert isinstance(trace, Mapping)
    return {
        "init_seed": result["init_seed"],
        "train_losses": _rounded_losses(result["train_losses"]),  # type: ignore[arg-type]
        "validation_losses": _rounded_losses(result["validation_losses"]),  # type: ignore[arg-type]
        "best_epoch": trace["best_epoch"],
        "epochs_run": trace["epochs_run"],
        "optimizer_steps": trace["optimizer_steps"],
    }


def train_phase_a_candidate(
    rows: Sequence[PhaseARow],
    output_dir: str | Path,
    config: V8TrainingConfig = V8TrainingConfig(),
    init_seeds: Sequence[int] = (101, 202, 303),
    dataset_path: str | Path | None = None,
) -> V8TrainingSummary:
    """Train every init seed, select by total validation loss, export one.

    Selection caveat (V8_DESIGN §6, deliberately NOT baked in as anything
    cleverer): validation loss is a gate, never a selector — it ranked v7
    seeds backwards once. The exported artifact records every seed's
    per-head losses so a later duel can overrule this pick.
    """

    check_v8_config(config)
    if not init_seeds:
        raise ValueError("at least one init seed is required")
    if len(set(init_seeds)) != len(init_seeds):
        raise ValueError("init seeds must be unique")
    model_version = config.model_version
    if not model_version:
        raise ValueError("config.model_version is required for export")
    output_path = Path(output_dir).expanduser().resolve()
    weights_path = output_path / f"{model_version}.weights.json"
    manifest_path = output_path / f"{model_version}.manifest.json"
    if weights_path.exists() or manifest_path.exists():
        raise FileExistsError(f"candidate artifact already exists for {model_version}")

    started = time.monotonic()
    results = []
    for init_seed in init_seeds:
        seed_config = replace(config, init_seed=init_seed)
        result = fit_phase_a(rows, seed_config)
        results.append(result)
        validation = result["validation_losses"]
        assert isinstance(validation, Mapping)
        print(
            f"seed {init_seed}: val total {validation['total']:.6f} "
            f"(fold_through {validation['fold_through']}, "
            f"range {validation['range']}, "
            f"equity_called {validation['equity_called']}), "
            f"best epoch {result['trace']['best_epoch']}",  # type: ignore[index]
            flush=True,
        )
    best = min(
        results,
        key=lambda result: float(result["validation_losses"]["total"]),  # type: ignore[index]
    )

    weights = best["weights"]
    assert isinstance(weights, Mapping)
    validate_v8_weight_shapes(weights)
    architecture = default_v8_architecture()
    architecture["dropout"] = float(config.dropout)
    validate_v8_architecture(architecture)

    output_path.mkdir(parents=True, exist_ok=True)
    weights_document = _round9(
        {
            "format": MODEL_FORMAT,
            "format_version": MODEL_FORMAT_VERSION_V8,
            "model_version": model_version,
            "feature_normalization": {
                "means": list(best["means"]),  # type: ignore[arg-type]
                "stds": list(best["stds"]),  # type: ignore[arg-type]
            },
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

    sidecar_generator: Mapping[str, object] | None = None
    if dataset_path is not None:
        sidecar_path = Path(dataset_path).parent / (
            Path(dataset_path).name.removesuffix(".jsonl.gz") + ".summary.json"
        )
        if sidecar_path.is_file():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if isinstance(sidecar, Mapping):
                    sidecar_generator = sidecar.get("generator")
            except (OSError, ValueError):
                sidecar_generator = None

    per_street_rows: dict[str, int] = {}
    for row in rows:
        per_street_rows[row.street] = per_street_rows.get(row.street, 0) + 1

    manifest = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION_V8,
        "model_version": model_version,
        "state": "candidate",
        "parent_version": None,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "feature_schema_version": schema3.SCHEMA_VERSION,
        "input_size": schema3.INPUT_SIZE_V8,
        "feature_names": list(schema3.FEATURE_NAMES_V8),
        "action_labels": list(BRANCH_LABELS_V8),
        "architecture": architecture,
        "weights_file": weights_path.name,
        "weights_sha256": weights_sha256,
        "training_window": {
            "dataset": None if dataset_path is None else str(dataset_path),
            "dataset_generator": (
                dict(sidecar_generator) if sidecar_generator else None
            ),
            "row_count": len(rows),
            "table_count": len({row.table_id for row in rows}),
            "label_coverage": _mask_counts(rows),
            "per_street_rows": dict(sorted(per_street_rows.items())),
        },
        "engine_parameters": {
            "safety_gates": DEFAULT_SAFETY_GATES.to_mapping(),
            "temperature_shaping": DEFAULT_TEMPERATURE_SHAPING.to_mapping(),
            "tracker_settings": DEFAULT_TRACKER_SETTINGS.to_mapping(),
            "bluff_settings": DEFAULT_BLUFF_SETTINGS.to_mapping(),
        },
        "serve": {
            # By construction the OOD guard watches only the context block
            # (V8_DESIGN §3): the raw card one-hots never enter it.
            "ood_guard_indices": list(schema3.CONTEXT_INDICES),
            "temperature": None,
            "note": (
                "Phase-A component heads only; the composed-value serve "
                "path (V8_DESIGN §4) is a separate work item and this "
                "artifact must not be deployed"
            ),
        },
        "training": {
            "objective": TRAINING_OBJECTIVE_V8_PHASE_A,
            "phase": "A",
            "optimizer": "adamw",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "warmup_steps": config.warmup_steps,
            "early_stop_patience": config.early_stop_patience,
            "epochs": config.epochs,
            "epochs_run": best["trace"]["epochs_run"],  # type: ignore[index]
            "best_epoch": best["trace"]["best_epoch"],  # type: ignore[index]
            "best_validation_loss_total": best["trace"][  # type: ignore[index]
                "best_validation_loss_total"
            ],
            "optimizer_steps": best["trace"]["optimizer_steps"],  # type: ignore[index]
            "gradient_clip": "global-norm 1.0",
            "batch_size": config.batch_size,
            "backend": "pytorch",
            "device": config.device,
            "device_name": best["device_name"],
            "parameter_count": best["parameter_count"],
            "split": {
                "method": "sha256(split_seed:table_id) hash, whole tables",
                "validation_fraction": config.validation_fraction,
                "split_seed": config.split_seed,
                "train_rows": best["train_rows"],
                "validation_rows": best["validation_rows"],
                "train_tables": best["train_tables"],
                "validation_tables": best["validation_tables"],
                "train_label_coverage": best["train_counts"],
                "validation_label_coverage": best["validation_counts"],
            },
            "init_seed": best["init_seed"],
            "init_seeds_evaluated": [_seed_run_record(result) for result in results],
            "seed_selection": (
                "minimum total validation loss; validation loss is a gate, "
                "never a selector (V8_DESIGN §6) — the seat-swapped duel "
                "remains the selector at evaluation time"
            ),
            "branch_targets": {
                "aggress_small": list(_BRANCH_SMALL),
                "aggress_large": list(_BRANCH_LARGE),
            },
            "heads_trained": list(SUPERVISED_HEADS),
            "residual_head": (
                "zero-initialized output, excluded from every Phase-A loss "
                "(zero gradient); exported untrained for Phase B"
            ),
            "targets": {
                "fold_through": (
                    "masked BCE per aggress branch on the observed everyone-"
                    "folded outcome; a row supervises only the branch its "
                    "realized size matched"
                ),
                "range": (
                    "masked cross-entropy over 8 strength-percentile octiles "
                    "of the strongest continuing opponent holding "
                    "(strength_metric.strength_percentile)"
                ),
                "equity_called": (
                    "masked MSE on hero's exact/MC pot share against the "
                    "actual continuing holdings; slots (aggress_small, "
                    "aggress_large, check_call); non-aggress rows (folds "
                    "included) supervise the check_call slot"
                ),
                "residual": "untrained in Phase A",
            },
            "input_normalization": (
                "context block only: per-feature z-score from training-split "
                "rows, std floor 0.05; card block raw (identity scales "
                "stored); scales in the weights file"
            ),
        },
        "evaluation": {
            "train_losses": _rounded_losses(best["train_losses"]),  # type: ignore[arg-type]
            "validation_losses": _rounded_losses(
                best["validation_losses"]  # type: ignore[arg-type]
            ),
            "calibration": best["calibration"],
        },
        "promotion": None,
    }
    validate_v8_manifest(manifest)
    weights_path.write_bytes(encoded + b"\n")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return V8TrainingSummary(
        rows=len(rows),
        train_rows=int(best["train_rows"]),  # type: ignore[arg-type]
        validation_rows=int(best["validation_rows"]),  # type: ignore[arg-type]
        selected_init_seed=int(best["init_seed"]),  # type: ignore[arg-type]
        train_losses=_rounded_losses(best["train_losses"]),  # type: ignore[arg-type]
        validation_losses=_rounded_losses(best["validation_losses"]),  # type: ignore[arg-type]
        seed_runs=tuple(_seed_run_record(result) for result in results),
        calibration=best["calibration"],  # type: ignore[arg-type]
        weights_sha256=weights_sha256,
        weights_path=weights_path,
        manifest_path=manifest_path,
        wall_time_seconds=time.monotonic() - started,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    defaults = V8TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(Path("artifacts") / "phase_a" / "phase-a-dataset.jsonl.gz"),
    )
    parser.add_argument("--output-dir", default="artifacts/candidates")
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--init-seeds", type=int, nargs="+", default=[101, 202, 303]
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument(
        "--learning-rate", type=float, default=defaults.learning_rate
    )
    parser.add_argument(
        "--weight-decay", type=float, default=defaults.weight_decay
    )
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument(
        "--warmup-steps", type=int, default=defaults.warmup_steps
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=defaults.early_stop_patience
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--validation-fraction", type=float, default=defaults.validation_fraction
    )
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default=defaults.device
    )
    args = parser.parse_args(argv)
    config = V8TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        warmup_steps=args.warmup_steps,
        early_stop_patience=args.early_stop_patience,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        device=args.device,
        model_version=args.model_version,
    )
    rows = load_phase_a_dataset(args.dataset)
    print(f"dataset: {args.dataset} ({len(rows)} rows)")
    summary = train_phase_a_candidate(
        rows,
        args.output_dir,
        config,
        init_seeds=tuple(args.init_seeds),
        dataset_path=args.dataset,
    )
    print(f"selected init seed: {summary.selected_init_seed}")
    print(f"train losses: {summary.train_losses}")
    print(f"validation losses: {summary.validation_losses}")
    print(f"manifest: {summary.manifest_path}")
    print(f"weights: {summary.weights_path}")
    print(f"weights_sha256: {summary.weights_sha256}")
    print(f"wall_time_seconds: {summary.wall_time_seconds:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BRANCH_LABELS_V8",
    "CARD_ENCODER_WIDTH",
    "CONTEXT_ENCODER_WIDTH",
    "CONTEXT_STD_FLOOR",
    "EQUITY_SLOTS",
    "FOLD_THROUGH_BRANCHES",
    "HEAD_TOWER_WIDTH",
    "MODEL_FAMILY_V8",
    "MODEL_FORMAT_VERSION_V8",
    "PhaseARow",
    "SUPERVISED_HEADS",
    "TRAINING_OBJECTIVE_V8_PHASE_A",
    "TRUNK_WIDTHS",
    "V8TrainingConfig",
    "V8TrainingSummary",
    "V8_HEAD_SIZES",
    "check_v8_config",
    "context_normalization",
    "default_v8_architecture",
    "fit_phase_a",
    "load_phase_a_dataset",
    "split_rows",
    "table_split_value",
    "train_phase_a_candidate",
    "validate_v8_architecture",
    "validate_v8_manifest",
    "validate_v8_weight_shapes",
]
