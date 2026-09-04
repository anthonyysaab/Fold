"""Phase-A supervised trainer for the v9 composed-value network (L3).

Fork of :mod:`engine.v8_trainer` for the v9 branch contract
(`engine/branch_contract_v9.py`), left as a sibling module so the v8
trainer stays byte-identical for the frozen format-3 line. The network
recipe — encoders, trunk, towers, AdamW schedule, early stopping, seed
selection — is the measured v7/v8 recipe verbatim; what changes is the
DATA CONTRACT, pinned in `.handoff/notes/V9_RESTRUCTURE_PLAN.md`
("L3/L4 DATA CONTRACTS") before any of this code existed:

- **The renamed label keys ARE the version guard.** Masks and labels are
  ``fold_through_active`` / ``fold_through_aggressive`` (the two wager
  lanes of ``FOLD_THROUGH_BRANCHES_V9``). A v8 dataset (``fold_through_
  small/large``) fails loudly here, and a v9 dataset fails loudly in the
  v8 loader — old files can never load with misrouted slots.
- **``to_call_zero`` is the lane-legality indicator.** Active-lane
  fold-through supervision is valid ONLY on free-spot rows (a call
  closes the action and defines no fold-through); aggressive supervision
  ONLY on priced rows (the contract masks the lane at ``to_call == 0``).
  The loader REJECTS violations instead of masking them away.
- **``read_temperature_x10``** carries g's temperature read as the raw
  int ``10·T`` (``aggression_sizing.read_to_context_int``). Consumers
  decode it through the ARTIFACT'S OWN sizing parameters and never
  recompute the read — ``round(x, 1)`` is banker's rounding, which torch
  cannot reproduce. Phase A's supervised losses do not consume it, but
  every row must carry a valid read so the same rows can flow into the
  Phase-B interleave and later audits without a re-harvest.
- **Equity labels route to ``EQUITY_SLOTS_V9`` positions BY NAME**
  (``passive`` / ``active`` / ``aggressive``), never by literal index:
  the taken wager lane for sized wagers (active-lane bets, aggressive
  raises); ``passive`` for a free-spot row without a wager (the
  checked-through conditional); ``active`` for every priced row without
  hero aggression (calls, and folds share that continuing-set-at-price
  conditional, exactly as v8's folds shared ``check_call``'s).

Head widths and slot orders all derive from ``branch_contract_v9``'s
tuples — the single definition site — and the architecture block is
``v8_trainer.default_v9_architecture`` (same shape as v8 by design; the
v9 change is slot MEANINGS, which is why the manifest carries format 4,
schema 4, ``BRANCH_LABELS_V9``, and a mandatory ``sizing`` record).

The manifest's ``sizing`` block is the composed sizing record the
dataset was built under (g identity + every rule-dial state). It is
resolved fail-loud: an explicit ``sizing_record`` argument must agree
with the dataset sidecar's record when the sidecar carries one; with no
explicit record the sidecar's is used; only when neither exists does the
module default (`engine.rules.composition.composed_sizing_record`, every
dial off) apply. Loading a record written under a foreign g identity
refuses.

Torch imports are function-local (``offline_trainer.py``'s pattern): the
module imports cleanly on the stdlib-only interpreter, and training runs
in the CUDA venv. This trainer writes immutable candidate artifacts only
— it never promotes, never touches ``artifacts/approved.json``, and its
manifest validator accepts no state but ``"candidate"``.

Usage (CUDA venv):
    python -m engine.v9_trainer \
        --model-version candidate-v9-0001 --init-seeds 101 202 303
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

from engine import schema4
from engine.branch_contract_v9 import (
    BRANCH_LABELS_V9,
    EQUITY_SLOTS_V9,
    FOLD_THROUGH_BRANCHES_V9,
    MODEL_FORMAT_VERSION_V9,
    V9_HEAD_SIZES,
)
from engine.dataset_provenance import describe, require_live_dataset
from engine.decision_engine import (
    DEFAULT_SAFETY_GATES,
    DEFAULT_TEMPERATURE_SHAPING,
)
from engine.learning_contract import MODEL_FORMAT
from engine.offline_trainer import (
    _assert_finite_weights,
    _round9,
    validate_training_device,
)
from engine.opponent_model import DEFAULT_TRACKER_SETTINGS
from engine.rules.composition import (
    composed_sizing_record,
    parameters_and_rules_from_record,
)
from engine.v8_trainer import (
    CARD_ENCODER_WIDTH,
    CONTEXT_ENCODER_WIDTH,
    CONTEXT_STD_FLOOR,
    HEAD_TOWER_WIDTH,
    TRUNK_WIDTHS,
    V8TrainingConfig,
    _require_finite_unit,
    check_v8_config,
    default_v9_architecture,
    split_rows,
    validate_v9_architecture,
)

TRAINING_OBJECTIVE_V9_PHASE_A = "phase_a_supervised_component_heads_v9"
SUPERVISED_HEADS_V9 = ("fold_through", "range", "equity_called")

#: The renamed mask/label keys, in FOLD_THROUGH_BRANCHES_V9 order. The
#: rename IS the dataset version bump: never keep old key names with new
#: semantics (pinned rule).
_FT_LABEL_NAMES_V9: tuple[str, ...] = tuple(
    f"fold_through_{branch}" for branch in FOLD_THROUGH_BRANCHES_V9
)
_FT_ACTIVE = FOLD_THROUGH_BRANCHES_V9.index("active")
_FT_AGGRESSIVE = FOLD_THROUGH_BRANCHES_V9.index("aggressive")
#: Equity slots resolved BY NAME once, at the single definition site.
_EQ_SLOT_PASSIVE = EQUITY_SLOTS_V9.index("passive")
_EQ_SLOT_ACTIVE = EQUITY_SLOTS_V9.index("active")
_EQ_SLOT_AGGRESSIVE = EQUITY_SLOTS_V9.index("aggressive")

#: The v8 key names, recognized only to refuse them with guidance.
_V8_FT_KEYS = frozenset({"fold_through_small", "fold_through_large"})


@dataclass(frozen=True, slots=True)
class PhaseARowV9:
    """One validated v9 Phase-A supervision row."""

    table_id: str
    street: str
    features: tuple[float, ...]
    fold_through_label: float
    fold_through_mask: tuple[int, ...]  # FOLD_THROUGH_BRANCHES_V9 order
    range_bucket: int
    range_mask: int
    equity_called: float
    equity_mask: int
    equity_slot: int  # index into EQUITY_SLOTS_V9, resolved by name
    to_call_zero: bool
    read_temperature_x10: int  # g's read, 10·T, decoded by consumers


# ---------------------------------------------------------------------------
# Dataset loading — fail-closed, the renamed keys are the version guard
# ---------------------------------------------------------------------------


def _parse_row_v9(document: Mapping[str, Any], line: int) -> PhaseARowV9:
    features = document.get("features")
    if not isinstance(features, list):
        raise ValueError(f"line {line}: features is not a list")
    if len(features) != schema4.INPUT_SIZE_V9:
        raise ValueError(
            f"line {line}: features must have {schema4.INPUT_SIZE_V9} "
            f"entries, found {len(features)}"
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
    if _V8_FT_KEYS & set(masks) or _V8_FT_KEYS & set(labels):
        raise ValueError(
            f"line {line}: fold_through_small/large are the v8 dataset's "
            "keys — this is not a v9 Phase-A dataset (the renamed keys "
            "are the version guard; v9 never relabels old corpora)"
        )
    mask_values: dict[str, int] = {}
    for name in (*_FT_LABEL_NAMES_V9, "range_bucket", "equity_called"):
        raw = masks.get(name)
        if raw not in (0, 1) or isinstance(raw, bool):
            raise ValueError(f"line {line}: mask {name} must be 0 or 1")
        mask_values[name] = int(raw)
    ft_mask = tuple(mask_values[name] for name in _FT_LABEL_NAMES_V9)
    if sum(ft_mask) > 1:
        raise ValueError(f"line {line}: both fold_through branch masks are set")

    to_call_zero = document.get("to_call_zero")
    if not isinstance(to_call_zero, bool):
        raise ValueError(f"line {line}: to_call_zero must be a boolean")
    # The lane-legality rules, rejected rather than masked away: an
    # active-lane fold-through exists only where the lane executes as a
    # BET (a call closes hero's action and buys no folds), and the
    # aggressive lane does not exist at a free spot under the contract.
    if ft_mask[_FT_ACTIVE] and not to_call_zero:
        raise ValueError(
            f"line {line}: fold_through_active is masked in on a priced "
            "row — a call defines no fold-through; active-lane "
            "supervision is valid only at to_call == 0"
        )
    if ft_mask[_FT_AGGRESSIVE] and to_call_zero:
        raise ValueError(
            f"line {line}: fold_through_aggressive is masked in on a "
            "free-spot row — the aggressive lane is escalation-only and "
            "masked at to_call == 0"
        )

    ft_label = 0.0
    if any(ft_mask):
        supervised = _FT_LABEL_NAMES_V9[ft_mask.index(1)]
        ft_label = _require_finite_unit(labels.get(supervised), supervised, line)
        if ft_label not in (0.0, 1.0):
            raise ValueError(
                f"line {line}: {supervised} label {ft_label!r} is not binary"
            )

    bucket = labels.get("range_bucket")
    if not isinstance(bucket, int) or isinstance(bucket, bool):
        raise ValueError(f"line {line}: range_bucket is not an integer")
    if mask_values["range_bucket"] and not 0 <= bucket < schema4.BELIEF_BUCKETS:
        raise ValueError(f"line {line}: range_bucket {bucket} out of range")

    equity = 0.0
    if mask_values["equity_called"]:
        equity = _require_finite_unit(
            labels.get("equity_called"), "equity_called", line
        )

    encoded = document.get("read_temperature_x10")
    if not isinstance(encoded, int) or isinstance(encoded, bool):
        raise ValueError(f"line {line}: read_temperature_x10 is not an integer")
    if not 0 <= encoded <= 1000:
        raise ValueError(
            f"line {line}: read_temperature_x10 {encoded} is not in [0, 1000]"
        )

    table_id = document.get("table_id")
    if not isinstance(table_id, str) or not table_id:
        raise ValueError(f"line {line}: missing table_id")
    street = document.get("street")
    if street not in ("preflop", "flop", "turn", "river"):
        raise ValueError(f"line {line}: unsupported street {street!r}")

    # Equity slot, resolved BY NAME (pinned rule): the taken wager lane
    # for sized wagers; otherwise the conditional the row observed —
    # checked-through (passive) at a free spot, the continuing set at
    # the existing price (active) on priced rows, folds included.
    if ft_mask[_FT_ACTIVE]:
        slot = _EQ_SLOT_ACTIVE
    elif ft_mask[_FT_AGGRESSIVE]:
        slot = _EQ_SLOT_AGGRESSIVE
    elif to_call_zero:
        slot = _EQ_SLOT_PASSIVE
    else:
        slot = _EQ_SLOT_ACTIVE
    return PhaseARowV9(
        table_id=table_id,
        street=str(street),
        features=tuple(values),
        fold_through_label=ft_label,
        fold_through_mask=ft_mask,
        range_bucket=max(0, min(schema4.BELIEF_BUCKETS - 1, bucket)),
        range_mask=mask_values["range_bucket"],
        equity_called=equity,
        equity_mask=mask_values["equity_called"],
        equity_slot=slot,
        to_call_zero=to_call_zero,
        read_temperature_x10=encoded,
    )


def load_phase_a_dataset_v9(path: str | Path) -> tuple[PhaseARowV9, ...]:
    """Load and validate a v9 Phase-A JSONL(.gz) dataset, fail-closed."""

    resolved = Path(path)
    opener = gzip.open if resolved.name.endswith(".gz") else open
    rows: list[PhaseARowV9] = []
    with opener(resolved, "rt", encoding="utf-8") as stream:  # type: ignore[operator]
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            document = json.loads(text)
            if not isinstance(document, Mapping):
                raise ValueError(f"line {line_number}: row is not an object")
            rows.append(_parse_row_v9(document, line_number))
    if not rows:
        raise ValueError(f"no rows in {resolved}")
    return tuple(rows)


# ---------------------------------------------------------------------------
# Normalization (schema-4 scoped; the split rule is shared with v8)
# ---------------------------------------------------------------------------


def context_normalization_v9(
    rows: Sequence[PhaseARowV9],
) -> tuple[list[float], list[float]]:
    """Schema-4 scales: context z-scored (floor 0.05), card identity.

    The v8 rule at the v9 partition — normalization arrays are
    schema-scoped and regenerated, never carried across (schema4's
    contract). Callers pass the training split only.
    """

    if not rows:
        raise ValueError("cannot compute normalization from zero rows")
    means = [0.0] * schema4.INPUT_SIZE_V9
    stds = [1.0] * schema4.INPUT_SIZE_V9
    count = len(rows)
    for index in schema4.CONTEXT_INDICES_V9:
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


def _mask_counts_v9(rows: Sequence[PhaseARowV9]) -> dict[str, int]:
    counts = {
        "rows": len(rows),
        "range_bucket": sum(row.range_mask for row in rows),
        "equity_called": sum(row.equity_mask for row in rows),
        "free_spot_rows": sum(1 for row in rows if row.to_call_zero),
    }
    for position, name in enumerate(_FT_LABEL_NAMES_V9):
        counts[name] = sum(row.fold_through_mask[position] for row in rows)
    return counts


# ---------------------------------------------------------------------------
# The network (one factory; the Phase-B fork reuses it, never re-states it)
# ---------------------------------------------------------------------------


def build_network_v9(dropout: float):
    """Construct the v9 network (torch), He-initialized, output heads zeroed.

    One definition for both trainer phases — the v8 line's twin network
    classes needed the same edit in the same commit to stay aligned, and
    a matching wrong edit on both sides would have passed parity cleanly.
    Call AFTER seeding torch: initialization draws from the global RNG.
    Same encoder/trunk/tower widths as v8 (the v9 change is the branch
    contract, never the shape); head sizes come from ``V9_HEAD_SIZES``.
    """

    import torch
    from torch import nn

    card_count = len(schema4.CARD_INDICES_V9)
    context_count = len(schema4.CONTEXT_INDICES_V9)

    class _NetworkV9(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.card_enc = nn.Linear(card_count, CARD_ENCODER_WIDTH)
            self.ctx_enc = nn.Linear(context_count, CONTEXT_ENCODER_WIDTH)
            self.card_ln = nn.LayerNorm(CARD_ENCODER_WIDTH)
            self.ctx_ln = nn.LayerNorm(CONTEXT_ENCODER_WIDTH)
            dims = [CARD_ENCODER_WIDTH + CONTEXT_ENCODER_WIDTH, *TRUNK_WIDTHS]
            self.trunk = nn.ModuleList(
                nn.Linear(dims[i], dims[i + 1]) for i in range(len(TRUNK_WIDTHS))
            )
            self.trunk_ln = nn.ModuleList(
                nn.LayerNorm(dims[i + 1]) for i in range(len(TRUNK_WIDTHS) - 1)
            )
            self.drop = nn.Dropout(float(dropout))
            self.towers = nn.ModuleDict(
                {
                    name: nn.Linear(TRUNK_WIDTHS[-1], HEAD_TOWER_WIDTH)
                    for name in V9_HEAD_SIZES
                }
            )
            self.outs = nn.ModuleDict(
                {
                    name: nn.Linear(HEAD_TOWER_WIDTH, size)
                    for name, size in V9_HEAD_SIZES.items()
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
                for name in V9_HEAD_SIZES
            }

    model = _NetworkV9()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=math.sqrt(2.0 / module.in_features))
                nn.init.zeros_(module.bias)
        # Zero output layers (all four heads, residual included): the model
        # starts at the constant predictor — the v7 recipe's largest single
        # init effect, kept verbatim.
        for name in V9_HEAD_SIZES:
            nn.init.zeros_(model.outs[name].weight)
            nn.init.zeros_(model.outs[name].bias)
    return model


def export_network_weights(model) -> dict[str, object]:
    """The JSON weight blocks for a fitted v9 network (either phase)."""

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
    return {
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
            for name in V9_HEAD_SIZES
        },
    }


# ---------------------------------------------------------------------------
# Torch fitting (function-local torch, offline_trainer.py's pattern)
# ---------------------------------------------------------------------------


def fit_phase_a_v9(
    rows: Sequence[PhaseARowV9], config: V8TrainingConfig
) -> dict[str, object]:
    """Fit one init seed; return weights, per-head losses, calibration.

    The v7/v8 optimization recipe verbatim (``v8_trainer.fit_phase_a``)
    over the v9 data contract: schema-4 partition, contract-derived head
    widths, equity slots by name. The residual head receives zero
    gradient — it belongs to the Phase-B composed objective.
    """

    check_v8_config(config)
    train_rows, validation_rows = split_rows(rows, config)
    if not train_rows:
        raise ValueError("training split is empty")
    if not validation_rows:
        raise ValueError(
            "validation split is empty; adjust validation_fraction or split_seed"
        )
    means, stds = context_normalization_v9(train_rows)

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

    card_indices = list(schema4.CARD_INDICES_V9)
    context_indices = list(schema4.CONTEXT_INDICES_V9)
    ft_width = len(FOLD_THROUGH_BRANCHES_V9)

    model = build_network_v9(config.dropout)
    model.to(device)

    mean_tensor = torch.tensor(means, dtype=torch.float32)
    std_tensor = torch.tensor(stds, dtype=torch.float32)

    def tensors(examples: Sequence[PhaseARowV9]) -> dict[str, object]:
        features = torch.tensor(
            [row.features for row in examples], dtype=torch.float32
        )
        features = ((features - mean_tensor) / std_tensor).to(device)
        return {
            "card": features[:, card_indices],
            "ctx": features[:, context_indices],
            "ft_target": torch.tensor(
                [[row.fold_through_label] * ft_width for row in examples],
                dtype=torch.float32,
                device=device,
            ),
            "ft_mask": torch.tensor(
                [
                    [float(flag) for flag in row.fold_through_mask]
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
            raise FloatingPointError("non-finite loss during v9 training")
        set_learning_rate()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite gradient during v9 training")
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
        ft_defined = sum(counts[name] for name in _FT_LABEL_NAMES_V9) > 0
        losses: dict[str, float | None] = {
            "fold_through": float(ft_loss) if ft_defined else None,
            "range": float(range_loss) if counts["range_bucket"] > 0 else None,
            "equity_called": (
                float(eq_loss) if counts["equity_called"] > 0 else None
            ),
        }
        losses["total"] = sum(value for value in losses.values() if value is not None)
        return losses

    train_counts = _mask_counts_v9(train_rows)
    validation_counts = _mask_counts_v9(validation_rows)

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
                raise FloatingPointError("non-finite parameter during v9 training")
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
            for branch in range(ft_width):
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

        bucket_predicted = [0.0] * schema4.BELIEF_BUCKETS
        bucket_observed = [0] * schema4.BELIEF_BUCKETS
        range_count = 0
        for row, probabilities in zip(validation_rows, range_probabilities):
            if not row.range_mask:
                continue
            range_count += 1
            bucket_observed[row.range_bucket] += 1
            for bucket in range(schema4.BELIEF_BUCKETS):
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
            for bucket in range(schema4.BELIEF_BUCKETS)
        ]

        eq_error = 0.0
        eq_count = 0
        slot_counts = [0] * len(EQUITY_SLOTS_V9)
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
                    for index, name in enumerate(EQUITY_SLOTS_V9)
                },
            },
        }

    weights = export_network_weights(model)
    _assert_finite_weights(weights, "v9 phase-A training")
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


def validate_v9_weight_shapes(weights: Mapping[str, object]) -> None:
    """Fail-closed structural check before a v9 artifact is written.

    A separate validator against the schema-4/contract constants, never a
    widened v8 one — numerically the two shapes agree today, which is
    exactly why the checks must diverge the moment either contract moves.
    """

    card = weights["card_encoder"]
    _require_matrix(
        "card_encoder.w", card["w"], CARD_ENCODER_WIDTH, schema4.CARD_BLOCK_SIZE_V9
    )
    _require_vector("card_encoder.b", card["b"], CARD_ENCODER_WIDTH)
    _require_vector("card_encoder.ln_g", card["ln_g"], CARD_ENCODER_WIDTH)
    _require_vector("card_encoder.ln_b", card["ln_b"], CARD_ENCODER_WIDTH)
    context = weights["context_encoder"]
    _require_matrix(
        "context_encoder.w",
        context["w"],
        CONTEXT_ENCODER_WIDTH,
        schema4.CONTEXT_BLOCK_SIZE_V9,
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
    if set(heads) != set(V9_HEAD_SIZES):
        raise ValueError(f"heads must be exactly {sorted(V9_HEAD_SIZES)}")
    for name, size in V9_HEAD_SIZES.items():
        head = heads[name]
        _require_matrix(
            f"heads.{name}.tower_w", head["tower_w"], HEAD_TOWER_WIDTH, TRUNK_WIDTHS[-1]
        )
        _require_vector(f"heads.{name}.tower_b", head["tower_b"], HEAD_TOWER_WIDTH)
        _require_matrix(f"heads.{name}.out_w", head["out_w"], size, HEAD_TOWER_WIDTH)
        _require_vector(f"heads.{name}.out_b", head["out_b"], size)


_REQUIRED_MANIFEST_KEYS_V9 = (
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
    "sizing",
    "weights_file",
    "weights_sha256",
    "training_window",
    "engine_parameters",
    "serve",
    "training",
    "evaluation",
    "promotion",
)


def validate_v9_manifest(manifest: Mapping[str, object]) -> None:
    """Structural contract for a format-4 candidate manifest.

    Mirrors ``v8_trainer.validate_v8_manifest`` at the v9 constants, plus
    the v9-only obligations: a mandatory, identity-checked composed
    ``sizing`` record (an artifact that cannot state its sizing cannot be
    served) and no inherited ``serve.margin_quantiles`` (v9 defines no
    hybrid mode). Accepts no state but ``"candidate"``: promotion is a
    separate, explicit, human-authorised act and never this module's.
    """

    for key in _REQUIRED_MANIFEST_KEYS_V9:
        if key not in manifest:
            raise ValueError(f"manifest is missing {key!r}")
    if manifest["format"] != MODEL_FORMAT:
        raise ValueError(f"format must be {MODEL_FORMAT!r}")
    if manifest["format_version"] != MODEL_FORMAT_VERSION_V9:
        raise ValueError(f"format_version must be {MODEL_FORMAT_VERSION_V9}")
    if manifest["state"] != "candidate":
        raise ValueError("a v9 trainer manifest state must be 'candidate'")
    if manifest["promotion"] is not None:
        raise ValueError("a v9 trainer manifest cannot carry a promotion record")
    if manifest["feature_schema_version"] != schema4.SCHEMA_VERSION_V9:
        raise ValueError(
            f"feature_schema_version must be {schema4.SCHEMA_VERSION_V9}"
        )
    if manifest["input_size"] != schema4.INPUT_SIZE_V9:
        raise ValueError(f"input_size must be {schema4.INPUT_SIZE_V9}")
    if list(manifest["feature_names"]) != list(schema4.FEATURE_NAMES_V9):
        raise ValueError("feature_names must match schema4.FEATURE_NAMES_V9")
    if list(manifest["action_labels"]) != list(BRANCH_LABELS_V9):
        raise ValueError(f"action_labels must be {list(BRANCH_LABELS_V9)}")
    validate_v9_architecture(manifest["architecture"])  # type: ignore[arg-type]
    sizing = manifest["sizing"]
    if not isinstance(sizing, Mapping):
        raise ValueError("manifest sizing must be a composed sizing record")
    try:
        parameters_and_rules_from_record(sizing)
    except ValueError as error:
        raise ValueError(f"manifest sizing record is invalid: {error}") from error
    serve = manifest.get("serve")
    if isinstance(serve, Mapping) and serve.get("margin_quantiles"):
        raise ValueError(
            "v9 defines no hybrid mode: serve.margin_quantiles must not be "
            "present"
        )
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


def _dataset_sidecar(dataset_path: str | Path | None) -> Mapping[str, object] | None:
    """The dataset's ``.summary.json`` sidecar document, when readable."""

    if dataset_path is None:
        return None
    sidecar_path = Path(dataset_path).parent / (
        Path(dataset_path).name.removesuffix(".jsonl.gz") + ".summary.json"
    )
    if not sidecar_path.is_file():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return sidecar if isinstance(sidecar, Mapping) else None


def resolve_sizing_record(
    sizing_record: Mapping[str, object] | None,
    sidecar: Mapping[str, object] | None,
) -> dict[str, object]:
    """The composed sizing record a v9 Phase-A manifest ships, fail-loud.

    Precedence: an explicit record must AGREE with the dataset sidecar's
    when the sidecar carries one (a disagreement means the features'
    baked costs and the manifest's declared sizing describe different
    g states — refuse, never pick); with no explicit record the
    sidecar's wins; the module default (every dial off) applies only
    when neither exists. Whatever is resolved must parse through
    ``parameters_and_rules_from_record`` — a bare or foreign-identity
    record refuses there.
    """

    def canonical(record: Mapping[str, object]) -> dict[str, object]:
        # JSON round-trip: a record fresh from composed_sizing_record()
        # carries tuples where one read back from a manifest or sidecar
        # carries lists. The canonical form is the on-disk one.
        return json.loads(json.dumps(record, sort_keys=True, allow_nan=False))

    recorded = sidecar.get("sizing") if sidecar is not None else None
    if recorded is not None and not isinstance(recorded, Mapping):
        raise ValueError("the dataset sidecar's sizing block is not a mapping")
    if sizing_record is not None and recorded is not None:
        if canonical(sizing_record) != canonical(recorded):
            raise ValueError(
                "the explicit sizing record disagrees with the dataset "
                "sidecar's — the dataset's baked costs and the manifest "
                "would describe different g states"
            )
    resolved = sizing_record if sizing_record is not None else recorded
    if resolved is None:
        resolved = composed_sizing_record()
    try:
        parameters_and_rules_from_record(resolved)
    except ValueError as error:
        raise ValueError(f"cannot resolve a sizing record: {error}") from error
    return canonical(resolved)


def train_phase_a_candidate_v9(
    rows: Sequence[PhaseARowV9],
    output_dir: str | Path,
    config: V8TrainingConfig = V8TrainingConfig(),
    init_seeds: Sequence[int] = (101, 202, 303),
    dataset_path: str | Path | None = None,
    sizing_record: Mapping[str, object] | None = None,
) -> dict[str, object]:
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
    sidecar = _dataset_sidecar(dataset_path)
    resolved_sizing = resolve_sizing_record(sizing_record, sidecar)
    output_path = Path(output_dir).expanduser().resolve()
    weights_path = output_path / f"{model_version}.weights.json"
    manifest_path = output_path / f"{model_version}.manifest.json"
    if weights_path.exists() or manifest_path.exists():
        raise FileExistsError(f"candidate artifact already exists for {model_version}")

    started = time.monotonic()
    results = []
    for init_seed in init_seeds:
        seed_config = replace(config, init_seed=init_seed)
        result = fit_phase_a_v9(rows, seed_config)
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
    validate_v9_weight_shapes(weights)
    architecture = default_v9_architecture()
    architecture["dropout"] = float(config.dropout)
    validate_v9_architecture(architecture)

    output_path.mkdir(parents=True, exist_ok=True)
    weights_document = _round9(
        {
            "format": MODEL_FORMAT,
            "format_version": MODEL_FORMAT_VERSION_V9,
            "model_version": model_version,
            "feature_normalization": {
                **schema4.normalization_stamp(),
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
    if sidecar is not None:
        generator_block = sidecar.get("generator")
        if isinstance(generator_block, Mapping):
            sidecar_generator = generator_block

    per_street_rows: dict[str, int] = {}
    for row in rows:
        per_street_rows[row.street] = per_street_rows.get(row.street, 0) + 1

    manifest = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION_V9,
        "model_version": model_version,
        "state": "candidate",
        "parent_version": None,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "feature_schema_version": schema4.SCHEMA_VERSION_V9,
        "input_size": schema4.INPUT_SIZE_V9,
        "feature_names": list(schema4.FEATURE_NAMES_V9),
        "action_labels": list(BRANCH_LABELS_V9),
        "architecture": architecture,
        "sizing": resolved_sizing,
        "weights_file": weights_path.name,
        "weights_sha256": weights_sha256,
        "training_window": {
            "dataset": None if dataset_path is None else str(dataset_path),
            "dataset_generator": (
                dict(sidecar_generator) if sidecar_generator else None
            ),
            "row_count": len(rows),
            "table_count": len({row.table_id for row in rows}),
            # The learning contract's own provenance key; same quantity as
            # ``table_count``. Emitted even though a Phase-A artifact is
            # never promotable, so the refusal comes from ``deployable``
            # below and not from an incidental missing field.
            "hand_count": len({row.table_id for row in rows}),
            "label_coverage": _mask_counts_v9(rows),
            "per_street_rows": dict(sorted(per_street_rows.items())),
        },
        "engine_parameters": {
            "safety_gates": DEFAULT_SAFETY_GATES.to_mapping(),
            "temperature_shaping": DEFAULT_TEMPERATURE_SHAPING.to_mapping(),
            "tracker_settings": DEFAULT_TRACKER_SETTINGS.to_mapping(),
            "bluff_settings": DEFAULT_BLUFF_SETTINGS.to_mapping(),
        },
        "serve": {
            # Machine-readable refusal. Phase-A trains the component heads
            # only; the composed-value serve path expects a Phase-B
            # artifact. The note below has said so since the fork, but a
            # note cannot stop a promotion and the promotion contract
            # reads THIS.
            "deployable": False,
            # By construction the OOD guard watches only the context block
            # (V8_DESIGN §3, unchanged): raw card one-hots never enter it.
            "ood_guard_indices": list(schema4.CONTEXT_INDICES_V9),
            "temperature": None,
            "note": (
                "Phase-A component heads only; the composed-value serve "
                "path (learned_policy_v9) expects a Phase-B artifact and "
                "this one must not be deployed"
            ),
        },
        "training": {
            "objective": TRAINING_OBJECTIVE_V9_PHASE_A,
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
            "heads_trained": list(SUPERVISED_HEADS_V9),
            "residual_head": (
                "zero-initialized output, excluded from every Phase-A loss "
                "(zero gradient); exported untrained for Phase B"
            ),
            "targets": {
                "fold_through": (
                    "masked BCE per wager lane (active, aggressive) on the "
                    "observed everyone-folded outcome; active supervises "
                    "only at to_call == 0 (its bet execution) and "
                    "aggressive only on priced rows — the loader rejects "
                    "violations"
                ),
                "range": (
                    "masked cross-entropy over 8 strength-percentile octiles "
                    "of the strongest continuing opponent holding "
                    "(strength_metric.strength_percentile)"
                ),
                "equity_called": (
                    "masked MSE on hero's exact/MC pot share against the "
                    "actual continuing holdings; slots (passive, active, "
                    "aggressive) routed by name — the taken wager lane for "
                    "sized wagers, passive for checked-through free spots, "
                    "active for every priced row without hero aggression "
                    "(folds share that conditional)"
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
    validate_v9_manifest(manifest)
    weights_path.write_bytes(encoded + b"\n")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": len(rows),
        "train_rows": int(best["train_rows"]),  # type: ignore[arg-type]
        "validation_rows": int(best["validation_rows"]),  # type: ignore[arg-type]
        "selected_init_seed": int(best["init_seed"]),  # type: ignore[arg-type]
        "train_losses": _rounded_losses(best["train_losses"]),  # type: ignore[arg-type]
        "validation_losses": _rounded_losses(best["validation_losses"]),  # type: ignore[arg-type]
        "seed_runs": tuple(_seed_run_record(result) for result in results),
        "calibration": best["calibration"],
        "weights_sha256": weights_sha256,
        "weights_path": weights_path,
        "manifest_path": manifest_path,
        "wall_time_seconds": time.monotonic() - started,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    defaults = V8TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Phase-A dataset (.jsonl.gz). Required: there is no safe "
            "default. It used to default to the Arena-built "
            "phase-a-dataset-v9.jsonl.gz, which the 2026-09-03 PHH switch "
            "retired, so a bare invocation trained on quarantined data and "
            "said nothing."
        ),
    )
    parser.add_argument(
        "--allow-retired-dataset",
        action="store_true",
        help=(
            "train on a corpus the project has retired (no live "
            "generator.source). Deliberate ablation only; record the arm."
        ),
    )
    parser.add_argument("--output-dir", default="artifacts/candidates")
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--init-seeds", type=int, nargs="+", default=[101, 202, 303]
    )
    parser.add_argument(
        "--sizing-record",
        default=None,
        help=(
            "path to a composed sizing record JSON; must agree with the "
            "dataset sidecar's record when one exists (default: sidecar, "
            "then the module default with every dial off)"
        ),
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
    sizing_record = None
    if args.sizing_record is not None:
        sizing_record = json.loads(
            Path(args.sizing_record).read_text(encoding="utf-8")
        )
    source = require_live_dataset(
        args.dataset, allow_retired=args.allow_retired_dataset
    )
    print(f"dataset provenance: {describe(args.dataset)}")
    rows = load_phase_a_dataset_v9(args.dataset)
    print(f"dataset: {args.dataset} ({len(rows)} rows, source {source})")
    summary = train_phase_a_candidate_v9(
        rows,
        args.output_dir,
        config,
        init_seeds=tuple(args.init_seeds),
        dataset_path=args.dataset,
        sizing_record=sizing_record,
    )
    print(f"selected init seed: {summary['selected_init_seed']}")
    print(f"train losses: {summary['train_losses']}")
    print(f"validation losses: {summary['validation_losses']}")
    print(f"manifest: {summary['manifest_path']}")
    print(f"weights: {summary['weights_path']}")
    print(f"weights_sha256: {summary['weights_sha256']}")
    print(f"wall_time_seconds: {summary['wall_time_seconds']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PhaseARowV9",
    "SUPERVISED_HEADS_V9",
    "TRAINING_OBJECTIVE_V9_PHASE_A",
    "build_network_v9",
    "context_normalization_v9",
    "export_network_weights",
    "fit_phase_a_v9",
    "load_phase_a_dataset_v9",
    "resolve_sizing_record",
    "train_phase_a_candidate_v9",
    "validate_v9_manifest",
    "validate_v9_weight_shapes",
]
