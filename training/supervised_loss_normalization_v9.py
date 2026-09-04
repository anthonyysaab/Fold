"""Constant-predictor normalizers for the v9 Phase-A supervised heads.

Phase 1 of ``artifacts/evaluations/v9-next-layer-plan-2026-09-02.md``
("fix the optimization, no architecture change"), pre-registered in
``artifacts/evaluations/v9-loss-rebalance-prereg-2026-09-02.md``.

The Phase-B objective divides the composed-value MSE by the training
targets' variance so that 1.0 means "the constant predictor" — and then
adds the three Phase-A supervised losses RAW. Measured on the schedule
sweep's control arm (2026-09-02, seed 401): ``range`` carries 48% of the
total validation loss, ``fold_through`` 21%, the value term 29% and
``equity_called`` 1.4%. Nobody chose that budget; it is the natural
scale of an 8-way NLL (ln 8), a BCE (ln 2) and an MSE on a [0, 1]
label. This module gives every supervised head the treatment the value
term already has: its loss divided by what a bias-only head would score
on the TRAINING split, so 1.0 means the constant predictor for all four
terms and a weight of 1.0 means equal footing.

The baselines, exactly — each is the masked loss the trainer computes,
minimized over a constant per output:

- ``fold_through``: one logit per wager lane, so the optimum is the
  mask-weighted label mean per lane; the baseline is the lane-count-
  weighted mean of the binary entropies (nats).
- ``range``: one logit per bucket, so the optimum is the bucket
  marginal over masked rows; the baseline is its entropy (nats).
- ``equity_called``: one output per slot, so the optimum is the
  per-slot label mean; the baseline is the pooled squared deviation
  about the slot means (the masked, slot-conditioned population
  variance).

Each baseline is floored at :data:`BASELINE_FLOOR` so a degenerate
split (a single label value, or a head with no masked rows at all)
cannot divide by zero; a head with no masked rows has zero loss anyway.
The unfloored value and the masked-row count are reported beside it so
a manifest shows when the floor engaged.

``raw`` mode is the shipped objective: every scale is the head's weight
alone (1.0 by default), which the trainer reduces to the original
unscaled sum so the 0001-0003 families reproduce bit-for-bit. The
default stays ``raw`` until the pre-registered experiment says
otherwise.

Stdlib only; nothing here imports torch, trains, or touches artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from training.v9_trainer import PhaseARowV9

#: The Phase-A heads the Phase-B trainer interleaves, in loss order.
SUPERVISED_HEADS_V9: tuple[str, ...] = ("fold_through", "range", "equity_called")

#: ``raw`` = the shipped objective (weights only); ``constant-predictor``
#: = weights divided by the training-split baselines.
SUPERVISED_NORMALIZATION_MODES: tuple[str, ...] = ("raw", "constant-predictor")

#: Floor on every baseline. Far below any real value (a BCE or NLL sits
#: near 0.1-2, the equity variance near 0.05) and only reachable on a
#: degenerate split.
BASELINE_FLOOR = 1e-3


@dataclass(frozen=True, slots=True)
class SupervisedLossConfigV9:
    """How the three Phase-A losses enter the Phase-B objective."""

    normalization: str = "raw"
    fold_through_weight: float = 1.0
    range_weight: float = 1.0
    equity_called_weight: float = 1.0

    def head_weights(self) -> dict[str, float]:
        return {
            "fold_through": float(self.fold_through_weight),
            "range": float(self.range_weight),
            "equity_called": float(self.equity_called_weight),
        }


def check_supervised_loss_config(config: SupervisedLossConfigV9) -> None:
    if config.normalization not in SUPERVISED_NORMALIZATION_MODES:
        raise ValueError(
            f"supervised normalization must be one of "
            f"{SUPERVISED_NORMALIZATION_MODES}, not {config.normalization!r}"
        )
    weights = config.head_weights()
    for name, value in weights.items():
        raw = getattr(config, f"{name}_weight")
        if isinstance(raw, bool) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} loss weight must be non-negative and finite")
    if all(value == 0.0 for value in weights.values()):
        raise ValueError(
            "at least one supervised head weight must be positive; switch the "
            "supervised term off with supervised_loss_weight 0 instead"
        )


@dataclass(frozen=True, slots=True)
class SupervisedBaselinesV9:
    """The constant predictor's masked losses on one row set.

    ``baselines`` is what the trainer divides by (floored);
    ``measured`` is the unfloored value; ``masked_rows`` counts the
    label-carrying rows (lane-rows for fold-through) each was computed
    from, so a floored head is visible in the manifest.
    """

    baselines: dict[str, float]
    measured: dict[str, float]
    masked_rows: dict[str, int]

    def as_record(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "baseline": self.baselines[name],
                "measured": self.measured[name],
                "masked_rows": self.masked_rows[name],
            }
            for name in SUPERVISED_HEADS_V9
        }


def _binary_entropy(probability: float) -> float:
    total = 0.0
    for q in (probability, 1.0 - probability):
        if q > 0.0:
            total -= q * math.log(q)
    return total


def constant_predictor_baselines(
    rows: Sequence[PhaseARowV9],
) -> SupervisedBaselinesV9:
    """The bias-only head's masked loss per Phase-A head (module docstring)."""

    lane_count: dict[int, float] = {}
    lane_positive: dict[int, float] = {}
    bucket_count: dict[int, float] = {}
    slot_values: dict[int, list[float]] = {}
    for row in rows:
        for lane, flag in enumerate(row.fold_through_mask):
            if flag:
                lane_count[lane] = lane_count.get(lane, 0.0) + 1.0
                lane_positive[lane] = lane_positive.get(lane, 0.0) + float(
                    row.fold_through_label
                )
        if row.range_mask:
            bucket_count[row.range_bucket] = bucket_count.get(row.range_bucket, 0.0) + 1.0
        if row.equity_mask:
            slot_values.setdefault(row.equity_slot, []).append(float(row.equity_called))

    ft_rows = sum(lane_count.values())
    ft_measured = 0.0
    if ft_rows > 0.0:
        ft_measured = sum(
            count * _binary_entropy(lane_positive[lane] / count)
            for lane, count in lane_count.items()
        ) / ft_rows

    range_rows = sum(bucket_count.values())
    range_measured = 0.0
    if range_rows > 0.0:
        range_measured = -sum(
            (count / range_rows) * math.log(count / range_rows)
            for count in bucket_count.values()
        )

    eq_rows = sum(len(values) for values in slot_values.values())
    eq_measured = 0.0
    if eq_rows > 0:
        deviation = 0.0
        for values in slot_values.values():
            mean = sum(values) / len(values)
            deviation += sum((value - mean) ** 2 for value in values)
        eq_measured = deviation / eq_rows

    measured = {
        "fold_through": ft_measured,
        "range": range_measured,
        "equity_called": eq_measured,
    }
    return SupervisedBaselinesV9(
        baselines={name: max(value, BASELINE_FLOOR) for name, value in measured.items()},
        measured=measured,
        masked_rows={
            "fold_through": int(ft_rows),
            "range": int(range_rows),
            "equity_called": int(eq_rows),
        },
    )


def supervised_head_scales(
    config: SupervisedLossConfigV9, baselines: Mapping[str, float]
) -> dict[str, float]:
    """The multiplier each head's loss gets before the supervised sum.

    ``raw``: the head weight alone (the baselines are ignored, so the
    shipped defaults give exactly 1.0 per head). ``constant-predictor``:
    weight / baseline.
    """

    check_supervised_loss_config(config)
    weights = config.head_weights()
    if config.normalization == "raw":
        return {name: weights[name] for name in SUPERVISED_HEADS_V9}
    return {
        name: weights[name] / float(baselines[name]) for name in SUPERVISED_HEADS_V9
    }
