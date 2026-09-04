"""Phase-B composed-value trainer for the v8 network (V8_DESIGN §5 Phase B).

Additive sibling of ``v8_trainer`` — the Phase-A path is untouched. This
module trains the same ``_NetworkV8`` architecture with the **composed-value
objective**: the value loss is applied to the centered Phase-B branch targets
*through* the §4 fixed arithmetic composition, so gradient reaches the
component heads only via the roles the composition assigns them
(``fold_through`` as fold equity, ``equity_called`` as showdown share,
``residual`` as the capped correction). The supervised Phase-A component
losses keep flowing throughout from the Phase-A dataset (V8_DESIGN §5:
"supervised component losses keep flowing from Phase A data"), so the heads
stay pinned to their observable ground truth while the composition learns
values.

Total loss per optimizer step::

    L = value_loss_weight · MSE_composed / Var(targets)      (Phase-B batch)
      + supervised_loss_weight · (L_ft + L_range + L_eq)     (Phase-A batch)
      + decoupled AdamW weight decay, with a separate rate on the
        residual head (V8_DESIGN §5: "weight decay on the residual")

Design decisions, each documented in the exported manifest and ablatable
from the CLI rather than trusted:

- **Value-loss normalization is estimated, not guessed.** The raw composed
  MSE lives in purse-normalized units (targets ~1e-2), three orders below
  the supervised losses; a hand-tuned weight would be a guessed constant.
  Instead the MSE is divided by the *population variance of the Phase-B
  training targets* (computed once from the training split), so a weight of
  1.0 means "equal footing with the supervised losses at the constant
  predictor". Both weights stay ablatable (``--value-loss-weight``,
  ``--supervised-loss-weight``).
- **Residual weight decay** applies to every parameter of the residual head
  (tower and output, biases included — zero is the correct prior for a
  correction head, so the usual no-bias-decay rule is deliberately not
  honoured there; everything else keeps the v7 rule). Rate
  ``--residual-weight-decay`` (default 0.1, 10x the base decay) — a
  hand-chosen regularizer strength, exposed for the §6 ablation battery,
  never on the value path itself.
- **The composition in torch mirrors ``compose_branch_values`` exactly**
  (sigmoid on fold-through logits, equities clamped into [0, 1], residual
  clamped at ``±cap·pot`` in purse units, centering over the emitted branch
  set). After export the trainer replays up to ``--parity-sample``
  validation decisions through the *stdlib* serve path
  (``_forward_v3`` + ``compose_branch_values``) and fails closed if any
  composed branch value disagrees with the torch path beyond tolerance —
  train/serve parity is checked, not assumed.

Corpus contract: the loader re-validates every Phase-B row fail-closed and
re-derives the E6 sizing (target, clamped to-amount, wager, pot-if-called)
from the row's own context, then asserts the derived numbers match the
harvester's recorded ``e6_target`` / ``e6_to_amount`` — an
impossible-by-construction cross-check that the trainer's arithmetic and the
harvester's arithmetic are the same arithmetic.

Torch imports are function-local (the ``offline_trainer`` pattern): the
module imports cleanly on the stdlib interpreter; training runs in the CUDA
venv. Artifacts are immutable candidates: state ``"candidate"``, promotion
``null``, ``artifacts/approved.json`` never read or written.

Usage (CUDA venv, repo root)::

    python -m training.v8_trainer_phase_b \
        --model-version candidate-v8-0002a --init-seeds 401
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
from training.offline_trainer import (
    _assert_finite_weights,
    _round9,
    validate_training_device,
)
from engine.opponent_model import DEFAULT_TRACKER_SETTINGS
from training.v8_trainer import (
    PhaseARow,
    V8TrainingConfig,
    check_v8_config,
    context_normalization,
    default_v8_architecture,
    load_phase_a_dataset,
    split_rows,
    table_split_value,
    validate_v8_manifest,
    validate_v8_weight_shapes,
)
from engine.architecture_v8 import (
    BRANCH_LABELS_V8,
    EQUITY_SLOTS,
    FOLD_THROUGH_BRANCHES,
    MODEL_FORMAT_VERSION_V8,
    V8_HEAD_SIZES,
    validate_v8_architecture,
)

TRAINING_OBJECTIVE_V8_PHASE_B = "phase_b_composed_value_v8"

#: Default corpus/dataset locations (repo-root relative, the project rule).
DEFAULT_PHASE_B_CORPUS = (
    Path("artifacts") / "phase_b" / "candidate-v8-0002.phase-b.jsonl.gz"
)
DEFAULT_PHASE_A_DATASET = (
    Path("artifacts") / "phase_a" / "phase-a-dataset.jsonl.gz"
)

#: §4 residual cap, duplicated numerically from
#: ``learned_policy_v8.RESIDUAL_CAP_POT_FRACTION`` (importing it would pull
#: the serve module into the trainer at import time; the parity check
#: imports lazily at run time and asserts through the real serve path, so a
#: drift between the two constants cannot survive a training run).
RESIDUAL_CAP_POT_FRACTION_DEFAULT = 0.05

_AGGRESS_BRANCHES = ("aggress_small", "aggress_large")
_E6_SPECS: dict[str, tuple[float, float]] = {
    "aggress_small": _BRANCH_SMALL,
    "aggress_large": _BRANCH_LARGE,
}
_E6_TOLERANCE = 1e-6
_CENTER_TOLERANCE = 1e-4
_PARITY_TOLERANCE = 1e-3
_STREETS = ("preflop", "flop", "turn", "river")


@dataclass(frozen=True, slots=True)
class PhaseBDecision:
    """One validated Phase-B decision with derived composition constants.

    All ``*_unit`` quantities are purse-normalized (chips / purse), exactly
    the units ``compose_branch_values`` works in. Aggress entries are ``None``
    when the branch was not emitted for this decision.
    """

    decision_id: str
    table_id: str
    street: str
    features: tuple[float, ...]
    emitted: tuple[str, ...]  # subset of BRANCH_LABELS_V8, corpus order
    targets: dict[str, float]  # centered reward, purse units, emitted only
    pot_unit: float
    cc_pot_unit: float  # (pot + to_call) / purse
    cc_cost_unit: float  # to_call / purse
    wager_unit: tuple[float | None, float | None]  # (small, large)
    pot_if_called_unit: tuple[float | None, float | None]
    context: dict[str, Any]  # raw ints for the stdlib parity check


@dataclass(frozen=True, slots=True)
class PhaseBTrainingConfig:
    """Phase-B knobs on top of the shared ``V8TrainingConfig``."""

    base: V8TrainingConfig
    value_loss_weight: float = 1.0
    supervised_loss_weight: float = 1.0
    residual_weight_decay: float = 0.1
    residual_cap_pot_fraction: float = RESIDUAL_CAP_POT_FRACTION_DEFAULT
    phase_a_batch_size: int = 256
    parity_sample: int = 64


def check_phase_b_config(config: PhaseBTrainingConfig) -> None:
    check_v8_config(config.base)
    for name in ("value_loss_weight", "supervised_loss_weight"):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be non-negative and finite")
    if config.value_loss_weight == 0.0 and config.supervised_loss_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")
    if (
        not math.isfinite(config.residual_weight_decay)
        or config.residual_weight_decay < 0.0
    ):
        raise ValueError("residual_weight_decay must be non-negative and finite")
    if (
        not math.isfinite(config.residual_cap_pot_fraction)
        or config.residual_cap_pot_fraction < 0.0
    ):
        raise ValueError("residual_cap_pot_fraction must be non-negative and finite")
    if config.phase_a_batch_size < 1:
        raise ValueError("phase_a_batch_size must be positive")
    if config.parity_sample < 1:
        raise ValueError("parity_sample must be positive")


# ---------------------------------------------------------------------------
# Corpus loading — fail-closed, with the E6 cross-check
# ---------------------------------------------------------------------------


def _finite(value: object, name: str, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{where}: {name} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where}: {name} is not finite")
    return number


def _derive_branch_units(
    context: Mapping[str, Any], branch: str, where: str, recorded: Mapping[str, Any]
) -> tuple[float, float, float]:
    """(wager, pot_if_called, e6_target) in chips, cross-checked vs the corpus.

    Reproduces ``compose_branch_values``'s arithmetic from the decision
    context, then asserts the harvester recorded the same target and clamped
    to-amount — the trainer's sizing and the harvest's sizing must be one
    arithmetic or the labels do not mean what the composition says.
    """

    pot = int(context["pot"])
    to_call = int(context["to_call"])
    contribution = int(context["contribution"])
    eff = max(1, int(context["effective_stack"]))
    legal = context.get("legal_range")
    if legal is None:
        raise ValueError(f"{where}: aggress branch without a legal range")
    low, high = int(legal[0]), int(legal[1])
    pot_fraction, stack_fraction = _E6_SPECS[branch]
    target = min(to_call + pot_fraction * (pot + to_call), stack_fraction * eff)
    to_amount = min(high, max(low, contribution + target))
    recorded_target = _finite(recorded.get("e6_target"), "e6_target", where)
    recorded_to = _finite(recorded.get("e6_to_amount"), "e6_to_amount", where)
    if abs(recorded_target - target) > _E6_TOLERANCE:
        raise ValueError(
            f"{where}: derived E6 target {target!r} does not match the "
            f"corpus's {recorded_target!r}"
        )
    if abs(recorded_to - to_amount) > _E6_TOLERANCE:
        raise ValueError(
            f"{where}: derived E6 to-amount {to_amount!r} does not match "
            f"the corpus's {recorded_to!r}"
        )
    wager = max(0.0, to_amount - contribution)
    pot_if_called = pot + 2.0 * wager - to_call
    return wager, pot_if_called, target


def _parse_decision(row: Mapping[str, Any], line: int) -> PhaseBDecision:
    where = f"line {line}"
    decision_id = row.get("decision_id")
    if not isinstance(decision_id, str) or not decision_id:
        raise ValueError(f"{where}: missing decision_id")
    where = f"line {line} ({decision_id})"
    table_id = row.get("table_id")
    if not isinstance(table_id, str) or not table_id:
        raise ValueError(f"{where}: missing table_id")
    street = row.get("street")
    if street not in _STREETS:
        raise ValueError(f"{where}: unknown street {street!r}")

    features = row.get("features")
    if not isinstance(features, list) or len(features) != schema3.INPUT_SIZE_V8:
        raise ValueError(
            f"{where}: features must be {schema3.INPUT_SIZE_V8} floats"
        )
    vector = tuple(_finite(value, "feature", where) for value in features)

    context = row.get("context")
    if not isinstance(context, Mapping):
        raise ValueError(f"{where}: missing context")
    purse = int(_finite(context.get("purse"), "purse", where))
    if purse < 1:
        raise ValueError(f"{where}: purse must be positive")
    big_blind = _finite(row.get("big_blind"), "big_blind", where)
    purse_bb = _finite(row.get("purse_bb"), "purse_bb", where)
    if big_blind <= 0 or purse_bb <= 0:
        raise ValueError(f"{where}: big_blind and purse_bb must be positive")
    if abs(purse_bb * big_blind - purse) > 1e-6 * max(1.0, purse):
        raise ValueError(
            f"{where}: purse_bb {purse_bb} x big_blind {big_blind} does not "
            f"reproduce context purse {purse}"
        )
    pot = int(_finite(context.get("pot"), "pot", where))
    to_call = int(_finite(context.get("to_call"), "to_call", where))

    branches = row.get("branches")
    if not isinstance(branches, list) or len(branches) < 2:
        raise ValueError(f"{where}: needs at least two emitted branches")
    emitted: list[str] = []
    targets: dict[str, float] = {}
    wager: dict[str, float] = {}
    pot_if_called: dict[str, float] = {}
    total_reward = 0.0
    for entry in branches:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where}: branch entry is not an object")
        label = entry.get("branch")
        if label not in BRANCH_LABELS_V8:
            raise ValueError(f"{where}: unknown branch label {label!r}")
        if label in emitted:
            raise ValueError(f"{where}: duplicate branch {label!r}")
        emitted.append(str(label))
        reward_bb = _finite(entry.get("reward_bb"), "reward_bb", where)
        total_reward += reward_bb
        # Purse units: reward_bb / purse_bb == reward_chips / purse — the
        # exact normalization the v7 targets used and §4 composes in.
        targets[str(label)] = reward_bb / purse_bb
        if label in _AGGRESS_BRANCHES:
            chips_wager, chips_pic, _ = _derive_branch_units(
                context, str(label), where, entry
            )
            wager[str(label)] = chips_wager / purse
            pot_if_called[str(label)] = chips_pic / purse
    if abs(total_reward) > _CENTER_TOLERANCE:
        raise ValueError(
            f"{where}: centered rewards sum to {total_reward!r}, not ~0"
        )

    return PhaseBDecision(
        decision_id=decision_id,
        table_id=table_id,
        street=str(street),
        features=vector,
        emitted=tuple(emitted),
        targets=targets,
        pot_unit=pot / purse,
        cc_pot_unit=(pot + to_call) / purse,
        cc_cost_unit=to_call / purse,
        wager_unit=(
            wager.get("aggress_small"),
            wager.get("aggress_large"),
        ),
        pot_if_called_unit=(
            pot_if_called.get("aggress_small"),
            pot_if_called.get("aggress_large"),
        ),
        context={
            "pot": pot,
            "to_call": to_call,
            "contribution": int(_finite(context.get("contribution"), "contribution", where)),
            "effective_stack": int(
                _finite(context.get("effective_stack"), "effective_stack", where)
            ),
            "purse": purse,
            "legal_range": (
                None
                if context.get("legal_range") is None
                else (int(context["legal_range"][0]), int(context["legal_range"][1]))
            ),
        },
    )


def load_phase_b_decisions(path: str | Path) -> tuple[PhaseBDecision, ...]:
    """Load and validate a Phase-B corpus into training decisions, fail-closed."""

    resolved = Path(path)
    decisions: list[PhaseBDecision] = []
    seen: set[str] = set()
    with gzip.open(resolved, "rt", encoding="utf-8") as stream:
        header = json.loads(stream.readline())
        if not isinstance(header, Mapping) or header.get("kind") != "phase-b-corpus":
            raise ValueError(f"{resolved}: not a phase-b corpus")
        if header.get("corpus_schema_version") != 1:
            raise ValueError(f"{resolved}: unsupported corpus schema version")
        if header.get("feature_schema_version") != schema3.SCHEMA_VERSION:
            raise ValueError(f"{resolved}: corpus feature schema is not schema 3")
        if header.get("input_size") != schema3.INPUT_SIZE_V8:
            raise ValueError(f"{resolved}: corpus input size does not match schema 3")
        if list(header.get("branch_labels") or []) != list(BRANCH_LABELS_V8):
            raise ValueError(f"{resolved}: corpus branch labels are not the v8 set")
        for line_number, line in enumerate(stream, start=2):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, Mapping):
                raise ValueError(f"line {line_number}: row is not an object")
            decision = _parse_decision(row, line_number)
            if decision.decision_id in seen:
                raise ValueError(
                    f"line {line_number}: duplicate decision_id "
                    f"{decision.decision_id!r}"
                )
            seen.add(decision.decision_id)
            decisions.append(decision)
    if not decisions:
        raise ValueError(f"no decisions in {resolved}")
    return tuple(decisions)


def split_decisions(
    decisions: Sequence[PhaseBDecision], config: V8TrainingConfig
) -> tuple[list[PhaseBDecision], list[PhaseBDecision]]:
    """The Phase-A table-hash split rule applied to Phase-B tables."""

    validation_tables = {
        table_id
        for table_id in {decision.table_id for decision in decisions}
        if table_split_value(config.split_seed, table_id)
        < config.validation_fraction
    }
    train = [d for d in decisions if d.table_id not in validation_tables]
    validation = [d for d in decisions if d.table_id in validation_tables]
    return train, validation


def value_target_variance(decisions: Sequence[PhaseBDecision]) -> float:
    """Population variance of every emitted branch target (purse units).

    The estimated normalizer for the composed-value MSE: dividing by this
    makes the loss ~1.0 at the constant (zero) predictor, which is what
    puts a 1.0 loss weight on equal footing with the supervised losses.
    Floored to avoid dividing by a degenerate corpus.
    """

    values = [
        target
        for decision in decisions
        for target in decision.targets.values()
    ]
    if not values:
        raise ValueError("no branch targets to compute a variance from")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return max(variance, 1e-8)


def compose_from_constants(
    head_outputs: Mapping[str, Sequence[float]],
    decision: PhaseBDecision,
    *,
    use_residual: bool = True,
    residual_cap_pot_fraction: float = RESIDUAL_CAP_POT_FRACTION_DEFAULT,
) -> dict[str, float]:
    """Pure-float composition over the decision's emitted branches.

    Same arithmetic as the torch path and as
    ``learned_policy_v8.compose_branch_values``; exists so the constants
    derivation is testable on the stdlib interpreter, and used by the
    export-time parity check as the torch-free twin.
    """

    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        expv = math.exp(value)
        return expv / (1.0 + expv)

    def _clip01(value: float) -> float:
        return min(1.0, max(0.0, value))

    fold_through = head_outputs["fold_through"]
    equity_called = head_outputs["equity_called"]
    residual = head_outputs["residual"]
    cap = abs(residual_cap_pot_fraction) * decision.pot_unit
    values: dict[str, float] = {}
    for label in decision.emitted:
        if label == "fold":
            values[label] = 0.0
        elif label == "check_call":
            equity_cc = _clip01(float(equity_called[EQUITY_SLOTS.index("check_call")]))
            values[label] = (
                equity_cc * decision.cc_pot_unit - decision.cc_cost_unit
            )
        else:
            slot = _AGGRESS_BRANCHES.index(label)
            wager = decision.wager_unit[slot]
            pot_if_called = decision.pot_if_called_unit[slot]
            assert wager is not None and pot_if_called is not None
            p_ft = _sigmoid(
                float(fold_through[FOLD_THROUGH_BRANCHES.index(label)])
            )
            equity_k = _clip01(float(equity_called[EQUITY_SLOTS.index(label)]))
            value = p_ft * decision.pot_unit + (1.0 - p_ft) * (
                equity_k * pot_if_called - wager
            )
            if use_residual:
                correction = float(residual[BRANCH_LABELS_V8.index(label)])
                value += min(cap, max(-cap, correction))
            values[label] = value
    return values


# ---------------------------------------------------------------------------
# Torch fitting
# ---------------------------------------------------------------------------


def fit_phase_b(
    phase_b: Sequence[PhaseBDecision],
    phase_a: Sequence[PhaseARow],
    config: PhaseBTrainingConfig,
) -> dict[str, object]:
    """Fit one init seed with the joint composed + supervised objective."""

    check_phase_b_config(config)
    base = config.base
    pb_train, pb_validation = split_decisions(phase_b, base)
    pa_train, pa_validation = split_rows(phase_a, base)
    if not pb_train or not pb_validation:
        raise ValueError("phase-b split produced an empty train or validation set")
    if not pa_train or not pa_validation:
        raise ValueError("phase-a split produced an empty train or validation set")
    # Context z-scores from the union of both training splits (documented in
    # the manifest): both loss terms consume the same normalization, and the
    # exported scales must serve every input the model was fitted on.
    means, stds = context_normalization([*pa_train, *pb_train])
    target_variance = value_target_variance(pb_train)

    import torch
    from torch import nn

    if base.device == "cuda":
        device_name = validate_training_device("cuda")
    else:
        device_name = "cpu"
    device = torch.device(base.device)
    torch.manual_seed(base.init_seed)
    if base.device == "cuda":
        torch.cuda.manual_seed_all(base.init_seed)

    card_indices = list(schema3.CARD_INDICES)
    context_indices = list(schema3.CONTEXT_INDICES)
    from training.v8_trainer import (
        CARD_ENCODER_WIDTH,
        CONTEXT_ENCODER_WIDTH,
        HEAD_TOWER_WIDTH,
        TRUNK_WIDTHS,
    )

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
            self.drop = nn.Dropout(float(base.dropout))
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
        for name in V8_HEAD_SIZES:
            nn.init.zeros_(model.outs[name].weight)
            nn.init.zeros_(model.outs[name].bias)
    model.to(device)

    mean_tensor = torch.tensor(means, dtype=torch.float32)
    std_tensor = torch.tensor(stds, dtype=torch.float32)

    def normalized_features(vectors: Sequence[Sequence[float]]):
        features = torch.tensor(list(vectors), dtype=torch.float32)
        return ((features - mean_tensor) / std_tensor).to(device)

    # ----- Phase-B tensors -------------------------------------------------
    def phase_b_tensors(decisions: Sequence[PhaseBDecision]) -> dict[str, object]:
        features = normalized_features([d.features for d in decisions])
        count = len(decisions)
        mask = torch.zeros((count, 4), dtype=torch.float32)
        target = torch.zeros((count, 4), dtype=torch.float32)
        pot_unit = torch.zeros(count, dtype=torch.float32)
        cc_pot = torch.zeros(count, dtype=torch.float32)
        cc_cost = torch.zeros(count, dtype=torch.float32)
        wager = torch.zeros((count, 2), dtype=torch.float32)
        pic = torch.zeros((count, 2), dtype=torch.float32)
        for index, decision in enumerate(decisions):
            pot_unit[index] = decision.pot_unit
            cc_pot[index] = decision.cc_pot_unit
            cc_cost[index] = decision.cc_cost_unit
            for label in decision.emitted:
                slot = BRANCH_LABELS_V8.index(label)
                mask[index, slot] = 1.0
                target[index, slot] = decision.targets[label]
            for aggress_slot, label in enumerate(_AGGRESS_BRANCHES):
                if decision.wager_unit[aggress_slot] is not None:
                    wager[index, aggress_slot] = decision.wager_unit[aggress_slot]
                    pic[index, aggress_slot] = decision.pot_if_called_unit[
                        aggress_slot
                    ]
        return {
            "card": features[:, card_indices],
            "ctx": features[:, context_indices],
            "mask": mask.to(device),
            "target": target.to(device),
            "pot_unit": pot_unit.to(device),
            "cc_pot": cc_pot.to(device),
            "cc_cost": cc_cost.to(device),
            "wager": wager.to(device),
            "pic": pic.to(device),
        }

    cap_fraction = abs(float(config.residual_cap_pot_fraction))

    def composed_values(outputs, data, indexes):
        """[batch, 4] composed values, mirroring compose_branch_values."""

        pot_unit = data["pot_unit"][indexes]
        equity = torch.clamp(outputs["equity_called"], 0.0, 1.0)
        p_ft = torch.sigmoid(outputs["fold_through"])
        cap = cap_fraction * pot_unit
        residual = torch.clamp(
            outputs["residual"][:, 2:4],
            min=-cap.unsqueeze(1),
            max=cap.unsqueeze(1),
        )
        v_fold = torch.zeros_like(pot_unit)
        v_cc = (
            equity[:, EQUITY_SLOTS.index("check_call")] * data["cc_pot"][indexes]
            - data["cc_cost"][indexes]
        )
        wager = data["wager"][indexes]
        pic = data["pic"][indexes]
        # equity slots 0/1 are aggress_small/aggress_large by EQUITY_SLOTS.
        v_aggress = p_ft * pot_unit.unsqueeze(1) + (1.0 - p_ft) * (
            equity[:, 0:2] * pic - wager
        )
        v_aggress = v_aggress + residual
        return torch.stack(
            [v_fold, v_cc, v_aggress[:, 0], v_aggress[:, 1]], dim=1
        )

    def value_loss(data, indexes):
        outputs = model(data["card"][indexes], data["ctx"][indexes])
        values = composed_values(outputs, data, indexes)
        mask = data["mask"][indexes]
        counts = mask.sum(dim=1).clamp(min=1.0)
        centered = values - (values * mask).sum(dim=1, keepdim=True) / counts.unsqueeze(1)
        errors = (centered - data["target"][indexes]).square() * mask
        per_decision = errors.sum(dim=1) / counts
        return per_decision.mean()

    # ----- Phase-A tensors (the v8_trainer construction, verbatim) ---------
    def phase_a_tensors(rows: Sequence[PhaseARow]) -> dict[str, object]:
        features = normalized_features([row.features for row in rows])
        return {
            "card": features[:, card_indices],
            "ctx": features[:, context_indices],
            "ft_target": torch.tensor(
                [[row.fold_through_label] * 2 for row in rows],
                dtype=torch.float32,
                device=device,
            ),
            "ft_mask": torch.tensor(
                [
                    [float(row.fold_through_mask[0]), float(row.fold_through_mask[1])]
                    for row in rows
                ],
                dtype=torch.float32,
                device=device,
            ),
            "range_target": torch.tensor(
                [row.range_bucket for row in rows], dtype=torch.long, device=device
            ),
            "range_mask": torch.tensor(
                [float(row.range_mask) for row in rows],
                dtype=torch.float32,
                device=device,
            ),
            "eq_target": torch.tensor(
                [row.equity_called for row in rows],
                dtype=torch.float32,
                device=device,
            ),
            "eq_slot": torch.tensor(
                [row.equity_slot for row in rows], dtype=torch.long, device=device
            ),
            "eq_mask": torch.tensor(
                [float(row.equity_mask) for row in rows],
                dtype=torch.float32,
                device=device,
            ),
        }

    def supervised_losses(data, indexes):
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

    pb_train_data = phase_b_tensors(pb_train)
    pb_validation_data = phase_b_tensors(pb_validation)
    pa_train_data = phase_a_tensors(pa_train)
    pa_validation_data = phase_a_tensors(pa_validation)

    # ----- Optimizer: three decoupled decay groups -------------------------
    residual_parameters, decay, no_decay = [], [], []
    for name, parameter in model.named_parameters():
        if "residual" in name:
            residual_parameters.append(parameter)
        elif name.endswith("bias") or "_ln" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(base.weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
            {
                "params": residual_parameters,
                "weight_decay": float(config.residual_weight_decay),
            },
        ],
        lr=base.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    steps_per_epoch = max(1, math.ceil(len(pb_train) / base.batch_size))
    total_steps = max(1, steps_per_epoch * base.epochs)
    step = 0

    def set_learning_rate() -> None:
        if step < base.warmup_steps:
            factor = (step + 1) / max(1, base.warmup_steps)
        else:
            progress = (step - base.warmup_steps) / max(
                1, total_steps - base.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            factor = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = base.learning_rate * factor

    def optimize(loss) -> None:
        nonlocal step
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss during v8 phase-b training")
        set_learning_rate()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite gradient during v8 phase-b training")
        optimizer.step()
        step += 1

    weight_value = float(config.value_loss_weight)
    weight_supervised = float(config.supervised_loss_weight)

    def evaluated_losses(pb_data, pa_data) -> dict[str, float]:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            pb_indexes = torch.arange(pb_data["mask"].shape[0], device=device)
            raw_value = float(value_loss(pb_data, pb_indexes))
            pa_indexes = torch.arange(pa_data["range_mask"].shape[0], device=device)
            ft_loss, range_loss, eq_loss = supervised_losses(pa_data, pa_indexes)
        if was_training:
            model.train()
        normalized = raw_value / target_variance
        supervised_total = float(ft_loss) + float(range_loss) + float(eq_loss)
        return {
            "value_mse": raw_value,
            "value_normalized": normalized,
            "fold_through": float(ft_loss),
            "range": float(range_loss),
            "equity_called": float(eq_loss),
            "supervised_total": supervised_total,
            "total": weight_value * normalized + weight_supervised * supervised_total,
        }

    pa_order_rng = random.Random(base.init_seed + 2)
    pa_order: list[int] = []

    def next_pa_batch() -> list[int]:
        nonlocal pa_order
        batch: list[int] = []
        while len(batch) < min(config.phase_a_batch_size, len(pa_train)):
            if not pa_order:
                pa_order = list(range(len(pa_train)))
                pa_order_rng.shuffle(pa_order)
            batch.append(pa_order.pop())
        return batch

    generator = random.Random(base.init_seed + 1)
    best_loss = math.inf
    best_state: dict[str, object] | None = None
    best_epoch = 0
    stale_epochs = 0
    epochs_run = 0
    model.train()
    for epoch in range(base.epochs):
        epochs_run = epoch + 1
        order = list(range(len(pb_train)))
        generator.shuffle(order)
        for start in range(0, len(order), base.batch_size):
            pb_indexes = torch.tensor(
                order[start : start + base.batch_size],
                dtype=torch.long,
                device=device,
            )
            loss = torch.zeros((), device=device)
            if weight_value > 0.0:
                loss = loss + weight_value * (
                    value_loss(pb_train_data, pb_indexes) / target_variance
                )
            if weight_supervised > 0.0:
                pa_indexes = torch.tensor(
                    next_pa_batch(), dtype=torch.long, device=device
                )
                ft_loss, range_loss, eq_loss = supervised_losses(
                    pa_train_data, pa_indexes
                )
                loss = loss + weight_supervised * (ft_loss + range_loss + eq_loss)
            optimize(loss)
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(
                    "non-finite parameter during v8 phase-b training"
                )
        epoch_losses = evaluated_losses(pb_validation_data, pa_validation_data)
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
            if stale_epochs >= base.early_stop_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    train_losses = evaluated_losses(pb_train_data, pa_train_data)
    validation_losses = evaluated_losses(pb_validation_data, pa_validation_data)

    # ----- Residual audit (V8_DESIGN §6.5) --------------------------------
    def residual_share() -> dict[str, object]:
        with torch.no_grad():
            indexes = torch.arange(pb_validation_data["mask"].shape[0], device=device)
            outputs = model(
                pb_validation_data["card"][indexes],
                pb_validation_data["ctx"][indexes],
            )
            values = composed_values(outputs, pb_validation_data, indexes)
            pot_unit = pb_validation_data["pot_unit"][indexes]
            cap = cap_fraction * pot_unit
            residual = torch.clamp(
                outputs["residual"][:, 2:4],
                min=-cap.unsqueeze(1),
                max=cap.unsqueeze(1),
            )
            aggress_mask = pb_validation_data["mask"][indexes][:, 2:4]
            abs_residual = (residual.abs() * aggress_mask).sum()
            abs_value = (values[:, 2:4].abs() * aggress_mask).sum()
            branches = aggress_mask.sum()
        return {
            "aggress_branches": int(branches),
            "sum_abs_capped_residual": round(float(abs_residual), 6),
            "sum_abs_composed_value": round(float(abs_value), 6),
            "share_of_abs_composed_value": (
                round(float(abs_residual) / float(abs_value), 6)
                if float(abs_value) > 0
                else None
            ),
        }

    def export_weights() -> dict[str, object]:
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
                for name in V8_HEAD_SIZES
            },
        }

    # ----- Train/serve parity: torch vs the stdlib compose path ------------
    def parity_check(weights: Mapping[str, object]) -> dict[str, object]:
        """Replay validation decisions through the stdlib serve arithmetic.

        The exported weights (rounded exactly as the artifact is) drive
        ``learned_policy_v8._forward_v3`` + ``compose_branch_values``; the
        torch model in eval mode drives ``composed_values``. Any branch
        value diverging beyond tolerance fails the export — the §4 promise
        is that training and serving are the same arithmetic.
        """

        from engine.learned_policy_v8 import (
            RESIDUAL_CAP_POT_FRACTION,
            compose_branch_values,
            _forward_v3,
        )

        if abs(RESIDUAL_CAP_POT_FRACTION - RESIDUAL_CAP_POT_FRACTION_DEFAULT) > 0:
            raise AssertionError(
                "the serve module's residual cap constant has drifted from "
                "the trainer's copy; reconcile before exporting"
            )
        rounded = _round9({"weights": weights})["weights"]
        architecture = default_v8_architecture()
        sample = pb_validation[: config.parity_sample]
        max_diff = 0.0
        with torch.no_grad():
            for offset, decision in enumerate(sample):
                indexes = torch.tensor([offset], dtype=torch.long, device=device)
                outputs = model(
                    pb_validation_data["card"][indexes],
                    pb_validation_data["ctx"][indexes],
                )
                torch_values = composed_values(
                    outputs, pb_validation_data, indexes
                )[0]
                normalized = tuple(
                    (value - mean) / max(1e-6, std)
                    for value, mean, std in zip(decision.features, means, stds)
                )
                stdlib_outputs = _forward_v3(architecture, rounded, normalized)
                stdlib_values, _ = compose_branch_values(
                    stdlib_outputs,
                    pot=decision.context["pot"],
                    to_call=decision.context["to_call"],
                    contribution=decision.context["contribution"],
                    effective_stack=decision.context["effective_stack"],
                    purse=decision.context["purse"],
                    legal_range=decision.context["legal_range"],
                    use_residual=True,
                    residual_cap_pot_fraction=cap_fraction,
                )
                for label in decision.emitted:
                    if label not in stdlib_values:
                        raise AssertionError(
                            f"parity: corpus emitted {label!r} at "
                            f"{decision.decision_id} but the stdlib "
                            "composition did not"
                        )
                    slot = BRANCH_LABELS_V8.index(label)
                    diff = abs(float(torch_values[slot]) - stdlib_values[label])
                    max_diff = max(max_diff, diff)
                    if diff > _PARITY_TOLERANCE:
                        raise AssertionError(
                            f"parity: branch {label!r} at "
                            f"{decision.decision_id} diverges by {diff} "
                            f"(> {_PARITY_TOLERANCE}) between the torch and "
                            "stdlib compositions"
                        )
        return {
            "decisions_checked": len(sample),
            "max_abs_value_diff": round(max_diff, 9),
            "tolerance": _PARITY_TOLERANCE,
        }

    weights = export_weights()
    _assert_finite_weights(weights, "v8 phase-B training")
    parity = parity_check(weights)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "weights": weights,
        "means": means,
        "stds": stds,
        "target_variance": target_variance,
        "pb_train_decisions": len(pb_train),
        "pb_validation_decisions": len(pb_validation),
        "pb_train_tables": len({d.table_id for d in pb_train}),
        "pb_validation_tables": len({d.table_id for d in pb_validation}),
        "pa_train_rows": len(pa_train),
        "pa_validation_rows": len(pa_validation),
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "residual_share": residual_share(),
        "parity_check": parity,
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
        "init_seed": base.init_seed,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _rounded(losses: Mapping[str, float]) -> dict[str, float]:
    return {name: round(value, 6) for name, value in losses.items()}


def _seed_record(result: Mapping[str, object]) -> dict[str, object]:
    trace = result["trace"]
    assert isinstance(trace, Mapping)
    return {
        "init_seed": result["init_seed"],
        "train_losses": _rounded(result["train_losses"]),  # type: ignore[arg-type]
        "validation_losses": _rounded(result["validation_losses"]),  # type: ignore[arg-type]
        "residual_share": result["residual_share"],
        "best_epoch": trace["best_epoch"],
        "epochs_run": trace["epochs_run"],
        "optimizer_steps": trace["optimizer_steps"],
    }


def train_phase_b_candidate(
    phase_b: Sequence[PhaseBDecision],
    phase_a: Sequence[PhaseARow],
    output_dir: str | Path,
    config: PhaseBTrainingConfig,
    init_seeds: Sequence[int] = (401, 402, 403),
    corpus_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> dict[str, object]:
    """Train every init seed, select by total validation loss, export one.

    Same caveat as Phase A, verbatim: validation loss is a gate, never a
    selector (V8_DESIGN §6.1) — every seed's losses are recorded so the
    seat-swapped duel can overrule this pick, and the project's practice is
    one artifact per seed (single-seed invocations) so all three can be
    gauntleted.
    """

    check_phase_b_config(config)
    if not init_seeds:
        raise ValueError("at least one init seed is required")
    if len(set(init_seeds)) != len(init_seeds):
        raise ValueError("init seeds must be unique")
    model_version = config.base.model_version
    if not model_version:
        raise ValueError("config.base.model_version is required for export")
    output_path = Path(output_dir).expanduser().resolve()
    weights_path = output_path / f"{model_version}.weights.json"
    manifest_path = output_path / f"{model_version}.manifest.json"
    if weights_path.exists() or manifest_path.exists():
        raise FileExistsError(f"candidate artifact already exists for {model_version}")

    started = time.monotonic()
    results = []
    for init_seed in init_seeds:
        seed_config = replace(
            config, base=replace(config.base, init_seed=init_seed)
        )
        result = fit_phase_b(phase_b, phase_a, seed_config)
        results.append(result)
        validation = result["validation_losses"]
        assert isinstance(validation, Mapping)
        print(
            f"seed {init_seed}: val total {validation['total']:.6f} "
            f"(value_norm {validation['value_normalized']:.6f}, "
            f"value_mse {validation['value_mse']:.8f}, "
            f"fold_through {validation['fold_through']:.6f}, "
            f"range {validation['range']:.6f}, "
            f"equity_called {validation['equity_called']:.6f}), "
            f"best epoch {result['trace']['best_epoch']}, "  # type: ignore[index]
            f"residual share {result['residual_share']['share_of_abs_composed_value']}, "  # type: ignore[index]
            f"parity max diff {result['parity_check']['max_abs_value_diff']}",  # type: ignore[index]
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
    architecture["dropout"] = float(config.base.dropout)
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
        weights_document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    weights_sha256 = hashlib.sha256(encoded).hexdigest()

    per_street: dict[str, int] = {}
    for decision in phase_b:
        per_street[decision.street] = per_street.get(decision.street, 0) + 1
    branch_counts: dict[str, int] = {}
    for decision in phase_b:
        for label in decision.emitted:
            branch_counts[label] = branch_counts.get(label, 0) + 1

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
            "phase_b_corpus": None if corpus_path is None else str(corpus_path),
            "phase_b_decisions": len(phase_b),
            "phase_b_tables": len({d.table_id for d in phase_b}),
            "phase_b_branch_counts": dict(sorted(branch_counts.items())),
            "phase_b_decisions_per_street": dict(sorted(per_street.items())),
            "phase_a_dataset": None if dataset_path is None else str(dataset_path),
            "phase_a_rows": len(phase_a),
            "row_count": len(phase_b) + len(phase_a),
        },
        "engine_parameters": {
            "safety_gates": DEFAULT_SAFETY_GATES.to_mapping(),
            "temperature_shaping": DEFAULT_TEMPERATURE_SHAPING.to_mapping(),
            "tracker_settings": DEFAULT_TRACKER_SETTINGS.to_mapping(),
            "bluff_settings": DEFAULT_BLUFF_SETTINGS.to_mapping(),
        },
        "serve": {
            "ood_guard_indices": list(schema3.CONTEXT_INDICES),
            "temperature": None,
            "note": (
                "Phase-B composed-value candidate: serve through "
                "learned_policy_v8.load_policy_v8 (the §4 composition). "
                "Promotion remains a separate, explicit, human-authorised "
                "act; this artifact must not be deployed by training"
            ),
        },
        "training": {
            "objective": TRAINING_OBJECTIVE_V8_PHASE_B,
            "phase": "B",
            "optimizer": "adamw",
            "learning_rate": config.base.learning_rate,
            "weight_decay": config.base.weight_decay,
            "residual_weight_decay": config.residual_weight_decay,
            "residual_decay_note": (
                "the residual head's tower and output parameters, biases "
                "included, form their own decoupled decay group (zero is "
                "the correct prior for a correction head); every other "
                "parameter keeps the v7 no-bias/no-LayerNorm decay rule"
            ),
            "dropout": config.base.dropout,
            "warmup_steps": config.base.warmup_steps,
            "early_stop_patience": config.base.early_stop_patience,
            "epochs": config.base.epochs,
            "epochs_run": best["trace"]["epochs_run"],  # type: ignore[index]
            "best_epoch": best["trace"]["best_epoch"],  # type: ignore[index]
            "best_validation_loss_total": best["trace"][  # type: ignore[index]
                "best_validation_loss_total"
            ],
            "optimizer_steps": best["trace"]["optimizer_steps"],  # type: ignore[index]
            "gradient_clip": "global-norm 1.0",
            "batch_size": config.base.batch_size,
            "phase_a_batch_size": config.phase_a_batch_size,
            "backend": "pytorch",
            "device": config.base.device,
            "device_name": best["device_name"],
            "parameter_count": best["parameter_count"],
            "loss": {
                "value_loss_weight": config.value_loss_weight,
                "supervised_loss_weight": config.supervised_loss_weight,
                "value_target_variance": round(float(best["target_variance"]), 9),  # type: ignore[arg-type]
                "value_normalization": (
                    "composed-value MSE divided by the population variance "
                    "of the Phase-B training targets (purse units) — an "
                    "estimated normalizer, so weight 1.0 means equal "
                    "footing with the supervised losses at the constant "
                    "predictor; both weights are CLI-ablatable"
                ),
                "composition": (
                    "centered within the decision over the corpus-emitted "
                    "branch set, through the §4 fixed arithmetic "
                    "(sigmoid fold-through, [0,1]-clamped equities, "
                    "residual clamped at ±cap·pot in purse units); "
                    "train/serve parity checked against "
                    "learned_policy_v8.compose_branch_values at export"
                ),
                "residual_cap_pot_fraction": config.residual_cap_pot_fraction,
                "supervised_source": (
                    "Phase-A dataset batches interleaved every optimizer "
                    "step (V8_DESIGN §5: component losses keep flowing)"
                ),
            },
            "split": {
                "method": "sha256(split_seed:table_id) hash, whole tables, "
                "applied independently to both datasets",
                "validation_fraction": config.base.validation_fraction,
                "split_seed": config.base.split_seed,
                "phase_b_train_decisions": best["pb_train_decisions"],
                "phase_b_validation_decisions": best["pb_validation_decisions"],
                "phase_b_train_tables": best["pb_train_tables"],
                "phase_b_validation_tables": best["pb_validation_tables"],
                "phase_a_train_rows": best["pa_train_rows"],
                "phase_a_validation_rows": best["pa_validation_rows"],
            },
            "input_normalization": (
                "context block z-scored from the union of the Phase-A and "
                "Phase-B training splits (std floor 0.05); card block raw"
            ),
            "init_seed": best["init_seed"],
            "init_seeds_evaluated": [_seed_record(result) for result in results],
            "seed_selection": (
                "minimum total validation loss; validation loss is a gate, "
                "never a selector (V8_DESIGN §6) — the seat-swapped duel "
                "remains the selector at evaluation time"
            ),
            "branch_targets": {
                "aggress_small": list(_BRANCH_SMALL),
                "aggress_large": list(_BRANCH_LARGE),
            },
        },
        "evaluation": {
            "train_losses": _rounded(best["train_losses"]),  # type: ignore[arg-type]
            "validation_losses": _rounded(best["validation_losses"]),  # type: ignore[arg-type]
            "residual_share": best["residual_share"],
            "parity_check": best["parity_check"],
        },
        "promotion": None,
    }
    validate_v8_manifest(manifest)
    weights_path.write_bytes(encoded + b"\n")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "selected_init_seed": best["init_seed"],
        "validation_losses": _rounded(best["validation_losses"]),  # type: ignore[arg-type]
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
    phase_defaults = PhaseBTrainingConfig(base=defaults)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-b-corpus", default=str(DEFAULT_PHASE_B_CORPUS))
    parser.add_argument("--phase-a-dataset", default=str(DEFAULT_PHASE_A_DATASET))
    parser.add_argument("--output-dir", default="artifacts/candidates")
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--init-seeds", type=int, nargs="+", default=[401, 402, 403]
    )
    parser.add_argument(
        "--value-loss-weight", type=float, default=phase_defaults.value_loss_weight
    )
    parser.add_argument(
        "--supervised-loss-weight",
        type=float,
        default=phase_defaults.supervised_loss_weight,
    )
    parser.add_argument(
        "--residual-weight-decay",
        type=float,
        default=phase_defaults.residual_weight_decay,
    )
    parser.add_argument(
        "--residual-cap-pot-fraction",
        type=float,
        default=phase_defaults.residual_cap_pot_fraction,
    )
    parser.add_argument(
        "--phase-a-batch-size", type=int, default=phase_defaults.phase_a_batch_size
    )
    parser.add_argument(
        "--parity-sample", type=int, default=phase_defaults.parity_sample
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument(
        "--early-stop-patience", type=int, default=defaults.early_stop_patience
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--validation-fraction", type=float, default=defaults.validation_fraction
    )
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=defaults.device)
    args = parser.parse_args(argv)

    config = PhaseBTrainingConfig(
        base=V8TrainingConfig(
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
        ),
        value_loss_weight=args.value_loss_weight,
        supervised_loss_weight=args.supervised_loss_weight,
        residual_weight_decay=args.residual_weight_decay,
        residual_cap_pot_fraction=args.residual_cap_pot_fraction,
        phase_a_batch_size=args.phase_a_batch_size,
        parity_sample=args.parity_sample,
    )
    decisions = load_phase_b_decisions(args.phase_b_corpus)
    print(f"phase-b corpus: {args.phase_b_corpus} ({len(decisions)} decisions)")
    rows = load_phase_a_dataset(args.phase_a_dataset)
    print(f"phase-a dataset: {args.phase_a_dataset} ({len(rows)} rows)")
    summary = train_phase_b_candidate(
        decisions,
        rows,
        args.output_dir,
        config,
        init_seeds=tuple(args.init_seeds),
        corpus_path=args.phase_b_corpus,
        dataset_path=args.phase_a_dataset,
    )
    print(f"selected init seed: {summary['selected_init_seed']}")
    print(f"validation losses: {summary['validation_losses']}")
    print(f"manifest: {summary['manifest_path']}")
    print(f"weights: {summary['weights_path']}")
    print(f"weights_sha256: {summary['weights_sha256']}")
    print(f"wall_time_seconds: {summary['wall_time_seconds']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PHASE_A_DATASET",
    "DEFAULT_PHASE_B_CORPUS",
    "PhaseBDecision",
    "PhaseBTrainingConfig",
    "RESIDUAL_CAP_POT_FRACTION_DEFAULT",
    "TRAINING_OBJECTIVE_V8_PHASE_B",
    "check_phase_b_config",
    "compose_from_constants",
    "fit_phase_b",
    "load_phase_b_decisions",
    "split_decisions",
    "train_phase_b_candidate",
    "value_target_variance",
]
