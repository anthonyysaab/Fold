"""Phase-B composed-value trainer for the v9 network (L3, second pass).

Fork of :mod:`engine.v8_trainer_phase_b` for the v9 branch contract,
left as a sibling module so the v8 trainer stays byte-identical for the
frozen format-3 line. The joint objective is unchanged — composed-value
MSE (normalized by the estimated target variance) plus the interleaved
Phase-A supervised losses, with the residual head in its own decoupled
weight-decay group — what changes is the DATA CONTRACT, pinned in
`.handoff/notes/V9_RESTRUCTURE_PLAN.md` ("L3/L4 DATA CONTRACTS") before
any of this code existed:

- **The corpus header gate**: ``corpus_schema_version`` must be 2, and
  version 1 — the stored v8 corpus — is refused BY NUMBER, never
  half-loaded. The header must also carry ``branch_labels ==
  BRANCH_LABELS_V9``, ``feature_schema_version`` 4, the full composed
  ``sizing`` record (g identity + every dial state used at harvest),
  ``belief_fit_source`` (the P3 fit the belief buckets were computed
  from), ``equity_trials`` (harvest == serve, one number — the exported
  manifest pins it at ``serve.equity_trials``), and the frozen
  instrument fields (``starting_stack``, ``big_blind``, ``seeds``).
- **Sizing is re-derived through frozen g from the RECORDED read.** A
  row's ``decision.context`` carries the raw int ``read_temperature_x10``
  (``10·T``); the loader decodes it through the HEADER'S OWN sizing
  parameters, runs the composed pipeline
  (``engine.rules.composition``), and asserts the harvester recorded
  the same ``sizing_target`` / ``sizing_to_amount`` — renamed from the
  v8 ``e6_*`` keys so old corpora fail loudly. Recomputing the read live
  would make this cross-check a tautology; consuming the recorded read
  is the point (only Python's banker's rounding ever produced it).
- **``decision.context`` IS ``compose_branch_values_v9``'s argument
  list, by design**: pot, to_call, contribution, effective_stack,
  purse, read_temperature_x10, street, bankroll, exposure,
  covered_allin_to_amounts, legal_labels, bet_range, raise_range. The
  export-time parity replay reconstructs the stdlib serve call from the
  stored context alone and fails closed on any branch-value
  disagreement — and keeps the emitted-set assertion as the emission-
  parity gate.
- **One shared wager-column helper** (:func:`wager_column_slice`,
  computed from ``BRANCH_LABELS_V9.index``) feeds the loss, the
  residual audit, and the residual clamp — a matching wrong literal on
  both parity sides would pass parity cleanly, so the slice is written
  once.
- **``v_active = where(to_call_zero, bet_composition,
  call_composition)``**: the active branch composes as a fold-through
  bet on free-spot rows and as the no-fold-equity call on priced rows,
  with the residual correction on wager executions only (the v8
  discipline, kept).

Torch imports are function-local (the ``offline_trainer`` pattern): the
module imports cleanly on the stdlib interpreter; training runs in the
CUDA venv. Artifacts are immutable candidates: state ``"candidate"``,
promotion ``null``, ``artifacts/approved.json`` never read or written.

Usage (CUDA venv, repo root)::

    python -m engine.v9_trainer_phase_b \
        --model-version candidate-v9-0002a --init-seeds 401
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
from engine.aggression_sizing import (
    SizingParameters,
    context_int_to_temperature,
)
from engine.branch_contract_v9 import (
    BRANCH_LABELS_V9,
    EQUITY_SLOTS_V9,
    FOLD_THROUGH_BRANCHES_V9,
    MODEL_FORMAT_VERSION_V9,
    branch_action,
)
from engine.decision_engine import (
    DEFAULT_SAFETY_GATES,
    DEFAULT_TEMPERATURE_SHAPING,
)
from engine.learned_policy_v8 import _clip01
from engine.learning_contract import MODEL_FORMAT
from engine.offline_trainer import (
    _assert_finite_weights,
    _round9,
    _sigmoid,
    validate_training_device,
)
from engine.opponent_model import DEFAULT_TRACKER_SETTINGS
from engine.rules.composition import (
    RuleLayerParams,
    compose_active_wager,
    compose_aggressive_target,
    parameters_and_rules_from_record,
)
from engine.v8_trainer import (
    V8TrainingConfig,
    default_v9_architecture,
    split_rows,
    validate_v9_architecture,
)
from engine.v8_trainer_phase_b import (
    RESIDUAL_CAP_POT_FRACTION_DEFAULT,
    PhaseBTrainingConfig,
    _finite,
    check_phase_b_config,
    split_decisions,
    value_target_variance,
)
from engine.v9_trainer import (
    PhaseARowV9,
    build_network_v9,
    context_normalization_v9,
    export_network_weights,
    load_phase_a_dataset_v9,
    validate_v9_manifest,
    validate_v9_weight_shapes,
)

TRAINING_OBJECTIVE_V9_PHASE_B = "phase_b_composed_value_v9"

#: Default locations (repo-root relative); the L4 harvester pins the
#: real corpus name when it exists.
DEFAULT_PHASE_B_CORPUS_V9 = (
    Path("artifacts") / "phase_b_v9" / "phase-b-corpus-v9.jsonl.gz"
)
DEFAULT_PHASE_A_DATASET_V9 = (
    Path("artifacts") / "phase_a_v9" / "phase-a-dataset-v9.jsonl.gz"
)

#: The wager-making branches, in contract order. They are exactly the
#: fold-through branches: a branch carries fold equity iff it makes a
#: wager, which is why one tuple serves both roles.
WAGER_LANES = FOLD_THROUGH_BRANCHES_V9

_SIZING_TOLERANCE = 1e-6
_CENTER_TOLERANCE = 1e-4
_PARITY_TOLERANCE = 1e-3
_STREETS = ("preflop", "flop", "turn", "river")

#: Slot indices resolved BY NAME once (pinned rule), never as literals.
_FT_ACTIVE = FOLD_THROUGH_BRANCHES_V9.index("active")
_FT_AGGRESSIVE = FOLD_THROUGH_BRANCHES_V9.index("aggressive")
_EQ_PASSIVE = EQUITY_SLOTS_V9.index("passive")
_EQ_ACTIVE = EQUITY_SLOTS_V9.index("active")
_EQ_AGGRESSIVE = EQUITY_SLOTS_V9.index("aggressive")
_LANE_ACTIVE = WAGER_LANES.index("active")
_LANE_AGGRESSIVE = WAGER_LANES.index("aggressive")

#: The v8 sizing keys, recognized only to refuse them with guidance.
_V8_SIZING_KEYS = frozenset({"e6_target", "e6_to_amount"})


def wager_column_slice() -> slice:
    """The wager lanes' column slice of a ``BRANCH_LABELS_V9``-wide array.

    Computed ONCE from ``BRANCH_LABELS_V9.index`` (pinned trainer rule)
    and reused by the loss, the residual clamp, and the residual audit —
    a wrong literal repeated on both parity sides would pass parity
    cleanly. Verifies the version-ledger invariant it relies on: the
    wager lanes are contiguous, in ``WAGER_LANES`` order, and sit LAST
    (so the sliced columns align with ``WAGER_LANES`` indices).
    """

    first = BRANCH_LABELS_V9.index(WAGER_LANES[0])
    span = [
        BRANCH_LABELS_V9[first + offset] for offset in range(len(WAGER_LANES))
    ]
    if tuple(span) != tuple(WAGER_LANES):
        raise AssertionError(
            "the wager lanes must be contiguous in BRANCH_LABELS_V9, in order"
        )
    if first + len(WAGER_LANES) != len(BRANCH_LABELS_V9):
        raise AssertionError("the wager lanes must sit last in BRANCH_LABELS_V9")
    return slice(first, first + len(WAGER_LANES))


WAGER_COLUMN_SLICE = wager_column_slice()


@dataclass(frozen=True, slots=True)
class PhaseBDecisionV9:
    """One validated v9 Phase-B decision with derived composition constants.

    All ``*_unit`` quantities are purse-normalized (chips / purse),
    exactly the units ``compose_branch_values_v9`` works in. Wager-lane
    entries (``WAGER_LANES`` order) are ``None`` when the lane did not
    execute a wager for this decision — ``aggressive`` when not emitted,
    ``active`` whenever the row is priced (a call has no size).
    ``context`` is the raw record the stdlib parity replay reconstructs
    the serve call from.
    """

    decision_id: str
    table_id: str
    street: str
    features: tuple[float, ...]
    emitted: tuple[str, ...]  # subset of BRANCH_LABELS_V9, slot order
    targets: dict[str, float]  # centered reward, purse units, emitted only
    to_call_zero: bool
    pot_unit: float
    cc_pot_unit: float  # (pot + to_call) / purse
    cc_cost_unit: float  # to_call / purse
    wager_unit: tuple[float | None, ...]  # WAGER_LANES order
    pot_if_called_unit: tuple[float | None, ...]
    context: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PhaseBCorpusV9:
    """A loaded v9 Phase-B corpus: validated header plus decisions."""

    sizing_record: dict[str, Any]  # the header's record, verbatim
    sizing: SizingParameters
    rules: RuleLayerParams
    belief_fit_source: object  # opaque P3 provenance (path or mapping)
    equity_trials: int
    starting_stack: int
    big_blind: int
    seeds: tuple[int, ...]
    decisions: tuple[PhaseBDecisionV9, ...]


# ---------------------------------------------------------------------------
# Corpus loading — fail-closed, sizes re-derived through frozen g
# ---------------------------------------------------------------------------


def _context_int(
    context: Mapping[str, Any], name: str, where: str, minimum: int = 0
) -> int:
    """A raw-int context field (the pinned contract: raw ints only)."""

    raw = context.get(name)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"{where}: context {name} must be a raw integer")
    if raw < minimum:
        raise ValueError(f"{where}: context {name} must be >= {minimum}")
    return raw


def _lane_range_from_context(
    context: Mapping[str, Any], name: str, where: str
) -> tuple[int, int] | None:
    raw = context.get(name)
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) != 2:
        raise ValueError(f"{where}: context {name} must be null or [min, max]")
    low, high = raw
    for bound in (low, high):
        if not isinstance(bound, int) or isinstance(bound, bool):
            raise ValueError(f"{where}: context {name} bounds must be integers")
    if low < 0 or high < low:
        raise ValueError(f"{where}: context {name} is not a valid range")
    return int(low), int(high)


def _derive_wager_units(
    branch: str,
    context: Mapping[str, Any],
    recorded: Mapping[str, Any],
    where: str,
    sizing: SizingParameters,
    rules: RuleLayerParams,
) -> tuple[float, float]:
    """(wager, pot_if_called) in chips, cross-checked against the corpus.

    Re-runs the composed sizing pipeline through FROZEN g from the row's
    RECORDED read (``10·T`` decoded through the header's own parameters
    — a live recompute would make the cross-check a tautology), then
    asserts the harvester recorded the same unclamped ``sizing_target``
    and range-clamped ``sizing_to_amount``: the trainer's sizing and the
    harvest's sizing must be one arithmetic or the labels do not mean
    what the composition says.
    """

    if _V8_SIZING_KEYS & set(recorded):
        raise ValueError(
            f"{where}: e6_target/e6_to_amount are the v8 corpus's keys — "
            "v9 rows carry sizing_target/sizing_to_amount (the rename is "
            "the version guard; stored corpora are never relabeled)"
        )
    boldness = sizing.boldness(
        context_int_to_temperature(context["read_temperature_x10"])
    )
    shared = dict(
        boldness=boldness,
        pot=context["pot"],
        effective_stack=context["effective_stack"],
        contribution=context["contribution"],
        street=context["street"],
        bankroll=context["bankroll"],
        exposure=context["exposure"],
        covered_allin_to_amounts=tuple(context["covered_allin_to_amounts"]),
        sizing=sizing,
        geometric=rules.geometric,
        snap=rules.snap,
        damper=rules.damper,
    )
    if branch == "aggressive":
        composed = compose_aggressive_target(to_call=context["to_call"], **shared)
        lane_range = context["raise_range"]
    else:
        composed = compose_active_wager(**shared)
        lane_range = context["bet_range"]
    to_amount = composed.to_amount
    if lane_range is not None:
        low, high = lane_range
        to_amount = min(high, max(low, to_amount))
    recorded_target = _finite(recorded.get("sizing_target"), "sizing_target", where)
    recorded_to = _finite(
        recorded.get("sizing_to_amount"), "sizing_to_amount", where
    )
    if abs(recorded_target - composed.target) > _SIZING_TOLERANCE:
        raise ValueError(
            f"{where}: derived sizing target {composed.target!r} does not "
            f"match the corpus's {recorded_target!r} for {branch}"
        )
    if abs(recorded_to - to_amount) > _SIZING_TOLERANCE:
        raise ValueError(
            f"{where}: derived sizing to-amount {to_amount!r} does not "
            f"match the corpus's {recorded_to!r} for {branch}"
        )
    contribution = context["contribution"]
    wager = max(0.0, to_amount - contribution)
    pot_if_called = context["pot"] + 2.0 * wager - context["to_call"]
    return wager, pot_if_called


def _parse_decision_v9(
    row: Mapping[str, Any],
    line: int,
    sizing: SizingParameters,
    rules: RuleLayerParams,
    header_big_blind: int,
) -> PhaseBDecisionV9:
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
    if not isinstance(features, list) or len(features) != schema4.INPUT_SIZE_V9:
        raise ValueError(
            f"{where}: features must be {schema4.INPUT_SIZE_V9} floats"
        )
    vector = tuple(_finite(value, "feature", where) for value in features)

    raw_context = row.get("context")
    if not isinstance(raw_context, Mapping):
        raise ValueError(f"{where}: missing context")
    context: dict[str, Any] = {
        name: _context_int(raw_context, name, where, minimum=minimum)
        for name, minimum in (
            ("pot", 0),
            ("to_call", 0),
            ("contribution", 0),
            ("effective_stack", 0),
            ("purse", 1),
            ("bankroll", 1),
            ("exposure", 0),
        )
    }
    encoded = _context_int(raw_context, "read_temperature_x10", where)
    if encoded > 1000:
        raise ValueError(
            f"{where}: context read_temperature_x10 {encoded} is not in [0, 1000]"
        )
    context["read_temperature_x10"] = encoded
    context_street = raw_context.get("street")
    if context_street != street:
        raise ValueError(
            f"{where}: context street {context_street!r} does not match the "
            f"row street {street!r}"
        )
    context["street"] = str(context_street)
    covered = raw_context.get("covered_allin_to_amounts")
    if not isinstance(covered, Sequence) or isinstance(covered, str):
        raise ValueError(
            f"{where}: context covered_allin_to_amounts must be a list"
        )
    for amount in covered:
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError(
                f"{where}: covered_allin_to_amounts entries must be raw "
                "non-negative integers"
            )
    context["covered_allin_to_amounts"] = [int(amount) for amount in covered]
    context["bet_range"] = _lane_range_from_context(raw_context, "bet_range", where)
    context["raise_range"] = _lane_range_from_context(
        raw_context, "raise_range", where
    )

    to_call = context["to_call"]
    to_call_zero = to_call == 0
    legal_labels = raw_context.get("legal_labels")
    if not isinstance(legal_labels, list) or len(legal_labels) < 2:
        raise ValueError(
            f"{where}: context legal_labels must list at least two branches"
        )
    seen_labels: list[str] = []
    for label in legal_labels:
        if label not in BRANCH_LABELS_V9:
            raise ValueError(f"{where}: unknown branch label {label!r}")
        if label in seen_labels:
            raise ValueError(f"{where}: duplicate legal label {label!r}")
        # The contract's own masking rules: raises on fatal at a free
        # check, passive at a price, aggressive at to_call == 0.
        branch_action(str(label), to_call)
        seen_labels.append(str(label))
    slot_ordered = sorted(seen_labels, key=BRANCH_LABELS_V9.index)
    if seen_labels != slot_ordered:
        raise ValueError(
            f"{where}: legal_labels must be recorded in slot order "
            f"{slot_ordered!r}, found {seen_labels!r}"
        )
    context["legal_labels"] = seen_labels

    purse = context["purse"]
    big_blind = _finite(row.get("big_blind"), "big_blind", where)
    if big_blind != header_big_blind:
        raise ValueError(
            f"{where}: row big_blind {big_blind!r} does not match the "
            f"header's {header_big_blind!r}"
        )
    purse_bb = _finite(row.get("purse_bb"), "purse_bb", where)
    if big_blind <= 0 or purse_bb <= 0:
        raise ValueError(f"{where}: big_blind and purse_bb must be positive")
    if abs(purse_bb * big_blind - purse) > 1e-6 * max(1.0, purse):
        raise ValueError(
            f"{where}: purse_bb {purse_bb} x big_blind {big_blind} does not "
            f"reproduce context purse {purse}"
        )

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
        if label not in BRANCH_LABELS_V9:
            raise ValueError(f"{where}: unknown branch label {label!r}")
        if label in emitted:
            raise ValueError(f"{where}: duplicate branch {label!r}")
        emitted.append(str(label))
        reward_bb = _finite(entry.get("reward_bb"), "reward_bb", where)
        total_reward += reward_bb
        # Purse units: reward_bb / purse_bb == reward_chips / purse — the
        # exact normalization the composition works in.
        targets[str(label)] = reward_bb / purse_bb
        is_wager_execution = label == "aggressive" or (
            label == "active" and to_call_zero
        )
        if is_wager_execution:
            chips_wager, chips_pic = _derive_wager_units(
                str(label), context, entry, where, sizing, rules
            )
            wager[str(label)] = chips_wager / purse
            pot_if_called[str(label)] = chips_pic / purse
        elif (
            {"sizing_target", "sizing_to_amount"} & set(entry)
            or _V8_SIZING_KEYS & set(entry)
        ):
            raise ValueError(
                f"{where}: branch {label!r} must not carry sizing fields — "
                "only wager executions have a size (aggressive, and active "
                "at to_call == 0; a call has no size)"
            )
    if emitted != context["legal_labels"]:
        raise ValueError(
            f"{where}: emitted branches {emitted!r} must equal the "
            f"context's legal_labels {context['legal_labels']!r} — under "
            "the v9 contract nothing is deduplicated away"
        )
    if abs(total_reward) > _CENTER_TOLERANCE:
        raise ValueError(
            f"{where}: centered rewards sum to {total_reward!r}, not ~0"
        )

    return PhaseBDecisionV9(
        decision_id=decision_id,
        table_id=table_id,
        street=str(street),
        features=vector,
        emitted=tuple(emitted),
        targets=targets,
        to_call_zero=to_call_zero,
        pot_unit=context["pot"] / purse,
        cc_pot_unit=(context["pot"] + to_call) / purse,
        cc_cost_unit=to_call / purse,
        wager_unit=tuple(wager.get(label) for label in WAGER_LANES),
        pot_if_called_unit=tuple(
            pot_if_called.get(label) for label in WAGER_LANES
        ),
        context=context,
    )


def load_phase_b_corpus_v9(path: str | Path) -> PhaseBCorpusV9:
    """Load and validate a v9 Phase-B corpus, fail-closed.

    The header gate runs before any row is read; every row is then
    re-derived and cross-checked against the header's own g record.
    """

    resolved = Path(path)
    decisions: list[PhaseBDecisionV9] = []
    seen: set[str] = set()
    with gzip.open(resolved, "rt", encoding="utf-8") as stream:
        header = json.loads(stream.readline())
        if not isinstance(header, Mapping) or header.get("kind") != "phase-b-corpus":
            raise ValueError(f"{resolved}: not a phase-b corpus")
        version = header.get("corpus_schema_version")
        if version == 1:
            raise ValueError(
                f"{resolved}: corpus_schema_version 1 is the v8 corpus — "
                "the v9 trainer refuses it (v9 trains from fresh harvests "
                "only; stored corpora are never relabeled)"
            )
        if version != 2:
            raise ValueError(
                f"{resolved}: unsupported corpus schema version {version!r}; "
                "the v9 trainer requires 2"
            )
        if header.get("feature_schema_version") != schema4.SCHEMA_VERSION_V9:
            raise ValueError(f"{resolved}: corpus feature schema is not schema 4")
        if header.get("input_size") != schema4.INPUT_SIZE_V9:
            raise ValueError(f"{resolved}: corpus input size does not match schema 4")
        if list(header.get("branch_labels") or []) != list(BRANCH_LABELS_V9):
            raise ValueError(
                f"{resolved}: corpus branch labels are not the v9 contract's"
            )
        sizing_record = header.get("sizing")
        if not isinstance(sizing_record, Mapping):
            raise ValueError(
                f"{resolved}: corpus header must carry the composed sizing "
                "record the harvest ran under"
            )
        try:
            sizing, rules = parameters_and_rules_from_record(sizing_record)
        except ValueError as error:
            raise ValueError(
                f"{resolved}: corpus sizing record is invalid: {error}"
            ) from error
        belief_fit_source = header.get("belief_fit_source")
        if not (
            (isinstance(belief_fit_source, str) and belief_fit_source)
            or isinstance(belief_fit_source, Mapping)
        ):
            raise ValueError(
                f"{resolved}: corpus header must record belief_fit_source "
                "(the P3 belief fit the buckets were computed from)"
            )
        equity_trials = header.get("equity_trials")
        if (
            not isinstance(equity_trials, int)
            or isinstance(equity_trials, bool)
            or equity_trials < 1
        ):
            raise ValueError(
                f"{resolved}: corpus header must record equity_trials as a "
                "positive integer (harvest == serve, one number)"
            )
        instrument: dict[str, int] = {}
        for name in ("starting_stack", "big_blind"):
            value = header.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"{resolved}: corpus header must record {name} as a "
                    "positive integer (the frozen-instrument fields)"
                )
            instrument[name] = value
        seeds = header.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(
                not isinstance(seed, int) or isinstance(seed, bool)
                for seed in seeds
            )
        ):
            raise ValueError(
                f"{resolved}: corpus header must record seeds as a non-empty "
                "list of integers (the frozen-instrument fields)"
            )
        for line_number, line in enumerate(stream, start=2):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, Mapping):
                raise ValueError(f"line {line_number}: row is not an object")
            decision = _parse_decision_v9(
                row, line_number, sizing, rules, instrument["big_blind"]
            )
            if decision.decision_id in seen:
                raise ValueError(
                    f"line {line_number}: duplicate decision_id "
                    f"{decision.decision_id!r}"
                )
            seen.add(decision.decision_id)
            decisions.append(decision)
    if not decisions:
        raise ValueError(f"no decisions in {resolved}")
    return PhaseBCorpusV9(
        sizing_record=dict(sizing_record),
        sizing=sizing,
        rules=rules,
        belief_fit_source=belief_fit_source,
        equity_trials=int(equity_trials),
        starting_stack=instrument["starting_stack"],
        big_blind=instrument["big_blind"],
        seeds=tuple(int(seed) for seed in seeds),
        decisions=tuple(decisions),
    )


def compose_from_constants_v9(
    head_outputs: Mapping[str, Sequence[float]],
    decision: PhaseBDecisionV9,
    *,
    use_residual: bool = True,
    residual_cap_pot_fraction: float = RESIDUAL_CAP_POT_FRACTION_DEFAULT,
) -> dict[str, float]:
    """Pure-float composition over the decision's derived constants.

    The torch-free twin of the training composition: the same arithmetic
    as ``learned_policy_v9.compose_branch_values_v9``, but over the
    LOADER-DERIVED wager constants instead of a live sizing run — so a
    stdlib test proving the two agree proves the loader's derivation and
    the serve path's sizing are one arithmetic.
    """

    fold_through = head_outputs["fold_through"]
    equity_called = head_outputs["equity_called"]
    residual = head_outputs["residual"]
    cap = abs(residual_cap_pot_fraction) * decision.pot_unit
    values: dict[str, float] = {}
    for label in decision.emitted:
        if label == "fatal":
            values[label] = 0.0
        elif label == "passive":
            eq = _clip01(float(equity_called[_EQ_PASSIVE]))
            values[label] = eq * decision.pot_unit
        elif label == "active" and not decision.to_call_zero:
            # A call buys no folds: no fold-through, no residual (the
            # correction belongs to sized wagers — the v8 discipline).
            eq = _clip01(float(equity_called[_EQ_ACTIVE]))
            values[label] = (
                eq * decision.cc_pot_unit - decision.cc_cost_unit
            )
        else:  # a wager execution: active as a bet, or aggressive
            lane = WAGER_LANES.index(label)
            wager = decision.wager_unit[lane]
            pot_if_called = decision.pot_if_called_unit[lane]
            assert wager is not None and pot_if_called is not None
            p_ft = _sigmoid(
                float(fold_through[FOLD_THROUGH_BRANCHES_V9.index(label)])
            )
            eq = _clip01(float(equity_called[EQUITY_SLOTS_V9.index(label)]))
            value = p_ft * decision.pot_unit + (1.0 - p_ft) * (
                eq * pot_if_called - wager
            )
            if use_residual:
                correction = float(residual[BRANCH_LABELS_V9.index(label)])
                value += min(cap, max(-cap, correction))
            values[label] = value
    return values


# ---------------------------------------------------------------------------
# Torch fitting
# ---------------------------------------------------------------------------


def fit_phase_b_v9(
    corpus: PhaseBCorpusV9,
    phase_a: Sequence[PhaseARowV9],
    config: PhaseBTrainingConfig,
) -> dict[str, object]:
    """Fit one init seed with the joint composed + supervised objective."""

    check_phase_b_config(config)
    base = config.base
    pb_train, pb_validation = split_decisions(corpus.decisions, base)
    pa_train, pa_validation = split_rows(phase_a, base)
    if not pb_train or not pb_validation:
        raise ValueError("phase-b split produced an empty train or validation set")
    if not pa_train or not pa_validation:
        raise ValueError("phase-a split produced an empty train or validation set")
    # Context z-scores from the union of both training splits (documented
    # in the manifest): both loss terms consume the same normalization,
    # and the exported scales must serve every input the model was
    # fitted on.
    means, stds = context_normalization_v9([*pa_train, *pb_train])
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

    card_indices = list(schema4.CARD_INDICES_V9)
    context_indices = list(schema4.CONTEXT_INDICES_V9)
    branch_count = len(BRANCH_LABELS_V9)
    lane_count = len(WAGER_LANES)
    ft_width = len(FOLD_THROUGH_BRANCHES_V9)

    model = build_network_v9(base.dropout)
    model.to(device)

    mean_tensor = torch.tensor(means, dtype=torch.float32)
    std_tensor = torch.tensor(stds, dtype=torch.float32)

    def normalized_features(vectors: Sequence[Sequence[float]]):
        features = torch.tensor(list(vectors), dtype=torch.float32)
        return ((features - mean_tensor) / std_tensor).to(device)

    # ----- Phase-B tensors -------------------------------------------------
    def phase_b_tensors(decisions: Sequence[PhaseBDecisionV9]) -> dict[str, object]:
        features = normalized_features([d.features for d in decisions])
        count = len(decisions)
        mask = torch.zeros((count, branch_count), dtype=torch.float32)
        target = torch.zeros((count, branch_count), dtype=torch.float32)
        pot_unit = torch.zeros(count, dtype=torch.float32)
        cc_pot = torch.zeros(count, dtype=torch.float32)
        cc_cost = torch.zeros(count, dtype=torch.float32)
        to_call_zero = torch.zeros(count, dtype=torch.float32)
        wager = torch.zeros((count, lane_count), dtype=torch.float32)
        pic = torch.zeros((count, lane_count), dtype=torch.float32)
        for index, decision in enumerate(decisions):
            pot_unit[index] = decision.pot_unit
            cc_pot[index] = decision.cc_pot_unit
            cc_cost[index] = decision.cc_cost_unit
            to_call_zero[index] = 1.0 if decision.to_call_zero else 0.0
            for label in decision.emitted:
                slot = BRANCH_LABELS_V9.index(label)
                mask[index, slot] = 1.0
                target[index, slot] = decision.targets[label]
            for lane in range(lane_count):
                if decision.wager_unit[lane] is not None:
                    wager[index, lane] = decision.wager_unit[lane]
                    pic[index, lane] = decision.pot_if_called_unit[lane]
        return {
            "card": features[:, card_indices],
            "ctx": features[:, context_indices],
            "mask": mask.to(device),
            "target": target.to(device),
            "pot_unit": pot_unit.to(device),
            "cc_pot": cc_pot.to(device),
            "cc_cost": cc_cost.to(device),
            "to_call_zero": to_call_zero.to(device),
            "wager": wager.to(device),
            "pic": pic.to(device),
        }

    cap_fraction = abs(float(config.residual_cap_pot_fraction))

    def composed_values(outputs, data, indexes):
        """[batch, 4] composed values, mirroring compose_branch_values_v9."""

        pot_unit = data["pot_unit"][indexes]
        equity = torch.clamp(outputs["equity_called"], 0.0, 1.0)
        p_ft = torch.sigmoid(outputs["fold_through"])
        cap = cap_fraction * pot_unit
        residual = torch.clamp(
            outputs["residual"][:, WAGER_COLUMN_SLICE],
            min=-cap.unsqueeze(1),
            max=cap.unsqueeze(1),
        )
        wager = data["wager"][indexes]
        pic = data["pic"][indexes]
        v_fatal = torch.zeros_like(pot_unit)
        v_passive = equity[:, _EQ_PASSIVE] * pot_unit
        v_active_call = (
            equity[:, _EQ_ACTIVE] * data["cc_pot"][indexes]
            - data["cc_cost"][indexes]
        )
        v_active_bet = (
            p_ft[:, _FT_ACTIVE] * pot_unit
            + (1.0 - p_ft[:, _FT_ACTIVE])
            * (equity[:, _EQ_ACTIVE] * pic[:, _LANE_ACTIVE] - wager[:, _LANE_ACTIVE])
            + residual[:, _LANE_ACTIVE]
        )
        # The pinned rule: the active branch composes as its bet on
        # free-spot rows and as its call on priced rows.
        v_active = torch.where(
            data["to_call_zero"][indexes] > 0.5, v_active_bet, v_active_call
        )
        v_aggressive = (
            p_ft[:, _FT_AGGRESSIVE] * pot_unit
            + (1.0 - p_ft[:, _FT_AGGRESSIVE])
            * (
                equity[:, _EQ_AGGRESSIVE] * pic[:, _LANE_AGGRESSIVE]
                - wager[:, _LANE_AGGRESSIVE]
            )
            + residual[:, _LANE_AGGRESSIVE]
        )
        per_branch = {
            "fatal": v_fatal,
            "passive": v_passive,
            "active": v_active,
            "aggressive": v_aggressive,
        }
        return torch.stack(
            [per_branch[label] for label in BRANCH_LABELS_V9], dim=1
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

    # ----- Phase-A tensors (the v9_trainer construction, verbatim) ---------
    def phase_a_tensors(rows: Sequence[PhaseARowV9]) -> dict[str, object]:
        features = normalized_features([row.features for row in rows])
        return {
            "card": features[:, card_indices],
            "ctx": features[:, context_indices],
            "ft_target": torch.tensor(
                [[row.fold_through_label] * ft_width for row in rows],
                dtype=torch.float32,
                device=device,
            ),
            "ft_mask": torch.tensor(
                [[float(flag) for flag in row.fold_through_mask] for row in rows],
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
            raise FloatingPointError("non-finite loss during v9 phase-b training")
        set_learning_rate()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite gradient during v9 phase-b training")
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
                    "non-finite parameter during v9 phase-b training"
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

    # ----- Residual audit (the WAGER_COLUMN_SLICE reuse site) --------------
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
                outputs["residual"][:, WAGER_COLUMN_SLICE],
                min=-cap.unsqueeze(1),
                max=cap.unsqueeze(1),
            )
            # Residual reaches wager EXECUTIONS only: mask the active
            # column down to free-spot rows (priced active is a call).
            wager_mask = pb_validation_data["mask"][indexes][
                :, WAGER_COLUMN_SLICE
            ].clone()
            wager_mask[:, _LANE_ACTIVE] *= pb_validation_data["to_call_zero"][
                indexes
            ]
            abs_residual = (residual.abs() * wager_mask).sum()
            abs_value = (
                values[:, WAGER_COLUMN_SLICE].abs() * wager_mask
            ).sum()
            branches = wager_mask.sum()
        return {
            "wager_executions": int(branches),
            "sum_abs_capped_residual": round(float(abs_residual), 6),
            "sum_abs_composed_value": round(float(abs_value), 6),
            "share_of_abs_composed_value": (
                round(float(abs_residual) / float(abs_value), 6)
                if float(abs_value) > 0
                else None
            ),
        }

    # ----- Train/serve parity: torch vs the stdlib compose path ------------
    def parity_check(weights: Mapping[str, object]) -> dict[str, object]:
        """Replay validation decisions through the stdlib serve arithmetic.

        The exported weights (rounded exactly as the artifact is) drive
        ``learned_policy_v8._forward_v3`` +
        ``learned_policy_v9.compose_branch_values_v9``, with the serve
        call reconstructed from each decision's RECORDED context
        (boldness decoded from the stored ``10·T`` through the corpus's
        own sizing parameters). Any branch value diverging beyond
        tolerance fails the export, and the emitted-set assertion is the
        emission-parity gate: the stdlib composition must produce
        exactly the branches the corpus emitted.
        """

        from engine.learned_policy_v8 import (
            RESIDUAL_CAP_POT_FRACTION,
            _forward_v3,
        )
        from engine.learned_policy_v9 import compose_branch_values_v9

        if abs(RESIDUAL_CAP_POT_FRACTION - RESIDUAL_CAP_POT_FRACTION_DEFAULT) > 0:
            raise AssertionError(
                "the serve module's residual cap constant has drifted from "
                "the trainer's copy; reconcile before exporting"
            )
        rounded = _round9({"weights": weights})["weights"]
        architecture = default_v9_architecture()
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
                context = decision.context
                boldness = corpus.sizing.boldness(
                    context_int_to_temperature(context["read_temperature_x10"])
                )
                stdlib_values, _ = compose_branch_values_v9(
                    stdlib_outputs,
                    pot=context["pot"],
                    to_call=context["to_call"],
                    contribution=context["contribution"],
                    effective_stack=context["effective_stack"],
                    purse=context["purse"],
                    boldness=boldness,
                    street=context["street"],
                    bankroll=context["bankroll"],
                    exposure=context["exposure"],
                    covered_allin_to_amounts=tuple(
                        context["covered_allin_to_amounts"]
                    ),
                    legal_labels=frozenset(decision.emitted),
                    bet_range=context["bet_range"],
                    raise_range=context["raise_range"],
                    sizing=corpus.sizing,
                    rules=corpus.rules,
                    use_residual=True,
                    residual_cap_pot_fraction=cap_fraction,
                )
                if set(stdlib_values) != set(decision.emitted):
                    raise AssertionError(
                        f"parity: the corpus emitted {sorted(decision.emitted)} "
                        f"at {decision.decision_id} but the stdlib composition "
                        f"emitted {sorted(stdlib_values)}"
                    )
                for label in decision.emitted:
                    slot = BRANCH_LABELS_V9.index(label)
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

    weights = export_network_weights(model)
    _assert_finite_weights(weights, "v9 phase-B training")
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


def train_phase_b_candidate_v9(
    corpus: PhaseBCorpusV9,
    phase_a: Sequence[PhaseARowV9],
    output_dir: str | Path,
    config: PhaseBTrainingConfig,
    init_seeds: Sequence[int] = (401, 402, 403),
    corpus_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> dict[str, object]:
    """Train every init seed, select by total validation loss, export one.

    Same caveat as every trainer here, verbatim: validation loss is a
    gate, never a selector (V8_DESIGN §6.1) — every seed's losses are
    recorded so the seat-swapped duel can overrule this pick, and the
    project's practice is one artifact per seed (single-seed
    invocations) so all can be gauntleted.
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
        result = fit_phase_b_v9(corpus, phase_a, seed_config)
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
    validate_v9_weight_shapes(weights)
    architecture = default_v9_architecture()
    architecture["dropout"] = float(config.base.dropout)
    validate_v9_architecture(architecture)

    output_path.mkdir(parents=True, exist_ok=True)
    weights_document = _round9(
        {
            "format": MODEL_FORMAT,
            "format_version": MODEL_FORMAT_VERSION_V9,
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

    decisions = corpus.decisions
    per_street: dict[str, int] = {}
    branch_counts: dict[str, int] = {}
    for decision in decisions:
        per_street[decision.street] = per_street.get(decision.street, 0) + 1
        for label in decision.emitted:
            branch_counts[label] = branch_counts.get(label, 0) + 1

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
        # Harvest == serve: the manifest ships the corpus header's own
        # composed record, so the serve path sizes under exactly the g
        # state (identity, parameters, dial states) the labels baked in.
        "sizing": json.loads(
            json.dumps(corpus.sizing_record, sort_keys=True, allow_nan=False)
        ),
        "weights_file": weights_path.name,
        "weights_sha256": weights_sha256,
        "training_window": {
            "phase_b_corpus": None if corpus_path is None else str(corpus_path),
            "phase_b_decisions": len(decisions),
            "phase_b_tables": len({d.table_id for d in decisions}),
            "phase_b_branch_counts": dict(sorted(branch_counts.items())),
            "phase_b_decisions_per_street": dict(sorted(per_street.items())),
            "phase_b_free_spot_decisions": sum(
                1 for d in decisions if d.to_call_zero
            ),
            "phase_a_dataset": None if dataset_path is None else str(dataset_path),
            "phase_a_rows": len(phase_a),
            "row_count": len(decisions) + len(phase_a),
            "belief_fit_source": corpus.belief_fit_source,
            "equity_trials": corpus.equity_trials,
            "instrument": {
                "starting_stack": corpus.starting_stack,
                "big_blind": corpus.big_blind,
                "seeds": list(corpus.seeds),
            },
        },
        "engine_parameters": {
            "safety_gates": DEFAULT_SAFETY_GATES.to_mapping(),
            "temperature_shaping": DEFAULT_TEMPERATURE_SHAPING.to_mapping(),
            "tracker_settings": DEFAULT_TRACKER_SETTINGS.to_mapping(),
            "bluff_settings": DEFAULT_BLUFF_SETTINGS.to_mapping(),
        },
        "serve": {
            "ood_guard_indices": list(schema4.CONTEXT_INDICES_V9),
            # Harvest == serve, one number (pinned): load_policy_v9
            # honours this pin when the caller passes no explicit value.
            "equity_trials": corpus.equity_trials,
            "temperature": None,
            "note": (
                "Phase-B composed-value candidate: serve through "
                "learned_policy_v9.load_policy_v9 (the v9 composition). "
                "Promotion remains a separate, explicit, human-authorised "
                "act; this artifact must not be deployed by training"
            ),
        },
        "training": {
            "objective": TRAINING_OBJECTIVE_V9_PHASE_B,
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
                    "branch set, through the v9 fixed arithmetic (sigmoid "
                    "fold-through on the wager lanes, [0,1]-clamped "
                    "equities, residual clamped at ±cap·pot on wager "
                    "executions only, v_active = where(to_call_zero, bet, "
                    "call)); train/serve parity checked against "
                    "learned_policy_v9.compose_branch_values_v9 at export "
                    "from each decision's recorded context"
                ),
                "residual_cap_pot_fraction": config.residual_cap_pot_fraction,
                "supervised_source": (
                    "v9 Phase-A dataset batches interleaved every optimizer "
                    "step (component losses keep flowing)"
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
        },
        "evaluation": {
            "train_losses": _rounded(best["train_losses"]),  # type: ignore[arg-type]
            "validation_losses": _rounded(best["validation_losses"]),  # type: ignore[arg-type]
            "residual_share": best["residual_share"],
            "parity_check": best["parity_check"],
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
    parser.add_argument(
        "--phase-b-corpus", default=str(DEFAULT_PHASE_B_CORPUS_V9)
    )
    parser.add_argument(
        "--phase-a-dataset", default=str(DEFAULT_PHASE_A_DATASET_V9)
    )
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
    corpus = load_phase_b_corpus_v9(args.phase_b_corpus)
    print(
        f"phase-b corpus: {args.phase_b_corpus} "
        f"({len(corpus.decisions)} decisions, "
        f"equity_trials {corpus.equity_trials})"
    )
    rows = load_phase_a_dataset_v9(args.phase_a_dataset)
    print(f"phase-a dataset: {args.phase_a_dataset} ({len(rows)} rows)")
    summary = train_phase_b_candidate_v9(
        corpus,
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
    "DEFAULT_PHASE_A_DATASET_V9",
    "DEFAULT_PHASE_B_CORPUS_V9",
    "PhaseBCorpusV9",
    "PhaseBDecisionV9",
    "TRAINING_OBJECTIVE_V9_PHASE_B",
    "WAGER_COLUMN_SLICE",
    "WAGER_LANES",
    "compose_from_constants_v9",
    "fit_phase_b_v9",
    "load_phase_b_corpus_v9",
    "train_phase_b_candidate_v9",
    "wager_column_slice",
]
