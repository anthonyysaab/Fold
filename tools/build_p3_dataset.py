"""Fit the P3 strength-aware opponent from the complete-information archive.

Precondition **P3** (`PLAYERS.md`, "What is missing, and what to build
first"): every scripted archetype is card-blind, so none of them can punish a
policy that bets a random hand, and every sizing and aggression conclusion
this project has drawn has been blocked on that. `DECISIONS.md` §4.4 says how
to unblock it — public arena replays reveal **every seat's hole cards,
including folders**, so the field's fold behaviour can be *measured* instead
of invented.

This tool does two things.

**1. Dataset.** It walks the raw table replays of both collections and emits
one row for every decision where a player faced chips to call. The actor's
*true* holding is known, so the row carries the canonical strength percentile
(``strength_metric.strength_percentile`` — the only strength scale this
project reports on, ``V8_DESIGN.md`` §2) alongside the price, the street, the
seat count, the position and a board-texture proxy, labelled with whether the
actor folded.

Replay parsing is **not re-derived**: the event reconstruction is imported
from ``tools.collect_foreign_play_data`` via ``tools.build_phase_a_dataset``,
the proven parser of this archive.

**2. Fit.** A logistic regression on those rows, pure stdlib, by ridge-damped
IRLS (Newton). Deterministic — no sampling, no shuffling, no seeds anywhere
in the fit. Inputs are standardised and the means/stds ship with the
coefficients. Streets get their own model where the data supports one and are
pooled where it does not, by a rule fixed **before** the counts were looked
at (:data:`MIN_STREET_ROWS` / :data:`MIN_STREET_MINORITY`). Validation is a
20% hold-out split **by table**, so no table contributes to both sides.

**The invariant that decides validity.** The fitted coefficient on
``strength_percentile`` must be **negative** (better hands fold less) and the
coefficient on the price must be **positive** (worse prices fold more). If
either sign is wrong, the fit is broken or the label is inverted, and the run
stops without shipping. This is enforced here, again in
``strength_aware_opponent.P3Fit.from_document`` when the artifact is loaded,
and again in the tests against the shipped JSON.

**The instrument is validated before the result** (`CLAUDE.md` working
habits; three measurement scripts produced confident wrong answers before
that rule was adopted). Before it touches a single replay, the tool fits
three impossible-by-construction cases and refuses to continue unless all
three come out right:

- known coefficients drawn from a known logistic must be recovered;
- inverting every label must flip every coefficient's sign;
- labels independent of the features must produce ~zero coefficients and
  AUC ~ 0.5.

Read-only over the archive; writes only under ``artifacts/p3/``. It does not
promote, deploy, touch ``artifacts/approved.json``, or open a network socket.

Usage::

    python -m tools.build_p3_dataset                 # dataset + fit
    python -m tools.build_p3_dataset --fit-only      # refit an existing dataset
    python -m tools.build_p3_dataset --selftest-only # just the instrument check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import replace
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from engine.game_state import ArenaSnapshotError
from engine.strength_aware_opponent import (
    FIT_FEATURE_NAMES,
    PRICE_INDEX,
    STREETS,
    STRENGTH_INDEX,
    LogisticModel,
    P3Fit,
    P3FitError,
    decision_features,
)
from engine.strength_metric import StrengthMetricError
from tools.build_phase_a_dataset import _hole_cards_by_seat
from tools.collect_foreign_play_data import (
    _read_json,
    _reconstruct_state,
    _unwrap_rpc,
)

DEFAULT_ROOTS = (
    Path("foreign play data") / "20260812T082057Z_poker-playground_s13_top15",
    Path("foreign play data") / "20260815T210237Z_poker-playground_s14_top15",
)
DEFAULT_OUTPUT_DIR = Path("artifacts") / "p3"
DATASET_NAME = "p3-dataset.jsonl"
DATASET_SUMMARY_NAME = "p3-dataset.summary.json"
FIT_NAME = "p3-fit.json"

#: Our own agent is in this archive. It is not "the field" — its measured
#: behaviour is 14.3% fold against the field's 55.8%, and its strength
#: separation is +0.020 against the field's +0.150 (`PLAYERS.md`). Fitting an
#: opponent that is supposed to model the field on our own refuted policy
#: would bake that policy's defect into the instrument meant to expose it.
DEFAULT_EXCLUDED_AGENTS = ("Fold-ver-4",)

#: Per-street fit eligibility, fixed before the row counts were looked at.
#: A street gets its own model only with enough rows *and* enough of the
#: rarer class to estimate a slope; otherwise it is pooled and the pooling is
#: reported. Both are recorded in the artifact and are ablatable.
MIN_STREET_ROWS = 300
MIN_STREET_MINORITY = 50

#: Amendment 2026-08-16, made after the first run and recorded here rather
#: than quietly: a thin **postflop** street must never fall back to a model
#: dominated by preflop rows. The reason is structural, not performance —
#: preflop has no board, so ``texture`` is identically 0 there and carries
#: exactly zero weight, and the preflop strength percentile comes from a
#: different construction (the committed 169-class lookup) than the postflop
#: exact enumeration (``V8_DESIGN.md`` §2). A model fitted 88% on preflop
#: rows cannot serve a turn decision. So thin postflop streets pool with each
#: other, and if that pool is still under the threshold it merges with the
#: qualifying postflop streets into one ``postflop`` model. The all-street
#: model survives only as the declared fallback for a street with no rows at
#: all. ``pooling_audit`` in the artifact scores the alternative that was not
#: taken on the held-out rows, so the choice is checkable rather than
#: asserted.
POSTFLOP_STREETS = ("flop", "turn", "river")

#: Ridge on the standardised non-intercept coefficients. Present for
#: conditioning and separation safety, not for regularisation strength: at
#: thousands of rows a penalty of 1.0 is negligible against the likelihood.
#: Recorded in the artifact and ablatable via ``--l2``.
DEFAULT_L2 = 1.0

#: The price-support clamp quantile (repair of the overbet blind spot,
#: 2026-08-16). Each shipped model's ``price_cap`` is this quantile of the
#: ``pot_odds`` observed on the model's **own training rows** — the fitted
#: support. At serve the price input is clamped from above at the cap, so
#: beyond it the fold probability is flat in bet size and a bigger overbet
#: buys no extra fold equity. Measured motivation: the archive's fields bet
#: at most ~1.1x pot (flop train max pot_odds 0.392), so every overbet was
#: an extrapolation, and the extrapolated positive price slope made a
#: zero-equity 4x-pot bluff *more* profitable than a half-pot one
#: (``tests/test_p3_audit.py`` ``OverbetPunishmentTests``, pre-repair).
#: The 0.99 is a quantile *choice*, not a fitted number: recorded in the
#: artifact and **ablatable** via ``--price-quantile``.
PRICE_SUPPORT_QUANTILE = 0.99

#: Hold-out share, by table hash.
VALIDATION_MODULUS = 5  # 1 in 5 tables -> 20%

_MAX_NEWTON_STEPS = 200
_NEWTON_TOLERANCE = 1e-10


class P3DatasetError(RuntimeError):
    """An impossible-by-construction invariant of this tool was violated.

    Deliberately not a ``ValueError``: per-decision degrade handling swallows
    malformed-replay ``ValueError``s, and an invariant violation must never
    be silently counted as a skip — it means this generator is wrong.
    """


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _is_validation_table(table_id: str) -> bool:
    """Deterministic 20% hold-out, split by table so nothing leaks."""

    digest = hashlib.sha256(table_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % VALIDATION_MODULUS == 0


#: Board cards visible on each street, by definition of the street.
_BOARD_LENGTH = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def decision_board(
    events: Sequence[Mapping[str, Any]], sequence: int, street: str
) -> tuple[str, ...]:
    """The board as it stood **when the actor decided**.

    Measured 2026-08-16, and the reason this function exists: the ``snapshot``
    attached to an ``ActionTaken`` event is the state *after* the action
    resolved. ``_reconstruct_state`` already repairs the actor's own stack and
    bet from the payload's ``stackBefore`` / ``actorCurrentBetBefore``, but it
    does not repair ``boardCards`` — so on the **last decision of a street**
    the reconstructed state carries the next street's cards. Reading strength
    off that board is a look-ahead leak: the actor would be scored on cards it
    had not seen.

    So the board is rebuilt from the ``StreetDealt`` events that precede this
    action, and cross-checked against the street's definitional card count
    (0 / 3 / 4 / 5). A mismatch raises and the decision is skipped and
    counted — never silently accepted.
    """

    board: list[str] = []
    for event in events:
        if not isinstance(event, Mapping) or event.get("type") != "StreetDealt":
            continue
        event_sequence = event.get("sequence")
        if not isinstance(event_sequence, int) or event_sequence >= sequence:
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        cumulative = payload.get("boardCards")
        if isinstance(cumulative, list) and all(
            isinstance(card, str) for card in cumulative
        ):
            board = [str(card) for card in cumulative]
            continue
        fresh = payload.get("cards")
        if isinstance(fresh, list) and all(isinstance(card, str) for card in fresh):
            board.extend(str(card) for card in fresh)
    expected = _BOARD_LENGTH.get(street)
    if expected is None:
        raise ValueError(f"unsupported street {street!r}")
    if len(board) != expected:
        raise ValueError(
            f"street {street!r} implies {expected} board cards but the "
            f"StreetDealt history gives {len(board)}"
        )
    return tuple(board)


def replay_rows(
    replay: Mapping[str, Any], *, excluded_agents: frozenset[str]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Every ``to_call > 0`` decision in one unwrapped replay, plus counts."""

    stats: Counter[str] = Counter()
    events = replay.get("events")
    if not isinstance(events, list):
        stats["replay_without_events"] += 1
        return [], stats
    if not isinstance(replay.get("table"), Mapping):
        stats["replay_without_table"] += 1
        return [], stats
    holes_by_seat = _hole_cards_by_seat(events)
    if not holes_by_seat:
        stats["replay_without_hole_cards"] += 1
        return [], stats

    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_type = event.get("type")
        if event_type == "TimeoutAction":
            # Not a decision the actor made. Counted, never labelled.
            stats["timeout_actions"] += 1
            continue
        if event_type != "ActionTaken":
            continue
        stats["action_events"] += 1
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            stats["skip:no_payload"] += 1
            continue
        agent_name = payload.get("agentName")
        if isinstance(agent_name, str) and agent_name in excluded_agents:
            stats["excluded_agent_decisions"] += 1
            continue
        allowed = payload.get("allowedActions")
        if not isinstance(allowed, Mapping):
            stats["skip:no_allowed_actions"] += 1
            continue
        call_chips = allowed.get("callChips")
        if isinstance(call_chips, bool) or not isinstance(call_chips, (int, float)):
            stats["skip:no_call_chips"] += 1
            continue
        if int(call_chips) <= 0:
            stats["not_facing_a_wager"] += 1
            continue
        available = allowed.get("availableActions") or []
        if "fold" not in available:
            # Facing a wager with no legal fold: the label would be a
            # constant, not a choice.
            stats["skip:fold_not_legal"] += 1
            continue
        actor_seat = payload.get("seatNumber")
        if not isinstance(actor_seat, int) or isinstance(actor_seat, bool):
            stats["skip:no_seat_number"] += 1
            continue
        hole = holes_by_seat.get(actor_seat)
        if hole is None:
            stats["skip:actor_hole_not_revealed"] += 1
            continue
        sequence = event.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            stats["skip:no_sequence"] += 1
            continue
        action = str(payload.get("action") or "").casefold()
        if not action:
            stats["skip:no_action"] += 1
            continue

        street = str(event.get("street") or "").casefold()
        try:
            state = dict(_reconstruct_state(dict(replay), dict(event)))
            snapshot_board = list(state.get("boardCards") or [])
            # Repair the look-ahead the shared reconstruction leaves behind.
            board = decision_board(events, sequence, street)
            if list(board) != snapshot_board:
                stats["board_corrected"] += 1
            state["boardCards"] = list(board)
            decision = decision_features(state, allowed, hole)
        except (
            ArenaSnapshotError,
            StrengthMetricError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            stats["skipped_decisions"] += 1
            stats[f"skip:{type(error).__name__}"] += 1
            continue

        table_id = str(state.get("tableId") or state.get("id") or "")
        if not table_id:
            stats["skip:no_table_id"] += 1
            continue

        row = {
            "table_id": table_id,
            "sequence": sequence,
            "seat": actor_seat,
            "actor_agent": agent_name if isinstance(agent_name, str) else "",
            "street": decision.street,
            "strength_percentile": decision.strength_percentile,
            "pot_odds": decision.pot_odds,
            "bet_to_pot": decision.bet_to_pot,
            "texture": decision.texture,
            "active_players": decision.active_players,
            "position_unit": decision.position_unit,
            "to_call": decision.to_call,
            "pot": decision.pot,
            "action": action,
            "label_fold": 1 if action == "fold" else 0,
            "split": "val" if _is_validation_table(table_id) else "train",
        }
        _validate_row(row)
        rows.append(row)
    return rows, stats


def _validate_row(row: Mapping[str, Any]) -> None:
    """Impossible-by-construction row checks, enforced at emission."""

    if not 0.0 <= row["strength_percentile"] <= 1.0:
        raise P3DatasetError(f"strength_percentile {row['strength_percentile']!r}")
    if not 0.0 < row["pot_odds"] < 1.0:
        # to_call > 0 and pot > 0 (blinds are posted before any decision), so
        # the price is strictly interior. A 0 or 1 here means the pot or the
        # call was misread.
        raise P3DatasetError(f"pot_odds {row['pot_odds']!r} is not in (0, 1)")
    if row["to_call"] <= 0:
        raise P3DatasetError("a row was emitted for a decision facing no wager")
    if not 0.0 <= row["texture"] <= 1.0:
        raise P3DatasetError(f"texture {row['texture']!r}")
    if not 0.0 <= row["position_unit"] <= 1.0:
        raise P3DatasetError(f"position_unit {row['position_unit']!r}")
    if row["active_players"] < 2:
        raise P3DatasetError(
            f"active_players {row['active_players']!r}: a decision needs "
            "at least two live seats"
        )
    if row["street"] == "preflop" and row["texture"] != 0.0:
        raise P3DatasetError("preflop texture must be 0 — there is no board")
    if row["label_fold"] not in (0, 1):
        raise P3DatasetError(f"label_fold {row['label_fold']!r}")
    if (row["action"] == "fold") != bool(row["label_fold"]):
        raise P3DatasetError("label_fold disagrees with the recorded action")


def _process_file(args: tuple[str, tuple[str, ...]]) -> tuple[
    list[dict[str, Any]], dict[str, int]
]:
    """Worker: one raw table file to its rows (importable for pickling)."""

    path_text, excluded = args
    path = Path(path_text)
    stats: Counter[str] = Counter()
    try:
        replay = _unwrap_rpc(_read_json(path))
    except (OSError, ValueError):
        stats["unreadable_files"] += 1
        return [], dict(stats)
    if not isinstance(replay, dict):
        stats["unreadable_files"] += 1
        return [], dict(stats)
    rows, replay_stats = replay_rows(replay, excluded_agents=frozenset(excluded))
    stats.update(replay_stats)
    return rows, dict(stats)


def build_dataset(
    roots: Sequence[Path],
    output: Path,
    *,
    excluded_agents: Sequence[str] = DEFAULT_EXCLUDED_AGENTS,
    workers: int = 1,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Walk the archive, emit rows, write the dataset and its sidecar."""

    files: list[tuple[str, Path]] = []
    for root in roots:
        table_dir = Path(root) / "raw" / "tables"
        if not table_dir.is_dir():
            raise FileNotFoundError(f"no raw tables directory under {root}")
        collection_files = sorted(table_dir.glob("*.json"))
        if not collection_files:
            raise FileNotFoundError(f"no raw table replays under {table_dir}")
        files.extend((Path(root).name, path) for path in collection_files)
    if limit is not None:
        files = files[:limit]

    started = time.monotonic()
    excluded = tuple(excluded_agents)
    work = [(str(path), excluded) for _, path in files]
    names = [name for name, _ in files]
    all_rows: list[dict[str, Any]] = []
    stats_by_collection: dict[str, Counter[str]] = {
        name: Counter() for name in dict.fromkeys(names)
    }
    tables_with_rows: Counter[str] = Counter()

    def _consume(collection: str, result: tuple[list[dict[str, Any]], dict[str, int]]):
        rows, stats = result
        all_rows.extend(rows)
        stats_by_collection[collection].update(stats)
        if rows:
            tables_with_rows[collection] += 1

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(
                pool.map(_process_file, work, chunksize=8)
            ):
                _consume(names[index], result)
                if (index + 1) % 100 == 0 or index + 1 == len(work):
                    print(
                        f"processed {index + 1}/{len(work)} files, "
                        f"{len(all_rows)} rows, "
                        f"{time.monotonic() - started:.0f}s",
                        flush=True,
                    )
    else:
        for index, item in enumerate(work):
            _consume(names[index], _process_file(item))
            if (index + 1) % 100 == 0 or index + 1 == len(work):
                print(
                    f"processed {index + 1}/{len(work)} files, "
                    f"{len(all_rows)} rows, {time.monotonic() - started:.0f}s",
                    flush=True,
                )

    if not all_rows:
        raise P3DatasetError("no decision rows were produced")

    # Deterministic order regardless of worker scheduling: the fit sums
    # floats in this order, so row order is part of reproducibility.
    all_rows.sort(key=lambda row: (row["table_id"], row["sequence"], row["seat"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(output)

    summary = {
        "generator": {
            "tool": "tools.build_p3_dataset",
            "roots": [str(root) for root in roots],
            "excluded_agents": list(excluded),
            "limit": limit,
            "validation_modulus": VALIDATION_MODULUS,
        },
        "counts": {
            "files": len(files),
            "tables_with_rows": sum(tables_with_rows.values()),
            "rows": len(all_rows),
            "folds": sum(row["label_fold"] for row in all_rows),
            "train_rows": sum(row["split"] == "train" for row in all_rows),
            "val_rows": sum(row["split"] == "val" for row in all_rows),
        },
        "per_street": _street_counts(all_rows),
        "per_collection": {
            name: {
                "tables_with_rows": tables_with_rows.get(name, 0),
                **dict(sorted(stats_by_collection[name].items())),
            }
            for name in dict.fromkeys(names)
        },
        "files": {"dataset": output.name},
    }
    sidecar = output.parent / DATASET_SUMMARY_NAME
    sidecar.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {output} ({len(all_rows)} rows)")
    print(f"wrote {sidecar}")
    return all_rows, summary


def _street_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        street: {"rows": 0, "folds": 0, "train": 0, "val": 0} for street in STREETS
    }
    for row in rows:
        entry = counts[row["street"]]
        entry["rows"] += 1
        entry["folds"] += int(row["label_fold"])
        entry[row["split"]] += 1
    return counts


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise P3DatasetError(f"dataset {path} is empty")
    return rows


# ---------------------------------------------------------------------------
# Logistic regression — pure stdlib, deterministic ridge-IRLS
# ---------------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    exponent = math.exp(max(z, -700.0))
    return exponent / (1.0 + exponent)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Small, dense, exact."""

    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise P3FitError("the Newton system is singular — fit is degenerate")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / pivot_value
            if factor == 0.0:
                continue
            for k in range(column, size + 1):
                augmented[row][k] -= factor * augmented[column][k]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = augmented[row][size]
        for k in range(row + 1, size):
            total -= augmented[row][k] * solution[k]
        solution[row] = total / augmented[row][row]
    return solution


def standardisation(
    design: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Column means and (population) stds, with a floor for constant columns."""

    if not design:
        raise P3FitError("cannot standardise an empty design matrix")
    width = len(design[0])
    count = len(design)
    means = [0.0] * width
    for row in design:
        for index in range(width):
            means[index] += row[index]
    means = [value / count for value in means]
    variances = [0.0] * width
    for row in design:
        for index in range(width):
            delta = row[index] - means[index]
            variances[index] += delta * delta
    stds = []
    for value in variances:
        std = math.sqrt(value / count)
        # A constant column carries no information; a std of 1.0 leaves its
        # centred values at exactly 0, so the ridge drives its coefficient to
        # 0 rather than the fit dividing by ~0.
        stds.append(std if std > 1e-9 else 1.0)
    return tuple(means), tuple(stds)


def fit_logistic(
    design: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    l2: float = DEFAULT_L2,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    feature_names: Sequence[str] = FIT_FEATURE_NAMES,
) -> LogisticModel:
    """Deterministic ridge-IRLS logistic regression. No RNG, no shuffling."""

    if len(design) != len(labels):
        raise P3FitError("design and labels disagree on row count")
    if not design:
        raise P3FitError("cannot fit an empty design matrix")
    width = len(design[0])
    if width != len(feature_names):
        raise P3FitError("design width does not match the feature contract")
    if any(len(row) != width for row in design):
        raise P3FitError("design matrix is ragged")
    if any(label not in (0, 1) for label in labels):
        raise P3FitError("labels must be 0 or 1")

    if mean is None or std is None:
        mean, std = standardisation(design)
    centred = [
        [(row[index] - mean[index]) / std[index] for index in range(width)]
        for row in design
    ]

    size = width + 1  # index 0 is the intercept
    weights = [0.0] * size
    for _ in range(_MAX_NEWTON_STEPS):
        gradient = [0.0] * size
        hessian = [[0.0] * size for _ in range(size)]
        for row, label in zip(centred, labels):
            z = weights[0]
            for index in range(width):
                z += weights[index + 1] * row[index]
            probability = _sigmoid(z)
            residual = label - probability
            variance = max(probability * (1.0 - probability), 1e-9)
            gradient[0] += residual
            hessian[0][0] += variance
            for a in range(width):
                gradient[a + 1] += residual * row[a]
                hessian[0][a + 1] += variance * row[a]
                hessian[a + 1][0] += variance * row[a]
                for b in range(a, width):
                    value = variance * row[a] * row[b]
                    hessian[a + 1][b + 1] += value
                    if a != b:
                        hessian[b + 1][a + 1] += value
        for index in range(1, size):  # ridge never penalises the intercept
            gradient[index] -= l2 * weights[index]
            hessian[index][index] += l2
        step = _solve(hessian, gradient)
        for index in range(size):
            weights[index] += step[index]
        if max(abs(value) for value in step) < _NEWTON_TOLERANCE:
            break

    return LogisticModel(
        intercept=weights[0],
        coefficients=tuple(weights[1:]),
        mean=tuple(mean),
        std=tuple(std),
        feature_names=tuple(feature_names),
    )


def price_support_quantile(values: Sequence[float], q: float) -> float:
    """Deterministic linear-interpolation quantile of ``values``.

    Used to derive each model's ``price_cap`` from its own training rows'
    observed ``pot_odds``. No RNG, no mutation of the input; validated in
    :func:`selftest` against closed forms before any real number is read.
    """

    if not values:
        raise P3FitError("cannot take a quantile of no observations")
    if not 0.0 <= q <= 1.0:
        raise P3FitError(f"quantile {q!r} is not in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def natural_scale(model: LogisticModel) -> dict[str, Any]:
    """The same model written on raw feature units, for readability only.

    Standardisation divides by a positive std, so signs are identical on
    both scales; this exists so a reader can see "a full sweep of the
    strength percentile moves the log-odds by X" without doing the algebra.
    """

    intercept = model.intercept
    coefficients = []
    for weight, centre, scale in zip(model.coefficients, model.mean, model.std):
        raw = weight / scale
        coefficients.append(raw)
        intercept -= raw * centre
    return {"intercept": intercept, "coefficients": coefficients}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def log_loss(predictions: Sequence[float], labels: Sequence[int]) -> float:
    total = 0.0
    for probability, label in zip(predictions, labels):
        clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
        total -= math.log(clipped) if label else math.log(1.0 - clipped)
    return total / len(labels)


def brier(predictions: Sequence[float], labels: Sequence[int]) -> float:
    return sum(
        (probability - label) ** 2
        for probability, label in zip(predictions, labels)
    ) / len(labels)


def auc(predictions: Sequence[float], labels: Sequence[int]) -> float | None:
    """Rank AUC with average ranks for ties; ``None`` if a class is absent."""

    count = len(labels)
    positives = sum(labels)
    negatives = count - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(count), key=lambda index: predictions[index])
    ranks = [0.0] * count
    index = 0
    while index < count:
        end = index
        while (
            end + 1 < count
            and predictions[order[end + 1]] == predictions[order[index]]
        ):
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    positive_rank_sum = sum(
        ranks[index] for index in range(count) if labels[index] == 1
    )
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _metrics(
    predictions: Sequence[float], labels: Sequence[int]
) -> dict[str, Any]:
    if not labels:
        return {"rows": 0}
    return {
        "rows": len(labels),
        "base_rate": sum(labels) / len(labels),
        "log_loss": log_loss(predictions, labels),
        "auc": auc(predictions, labels),
        "brier": brier(predictions, labels),
    }


def calibration_table(
    predictions: Sequence[float], labels: Sequence[int], bins: int = 10
) -> list[dict[str, Any]]:
    """Predicted vs observed fold rate by equal-count decile of prediction."""

    count = len(labels)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: (predictions[index], index))
    table: list[dict[str, Any]] = []
    for bucket in range(bins):
        start = bucket * count // bins
        end = (bucket + 1) * count // bins
        if end <= start:
            continue
        members = order[start:end]
        table.append(
            {
                "decile": bucket + 1,
                "rows": len(members),
                "predicted_low": predictions[members[0]],
                "predicted_high": predictions[members[-1]],
                "mean_predicted": sum(predictions[i] for i in members) / len(members),
                "observed": sum(labels[i] for i in members) / len(members),
            }
        )
    return table


# ---------------------------------------------------------------------------
# Instrument validation — run BEFORE any real number is believed
# ---------------------------------------------------------------------------


def _synthetic(
    seed: int, count: int, truth: Sequence[float], intercept: float
) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(seed)
    design: list[list[float]] = []
    labels: list[int] = []
    for _ in range(count):
        row = [
            rng.random(),  # strength_percentile-like, [0, 1]
            0.1 + 0.6 * rng.random(),  # pot_odds-like
            rng.random(),  # texture-like
            float(rng.randint(2, 6)),  # active_players-like
            rng.random(),  # position_unit-like
        ]
        z = intercept + sum(w * x for w, x in zip(truth, row))
        design.append(row)
        labels.append(1 if rng.random() < _sigmoid(z) else 0)
    return design, labels


def selftest(l2: float = 1e-6, verbose: bool = True) -> dict[str, Any]:
    """Three impossible-by-construction checks on the fitter itself.

    House rule (`CLAUDE.md`): validate any measurement script against an
    invariant that cannot hold if it is wrong, *before* believing its
    numbers. A logistic fitter can be wrong in exactly the ways that would
    make the P3 sign invariant meaningless — a flipped label convention, a
    sign error in the gradient, a broken solve — and all three are visible
    here without touching a replay.
    """

    truth = [-3.0, 2.5, 0.4, 0.3, -0.5]
    intercept = 0.2
    design, labels = _synthetic(20260816, 20000, truth, intercept)
    model = fit_logistic(design, labels, l2=l2)
    raw = natural_scale(model)
    recovered = raw["coefficients"]
    errors = [abs(a - b) for a, b in zip(recovered, truth)]
    recovery_ok = max(errors) < 0.35 and all(
        (a > 0) == (b > 0) for a, b in zip(recovered, truth)
    )

    inverted = fit_logistic(design, [1 - label for label in labels], l2=l2)
    inverted_raw = natural_scale(inverted)["coefficients"]
    inversion_ok = all(
        (a < 0) != (b < 0) and abs(a + b) < 0.35
        for a, b in zip(recovered, inverted_raw)
    )

    null_rng = random.Random(99)
    null_labels = [1 if null_rng.random() < 0.5 else 0 for _ in design]
    null_model = fit_logistic(design, null_labels, l2=l2)
    null_raw = natural_scale(null_model)["coefficients"]
    null_predictions = [null_model.predict(row) for row in design]
    null_auc = auc(null_predictions, null_labels) or 0.0
    null_ok = max(abs(value) for value in null_raw) < 0.20 and abs(
        null_auc - 0.5
    ) < 0.02

    # The quantile helper that derives the price-support cap, against
    # closed forms it cannot get right by accident.
    quantile_ok = (
        price_support_quantile([5.0], 0.99) == 5.0
        and price_support_quantile([float(v) for v in range(101)], 0.5) == 50.0
        and price_support_quantile([float(v) for v in range(101)], 0.99) == 99.0
        and price_support_quantile([float(v) for v in range(101)], 1.0) == 100.0
        and abs(price_support_quantile([0.0, 1.0], 0.75) - 0.75) < 1e-15
        and price_support_quantile([3.0, 1.0, 2.0], 0.0) == 1.0
        and price_support_quantile([3.0, 1.0, 2.0], 1.0) == 3.0
    )

    report = {
        "recovery": {
            "truth": list(truth),
            "recovered": recovered,
            "max_abs_error": max(errors),
            "passed": recovery_ok,
        },
        "label_inversion": {
            "recovered": inverted_raw,
            "passed": inversion_ok,
        },
        "null_labels": {
            "max_abs_coefficient": max(abs(value) for value in null_raw),
            "auc": null_auc,
            "passed": null_ok,
        },
        "price_quantile_closed_forms": {
            "passed": quantile_ok,
        },
        "passed": bool(recovery_ok and inversion_ok and null_ok and quantile_ok),
    }
    if verbose:
        print("instrument validation (before any real number is believed)")
        print(f"  known-coefficient recovery : {report['recovery']['passed']} "
              f"(max abs error {report['recovery']['max_abs_error']:.4f})")
        print(f"  label inversion flips signs: {report['label_inversion']['passed']}")
        print(f"  null labels give ~0 / AUC .5: {report['null_labels']['passed']} "
              f"(auc {null_auc:.4f})")
        print(f"  quantile closed forms      : {quantile_ok}")
    if not report["passed"]:
        raise P3FitError(
            "the logistic fitter failed its own impossible-by-construction "
            f"checks; no number from it is believable: {report}"
        )
    return report


# ---------------------------------------------------------------------------
# Fit assembly
# ---------------------------------------------------------------------------


def _design(rows: Sequence[Mapping[str, Any]]) -> tuple[
    list[list[float]], list[int]
]:
    design = [
        [float(row[name]) for name in FIT_FEATURE_NAMES] for row in rows
    ]
    labels = [int(row["label_fold"]) for row in rows]
    return design, labels


def _street_plan(rows: Sequence[Mapping[str, Any]]) -> tuple[
    dict[str, str], dict[str, list[str]], dict[str, Any]
]:
    """Decide which streets get their own model and which are pooled.

    The rule is fixed above, before the counts were seen: a street needs
    :data:`MIN_STREET_ROWS` training rows and :data:`MIN_STREET_MINORITY` of
    the rarer class. Everything that fails is pooled into one model; if the
    pool itself is too thin, those streets fall back to the all-street model.
    """

    train = [row for row in rows if row["split"] == "train"]
    per_street: dict[str, dict[str, int]] = {}
    for street in STREETS:
        members = [row for row in train if row["street"] == street]
        folds = sum(row["label_fold"] for row in members)
        per_street[street] = {
            "train_rows": len(members),
            "folds": folds,
            "non_folds": len(members) - folds,
            "minority": min(folds, len(members) - folds),
        }

    def _qualifies(streets: Sequence[str]) -> tuple[bool, int, int]:
        members = [row for row in train if row["street"] in streets]
        folds = sum(row["label_fold"] for row in members)
        minority = min(folds, len(members) - folds)
        return (
            len(members) >= MIN_STREET_ROWS and minority >= MIN_STREET_MINORITY,
            len(members),
            minority,
        )

    own: list[str] = []
    thin: list[str] = []
    for street in STREETS:
        qualified, _, _ = _qualifies([street])
        if qualified:
            own.append(street)
        elif per_street[street]["train_rows"] > 0:
            thin.append(street)

    members_by_model: dict[str, list[str]] = {street: [street] for street in own}
    street_to_model: dict[str, str] = {street: street for street in own}

    thin_postflop = [street for street in thin if street in POSTFLOP_STREETS]
    thin_preflop = [street for street in thin if street not in POSTFLOP_STREETS]
    pooling: dict[str, Any] = {}

    if thin_postflop:
        qualified, rows_count, minority = _qualifies(thin_postflop)
        if qualified:
            key = "+".join(thin_postflop)
            members_by_model[key] = list(thin_postflop)
        else:
            # Merge with the postflop streets that did qualify. Never with
            # preflop (see POSTFLOP_STREETS).
            key = "postflop"
            members_by_model[key] = [
                street
                for street in POSTFLOP_STREETS
                if per_street[street]["train_rows"] > 0
            ]
        for street in thin_postflop:
            street_to_model[street] = key
        _, merged_rows, merged_minority = _qualifies(members_by_model[key])
        pooling["postflop"] = {
            "thin_streets": thin_postflop,
            "model": key,
            "model_streets": members_by_model[key],
            "thin_only_train_rows": rows_count,
            "thin_only_minority": minority,
            "model_train_rows": merged_rows,
            "model_minority": merged_minority,
        }

    for street in thin_preflop:
        street_to_model[street] = "all"
    if thin_preflop:
        pooling["preflop"] = {"thin_streets": thin_preflop, "model": "all"}

    # Always fitted, and the declared fallback for a street with no rows.
    members_by_model["all"] = list(STREETS)
    for street in STREETS:
        street_to_model.setdefault(street, "all")

    plan = {
        "rule": {
            "min_street_rows": MIN_STREET_ROWS,
            "min_street_minority": MIN_STREET_MINORITY,
            "stated": "thresholds fixed before the row counts were inspected",
            "amendment": (
                "a thin postflop street pools with postflop streets only, "
                "never with preflop: preflop has no board (texture is "
                "identically 0 and takes exactly zero weight) and its "
                "strength percentile is a different construction from the "
                "postflop enumeration (V8_DESIGN.md §2). Structural, not "
                "performance-selected; see pooling_audit."
            ),
            "ablatable": True,
        },
        "per_street_train": per_street,
        "own_model": own,
        "thin": thin,
        "pooling": pooling,
    }
    return street_to_model, members_by_model, plan


def _band(
    models: Mapping[str, LogisticModel],
    members_by_model: Mapping[str, list[str]],
    rows: Sequence[Mapping[str, Any]],
    street_to_model: Mapping[str, str],
) -> dict[str, Any]:
    """Derive the clamp band from the data rather than choosing one.

    The band is a **rail, not a shaper**. Its job is to stop a model
    extrapolated off its support from folding always — an opponent that
    always folds is unbeatable-by-folding and would flatter any aggressive
    candidate, which is the opposite of what P3 is for. So it is the union
    of two measured envelopes over the training rows:

    - the range of the fit's own predictions on real decisions, so on-support
      play is never clamped (the clamp must not silently reshape the fitted
      response), and
    - the Laplace-smoothed observed fold rate of the most-calling and
      most-folding prediction deciles, so the rail sits where the field
      actually lives.

    A final Laplace guard from the sample size keeps it strictly inside
    (0, 1) whatever the above produce.
    """

    train = [row for row in rows if row["split"] == "train"]
    predictions: list[float] = []
    labels: list[int] = []
    for row in train:
        model = models[street_to_model.get(row["street"], "all")]
        predictions.append(
            model.predict([float(row[name]) for name in FIT_FEATURE_NAMES])
        )
        labels.append(int(row["label_fold"]))
    table = calibration_table(predictions, labels)
    smoothed = [
        (entry["observed"] * entry["rows"] + 1.0) / (entry["rows"] + 2.0)
        for entry in table
    ]
    low = min([min(predictions), *smoothed])
    high = max([max(predictions), *smoothed])
    guard_low = 1.0 / (len(train) + 2.0)
    guard_high = (len(train) + 1.0) / (len(train) + 2.0)
    low = max(low, guard_low)
    high = min(high, guard_high)
    if not low < high:
        raise P3FitError("derived clamp band collapsed")
    return {
        "low": low,
        "high": high,
        "method": "union of the fit's predicted range on training decisions "
        "and the Laplace-smoothed observed fold rate of the extreme "
        "prediction deciles, guarded to stay strictly inside (0, 1)",
        "ablatable": True,
        "predicted_range": [min(predictions), max(predictions)],
        "decile_observed_range": [min(smoothed), max(smoothed)],
        "guard": [guard_low, guard_high],
    }


def fit_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    l2: float = DEFAULT_L2,
    price_quantile: float = PRICE_SUPPORT_QUANTILE,
) -> dict[str, Any]:
    """Fit every model, score it, and assemble the artifact document."""

    street_to_model, members_by_model, plan = _street_plan(rows)

    models: dict[str, LogisticModel] = {}
    reports: dict[str, Any] = {}
    price_support: dict[str, Any] = {}
    for key, streets in members_by_model.items():
        members = [row for row in rows if row["street"] in streets]
        train = [row for row in members if row["split"] == "train"]
        validation = [row for row in members if row["split"] == "val"]
        if not train:
            raise P3FitError(f"model {key!r} has no training rows")
        train_design, train_labels = _design(train)
        model = fit_logistic(train_design, train_labels, l2=l2)
        # Attach the price-support clamp BEFORE any prediction is made, so
        # every metric, calibration table and band below describes the model
        # that actually ships (clamped), not the unclamped fit.
        train_prices = [float(row["pot_odds"]) for row in train]
        cap = price_support_quantile(train_prices, price_quantile)
        model = replace(model, price_cap=cap)
        price_support[key] = {
            "train_rows": len(train_prices),
            "price_cap": cap,
            "train_price_max": max(train_prices),
            "train_price_median": price_support_quantile(train_prices, 0.5),
        }
        models[key] = model

        train_predictions = [model.predict(row) for row in train_design]
        validation_design, validation_labels = _design(validation)
        validation_predictions = [model.predict(row) for row in validation_design]

        full_design, full_labels = _design(members)
        full_model = fit_logistic(full_design, full_labels, l2=l2)

        reports[key] = {
            "streets": list(streets),
            **model.to_document(),
            "natural_scale": natural_scale(model),
            "coefficients_by_name": dict(
                zip(FIT_FEATURE_NAMES, model.coefficients)
            ),
            "metrics": {
                "train": _metrics(train_predictions, train_labels),
                "val": _metrics(validation_predictions, validation_labels),
            },
            "calibration_val": calibration_table(
                validation_predictions, validation_labels
            ),
            "calibration_train": calibration_table(
                train_predictions, train_labels
            ),
            # Shipped coefficients are the train-split ones — the exact
            # parameters the validation metrics and calibration table above
            # describe. The all-rows refit is recorded for a later ablation,
            # never silently substituted.
            "full_data_refit": {
                **full_model.to_document(),
                "natural_scale": natural_scale(full_model),
                "rows": len(members),
            },
        }

    # ---- THE INVARIANT THAT DECIDES VALIDITY -----------------------------
    # Checked here, before any metric is printed or any file is written.
    violations = []
    for key, model in sorted(models.items()):
        try:
            model.check_sign_invariant(key)
        except P3FitError as error:
            violations.append(str(error))
        full = reports[key]["full_data_refit"]
        if not full["coefficients"][STRENGTH_INDEX] < 0.0:
            violations.append(f"{key} (full-data refit): strength coefficient >= 0")
        if not full["coefficients"][PRICE_INDEX] > 0.0:
            violations.append(f"{key} (full-data refit): price coefficient <= 0")
    if violations:
        raise P3FitError(
            "SIGN INVARIANT FAILED — the fit is broken or the label is "
            "inverted; nothing is shipped. " + "; ".join(violations)
        )

    # ---- pooling audit ----------------------------------------------------
    # The amendment above is argued structurally, so it must not be taken on
    # trust: score the model each pooled street was assigned AND the
    # all-street model it would otherwise have fallen back to, on the same
    # held-out rows. Row counts are printed with the numbers because at this
    # sample size the comparison may well be unresolvable — and an
    # unresolvable comparison is reported as unresolvable, not as a win.
    pooling_audit: dict[str, Any] = {}
    for group in plan["pooling"].values():
        streets = list(group["thin_streets"])
        held_out = [
            row
            for row in rows
            if row["street"] in streets and row["split"] == "val"
        ]
        design, labels = _design(held_out)
        entry: dict[str, Any] = {"streets": streets, "val_rows": len(held_out)}
        for candidate in dict.fromkeys([group["model"], "all"]):
            model = models[candidate]
            entry[candidate] = _metrics(
                [model.predict(row) for row in design], labels
            )
        entry["assigned"] = group["model"]
        pooling_audit["+".join(streets)] = entry

    band = _band(models, members_by_model, rows, street_to_model)

    document = {
        "generator": {
            "tool": "tools.build_p3_dataset",
            "what_this_is": (
                "a fitted model of the observed field's fold frequency, "
                "conditional on the actor's true hand strength and the price "
                "it was laid"
            ),
            "what_this_is_not": (
                "an optimal player, an exploitative player, or a "
                "range-balanced one; it makes no claim beyond folding more "
                "with worse hands and worse prices"
            ),
            "l2": l2,
            "l2_ablatable": True,
            "optimizer": "ridge-damped IRLS (Newton), deterministic, no RNG",
            "validation_split": f"1 in {VALIDATION_MODULUS} tables by sha256 "
            "of table_id",
        },
        "features": list(FIT_FEATURE_NAMES),
        "excluded_regressors": {
            "bet_to_pot": "recorded per row but not fitted: pot_odds == "
            "bet_to_pot / (1 + bet_to_pot) exactly, so fitting both is exact "
            "collinearity and the price sign invariant becomes unreadable"
        },
        "price_support": {
            "clamped_feature": "pot_odds",
            "why": (
                "repair of the overbet blind spot (2026-08-16): the archive's "
                "fields bet at most ~1.1x pot, so every overbet price is an "
                "extrapolation, and the extrapolated positive price slope "
                "made a zero-equity bluff's EV RISE with bet size. Above the "
                "cap the fold probability is flat in size, so a bigger "
                "overbet buys no extra fold equity."
            ),
            "method": (
                "linear-interpolation quantile of the pot_odds observed on "
                "the model's own training rows (the fitted support), applied "
                "as an upper clamp on the price input before the logit; "
                "derived per model, never hand-authored"
            ),
            "side": "upper only — below-support prices are not clamped",
            "quantile": price_quantile,
            "quantile_ablatable": True,
            "bet_to_pot_note": (
                "pot_odds is the only fitted price regressor; bet_to_pot = "
                "pot_odds / (1 - pot_odds) is capped by implication"
            ),
            "full_data_refit_note": (
                "caps attach to the shipped train-split models; the recorded "
                "full_data_refit blocks remain unclamped fit records"
            ),
            "per_model": price_support,
        },
        "street_plan": plan,
        "pooling_audit": pooling_audit,
        "street_to_model": street_to_model,
        "fallback_model": "all",
        "models": {key: models[key].to_document() for key in models},
        "reports": reports,
        "band": band,
        "invariants": {
            "strength_coefficient_negative": True,
            "price_coefficient_positive": True,
            "checked_on": ["train-split models", "full-data refits"],
        },
    }
    return document


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_fit(document: Mapping[str, Any]) -> None:
    plan = document["street_plan"]
    print("\nstreet plan")
    for street in STREETS:
        entry = plan["per_street_train"][street]
        print(
            f"  {street:<8} train_rows {entry['train_rows']:>6}  "
            f"folds {entry['folds']:>6}  minority {entry['minority']:>6}  "
            f"-> model {document['street_to_model'][street]!r}"
        )
    print("\nmodels")
    for key, report in document["reports"].items():
        train = report["metrics"]["train"]
        validation = report["metrics"]["val"]
        print(f"\n  [{key}] streets={report['streets']}")
        print("    coefficients (standardised):")
        for name, value in report["coefficients_by_name"].items():
            print(f"      {name:<20} {value:+.4f}")
        print(f"    intercept {report['intercept']:+.4f}")
        print(
            f"    train  rows {train['rows']:>6} base {train['base_rate']:.4f} "
            f"logloss {train['log_loss']:.4f} auc "
            f"{train['auc']:.4f} brier {train['brier']:.4f}"
        )
        if validation["rows"]:
            auc_text = (
                f"{validation['auc']:.4f}"
                if validation["auc"] is not None
                else "n/a"
            )
            print(
                f"    val    rows {validation['rows']:>6} base "
                f"{validation['base_rate']:.4f} logloss "
                f"{validation['log_loss']:.4f} auc {auc_text} brier "
                f"{validation['brier']:.4f}"
            )
        print("    held-out calibration (decile: n, predicted -> observed)")
        for entry in report["calibration_val"]:
            print(
                f"      {entry['decile']:>2}  n={entry['rows']:>5}  "
                f"{entry['mean_predicted']:.4f} -> {entry['observed']:.4f}"
            )
    if document.get("pooling_audit"):
        print("\npooling audit (held-out rows of the pooled streets)")
        for key, entry in document["pooling_audit"].items():
            print(f"  {key}: val rows {entry['val_rows']} (assigned "
                  f"{entry['assigned']!r})")
            for name, value in entry.items():
                if not isinstance(value, dict) or "log_loss" not in value:
                    continue
                auc_text = (
                    f"{value['auc']:.4f}" if value["auc"] is not None else "n/a"
                )
                print(
                    f"    {name:<10} logloss {value['log_loss']:.4f} "
                    f"auc {auc_text} brier {value['brier']:.4f}"
                )
    band = document["band"]
    print(f"\nclamp band [{band['low']:.4f}, {band['high']:.4f}] ({band['method']})")
    support = document.get("price_support")
    if support:
        print(
            f"price-support clamp (quantile {support['quantile']}, upper only): "
            + ", ".join(
                f"{key}={entry['price_cap']:.4f}"
                for key, entry in support["per_model"].items()
            )
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the P3 dataset and fit the strength-aware opponent."
    )
    parser.add_argument(
        "--roots", nargs="+", default=[str(root) for root in DEFAULT_ROOTS]
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    parser.add_argument(
        "--price-quantile",
        type=float,
        default=PRICE_SUPPORT_QUANTILE,
        help="fitted-support quantile for the per-model price_cap (ablatable)",
    )
    parser.add_argument(
        "--exclude-agents",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_AGENTS),
        help="agent names to leave out of the field model",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="reuse the dataset already on disk instead of rebuilding it",
    )
    parser.add_argument("--selftest-only", action="store_true")
    parser.add_argument("--skip-selftest", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("workers must be >= 1")

    if not args.skip_selftest:
        selftest()
    if args.selftest_only:
        return 0

    output_dir = Path(args.output_dir)
    dataset_path = output_dir / DATASET_NAME
    if args.fit_only:
        rows = load_dataset(dataset_path)
        print(f"loaded {len(rows)} rows from {dataset_path}")
    else:
        rows, _ = build_dataset(
            [Path(root) for root in args.roots],
            dataset_path,
            excluded_agents=tuple(args.exclude_agents),
            workers=args.workers,
            limit=args.limit,
        )

    document = fit_from_rows(rows, l2=args.l2, price_quantile=args.price_quantile)
    document["dataset"] = {
        "path": str(dataset_path),
        "rows": len(rows),
        "folds": sum(row["label_fold"] for row in rows),
        "per_street": _street_counts(rows),
    }

    fit_path = output_dir / FIT_NAME
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = fit_path.with_suffix(fit_path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(fit_path)

    # Round-trip through the loader the agent uses: the artifact is only
    # shipped if the serve path can read it and it passes the sign invariant
    # again on the way in.
    reloaded = P3Fit.load(fit_path)
    reloaded.check_sign_invariant()

    _print_fit(document)
    print(f"\nwrote {fit_path}")
    print(
        f"reloaded {len(reloaded.models)} models; sign invariant holds on all "
        "of them"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
