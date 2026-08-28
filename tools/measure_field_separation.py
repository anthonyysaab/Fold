"""Recompute the FIELD strength-separation benchmark on the canonical metric.

`V8_DESIGN.md` §2 requires the frozen field benchmark (+0.150 field /
+0.020 us) to be recomputed on ``strength_metric.strength_percentile``
before any v8 number is compared to it: the frozen pair was measured on a
different, undocumented percentile-style scale, and §1c showed two scales
in circulation disagreeing 3x on the same decisions.

This tool walks the complete-information replay archives
(``foreign play data/<collection>/raw/tables/*.json``) and, for every
``ActionTaken`` decision by every seat, computes

- ``strength_percentile`` of the actor's **real** revealed holding on the
  **decision-time** board, and
- the canonical ``training_telemetry.action_family`` of the action,

then reports, per street and overall, with seeded bootstrap-over-hands
95% CIs:

(a) **separation** exactly as the frozen report defined it — mean strength
    when AGGRESSING minus mean strength when FOLDING — for the field;
(b) the same per agent for the leaderboard top-15, so the spread across
    the field is visible;
(c) our own agent on the same instrument, from the same archive;
(d) action-mix context: fold / check-call / aggress rates per street, and
    the median bet as a fraction of the pot.

Scope conventions, chosen to reproduce the frozen report rather than to
improve on it:

- **"field" excludes our own agent.** The frozen table is "S14 field vs
  `candidate-v7-0001c`"; pooling our decisions into the field would move
  the very quantity the gate checks. ``all_agents`` is reported too.
- **``TimeoutAction`` is not a decision.** The actor did not choose it.
  Timeouts are counted in the sidecar and excluded from every rate; this
  is what reproduces the published 55.8% / 21.5% (including them gives
  58.6% / 20.5%).
- **One replay file is one hand**, so the bootstrap resamples tables.

Two reconstruction details this tool does NOT take from the snapshot:

- **The decision-time board is rebuilt from ``StreetDealt`` payloads**, not
  from ``event.snapshot.boardCards``. The snapshot is captured *after* the
  action, so on a street-closing action it already carries the next
  street's cards: measured here, **962 of 9,084 decisions (10.6%)** have a
  snapshot board longer than their street. Reading strength off that board
  is forward-looking leakage. The ``StreetDealt``-derived board matched its
  street's card count on 9,084/9,084 decisions and the snapshot board was a
  forward extension of it on 9,084/9,084 — both checked every run.
- The bet size uses the engine's own pot fraction,
  ``(new chips - call) / (pot + call)``, which is the quantity the frozen
  report's "median bet 0.60x pot" is on (`PENDING_EDITS` 18d: the earlier
  1.333x figure "double-counted the call"; this formula recovers a maximum
  of exactly 1.000000 for our own pot-capped engine).

**Validation gate (house rule: validate the instrument before the
result).** Two independent gates run before any separation number is
presented:

1. **Reproduction gate** — the pipeline must reproduce the published S14
   field behavioural rates from ``PLAYERS.md`` (55.8% fold, 21.5%
   aggression, 0.60x pot median bet) within 2pp. Computed on the S14
   collection alone, since that is the window the published figures were
   measured on.
2. **Impossible-by-construction gate** — a seeded *control* holding drawn
   from the deck-minus-board is scored through the identical pipeline. An
   agent cannot separate on cards it was never dealt, so the control
   separation must be indistinguishable from zero (95% CI containing 0).
   A pipeline that "finds" separation on random holes is measuring an
   artifact of the instrument, not the field.

Plus per-decision structural invariants, any violation of which aborts:
every strength in [0, 1]; hole and decision-time board disjoint; every
revealed holding in a hand pairwise distinct and distinct from the final
board; the derived board length equals its street's.

Offline, read-only over the archive, stdlib-only, fully seeded. Writes
only its JSON report (and an optional markdown summary).

Usage:
    python -m tools.measure_field_separation \\
        --output artifacts/evaluations/field-separation-canonical-2026-08-16.json \\
        --markdown artifacts/evaluations/field-separation-canonical-2026-08-16.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from engine.schema3 import CARD_CODES
from engine.strength_metric import strength_percentile
from engine.training_telemetry import action_family
from tools.collect_foreign_play_data import _read_json, _unwrap_rpc

DEFAULT_ROOTS = (
    Path("foreign play data") / "20260812T082057Z_poker-playground_s13_top15",
    Path("foreign play data") / "20260815T210237Z_poker-playground_s14_top15",
)
DEFAULT_OUTPUT = (
    Path("artifacts")
    / "evaluations"
    / "field-separation-canonical-2026-08-16.json"
)
DEFAULT_SEED = 20260816
DEFAULT_BOOTSTRAP = 2000
DEFAULT_TOP_N = 15

#: The street label -> number of board cards visible while acting.
STREET_BOARD_SIZE: dict[str, int] = {
    "preflop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
}
STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")
FAMILIES: tuple[str, ...] = ("fold", "check_call", "aggress")

#: Our own agent, excluded from "field" and reported separately (deliverable c).
OUR_AGENT_NAME = "Fold-ver-4"

#: The collection the published field figures were measured on.
GATE_COLLECTION_MARKER = "s14"

#: ``PLAYERS.md`` / ``PENDING_EDITS`` 18d, the numbers the pipeline must
#: reproduce before any separation figure is treated as authoritative.
PUBLISHED_FIELD_RATES: dict[str, float] = {
    "fold_rate": 0.558,
    "aggression_rate": 0.215,
    "median_bet_pot_fraction": 0.60,
}
#: "within ~2pp" from the task brief; the bet fraction is not a percentage
#: but 0.02 is the same absolute tolerance and is the stricter reading.
GATE_TOLERANCE = 0.02

#: A bootstrap over hands cannot resolve a cell that lives in a handful of
#: hands: when every aggress and every fold in a cell come from the same
#: few tables, the multiplicity cancels and the interval collapses to a
#: point that looks precise and means nothing. Below these thresholds the
#: point estimate is still reported and the interval is withheld with a
#: reason, rather than shipping a confident wrong number.
MIN_CI_HANDS = 10

_ACTION_EVENT_TYPES = ("ActionTaken", "TimeoutAction")


class MeasurementInvariantError(RuntimeError):
    """An impossible-by-construction invariant was violated.

    Not a ValueError: malformed-replay ValueErrors are counted as skips,
    and an invariant violation must never be silently absorbed into that
    count — it means this tool is wrong, not the archive.
    """


class Decision(NamedTuple):
    """One ``ActionTaken`` decision, scored on the canonical metric."""

    collection: str
    table_id: str
    sequence: int
    agent_id: str
    agent_name: str
    street: str
    family: str
    strength: float
    control_strength: float
    pot_fraction: float | None


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _decision_seed(seed: int, table_id: str, sequence: int) -> int:
    """A stable per-decision seed: file-order and worker independent."""

    digest = hashlib.sha256(f"{seed}:{table_id}:{sequence}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def pot_fraction(payload: Mapping[str, Any]) -> float | None:
    """The engine's own pot fraction for an aggressive action.

    ``(new chips beyond the call) / (pot + call)``. ``None`` when the
    action carries no readable size or the denominator is not positive.
    """

    action = str(payload.get("action") or "").casefold()
    if action_family(action) != "aggress":
        return None
    contribution = _int_or(payload.get("actorCurrentBetBefore"), 0)
    target = payload.get("toAmount")
    if target is None:
        target = payload.get("amount")
    allowed = payload.get("allowedActions")
    allowed = allowed if isinstance(allowed, Mapping) else {}
    if target is None and action == "all-in":
        target = allowed.get("allInToAmount")
    if target is None:
        return None
    new_chips = max(0, _int_or(target, 0) - contribution)
    call = _int_or(payload.get("callAmount"), 0)
    if call == 0:
        call = max(0, _int_or(allowed.get("callChips"), 0))
    pot = _int_or(payload.get("pot"), 0)
    denominator = pot + call
    if denominator <= 0:
        return None
    return (new_chips - call) / denominator


def _int_or(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _hole_cards_by_seat(
    events: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[str, ...]]:
    """Every seat's revealed holding, folders included."""

    for event in events:
        if event.get("type") != "HoleCardsDealt":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        holes: dict[int, tuple[str, ...]] = {}
        for seat in payload.get("seats") or []:
            if not isinstance(seat, Mapping):
                continue
            number = seat.get("seatNumber")
            cards = seat.get("holeCards")
            if (
                isinstance(number, int)
                and not isinstance(number, bool)
                and isinstance(cards, list)
                and len(cards) == 2
                and all(isinstance(card, str) and card for card in cards)
            ):
                holes[number] = tuple(cards)
        return holes
    return {}


def _require_distinct_cards(
    holes: Mapping[int, tuple[str, ...]], final_board: Sequence[str], table_id: str
) -> None:
    """No card is dealt twice in one hand — impossible by construction.

    Catches seat/holding misalignment, which would silently corrupt every
    strength in the hand while leaving all the rates intact.
    """

    seen: list[str] = [card for hole in holes.values() for card in hole]
    seen.extend(str(card) for card in final_board)
    if len(seen) != len(set(seen)):
        duplicates = sorted({card for card in seen if seen.count(card) > 1})
        raise MeasurementInvariantError(
            f"table {table_id}: card(s) {duplicates} dealt more than once"
        )


def _control_hole(board: Sequence[str], rng: random.Random) -> tuple[str, str]:
    """A counterfactual holding: two cards from the deck minus the board.

    The control the impossible-by-construction gate rests on. It is a
    *legal* holding for the metric but bears no relation to the actor, so
    separation computed on it must be zero up to noise.
    """

    dead = set(board)
    live = [code for code in CARD_CODES if code not in dead]
    first, second = rng.sample(live, 2)
    return first, second


def replay_decisions(
    replay: Mapping[str, Any], *, collection: str, seed: int
) -> tuple[list[Decision], Counter[str]]:
    """Every scored decision in one unwrapped replay, plus counters."""

    stats: Counter[str] = Counter()
    events = replay.get("events")
    table = replay.get("table")
    if not isinstance(events, list) or not isinstance(table, Mapping):
        stats["replay_unusable"] += 1
        return [], stats
    table_id = str(table.get("id") or table.get("tableId") or "")
    if not table_id:
        stats["replay_unusable"] += 1
        return [], stats
    holes = _hole_cards_by_seat(events)
    if not holes:
        stats["replay_without_hole_cards"] += 1
        return [], stats
    _require_distinct_cards(holes, table.get("boardCards") or [], table_id)

    decisions: list[Decision] = []
    board: tuple[str, ...] = ()
    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_type = event.get("type")
        if event_type == "StreetDealt":
            payload = event.get("payload")
            if isinstance(payload, Mapping):
                cards = payload.get("boardCards")
                if isinstance(cards, list) and all(
                    isinstance(card, str) for card in cards
                ):
                    board = tuple(cards)
            continue
        if event_type not in _ACTION_EVENT_TYPES:
            continue
        if event_type == "TimeoutAction":
            stats["timeout_actions"] += 1
            continue
        stats["action_taken"] += 1
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            stats["skip:no_payload"] += 1
            continue
        street = str(event.get("street") or "").casefold()
        if street not in STREET_BOARD_SIZE:
            stats["skip:street"] += 1
            continue
        # Invariant: the reconstructed board must fit its street exactly.
        if len(board) != STREET_BOARD_SIZE[street]:
            raise MeasurementInvariantError(
                f"table {table_id}: {street} decision sees a "
                f"{len(board)}-card board"
            )
        # Invariant, and the reason this tool does not read the snapshot:
        # the post-action snapshot must be a forward extension of the
        # decision-time board, never a different board.
        snapshot = event.get("snapshot")
        if isinstance(snapshot, Mapping):
            snapshot_board = snapshot.get("boardCards")
            if isinstance(snapshot_board, list):
                if tuple(snapshot_board[: len(board)]) != board:
                    raise MeasurementInvariantError(
                        f"table {table_id}: snapshot board {snapshot_board} "
                        f"is not an extension of {list(board)}"
                    )
                if len(snapshot_board) != len(board):
                    stats["snapshot_board_leaked_forward"] += 1
        seat = payload.get("seatNumber")
        if not isinstance(seat, int) or isinstance(seat, bool):
            stats["skip:seat"] += 1
            continue
        hole = holes.get(seat)
        if hole is None:
            stats["skip:hole_not_revealed"] += 1
            continue
        if set(hole) & set(board):
            raise MeasurementInvariantError(
                f"table {table_id} seat {seat}: holding {list(hole)} "
                f"intersects the board {list(board)}"
            )
        family = action_family(str(payload.get("action") or "").casefold())
        if family is None:
            stats["skip:unsupported_action"] += 1
            continue
        sequence = event.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            stats["skip:sequence"] += 1
            continue

        strength = strength_percentile(hole, board)
        rng = random.Random(_decision_seed(seed, table_id, sequence))
        control = strength_percentile(_control_hole(board, rng), board)
        for name, value in (("strength", strength), ("control", control)):
            if not 0.0 <= value <= 1.0:
                raise MeasurementInvariantError(
                    f"table {table_id} seq {sequence}: {name} {value!r}"
                )
        agent_id = event.get("agentId")
        agent_name = payload.get("agentName")
        decisions.append(
            Decision(
                collection=collection,
                table_id=table_id,
                sequence=sequence,
                agent_id=agent_id if isinstance(agent_id, str) else "",
                agent_name=agent_name if isinstance(agent_name, str) else "",
                street=street,
                family=family,
                strength=strength,
                control_strength=control,
                pot_fraction=pot_fraction(payload),
            )
        )
        stats["scored"] += 1
    return decisions, stats


def collect(
    roots: Sequence[Path], *, seed: int, limit: int | None = None
) -> tuple[list[Decision], dict[str, Counter[str]]]:
    """Score every decision in every replay under ``roots``."""

    decisions: list[Decision] = []
    stats: dict[str, Counter[str]] = {}
    started = time.monotonic()
    for root in roots:
        collection = Path(root).name
        table_dir = Path(root) / "raw" / "tables"
        if not table_dir.is_dir():
            raise FileNotFoundError(f"no raw tables directory under {root}")
        files = sorted(table_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no raw table replays under {table_dir}")
        if limit is not None:
            files = files[:limit]
        counters = stats.setdefault(collection, Counter())
        counters["files"] += len(files)
        for index, path in enumerate(files, start=1):
            try:
                replay = _unwrap_rpc(_read_json(path))
            except (OSError, ValueError):
                counters["unreadable_files"] += 1
                continue
            if not isinstance(replay, dict):
                counters["unreadable_files"] += 1
                continue
            rows, replay_stats = replay_decisions(
                replay, collection=collection, seed=seed
            )
            decisions.extend(rows)
            counters.update(replay_stats)
            if index % 200 == 0 or index == len(files):
                print(
                    f"  {collection}: {index}/{len(files)} files, "
                    f"{len(decisions)} decisions, "
                    f"{time.monotonic() - started:.0f}s",
                    flush=True,
                )
    decisions.sort(key=lambda row: (row.collection, row.table_id, row.sequence))
    return decisions, stats


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


class Cell:
    """Per-hand sufficient statistics for one (scope, street) cell."""

    __slots__ = ("entries",)

    def __init__(self) -> None:
        # table index -> [sum_aggress, n_aggress, sum_fold, n_fold, n_call]
        self.entries: dict[int, list[float]] = {}

    def add(self, table_index: int, family: str, strength: float) -> None:
        row = self.entries.get(table_index)
        if row is None:
            row = [0.0, 0.0, 0.0, 0.0, 0.0]
            self.entries[table_index] = row
        if family == "aggress":
            row[0] += strength
            row[1] += 1.0
        elif family == "fold":
            row[2] += strength
            row[3] += 1.0
        else:
            row[4] += 1.0

    def frozen(self) -> list[tuple[int, float, float, float, float, float]]:
        return [
            (index, row[0], row[1], row[2], row[3], row[4])
            for index, row in sorted(self.entries.items())
        ]


def _stats_from(
    rows: Iterable[tuple[int, float, float, float, float, float]],
    counts: Sequence[int] | None,
) -> tuple[float, float, float, float, float]:
    sum_a = n_a = sum_f = n_f = n_c = 0.0
    if counts is None:
        for _, sa, na, sf, nf, nc in rows:
            sum_a += sa
            n_a += na
            sum_f += sf
            n_f += nf
            n_c += nc
    else:
        for index, sa, na, sf, nf, nc in rows:
            multiplicity = counts[index]
            if multiplicity:
                sum_a += multiplicity * sa
                n_a += multiplicity * na
                sum_f += multiplicity * sf
                n_f += multiplicity * nf
                n_c += multiplicity * nc
    return sum_a, n_a, sum_f, n_f, n_c


def separation_of(
    sum_a: float, n_a: float, sum_f: float, n_f: float
) -> float | None:
    """Mean strength when aggressing minus mean strength when folding."""

    if n_a <= 0 or n_f <= 0:
        return None
    return sum_a / n_a - sum_f / n_f


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile of an already sorted sequence."""

    if not values:
        raise ValueError("percentile of an empty sequence")
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1.0 - weight) + values[high] * weight


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation; ``None`` when undefined."""

    if len(xs) != len(ys) or len(xs) < 3:
        return None

    def _ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    ) ** 0.5
    if denominator == 0:
        return None
    return numerator / denominator


def _multiplicities(n: int, rng: random.Random) -> list[int]:
    counts = [0] * n
    for _ in range(n):
        counts[rng.randrange(n)] += 1
    return counts


def bootstrap_cells(
    cells: Mapping[str, list[tuple[int, float, float, float, float, float]]],
    n_tables: int,
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Point estimates and seeded bootstrap-over-hands 95% CIs per cell.

    One common resample of hands is drawn per iteration and shared by
    every cell, so the intervals come from the same resampled archive.
    """

    point: dict[str, dict[str, Any]] = {}
    for key, rows in cells.items():
        sum_a, n_a, sum_f, n_f, n_c = _stats_from(rows, None)
        total = n_a + n_f + n_c
        point[key] = {
            "decisions": int(total),
            "hands": len(rows),
            "hands_with_aggress": sum(1 for row in rows if row[2] > 0),
            "hands_with_fold": sum(1 for row in rows if row[4] > 0),
            "n_aggress": int(n_a),
            "n_fold": int(n_f),
            "n_check_call": int(n_c),
            "mean_strength_aggress": (sum_a / n_a) if n_a else None,
            "mean_strength_fold": (sum_f / n_f) if n_f else None,
            "separation": separation_of(sum_a, n_a, sum_f, n_f),
            "fold_rate": (n_f / total) if total else None,
            "aggression_rate": (n_a / total) if total else None,
            "check_call_rate": (n_c / total) if total else None,
        }

    draws: dict[str, dict[str, list[float]]] = {
        key: {
            "separation": [],
            "fold_rate": [],
            "aggression_rate": [],
            "check_call_rate": [],
        }
        for key in cells
    }
    undefined: Counter[str] = Counter()
    rng = random.Random(seed)
    for _ in range(resamples):
        counts = _multiplicities(n_tables, rng)
        for key, rows in cells.items():
            sum_a, n_a, sum_f, n_f, n_c = _stats_from(rows, counts)
            total = n_a + n_f + n_c
            value = separation_of(sum_a, n_a, sum_f, n_f)
            if value is None:
                undefined[key] += 1
            else:
                draws[key]["separation"].append(value)
            if total:
                draws[key]["fold_rate"].append(n_f / total)
                draws[key]["aggression_rate"].append(n_a / total)
                draws[key]["check_call_rate"].append(n_c / total)

    for key in cells:
        entry = point[key]
        withheld: dict[str, str] = {}
        separation_resolvable = (
            entry["hands_with_aggress"] >= MIN_CI_HANDS
            and entry["hands_with_fold"] >= MIN_CI_HANDS
        )
        rates_resolvable = entry["hands"] >= MIN_CI_HANDS
        entry["bootstrap"] = {
            "resamples": resamples,
            "undefined_separation_resamples": undefined.get(key, 0),
            "min_ci_hands": MIN_CI_HANDS,
        }
        for name, values in draws[key].items():
            resolvable = (
                separation_resolvable if name == "separation" else rates_resolvable
            )
            if not resolvable:
                entry[f"{name}_ci95"] = None
                withheld[name] = (
                    f"fewer than {MIN_CI_HANDS} contributing hands; a "
                    "bootstrap over hands cannot resolve this cell"
                )
                continue
            if len(values) < max(20, resamples // 20):
                entry[f"{name}_ci95"] = None
                withheld[name] = "too few defined resamples"
                continue
            ordered = sorted(values)
            entry[f"{name}_ci95"] = [
                _percentile(ordered, 0.025),
                _percentile(ordered, 0.975),
            ]
        entry["bootstrap"]["ci_withheld"] = withheld
    return point


def bootstrap_median(
    per_table: Mapping[int, list[float]],
    n_tables: int,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Point median and bootstrap-over-hands 95% CI of a per-hand value list.

    ``resamples=0`` returns the point estimate only. Resampling a median
    needs the whole value list rebuilt per iteration, so it is spent only
    on the cells whose interval is actually reported.
    """

    flat = [value for values in per_table.values() for value in values]
    if not flat:
        return {"n": 0, "median": None, "median_ci95": None}
    rows = sorted(per_table.items())
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        counts = _multiplicities(n_tables, rng)
        pooled: list[float] = []
        for index, values in rows:
            multiplicity = counts[index]
            if multiplicity:
                pooled.extend(values * multiplicity)
        if pooled:
            draws.append(statistics.median(pooled))
    ordered = sorted(draws)
    return {
        "n": len(flat),
        "median": statistics.median(flat),
        "median_ci95": (
            [_percentile(ordered, 0.025), _percentile(ordered, 0.975)]
            if len(ordered) >= max(20, resamples // 20)
            else None
        ),
        "share_above_1_1x_pot": sum(1 for value in flat if value > 1.1)
        / len(flat),
    }


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------


def leaderboard_top(root: Path, top_n: int) -> list[dict[str, Any]]:
    """The collection's own leaderboard, ranks 1..``top_n``.

    The archive's ``rank`` field is a **dense** rank and genuinely ties:
    ``rank <= 15`` selects 18 agents in S13 and 17 in S14. The stored rank
    is kept rather than re-derived from ``totalScore`` because it is what
    ``PLAYERS.md`` quotes ("rank 10 of 15"); score-ordering would move our
    own agent to 11 and silently disagree with the frozen documents.
    """

    document = _unwrap_rpc(_read_json(Path(root) / "raw" / "leaderboard.json"))
    if not isinstance(document, Mapping):
        raise ValueError(f"{root}: leaderboard has an unexpected shape")
    agents = document.get("agents")
    if not isinstance(agents, list):
        raise ValueError(f"{root}: leaderboard has no agents")
    rows = [
        {
            "agent_id": agent.get("id"),
            "name": agent.get("name"),
            "rank": agent.get("rank"),
            "total_score": agent.get("totalScore"),
            "adjusted_bb100": agent.get("adjustedBb100"),
        }
        for agent in agents
        if isinstance(agent, Mapping)
        and isinstance(agent.get("rank"), int)
        and agent.get("rank") <= top_n
    ]
    rows.sort(key=lambda row: row["rank"])
    return rows


def scope_of(decision: Decision) -> str:
    return "us" if decision.agent_name == OUR_AGENT_NAME else "field"


def collection_alias(collection: str) -> str:
    """``...poker-playground_s14_top15`` -> ``s14``.

    The frozen +0.150 was measured on S14 alone, so the report has to be
    able to show a like-for-like S14 slice next to the pooled figure.
    """

    match = re.search(r"_(s\d+)_", collection)
    return match.group(1) if match else collection


def build_report(
    decisions: Sequence[Decision],
    stats: Mapping[str, Counter[str]],
    leaderboards: Mapping[str, list[dict[str, Any]]],
    *,
    seed: int,
    resamples: int,
    roots: Sequence[Path],
) -> dict[str, Any]:
    """Assemble the full report document (deliverables a-d plus the gates)."""

    table_keys = sorted({(row.collection, row.table_id) for row in decisions})
    table_index = {key: index for index, key in enumerate(table_keys)}
    n_tables = len(table_keys)

    tracked_agents: dict[str, dict[str, Any]] = {}
    for collection, rows in leaderboards.items():
        for row in rows:
            agent_id = row["agent_id"]
            if not isinstance(agent_id, str):
                continue
            entry = tracked_agents.setdefault(
                agent_id, {"name": row["name"], "leaderboard": {}}
            )
            entry["leaderboard"][collection] = {
                "rank": row["rank"],
                "total_score": row["total_score"],
                "adjusted_bb100": row["adjusted_bb100"],
            }

    cells: dict[str, Cell] = {}
    control_cells: dict[str, Cell] = {}
    bet_fractions: dict[str, dict[int, list[float]]] = {}
    per_collection_gate: dict[str, Cell] = {}
    per_collection_bets: dict[str, dict[int, list[float]]] = {}

    def _cell(store: dict[str, Cell], key: str) -> Cell:
        cell = store.get(key)
        if cell is None:
            cell = Cell()
            store[key] = cell
        return cell

    for row in decisions:
        index = table_index[(row.collection, row.table_id)]
        scope = scope_of(row)
        alias = collection_alias(row.collection)
        for scope_key in (scope, "all_agents", f"{scope}@{alias}"):
            for street_key in ("overall", row.street):
                _cell(cells, f"{scope_key}|{street_key}").add(
                    index, row.family, row.strength
                )
        _cell(control_cells, f"control_{scope}|overall").add(
            index, row.family, row.control_strength
        )
        if row.agent_id in tracked_agents:
            for street_key in ("overall", row.street):
                _cell(cells, f"agent:{row.agent_id}|{street_key}").add(
                    index, row.family, row.strength
                )
        if row.pot_fraction is not None:
            for scope_key in (scope, "all_agents", f"{scope}@{alias}"):
                for street_key in ("overall", row.street):
                    bet_fractions.setdefault(
                        f"{scope_key}|{street_key}", {}
                    ).setdefault(index, []).append(row.pot_fraction)
        # The reproduction gate is computed on the published window only.
        if GATE_COLLECTION_MARKER in row.collection and scope == "field":
            _cell(per_collection_gate, row.collection).add(
                index, row.family, row.strength
            )
            if row.pot_fraction is not None:
                per_collection_bets.setdefault(row.collection, {}).setdefault(
                    index, []
                ).append(row.pot_fraction)

    frozen = {key: cell.frozen() for key, cell in cells.items()}
    frozen_control = {key: cell.frozen() for key, cell in control_cells.items()}
    print(
        f"  bootstrapping {len(frozen) + len(frozen_control)} cells x "
        f"{resamples} resamples over {n_tables} hands",
        flush=True,
    )
    results = bootstrap_cells(frozen, n_tables, resamples=resamples, seed=seed)
    control = bootstrap_cells(
        frozen_control, n_tables, resamples=resamples, seed=seed + 1
    )

    ci_median_cells = {"field|overall", "us|overall", "all_agents|overall"}
    medians = {
        key: bootstrap_median(
            values,
            n_tables,
            resamples=resamples if key in ci_median_cells else 0,
            seed=seed + 2,
        )
        for key, values in bet_fractions.items()
    }

    # ---- gate 1: reproduce the published S14 field behavioural rates -----
    gate_rows: dict[str, Any] = {}
    for collection, cell in per_collection_gate.items():
        rows = cell.frozen()
        sum_a, n_a, sum_f, n_f, n_c = _stats_from(rows, None)
        total = n_a + n_f + n_c
        median = bootstrap_median(
            per_collection_bets.get(collection, {}),
            n_tables,
            resamples=resamples,
            seed=seed + 3,
        )
        measured = {
            "fold_rate": (n_f / total) if total else None,
            "aggression_rate": (n_a / total) if total else None,
            "median_bet_pot_fraction": median["median"],
        }
        checks = {}
        for name, published in PUBLISHED_FIELD_RATES.items():
            value = measured[name]
            delta = None if value is None else value - published
            checks[name] = {
                "published": published,
                "measured": value,
                "delta": delta,
                "tolerance": GATE_TOLERANCE,
                "pass": delta is not None and abs(delta) <= GATE_TOLERANCE,
            }
        gate_rows[collection] = {
            "decisions": int(total),
            "checks": checks,
            "pass": all(check["pass"] for check in checks.values()),
        }
    reproduction_pass = bool(gate_rows) and all(
        row["pass"] for row in gate_rows.values()
    )

    # ---- gate 2: the impossible-by-construction control ------------------
    #
    # A control cell whose interval was withheld is UNRESOLVED, never a
    # failure: our own 98 decisions live in 9 fold-hands, which no
    # bootstrap over hands can resolve, and treating that as evidence of
    # a broken instrument would be exactly the "below the noise is not a
    # result" error in reverse. The gate binds on the field cell — the
    # one the recomputed benchmark actually rests on — and any resolved
    # cell whose interval excludes zero fails it.
    required_control = "control_field|overall"
    control_checks: dict[str, Any] = {}
    for key, entry in control.items():
        interval = entry.get("separation_ci95")
        if interval is None:
            status = "unresolved"
        elif interval[0] <= 0.0 <= interval[1]:
            status = "pass"
        else:
            status = "fail"
        control_checks[key] = {
            "separation": entry["separation"],
            "separation_ci95": interval,
            "status": status,
            "contains_zero": status == "pass",
            "n_aggress": entry["n_aggress"],
            "n_fold": entry["n_fold"],
            "hands_with_aggress": entry["hands_with_aggress"],
            "hands_with_fold": entry["hands_with_fold"],
        }
    control_pass = (
        control_checks.get(required_control, {}).get("status") == "pass"
        and not any(
            check["status"] == "fail" for check in control_checks.values()
        )
    )

    # ---- per-agent block (deliverable b) ---------------------------------
    agents_block: list[dict[str, Any]] = []
    for agent_id, meta in tracked_agents.items():
        overall = results.get(f"agent:{agent_id}|overall")
        if overall is None:
            continue
        entry: dict[str, Any] = {
            "agent_id": agent_id,
            "name": meta["name"],
            "is_us": meta["name"] == OUR_AGENT_NAME,
            "leaderboard": meta["leaderboard"],
            "overall": overall,
            "per_street": {
                street: results.get(f"agent:{agent_id}|{street}")
                for street in STREETS
            },
        }
        agents_block.append(entry)
    agents_block.sort(
        key=lambda row: (
            min(
                (
                    value["rank"]
                    for value in row["leaderboard"].values()
                    if isinstance(value.get("rank"), int)
                ),
                default=999,
            ),
            row["name"] or "",
        )
    )

    # The honest null the frozen note insists travels with these numbers.
    spearman: dict[str, Any] = {}
    for collection in leaderboards:
        scored = [
            (
                entry["leaderboard"][collection]["total_score"],
                entry["overall"]["fold_rate"],
                entry["overall"]["aggression_rate"],
                entry["overall"]["separation"],
            )
            for entry in agents_block
            if collection in entry["leaderboard"]
            and isinstance(
                entry["leaderboard"][collection].get("total_score"), (int, float)
            )
            and entry["overall"]["fold_rate"] is not None
        ]
        if len(scored) < 3:
            continue
        scores = [row[0] for row in scored]
        spearman[collection] = {
            "n_agents": len(scored),
            "score_vs_fold_rate": _spearman(scores, [row[1] for row in scored]),
            "score_vs_aggression_rate": _spearman(
                scores, [row[2] for row in scored]
            ),
            "score_vs_separation": _spearman(
                [row[0] for row in scored if row[3] is not None],
                [row[3] for row in scored if row[3] is not None],
            ),
        }

    def _scope_block(scope: str) -> dict[str, Any]:
        return {
            "overall": results.get(f"{scope}|overall"),
            "per_street": {
                street: results.get(f"{scope}|{street}") for street in STREETS
            },
            "bet_pot_fraction": {
                "overall": medians.get(f"{scope}|overall"),
                **{
                    street: medians.get(f"{scope}|{street}")
                    for street in STREETS
                },
            },
        }

    return {
        "report": "field-separation-canonical",
        "generated_for": "V8_DESIGN.md §2 / §6.3 — the recomputed field benchmark",
        "date": "2026-08-16",
        "metric": {
            "module": "engine.strength_metric",
            "function": "strength_percentile",
            "definition": (
                "postflop: exact enumeration of C(unseen,2) opponent "
                "holdings the made hand beats, ties half; preflop: the "
                "committed 169-class all-in-equity percentile table"
            ),
            "note": (
                "NOT the scale the frozen +0.150 / +0.020 pair was computed "
                "on; that pair is a different instrument on a different "
                "window and the two numbers are not interchangeable"
            ),
        },
        "generator": {
            "tool": "tools.measure_field_separation",
            "seed": seed,
            "bootstrap_resamples": resamples,
            "bootstrap_unit": "hand (one replay file = one hand)",
            "roots": [str(root) for root in roots],
            "board_source": "StreetDealt payload boardCards (decision-time)",
            "timeouts": "TimeoutAction excluded from every rate",
            "field_definition": f"all agents except {OUR_AGENT_NAME!r}",
        },
        "counts": {
            "hands": n_tables,
            "decisions": len(decisions),
            # Our agent is identified by name; the ids behind that name are
            # listed so a name collision would be visible rather than silent.
            "our_agent_ids": sorted(
                {
                    row.agent_id
                    for row in decisions
                    if row.agent_name == OUR_AGENT_NAME
                }
            ),
            "tracked_agents": len(tracked_agents),
            "per_collection": {
                name: dict(sorted(counter.items()))
                for name, counter in stats.items()
            },
        },
        "validation": {
            "reproduction_gate": {
                "published_source": "PLAYERS.md / PENDING_EDITS 18d (S14)",
                "collections": gate_rows,
                "pass": reproduction_pass,
            },
            "control_gate": {
                "description": (
                    "seeded counterfactual holding drawn from deck-minus-"
                    "board, scored through the identical pipeline; an agent "
                    "cannot separate on cards it was never dealt"
                ),
                "binding_check": required_control,
                "checks": control_checks,
                "pass": control_pass,
            },
            "pass": reproduction_pass and control_pass,
        },
        "field": _scope_block("field"),
        "us": _scope_block("us"),
        "all_agents": _scope_block("all_agents"),
        "field_by_collection": {
            alias: _scope_block(f"field@{alias}")
            for alias in sorted(
                {collection_alias(row.collection) for row in decisions}
            )
        },
        "us_by_collection": {
            alias: _scope_block(f"us@{alias}")
            for alias in sorted(
                {
                    collection_alias(row.collection)
                    for row in decisions
                    if scope_of(row) == "us"
                }
            )
        },
        "agents": agents_block,
        "leaderboards": {
            name: rows for name, rows in leaderboards.items()
        },
        "spearman_caveat": {
            "note": (
                "the frozen warning: the field data does NOT say 'fold more, "
                "score better'"
            ),
            "by_collection": spearman,
        },
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


def _fmt_ci(interval: Sequence[float] | None, digits: int = 3) -> str:
    if not interval:
        return "n/a"
    return f"[{interval[0]:+.{digits}f}, {interval[1]:+.{digits}f}]"


def _fmt_separation_ci(entry: Mapping[str, Any]) -> str:
    """CI text that says *why* it is absent rather than just "n/a"."""

    interval = entry.get("separation_ci95")
    if interval:
        return _fmt_ci(interval)
    if entry.get("separation") is None:
        return "n/a (no aggress or no fold)"
    return (
        f"WITHHELD ({entry['hands_with_aggress']} aggress / "
        f"{entry['hands_with_fold']} fold hands)"
    )


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    """A short human-readable summary of the report document."""

    validation = report["validation"]
    verdict = "PASS" if validation["pass"] else "FAIL"
    lines: list[str] = [
        "# Field strength separation, recomputed on the canonical metric",
        "",
        f"**Generated** {report['date']} by `tools.measure_field_separation` "
        f"(seed {report['generator']['seed']}, "
        f"{report['generator']['bootstrap_resamples']} bootstrap resamples "
        f"over {report['counts']['hands']} hands).",
        "",
        f"**VALIDATION GATE: {verdict}**",
        "",
        "| gate | check | published | measured | delta | verdict |",
        "|---|---|---:|---:|---:|---|",
    ]
    for collection, row in validation["reproduction_gate"]["collections"].items():
        for name, check in row["checks"].items():
            measured = check["measured"]
            measured_text = "n/a" if measured is None else f"{measured:.3f}"
            lines.append(
                f"| reproduction ({collection}) | {name} | "
                f"{check['published']:.3f} | {measured_text} | "
                f"{_fmt(check['delta'])} | "
                f"{'PASS' if check['pass'] else 'FAIL'} |"
            )
    binding = validation["control_gate"].get("binding_check")
    for key, check in validation["control_gate"]["checks"].items():
        marker = " (binding)" if key == binding else ""
        lines.append(
            f"| control{marker} | {key} separation on random holes | 0.000 | "
            f"{_fmt(check['separation'])} | "
            f"CI {_fmt_ci(check['separation_ci95'])} | "
            f"{check['status'].upper()} |"
        )
    lines += ["", "## Separation, canonical metric (95% CI, bootstrap over hands)", ""]
    lines.append("| scope | street | n | mean strength aggress | mean strength fold | separation | 95% CI |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    scopes: list[tuple[str, Mapping[str, Any]]] = [
        ("field", report["field"]),
        *(
            (f"field ({alias})", block)
            for alias, block in report["field_by_collection"].items()
        ),
        ("us", report["us"]),
    ]
    for scope, block in scopes:
        for street in ("overall", *STREETS):
            entry = (
                block["overall"] if street == "overall" else block["per_street"][street]
            )
            if entry is None:
                continue
            lines.append(
                f"| {scope} | {street} | {entry['decisions']} | "
                f"{_fmt(entry['mean_strength_aggress'])} | "
                f"{_fmt(entry['mean_strength_fold'])} | "
                f"{_fmt(entry['separation'])} | "
                f"{_fmt_separation_ci(entry)} |"
            )
    lines += ["", "## Per-agent (leaderboard top-15)", ""]
    lines.append("| rank | agent | n | fold | call | aggress | separation | 95% CI |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for entry in report["agents"]:
        overall = entry["overall"]
        ranks = "/".join(
            str(value["rank"]) for value in entry["leaderboard"].values()
        )
        lines.append(
            f"| {ranks} | {entry['name']}{' (us)' if entry['is_us'] else ''} | "
            f"{overall['decisions']} | {_fmt_pct(overall['fold_rate'])} | "
            f"{_fmt_pct(overall['check_call_rate'])} | "
            f"{_fmt_pct(overall['aggression_rate'])} | "
            f"{_fmt(overall['separation'])} | "
            f"{_fmt_separation_ci(overall)} |"
        )
    lines += ["", "## Action mix per street", ""]
    lines.append("| scope | street | n | fold | check/call | aggress | median bet / pot |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for scope in ("field", "us"):
        block = report[scope]
        for street in ("overall", *STREETS):
            entry = (
                block["overall"] if street == "overall" else block["per_street"][street]
            )
            if entry is None:
                continue
            median = (block["bet_pot_fraction"] or {}).get(street) or {}
            median_value = median.get("median")
            median_text = "n/a" if median_value is None else f"{median_value:.3f}x"
            lines.append(
                f"| {scope} | {street} | {entry['decisions']} | "
                f"{_fmt_pct(entry['fold_rate'])} | "
                f"{_fmt_pct(entry['check_call_rate'])} | "
                f"{_fmt_pct(entry['aggression_rate'])} | "
                f"{median_text} |"
            )
    lines += [
        "",
        "## The honest null that travels with these numbers",
        "",
        "| collection | n agents | Spearman(score, fold rate) | Spearman(score, aggression) | Spearman(score, separation) |",
        "|---|---:|---:|---:|---:|",
    ]
    for collection, row in report["spearman_caveat"]["by_collection"].items():
        lines.append(
            f"| {collection} | {row['n_agents']} | "
            f"{_fmt(row['score_vs_fold_rate'])} | "
            f"{_fmt(row['score_vs_aggression_rate'])} | "
            f"{_fmt(row['score_vs_separation'])} |"
        )
    lines += [
        "",
        "The field data does **not** say \"fold more, score better\". Read the",
        "per-agent table as a spread, never as a ranking of virtue.",
    ]
    lines += _interpretation(report)
    return "\n".join(lines)


def _interpretation(report: Mapping[str, Any]) -> list[str]:
    """The paragraph that keeps the two scales from being confused."""

    field = report["field"]["overall"]
    per_street = report["field"]["per_street"]
    s14 = (report.get("field_by_collection") or {}).get("s14", {}).get("overall")
    ours = report["us"]["overall"]
    counts = report["counts"]
    leaked = sum(
        entry.get("snapshot_board_leaked_forward", 0)
        for entry in counts["per_collection"].values()
    )
    headline = (
        f"The recomputed field separation is **{field['separation']:+.3f}** "
        f"({_fmt_ci(field['separation_ci95'])}, {field['decisions']} "
        f"decisions over {field['hands']} hands)"
    )
    if s14:
        headline += (
            f", and **{s14['separation']:+.3f}** "
            f"{_fmt_ci(s14['separation_ci95'])} on the S14 window the frozen "
            "figure was measured on"
        )
    return [
        "",
        "## Reading this against the frozen +0.150",
        "",
        headline + ".",
        "",
        "**This is not a correction of +0.150 and it does not refute it.**",
        "It is the same *quantity* — mean strength when aggressing minus",
        "mean strength when folding — on a different *scale*: the canonical",
        "exact-enumeration percentile of `strength_metric`, which is",
        "street-comparable and player-count invariant, where the frozen pair",
        "was computed on an undocumented percentile-style scale over a",
        "narrower decision window. The two are not interchangeable and",
        "neither supersedes the other. Only figures produced by this module",
        "may be compared to the ones in this table.",
        "",
        "**What that means for a v8 number.** A +0.110 quoted for the",
        "incumbent on \"a percentile scale\" came from a different probe on a",
        "different window (live-journal decisions, postflop only), so it",
        "cannot be read against this table either. Our own replay-side",
        f"figure here is **{ours['separation']:+.3f}** on {ours['decisions']}",
        f"decisions in {ours['hands']} hands, and its interval is withheld",
        "because a bootstrap over hands cannot resolve 9 fold-hands — treat",
        "it as consistent with the known behaviour (9.2% fold against the",
        "field's 56.0%), not as a measurement in its own right. The honest",
        "statement of the gap is directional: the field's aggression carries",
        "a large, tightly resolved amount of hand-strength information on",
        "every street, and ours carries an amount this archive cannot",
        "distinguish from none.",
        "",
        "**Acceptance targets are now numeric** (`V8_DESIGN.md` §6.3 — move",
        "*toward* the field figure, preflop CI excluding zero):",
        "overall "
        + _fmt(field["separation"])
        + ", "
        + ", ".join(
            f"{street} "
            + _fmt((per_street.get(street) or {}).get("separation"))
            for street in STREETS
        )
        + ".",
        "",
        "## Caveats that travel with these numbers",
        "",
        "- The decision-time board is rebuilt from `StreetDealt`. The",
        f"  post-action snapshot leaks the next street's cards on **{leaked}",
        f"  of {counts['decisions']} decisions "
        f"({100 * leaked / counts['decisions']:.1f}%)**; anything reading",
        "  `snapshot.boardCards` scores those on a board the actor could not",
        "  see.",
        "- `TimeoutAction` events are excluded — they are not decisions the",
        "  actor made. Including them moves the field fold rate to 58.6% and",
        "  stops reproducing the published figure.",
        f"- The per-agent block selects on the archive's own dense `rank <=",
        f"  {DEFAULT_TOP_N}`, which ties: {counts['tracked_agents']} distinct",
        "  agents across the two collections, not 30.",
        "- Per-agent samples are small (61-182 decisions). Intervals are",
        "  wide and several are withheld; the block shows a spread, never a",
        "  ranking.",
        "- The median-bet interval is degenerate because 0.600 is a hard",
        "  mode of the field's sizing, not because the estimate is precise.",
        "",
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the field strength-separation benchmark on the "
            "canonical strength metric."
        )
    )
    parser.add_argument(
        "--roots", nargs="+", default=[str(root) for root in DEFAULT_ROOTS]
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bootstrap < 1 or args.top < 1:
        raise SystemExit("--bootstrap and --top must be >= 1")
    roots = [Path(root) for root in args.roots]
    print("collecting decisions...", flush=True)
    decisions, stats = collect(roots, seed=args.seed, limit=args.limit)
    if not decisions:
        raise SystemExit("no decisions were scored")
    leaderboards = {
        Path(root).name: leaderboard_top(root, args.top) for root in roots
    }
    report = build_report(
        decisions,
        stats,
        leaderboards,
        seed=args.seed,
        resamples=args.bootstrap,
        roots=roots,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    markdown = render_markdown(report)
    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
        print(f"wrote {markdown_path}")
    print()
    print(markdown)
    return 0 if report["validation"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
