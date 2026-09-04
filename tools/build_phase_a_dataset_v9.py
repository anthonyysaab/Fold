"""Build the v9 Phase-A supervised dataset from complete-information replays.

Sibling of ``tools.build_phase_a_dataset`` — the v8 builder and its
stored dataset stay byte-untouched, and the replay walking, label
windows (continuing opponents, strongest-holding octile, exact/MC
showdown equity), determinism discipline, and per-decision seeding are
IMPORTED from it verbatim. What changes is the v9 data contract, pinned
in `.handoff/notes/V9_RESTRUCTURE_PLAN.md` ("L3/L4 DATA CONTRACTS" +
"L3 LANDED") and enforced by the landed trainer loader
(``engine.v9_trainer.load_phase_a_dataset_v9``):

- **Renamed keys are the version guard**: labels/masks are
  ``fold_through_active`` / ``fold_through_aggressive``; every row
  carries ``to_call_zero`` (a JSON bool) and ``read_temperature_x10``
  (g's read as the raw int ``10·T``).
- **The fold-through gate keys on raw wager actions** ``{bet, raise,
  all-in}``, never on a branch or family — the v9 ``active`` branch
  contains calls, which define no fold-through, and no validator would
  catch that mislabel (the pinned rationale). The lane is the state's:
  ``to_call == 0`` supervises ``fold_through_active`` (the bet
  execution), a priced wager supervises ``fold_through_aggressive``.
  One carve-out the v8 builder lacked: an all-in at a price that does
  NOT exceed the call-to amount is economically a call — it puts nobody
  to a decision and buys no folds — so it supervises neither lane and
  is counted (``allin_call_for_less``) instead of mislabeled.
- **The midpoint rule is DELETED** (there are no fixed-size branches to
  assign); the realized size is recorded instead, raw and normalized
  (``realized``: action, to-amount, new chips, wager per effective
  stack), so any later sizing analysis reads the data rather than a
  discretization of it.
- **Features are schema 4** (``extract_features_v9``) with the FITTED
  P3 belief provider — the same provider the v9 serve path defaults to,
  so the eight bucket inputs are live in training exactly as they are
  at serve; per-decision provider degrades are counted. The read
  (``read_temperature_x10``) is the extractor's own convention
  (unconditioned multiway MC at the schema-frozen
  ``feature_extract_v8._EQUITY_TRIALS`` with this decision's seed), and
  the vector's ``equity_multiway`` slot is asserted equal to it per row
  — one computation, two consumers, checked not assumed.
- **The sidecar carries the composed ``sizing`` record** the features
  were extracted under (every dial OFF today). The v9 Phase-A trainer's
  ``resolve_sizing_record`` reads exactly this block, so the manifest a
  training run ships describes the same g state the vectors baked.

Everything else — equity labels vs the actual continuing holdings, the
strongest-continuing octile, the everyone-folded winner invariant, the
skip accounting, byte-identical reruns — is the v8 builder's behavior,
reused or reproduced unchanged.

Offline and read-only over the archive; writes only the dataset and its
sidecar under ``artifacts/phase_a_v9/``. No Arena requests, no
credentials, no promotion.

Usage:
    python -m tools.build_phase_a_dataset_v9
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import random
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

from engine import schema4
from engine.aggression_sizing import read_to_context_int, table_temperature
from engine.feature_extract_v8 import _EQUITY_TRIALS
from engine.feature_extract_v9 import extract_features_v9
from engine.game_state import (
    _hero_and_seats,
    active_opponent_count,
    effective_stack_chips,
)
from engine.hand_strength import estimate_equity
from engine.p3_belief_provider import P3BeliefProvider
from engine.rules.composition import composed_sizing_record
from engine.strength_metric import strength_percentile
from tools.build_p3_dataset import decision_board
from tools.build_phase_a_dataset import (
    DEFAULT_EQUITY_TRIALS,
    DEFAULT_POTENTIAL_TRIALS,
    DEFAULT_ROOTS,
    DEFAULT_SEED,
    PhaseAInvariantError,
    _continuing_opponents,
    _decision_seed,
    _hole_cards_by_seat,
    _in_hand_opponents,
    _require_unit,
    _showdown_equity,
)
from tools.collect_foreign_play_data import (
    _action_size,
    _read_json,
    _reconstruct_state,
    _unwrap_rpc,
)

DEFAULT_OUTPUT_V9 = Path("artifacts") / "phase_a_v9" / "phase-a-dataset-v9.jsonl.gz"

_ACTION_EVENT_TYPES = ("ActionTaken", "TimeoutAction")
_STREETS = ("preflop", "flop", "turn", "river")
#: The definitional board size per street — the invariant the look-ahead
#: repair below restores.
_BOARD_LENGTH_V9 = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
_LABEL_NAMES_V9 = (
    "fold_through_active",
    "fold_through_aggressive",
    "range_bucket",
    "equity_called",
)
#: The pinned gate: raw wager actions, never a branch or family.
_WAGER_ACTIONS = frozenset({"bet", "raise", "all-in"})
_EQUITY_MULTIWAY_INDEX = schema4.feature_index_v9("equity_multiway")

#: One fitted provider per process (workers re-create it lazily); the
#: parent records its provenance in the sidecar.
_PROVIDER: P3BeliefProvider | None = None


def _belief_provider() -> P3BeliefProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = P3BeliefProvider.from_artifact()
    return _PROVIDER


def wager_lane(
    action: str, to_call: int, to_amount: int | None, call_to_amount: int
) -> str | None:
    """Which fold-through lane a realized wager supervises, if any.

    ``None`` for non-wager actions, for wagers whose size could not be
    read, and for the call-for-less all-in (a priced all-in whose
    to-amount does not exceed the call-to amount: nobody faces new
    chips, so there is no fold-through to observe). The lane is the
    state's: a free-spot wager is the active branch's bet execution, a
    priced escalation is the aggressive branch's.
    """

    if action not in _WAGER_ACTIONS or to_amount is None:
        return None
    if to_call == 0:
        return "fold_through_active"
    if to_amount <= call_to_amount:
        return None
    return "fold_through_aggressive"


def _decision_row_v9(
    replay: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    decision_index: int,
    holes_by_seat: Mapping[int, list[str]],
    winner_agent_ids: frozenset[str],
    stats: Counter,
    *,
    seed: int,
    equity_trials: int,
    potential_trials: int,
) -> dict[str, Any]:
    """One v9 dataset row for one ``ActionTaken`` decision, validated."""

    event = events[decision_index]
    payload = event["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("action event payload is not an object")
    state = _reconstruct_state(dict(replay), dict(event))
    table_id = str(state.get("tableId") or state.get("id") or "")
    if not table_id:
        raise ValueError("replay has no table id")
    sequence = event.get("sequence")
    if not isinstance(sequence, int):
        raise ValueError("action event has no integer sequence")
    actor_seat = payload.get("seatNumber")
    if not isinstance(actor_seat, int):
        raise ValueError("action event has no integer seatNumber")
    action = str(payload.get("action") or "").casefold()
    street = str(event.get("street") or "").casefold()
    if street not in _STREETS:
        raise ValueError(f"unsupported street {event.get('street')!r}")

    # Repair the board look-ahead the SHARED reconstruction leaves behind.
    #
    # `_reconstruct_state` rewinds the actor's own stack and street bet
    # from the payload's `stackBefore` / `actorCurrentBetBefore`, but it
    # does NOT rewind `boardCards`, and the event snapshot is the table
    # AFTER the action resolved. So on every action that closes a
    # betting round the reconstructed state already carries the next
    # street's cards. Measured on this archive: 962 of 9,084 decisions
    # (10.6%), independently reproduced here at 10.1% on a 444-decision
    # sample — preflop decisions seeing the flop, flop seeing the turn,
    # turn seeing the river.
    #
    # For v9 that is not a cosmetic leak. The board feeds the schema-4
    # card planes and board tiers, BOTH labels (`range_bucket`'s
    # strength percentile and `equity_called`'s showdown equity), the
    # `equity_multiway` feature, AND g's recorded `read_temperature_x10`
    # — so one un-rewound field would train the network, its labels and
    # its sizing read on cards the actor had not seen. Phase-A rows are
    # co-trained inside Phase B, so it would poison both phases.
    #
    # The repair is not new work: `build_p3_dataset.decision_board`
    # already rebuilds the board from the preceding `StreetDealt`
    # events, cross-checks it against the street's definitional card
    # count, and raises rather than guessing. It is applied there and in
    # `measure_field_separation`, and was simply never applied to the
    # Phase-A builders. Fixed in the v9 SIBLING ONLY — the shared
    # `_reconstruct_state` feeds the frozen v8 tool and the foreign-play
    # collection that frozen reports cite, and must not move.
    snapshot_board = [str(card) for card in state.get("boardCards") or []]
    true_board = [str(card) for card in decision_board(events, sequence, street)]
    if true_board != snapshot_board:
        stats["board_corrected"] += 1
        state["boardCards"] = true_board
    if len(true_board) != _BOARD_LENGTH_V9[street]:  # pragma: no cover
        # decision_board enforces this itself; asserted again here so the
        # invariant is stated where the row is built, not only upstream.
        raise PhaseAInvariantError(
            f"street {street!r} implies {_BOARD_LENGTH_V9[street]} board "
            f"cards, got {len(true_board)}"
        )

    allowed = state.get("allowedActions")
    if not isinstance(allowed, Mapping):
        raise ValueError("reconstructed state lacks allowedActions")
    to_call_raw = allowed.get("callChips") or 0
    if isinstance(to_call_raw, bool) or not isinstance(to_call_raw, int):
        raise ValueError("allowedActions.callChips is not an integer")
    to_call = max(0, to_call_raw)
    to_call_zero = to_call == 0

    decision_seed = _decision_seed(seed, table_id, sequence)
    provider = _belief_provider()
    features = extract_features_v9(
        state,
        belief_provider=provider,
        potential_trials=potential_trials,
        seed=decision_seed,
    )
    if provider.last_degrade_reason is not None:
        stats["belief_degrades"] += 1
    schema4.require_vector_v9(features)

    # g's read, the extractor's own convention (same estimator, trials,
    # and seed as the vector's equity_multiway — asserted below).
    hero, _ = _hero_and_seats(state)
    hero_hole_state = [str(card) for card in hero.get("holeCards") or ()]
    if len(hero_hole_state) != 2:
        raise ValueError("reconstructed state lacks the actor's hole cards")
    board = [str(card) for card in state.get("boardCards") or []]
    opponents = active_opponent_count(state)
    equity_read = (
        1.0
        if opponents < 1
        else estimate_equity(
            (hero_hole_state[0], hero_hole_state[1]),
            tuple(board),
            opponents,
            trials=_EQUITY_TRIALS,
            seed=decision_seed,
        )
    )
    reading = table_temperature(state, allowed, equity_read)
    if reading is None:
        raise ValueError("temperature read failed on a reconstructed state")
    encoded = read_to_context_int(reading.temperature)
    if features[_EQUITY_MULTIWAY_INDEX] != equity_read:
        raise PhaseAInvariantError(
            f"table {table_id} seq {sequence}: the vector's equity_multiway "
            f"{features[_EQUITY_MULTIWAY_INDEX]!r} is not the recorded "
            f"read's equity {equity_read!r}"
        )

    hero_hole = holes_by_seat.get(actor_seat)
    if hero_hole is None:
        raise ValueError("actor holding is not revealed in HoleCardsDealt")

    in_hand = _in_hand_opponents(state, actor_seat)
    continuing = _continuing_opponents(events, decision_index, in_hand)

    labels: dict[str, Any] = {name: 0.0 for name in _LABEL_NAMES_V9}
    labels["range_bucket"] = 0
    masks: dict[str, int] = {name: 0 for name in _LABEL_NAMES_V9}
    realized: dict[str, Any] | None = None

    # --- fold_through: gated on raw wager actions, lane by state ----------
    if action in _WAGER_ACTIONS:
        outcome = 1.0 if not continuing else 0.0
        to_amount, new_chips = _action_size(dict(payload))
        contribution_raw = hero.get("currentBetChips") or 0
        contribution = (
            int(contribution_raw)
            if isinstance(contribution_raw, int)
            and not isinstance(contribution_raw, bool)
            else 0
        )
        lane = wager_lane(action, to_call, to_amount, contribution + to_call)
        if to_amount is None:
            stats["unsized_wagers"] += 1
        else:
            realized = {
                "action": action,
                "to_amount": int(to_amount),
                "new_chips": int(new_chips),
                "wager_per_effective": float(new_chips)
                / max(1, effective_stack_chips(state)),
            }
            if lane is None:
                stats["allin_call_for_less"] += 1
        if lane is not None:
            labels[lane] = outcome
            masks[lane] = 1
            if outcome == 1.0:
                actor_agent_id = event.get("agentId")
                if (
                    isinstance(actor_agent_id, str)
                    and winner_agent_ids
                    and actor_agent_id not in winner_agent_ids
                ):
                    raise PhaseAInvariantError(
                        f"table {table_id} seq {sequence}: everyone folded "
                        "to this wager but the actor is not a settled winner"
                    )

    # --- range_bucket / equity_called: the v8 mechanics, verbatim ----------
    continuing_holes = [
        holes_by_seat[number]
        for number in sorted(continuing)
        if number in holes_by_seat
    ]
    if continuing and len(continuing_holes) == len(continuing):
        strongest = max(
            strength_percentile(hole, board) for hole in continuing_holes
        )
        _require_unit(strongest, "continuing strength percentile")
        labels["range_bucket"] = min(
            schema4.BELIEF_BUCKETS - 1,
            int(strongest * schema4.BELIEF_BUCKETS),
        )
        masks["range_bucket"] = 1

        rng = random.Random(decision_seed)
        equity = _showdown_equity(
            hero_hole,
            continuing_holes,
            board,
            list(holes_by_seat.values()),
            rng,
            equity_trials,
        )
        _require_unit(equity, "equity_called")
        labels["equity_called"] = equity
        masks["equity_called"] = 1

    # --- the generator-side guarantees of what the loader enforces --------
    if masks["fold_through_active"] and masks["fold_through_aggressive"]:
        raise PhaseAInvariantError("both fold_through lane masks are set")
    if masks["fold_through_active"] and not to_call_zero:
        raise PhaseAInvariantError(
            "fold_through_active is masked in on a priced decision"
        )
    if masks["fold_through_aggressive"] and to_call_zero:
        raise PhaseAInvariantError(
            "fold_through_aggressive is masked in on a free-spot decision"
        )
    if not isinstance(labels["range_bucket"], int) or not (
        0 <= labels["range_bucket"] < schema4.BELIEF_BUCKETS
    ):
        raise PhaseAInvariantError(f"range_bucket {labels['range_bucket']!r}")
    _require_unit(labels["equity_called"], "equity_called")

    agent_name = payload.get("agentName")
    row: dict[str, Any] = {
        "table_id": table_id,
        "seat": actor_seat,
        "street": street,
        "actor_agent": agent_name if isinstance(agent_name, str) else "",
        "sequence": sequence,
        "to_call_zero": to_call_zero,
        "read_temperature_x10": encoded,
        "features": list(features),
        "labels": labels,
        "masks": masks,
    }
    if realized is not None:
        row["realized"] = realized
    return row


def replay_rows_v9(
    replay: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    equity_trials: int = DEFAULT_EQUITY_TRIALS,
    potential_trials: int = DEFAULT_POTENTIAL_TRIALS,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """All v9 dataset rows for one unwrapped replay, plus counts."""

    stats: Counter[str] = Counter()
    events = replay.get("events")
    if not isinstance(events, list):
        stats["replay_without_events"] += 1
        return [], stats
    table = replay.get("table")
    if not isinstance(table, Mapping):
        stats["replay_without_table"] += 1
        return [], stats
    holes_by_seat = _hole_cards_by_seat(events)
    if not holes_by_seat:
        stats["replay_without_hole_cards"] += 1
        return [], stats
    winner_agent_ids = frozenset(
        winner.get("agentId")
        for winner in table.get("winners") or []
        if isinstance(winner, Mapping) and isinstance(winner.get("agentId"), str)
    )
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        event_type = event.get("type") if isinstance(event, Mapping) else None
        if event_type == "TimeoutAction":
            stats["timeout_actions"] += 1
            continue
        if event_type != "ActionTaken":
            continue
        try:
            rows.append(
                _decision_row_v9(
                    replay,
                    events,
                    index,
                    holes_by_seat,
                    winner_agent_ids,
                    stats,
                    seed=seed,
                    equity_trials=equity_trials,
                    potential_trials=potential_trials,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            stats["skipped_decisions"] += 1
            stats[f"skip:{type(error).__name__}"] += 1
    return rows, stats


def _process_file_v9(
    args: tuple[str, int, int, int],
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Worker: one raw table file to its v9 rows (importable for pickling)."""

    path_text, seed, equity_trials, potential_trials = args
    path = Path(path_text)
    stats: Counter[str] = Counter()
    try:
        replay = _unwrap_rpc(_read_json(path))
    except (OSError, ValueError):
        stats["unreadable_files"] += 1
        return path.name, [], dict(stats)
    if not isinstance(replay, dict):
        stats["unreadable_files"] += 1
        return path.name, [], dict(stats)
    rows, replay_stats = replay_rows_v9(
        replay,
        seed=seed,
        equity_trials=equity_trials,
        potential_trials=potential_trials,
    )
    stats.update(replay_stats)
    return path.name, rows, dict(stats)


def _coverage_v9(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {
        street: {
            "rows": 0,
            "free_spot_rows": 0,
            **{name: 0 for name in _LABEL_NAMES_V9},
        }
        for street in _STREETS
    }
    for row in rows:
        entry = coverage[row["street"]]
        entry["rows"] += 1
        if row["to_call_zero"]:
            entry["free_spot_rows"] += 1
        for name in _LABEL_NAMES_V9:
            entry[name] += row["masks"][name]
    return coverage


def _print_summary_v9(
    coverage: Mapping[str, Mapping[str, int]], totals: Mapping[str, int]
) -> None:
    names = ("free_spot_rows", *_LABEL_NAMES_V9)
    header = f"{'street':<9} {'rows':>7} " + " ".join(
        f"{name:>23}" for name in names
    )
    print(header)
    print("-" * len(header))
    for street in _STREETS:
        entry = coverage[street]
        print(
            f"{street:<9} {entry['rows']:>7} "
            + " ".join(f"{entry[name]:>23}" for name in names)
        )
    print(
        f"{'total':<9} {totals['rows']:>7} "
        + " ".join(f"{totals[name]:>23}" for name in names)
    )


class PhaseARowSink:
    """The shared v9 Phase-A row sink — one mechanism, two builders.

    Everything downstream of row generation for the Phase-A pipeline
    lives here once: the chunked flush sorted by
    ``(table_id, sequence)``, the per-table dedupe, the heapq k-way
    merge, the gzip archive with ``mtime=0`` (identical inputs give
    byte-identical archives), the atomic replace, the per-street label
    coverage accumulation, the ``.summary.json`` sidecar, and the
    trainer self-load that proves the written dataset parses. Used by
    ``build_dataset_v9`` (Arena roots) and by
    ``tools.build_phase_a_dataset_phh`` (PHH hand-history files).

    Dedupe keys on the TABLE ID, per row, not per file: one ``.phhs``
    file holds many hands (many table ids), so a duplicate table's rows
    drop wholesale while the file's other hands stay. On the Arena
    roots one file is one table, so this is exactly the pre-sink
    behaviour — the frozen oracle ``ecb4739df9d1b9ec`` pins it.
    """

    def __init__(
        self,
        output: Path,
        *,
        dedupe_tables: bool = True,
        chunk_rows: int = 50_000,
    ) -> None:
        self._output = Path(output)
        self._dedupe_tables = dedupe_tables
        self._chunk_row_limit = chunk_rows
        self._output.parent.mkdir(parents=True, exist_ok=True)
        self._temporary_dir = tempfile.TemporaryDirectory(
            prefix="phase-a-v9-chunks-", dir=self._output.parent
        )
        self._chunk_files: list[Path] = []
        self._chunk: list[dict[str, Any]] = []
        self._seen_tables: set[str] = set()
        self._duplicate_rows_dropped = 0
        self._tables_with_rows: Counter[str] = Counter()
        self._collection_order: list[str] = []
        self._stats_by_collection: dict[str, Counter[str]] = {}
        self._row_count = 0
        self._coverage: dict[str, dict[str, int]] | None = None
        self._totals: dict[str, int] = {}
        self._sidecar_path: Path | None = None

    @property
    def output(self) -> Path:
        return self._output

    @property
    def duplicate_rows_dropped(self) -> int:
        return self._duplicate_rows_dropped

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def coverage(self) -> dict[str, dict[str, int]]:
        if self._coverage is None:
            raise RuntimeError("finish() must run before reading coverage")
        return self._coverage

    @property
    def totals(self) -> dict[str, int]:
        return self._totals

    @property
    def sidecar_path(self) -> Path:
        if self._sidecar_path is None:
            raise RuntimeError("finish() must run before reading sidecar_path")
        return self._sidecar_path

    @property
    def combined_stats(self) -> dict[str, int]:
        """Every per-collection counter folded into one (for sidecars)."""
        combined: Counter[str] = Counter()
        for counter in self._stats_by_collection.values():
            combined.update(counter)
        return dict(combined)

    def consume(
        self,
        collection: str,
        rows: Sequence[Mapping[str, Any]],
        stats: Mapping[str, int] | None = None,
    ) -> None:
        """Feed one collection's rows into the pending chunk.

        With ``dedupe_tables`` on, rows whose table id was already
        consumed drop wholesale and are counted in
        ``duplicate_rows_dropped``; ``tables_with_rows`` counts the
        kept tables. ``stats`` is folded into the per-collection
        counters that the sidecar reports.
        """

        self._collection_order.append(collection)
        counter = self._stats_by_collection.setdefault(collection, Counter())
        if stats:
            counter.update(stats)
        if not rows:
            return
        if self._dedupe_tables:
            order: list[str] = []
            groups: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                table_id = str(row["table_id"])
                if table_id not in groups:
                    groups[table_id] = []
                    order.append(table_id)
                groups[table_id].append(row)
            kept: list[dict[str, Any]] = []
            new_tables = 0
            for table_id in order:
                group = groups[table_id]
                if table_id in self._seen_tables:
                    self._duplicate_rows_dropped += len(group)
                    continue
                self._seen_tables.add(table_id)
                new_tables += 1
                kept.extend(group)
            if new_tables:
                self._tables_with_rows[collection] += new_tables
        else:
            kept = list(rows)
            self._tables_with_rows[collection] += len(
                {str(row["table_id"]) for row in kept}
            )
        self._chunk.extend(kept)
        if len(self._chunk) >= self._chunk_row_limit:
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        if not self._chunk:
            return
        self._chunk.sort(key=lambda row: (row["table_id"], row["sequence"]))
        chunk_path = (
            Path(self._temporary_dir.name)
            / f"chunk-{len(self._chunk_files):05d}.jsonl"
        )
        with chunk_path.open("w", encoding="utf-8") as stream:
            for row in self._chunk:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._chunk_files.append(chunk_path)
        self._chunk.clear()

    @staticmethod
    def _chunk_rows(path: Path):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    @staticmethod
    def _sort_key(row: Mapping[str, Any]) -> tuple[str, int]:
        return (row["table_id"], row["sequence"])

    def finish(
        self,
        *,
        generator: Mapping[str, Any],
        skipped_roots: Mapping[str, str] | None = None,
        file_count: int | None = None,
        counts_extra: Mapping[str, Any] | None = None,
        label_coverage_in_generator: bool = False,
    ) -> dict[str, Any]:
        """Flush, k-way merge into the gzip archive, write the sidecar.

        The sidecar's ``generator`` block is caller-shaped (Arena vs
        PHH provenance); everything else the sink owns.
        ``label_coverage_in_generator`` adds the totals into the
        generator block as ``label_coverage`` (the PHH sidecar records
        it there; the Arena sidecar keeps it in ``counts`` only). The
        build ends by loading the written archive through the trainer's
        own ``load_phase_a_dataset_v9`` — a dataset this sink blesses
        is one the trainer provably accepts.
        """

        self._flush_chunk()
        if not self._chunk_files:
            raise ValueError("no decision rows were produced")

        # k-way merge in (table_id, sequence) order — the exact order
        # the in-memory sort of the whole set would produce.
        # heapq.merge is stable across equal keys, so rows from equal
        # (table_id, sequence) keys keep file order, matching the old
        # global sort byte for byte.
        coverage: dict[str, dict[str, int]] = {
            street: {
                "rows": 0,
                "free_spot_rows": 0,
                **{name: 0 for name in _LABEL_NAMES_V9},
            }
            for street in _STREETS
        }
        temporary = self._output.with_suffix(self._output.suffix + ".tmp")
        row_count = 0
        with open(temporary, "wb") as raw_stream:
            # mtime=0 so identical inputs give byte-identical archives.
            with gzip.GzipFile(
                fileobj=raw_stream, mode="wb", filename="", mtime=0
            ) as stream:
                for row in heapq.merge(
                    *(self._chunk_rows(path) for path in self._chunk_files),
                    key=self._sort_key,
                ):
                    stream.write(
                        (json.dumps(row, separators=(",", ":")) + "\n").encode()
                    )
                    row_count += 1
                    entry = coverage[row["street"]]
                    entry["rows"] += 1
                    if row["to_call_zero"]:
                        entry["free_spot_rows"] += 1
                    for name in _LABEL_NAMES_V9:
                        entry[name] += row["masks"][name]
        temporary.replace(self._output)
        self._temporary_dir.cleanup()
        self._row_count = row_count
        self._coverage = coverage

        totals: dict[str, int] = {"rows": row_count}
        for name in ("free_spot_rows", *_LABEL_NAMES_V9):
            totals[name] = sum(coverage[street][name] for street in _STREETS)
        self._totals = totals

        collection_names = list(dict.fromkeys(self._collection_order))
        collection_stats = {
            name: dict(sorted(self._stats_by_collection[name].items()))
            for name in collection_names
        }
        counts: dict[str, Any] = {
            "files": (
                file_count
                if file_count is not None
                else len(collection_names)
            ),
            "tables_with_rows": sum(self._tables_with_rows.values()),
            "rows": row_count,
            "duplicate_table_rows_dropped": self._duplicate_rows_dropped,
            "label_coverage": totals,
        }
        if counts_extra:
            counts.update(counts_extra)
        sidecar_document: dict[str, Any] = {
            "schema_version": schema4.SCHEMA_VERSION_V9,
            "input_size": schema4.INPUT_SIZE_V9,
            # The composed record the vectors' costs were extracted
            # under — the v9 Phase-A trainer's resolve_sizing_record
            # reads THIS, so a training run's manifest describes the
            # same g state the features baked. Canonical (JSON) form.
            "sizing": json.loads(
                json.dumps(
                    composed_sizing_record(), sort_keys=True, allow_nan=False
                )
            ),
            "generator": dict(generator),
            "counts": counts,
            "dedupe_tables": self._dedupe_tables,
            "skipped_roots": dict(skipped_roots or {}),
            "per_street": coverage,
            "per_collection": {
                name: {
                    "tables_with_rows": self._tables_with_rows.get(name, 0),
                    **collection_stats.get(name, {}),
                }
                for name in collection_names
            },
            "files": {"dataset": self._output.name},
        }
        if label_coverage_in_generator:
            sidecar_document["generator"]["label_coverage"] = totals
        self._sidecar_path = self._output.parent / (
            self._output.name.removesuffix(".jsonl.gz") + ".summary.json"
        )
        self._sidecar_path.write_text(
            json.dumps(sidecar_document, indent=2) + "\n", encoding="utf-8"
        )

        # The proof, not a formality: the trainer's loader is the contract.
        from engine.v9_trainer import load_phase_a_dataset_v9

        loaded = load_phase_a_dataset_v9(self._output)
        if len(loaded) != row_count:
            raise PhaseAInvariantError(
                f"the trainer loaded {len(loaded)} rows from a dataset "
                f"written with {row_count}"
            )
        return sidecar_document


def _consume_files(
    sink: PhaseARowSink,
    work: Sequence[tuple[Any, ...]],
    collection_names: Sequence[str],
    process: Callable[
        [tuple[Any, ...]],
        tuple[str, list[dict[str, Any]], dict[str, int]],
    ],
    *,
    workers: int,
    noun: str = "files",
) -> None:
    """Map ``process`` over ``work`` and feed every result into the sink.

    ``process`` must be a module-level function so ``workers > 1``
    pickles it; progress is printed per 100 units, identical for the
    Arena (``noun="files"``) and PHH (``noun="roots"``) builders.
    """

    started = time.monotonic()
    total_rows = 0

    def _report(index: int) -> None:
        elapsed = time.monotonic() - started
        print(
            f"processed {index + 1}/{len(work)} {noun}, {total_rows} rows "
            f"({sink.duplicate_rows_dropped} duplicate rows dropped), "
            f"{elapsed:.0f}s",
            flush=True,
        )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(
                pool.map(process, work, chunksize=8)
            ):
                sink.consume(collection_names[index], result[1], result[2])
                total_rows += len(result[1])
                if (index + 1) % 100 == 0 or index + 1 == len(work):
                    _report(index)
    else:
        for index, item in enumerate(work):
            result = process(item)
            sink.consume(collection_names[index], result[1], result[2])
            total_rows += len(result[1])
            if (index + 1) % 100 == 0 or index + 1 == len(work):
                _report(index)


def build_dataset_v9(
    roots: Sequence[Path],
    output: Path,
    *,
    seed: int = DEFAULT_SEED,
    equity_trials: int = DEFAULT_EQUITY_TRIALS,
    potential_trials: int = DEFAULT_POTENTIAL_TRIALS,
    workers: int = 1,
    limit: int | None = None,
    dedupe_tables: bool = True,
    chunk_rows: int = 50_000,
) -> dict[str, Any]:
    """Build the v9 dataset and sidecar; return the sidecar document.

    The build ends by loading the written dataset through the TRAINER's
    own ``load_phase_a_dataset_v9`` — a dataset this tool blesses is one
    the trainer provably accepts (the L3 LANDED rule).

    All row-sink machinery (chunk flush, per-table dedupe, k-way merge,
    byte-deterministic gzip, atomic replace, coverage, sidecar,
    self-load) lives in the shared ``PhaseARowSink`` — also used by
    ``tools.build_phase_a_dataset_phh``. The widened-roots run
    (2026-09-02 diagnosis, ~1.14M rows) cannot hold every row in RAM,
    so rows stream through fixed-size sorted chunks that are k-way
    merged into the final archive — the same ``(table_id, sequence)``
    order the in-memory sort produced, byte for byte. ``dedupe_tables``
    (on by default) drops rows whose table id was already emitted: the
    widened archive contains 536 byte-identical duplicate table files
    between collections (md5-verified in the diagnosis), which would
    double-count training rows. The default roots contain no
    duplicates, so a default rebuild is byte-identical to the
    pre-streaming builder. Roots whose ``raw/tables`` is missing or
    empty are skipped with a warning (the container directories and two
    empty leaf collections in the archive).
    """

    files: list[tuple[str, Path]] = []
    skipped_roots: list[tuple[str, str]] = []
    for root in roots:
        table_dir = Path(root) / "raw" / "tables"
        if not table_dir.is_dir():
            skipped_roots.append((str(root), "no raw/tables directory"))
            continue
        collection_files = sorted(table_dir.glob("*.json"))
        if not collection_files:
            skipped_roots.append((str(root), "no raw table replays"))
            continue
        files.extend((Path(root).name, path) for path in collection_files)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError("no decision rows were produced (no readable roots)")
    for root, reason in skipped_roots:
        print(f"skipping root {root}: {reason}", flush=True)

    started = time.monotonic()
    sink = PhaseARowSink(
        output, dedupe_tables=dedupe_tables, chunk_rows=chunk_rows
    )
    work = [
        (str(path), seed, equity_trials, potential_trials) for _, path in files
    ]
    collection_names = [name for name, _ in files]
    _consume_files(sink, work, collection_names, _process_file_v9, workers=workers)

    sidecar_document = sink.finish(
        generator={
            "tool": "tools.build_phase_a_dataset_v9",
            "seed": seed,
            "equity_trials": equity_trials,
            "potential_trials": potential_trials,
            "read_equity_trials": _EQUITY_TRIALS,
            "belief_provider": "P3BeliefProvider",
            "belief_fit_source": _belief_provider().fit_source,
            "roots": [str(root) for root in roots],
            "limit": limit,
        },
        skipped_roots=dict(skipped_roots),
        file_count=len(files),
    )

    _print_summary_v9(sink.coverage, sink.totals)
    elapsed = time.monotonic() - started
    print(f"\nwrote {sink.output} ({sink.row_count} rows) in {elapsed:.0f}s")
    print(f"trainer loader accepted the dataset ({sink.row_count} rows)")
    if sink.duplicate_rows_dropped:
        print(f"dropped {sink.duplicate_rows_dropped} duplicate table rows")
    print(f"wrote {sink.sidecar_path}")
    return sidecar_document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the v9 Phase-A supervised dataset from replays."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[str(root) for root in DEFAULT_ROOTS],
        help="collection directories containing raw/tables/*.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "where to write the dataset. Required: the old default was "
            "the frozen artifact's own path, so a bare rerun overwrote "
            "the ecb4739df9d1b9ec oracle in place."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--equity-trials", type=int, default=DEFAULT_EQUITY_TRIALS
    )
    parser.add_argument(
        "--potential-trials", type=int, default=DEFAULT_POTENTIAL_TRIALS
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N files"
    )
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "drop rows whose table id was already emitted (default: on; the "
            "widened archive contains byte-identical duplicate table files "
            "between collections)"
        ),
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=50_000,
        help="rows per in-memory sort chunk before the k-way merge",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1 or args.equity_trials < 1 or args.potential_trials < 1:
        raise SystemExit("workers, equity-trials, potential-trials must be >= 1")
    if args.chunk_rows < 1:
        raise SystemExit("--chunk-rows must be >= 1")
    build_dataset_v9(
        [Path(root) for root in args.roots],
        Path(args.output),
        seed=args.seed,
        equity_trials=args.equity_trials,
        potential_trials=args.potential_trials,
        workers=args.workers,
        limit=args.limit,
        dedupe_tables=args.dedupe,
        chunk_rows=args.chunk_rows,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
