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
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

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


def build_dataset_v9(
    roots: Sequence[Path],
    output: Path,
    *,
    seed: int = DEFAULT_SEED,
    equity_trials: int = DEFAULT_EQUITY_TRIALS,
    potential_trials: int = DEFAULT_POTENTIAL_TRIALS,
    workers: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build the v9 dataset and sidecar; return the sidecar document.

    The build ends by loading the written dataset through the TRAINER's
    own ``load_phase_a_dataset_v9`` — a dataset this tool blesses is one
    the trainer provably accepts (the L3 LANDED rule).
    """

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
    all_rows: list[dict[str, Any]] = []
    stats_by_collection: dict[str, Counter[str]] = {
        name: Counter() for name, _ in files
    }
    tables_with_rows: Counter[str] = Counter()
    work = [
        (str(path), seed, equity_trials, potential_trials) for _, path in files
    ]
    collection_names = [name for name, _ in files]

    def _consume(
        collection: str, result: tuple[str, list[dict[str, Any]], dict[str, int]]
    ) -> None:
        _, rows, stats = result
        all_rows.extend(rows)
        stats_by_collection[collection].update(stats)
        if rows:
            tables_with_rows[collection] += 1

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(
                pool.map(_process_file_v9, work, chunksize=8)
            ):
                _consume(collection_names[index], result)
                if (index + 1) % 100 == 0 or index + 1 == len(work):
                    elapsed = time.monotonic() - started
                    print(
                        f"processed {index + 1}/{len(work)} files, "
                        f"{len(all_rows)} rows, {elapsed:.0f}s",
                        flush=True,
                    )
    else:
        for index, item in enumerate(work):
            _consume(collection_names[index], _process_file_v9(item))
            if (index + 1) % 100 == 0 or index + 1 == len(work):
                elapsed = time.monotonic() - started
                print(
                    f"processed {index + 1}/{len(work)} files, "
                    f"{len(all_rows)} rows, {elapsed:.0f}s",
                    flush=True,
                )

    if not all_rows:
        raise ValueError("no decision rows were produced")

    # Deterministic order regardless of worker scheduling.
    all_rows.sort(key=lambda row: (row["table_id"], row["sequence"]))

    coverage = _coverage_v9(all_rows)
    totals: dict[str, int] = {"rows": len(all_rows)}
    for name in ("free_spot_rows", *_LABEL_NAMES_V9):
        totals[name] = sum(coverage[street][name] for street in _STREETS)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "wb") as raw_stream:
        # mtime=0 so identical inputs give byte-identical archives.
        with gzip.GzipFile(
            fileobj=raw_stream, mode="wb", filename="", mtime=0
        ) as stream:
            for row in all_rows:
                stream.write(
                    (json.dumps(row, separators=(",", ":")) + "\n").encode()
                )
    temporary.replace(output)

    collection_stats = {
        name: dict(sorted(stats_by_collection[name].items()))
        for name in dict.fromkeys(collection_names)
    }
    sidecar_document: dict[str, Any] = {
        "schema_version": schema4.SCHEMA_VERSION_V9,
        "input_size": schema4.INPUT_SIZE_V9,
        # The composed record the vectors' costs were extracted under —
        # the v9 Phase-A trainer's resolve_sizing_record reads THIS, so
        # a training run's manifest describes the same g state the
        # features baked. Canonical (JSON) form.
        "sizing": json.loads(
            json.dumps(composed_sizing_record(), sort_keys=True, allow_nan=False)
        ),
        "generator": {
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
        "counts": {
            "files": len(files),
            "tables_with_rows": sum(tables_with_rows.values()),
            "rows": len(all_rows),
            "label_coverage": totals,
        },
        "per_street": coverage,
        "per_collection": {
            name: {
                "tables_with_rows": tables_with_rows.get(name, 0),
                **collection_stats.get(name, {}),
            }
            for name in dict.fromkeys(collection_names)
        },
        "files": {"dataset": output.name},
    }
    sidecar = output.parent / (
        output.name.removesuffix(".jsonl.gz") + ".summary.json"
    )
    sidecar.write_text(
        json.dumps(sidecar_document, indent=2) + "\n", encoding="utf-8"
    )

    # The proof, not a formality: the trainer's loader is the contract.
    from engine.v9_trainer import load_phase_a_dataset_v9

    loaded = load_phase_a_dataset_v9(output)
    if len(loaded) != len(all_rows):
        raise PhaseAInvariantError(
            f"the trainer loaded {len(loaded)} rows from a dataset written "
            f"with {len(all_rows)}"
        )

    _print_summary_v9(coverage, totals)
    elapsed = time.monotonic() - started
    print(f"\nwrote {output} ({len(all_rows)} rows) in {elapsed:.0f}s")
    print(f"trainer loader accepted the dataset ({len(loaded)} rows)")
    print(f"wrote {sidecar}")
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
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_V9))
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1 or args.equity_trials < 1 or args.potential_trials < 1:
        raise SystemExit("workers, equity-trials, potential-trials must be >= 1")
    build_dataset_v9(
        [Path(root) for root in args.roots],
        Path(args.output),
        seed=args.seed,
        equity_trials=args.equity_trials,
        potential_trials=args.potential_trials,
        workers=args.workers,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
