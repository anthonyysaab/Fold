"""Build the Phase-A supervised dataset from complete-information replays.

V8_DESIGN.md §5 Phase A: the raw foreign replay archive reveals every
seat's hole cards (folders included), so the three composed-value heads
have observable ground truth in real hands. This tool walks the raw table
replays (``foreign play data/<collection>/raw/tables/*.json``), and for
every ``ActionTaken`` decision of every seat emits one JSONL row:

- ``features`` — the full schema-3 vector (413 floats) from
  ``feature_extract_v8.extract_features_v8`` with the neutral belief
  provider, on the decision-time snapshot reconstructed exactly as
  ``tools.collect_foreign_play_data._reconstruct_state`` does (that module
  is the proven parser of this archive; this tool imports it rather than
  re-deriving the event-field reading).
- ``labels`` / ``masks`` — the Phase-A supervision with defined-masks
  (masked losses are the trainer's job):

  * ``fold_through_small`` / ``fold_through_large`` — defined only when
    the actor aggressed (canonical ``training_telemetry.action_family``).
    One observed outcome shared by both branch labels: 1.0 when every
    remaining opponent folded before the next street / showdown, else
    0.0. The mask records which E6 branch the realized size matched:
    realized new-chip wager <= midpoint of the small/large branch targets
    -> small, else large (targets computed with the same
    ``min(call + fraction*(pot+call), stack_fraction*eff)`` rule as
    ``feature_extract_v8._branch_block``, unclamped).
  * ``range_bucket`` — the strength-percentile octile (canonical
    ``strength_metric.strength_percentile`` on the decision-time board)
    of the STRONGEST holding among opponents who did not fold to the
    action; defined when at least one opponent continued.
  * ``equity_called`` — hero's showdown pot share against the actual
    continuing holdings, completing the actual current board: exact by
    enumeration when at most one board card remains (turn/river
    decisions), seeded Monte-Carlo (``--equity-trials``, default 500,
    ablatable) preflop/flop.

Documented approximations and conventions:

- ``allowedActions`` are NOT approximated: every raw ``ActionTaken``
  payload carries the engine's own ``allowedActions`` verbatim, and the
  reconstruction uses them directly. A decision without them is skipped
  and counted in the sidecar.
- "Remaining opponent" means a seat whose status was not folded/settled
  at decision time (all-in seats count — they cannot fold, so an all-in
  caller correctly forces ``fold_through`` to 0). "Continued" means the
  seat did not fold between this action and the street boundary (next
  ``StreetDealt`` / ``Showdown``) or hand end.
- ``all-in`` is an aggress by the canonical ``action_family`` even when
  it does not raise the current bet (a call-for-less); the branch mask
  still assigns it by realized size.
- ``TimeoutAction`` events produce no dataset row (they are not decisions
  the actor made) but their folds do drive the label windows; they are
  counted in the sidecar.
- Equity runouts draw from the true remaining deck: every revealed hole
  card (folders included) plus the board is dead. Ties split the pot
  share equally among the winners; side-pot structure is ignored.
- The range percentile is computed on the decision-time board (the head
  is "who is still in at this price", asked at the decision).

Impossible-by-construction validations (house rule), enforced per row:
the vector passes ``schema3.require_vector``; every equity/percentile
label sits in [0, 1] and ``range_bucket`` in 0..7; ``fold_through`` masks
are set only on aggress decisions (exactly one branch mask per sized
aggress); and a ``fold_through`` of 1.0 requires the actor to be listed
in the replay's own settled ``winners`` — an independent record of "the
bet took the pot".

Offline and read-only over the archive; writes only the dataset and its
sidecar under ``artifacts/phase_a/``. Deterministic: every stochastic
quantity is seeded per decision from ``(seed, table_id, sequence)``, so
row content is independent of worker count and file order, and the gzip
stream is written with a zeroed mtime so byte-identical reruns are
byte-identical outputs.

Usage:
    python -m tools.build_phase_a_dataset
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from devfun_poker_playground import schema3
from devfun_poker_playground._vendor.treys import Card, Evaluator
from devfun_poker_playground.belief_provider import NeutralBeliefProvider
from devfun_poker_playground.feature_extract_v8 import (
    _BRANCH_LARGE,
    _BRANCH_SMALL,
    extract_features_v8,
)
from devfun_poker_playground.game_state import effective_stack_chips
from devfun_poker_playground.strength_metric import strength_percentile
from devfun_poker_playground.training_telemetry import action_family
from tools.collect_foreign_play_data import (
    _action_size,
    _read_json,
    _reconstruct_state,
    _unwrap_rpc,
)

DEFAULT_ROOTS = (
    Path("foreign play data") / "20260812T082057Z_poker-playground_s13_top15",
    Path("foreign play data") / "20260815T210237Z_poker-playground_s14_top15",
)
DEFAULT_OUTPUT = Path("artifacts") / "phase_a" / "phase-a-dataset.jsonl.gz"
DEFAULT_EQUITY_TRIALS = 500
DEFAULT_POTENTIAL_TRIALS = 400
DEFAULT_SEED = 7

_ACTION_EVENT_TYPES = ("ActionTaken", "TimeoutAction")
_BOUNDARY_EVENT_TYPES = ("StreetDealt", "Showdown")
_OUT_OF_HAND_STATUSES = ("folded", "settled")
_STREETS = ("preflop", "flop", "turn", "river")
_LABEL_NAMES = (
    "fold_through_small",
    "fold_through_large",
    "range_bucket",
    "equity_called",
)


class PhaseAInvariantError(RuntimeError):
    """An impossible-by-construction dataset invariant was violated.

    Deliberately not a ValueError: per-decision degrade handling swallows
    malformed-replay ValueErrors, and an invariant violation must never be
    silently counted as a skip — it means the generator itself is wrong.
    """


_EVALUATOR: Evaluator | None = None


def _evaluator() -> Evaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = Evaluator()
    return _EVALUATOR


def _decision_seed(seed: int, table_id: str, sequence: int) -> int:
    """A stable per-decision seed: order- and worker-count-independent."""

    digest = hashlib.sha256(f"{seed}:{table_id}:{sequence}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _hole_cards_by_seat(events: Sequence[Mapping[str, Any]]) -> dict[int, list[str]]:
    """Every seat's revealed holding, from the ``HoleCardsDealt`` payload."""

    for event in events:
        if event.get("type") != "HoleCardsDealt":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        holes: dict[int, list[str]] = {}
        for seat in payload.get("seats") or []:
            if not isinstance(seat, Mapping):
                continue
            number = seat.get("seatNumber")
            cards = seat.get("holeCards")
            if (
                isinstance(number, int)
                and isinstance(cards, list)
                and len(cards) == 2
                and all(isinstance(card, str) and card for card in cards)
            ):
                holes[number] = list(cards)
        return holes
    return {}


def _in_hand_opponents(
    state: Mapping[str, Any], actor_seat: int
) -> set[int]:
    """Seats still in the hand at decision time, excluding the actor.

    Same status rule as ``game_state._active_seats``: folded/settled are
    out; ``AllIn`` seats are in (they cannot fold, and they contest the
    pot).
    """

    numbers: set[int] = set()
    for seat in state.get("seats") or []:
        if not isinstance(seat, Mapping):
            continue
        number = seat.get("seatNumber")
        if not isinstance(number, int) or number == actor_seat:
            continue
        status = str(seat.get("status") or "").casefold()
        if status not in _OUT_OF_HAND_STATUSES:
            numbers.add(number)
    return numbers


def _continuing_opponents(
    events: Sequence[Mapping[str, Any]],
    decision_index: int,
    in_hand: set[int],
) -> set[int]:
    """Opponents who did not fold before the street boundary or hand end."""

    continuing = set(in_hand)
    for event in events[decision_index + 1 :]:
        event_type = event.get("type")
        if event_type in _BOUNDARY_EVENT_TYPES:
            break
        if event_type not in _ACTION_EVENT_TYPES:
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get("action") or "").casefold() == "fold":
            seat = payload.get("seatNumber")
            if isinstance(seat, int):
                continuing.discard(seat)
    return continuing


def _showdown_equity(
    hero_hole: Sequence[str],
    opponent_holes: Sequence[Sequence[str]],
    board: Sequence[str],
    all_revealed: Sequence[Sequence[str]],
    rng: random.Random,
    equity_trials: int,
) -> float:
    """Hero's expected pot share completing the actual current board.

    Exact by enumeration when at most one board card remains; seeded
    Monte-Carlo otherwise. The remaining deck excludes the board and every
    revealed holding (folders included): with complete information those
    cards cannot arrive on the board. Ties split equally among winners.
    """

    dead = {card for hole in all_revealed for card in hole}
    dead.update(board)
    unseen = [code for code in schema3.CARD_CODES if code not in dead]
    missing = 5 - len(board)
    evaluator = _evaluator()
    hero_ints = [Card.new(card) for card in hero_hole]
    opponent_ints = [
        [Card.new(card) for card in hole] for hole in opponent_holes
    ]
    board_ints = [Card.new(card) for card in board]

    def _share(runout: list[int]) -> float:
        hero_rank = evaluator.evaluate(runout, hero_ints)
        ranks = [evaluator.evaluate(runout, hole) for hole in opponent_ints]
        best = min([hero_rank, *ranks])
        if hero_rank > best:  # treys: lower rank is stronger
            return 0.0
        winners = 1 + sum(rank == best for rank in ranks)
        return 1.0 / winners

    if missing == 0:
        return _share(board_ints)
    if missing == 1:
        total = 0.0
        for code in unseen:
            total += _share(board_ints + [Card.new(code)])
        return total / len(unseen)
    total = 0.0
    for _ in range(equity_trials):
        drawn = rng.sample(unseen, missing)
        total += _share(board_ints + [Card.new(code) for code in drawn])
    return total / equity_trials


def _branch_targets(
    pot: int, to_call: int, effective_stack: int
) -> tuple[float, float]:
    """The E6 small/large total-wager targets, unclamped.

    Same arithmetic as ``feature_extract_v8._branch_block._target`` —
    the constants are imported, never restated.
    """

    eff = max(1, effective_stack)

    def _target(spec: tuple[float, float]) -> float:
        pot_fraction, stack_fraction = spec
        return min(
            to_call + pot_fraction * (pot + to_call), stack_fraction * eff
        )

    return _target(_BRANCH_SMALL), _target(_BRANCH_LARGE)


def _require_unit(value: float, name: str) -> None:
    if not (isinstance(value, float) and 0.0 <= value <= 1.0):
        raise PhaseAInvariantError(f"{name} {value!r} is not in [0, 1]")


def _decision_row(
    replay: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    decision_index: int,
    holes_by_seat: Mapping[int, list[str]],
    winner_agent_ids: frozenset[str],
    *,
    seed: int,
    equity_trials: int,
    potential_trials: int,
) -> dict[str, Any]:
    """One dataset row for one ``ActionTaken`` decision, fully validated."""

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
    family = action_family(action)
    if family is None:
        raise ValueError(f"unsupported action {action!r}")
    street = str(event.get("street") or "").casefold()
    if street not in _STREETS:
        raise ValueError(f"unsupported street {event.get('street')!r}")

    decision_seed = _decision_seed(seed, table_id, sequence)
    features = extract_features_v8(
        state,
        belief_provider=NeutralBeliefProvider(),
        potential_trials=potential_trials,
        seed=decision_seed,
    )
    schema3.require_vector(features)

    board = [str(card) for card in state.get("boardCards") or []]
    hero_hole = holes_by_seat.get(actor_seat)
    if hero_hole is None:
        raise ValueError("actor holding is not revealed in HoleCardsDealt")

    in_hand = _in_hand_opponents(state, actor_seat)
    continuing = _continuing_opponents(events, decision_index, in_hand)

    labels: dict[str, Any] = {name: 0.0 for name in _LABEL_NAMES}
    labels["range_bucket"] = 0
    masks: dict[str, int] = {name: 0 for name in _LABEL_NAMES}

    # --- fold_through: defined only on aggress decisions ------------------
    if family == "aggress":
        outcome = 1.0 if not continuing else 0.0
        to_amount, new_chips = _action_size(dict(payload))
        if to_amount is not None:
            allowed = payload.get("allowedActions")
            to_call = 0
            if isinstance(allowed, Mapping):
                call_chips = allowed.get("callChips")
                if isinstance(call_chips, int) and not isinstance(
                    call_chips, bool
                ):
                    to_call = max(0, call_chips)
            pot = payload.get("pot")
            pot = pot if isinstance(pot, int) and not isinstance(pot, bool) else 0
            small_target, large_target = _branch_targets(
                pot, to_call, effective_stack_chips(state)
            )
            midpoint = (small_target + large_target) / 2.0
            branch = (
                "fold_through_small"
                if new_chips <= midpoint
                else "fold_through_large"
            )
            labels["fold_through_small"] = outcome
            labels["fold_through_large"] = outcome
            masks[branch] = 1
            if outcome == 1.0:
                actor_agent_id = event.get("agentId")
                if (
                    isinstance(actor_agent_id, str)
                    and winner_agent_ids
                    and actor_agent_id not in winner_agent_ids
                ):
                    raise PhaseAInvariantError(
                        f"table {table_id} seq {sequence}: everyone folded "
                        "to this bet but the actor is not a settled winner"
                    )
        # A sized aggress could not be read: leave both masks 0 (counted
        # by the caller via the returned row's masks).

    # --- range_bucket: strongest continuing holding ------------------------
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
            schema3.BELIEF_BUCKETS - 1,
            int(strongest * schema3.BELIEF_BUCKETS),
        )
        masks["range_bucket"] = 1

        # --- equity_called: vs the actual continuing holdings -------------
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

    if masks["fold_through_small"] and masks["fold_through_large"]:
        raise PhaseAInvariantError("both fold_through branch masks are set")
    if family != "aggress" and (
        masks["fold_through_small"] or masks["fold_through_large"]
    ):
        raise PhaseAInvariantError("fold_through defined on a non-aggress")
    if not isinstance(labels["range_bucket"], int) or not (
        0 <= labels["range_bucket"] < schema3.BELIEF_BUCKETS
    ):
        raise PhaseAInvariantError(f"range_bucket {labels['range_bucket']!r}")
    _require_unit(labels["equity_called"], "equity_called")
    _require_unit(labels["fold_through_small"], "fold_through_small")

    agent_name = payload.get("agentName")
    return {
        "table_id": table_id,
        "seat": actor_seat,
        "street": street,
        "actor_agent": agent_name if isinstance(agent_name, str) else "",
        "sequence": sequence,
        "features": list(features),
        "labels": labels,
        "masks": masks,
    }


def replay_rows(
    replay: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    equity_trials: int = DEFAULT_EQUITY_TRIALS,
    potential_trials: int = DEFAULT_POTENTIAL_TRIALS,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """All dataset rows for one unwrapped replay, plus skip/event counts."""

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
                _decision_row(
                    replay,
                    events,
                    index,
                    holes_by_seat,
                    winner_agent_ids,
                    seed=seed,
                    equity_trials=equity_trials,
                    potential_trials=potential_trials,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            stats["skipped_decisions"] += 1
            stats[f"skip:{type(error).__name__}"] += 1
    return rows, stats


def _process_file(
    args: tuple[str, int, int, int],
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Worker: one raw table file to its rows (importable for pickling)."""

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
    rows, replay_stats = replay_rows(
        replay,
        seed=seed,
        equity_trials=equity_trials,
        potential_trials=potential_trials,
    )
    stats.update(replay_stats)
    return path.name, rows, dict(stats)


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {
        street: {"rows": 0, **{name: 0 for name in _LABEL_NAMES}}
        for street in _STREETS
    }
    for row in rows:
        entry = coverage[row["street"]]
        entry["rows"] += 1
        for name in _LABEL_NAMES:
            entry[name] += row["masks"][name]
    return coverage


def _print_summary(
    coverage: Mapping[str, Mapping[str, int]], totals: Mapping[str, int]
) -> None:
    header = f"{'street':<9} {'rows':>7} " + " ".join(
        f"{name:>18}" for name in _LABEL_NAMES
    )
    print(header)
    print("-" * len(header))
    for street in _STREETS:
        entry = coverage[street]
        print(
            f"{street:<9} {entry['rows']:>7} "
            + " ".join(f"{entry[name]:>18}" for name in _LABEL_NAMES)
        )
    print(
        f"{'total':<9} {totals['rows']:>7} "
        + " ".join(f"{totals[name]:>18}" for name in _LABEL_NAMES)
    )


def build_dataset(
    roots: Sequence[Path],
    output: Path,
    *,
    seed: int = DEFAULT_SEED,
    equity_trials: int = DEFAULT_EQUITY_TRIALS,
    potential_trials: int = DEFAULT_POTENTIAL_TRIALS,
    workers: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build the dataset and sidecar; return the sidecar document."""

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
                pool.map(_process_file, work, chunksize=8)
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
            _consume(collection_names[index], _process_file(item))
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

    coverage = _coverage(all_rows)
    totals: dict[str, int] = {"rows": len(all_rows)}
    for name in _LABEL_NAMES:
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
        "schema_version": schema3.SCHEMA_VERSION,
        "input_size": schema3.INPUT_SIZE_V8,
        "generator": {
            "tool": "tools.build_phase_a_dataset",
            "seed": seed,
            "equity_trials": equity_trials,
            "potential_trials": potential_trials,
            "branch_small": list(_BRANCH_SMALL),
            "branch_large": list(_BRANCH_LARGE),
            "belief_provider": "NeutralBeliefProvider",
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

    _print_summary(coverage, totals)
    elapsed = time.monotonic() - started
    print(f"\nwrote {output} ({len(all_rows)} rows) in {elapsed:.0f}s")
    print(f"wrote {sidecar}")
    return sidecar_document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Phase-A supervised dataset from replays."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[str(root) for root in DEFAULT_ROOTS],
        help="collection directories containing raw/tables/*.json",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
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
    build_dataset(
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
