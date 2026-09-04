"""Build the v9 Phase-A supervised dataset from PHH hand-history files.

Sibling of ``tools.build_phase_a_dataset_v9`` — the Arena builder. This
module owns ONE mechanism: turning PHH files (``.phh`` / ``.phhs``)
into the same schema-4 supervised rows the Arena builder produces, via
the ``tools.phh_replay`` adapter, and sinking them through the shared
``PhaseARowSink``. Row semantics, labels, per-decision seeding and the
sidecar contract are the Arena builder's, unchanged:
``replay_rows_v9`` is imported verbatim, so the per-decision seed stays
``sha256(seed:table_id:sequence)`` — a PHH table id
(``phh/pluribus/<session>/<n>``, set by the adapter) gets exactly the
Arena discipline.

The differences are provenance, not mechanics. Each root is converted
through ``tools.phh_replay.replays_from_root`` — the adapter's designed
entry point for this tool: it walks the root for ``*.phh`` / ``*.phhs``
in sorted order and yields ``(table_id, replay)`` pairs with the
root-scoped ids (``phh/pluribus/<session>/<n>``; per-file
``replays_from_path`` ids are file-stem-scoped and would collide
between sessions, so this tool does not use it). Refusals ride in a
``RefusalCounter`` the worker owns and reports by reason (non-``NT``
variants, antes, straddles, null stacks, bad blinds, parse errors).
The sidecar's ``generator`` block records ``source = "phh"``, the
dataset git commit (``git -C phh-dataset rev-parse HEAD``, ``unknown``
on failure), the roots, the adapter version, the refusal counters by
reason, ``skipped_decisions``, ``timeout_actions`` and the label
coverage. Refused hands produce no rows and are counted; the sink's
per-table dedupe keys on the hand's table id, so a ``.phhs`` file
holding many hands dedupes per hand, not per file.

``--workers`` shards the walk PER FILE. It used to build one work item
per root, which made the flag a no-op for the documented one-root run
(`PROCEDURES.md` section 16) -- measured 2026-09-03: ``--workers 12``
and ``--workers 1`` were within 0.6 s of each other on 20 hands. Ids
come from ``tools.phh_replay.root_table_base``, the same rule
``replays_from_root`` applies, so the per-decision seed and therefore
the dataset bytes are identical at any worker count -- pinned by
``tests/test_build_phase_a_dataset_phh.py``. Do NOT shard by passing
per-session ``--roots`` instead: that changes the id prefix from
``phh/pluribus/<session>/<n>`` to ``phh/<session>/<n>`` and silently
rewrites every seed. ``--limit`` caps hands PER ROOT, which a per-file
shard cannot enforce without coordinating workers, so setting it falls
back to the per-root path (it is a smoke-test flag; the production run
does not pass it). The shard is invisible in the output: the
dataset bytes are unchanged, and so is the sidecar, because the sink is
handed the ROOT as each shard's collection name and aggregates them.
Keying it per file instead cost 813 KB of ``{'hands': 1}`` entries in a
tracked record on the first 10,000-hand build -- 99.7% of the file.

Offline and read-only over ``phh-dataset/``; writes only the dataset
and its sidecar under ``artifacts/phase_a_v9/``. No Arena requests, no
credentials, no promotion.

Usage:
    python -m tools.build_phase_a_dataset_phh
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engine.feature_extract_v8 import _EQUITY_TRIALS
from tools.build_phase_a_dataset import (
    DEFAULT_EQUITY_TRIALS,
    DEFAULT_POTENTIAL_TRIALS,
    DEFAULT_SEED,
)
from tools.build_phase_a_dataset_v9 import (
    PhaseARowSink,
    _belief_provider,
    _consume_files,
    _print_summary_v9,
    replay_rows_v9,
)

DEFAULT_ROOTS_PHH = (Path("phh-dataset") / "data" / "pluribus",)
DEFAULT_OUTPUT_PHH = (
    Path("artifacts")
    / "phase_a_v9"
    / "phase-a-dataset-v9-pluribus.jsonl.gz"
)

#: The dataset's git root (a junction to ``D:\\phh-dataset``); the
#: commit the rows were built from is recorded in every sidecar.
_DATASET_GIT_ROOT = Path("phh-dataset")
_PHH_SUFFIXES = (".phh", ".phhs")


def _dataset_commit() -> str:
    """The ``phh-dataset`` commit the rows were built from, or ``unknown``."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(_DATASET_GIT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return "unknown"


def _phh_replay_version() -> str:
    """The adapter's own version stamp, or ``unknown``."""

    from tools.phh_replay import PHH_REPLAY_VERSION

    return str(PHH_REPLAY_VERSION)


def _phh_files(root: Path) -> list[Path]:
    """The root's ``*.phh`` / ``*.phhs`` files, the adapter's walk order.

    Kept here so the parent process can build the per-file work list
    without importing the adapter (and therefore pokerkit); it MUST
    stay equal to ``tools.phh_replay.root_files``, which
    ``tests/test_build_phase_a_dataset_phh.py`` pins.
    """
    found: set[Path] = set()
    for suffix in _PHH_SUFFIXES:
        found.update(root.rglob("*" + suffix))
    return sorted(found)


def _process_phh_root(
    args: tuple[str, int, int, int, int | None],
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Worker: one root's PHH hands to v9 rows (importable for pickling).

    The adapter is imported lazily (it needs pokerkit) and its
    ``replays_from_root`` yields ``(table_id, replay)`` pairs with the
    root-scoped ids; refused hands land in the ``RefusalCounter``.
    ``limit`` caps the hands read from this root.
    """

    root_text, seed, equity_trials, potential_trials, limit = args
    from tools.phh_replay import RefusalCounter, replays_from_root

    stats: Counter[str] = Counter()
    counter = RefusalCounter()
    rows: list[dict[str, Any]] = []
    try:
        pairs = replays_from_root(root_text, refusals=counter)
        if limit is not None:
            pairs = itertools.islice(pairs, limit)
        for table_id, replay in pairs:
            stats["hands"] += 1
            hand_rows, replay_stats = replay_rows_v9(
                replay,
                seed=seed,
                equity_trials=equity_trials,
                potential_trials=potential_trials,
            )
            rows.extend(hand_rows)
            stats.update(replay_stats)
    except (OSError, ValueError) as error:
        stats["unreadable_roots"] += 1
        stats[f"skip:{type(error).__name__}"] += 1
        return root_text, [], dict(stats)
    for reason, count in counter.counts.items():
        stats[f"refusal:{reason}"] += count
    return root_text, rows, dict(stats)


def _process_phh_file(
    args: tuple[str, str, int, int, int],
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Worker: one PHH file's hands to v9 rows (importable for pickling).

    The per-file shard of :func:`_process_phh_root`. Ids come from the
    adapter's ``replays_from_file_in_root``, i.e. the root-scoped ids
    ``replays_from_root`` would give this file, so the per-decision seed
    ``sha256(seed:table_id:sequence)`` is unchanged and sharding moves
    no dataset byte. Refusals ride in this worker's own counter and are
    summed by the sink.
    """

    root_text, path_text, seed, equity_trials, potential_trials = args
    from tools.phh_replay import RefusalCounter, replays_from_file_in_root

    stats: Counter[str] = Counter()
    counter = RefusalCounter()
    rows: list[dict[str, Any]] = []
    try:
        for _table_id, replay in replays_from_file_in_root(
            root_text, path_text, refusals=counter
        ):
            stats["hands"] += 1
            hand_rows, replay_stats = replay_rows_v9(
                replay,
                seed=seed,
                equity_trials=equity_trials,
                potential_trials=potential_trials,
            )
            rows.extend(hand_rows)
            stats.update(replay_stats)
    except (OSError, ValueError) as error:
        stats["unreadable_files"] += 1
        stats[f"skip:{type(error).__name__}"] += 1
        return path_text, [], dict(stats)
    for reason, count in counter.counts.items():
        stats[f"refusal:{reason}"] += count
    return path_text, rows, dict(stats)


def build_phase_a_dataset_phh(
    roots: Sequence[Path],
    output: Path,
    *,
    seed: int = DEFAULT_SEED,
    equity_trials: int = DEFAULT_EQUITY_TRIALS,
    potential_trials: int = DEFAULT_POTENTIAL_TRIALS,
    workers: int = 1,
    limit: int | None = None,
    chunk_rows: int = 50_000,
) -> dict[str, Any]:
    """Build the v9 dataset and sidecar from PHH files; return the sidecar.

    Each root is converted through the adapter's ``replays_from_root``
    and its rows fed to the shared ``PhaseARowSink``, so chunking,
    per-table dedupe, the k-way merge, byte-deterministic gzip, the
    atomic replace, the sidecar and the trainer self-load are the Arena
    builder's mechanics. Roots that do not exist or hold no PHH files
    are skipped with a warning. ``limit`` caps the hands read from each
    root (files for single-hand-per-file datasets like Pluribus).
    """

    usable: list[str] = []
    file_counts: dict[str, int] = {}
    skipped_roots: list[tuple[str, str]] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            skipped_roots.append((str(root), "no such directory"))
            continue
        files = _phh_files(root_path)
        if not files:
            skipped_roots.append((str(root), "no .phh or .phhs files"))
            continue
        usable.append(str(root))
        file_counts[str(root)] = len(files)
    if not usable:
        raise ValueError("no .phh or .phhs files were found (no readable roots)")
    for root, reason in skipped_roots:
        print(f"skipping root {root}: {reason}", flush=True)

    started = time.monotonic()
    sink = PhaseARowSink(output, dedupe_tables=True, chunk_rows=chunk_rows)
    if limit is None:
        shards = [
            (root, str(path))
            for root in usable
            for path in _phh_files(Path(root))
        ]
        _consume_files(
            sink,
            [
                (root, path, seed, equity_trials, potential_trials)
                for root, path in shards
            ],
            # The collection name is the ROOT, not the shard: the sink
            # aggregates repeated names and de-duplicates them for the
            # sidecar, so per-collection coverage stays keyed the way
            # the Arena builder keys it and the shard leaves no trace
            # in the record -- just as it leaves none in the bytes.
            [root for root, _path in shards],
            _process_phh_file,
            workers=workers,
            noun="files",
        )
    else:
        # --limit caps hands per ROOT; a per-file shard cannot enforce
        # that without coordinating workers, so keep the per-root path.
        _consume_files(
            sink,
            [
                (root, seed, equity_trials, potential_trials, limit)
                for root in usable
            ],
            usable,
            _process_phh_root,
            workers=workers,
            noun="roots",
        )

    combined = sink.combined_stats
    refusals = {
        key.removeprefix("refusal:"): value
        for key, value in sorted(combined.items())
        if key.startswith("refusal:")
    }
    if limit is not None:
        file_count = sum(min(count, limit) for count in file_counts.values())
    else:
        file_count = sum(file_counts.values())
    sidecar_document = sink.finish(
        generator={
            "tool": "tools.build_phase_a_dataset_phh",
            "source": "phh",
            "seed": seed,
            "equity_trials": equity_trials,
            "potential_trials": potential_trials,
            "read_equity_trials": _EQUITY_TRIALS,
            "belief_provider": "P3BeliefProvider",
            "belief_fit_source": _belief_provider().fit_source,
            "dataset_commit": _dataset_commit(),
            "roots": [str(root) for root in roots],
            "limit": limit,
            "phh_replay_version": _phh_replay_version(),
            "refusals": refusals,
            "skipped_decisions": combined.get("skipped_decisions", 0),
            "timeout_actions": combined.get("timeout_actions", 0),
        },
        skipped_roots=dict(skipped_roots),
        file_count=file_count,
        counts_extra={"hands": combined.get("hands", 0)},
        label_coverage_in_generator=True,
    )

    _print_summary_v9(sink.coverage, sink.totals)
    elapsed = time.monotonic() - started
    print(f"\nwrote {sink.output} ({sink.row_count} rows) in {elapsed:.0f}s")
    print(f"trainer loader accepted the dataset ({sink.row_count} rows)")
    if sink.duplicate_rows_dropped:
        print(f"dropped {sink.duplicate_rows_dropped} duplicate table rows")
    if refusals:
        print(f"refused hands by reason: {refusals}")
    print(f"wrote {sink.sidecar_path}")
    return sidecar_document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the v9 Phase-A supervised dataset from PHH files."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[str(root) for root in DEFAULT_ROOTS_PHH],
        help="directories walked recursively for *.phh and *.phhs",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PHH))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--equity-trials", type=int, default=DEFAULT_EQUITY_TRIALS
    )
    parser.add_argument(
        "--potential-trials", type=int, default=DEFAULT_POTENTIAL_TRIALS
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "only process the first N hands per root (files for "
            "single-hand-per-file datasets like Pluribus)"
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
    build_phase_a_dataset_phh(
        [Path(root) for root in args.roots],
        Path(args.output),
        seed=args.seed,
        equity_trials=args.equity_trials,
        potential_trials=args.potential_trials,
        workers=args.workers,
        limit=args.limit,
        chunk_rows=args.chunk_rows,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
