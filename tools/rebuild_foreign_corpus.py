"""Complete the foreign corpus without rewriting a frozen collection.

The archive holds replays that were inventoried but never downloaded, and
downloaded replays whose decisions never reached a CSV. This tool resumes
those downloads and derives only the missing rows into a sibling *delta*
CSV that shares the frozen file's exact header. Frozen CSVs, manifests, and
candidate reports are never modified, satisfying the rule that a data
rebuild creates a new corpus instead of rewritten history.

Delta rows are selected per ``(agent_id, table_id)`` receipt pair, so a
table that is already in the frozen CSV still contributes rows for agents
whose membership was only inventoried later.

Phases are explicit. ``--derive`` is offline. ``--fetch`` is the only path
that contacts Arena, and it reads public replay endpoints only.

Example:
    python -m tools.rebuild_foreign_corpus --season 9 --season 11 --fetch
    python -m tools.rebuild_foreign_corpus --derive
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from devfun_poker_playground.decision_engine import DecisionEngine
from tools.collect_foreign_play_data import (
    _decision_row,
    _fetch_replays,
    _read_json,
    _receipt_from_table,
    _sha256,
    _unwrap_rpc,
    _write_json,
)

DECISION_EVENT_TYPES = {"ActionTaken", "TimeoutAction"}
DELTA_CSV_SUFFIX = ".delta.csv"


def _agent_id_from_dir(agent_dir: Path) -> str:
    return agent_dir.name.split("-", 1)[-1]


def cached_receipts(
    raw: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, bool]]:
    """Rebuild the collector's receipt map from cached history pages only.

    Also reports, per agent, whether the cached history is the exhausted
    pagination or only the old at-most-50 receipt sample. Sampling depth
    decides eligibility for scope expansion: a 50-hand agent mixed into a
    full-history season would over-represent its most recent hands.
    """

    receipts_by_agent: dict[str, dict[str, dict[str, Any]]] = {}
    paginated: dict[str, bool] = {}
    agents_dir = raw / "agents"
    if not agents_dir.is_dir():
        return receipts_by_agent, paginated
    for agent_dir in sorted(path for path in agents_dir.iterdir() if path.is_dir()):
        agent_id = _agent_id_from_dir(agent_dir)
        receipts: dict[str, dict[str, Any]] = {}
        pages = sorted(agent_dir.glob("tables-*.json"))
        paginated[agent_id] = bool(pages)
        if pages:
            for page_path in pages:
                page = _unwrap_rpc(_read_json(page_path))
                if not isinstance(page, dict):
                    continue
                for table in page.get("tables") or []:
                    if (
                        not isinstance(table, dict)
                        or table.get("status") != "Completed"
                    ):
                        continue
                    try:
                        receipt = _receipt_from_table(table, agent_id)
                    except (KeyError, TypeError, ValueError):
                        continue
                    receipts[receipt["tableId"]] = receipt
        else:
            recent = agent_dir / "receipts.json"
            if recent.exists():
                for receipt in _read_json(recent) or []:
                    if isinstance(receipt, dict) and isinstance(
                        receipt.get("tableId"), str
                    ):
                        receipts[receipt["tableId"]] = receipt
        receipts_by_agent[agent_id] = receipts
    return receipts_by_agent, paginated


def archived_agents(
    raw: Path, known: dict[str, Any], paginated: dict[str, bool]
) -> dict[str, Any]:
    """Resolve fully-paginated agent folders that the manifest never listed.

    Seasons 11 and 13 were frozen at the top 10 while the archive kept the
    exhausted history of the top 15. Those extra agents are the same public
    teacher material; their rank comes from the folder prefix the collector
    wrote, and their scores from the cached leaderboard snapshot.
    """

    leaderboard_path = raw / "leaderboard.json"
    if not leaderboard_path.exists():
        return {}
    leaderboard = _unwrap_rpc(_read_json(leaderboard_path))
    if not isinstance(leaderboard, dict):
        return {}
    by_id = {
        entry["id"]: entry
        for entry in leaderboard.get("agents") or []
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    resolved: dict[str, Any] = {}
    for agent_dir in sorted(
        path for path in (raw / "agents").iterdir() if path.is_dir()
    ):
        agent_id = _agent_id_from_dir(agent_dir)
        if agent_id in known or not paginated.get(agent_id):
            continue
        entry = by_id.get(agent_id)
        if entry is None or not entry.get("claimed"):
            continue
        rank_prefix = agent_dir.name.split("-", 1)[0]
        resolved[agent_id] = {
            "rank": int(rank_prefix) if rank_prefix.isdigit() else 0,
            "id": agent_id,
            "name": entry.get("name"),
            "totalScore": entry.get("totalScore"),
            "totalSubmissions": entry.get("totalSubmissions"),
        }
    return resolved


def frozen_pairs(csv_path: Path) -> tuple[set[tuple[str, str]], list[str]]:
    """Return the derived ``(agent_id, table_id)`` pairs and the frozen header."""

    pairs: set[tuple[str, str]] = set()
    with csv_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        header = list(reader.fieldnames or ())
        for row in reader:
            pairs.add((row.get("agent_id") or "", row.get("table_id") or ""))
    return pairs, header


class Collection:
    """One season folder: frozen outputs plus everything cached under raw/."""

    def __init__(self, directory: Path, *, agent_scope: str = "manifest") -> None:
        self.directory = directory
        self.manifest = _read_json(directory / "manifest.json")
        self.raw = directory / "raw"
        scope = self.manifest.get("scope") or {}
        self.season = scope.get("season_number")
        self.arena = scope.get("arena_slug") or "poker-playground"
        self.agents = {
            agent["id"]: agent for agent in self.manifest.get("agents") or []
        }
        files = self.manifest.get("files") or {}
        self.csv_name = files.get("decision_csv") or "top15_decisions.csv"
        self.csv_path = directory / self.csv_name
        self.receipts_by_agent, paginated = cached_receipts(self.raw)
        self.added_agents: dict[str, Any] = {}
        if agent_scope == "archive":
            self.added_agents = archived_agents(self.raw, self.agents, paginated)
            self.agents.update(self.added_agents)
        # Everything downstream -- inventory, fetch targets, delta rows --
        # stays inside the selected agent scope.
        self.receipts_by_agent = {
            agent_id: receipts
            for agent_id, receipts in self.receipts_by_agent.items()
            if agent_id in self.agents
        }
        self.pairs = {
            (agent_id, table_id)
            for agent_id, receipts in self.receipts_by_agent.items()
            for table_id in receipts
        }
        self.inventory = {table_id for _, table_id in self.pairs}
        self.downloaded = {
            path.stem for path in (self.raw / "tables").glob("*.json") if path.is_file()
        }
        self.derived_pairs, self.header = frozen_pairs(self.csv_path)

    @property
    def delta_csv(self) -> Path:
        return self.directory / (self.csv_name.removesuffix(".csv") + DELTA_CSV_SUFFIX)

    @property
    def delta_manifest(self) -> Path:
        return self.directory / "manifest.delta.json"

    def missing_replays(self) -> set[str]:
        return self.inventory - self.downloaded

    def pending_pairs(self) -> set[tuple[str, str]]:
        """Receipt pairs that are downloadable now and not yet in the CSV."""

        return {
            pair
            for pair in self.pairs - self.derived_pairs
            if pair[1] in self.downloaded and pair[0] in self.agents
        }


def fetch_missing(
    collection: Collection, workers: int, dry_run: bool
) -> dict[str, Any]:
    """Download inventoried replays whose bodies are absent (network)."""

    missing = collection.missing_replays()
    print(
        f"season {collection.season}: {len(collection.inventory)} inventoried, "
        f"{len(collection.downloaded)} on disk, {len(missing)} to fetch",
        flush=True,
    )
    if dry_run or not missing:
        return {"requested": len(missing), "fetched": 0, "dry_run": dry_run}
    started = time.monotonic()
    _fetch_replays(collection.raw, collection.inventory, workers)
    on_disk_now = {
        path.stem
        for path in (collection.raw / "tables").glob("*.json")
        if path.is_file()
    }
    fetched = len(on_disk_now - collection.downloaded)
    collection.downloaded = on_disk_now
    return {
        "requested": len(missing),
        "fetched": fetched,
        "still_missing": len(collection.inventory - on_disk_now),
        "seconds": round(time.monotonic() - started, 1),
    }


def derive_delta(collection: Collection, dry_run: bool) -> dict[str, Any]:
    """Write the rows that the frozen CSV never received (offline)."""

    pending = collection.pending_pairs()
    by_table: dict[str, set[str]] = {}
    for agent_id, table_id in pending:
        by_table.setdefault(table_id, set()).add(agent_id)
    print(
        f"season {collection.season}: {len(pending)} pending receipt pairs "
        f"across {len(by_table)} replays",
        flush=True,
    )
    if dry_run or not by_table:
        return {
            "pending_pairs": len(pending),
            "pending_tables": len(by_table),
            "rows": 0,
            "dry_run": dry_run,
        }

    engine = DecisionEngine(equity_trials=100, seed=7)
    feature_columns = [
        name for name in collection.header if name.startswith("feature_")
    ]
    skipped: list[dict[str, object]] = []
    represented: set[str] = set()
    rows = 0
    eligible = 0
    timeouts = 0
    started = time.monotonic()
    temporary = collection.delta_csv.with_suffix(collection.delta_csv.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=collection.header, extrasaction="ignore"
        )
        writer.writeheader()
        for completed, table_id in enumerate(sorted(by_table), 1):
            replay = _unwrap_rpc(
                _read_json(collection.raw / "tables" / f"{table_id}.json")
            )
            if not isinstance(replay, dict):
                skipped.append(
                    {"table_id": table_id, "error": "replay is not an object"}
                )
                continue
            wanted_agents = by_table[table_id]
            for event in replay.get("events") or []:
                agent_id = event.get("agentId")
                if event.get("type") not in DECISION_EVENT_TYPES:
                    continue
                if agent_id not in wanted_agents:
                    continue
                receipt = collection.receipts_by_agent.get(agent_id, {}).get(table_id)
                if receipt is None:
                    continue
                try:
                    row = _decision_row(
                        replay,
                        event,
                        collection.agents[agent_id],
                        receipt,
                        engine,
                        arena_slug=collection.arena,
                        season_number=collection.season,
                    )
                    missing_columns = [
                        column for column in collection.header if column not in row
                    ]
                    if missing_columns:
                        raise ValueError(
                            f"derived row lacks frozen columns: {missing_columns[:3]}"
                        )
                    if any(
                        not math.isfinite(float(row[column]))
                        for column in feature_columns
                    ):
                        raise ValueError("decision contains a non-finite model feature")
                    writer.writerow(row)
                    rows += 1
                    eligible += row["teacher_eligible"] is True
                    timeouts += row["event_type"] == "TimeoutAction"
                    represented.add(agent_id)
                except (KeyError, TypeError, ValueError) as error:
                    skipped.append(
                        {
                            "table_id": table_id,
                            "event_sequence": event.get("sequence"),
                            "agent_id": agent_id,
                            "error": str(error),
                        }
                    )
            if completed % 500 == 0 or completed == len(by_table):
                rate = completed / max(1e-9, time.monotonic() - started)
                print(
                    f"   derived {completed}/{len(by_table)} replays; rows {rows}; "
                    f"{rate:.1f} replays/s",
                    flush=True,
                )
    if rows == 0:
        temporary.unlink(missing_ok=True)
        return {
            "pending_pairs": len(pending),
            "pending_tables": len(by_table),
            "rows": 0,
            "skipped": len(skipped),
        }
    temporary.replace(collection.delta_csv)
    _write_json(collection.directory / "skipped_decisions.delta.json", skipped)

    summary = {
        "pending_pairs": len(pending),
        "pending_tables": len(by_table),
        "rows": rows,
        "eligible_rows": eligible,
        "timeout_rows": timeouts,
        "skipped": len(skipped),
        "represented_agents": len(represented),
        "seconds": round(time.monotonic() - started, 1),
    }
    _write_json(
        collection.delta_manifest,
        {
            "schema_version": 1,
            "kind": "delta",
            "derived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "season_number": collection.season,
            "arena_slug": collection.arena,
            "parent": {
                "collection": collection.directory.name,
                "decision_csv": collection.csv_name,
                "decision_csv_sha256": (collection.manifest.get("files") or {}).get(
                    "decision_csv_sha256"
                ),
                "note": (
                    "The parent CSV is frozen. This delta holds only rows whose "
                    "(agent_id, table_id) receipt pair was absent from it, so the "
                    "two files are disjoint and load together without duplicates."
                ),
            },
            "counts": {
                **summary,
                "inventory_tables": len(collection.inventory),
                "downloaded_replays": len(collection.downloaded),
                "still_missing_replays": len(collection.missing_replays()),
            },
            "agent_scope": {
                "agents_in_scope": len(collection.agents),
                "manifest_agents": len(collection.manifest.get("agents") or []),
                "added_from_archive": [
                    {"rank": agent["rank"], "id": agent["id"], "name": agent["name"]}
                    for agent in sorted(
                        collection.added_agents.values(), key=lambda a: a["rank"]
                    )
                ],
            },
            "files": {
                "decision_csv": collection.delta_csv.name,
                "decision_csv_sha256": _sha256(collection.delta_csv),
                "skipped_decisions": "skipped_decisions.delta.json",
            },
            "model_features": {
                "count": len(feature_columns),
                "header_source": collection.csv_name,
                "equity_trials": 100,
                "equity_seed": 7,
            },
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="foreign play data/last 5 seasons top 15")
    parser.add_argument(
        "--season",
        type=int,
        action="append",
        help="restrict to a season number; repeatable (default: every collection)",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download inventoried replays that are missing (contacts Arena)",
    )
    parser.add_argument(
        "--derive", action="store_true", help="write delta CSVs from cached replays"
    )
    parser.add_argument(
        "--agent-scope",
        choices=("manifest", "archive"),
        default="manifest",
        help=(
            "manifest: only the agents the frozen manifest listed. "
            "archive: also include agent folders whose full paginated history "
            "is cached but which the manifest omitted (season 11 ranks 11-15). "
            "Agents holding only the old 50-hand receipt sample are never added."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what each phase would do and change nothing",
    )
    parser.add_argument("--report", help="optional JSON summary path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.fetch and not args.derive:
        raise SystemExit("choose at least one phase: --fetch and/or --derive")
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be 1-8")
    root = Path(args.root)
    directories = sorted(
        path for path in root.iterdir() if (path / "manifest.json").is_file()
    )
    summaries: list[dict[str, Any]] = []
    for directory in directories:
        collection = Collection(directory, agent_scope=args.agent_scope)
        if args.season and collection.season not in args.season:
            continue
        summary: dict[str, Any] = {
            "collection": directory.name,
            "season": collection.season,
            "agent_scope": args.agent_scope,
            "agents_in_scope": len(collection.agents),
            "agents_added_from_archive": sorted(collection.added_agents),
        }
        if args.fetch:
            summary["fetch"] = fetch_missing(collection, args.workers, args.dry_run)
        if args.derive:
            summary["derive"] = derive_delta(collection, args.dry_run)
        summaries.append(summary)
    print(json.dumps(summaries, indent=2))
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
