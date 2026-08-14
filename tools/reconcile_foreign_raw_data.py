"""Reconcile the raw foreign archive against its derived CSV manifests.

Answers the mandatory stale-data question before a new training corpus is
frozen: which inventoried replays were never downloaded, which downloaded
replays are outside the current CSV, and which of those actually carry a
selected-agent decision that a rebuild would turn into new rows.

Read-only and offline. It makes no network request, writes no CSV, and
never edits an existing manifest.

Example:
    python -m tools.reconcile_foreign_raw_data \
        --report .handoff/notes/foreign_reconciliation.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DECISION_EVENT_TYPES = {"ActionTaken", "TimeoutAction"}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _unwrap_rpc(document: Any) -> Any:
    try:
        return document["result"]["data"]["json"]
    except (KeyError, TypeError, IndexError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _agent_directories(raw: Path) -> list[Path]:
    agents = raw / "agents"
    if not agents.is_dir():
        return []
    return sorted(path for path in agents.iterdir() if path.is_dir())


def _inventory_from_pages(
    agent_dir: Path, agent_id: str
) -> tuple[set[str], int, dict[str, Any]]:
    """Replay the cached pagination for one agent exactly as the collector did."""

    tables: set[str] = set()
    memberships = 0
    cursor = 0
    pages = 0
    truncated_at: int | None = None
    while True:
        path = agent_dir / f"tables-{cursor:06d}.json"
        if not path.exists():
            truncated_at = cursor
            break
        page = _unwrap_rpc(_read_json(path))
        if not isinstance(page, dict) or not isinstance(page.get("tables"), list):
            truncated_at = cursor
            break
        pages += 1
        for table in page["tables"]:
            if not isinstance(table, dict) or table.get("status") != "Completed":
                continue
            table_id = table.get("id")
            if not isinstance(table_id, str) or not table_id:
                continue
            seat = next(
                (
                    value
                    for value in table.get("seats") or []
                    if isinstance(value, dict) and value.get("agentId") == agent_id
                ),
                None,
            )
            if seat is None or not isinstance(seat.get("chipDelta"), int):
                continue
            memberships += 1
            tables.add(table_id)
        next_cursor = page.get("nextCursor")
        if next_cursor is None:
            break
        if not isinstance(next_cursor, int) or next_cursor <= cursor:
            truncated_at = cursor
            break
        cursor = next_cursor
    return (
        tables,
        memberships,
        {"pages": pages, "exhausted": truncated_at is None, "stopped_at": truncated_at},
    )


def _inventory_from_receipts(agent_dir: Path) -> tuple[set[str], int]:
    path = agent_dir / "receipts.json"
    if not path.exists():
        return set(), 0
    receipts = _read_json(path)
    if not isinstance(receipts, list):
        return set(), 0
    tables = {
        receipt["tableId"]
        for receipt in receipts
        if isinstance(receipt, dict) and isinstance(receipt.get("tableId"), str)
    }
    return tables, len(receipts)


def collection_inventory(raw: Path) -> dict[str, Any]:
    """Union the per-agent table inventory cached under ``raw/agents``."""

    per_agent: list[dict[str, Any]] = []
    inventory: set[str] = set()
    ownership: dict[str, set[str]] = {}
    memberships = 0
    modes: Counter[str] = Counter()
    for agent_dir in _agent_directories(raw):
        agent_id = agent_dir.name.split("-", 1)[-1]
        if list(agent_dir.glob("tables-*.json")):
            tables, count, pagination = _inventory_from_pages(agent_dir, agent_id)
            modes["all_paginated_tables"] += 1
        else:
            tables, count = _inventory_from_receipts(agent_dir)
            pagination = {"pages": 0, "exhausted": True, "stopped_at": None}
            modes["recent"] += 1
        memberships += count
        inventory |= tables
        ownership[agent_id] = tables
        per_agent.append(
            {
                "agent_dir": agent_dir.name,
                "agent_id": agent_id,
                "tables": len(tables),
                "memberships": count,
                **pagination,
            }
        )
    return {
        "mode": modes.most_common(1)[0][0] if modes else "none",
        "agents": per_agent,
        "table_ids": inventory,
        "ownership": ownership,
        "memberships": memberships,
        "incomplete_agents": [
            entry["agent_dir"] for entry in per_agent if not entry["exhausted"]
        ],
    }


def csv_summary(path: Path) -> dict[str, Any]:
    """Row, eligibility, duplicate, and coverage facts for one decision CSV."""

    tables: set[str] = set()
    keys: Counter[tuple[str, str, str, str]] = Counter()
    rows = 0
    eligible = 0
    eligible_keys: set[tuple[str, str, str, str]] = set()
    agents: set[str] = set()
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            rows += 1
            table_id = row.get("table_id") or ""
            tables.add(table_id)
            agents.add(row.get("agent_id") or "")
            key = (
                row.get("season_number") or "",
                row.get("agent_id") or "",
                table_id,
                row.get("event_sequence") or "",
            )
            keys[key] += 1
            if str(row.get("teacher_eligible")) == "True":
                eligible += 1
                eligible_keys.add(key)
    return {
        "rows": rows,
        "eligible_rows": eligible,
        "unique_tables": len(tables),
        "table_ids": tables,
        "duplicate_keys": sum(count - 1 for count in keys.values() if count > 1),
        "eligible_keys": eligible_keys,
        "represented_agents": len(agents - {""}),
    }


def decision_agents(path: Path) -> set[str] | str:
    """Agents that acted in one replay, or a string naming why it is unusable."""

    try:
        document = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return "unreadable"
    replay = _unwrap_rpc(document)
    if not isinstance(replay, dict):
        return "unexpected_shape"
    events = replay.get("events")
    if not isinstance(events, list):
        return "unexpected_shape"
    return {
        event["agentId"]
        for event in events
        if isinstance(event, dict)
        and event.get("type") in DECISION_EVENT_TYPES
        and isinstance(event.get("agentId"), str)
    }


def classify_replay(path: Path, receipt_agents: set[str]) -> str:
    """Say whether an on-disk replay would yield selected-agent decision rows."""

    actors = decision_agents(path)
    if isinstance(actors, str):
        return actors
    if actors & receipt_agents:
        return "has_selected_agent_decision"
    return "no_selected_agent_decision"


def reconcile_collection(directory: Path, *, deep: bool) -> dict[str, Any]:
    """Compare one collection's manifest, raw inventory, replays, and CSV."""

    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    raw = directory / "raw"
    csv_name = (manifest.get("files") or {}).get(
        "decision_csv"
    ) or "top15_decisions.csv"
    csv_path = directory / csv_name

    inventory = collection_inventory(raw)
    downloaded = {
        path.stem for path in (raw / "tables").glob("*.json") if path.is_file()
    }
    summary = csv_summary(csv_path) if csv_path.exists() else None

    # A completed collection is the frozen CSV plus its delta. Auditing only
    # the frozen half would report every delta-derived table as a gap.
    delta_path = directory / (csv_name.removesuffix(".csv") + ".delta.csv")
    delta_manifest_path = directory / "manifest.delta.json"
    delta = csv_summary(delta_path) if delta_path.exists() else None
    delta_manifest = (
        _read_json(delta_manifest_path) if delta_manifest_path.exists() else None
    )

    inventoried = inventory["table_ids"]
    csv_tables = (summary["table_ids"] if summary else set()) | (
        delta["table_ids"] if delta else set()
    )
    missing_replays = inventoried - downloaded
    orphan_replays = downloaded - inventoried
    covered = downloaded & inventoried
    underived = covered - csv_tables
    csv_without_replay = csv_tables - downloaded

    owner_of: dict[str, set[str]] = {}
    for agent_id, tables in inventory["ownership"].items():
        for table_id in tables:
            owner_of.setdefault(table_id, set()).add(agent_id)

    # A collection's corpus covers the agents its manifest lists. The archive
    # can hold more: season 11 kept the exhausted history of five agents the
    # manifest omitted. Counting those as derivable without saying so reads
    # as "the CSV is missing rows" when the truth is "the scope was narrower".
    manifest_agent_ids = {
        agent["id"] for agent in manifest.get("agents") or [] if isinstance(agent, dict)
    }
    paginated_ids = {
        entry["agent_id"] for entry in inventory["agents"] if entry["pages"] > 0
    }
    sampled_ids = {
        entry["agent_id"] for entry in inventory["agents"] if entry["pages"] == 0
    }

    scan_targets = sorted(underived)
    if not deep:
        scan_targets = scan_targets[: min(len(scan_targets), 5000)]
    classes: Counter[str] = Counter()
    derivable: list[str] = []
    archive_only_derivable = 0
    for table_id in scan_targets:
        owners = owner_of.get(table_id, set())
        actors = decision_agents(raw / "tables" / f"{table_id}.json")
        if isinstance(actors, str):
            classes[actors] += 1
            continue
        acting_owners = actors & owners
        if acting_owners & manifest_agent_ids:
            classes["has_selected_agent_decision"] += 1
            derivable.append(table_id)
        elif acting_owners & paginated_ids:
            classes["decision_by_archive_only_agent"] += 1
            archive_only_derivable += 1
        else:
            classes["no_selected_agent_decision"] += 1

    orphan_classes: Counter[str] = Counter()
    for table_id in sorted(orphan_replays)[:2000]:
        orphan_classes[
            classify_replay(raw / "tables" / f"{table_id}.json", set(owner_of))
        ] += 1

    recorded_sha = (manifest.get("files") or {}).get("decision_csv_sha256")
    actual_sha = _sha256(csv_path) if csv_path.exists() else None

    return {
        "collection": directory.name,
        "season": (manifest.get("scope") or {}).get("season_number"),
        "season_status": (manifest.get("scope") or {}).get("season_status"),
        "manifest_history_mode": (manifest.get("scope") or {}).get("history_mode"),
        "disk_history_mode": inventory["mode"],
        "manifest_counts": manifest.get("counts") or {},
        "inventory": {
            "memberships": inventory["memberships"],
            "unique_tables": len(inventoried),
            "incomplete_agents": inventory["incomplete_agents"],
        },
        "agent_scope": {
            "manifest_agents": len(manifest_agent_ids),
            "agent_folders": len(inventory["agents"]),
            "full_history_outside_manifest": sorted(paginated_ids - manifest_agent_ids),
            "sampled_only_outside_manifest": sorted(sampled_ids - manifest_agent_ids),
        },
        "replays_on_disk": len(downloaded),
        "csv": {
            "name": csv_name,
            "rows": summary["rows"] if summary else 0,
            "eligible_rows": summary["eligible_rows"] if summary else 0,
            "unique_tables": summary["unique_tables"] if summary else 0,
            "duplicate_keys": summary["duplicate_keys"] if summary else 0,
            "represented_agents": summary["represented_agents"] if summary else 0,
            "sha256_recorded": recorded_sha,
            "sha256_actual": actual_sha,
            "sha256_matches": recorded_sha == actual_sha,
        },
        "delta": (
            {
                "name": delta_path.name,
                "rows": delta["rows"],
                "eligible_rows": delta["eligible_rows"],
                "unique_tables": delta["unique_tables"],
                "duplicate_keys": delta["duplicate_keys"],
                "represented_agents": delta["represented_agents"],
                "sha256_recorded": (
                    ((delta_manifest or {}).get("files") or {}).get(
                        "decision_csv_sha256"
                    )
                ),
                "sha256_actual": _sha256(delta_path),
                "sha256_matches": (
                    ((delta_manifest or {}).get("files") or {}).get(
                        "decision_csv_sha256"
                    )
                    == _sha256(delta_path)
                ),
                "keys_shared_with_frozen": len(
                    (summary["eligible_keys"] if summary else set())
                    & delta["eligible_keys"]
                ),
            }
            if delta
            else None
        ),
        "corpus_eligible_rows": (summary["eligible_rows"] if summary else 0)
        + (delta["eligible_rows"] if delta else 0),
        "gaps": {
            "inventoried_missing_replay": len(missing_replays),
            "downloaded_not_inventoried": len(orphan_replays),
            "downloaded_not_in_csv": len(underived),
            "csv_table_without_replay": len(csv_without_replay),
        },
        "underived_classification": dict(classes),
        "archive_only_derivable_tables": archive_only_derivable,
        "underived_scanned": len(scan_targets),
        "underived_truncated": len(scan_targets) != len(underived),
        "orphan_classification": dict(orphan_classes),
        "derivable_new_tables": len(derivable),
        "_eligible_keys": (summary["eligible_keys"] if summary else set())
        | (delta["eligible_keys"] if delta else set()),
    }


def cross_collection_audit(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Confirm the trainer-facing eligible rows stay unique across seasons."""

    seen: Counter[tuple[str, str, str, str]] = Counter()
    for report in reports:
        seen.update(report["_eligible_keys"])
    duplicates = sum(count - 1 for count in seen.values() if count > 1)
    return {
        "eligible_rows_total": sum(
            report["corpus_eligible_rows"] for report in reports
        ),
        "unique_eligible_keys": len(seen),
        "cross_collection_duplicate_keys": duplicates,
    }


def _print_report(reports: list[dict[str, Any]], totals: dict[str, Any]) -> None:
    for report in reports:
        gaps = report["gaps"]
        print(f"\n== season {report['season']}  {report['collection']}")
        print(
            f"   inventory: {report['inventory']['unique_tables']} unique tables from "
            f"{report['inventory']['memberships']} memberships "
            f"({report['disk_history_mode']}); replays on disk: "
            f"{report['replays_on_disk']}"
        )
        print(
            f"   csv: {report['csv']['rows']} rows, "
            f"{report['csv']['eligible_rows']} eligible, "
            f"{report['csv']['unique_tables']} tables, "
            f"duplicates {report['csv']['duplicate_keys']}, "
            f"checksum {'ok' if report['csv']['sha256_matches'] else 'MISMATCH'}"
        )
        print(
            f"   gaps: inventoried-missing-replay {gaps['inventoried_missing_replay']}, "
            f"downloaded-not-inventoried {gaps['downloaded_not_inventoried']}, "
            f"downloaded-not-in-csv {gaps['downloaded_not_in_csv']}, "
            f"csv-table-without-replay {gaps['csv_table_without_replay']}"
        )
        if report["delta"]:
            entry = report["delta"]
            print(
                f"   delta: {entry['rows']} rows, {entry['eligible_rows']} eligible, "
                f"{entry['unique_tables']} tables, {entry['represented_agents']} agents, "
                f"duplicates {entry['duplicate_keys']}, shared keys with frozen "
                f"{entry['keys_shared_with_frozen']}, checksum "
                f"{'ok' if entry['sha256_matches'] else 'MISMATCH'}"
            )
        scope = report["agent_scope"]
        print(
            f"   agents: {scope['manifest_agents']} in manifest, "
            f"{scope['agent_folders']} folders on disk; full history outside the "
            f"manifest: {len(scope['full_history_outside_manifest'])}; "
            f"50-hand sample only: {len(scope['sampled_only_outside_manifest'])}"
        )
        if report["underived_classification"]:
            print(f"   underived replays: {report['underived_classification']}")
        if report["orphan_classification"]:
            print(f"   orphan replays: {report['orphan_classification']}")
        if report["inventory"]["incomplete_agents"]:
            print(
                "   INCOMPLETE inventory pagination: "
                f"{report['inventory']['incomplete_agents']}"
            )
    print(
        f"\ntotal eligible rows {totals['eligible_rows_total']}; unique keys "
        f"{totals['unique_eligible_keys']}; cross-collection duplicates "
        f"{totals['cross_collection_duplicate_keys']}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="foreign play data/last 5 seasons top 15",
        help="directory holding the per-season collection folders",
    )
    parser.add_argument(
        "--report", help="optional path for the machine-readable JSON report"
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="scan every underived replay instead of the first 5000 per season",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root)
    directories = sorted(
        path for path in root.iterdir() if (path / "manifest.json").is_file()
    )
    if not directories:
        raise SystemExit(f"no collection folders under {root}")
    reports = [
        reconcile_collection(directory, deep=args.deep) for directory in directories
    ]
    totals = cross_collection_audit(reports)
    _print_report(reports, totals)
    for report in reports:
        report.pop("_eligible_keys", None)
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {"root": str(root), "collections": reports, "totals": totals}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
