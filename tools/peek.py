"""Inspect one Arena competition. Usage: python tools/peek.py <competitionId>."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path


def load_key() -> str:
    path = os.environ.get("ARENA_CREDENTIALS")
    if not path:
        raise SystemExit("set ARENA_CREDENTIALS to your .arena-credentials path")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("apiKey", "api_key", "key", "apiKeyValue", "token"):
        if isinstance(data.get(field), str) and data[field]:
            return data[field]
    raise SystemExit(f"no api key field in {path}")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit(__doc__)
    request = urllib.request.Request(
        "https://arena.dev.fun/api/arena/texas/pending-actions?competitionId="
        + arguments[0],
        headers={"x-arena-api-key": load_key(), "User-Agent": "curl/8.9.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read())

    participant = data.get("participant") or {}
    runner = data.get("runner") or {}
    lobby = data.get("lobby")
    tables = data.get("tables") or []
    active_tables = data.get("activeTables") or []
    fields = (
        "chipState",
        "totalChips",
        "tableChips",
        "bankrollChips",
        "initialChips",
    )
    print("participant:", {field: participant.get(field) for field in fields})
    print("runner:", runner)
    print("lobby:", json.dumps(lobby)[:200] if lobby else None)
    print(f"tables (my turn): {len(tables)}")
    for table in tables:
        fields = (
            "tableId",
            "street",
            "selfSeatNumber",
            "currentSeatNumber",
            "actionDeadlineAt",
        )
        print(
            "  turn-table:",
            {field: table.get(field) for field in fields},
            "allowed?",
            bool(table.get("allowedActions")),
        )
    print(f"activeTables: {len(active_tables)}")
    for table in active_tables:
        fields = (
            "tableId",
            "street",
            "status",
            "selfSeatNumber",
            "currentSeatNumber",
            "agentTableStatus",
            "selfStackChips",
        )
        print("  active:", {field: table.get(field) for field in fields})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
