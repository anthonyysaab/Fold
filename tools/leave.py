"""Leave one Arena competition. Usage: python tools/leave.py <competitionId>."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
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
        "https://arena.dev.fun/api/arena/texas/leave",
        data=json.dumps({"competitionId": arguments[0]}).encode(),
        method="POST",
        headers={
            "x-arena-api-key": load_key(),
            "Content-Type": "application/json",
            "User-Agent": "curl/8.9.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print("leave:", response.read().decode("utf-8", "replace")[:200])
    except urllib.error.HTTPError as error:
        print(
            "leave HTTP",
            error.code,
            error.read().decode("utf-8", "replace")[:200],
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
