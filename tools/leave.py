"""Leave one Arena competition. Usage: python tools/leave.py <competitionId>."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


#: Repo-root fallback, mirroring `tools/api.py`. `.arena-credentials` is
#: gitignored. Without this fallback the seat-release command documented in
#: `DECISIONS.md` section 1 -- the ONLY recovery, since there is no unclaim
#: endpoint -- fails whenever the environment variable happens to be unset.
DEFAULT_CREDENTIALS = Path(__file__).resolve().parent.parent / ".arena-credentials"


def credentials_path() -> Path:
    """Locate the credentials file without reading or printing its contents."""

    configured = os.environ.get("ARENA_CREDENTIALS")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise SystemExit(f"ARENA_CREDENTIALS points at a missing file: {path}")
        return path
    if DEFAULT_CREDENTIALS.is_file():
        return DEFAULT_CREDENTIALS
    raise SystemExit(
        "no Arena credentials found; set ARENA_CREDENTIALS or place the file at "
        f"{DEFAULT_CREDENTIALS}"
    )


def load_key() -> str:
    path = credentials_path()
    data = json.loads(path.read_text(encoding="utf-8"))
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
