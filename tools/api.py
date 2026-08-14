"""Authenticated dev.fun Arena GET helper.

Reads the API key from the credentials file (never prints it) and issues
read-only GETs to Arena. Usage: ``python tools/api.py <path-and-query> [...]``.

The credentials file is ``ARENA_CREDENTIALS`` when set, otherwise the
repository's own ``.arena-credentials`` (gitignored), so routine commands need
no environment setup.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://arena.dev.fun"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CREDENTIALS = REPO_ROOT / ".arena-credentials"


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
    raise SystemExit(f"no api key field in {path}; fields={list(data)}")


def get(key: str, path: str) -> tuple[int, str]:
    if not path.startswith("/") or path.startswith("//"):
        raise ValueError("Arena API paths must start with one slash")
    url = BASE + path
    request = urllib.request.Request(
        url, headers={"x-arena-api-key": key, "User-Agent": "curl/8.9.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def main() -> int:
    key = load_key()
    paths = sys.argv[1:] or ["/api/arena/submissions/settings"]
    for path in paths:
        try:
            status, body = get(key, path)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"## GET {path} -> HTTP {status}")
        print(body[:2000])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
