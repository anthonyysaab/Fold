"""Authenticated dev.fun Arena GET helper.

Reads the API key from .arena-credentials (never prints it) and issues
read-only GETs so we can check sandbox submission access before building a
bundle. Usage: python devfun_api.py <path-and-query> [...]
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CRED = Path(r"C:\Users\user\devfun-poker-playground\.arena-credentials")
BASE = "https://arena.dev.fun"


def load_key() -> str:
    data = json.loads(CRED.read_text(encoding="utf-8"))
    for field in ("apiKey", "api_key", "key", "apiKeyValue", "token"):
        if isinstance(data.get(field), str) and data[field]:
            return data[field]
    raise SystemExit(f"no api key field in .arena-credentials; fields={list(data)}")


def get(key: str, path: str) -> tuple[int, str]:
    url = path if path.startswith("http") else BASE + path
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
        status, body = get(key, path)
        print(f"## GET {path} -> HTTP {status}")
        print(body[:2000])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
