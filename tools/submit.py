"""Submit strategy.py or bundle.zip to a dev.fun Arena sandbox competition.

Usage:
    python tools/submit.py <path> [competitionId]

The API key is read from the credentials file at $ARENA_CREDENTIALS (a
.arena-credentials JSON written by the arena client) and is never printed.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://arena.dev.fun/api/arena/submissions"
DEFAULT_COMPETITION = "seed_poker_eval_s1"


def load_key() -> str:
    path = os.environ.get("ARENA_CREDENTIALS")
    if not path:
        raise SystemExit("set ARENA_CREDENTIALS to your .arena-credentials path")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("apiKey", "api_key", "key", "apiKeyValue", "token"):
        if isinstance(data.get(field), str) and data[field]:
            return data[field]
    raise SystemExit(f"no api key field in {path}; fields={list(data)}")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    file_path = Path(sys.argv[1])
    competition = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_COMPETITION
    filename = "bundle.zip" if file_path.suffix == ".zip" else "strategy.py"
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    key = load_key()
    boundary = "----devfun" + os.urandom(12).hex()
    body = bytearray()

    def field(name: str, value: str) -> None:
        body.extend(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            ).encode()
        )

    field("competitionId", competition)
    field("template", "static-agent")
    body.extend(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        URL,
        data=bytes(body),
        method="POST",
        headers={
            "x-arena-api-key": key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "curl/8.9.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            print("HTTP", response.status)
            print(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        print("HTTP", error.code)
        print(error.read().decode("utf-8", "replace"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
