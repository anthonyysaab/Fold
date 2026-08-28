"""Poll a dev.fun Arena submission until it reaches a terminal status.

Usage:
    python tools/poll.py <submissionId>

Reads the API key from $ARENA_CREDENTIALS (never printed). The sandbox eval can
be slow and occasionally returns TimedOut / sandbox_eval_incomplete ("usually
temporary, resubmit"); failed validations do not count against the daily limit.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

TERMINAL = {"Succeeded", "Failed", "Cancelled", "TimedOut", "Discarded"}


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


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    key = load_key()
    url = f"https://arena.dev.fun/api/arena/submissions/{sys.argv[1]}"
    data = {}
    for _ in range(210):
        request = urllib.request.Request(
            url, headers={"x-arena-api-key": key, "User-Agent": "curl/8.9.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as error:  # transient network errors: keep polling
            print("poll error:", repr(error)[:120])
            time.sleep(10)
            continue
        status = data.get("status")
        hands = data.get("completedHands")
        print(f"status={status} hands={hands}/{data.get('targetHands')}")
        if status in TERMINAL:
            print("TERMINAL:", json.dumps(data))
            return 0 if status == "Succeeded" else 1
        time.sleep(10)
    print("TIMED_OUT_POLLING:", json.dumps(data))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
