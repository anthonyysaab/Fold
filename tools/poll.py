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


def load_key() -> str:
    path = os.environ.get("ARENA_CREDENTIALS")
    if not path:
        raise SystemExit("set ARENA_CREDENTIALS to your .arena-credentials path")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
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
            return 0
        time.sleep(10)
    print("TIMED_OUT_POLLING:", json.dumps(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
