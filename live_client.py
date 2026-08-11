"""dev.fun Arena live-play client for the Poker Playground policy.

Joins a live Texas Hold'em competition, polls pending actions, and plays each
decision with the torch-free policy over the live table (which is exactly the
snapshot contract the policy already consumes). Bounded session: plays for
--seconds, then leaves the lobby cleanly. --loose swaps in the aggressive
multiway variant (see multiway.py).

HARD STOPS (owner-gated money): a 402 entry fee on join, or busting so the
bankroll can no longer cover a buy-in. Never rebuys, pays, or requeues on a
bust on its own.

Usage:
    ARENA_CREDENTIALS=/path/.arena-credentials \\
    python live_client.py <competitionId> [--seconds N] [--loose]
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from devfun_poker_playground.equity import prewarm
from devfun_poker_playground.pure_model import PurePolicy

BASE = "https://arena.dev.fun"
WEIGHTS = HERE / "artifacts" / "tiny-policy-pure.json"


def load_key() -> str:
    data = json.loads(Path(os.environ["ARENA_CREDENTIALS"]).read_text(encoding="utf-8"))
    for field in ("apiKey", "api_key", "key", "apiKeyValue", "token"):
        if isinstance(data.get(field), str) and data[field]:
            return data[field]
    raise SystemExit("no api key field in ARENA_CREDENTIALS")


KEY = load_key()


def _req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"x-arena-api-key": KEY, "Content-Type": "application/json",
                 "User-Agent": "curl/8.9.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw}
    except Exception as e:
        return 0, {"error": repr(e)[:120]}


def now_ms():
    return int(time.time() * 1000)


def make_policy(loose):
    if loose:
        from multiway import MultiwayPolicy
        return MultiwayPolicy(weights_path=str(WEIGHTS), equity_trials=200)
    return PurePolicy(weights_path=str(WEIGHTS), equity_trials=200)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    competition = sys.argv[1]
    seconds = int(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 600
    loose = "--loose" in sys.argv

    prewarm()
    policy = make_policy(loose)
    print(f"policy={'MULTIWAY-LOOSE' if loose else 'tight-v3'} competition={competition} seconds={seconds}")

    status, resp = _req("POST", "/api/arena/texas/join", {"competitionId": competition})
    already = status == 409 and "concurren" in json.dumps(resp).lower()
    if status == 402:
        print(f"STOP: entry fee (402) — {resp.get('message') or resp}. Owner approval needed.")
        return 2
    if status == 403:
        print(f"STOP: access denied (403) — {resp.get('message') or resp}.")
        return 3
    if already:
        print("already seated (reconnecting) — skipping join")
    elif status not in (200, 201):
        print(f"STOP: join failed HTTP {status} — {resp}")
        return 4
    else:
        print(f"joined: {json.dumps(resp)[:200]}")

    start = time.time()
    start_chips = None if already else ((resp.get("participant") or {}).get("totalChips"))
    if not isinstance(start_chips, int):
        start_chips = None
    acted, hands = 0, set()
    deadline = start + seconds

    while time.time() < deadline:
        status, pending = _req("GET", f"/api/arena/texas/pending-actions?competitionId={competition}")
        if status != 200:
            time.sleep(2)
            continue
        part = pending.get("participant") or {}
        if start_chips is None and isinstance(part.get("totalChips"), int):
            start_chips = part["totalChips"]
        if part.get("chipState") == "busted":
            print(f"STOP: busted (totalChips={part.get('totalChips')}). Rebuy is owner-gated — pausing.")
            break
        my_turn = [t for t in (pending.get("tables") or []) if t.get("allowedActions")]
        my_turn.sort(key=lambda t: t.get("actionDeadlineAt") or 0)
        if not my_turn:
            time.sleep(1.5)
            continue
        for table in my_turn:
            if str(table.get("street") or "").casefold() not in ("preflop", "flop", "turn", "river"):
                continue
            ddl = table.get("actionDeadlineAt")
            budget = max(0.2, (ddl - now_ms()) / 1000.0 - 0.6) if isinstance(ddl, (int, float)) else 8.0
            try:
                payload = policy.decide(table, deadline_s=budget)
            except Exception as exc:
                avail = (table.get("allowedActions") or {}).get("availableActions") or []
                fb = "check" if "check" in avail else ("call" if "call" in avail else "fold")
                payload = {"action": fb, "message": "safe line"}
                print(f"  decide error ({exc!r:.50}); fallback {fb}")
            body = {"tableId": table.get("tableId"), "action": payload.get("action"),
                    "message": payload.get("message") or "range-aware line"}
            if payload.get("amount") is not None:
                body["amount"] = int(payload["amount"])
            if (table.get("allowedActions") or {}).get("reasoningRequired") and payload.get("reasoning"):
                body["reasoning"] = payload["reasoning"]
            st, ar = _req("POST", "/api/arena/texas/action", body)
            acted += 1
            hands.add(table.get("id") or table.get("tableId"))
            print(f"  act[{acted}] {table.get('street')} {body['action']}"
                  f"{' ' + str(body.get('amount')) if 'amount' in body else ''} -> "
                  f"{'ok' if st == 200 else 'HTTP ' + str(st)}"
                  + (f" {ar.get('error') or ar.get('message')}" if st != 200 else ""))
            if st == 402:
                print("STOP: action requires payment (402). Owner approval needed.")
                deadline = 0
                break

    _req("POST", "/api/arena/texas/leave", {"competitionId": competition})
    for _ in range(10):
        st, p = _req("GET", f"/api/arena/texas/pending-actions?competitionId={competition}")
        if st == 200 and ((p or {}).get("runner") or {}).get("canStopSafely"):
            break
        time.sleep(2)
    st, me = _req("GET", f"/api/arena/texas/pending-actions?competitionId={competition}")
    end_chips = ((me or {}).get("participant") or {}).get("totalChips")
    delta = None if (start_chips is None or end_chips is None) else end_chips - start_chips
    print(f"=== session done: {acted} actions across ~{len(hands)} table-hands; "
          f"chips {start_chips} -> {end_chips} (delta {delta}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
