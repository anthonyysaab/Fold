"""One-shot dump of the live pending-actions state for diagnosis."""
import json, os, urllib.request
from pathlib import Path

key = None
data = json.loads(Path(os.environ["ARENA_CREDENTIALS"]).read_text())
for f in ("apiKey", "api_key", "key", "token"):
    if isinstance(data.get(f), str):
        key = data[f]; break

req = urllib.request.Request(
    "https://arena.dev.fun/api/arena/texas/pending-actions?competitionId=cmsg35zvs001hbagh1wdjc1me",
    headers={"x-arena-api-key": key, "User-Agent": "curl/8.9.1"})
d = json.loads(urllib.request.urlopen(req, timeout=30).read())

part = d.get("participant") or {}
runner = d.get("runner") or {}
lobby = d.get("lobby")
tables = d.get("tables") or []
active = d.get("activeTables") or []
print("participant:", {k: part.get(k) for k in ("chipState", "totalChips", "tableChips", "bankrollChips", "initialChips")})
print("runner:", runner)
print("lobby:", json.dumps(lobby)[:200] if lobby else None)
print(f"tables (my turn): {len(tables)}")
for t in tables:
    print("  turn-table:", {k: t.get(k) for k in ("tableId", "street", "selfSeatNumber", "currentSeatNumber", "actionDeadlineAt")},
          "allowed?", bool(t.get("allowedActions")))
print(f"activeTables: {len(active)}")
for t in active:
    print("  active:", {k: t.get(k) for k in ("tableId", "street", "status", "selfSeatNumber", "currentSeatNumber", "agentTableStatus", "selfStackChips")})
