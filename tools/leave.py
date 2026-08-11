"""Leave the matchmaking queue / table for a competition. Usage: python leave.py <competitionId>"""
import json, os, sys, urllib.request
from pathlib import Path

d = json.loads(Path(os.environ["ARENA_CREDENTIALS"]).read_text())
key = next(d[k] for k in ("apiKey", "api_key", "key", "token") if isinstance(d.get(k), str))
comp = sys.argv[1] if len(sys.argv) > 1 else "cmsg35zvs001hbagh1wdjc1me"
req = urllib.request.Request(
    "https://arena.dev.fun/api/arena/texas/leave",
    data=json.dumps({"competitionId": comp}).encode(), method="POST",
    headers={"x-arena-api-key": key, "Content-Type": "application/json", "User-Agent": "curl/8.9.1"})
try:
    print("leave:", urllib.request.urlopen(req, timeout=30).read().decode()[:200])
except urllib.error.HTTPError as e:
    print("leave HTTP", e.code, e.read().decode()[:200])
