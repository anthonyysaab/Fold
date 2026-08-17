#!/usr/bin/env bash
# One command: play live Arena poker continuously until stopped.
#
# Serves whatever artifacts/approved.json names once a candidate is promoted,
# and the heuristic otherwise; pass --aggressive or --standard to force a
# heuristic, or any live_session.py flag (for example --session-seconds 3600).
#
# Credentials resolve from this folder's .arena-credentials automatically.
# Ctrl+C, SIGTERM (systemctl stop), and SIGHUP all release the table cleanly.
set -euo pipefail
cd "$(dirname "$0")"

# Prefer an explicit interpreter, then python3, then python: an old server may
# ship either name, and the live path needs 3.11+ for the typing syntax used.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        echo "play.sh: no python3 or python on PATH" >&2
        exit 127
    fi
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "play.sh: need Python 3.11+, found $("$PYTHON" -V 2>&1)" >&2
    exit 1
fi

if [ ! -f .arena-credentials ] && [ -z "${ARENA_CREDENTIALS:-}" ]; then
    echo "play.sh: no .arena-credentials here and ARENA_CREDENTIALS is unset" >&2
    exit 1
fi

# Unbuffered so journalctl/tee show decisions as they happen.
exec "$PYTHON" -u live_session.py "$@"
