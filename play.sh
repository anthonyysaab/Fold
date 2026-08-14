#!/usr/bin/env bash
# One command: play live Arena poker continuously until stopped.
# Credentials resolve from this folder's .arena-credentials automatically.
# Ctrl+C, SIGTERM (systemctl stop), and SIGHUP all release the table cleanly.
set -u
cd "$(dirname "$0")"
exec python3 live_session.py "$@"
