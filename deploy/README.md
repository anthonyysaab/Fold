# Deploy folder

Deployment adapters. Nothing here is imported by the live runner.

- `README.md` is this folder map.
- `devfun-arena/strategy.py` is the Eval sandbox adapter: `choose_action(table)` over the live snapshot dialect, guarded so every return path emits a legal action, with an inert programmatic zero network satisfying the policy's weights contract (the sandbox blocks filesystem reads).
- `devfun-arena/build_bundle.py` assembles `bundle.zip` at the repo root from the live decision closure (training-only modules dropped, 256 KB uncompressed harness cap) and refuses to ship a bundle that fails its local smoke gate.
- `live_session.service` is the systemd unit for unattended Linux/Raspberry Pi hosting of `live_session.py`. `systemctl stop` releases the table cleanly via SIGTERM; `Restart=no` on purpose because the supervisor restarts its own sessions and its money hard-stops are owner decisions a service manager must not retry past.

Submit a built bundle with `python tools/submit.py bundle.zip` (defaults to the Eval S1 competition) and poll it with `python tools/poll.py <submissionId>`.
