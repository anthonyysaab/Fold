# deploy — running the agent unattended

Revision 2026-09-02. Money rules apply unchanged (`CLAUDE.md` §1). Condensed
from the 2026-08-14 Linux runbook, corrected against the current code;
the long form is `archive/docs-superseded-2026-09-02/.../SESSION_AUDIT_AND_LINUX_RUNBOOK.md`
(reference only). Nothing in this folder is imported by the runner.

## Files

| file | role |
|---|---|
| `live_session.service` | systemd unit template (placeholders: paths, user) |
| `devfun-arena/strategy.py`, `devfun-arena/build_bundle.py` | the Eval-sandbox adapter and bundle builder (`bundle.zip`, 256 KB harness cap; the learned model does not fit) — dormant path, submit with `tools/submit.py`, poll with `tools/poll.py` |
| `chipzen/` | Chipzen (chipzen.ai) upload bot: the heuristic-aggressive-v6 port (`bot.py` maps Chipzen `GameState` to the Arena-shaped table; the policy carries no network at all since P2, so the deterministic equity rules drive every decision — same as `--aggressive`), full-cython Dockerfile, local probes, and the upload artifact `../0fold-heuristic-v1.tar.gz` (gitignored; 23 MB compressed, caps 250 MB/200 MB; caps verified 2026-09-02) |

## Chipzen upload (chipzen.ai/get-started-upload)

Free-to-play; nothing here touches the Arena API or the money rules. The
learned artifact does not ship: torch blows the 200 MB image cap and the
seccomp profile. Rebuild loop (Docker Desktop running, Git Bash for the
binary pipe — PowerShell corrupts it):

1. `python deploy/chipzen/smoke_probes.py` — engine healthy, legal actions, < 2 s/decision
2. `chipzen-sdk validate deploy/chipzen --check-connectivity` — the platform's own upload checks
3. `docker build -t 0fold-heuristic:v1 deploy/chipzen`
4. Git Bash: `docker save 0fold-heuristic:v1 | gzip > deploy/0fold-heuristic-v1.tar.gz`
5. `gzip -t deploy/0fold-heuristic-v1.tar.gz` before uploading

The image ships no readable `.py`: builder-stage `cythonize -i -3` compiles
`bot.py` and the trimmed engine closure (modules trimmed to the runtime
import closure; `training_telemetry`/the v8-v9 trainer chain is unreachable
on the serve path and is dropped). Runtime: alpine, non-root, `python -u`,
`CHIPZEN_WS_URL`/`CHIPZEN_TOKEN` from env. The engine's own `deadline_s=4.0`
decide path plus the check-call-fold fallback chain guarantee a legal,
in-time action on every turn.

## Windows (the only host used so far)

`.\play.cmd` or `python -u live_session.py` from the repo root, in a console
you can Ctrl+C **once**. Preconditions and the stop procedure:
`.handoff/PROCEDURES.md` §10–§12.

## Linux, systemd — procedure

1. **Host.** Debian 12+/Ubuntu 24.04 (Python ≥ 3.11; `from datetime import UTC`
   is the floor). `apt install python3 rsync openssh-server chrony`; enable
   `chrony` and `chrony-wait.service`; mask `sleep/suspend/hibernate/hybrid-sleep.target`.
   No pip, no venv: the serve path is stdlib. The clock is load-bearing: TLS
   and the per-decision budget both read it.
2. **Account and state.** System user `arena`; releases under
   `/opt/fold-ver-4/releases/<stamp>` with `current` symlinked (`ln -sfn`);
   state under `/var/lib/fold-ver-4/` (`.arena-credentials` 600 arena:arena,
   `.arena-training.jsonl` 640, `runs/`), symlinked **into each release**
   (`credentials_path()` resolves symlinks from the script directory).
3. **Payload.** Copy the **working tree**, never a bare clone: exclude `.git`,
   `runs`, `foreign play data`, `archive/data`, `.handoff`, `.claude`, caches,
   `.arena-credentials`, `.arena-training.jsonl`. Ship `MANIFEST.sha256`
   beside the payload and verify from the parent directory. Credentials
   travel out of band. `chmod +x play.sh` after transfer.
4. **Offline checks** from `/opt/fold-ver-4/current`: Python version; CWD ==
   script dir; `approved.json` has no backslash; stdlib-only import of
   `live_session` + `load_approved`; `python3 -m unittest discover -s tests
   -p "test_*.py"` (pytest is not on the box; one pytest-only file is not
   collected); `run_agent.py --help` needs no credentials.
5. **Unit** (`deploy/live_session.service` corrected): `User=arena`,
   `WorkingDirectory=/opt/fold-ver-4/current`,
   `ExecStart=/opt/fold-ver-4/current/play.sh --learned` (fail-loud guard; the
   approved pointer is already the default),
   `ExecStartPre=/usr/bin/test ! -e /var/lib/fold-ver-4/HALT`, **`Restart=no`**,
   **`TimeoutStopSec=600`** (a clean leave can take ~470 s of polls),
   `RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK`,
   `ReadWritePaths=/var/lib/fold-ver-4`, journald rate limits off,
   persistent journald. **Never put `--competition` in `ExecStart`.** Verify
   DNS inside the unit's namespace with `systemd-run`.
6. **First supervised run — OWNER.** The Windows runner must be stopped
   (there is no interlock). `sudo -u arena env PYTHONUNBUFFERED=1 ./play.sh
   --learned --session-seconds 900`; require the `policy learned:<v>` banner
   and a free Playground; Ctrl+C once; require `left <id> cleanly`. Check the
   `deadline_budget_s` distribution in the journal: >1% at the 0.2 s floor
   means the box or the clock is too slow.
7. **Enable.** `systemctl enable --now live_session`; test `systemctl stop`
   (clean leave inside 600 s) and a reboot.
8. **Monitor `stop_reason`, not exit status** — `live_session.main()` returns
   0 on a bust and on 402/403. A timer script that reads
   `runs/<latest>/run.json`, writes `HALT` on a money stop, and never
   auto-restarts a money stop. Add an off-box heartbeat: nothing on the box
   distinguishes "never came back" from "playing".
9. **Artifact update without restart.** Weights and manifest first, pointer
   last, atomic `mv`; confirm `refreshed learned policy` in the log. The swap
   trigger is the sha of `approved.json`, so a re-shipped identical pointer
   does nothing. Pull `runs/` before any restart: the next start purges it.
10. **Telemetry return.** Pull-only (`rsync --partial-dir`); never push into
    the server's journal; never concatenate into `.arena-training.jsonl`;
    quarantine any window where two runners overlapped.
11. **Rollback.** Tier 1 stop (`systemctl stop`, then `tools.leave` if
    unconfirmed). Tier 2 `python -m tools.promote_candidate --rollback` on
    Windows (available: `previous` = `candidate-v7-0001c`), then ship as in 9.
    Tier 3 flip `current` to the previous release after recreating the three
    state symlinks — this also rolls the served artifact back.

## Residual risks (unchanged since 2026-08-14 unless noted)

- No one-live-seat interlock: a 409 already-seated is treated as a reconnect.
- The money gate does not persist across a boot without the `HALT` sentinel.
- `runs/` purge is delayed-action: the restart after a hot-swap deletes the
  previous version's archives.
- Discovery failure: `discover_competition_with_retry` now retries with the
  restart back-off (the 2026-08-14 "ends play permanently" defect is fixed);
  a busted or 402/403 stop remains terminal by design.
- Runner exit codes 5 and 6 retry through the back-off while the agent may
  still be seated (defect 22 in `.handoff/PENDING_EDITS.md`).
- Hosting 24/7 does not change what is hosted: every instrument runs at
  60 bb; live is 500–2,900 bb; all three busts were deep-stack stack-offs.
