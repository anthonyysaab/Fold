"""Play live Arena sessions back to back until this machine stops.

One command starts it. Each finished session is restarted immediately, and the
free Playground competition is re-discovered every session so a season
rollover does not need a new command.

HARD STOPS (owner-gated money; never retried automatically):

* a 402 entry fee or a 403 refusal when joining,
* a busted or empty bankroll -- rebuy stays owner-gated,
* any competition that is not the free Playground. Two active Arena
  competitions charge real money (``Eval Open`` bills per run, ``Tournament``
  needs a buy-in), so discovery refuses anything whose name or description
  looks paid, and an explicit ``--competition`` is required to override.

This is a foreground console process, not a daemon or a service: it holds the
window it runs in and ends with your login session.

Clean shutdown: Ctrl+C and Ctrl+Break arrive as Python signals, and closing the
window, logging off, or shutting Windows down are caught through a console
control handler (CPython does not deliver those three as signals), so every
route out releases the table and the agent is never seated with no runner
answering. Windows allows only a few seconds for that handler, so it does the
one leave request and nothing else. A hard power cut cannot be intercepted;
Arena then times out the remaining hands.

Usage::

    python live_session.py                       # discover Playground, play on
    python live_session.py --session-seconds 7200
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import signal
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_agent
from devfun_poker_playground import poker_policy

# Substrings that mark a competition as costing money. Conservative on
# purpose: a false positive only refuses to auto-join, which is recoverable,
# while a false negative could spend the owner's funds.
PAID_MARKERS = (
    "pay",
    "paid",
    "buy in",
    "buy-in",
    "buyin",
    "entry fee",
    "fee",
    "usd",
    "usdt",
    "stake",
    "wager",
)
FREE_COMPETITION_MARKER = "playground"

RESTART_BACKOFF_S = (5, 15, 45, 120, 300)

ARCHIVE_ROOT = Path(__file__).resolve().parent / "runs"
DEPLOYMENT_MARKER = "DEPLOYED"


def policy_identity(standard: bool) -> str:
    """The deployed policy version the archive is studying."""

    chosen = (
        poker_policy.PokerPolicy if standard else poker_policy.AggressivePokerPolicy
    )
    return str(chosen.policy_version)


def sync_archive_to_deployment(
    root: Path, identity: str, *, keep: bool = False
) -> tuple[int, str | None]:
    """Reset the archive when the deployed agent changed (owner policy).

    The archive studies the CURRENT deployment, so a version change purges
    the previous version's run folders and stamps the new identity in
    ``runs/DEPLOYED``. Runs from before version stamping existed (no marker)
    are adopted, not deleted; ``keep=True`` adopts across a version change
    instead of purging. The telemetry journal is never touched -- it is the
    permanent training record and stamps every record with its own
    ``policy_version``.

    Returns ``(purged run count, previous identity or None)``.
    """

    root.mkdir(parents=True, exist_ok=True)
    marker = root / DEPLOYMENT_MARKER
    previous = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
    purged = 0
    if previous is not None and previous != identity and not keep:
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                purged += 1
    marker.write_text(identity + "\n", encoding="utf-8")
    return purged, previous


def _utc_stamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%dT%H%M%SZ")


class _Tee(io.TextIOBase):
    """Write-through to the console plus the session's archive log."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class RunArchive:
    """Persist each session's console log and outcome for later study.

    Layout: ``runs/<launch-utc>/session-NNN.log`` (the runner's full output),
    ``session-NNN.json`` (boundary facts: chips, timing, exit code, telemetry
    byte offsets), and ``run.json`` (the launch summary, rewritten at every
    boundary so a hard kill still leaves a readable record). Telemetry offsets
    let a session's journal slice be recovered exactly; the logs themselves
    contain no hole cards. Archive failures never stop play.
    """

    def __init__(self, root: Path, launched_at: float, args: argparse.Namespace):
        self.directory = root / _utc_stamp(launched_at)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._run: dict[str, Any] = {
            "launched_at": _utc_stamp(launched_at),
            "session_seconds": args.session_seconds,
            "policy": "standard" if args.standard else "aggressive",
            "policy_version": policy_identity(args.standard),
            "min_chips": args.min_chips,
            "telemetry": not args.no_telemetry,
            "sessions": 0,
            "stop_reason": None,
        }
        self._write_run()

    def _write_run(self) -> None:
        (self.directory / "run.json").write_text(
            json.dumps(self._run, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _telemetry_bytes() -> int | None:
        journal = Path(".arena-training.jsonl")
        try:
            return journal.stat().st_size if journal.exists() else 0
        except OSError:
            return None

    def open_session(
        self, index: int, competition: str, label: str, chips: int | None
    ) -> tuple[Any, dict[str, Any]]:
        """Return (log handle, manifest) for one session about to start."""

        manifest = {
            "index": index,
            "competition_id": competition,
            "competition_name": label,
            "started_at": _utc_stamp(time.time()),
            "chips_before": chips,
            "telemetry_bytes_before": self._telemetry_bytes(),
            "log": f"session-{index:03d}.log",
        }
        # Line-buffered so every played hand reaches disk immediately; block
        # buffering would hold a quiet session's whole log in memory for
        # hours, where a hard kill loses it.
        handle = (self.directory / manifest["log"]).open(
            "w", encoding="utf-8", buffering=1
        )
        return handle, manifest

    def close_session(
        self, manifest: dict[str, Any], exit_code: int | None, chips: int | None
    ) -> None:
        manifest.update(
            ended_at=_utc_stamp(time.time()),
            exit_code=exit_code,
            chips_after=chips,
            telemetry_bytes_after=self._telemetry_bytes(),
        )
        path = self.directory / f"session-{manifest['index']:03d}.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._run["sessions"] = manifest["index"]
        self._write_run()

    def finish(self, stop_reason: str) -> None:
        self._run["stop_reason"] = stop_reason
        self._write_run()


def is_free_playground(competition: Mapping[str, Any]) -> bool:
    """True only for the free Texas Hold'em Playground competition."""

    if competition.get("gameType") != "TexasHoldem":
        return False
    name = str(competition.get("name") or "")
    if FREE_COMPETITION_MARKER not in name.casefold():
        return False
    haystack = f"{name} {competition.get('description') or ''}".casefold()
    return not any(marker in haystack for marker in PAID_MARKERS)


def bankroll_stop_reason(participant: Mapping[str, Any] | None) -> str | None:
    """Reason to stop before joining again, or None to keep playing."""

    if not participant:
        return None
    if str(participant.get("chipState") or "").casefold() == "busted":
        return f"bankroll busted (totalChips={participant.get('totalChips')})"
    total = participant.get("totalChips")
    if isinstance(total, int) and total <= 0:
        return "bankroll is empty"
    return None


def exit_code_stop_reason(code: int) -> str | None:
    """Runner exit codes that must never be retried automatically."""

    return {
        2: "Arena asked for an entry fee (402); owner approval needed",
        3: "Arena refused access (403)",
    }.get(code)


def discover_competition(api_key: str) -> tuple[str | None, str]:
    """Find the free Playground competition; never return a paid one."""

    status, body = run_agent.request_arena(
        api_key, "GET", "/api/arena/competition/list-active"
    )
    if status != 200 or not isinstance(body, list):
        return None, f"could not list active competitions (HTTP {status})"
    free = [
        item for item in body if isinstance(item, Mapping) and is_free_playground(item)
    ]
    if not free:
        names = ", ".join(
            str(item.get("name")) for item in body if isinstance(item, Mapping)
        )
        return None, f"no free Playground competition is active (saw: {names})"
    # Newest season first, so a rollover is picked up automatically.
    free.sort(key=lambda item: item.get("seasonNumber") or 0, reverse=True)
    chosen = free[0]
    return str(chosen.get("id")), str(chosen.get("name"))


def participant_state(api_key: str, competition: str) -> Mapping[str, Any] | None:
    status, pending = run_agent.request_arena(
        api_key,
        "GET",
        f"/api/arena/texas/pending-actions?competitionId={competition}",
    )
    if status != 200 or not isinstance(pending, Mapping):
        return None
    participant = pending.get("participant")
    return participant if isinstance(participant, Mapping) else None


def leave_quietly(api_key: str, competition: str) -> None:
    """Best-effort table release; never raises during shutdown."""

    try:
        run_agent.request_arena(
            api_key, "POST", "/api/arena/texas/leave", {"competitionId": competition}
        )
        print(f"left {competition} cleanly")
    except Exception as error:  # shutdown path must not raise
        print(f"could not confirm leave: {error!r:.80}")


# Console control events Windows delivers but CPython does not turn into
# signals, so a `finally` alone would not release the table for them.
CTRL_CLOSE_EVENT = 2
CTRL_LOGOFF_EVENT = 5
CTRL_SHUTDOWN_EVENT = 6
_CONSOLE_HANDLER_REFS: list[Any] = []


def install_stop_signals() -> None:
    """Map termination signals onto the KeyboardInterrupt shutdown path.

    On Linux/macOS a service manager stops the process with SIGTERM (and a
    dropped SSH session sends SIGHUP); neither runs ``finally`` blocks by
    default, which would leave the agent seated. Windows never delivers
    SIGTERM to console apps -- the console control handler covers it there.
    """

    def raise_interrupt(*_: object) -> None:
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is not None:
            try:
                signal.signal(number, raise_interrupt)
            except (ValueError, OSError):  # non-main thread or exotic platform
                pass


def install_console_shutdown_handler(on_stop: Callable[[], None]) -> bool:
    """Run ``on_stop`` when the window closes, you log off, or Windows stops.

    Returns whether the handler was installed. Windows grants only a few
    seconds before killing the process, so ``on_stop`` must be one quick call.
    """

    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        from ctypes import wintypes

        prototype = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        closing = {CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT}

        def handler(event: int) -> bool:
            if event not in closing:
                return False  # let Ctrl+C and Ctrl+Break reach Python
            on_stop()
            return True

        callback = prototype(handler)
        if not ctypes.windll.kernel32.SetConsoleCtrlHandler(callback, True):
            return False
        # The callback must outlive this frame; if it is collected, Windows
        # calls freed memory during shutdown.
        _CONSOLE_HANDLER_REFS.append(callback)
        return True
    except Exception:  # never block startup over a shutdown nicety
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play live Arena sessions continuously until stopped."
    )
    parser.add_argument(
        "--competition",
        help="explicit competition id (overrides free-Playground discovery)",
    )
    parser.add_argument(
        "--session-seconds",
        type=int,
        default=21600,
        help=(
            "length of each session before a clean restart (default 21600, "
            "six hours). Longer sessions are denser: every restart forfeits "
            "the lobby queue position, and matchmaking is the throughput "
            "bottleneck."
        ),
    )
    parser.add_argument(
        "--min-chips",
        type=int,
        default=0,
        help="stop when the bankroll falls to this many chips (default 0)",
    )
    parser.add_argument(
        "--standard",
        action="store_true",
        help="play the standard policy instead of the aggressive one",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="do not record the local training journal",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="do not archive session logs and manifests under runs/",
    )
    parser.add_argument(
        "--keep-old-runs",
        action="store_true",
        help=(
            "keep archived runs from a previous deployment instead of "
            "purging them when the policy version changed"
        ),
    )
    args = parser.parse_args(argv)
    if args.session_seconds <= 0:
        parser.error("--session-seconds must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key, _ = run_agent.load_credentials()

    # The console handler fires on a different thread than the loop, so the
    # seated competition and the released flag live in one shared place.
    seat: dict[str, Any] = {"competition": None, "released": False}

    def release_table() -> None:
        """Leave the table once, from whichever exit route arrives first."""

        competition = seat["competition"]
        if competition is None or seat["released"]:
            return
        seat["released"] = True
        leave_quietly(api_key, competition)

    install_stop_signals()
    install_console_shutdown_handler(release_table)

    started = time.time()
    sessions = 0
    failures = 0
    stop_reason = "interrupted"

    archive: RunArchive | None = None
    if not args.no_archive:
        try:
            identity = policy_identity(args.standard)
            purged, previous = sync_archive_to_deployment(
                ARCHIVE_ROOT, identity, keep=args.keep_old_runs
            )
            if purged:
                print(
                    f"new deployment {previous} -> {identity}: "
                    f"purged {purged} archived run(s) of the old agent"
                )
            elif previous != identity:
                print(f"archive now tracking deployment {identity}")
            archive = RunArchive(ARCHIVE_ROOT, started, args)
            print(f"archiving this run under {archive.directory}")
        except OSError as error:
            print(f"archive disabled (cannot write runs/): {error}")

    print(
        "continuous live play; Ctrl+C to stop cleanly. "
        f"session length {args.session_seconds}s, "
        f"policy {'standard' if args.standard else 'aggressive'}"
    )
    try:
        while True:
            competition = args.competition
            if competition is None:
                competition, label = discover_competition(api_key)
                if competition is None:
                    stop_reason = label
                    break
            else:
                label = competition
            seat["competition"] = competition

            participant = participant_state(api_key, competition)
            reason = bankroll_stop_reason(participant)
            if reason is None and participant is not None:
                total = participant.get("totalChips")
                if isinstance(total, int) and total <= args.min_chips:
                    reason = f"bankroll {total} reached the --min-chips floor"
            if reason is not None:
                stop_reason = f"{reason}; rebuy is owner-gated"
                break

            sessions += 1
            uptime = time.time() - started
            chips = (participant or {}).get("totalChips")
            print(
                f"\n=== session {sessions} on {label} "
                f"(chips {chips}, uptime {uptime / 3600:.1f}h) ==="
            )

            runner_argv = [competition, "--seconds", str(args.session_seconds)]
            if not args.standard:
                runner_argv.append("--aggressive")
            if not args.no_telemetry:
                runner_argv.append("--telemetry")

            log_handle, manifest = None, None
            if archive is not None:
                try:
                    log_handle, manifest = archive.open_session(
                        sessions, competition, label, chips
                    )
                except OSError as error:
                    print(f"session archive skipped: {error}")

            code: int | None = None
            try:
                if log_handle is not None:
                    with contextlib.redirect_stdout(_Tee(sys.stdout, log_handle)):
                        code = run_agent.main(runner_argv)
                else:
                    code = run_agent.main(runner_argv)
            except KeyboardInterrupt:
                raise
            except Exception as error:  # a crashed session must not end the run
                code = 1
                print(f"session raised {error!r:.120}")
            finally:
                if log_handle is not None:
                    with contextlib.suppress(OSError):
                        log_handle.close()
                if archive is not None and manifest is not None:
                    after = participant_state(api_key, competition)
                    with contextlib.suppress(OSError):
                        archive.close_session(
                            manifest, code, (after or {}).get("totalChips")
                        )

            money_stop = exit_code_stop_reason(code)
            if money_stop is not None:
                stop_reason = money_stop
                break

            if code == 0:
                failures = 0
                continue

            failures += 1
            if failures > len(RESTART_BACKOFF_S):
                stop_reason = f"{failures} consecutive failed sessions"
                break
            delay = RESTART_BACKOFF_S[failures - 1]
            print(f"session exited {code}; retrying in {delay}s")
            time.sleep(delay)
    except KeyboardInterrupt:
        stop_reason = "stopped by you"
    finally:
        release_table()
        if archive is not None:
            with contextlib.suppress(OSError):
                archive.finish(stop_reason)
        print(
            f"\n=== STOPPED: {stop_reason}. "
            f"{sessions} session(s) over {(time.time() - started) / 3600:.1f}h ==="
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
