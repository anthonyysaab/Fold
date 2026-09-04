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

EXIT CODES. 0 means a deliberate stop: you interrupted it, or an owner-gated
money guard fired. Anything else is a failure and says so loudly on the way
out -- 7 when competition discovery never recovered, 8 when consecutive
sessions kept failing, 9 when the abandoned-seat reconciliation refused to
join because the seat was not verifiably free. A supervisor that dies on
failure must never report success, because "exited 0" is what a clean
shutdown looks like.

This is a foreground console process, not a daemon or a service: it holds the
window it runs in and ends with your login session.

Clean shutdown: Ctrl+C and Ctrl+Break arrive as Python signals, and closing the
window, logging off, or shutting Windows down are caught through a console
control handler (CPython does not deliver those three as signals), so every
route out releases the table and the agent is never seated with no runner
answering. Windows allows only a few seconds for that handler, so it does the
one leave request and nothing else. A hard power cut cannot be intercepted;
Arena then times out the remaining hands.

DEFECT 22 -- stops that never reach Python. A SIGKILL-class kill (Task
Manager, ``Stop-Process -Force``, ``Popen.terminate()`` on Windows) runs no
in-process code at all, so no handler releases the table and nothing finishes
the archive: observed 2026-09-01/02 at ``runs/2026-09-01T235328Z`` and
``runs/2026-09-02T023838Z`` (``stop_reason: null``, no session record). Two
compensating layers. First, the stop intent -- the reason plus the seated
competition -- is written to ``run.json`` *before* any network leave, so a
process killed during the leave (a second Ctrl+C, the console handler's grace
window expiring) still leaves evidence of what it was doing. Second,
``--reconcile-abandoned-seat`` (default OFF, owner-enabled) makes the next
start release the seat left by a previous run that never verifiably released
one -- no clean stop, or a clean stop whose leave Arena never confirmed
(``seat_release_confirmed: false``). It verifies through the read-only peek
and calls the leave endpoint before any join. Releasing is allowed there,
joining is not, and nothing rebuys. Every seat check on that path fails
closed: a seat that cannot be read is not a free seat.

Usage::

    python live_session.py                       # discover Playground, play on
    python live_session.py --session-seconds 7200
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_agent
from engine import poker_policy

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

# Competition discovery is a network call, so one bad answer is expected and
# only a run of them is real. Retried on this backoff before giving up; giving
# up exits nonzero, because a supervisor that ends on failure with a success
# code is indistinguishable from a clean shutdown.
DISCOVERY_BACKOFF_S = (5, 15, 45, 120, 300)

# Supervisor exit codes. 0 is reserved for a deliberate stop -- you pressed
# Ctrl+C, or an owner-gated money guard fired -- so a service manager
# configured with Restart=on-failure never restarts past one of those.
EXIT_OK = 0
EXIT_DISCOVERY_FAILED = 7
EXIT_SESSION_FAILURES = 8
EXIT_RECONCILIATION_FAILED = 9

# A bankroll stop ends the supervisor, so it gets the same confirmation the
# runner now applies to a busted reading: one transient sample must not end
# an unattended run. See run_agent.confirm_bust.
BANKROLL_CONFIRMATION_POLLS = 2
BANKROLL_CONFIRMATION_DELAY_S = 2.0

ARCHIVE_ROOT = Path(__file__).resolve().parent / "runs"
ARTIFACTS_ROOT = Path(__file__).resolve().parent / "artifacts"
DEPLOYMENT_MARKER = "DEPLOYED"


def policy_identity(standard: bool, learned: bool = False) -> str:
    """The deployed policy version the archive is studying.

    With ``learned`` the identity comes from the approved pointer, so the
    archive is scoped to the artifact actually being served and a promotion
    counts as a deployment change like any other version bump.
    """

    if learned:
        from engine.learned_policy import load_approved

        return str(load_approved("artifacts", equity_trials=1).policy_version)
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
        # The console control handler writes the stop intent on a different
        # thread, so run.json rewrites are locked and atomic.
        self._lock = threading.Lock()
        self._run: dict[str, Any] = {
            "launched_at": _utc_stamp(launched_at),
            "session_seconds": args.session_seconds,
            "policy": (
                "learned"
                if getattr(args, "learned", False)
                else "standard"
                if args.standard
                else "aggressive"
            ),
            "policy_version": policy_identity(
                args.standard, getattr(args, "learned", False)
            ),
            "min_chips": args.min_chips,
            "telemetry": not args.no_telemetry,
            "sessions": 0,
            "stop_reason": None,
            "competition_id": None,
            # None until a release is attempted; False is a seat that may
            # still be held (see record_release and previous_unclean_run).
            "seat_release_confirmed": None,
        }
        self._write_run()

    def _write_run(self) -> None:
        with self._lock:
            payload = json.dumps(dict(self._run), indent=2)
        path = self.directory / "run.json"
        temporary = path.with_name("run.json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)

    def mark_seated(self, competition: str) -> None:
        """Record which competition the agent is about to join (defect 22).

        Written before the runner joins, so a supervisor killed mid-session
        still leaves the competition id in ``run.json`` -- the evidence the
        next start's reconciliation needs when no session manifest exists.
        """

        with self._lock:
            self._run["competition_id"] = competition
        self._write_run()

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
        with self._lock:
            self._run["sessions"] = manifest["index"]
        self._write_run()

    def record_release(self, confirmed: bool) -> None:
        """Record whether Arena confirmed the shutdown leave (LIMITS 6).

        ``finish`` writes the stop reason *before* the network leave, so a
        clean ``stop_reason`` alone cannot say whether the seat was actually
        released. This is the field that can: ``False`` marks the run for the
        next start's reconciliation even though it stopped cleanly.

        The caller writes ``False`` before the leave and ``True`` only once
        Arena confirms it -- the same order defect 22 forced on the stop
        reason. A kill inside the leave then leaves an unreleased seat on
        disk, which is the recoverable answer; the unrecoverable one is a
        clean record over a held seat.
        """

        with self._lock:
            self._run["seat_release_confirmed"] = bool(confirmed)
        self._write_run()

    def finish(self, stop_reason: str) -> None:
        with self._lock:
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


def bankroll_stop_reason(
    participant: Mapping[str, Any] | None, *, min_chips: int = 0
) -> str | None:
    """Reason to stop before joining again, or None to keep playing."""

    if not participant:
        return None
    if str(participant.get("chipState") or "").casefold() == "busted":
        return f"bankroll busted (totalChips={participant.get('totalChips')})"
    total = participant.get("totalChips")
    if isinstance(total, bool) or not isinstance(total, int):
        return None
    if total <= 0:
        return "bankroll is empty"
    if total <= min_chips:
        return f"bankroll {total} reached the --min-chips floor"
    return None


def confirm_bankroll_stop(
    api_key: str,
    competition: str,
    reason: str,
    *,
    min_chips: int = 0,
    polls: int = BANKROLL_CONFIRMATION_POLLS,
    delay_s: float = BANKROLL_CONFIRMATION_DELAY_S,
) -> str | None:
    """The bankroll stop reason if re-polls agree, else ``None`` to play on.

    Ending the supervisor is the most expensive thing this loop can do, and a
    single ``chipState: busted`` sample has been observed to be wrong (see
    ``run_agent.confirm_bust``). A real empty bankroll reads the same way on
    every poll, so confirmation costs seconds and never suppresses it.
    """

    for attempt in range(1, polls + 1):
        time.sleep(delay_s)
        participant = participant_state(api_key, competition)
        if participant is None:
            print(
                f"  !! BANKROLL STOP REJECTED: re-poll {attempt}/{polls} was "
                f"unreadable, so {reason!r} is unconfirmed. Playing on."
            )
            return None
        again = bankroll_stop_reason(participant, min_chips=min_chips)
        if again is None:
            print(
                f"  !! BANKROLL STOP REJECTED: {reason!r} was contradicted by "
                f"re-poll {attempt}/{polls} (chipState="
                f"{participant.get('chipState')!r}, totalChips="
                f"{participant.get('totalChips')}). Playing on."
            )
            return None
        reason = again
    return reason


def exit_code_stop_reason(code: int) -> str | None:
    """Runner exit codes that must never be retried automatically.

    Only the owner-gated money guards live here. Everything else is retried
    on ``RESTART_BACKOFF_S``; ``RETRY_NOTES`` records why for the exits where
    the choice is not obvious.
    """

    return {
        2: "Arena asked for an entry fee (402); owner approval needed",
        3: "Arena refused access (403)",
    }.get(code)


# Why the non-obvious runner exits are auto-retried rather than fatal
# (PENDING_EDITS item 14). Recorded in code so the decision is auditable.
RETRY_NOTES = {
    5: (
        "the runner exhausted its bounded retry on malformed HTTP 200s. A new "
        "session opens new connections and the restart backoff grows, and a "
        "run of consecutive failures still ends the supervisor nonzero, so "
        "retrying is bounded and correct"
    ),
    6: (
        "the runner could not confirm its leave, so the agent may still be "
        "seated. Retrying re-attaches a runner to that seat and answers its "
        "hands; halting instead would leave the agent seated with nobody "
        "answering, which is strictly worse. Money is not the exposure: the "
        "supervisor only ever auto-joins the free Playground "
        "(is_free_playground), and paid entry still hard-stops on 402/403. The "
        "supervisor tries one more leave before restarting"
    ),
}


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


def verify_explicit_competition(
    api_key: str, competition: str
) -> tuple[bool, str]:
    """Check an operator-supplied competition id against the money guard.

    ``--competition`` used to skip discovery entirely, and with it
    :func:`is_free_playground` -- the only thing standing between a
    mistyped id and a paid competition. The active list has carried a
    real-money entry (``[Poker] Eval Open S4``, paid per run in USD.T0)
    and a buy-in tournament alongside the free Playground, so an
    unchecked id is a live financial hazard, not a theoretical one.

    Fail-closed on every path: an id absent from the active list, an
    unreadable list, and a competition that fails the free-Playground
    test are all refusals. Returns ``(ok, label_or_reason)``.
    """

    status, body = run_agent.request_arena(
        api_key, "GET", "/api/arena/competition/list-active"
    )
    if status != 200 or not isinstance(body, list):
        return False, (
            f"cannot verify --competition {competition!r}: could not list "
            f"active competitions (HTTP {status})"
        )
    for item in body:
        if not isinstance(item, Mapping) or str(item.get("id")) != competition:
            continue
        name = str(item.get("name"))
        if is_free_playground(item):
            return True, name
        return False, (
            f"--competition {competition!r} is {name!r}, which is not the "
            "free Playground; refusing to join"
        )
    return False, (
        f"--competition {competition!r} is not in the active list; refusing "
        "to join an unverifiable competition"
    )


def discover_competition_with_retry(
    api_key: str,
    *,
    backoff_s: Sequence[float] = DISCOVERY_BACKOFF_S,
) -> tuple[str | None, str]:
    """Discovery, retried through transient failures before giving up.

    A single failed list call used to end the supervisor silently, and with
    exit 0. Discovery is one HTTP call against a live service, so one failure
    says nothing; only a run of them does. Both failure shapes are retried --
    an unreadable list response and a list with no free Playground in it --
    because a season rollover can make the second one momentarily true.

    Returns ``(competition id, label)`` or ``(None, reason)`` once the whole
    backoff is spent; the caller must treat ``None`` as a nonzero exit.
    """

    attempts = len(backoff_s) + 1
    reason = "competition discovery was never attempted"
    for attempt in range(1, attempts + 1):
        competition, label = discover_competition(api_key)
        if competition is not None:
            if attempt > 1:
                print(f"competition discovery recovered on attempt {attempt}")
            return competition, label
        reason = label
        if attempt < attempts:
            delay = backoff_s[attempt - 1]
            print(
                f"competition discovery failed (attempt {attempt}/{attempts}): "
                f"{reason}; retrying in {delay}s"
            )
            time.sleep(delay)
    return None, f"{reason} (gave up after {attempts} attempts)"


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


def leave_quietly(api_key: str, competition: str) -> bool:
    """Best-effort table release; never raises during shutdown.

    One request and no retry -- the console control handler gets only a few
    seconds -- but the answer is *read*: ``True`` only for a 2xx with a JSON
    body, exactly the bar :func:`run_agent.confirm_leave` applies. Until
    2026-09-03 this printed ``left <comp> cleanly`` for every answer, HTTP 503
    included, so an unreleased seat looked like a clean stop and no later
    start re-examined it (LIMITS 6). An unconfirmed leave now says so and
    names the recovery command; the caller records it.
    """

    try:
        status, response = run_agent.request_arena(
            api_key, "POST", "/api/arena/texas/leave", {"competitionId": competition}
        )
    except Exception as error:  # shutdown path must not raise
        print(f"could not confirm leave: {error!r:.80}")
        return False
    if 200 <= status < 300 and isinstance(response, Mapping):
        print(f"left {competition} cleanly")
        return True
    print(
        f"WARNING: the leave for {competition} was not confirmed (HTTP {status}); "
        f"the seat may still be held. Recover with: "
        f"python -m tools.leave {competition}"
    )
    return False


def previous_unclean_run(root: Path) -> Path | None:
    """The newest run folder whose ``run.json`` shows no released seat, else None.

    Two signatures, both meaning "this run may still hold a seat". Every clean
    exit writes a stop reason via :meth:`RunArchive.finish`, so a missing or
    null ``stop_reason`` is the defect-22 signature: the previous supervisor
    died without its release path. A run that *did* stop cleanly but whose
    leave Arena never confirmed carries ``seat_release_confirmed: false``
    (:meth:`RunArchive.record_release`); its stop reason is clean, so only
    that field distinguishes it from a released seat. An absent field is not a
    failed release -- runs written before the field existed stay clean.

    Folder names are UTC stamps, so the lexicographically last one is newest.
    """

    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        run_path = child / "run.json"
        if not run_path.is_file():
            continue
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(run, Mapping):
            continue
        if run.get("stop_reason") is None or run.get("seat_release_confirmed") is False:
            candidates.append(child)
    return max(candidates, key=lambda path: path.name) if candidates else None


def competition_of_unclean_run(run_dir: Path) -> str | None:
    """The competition a dead supervisor may still be seated in, or None.

    ``run.json`` has carried ``competition_id`` since the defect-22 fix, and
    session manifests carried it before, so either record answers. A kill
    before either write leaves nothing to release.
    """

    try:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        run = {}
    if isinstance(run, Mapping):
        competition = run.get("competition_id")
        if isinstance(competition, str) and competition:
            return competition
    manifests = sorted(
        run_dir.glob("session-*.json"), key=lambda path: path.name, reverse=True
    )
    for manifest in manifests:
        try:
            session = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(session, Mapping):
            competition = session.get("competition_id")
            if isinstance(competition, str) and competition:
                return competition
    return None


def is_seated_snapshot(snapshot: Mapping | None) -> bool:
    """Whether a validated pending snapshot shows a seat or queue entry.

    Mirrors the runner's ``seated_or_queued`` check; ``runner`` is excluded
    because the envelope carries one alongside an empty seat.
    """

    if snapshot is None:
        return False
    return bool(
        snapshot.get("lobby") or snapshot.get("tables") or snapshot.get("activeTables")
    )


def peek_seat(api_key: str, competition: str) -> tuple[Mapping | None, int]:
    """The read-only seat snapshot for ``competition``, plus the HTTP status.

    The single seat probe on the release path, so every caller means the same
    thing by an unreadable answer. ``None`` is "unverified", never "empty": a
    non-200, or a 200 whose body fails ``run_agent.validate_pending_snapshot``.
    The pre- and post-leave peeks in :func:`reconcile_abandoned_seat`
    disagreed on exactly that point until 2026-09-03 -- the second one read
    ``None`` through :func:`is_seated_snapshot`, which answers False for
    ``None``, and fell through to a join. One helper, one meaning.
    """

    status, payload = run_agent.request_arena(
        api_key,
        "GET",
        f"/api/arena/texas/pending-actions?competitionId={competition}",
    )
    snapshot = run_agent.validate_pending_snapshot(payload) if status == 200 else None
    return snapshot, status


def _stamp_reconciled_run(run_dir: Path, reason: str) -> None:
    """Record a reconciliation outcome in the previous run's ``run.json``.

    A stamped folder is no longer selected by :func:`previous_unclean_run`, so
    the next start skips it instead of sending the leave again. Called only
    once the seat is *verified* free, and it must clear both selection
    signatures -- a run whose leave went unconfirmed would otherwise be
    re-attempted at every start forever.

    A run with no stop reason (the hard-kill signature) takes ``reason`` as
    its stop reason. A run that stopped cleanly but could not confirm its
    leave keeps the stop reason it already has: that is the evidence of how it
    stopped, and ``reconciled_by`` carries what reconciliation did. Never
    fails startup.
    """

    run_path = run_dir / "run.json"
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(run, Mapping):
        return
    if run.get("stop_reason") is None:
        run["stop_reason"] = reason
    run["seat_release_confirmed"] = True
    run["reconciled_by"] = f"next-start reconciliation (defect 22 dial): {reason}"
    try:
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    except OSError:
        pass


def reconcile_abandoned_seat(api_key: str, *, root: Path | None = None) -> str | None:
    """Release a seat left by a previous supervisor that died without its
    release path (defect 22).

    The default-OFF dial behind ``--reconcile-abandoned-seat``. When the
    newest run folder shows no verified seat release -- no clean stop, or a
    clean stop whose leave Arena never confirmed -- the previous supervisor
    may have left the agent seated; that seat must be free before this process
    joins anything, because joining a seat that still holds the old agent
    silently reconnects a second runner to it (DECISIONS §1.2). Only the
    read-only peek and the leave endpoint are used here: releasing is allowed,
    joining is not, and nothing rebuys.

    Returns a refusal reason that must halt startup, or None to play on. On
    a refusal the previous run stays unstamped so every restart re-attempts
    the release until it succeeds; a released or empty seat is stamped so
    later starts do not repeat the leave.

    Both seat peeks go through :func:`peek_seat` and both fail closed: an
    answer that could not be read is a refusal, never a free seat. The
    post-leave peek used to test only :func:`is_seated_snapshot`, which is
    False for ``None``, so an HTTP 500 or an unusable 200 after the leave
    stamped the run and joined anyway -- and the stamp meant no later start
    ever re-attempted that release.
    """

    root = root or ARCHIVE_ROOT
    run_dir = previous_unclean_run(root)
    if run_dir is None:
        return None
    competition = competition_of_unclean_run(run_dir)
    if competition is None:
        _stamp_reconciled_run(
            run_dir, "no verified seat release and no competition recorded"
        )
        print(
            f"reconciliation: previous run {run_dir.name} left no verified seat "
            "release but recorded no competition; playing on"
        )
        return None
    snapshot, status = peek_seat(api_key, competition)
    if snapshot is None:
        return (
            f"previous run {run_dir.name} left no verified seat release and "
            f"the seat for {competition} could not be verified (HTTP {status}); "
            f"refusing to join. Recover with: python -m tools.leave {competition}"
        )
    if not is_seated_snapshot(snapshot):
        _stamp_reconciled_run(
            run_dir, "no verified seat release; verified not seated at next start"
        )
        print(
            f"reconciliation: previous run {run_dir.name} left no verified seat "
            f"release but {competition} shows no seat; playing on"
        )
        return None
    print(
        f"reconciliation: previous run {run_dir.name} left {competition} still "
        "seated; releasing before any join"
    )
    if not run_agent.confirm_leave(api_key, competition):
        return (
            f"the seat for {competition} left by the previous run could not be "
            f"released; refusing to join. Recover with: "
            f"python -m tools.leave {competition}"
        )
    snapshot, status = peek_seat(api_key, competition)
    if snapshot is None:
        # Symmetric with the pre-leave peek: an unread answer is not a free
        # seat. Unstamped, so the next start re-attempts the release.
        return (
            f"the leave for {competition} was sent but the release could not "
            f"be verified (HTTP {status}); refusing to join. Recover with: "
            f"python -m tools.leave {competition}"
        )
    if is_seated_snapshot(snapshot):
        return (
            f"the leave for {competition} was sent but the seat still reads "
            f"occupied; refusing to join. Recover with: "
            f"python -m tools.leave {competition}"
        )
    _stamp_reconciled_run(
        run_dir, "no verified seat release; seat released by the next start"
    )
    print("reconciliation: seat released; playing on")
    return None


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
        "--learned",
        action="store_true",
        help=(
            "serve the approved learned artifact from artifacts/approved.json. "
            "This is already the default whenever an approved artifact exists, "
            "so pass it only to fail loudly when one does not"
        ),
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help=(
            "force the heuristic aggressive policy even when an approved "
            "learned artifact exists"
        ),
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
    parser.add_argument(
        "--reconcile-abandoned-seat",
        action="store_true",
        help=(
            "defect-22 dial (default OFF, owner-enabled): when the previous "
            "run.json shows no clean stop, verify the seat through the "
            "read-only peek and call the leave endpoint before any join. "
            "Releasing is allowed, joining is not, and nothing rebuys"
        ),
    )
    args = parser.parse_args(argv)
    if args.session_seconds <= 0:
        parser.error("--session-seconds must be greater than zero")
    explicit = sum((args.learned, args.aggressive, args.standard))
    if explicit > 1:
        parser.error(
            "--learned, --aggressive, and --standard select different policies"
        )
    approved = (ARTIFACTS_ROOT / "approved.json").exists()
    if args.learned and not approved:
        parser.error(
            "--learned needs an approved artifact; run tools/promote_candidate.py first"
        )
    # "Play" means play what is deployed. Once a candidate is promoted, a bare
    # invocation must not silently serve a different policy than the approved
    # pointer names; the heuristics stay available behind an explicit flag.
    if explicit == 0 and approved:
        args.learned = True
        args.policy_defaulted = True
    else:
        args.policy_defaulted = False
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key, _ = run_agent.load_credentials()

    # Defect-22 dial: release a seat the previous supervisor may have left
    # behind, before this process joins anything. Default OFF.
    if args.reconcile_abandoned_seat:
        refusal = reconcile_abandoned_seat(api_key)
        if refusal is not None:
            print(
                "!!! SUPERVISOR REFUSED TO JOIN: "
                f"{refusal}. This is a deliberate halt: the seat is not "
                "verifiably free. !!!"
            )
            return EXIT_RECONCILIATION_FAILED

    # The console handler fires on a different thread than the loop, so the
    # seated competition and the released flag live in one shared place.
    seat: dict[str, Any] = {"competition": None, "released": False}
    archive_cell: dict[str, RunArchive | None] = {"archive": None}

    def release_table() -> None:
        """Leave the table once, from whichever exit route arrives first.

        The leave answer is recorded, not assumed: an unconfirmed release
        leaves a clean-looking ``stop_reason`` behind, and only
        ``seat_release_confirmed`` tells the next start to re-examine it.
        """

        competition = seat["competition"]
        if competition is None or seat["released"]:
            return
        seat["released"] = True
        archive = archive_cell["archive"]
        if archive is not None:
            # Intent before the network call, exactly like the stop reason: a
            # kill inside the leave must leave "unreleased" on disk for the
            # next start to reconcile, not a clean-looking record.
            with contextlib.suppress(OSError):
                archive.record_release(False)
        confirmed = leave_quietly(api_key, competition)
        if archive is not None and confirmed:
            with contextlib.suppress(OSError):
                archive.record_release(True)

    def console_stop() -> None:
        """Window close / logoff / shutdown: record the stop, then leave.

        The leave is one network call and Windows grants the handler only a
        few seconds, so the local ``run.json`` write must come first: if the
        grace window expires mid-leave, the evidence survives (defect 22).
        """

        archive = archive_cell["archive"]
        if archive is not None:
            with contextlib.suppress(OSError):
                archive.finish("console closed, logged off, or shutting down")
        release_table()

    install_stop_signals()
    install_console_shutdown_handler(console_stop)

    started = time.time()
    sessions = 0
    failures = 0
    stop_reason = "interrupted"
    exit_code = EXIT_OK

    archive: RunArchive | None = None
    if not args.no_archive:
        try:
            identity = policy_identity(args.standard, args.learned)
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
            archive_cell["archive"] = archive
            print(f"archiving this run under {archive.directory}")
        except OSError as error:
            print(f"archive disabled (cannot write runs/): {error}")

    if args.learned:
        served = f"learned:{policy_identity(args.standard, True)}"
        if getattr(args, "policy_defaulted", False):
            served += " (approved artifact; --aggressive or --standard to override)"
    else:
        served = "standard" if args.standard else "aggressive"
    print(
        "continuous live play; Ctrl+C to stop cleanly. "
        f"session length {args.session_seconds}s, policy {served}"
    )
    try:
        while True:
            competition = args.competition
            if competition is None:
                competition, label = discover_competition_with_retry(api_key)
                if competition is None:
                    stop_reason = label
                    exit_code = EXIT_DISCOVERY_FAILED
                    break
            else:
                # An explicit id still has to clear the money guard.
                ok, label = verify_explicit_competition(api_key, competition)
                if not ok:
                    stop_reason = label
                    exit_code = EXIT_DISCOVERY_FAILED
                    break
            seat["competition"] = competition
            if archive is not None:
                with contextlib.suppress(OSError):
                    archive.mark_seated(competition)

            participant = participant_state(api_key, competition)
            reason = bankroll_stop_reason(participant, min_chips=args.min_chips)
            if reason is not None:
                reason = confirm_bankroll_stop(
                    api_key, competition, reason, min_chips=args.min_chips
                )
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
            if args.learned:
                runner_argv.append("--learned")
            elif not args.standard:
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
                exit_code = EXIT_SESSION_FAILURES
                break
            if code == 6:
                # The runner could not confirm its own leave; try once more
                # from here so the restart begins from a released seat. The
                # answer is deliberately not acted on: RETRY_NOTES[6] -- if
                # this one is unconfirmed too, re-attaching a runner to that
                # seat still beats leaving it seated with nobody answering.
                print("  unconfirmed leave: releasing the seat once more")
                leave_quietly(api_key, competition)
            delay = RESTART_BACKOFF_S[failures - 1]
            note = RETRY_NOTES.get(code)
            print(
                f"session exited {code}; retrying in {delay}s"
                + (f" -- {note}" if note else "")
            )
            time.sleep(delay)
    except KeyboardInterrupt:
        stop_reason = "stopped by you"
        exit_code = EXIT_OK
    finally:
        # Write the stop intent to run.json BEFORE the network leave: the
        # leave can outlive a second Ctrl+C or the console handler's grace
        # window, and a kill during it must still leave evidence (defect 22).
        if archive is not None:
            with contextlib.suppress(OSError):
                archive.finish(stop_reason)
        release_table()
        print(
            f"\n=== STOPPED: {stop_reason}. "
            f"{sessions} session(s) over {(time.time() - started) / 3600:.1f}h ==="
        )
        if exit_code != EXIT_OK:
            print(
                f"!!! SUPERVISOR FAILED (exit {exit_code}): {stop_reason}. "
                "This is not a clean shutdown -- nothing is playing. !!!"
            )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
