"""Run the poker policy against a live dev.fun Arena competition.

Joins a live Texas Hold'em competition, polls pending actions, and plays each
decision with the dependency-free policy. The session runs for ``--seconds``
and then leaves the competition cleanly. ``--aggressive`` selects the current
higher-volume policy settings.

HARD STOPS (owner-gated money): a 402 entry fee on join, or busting so the
bankroll can no longer cover a buy-in. Never rebuys, pays, or requeues on a
bust on its own.

The session exits 0 only when Arena confirms the leave; repeated unusable
pending-state responses and unconfirmed departures exit nonzero.

Credentials come from ``ARENA_CREDENTIALS`` when set, otherwise this
repository's gitignored ``.arena-credentials``.

Usage:
    python run_agent.py <competition_id> [--seconds N] [--aggressive]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from devfun_poker_playground.decision_engine import safest_passive_action
from devfun_poker_playground.poker_policy import build_policy
from devfun_poker_playground.training_telemetry import (
    action_response_matches,
    make_decision_record,
    parse_replay_receipts,
    replay_path,
    TelemetryError,
    TelemetryLog,
)

BASE = "https://arena.dev.fun"

# Consecutive HTTP-200 responses with unusable bodies before the session stops
# instead of treating them as "no pending turns".
MAX_INVALID_PENDING_RESPONSES = 3
LEAVE_ATTEMPTS = 2


class MalformedResponse:
    """Arena HTTP success whose body was not JSON; never usable as state."""

    def __init__(self, raw: str) -> None:
        self.raw = raw[:200]

    def __repr__(self) -> str:
        return f"MalformedResponse(raw={self.raw!r})"


def credentials_path() -> Path:
    """``ARENA_CREDENTIALS`` when set, else this repository's own file.

    The runner never imports from ``tools/``, so this small resolver mirrors
    ``tools.api.credentials_path`` rather than sharing it.
    """

    configured = os.environ.get("ARENA_CREDENTIALS")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise SystemExit(f"ARENA_CREDENTIALS points at a missing file: {path}")
        return path
    default = Path(__file__).resolve().parent / ".arena-credentials"
    if default.is_file():
        return default
    raise SystemExit(
        f"no Arena credentials found; set ARENA_CREDENTIALS or place the file at {default}"
    )


def load_credentials() -> tuple[str, str | None]:
    credentials = credentials_path()
    data = json.loads(credentials.read_text(encoding="utf-8"))
    api_key = next(
        (
            data[field]
            for field in ("apiKey", "api_key", "key", "apiKeyValue", "token")
            if isinstance(data.get(field), str) and data[field]
        ),
        None,
    )
    if api_key is None:
        raise SystemExit("no api key field in ARENA_CREDENTIALS")
    agent_id = next(
        (
            data[field]
            for field in ("agentId", "agent_id")
            if isinstance(data.get(field), str) and data[field]
        ),
        None,
    )
    return api_key, agent_id


def load_api_key() -> str:
    return load_credentials()[0]


def request_arena(api_key, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "curl/8.9.1",
    }
    if api_key is not None:
        headers["x-arena-api-key"] = api_key
    request = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw or "{}")
            except json.JSONDecodeError:
                # A success status must never masquerade as an empty snapshot.
                return r.status, MalformedResponse(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            decoded = json.loads(raw)
        except Exception:
            decoded = None
        if isinstance(decoded, Mapping):
            return e.code, decoded
        return e.code, {"raw": raw}
    except Exception as e:
        return 0, {"error": repr(e)[:120]}


def validate_pending_snapshot(payload: object) -> Mapping | None:
    """The pending envelope, or ``None`` unless every consumed shape is a mapping.

    Arena can answer HTTP 200 with a non-JSON or non-mapping body; treating
    that as empty state silently misses turns, so the caller must retry or
    stop when this returns ``None``.
    """

    if not isinstance(payload, Mapping):
        return None
    for key in ("participant", "runner"):
        value = payload.get(key)
        if value is not None and not isinstance(value, Mapping):
            return None
    tables = payload.get("tables")
    if tables is None:
        return payload
    if not isinstance(tables, list):
        return None
    if any(not isinstance(table, Mapping) for table in tables):
        return None
    return payload


def participant_total_chips(payload: object) -> int | None:
    """``participant.totalChips`` when the response carries the expected shape."""

    if not isinstance(payload, Mapping):
        return None
    participant = payload.get("participant")
    if not isinstance(participant, Mapping):
        return None
    chips = participant.get("totalChips")
    if isinstance(chips, bool) or not isinstance(chips, int):
        return None
    return chips


def confirm_leave(api_key: str, competition: str) -> bool:
    """POST leave with at most one re-attempt; True only on a confirmed 2xx."""

    for attempt in range(1, LEAVE_ATTEMPTS + 1):
        status, response = request_arena(
            api_key,
            "POST",
            "/api/arena/texas/leave",
            {"competitionId": competition},
        )
        if 200 <= status < 300 and isinstance(response, Mapping):
            return True
        print(
            f"  leave attempt {attempt}/{LEAVE_ATTEMPTS} unconfirmed "
            f"(HTTP {status}): {json.dumps(response, default=str)[:160]}"
        )
    return False


def now_ms():
    return int(time.time() * 1000)


def is_reconnectable_join_conflict(status: int, response: object) -> bool:
    """Recognize Arena's already-seated and already-queued 409 responses."""

    if status != 409:
        return False
    detail = json.dumps(response, default=str).casefold()
    return "concurren" in detail or "already_queued" in detail


def reconcile_telemetry(
    telemetry: TelemetryLog,
    *,
    agent_id: str,
    competition_id: str,
) -> int:
    """Fetch public settlements and append results for known table IDs."""

    status, response = request_arena(
        None,
        "GET",
        replay_path(agent_id, competition_id),
    )
    if status != 200:
        print(f"  telemetry replay check failed HTTP {status}")
        return 0
    try:
        added = telemetry.append_settlements(
            competition_id,
            parse_replay_receipts(response),
        )
    except (OSError, TelemetryError, ValueError) as error:
        print(f"  telemetry replay check rejected: {error}")
        return 0
    if added:
        print(f"  telemetry settled {added} hand(s)")
    return added


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the poker policy against a live Arena competition."
    )
    parser.add_argument("competition_id", help="Arena competition to join")
    parser.add_argument("--seconds", type=int, default=600, help="session length")
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="use the current higher-volume policy settings",
    )
    parser.add_argument(
        "--telemetry",
        nargs="?",
        const=".arena-training.jsonl",
        metavar="PATH",
        help="append local training telemetry (default: .arena-training.jsonl)",
    )
    parser.add_argument(
        "--learned",
        action="store_true",
        help=(
            "play the approved learned artifact from artifacts/approved.json "
            "and refresh it between hands"
        ),
    )
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")
    if args.learned and args.aggressive:
        parser.error("--learned and --aggressive select different policies")
    return args


def main(argv=None):
    args = parse_args(argv)
    api_key, agent_id = load_credentials()
    competition = args.competition_id
    approved_token = None
    if args.learned:
        from devfun_poker_playground.learned_policy import (
            approved_fingerprint,
            load_approved,
        )

        policy = load_approved("artifacts")
        approved_token = approved_fingerprint("artifacts")
        mode = f"learned:{policy.policy_version}"
    else:
        policy = build_policy(aggressive=args.aggressive)
        mode = "aggressive" if args.aggressive else "standard"
    telemetry: TelemetryLog | None = None
    if args.telemetry is not None:
        if agent_id is None:
            raise SystemExit("telemetry requires agentId in ARENA_CREDENTIALS")
        try:
            telemetry = TelemetryLog(args.telemetry)
        except (OSError, TelemetryError) as error:
            raise SystemExit(f"cannot open telemetry: {error}") from error
    print(
        f"policy={mode} competition={competition} seconds={args.seconds}"
        + (f" telemetry={telemetry.path}" if telemetry is not None else "")
    )

    status, response = request_arena(
        api_key,
        "POST",
        "/api/arena/texas/join",
        {"competitionId": competition},
    )
    already_joined = is_reconnectable_join_conflict(status, response)
    if status == 402:
        print(
            f"STOP: entry fee (402): {response.get('message') or response}. Owner approval needed."
        )
        return 2
    if status == 403:
        print(f"STOP: access denied (403): {response.get('message') or response}.")
        return 3
    if already_joined:
        print("already seated or queued (reconnecting), skipping join")
    elif status not in (200, 201):
        print(f"STOP: join failed HTTP {status}: {response}")
        return 4
    else:
        print(f"joined: {json.dumps(response, default=str)[:200]}")

    start = time.time()
    start_chips = None if already_joined else participant_total_chips(response)
    action_attempts, actions_accepted, hand_ids = 0, 0, set()
    deadline = start + args.seconds
    replay_check_needed = telemetry is not None
    invalid_pending_responses = 0
    session_exit = 0

    try:
        while time.time() < deadline:
            status, payload = request_arena(
                api_key,
                "GET",
                f"/api/arena/texas/pending-actions?competitionId={competition}",
            )
            if status != 200:
                time.sleep(2)
                continue
            pending = validate_pending_snapshot(payload)
            if pending is None:
                invalid_pending_responses += 1
                print(
                    "  unusable pending-actions body on HTTP 200 "
                    f"({invalid_pending_responses}/{MAX_INVALID_PENDING_RESPONSES}): "
                    f"{payload!r:.120}"
                )
                if invalid_pending_responses >= MAX_INVALID_PENDING_RESPONSES:
                    print(
                        "STOP: Arena keeps answering success with an unusable "
                        "pending state; stopping rather than playing on as if "
                        "no turns were pending."
                    )
                    session_exit = 5
                    break
                time.sleep(2)
                continue
            invalid_pending_responses = 0
            participant = pending.get("participant") or {}
            if start_chips is None:
                start_chips = participant_total_chips(pending)
            if participant.get("chipState") == "busted":
                print(
                    f"STOP: busted (totalChips={participant.get('totalChips')}). "
                    "Rebuy is owner-gated, pausing."
                )
                break
            pending_tables = [
                table
                for table in (pending.get("tables") or [])
                if table.get("allowedActions")
            ]
            pending_tables.sort(key=lambda table: table.get("actionDeadlineAt") or 0)
            if not pending_tables:
                if telemetry is not None and replay_check_needed:
                    reconcile_telemetry(
                        telemetry,
                        agent_id=agent_id,
                        competition_id=competition,
                    )
                    replay_check_needed = False
                if args.learned:
                    # Between hands is the only safe point to swap weights.
                    from devfun_poker_playground.learned_policy import (
                        approved_fingerprint,
                        load_approved,
                        LearnedPolicyError,
                    )

                    token = approved_fingerprint("artifacts")
                    if token is not None and token != approved_token:
                        try:
                            refreshed = load_approved("artifacts")
                        except LearnedPolicyError as error:
                            print(
                                f"  approved refresh failed, keeping current: {error}"
                            )
                        else:
                            # Keep the session's opponent evidence across swaps.
                            refreshed.opponent_tracker = policy.opponent_tracker
                            policy = refreshed
                            approved_token = token
                            print(
                                f"  refreshed learned policy -> {policy.policy_version}"
                            )
                time.sleep(1.5)
                continue
            for table in pending_tables:
                if str(table.get("street") or "").casefold() not in (
                    "preflop",
                    "flop",
                    "turn",
                    "river",
                ):
                    continue
                table_id = table.get("tableId")
                if not isinstance(table_id, str) or not table_id:
                    print("  skipped snapshot without tableId")
                    continue
                table_competition = table.get("competitionId")
                if table_competition is not None and table_competition != competition:
                    print("  skipped snapshot for a different competition")
                    continue
                action_deadline_ms = table.get("actionDeadlineAt")
                budget = (
                    max(0.2, (action_deadline_ms - now_ms()) / 1000.0 - 0.6)
                    if isinstance(action_deadline_ms, (int, float))
                    else 8.0
                )
                decision = None
                fallback_reason = None
                try:
                    decision = policy.decide_with_diagnostics(table, deadline_s=budget)
                    payload = decision.to_payload()
                    temperature = decision.situation_temperature
                except Exception as exc:
                    available_actions = (table.get("allowedActions") or {}).get(
                        "availableActions"
                    ) or []
                    fallback = safest_passive_action(available_actions)
                    if fallback is None:
                        print(f"  decide error ({exc!r:.50}); no safe fallback action")
                        continue
                    payload = {"action": fallback, "message": "safe line"}
                    temperature = None
                    fallback_reason = "exception"
                    print(f"  decide error ({exc!r:.50}); fallback {fallback}")
                body = {
                    "tableId": table_id,
                    "action": payload.get("action"),
                    "message": payload.get("message") or "range-aware line",
                }
                if payload.get("amount") is not None:
                    body["amount"] = int(payload["amount"])
                if (table.get("allowedActions") or {}).get(
                    "reasoningRequired"
                ) and payload.get("reasoning"):
                    body["reasoning"] = payload["reasoning"]
                action_status, action_response = request_arena(
                    api_key, "POST", "/api/arena/texas/action", body
                )
                action_attempts += 1
                accepted = 200 <= action_status < 300
                identity_verified = bool(
                    accepted
                    and agent_id is not None
                    and action_response_matches(
                        action_response,
                        table_id=table_id,
                        competition_id=competition,
                        agent_id=agent_id,
                    )
                )
                if accepted:
                    actions_accepted += 1
                    hand_ids.add(table_id)
                response_error = None
                if not accepted and isinstance(action_response, Mapping):
                    response_error = action_response.get(
                        "error"
                    ) or action_response.get("message")
                if telemetry is not None:
                    try:
                        telemetry.append(
                            make_decision_record(
                                competition_id=competition,
                                policy_version=policy.policy_version,
                                table=table,
                                payload=payload,
                                decision=decision,
                                deadline_budget_s=budget,
                                fallback_reason=fallback_reason,
                                action_status=action_status,
                                identity_verified=identity_verified,
                                response_error=response_error,
                            )
                        )
                    except (OSError, TelemetryError, ValueError) as error:
                        print(
                            "STOP: Arena processed the action, but telemetry could "
                            f"not be saved: {error}. Do not resubmit it."
                        )
                        deadline = 0
                        break
                    replay_check_needed = True
                if accepted and agent_id is not None and not identity_verified:
                    print(
                        "STOP: Arena returned success with an unexpected action "
                        "identity. Do not resubmit blindly."
                    )
                    deadline = 0
                print(
                    f"  act[{action_attempts}] {table.get('street')} {body['action']}"
                    f"{' ' + str(body.get('amount')) if 'amount' in body else ''} -> "
                    f"{'ok' if accepted else 'HTTP ' + str(action_status)}"
                    + (
                        f" temp={temperature.temperature:.1f}/{temperature.band}"
                        f" bold={decision.temperature_boldness:+.2f}"
                        f" rng={decision.opponent_range_width:.2f}"
                        if temperature is not None
                        else ""
                    )
                    + (
                        f" bluff={decision.bluff_kind}"
                        if decision is not None and decision.bluff_kind is not None
                        else ""
                    )
                    + (
                        " hyper"
                        if decision is not None and decision.hyper_aggression
                        else ""
                    )
                    + (
                        " identity-unverified"
                        if accepted and not identity_verified and telemetry is not None
                        else ""
                    )
                    + (
                        f" {response_error}"
                        if not accepted and response_error is not None
                        else ""
                    )
                )
                if action_status == 402:
                    print("STOP: action requires payment (402). Owner approval needed.")
                    deadline = 0
                    break
    finally:
        # The shutdown flow must run however the session stopped -- bust,
        # timer, STOP path, or an escaping error -- and must never raise over
        # the original stop reason.
        departed = confirm_leave(api_key, competition)
        for _ in range(10):
            status, payload = request_arena(
                api_key,
                "GET",
                f"/api/arena/texas/pending-actions?competitionId={competition}",
            )
            snapshot = validate_pending_snapshot(payload) if status == 200 else None
            if snapshot is not None and (snapshot.get("runner") or {}).get(
                "canStopSafely"
            ):
                break
            time.sleep(2)
        _, final_payload = request_arena(
            api_key,
            "GET",
            f"/api/arena/texas/pending-actions?competitionId={competition}",
        )
        if telemetry is not None:
            # Settle the stop-triggering hand; its failure must not replace
            # the original exit reason.
            try:
                reconcile_telemetry(
                    telemetry,
                    agent_id=agent_id,
                    competition_id=competition,
                )
            except Exception as error:
                print(f"  final settlement reconcile failed: {error!r:.120}")
        end_chips = participant_total_chips(final_payload)
        delta = (
            None
            if (start_chips is None or end_chips is None)
            else end_chips - start_chips
        )
        print(
            f"=== session done: {actions_accepted}/{action_attempts} accepted actions "
            f"across ~{len(hand_ids)} observed hand/table ids; "
            f"chips {start_chips} -> {end_chips} (delta {delta}) ==="
        )
        if not departed:
            print(
                "WARNING: Arena did not confirm the leave; the agent may still "
                f"be seated. Recover with: python -m tools.leave {competition}"
            )
    if departed:
        return session_exit
    return session_exit or 6


if __name__ == "__main__":
    raise SystemExit(main())
