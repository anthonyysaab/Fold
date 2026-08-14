"""Collect public Arena poker decisions from the current top agents.

Usage:
    python -m tools.collect_foreign_play_data
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from devfun_poker_playground.decision_engine import DecisionEngine
from devfun_poker_playground.game_state import (
    effective_stack_chips,
    features_from_table,
)
from devfun_poker_playground.learning_contract import LEARNING_FEATURE_NAMES

BASE_URL = "https://arena.dev.fun"
USER_AGENT = "Fold-ver-4-public-data-collector/1.0"
ACTION_FAMILIES = {
    "fold": "fold",
    "check": "check_call",
    "call": "check_call",
    "bet": "aggress",
    "raise": "aggress",
    "all-in": "aggress",
}
_REQUEST_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0


def _wait_for_request_slot() -> None:
    global _NEXT_REQUEST_AT
    with _REQUEST_LOCK:
        now = time.monotonic()
        request_at = max(now, _NEXT_REQUEST_AT)
        _NEXT_REQUEST_AT = request_at + 0.12
    time.sleep(max(0.0, request_at - now))


def _pause_requests(seconds: float) -> None:
    global _NEXT_REQUEST_AT
    with _REQUEST_LOCK:
        _NEXT_REQUEST_AT = max(_NEXT_REQUEST_AT, time.monotonic() + seconds)


def _rpc_url(procedure: str, payload: dict[str, object] | None = None) -> str:
    if payload is None:
        envelope: dict[str, object] = {
            "json": None,
            "meta": {"values": ["undefined"]},
        }
    else:
        envelope = {"json": payload}
    query = urllib.parse.urlencode(
        {"input": json.dumps(envelope, separators=(",", ":"))}
    )
    return f"{BASE_URL}/api/{procedure}?{query}"


def _get_json(url: str, attempts: int = 8) -> object:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    for attempt in range(attempts):
        _wait_for_request_slot()
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt + 1 == attempts:
                raise RuntimeError(f"failed to fetch {url}: {error}") from error
            if isinstance(error, urllib.error.HTTPError) and error.code not in {
                429,
                500,
                502,
                503,
                504,
            }:
                raise RuntimeError(f"failed to fetch {url}: {error}") from error
            delay = 2**attempt
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                retry_after = error.headers.get("Retry-After")
                delay = max(15, min(120, 15 * 2**attempt))
                if retry_after and retry_after.isdigit():
                    delay = max(delay, int(retry_after))
                _pause_requests(delay)
            else:
                time.sleep(delay)
    raise AssertionError("retry loop ended unexpectedly")


def _unwrap_rpc(document: object) -> object:
    try:
        return document["result"]["data"]["json"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError("Arena RPC response has an unexpected shape") from error


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _iso_ms(value: object) -> str:
    if not isinstance(value, int):
        return ""
    return datetime.fromtimestamp(value / 1000, UTC).isoformat().replace("+00:00", "Z")


def _timestamp_ms(value: object) -> int:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise ValueError("settlement time must be an integer or ISO timestamp")
    return round(
        datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000
    )


def _receipt_from_table(table: dict[str, Any], agent_id: str) -> dict[str, Any]:
    table_id = table.get("id")
    if not isinstance(table_id, str) or not table_id:
        raise ValueError("Texas table summary is missing its id")
    seat = next(
        (
            value
            for value in table.get("seats") or []
            if isinstance(value, dict) and value.get("agentId") == agent_id
        ),
        None,
    )
    if seat is None or not isinstance(seat.get("chipDelta"), int):
        raise ValueError(f"Texas table {table_id} is missing the agent result")
    winner_ids = {
        value.get("agentId")
        for value in table.get("winners") or []
        if isinstance(value, dict)
    }
    winner = next(
        (
            value
            for value in table.get("seats") or []
            if isinstance(value, dict) and value.get("agentId") in winner_ids
        ),
        None,
    )
    return {
        "handId": table_id,
        "tableId": table_id,
        "settledAt": _timestamp_ms(table.get("endedAt")),
        "chipDelta": seat["chipDelta"],
        "winnerHandle": winner.get("agentHandle") if winner else None,
    }


def _prior_events(events: list[dict[str, Any]], sequence: int) -> list[dict[str, Any]]:
    return [
        {
            "type": event.get("type"),
            "street": event.get("street"),
            "summary": event.get("payload"),
        }
        for event in events
        if isinstance(event.get("sequence"), int) and event["sequence"] < sequence
    ]


def _reconstruct_state(replay: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    snapshot = event.get("snapshot")
    if not isinstance(payload, dict) or not isinstance(snapshot, dict):
        raise ValueError("action event is missing payload or snapshot")
    seats = copy.deepcopy(snapshot.get("seats"))
    if not isinstance(seats, list):
        raise ValueError("action snapshot is missing seats")

    seat_number = payload.get("seatNumber")
    hero = next(
        (
            seat
            for seat in seats
            if isinstance(seat, dict) and seat.get("seatNumber") == seat_number
        ),
        None,
    )
    if hero is None:
        raise ValueError("action actor is missing from snapshot seats")

    actor_bet_before = int(payload.get("actorCurrentBetBefore") or 0)
    actor_bet_after = int(hero.get("currentBetChips") or 0)
    committed_after = int(hero.get("totalCommittedChips") or 0)
    hero.update(
        {
            "status": "Active",
            "stackChips": int(payload["stackBefore"]),
            "currentBetChips": actor_bet_before,
            "totalCommittedChips": max(
                0, committed_after - max(0, actor_bet_after - actor_bet_before)
            ),
        }
    )

    table = replay.get("table")
    if not isinstance(table, dict):
        raise ValueError("replay is missing table metadata")
    return {
        **table,
        **snapshot,
        "street": event.get("street"),
        "potChips": int(payload.get("pot") or 0),
        "currentBet": int(payload.get("currentBetBefore") or 0),
        "minRaiseTo": payload.get("minRaiseToBefore"),
        "currentSeatNumber": seat_number,
        "selfSeatNumber": seat_number,
        "allowedActions": payload.get("allowedActions"),
        "seats": seats,
        "recentEvents": _prior_events(replay.get("events", []), event["sequence"]),
    }


def _action_size(payload: dict[str, Any]) -> tuple[int | None, int]:
    action = str(payload.get("action") or "").casefold()
    contribution = int(payload.get("actorCurrentBetBefore") or 0)
    if action == "call":
        new_chips = int(payload.get("callAmount") or 0)
        return contribution + new_chips, new_chips
    if action in {"bet", "raise", "all-in"}:
        target = payload.get("toAmount")
        if target is None:
            target = payload.get("amount")
        if target is None and action == "all-in":
            allowed = payload.get("allowedActions") or {}
            target = allowed.get("allInToAmount")
        if target is None:
            return None, 0
        target = int(target)
        return target, max(0, target - contribution)
    return None, 0


def _legal_family_count(allowed: dict[str, Any]) -> int:
    actions = allowed.get("availableActions") or []
    return len(
        {ACTION_FAMILIES[action] for action in actions if action in ACTION_FAMILIES}
    )


def _decision_row(
    replay: dict[str, Any],
    event: dict[str, Any],
    agent: dict[str, Any],
    receipt: dict[str, Any],
    engine: DecisionEngine,
    *,
    arena_slug: str = "poker-playground",
    season_number: int = 13,
) -> dict[str, object]:
    state = _reconstruct_state(replay, event)
    payload = event["payload"]
    allowed = payload.get("allowedActions")
    if not isinstance(allowed, dict):
        raise ValueError("action event is missing allowedActions")
    action = str(payload.get("action") or "").casefold()
    action_family = ACTION_FAMILIES.get(action, "")
    action_to, action_new = _action_size(payload)

    base_features = features_from_table(state)
    top_fraction = engine._call_top_fraction(state, allowed)
    equity = engine._equity(state, top_fraction=top_fraction)
    if equity is None:
        raise ValueError("equity calculation was disabled")
    temperature = engine._situation_temperature(state, allowed, equity)
    learning_features = engine._learning_features(
        state, allowed, base_features, equity, temperature
    )
    if len(learning_features) != len(LEARNING_FEATURE_NAMES):
        raise AssertionError("learning feature count changed during collection")

    table = replay["table"]
    hero = next(
        seat for seat in state["seats"] if seat.get("agentId") == event.get("agentId")
    )
    effective_stack = effective_stack_chips(state)
    big_blind = int(state["bigBlindChips"])
    legal_actions = [str(value) for value in allowed.get("availableActions") or []]
    selected_is_legal = action in legal_actions
    legal_family_count = _legal_family_count(allowed)
    teacher_eligible = (
        event.get("type") == "ActionTaken"
        and bool(action_family)
        and selected_is_legal
        and legal_family_count >= 2
    )
    if event.get("type") == "TimeoutAction":
        exclusion_reason = "timeout_action"
    elif not action_family:
        exclusion_reason = "unsupported_action"
    elif not selected_is_legal:
        exclusion_reason = "selected_action_not_legal"
    elif legal_family_count < 2:
        exclusion_reason = "forced_action_family"
    else:
        exclusion_reason = ""

    winners = table.get("winners") or []
    hero_winner = next(
        (winner for winner in winners if winner.get("agentId") == event.get("agentId")),
        None,
    )
    chip_delta = int(receipt["chipDelta"])
    row: dict[str, object] = {
        "arena_slug": arena_slug,
        "season_number": season_number,
        "competition_id": table.get("competitionId"),
        "agent_rank": agent["rank"],
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "agent_score_at_collection": agent.get("totalScore"),
        "agent_hands_at_collection": agent.get("totalSubmissions"),
        "hand_id": receipt.get("handId"),
        "table_id": table.get("tableId") or table.get("id"),
        "table_number": table.get("tableNumber"),
        "hand_started_at": _iso_ms(table.get("startedAt")),
        "hand_settled_at": _iso_ms(receipt.get("settledAt")),
        "hand_chip_delta": chip_delta,
        "hand_reward_bb": chip_delta / big_blind,
        "hand_result": "win"
        if chip_delta > 0
        else "loss"
        if chip_delta < 0
        else "even",
        "hand_winner_handle": receipt.get("winnerHandle"),
        "hero_won_hand": bool(hero_winner),
        "hero_winning_hand": hero_winner.get("handName") if hero_winner else "",
        "showdown_reached": any(
            item.get("type") == "Showdown" for item in replay.get("events", [])
        ),
        "event_sequence": event.get("sequence"),
        "decision_at": _iso_ms(event.get("occurredAt")),
        "event_type": event.get("type"),
        "street": str(event.get("street") or "").casefold(),
        "dealer_seat": payload.get("dealerSeatNumber"),
        "hero_seat": payload.get("seatNumber"),
        "hero_hole_card_1": (hero.get("holeCards") or ["", ""])[0],
        "hero_hole_card_2": (hero.get("holeCards") or ["", ""])[1],
        "board_cards": "|".join(state.get("boardCards") or []),
        "player_count": len(state["seats"]),
        "active_player_count": sum(
            str(seat.get("status") or "").casefold() not in {"folded", "settled"}
            for seat in state["seats"]
        ),
        "small_blind_chips": state.get("smallBlindChips"),
        "big_blind_chips": big_blind,
        "pot_before_chips": payload.get("pot"),
        "current_bet_before_chips": payload.get("currentBetBefore"),
        "hero_bet_before_chips": payload.get("actorCurrentBetBefore"),
        "hero_stack_before_chips": payload.get("stackBefore"),
        "hero_total_committed_before_chips": hero.get("totalCommittedChips"),
        "effective_stack_before_chips": effective_stack,
        "call_chips": allowed.get("callChips"),
        "call_to_amount": allowed.get("callToAmount"),
        "min_raise_to": allowed.get("minRaiseTo"),
        "max_commit": allowed.get("maxCommit"),
        "legal_actions": "|".join(legal_actions),
        "legal_family_count": legal_family_count,
        "can_fold": bool(allowed.get("canFold")),
        "can_check": bool(allowed.get("canCheck")),
        "can_call": bool(allowed.get("canCall")),
        "can_bet": bool(allowed.get("canBet")),
        "can_raise": bool(allowed.get("canRaise")),
        "can_all_in": bool(allowed.get("canAllIn")),
        "action": action,
        "action_family": action_family,
        "action_to_amount": action_to if action_to is not None else "",
        "action_new_chips": action_new,
        "action_effective_stack_fraction": action_new / max(1, effective_stack),
        "selected_action_is_legal": selected_is_legal,
        "teacher_eligible": teacher_eligible,
        "teacher_exclusion_reason": exclusion_reason,
        "equity_top_fraction": top_fraction,
    }
    row.update(
        {
            f"feature_{name}": value
            for name, value in zip(
                LEARNING_FEATURE_NAMES, learning_features, strict=True
            )
        }
    )
    return row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _agent_history(
    raw: Path, agent: dict[str, Any], competition_id: str
) -> tuple[str, dict[str, dict[str, Any]], int]:
    agent_dir = raw / "agents" / f"{agent['rank']:02d}-{agent['id']}"
    receipts: dict[str, dict[str, Any]] = {}
    cursor: int | None = 0
    pages = 0
    seen_cursors: set[int] = set()
    while cursor is not None:
        if cursor in seen_cursors:
            raise ValueError(f"Texas table history repeated cursor {cursor}")
        seen_cursors.add(cursor)
        path = agent_dir / f"tables-{cursor:06d}.json"
        if path.exists():
            document = _read_json(path)
        else:
            document = _get_json(
                _rpc_url(
                    "arena.getTexasTables",
                    {
                        "arenaId": competition_id,
                        "agentId": agent["id"],
                        "limit": 100,
                        "cursor": cursor,
                    },
                )
            )
            _write_json(path, document)
        page = _unwrap_rpc(document)
        if not isinstance(page, dict) or not isinstance(page.get("tables"), list):
            raise ValueError("Texas table-history page has an unexpected shape")
        for table in page["tables"]:
            if not isinstance(table, dict) or table.get("status") != "Completed":
                continue
            receipt = _receipt_from_table(table, agent["id"])
            receipts[receipt["tableId"]] = receipt
        pages += 1
        next_cursor = page.get("nextCursor")
        if next_cursor is None:
            cursor = None
        elif isinstance(next_cursor, int) and next_cursor > cursor:
            cursor = next_cursor
        else:
            raise ValueError("Texas table history returned an invalid next cursor")
    return agent["id"], receipts, pages


def _all_history_receipts(
    raw: Path,
    agents: list[dict[str, Any]],
    competition_id: str,
    workers: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    receipts_by_agent: dict[str, dict[str, dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(agents))) as pool:
        futures = {
            pool.submit(_agent_history, raw, agent, competition_id): agent
            for agent in agents
        }
        for future in as_completed(futures):
            agent_id, receipts, pages = future.result()
            receipts_by_agent[agent_id] = receipts
            print(
                f"history agent rank {futures[future]['rank']}: "
                f"{len(receipts)} tables in {pages} pages",
                flush=True,
            )
    return receipts_by_agent


def _fetch_replays(raw: Path, table_ids: set[str], workers: int) -> None:
    table_dir = raw / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        table_id
        for table_id in sorted(table_ids)
        if not (table_dir / f"{table_id}.json").exists()
    ]
    cached = len(table_ids) - len(missing)
    print(
        f"unique tables: {len(table_ids)}; cached: {cached}; missing: {len(missing)}",
        flush=True,
    )
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(missing), 500):
            batch = missing[offset : offset + 500]
            futures = {
                pool.submit(
                    _get_json,
                    _rpc_url("arena.getTexasReplay", {"tableId": table_id}),
                ): table_id
                for table_id in batch
            }
            for future in as_completed(futures):
                table_id = futures[future]
                _write_json(table_dir / f"{table_id}.json", future.result())
                completed += 1
                if completed % 100 == 0 or completed == len(missing):
                    print(
                        f"fetched missing replays: {completed}/{len(missing)}",
                        flush=True,
                    )


def collect(args: argparse.Namespace) -> Path:
    collected_at = datetime.now(UTC)
    stamp = collected_at.strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.output_root)
        / f"{stamp}_{args.arena}_s{args.season}_top{args.top}"
    )
    raw = output / "raw"
    arenas_path = raw / "arenas.json"
    arenas_document = (
        _read_json(arenas_path)
        if arenas_path.exists()
        else _get_json(_rpc_url("arena.listArenas"))
    )
    arenas = _unwrap_rpc(arenas_document)
    arena = next(item for item in arenas if item.get("slug") == args.arena)
    season = next(
        item
        for item in arena["competitions"]
        if item.get("seasonNumber") == args.season
    )
    competition_id = season["id"]

    leaderboard_path = raw / "leaderboard.json"
    leaderboard_document = (
        _read_json(leaderboard_path)
        if leaderboard_path.exists()
        else _get_json(_rpc_url("arena.getLeaderboard", {"arenaId": competition_id}))
    )
    leaderboard = _unwrap_rpc(leaderboard_document)
    agents = sorted(
        (item for item in leaderboard["agents"] if item.get("claimed")),
        key=lambda item: item["totalScore"],
        reverse=True,
    )[: args.top]
    for rank, agent in enumerate(agents, 1):
        agent["rank"] = rank

    _write_json(arenas_path, arenas_document)
    _write_json(leaderboard_path, leaderboard_document)

    if args.all_hands:
        receipts_by_agent = _all_history_receipts(
            raw, agents, competition_id, args.workers
        )
    else:
        receipts_by_agent: dict[str, dict[str, dict[str, Any]]] = {}
        for agent in agents:
            url = (
                f"{BASE_URL}/api/arena/agent/{agent['id']}/replays?"
                + urllib.parse.urlencode(
                    {"competitionId": competition_id, "limit": args.hands}
                )
            )
            receipts = _get_json(url)
            if not isinstance(receipts, list):
                raise ValueError(f"receipt response for {agent['id']} is not an array")
            agent_dir = raw / "agents" / f"{agent['rank']:02d}-{agent['id']}"
            _write_json(agent_dir / "receipts.json", receipts)
            receipts_by_agent[agent["id"]] = {
                receipt["tableId"]: receipt for receipt in receipts
            }
    table_ids = {
        table_id for receipts in receipts_by_agent.values() for table_id in receipts
    }
    print(
        f"top agents: {len(agents)}; receipts: "
        f"{sum(len(value) for value in receipts_by_agent.values())}; "
        f"unique tables: {len(table_ids)}",
        flush=True,
    )
    _fetch_replays(raw, table_ids, args.workers)

    top_by_id = {agent["id"]: agent for agent in agents}
    engine = DecisionEngine(equity_trials=100, seed=7)
    feature_columns = [f"feature_{name}" for name in LEARNING_FEATURE_NAMES]
    skipped: list[dict[str, object]] = []
    represented: set[object] = set()
    row_count = 0
    eligible_count = 0
    timeout_count = 0
    csv_path = output / "top15_decisions.csv"
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    temporary_csv.parent.mkdir(parents=True, exist_ok=True)
    writer: csv.DictWriter[str] | None = None
    with temporary_csv.open("w", encoding="utf-8", newline="") as destination:
        for completed, table_id in enumerate(sorted(table_ids), 1):
            document = _read_json(raw / "tables" / f"{table_id}.json")
            replay = _unwrap_rpc(document)
            if not isinstance(replay, dict):
                raise ValueError(f"table replay {table_id} is not an object")
            for event in replay.get("events", []):
                agent_id = event.get("agentId")
                if event.get("type") not in {"ActionTaken", "TimeoutAction"}:
                    continue
                if agent_id not in top_by_id:
                    continue
                receipt = receipts_by_agent.get(agent_id, {}).get(table_id)
                if receipt is None:
                    continue
                try:
                    row = _decision_row(
                        replay,
                        event,
                        top_by_id[agent_id],
                        receipt,
                        engine,
                        arena_slug=args.arena,
                        season_number=args.season,
                    )
                    if any(
                        not math.isfinite(float(row[column]))
                        for column in feature_columns
                    ):
                        raise ValueError("decision contains a non-finite model feature")
                    if writer is None:
                        writer = csv.DictWriter(destination, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    row_count += 1
                    eligible_count += row["teacher_eligible"] is True
                    timeout_count += row["event_type"] == "TimeoutAction"
                    represented.add(agent_id)
                except (KeyError, TypeError, ValueError) as error:
                    skipped.append(
                        {
                            "table_id": table_id,
                            "event_sequence": event.get("sequence"),
                            "agent_id": agent_id,
                            "error": str(error),
                        }
                    )
            if completed % 500 == 0 or completed == len(table_ids):
                print(
                    f"derived replays: {completed}/{len(table_ids)}; rows: {row_count}",
                    flush=True,
                )
    if writer is None:
        temporary_csv.unlink(missing_ok=True)
        raise ValueError("no decision rows were produced")
    temporary_csv.replace(csv_path)
    _write_json(output / "skipped_decisions.json", skipped)
    missing_agents = [agent["id"] for agent in agents if agent["id"] not in represented]
    manifest = {
        "schema_version": 1,
        "collected_at": collected_at.isoformat().replace("+00:00", "Z"),
        "source": BASE_URL,
        "scope": {
            "arena_slug": args.arena,
            "arena_name": arena["name"],
            "season_number": args.season,
            "competition_id": competition_id,
            "season_status": season["status"],
            "ranking": "claimed agents sorted by totalScore descending",
            "top_agent_count": len(agents),
            "history_mode": "all_paginated_tables" if args.all_hands else "recent",
            "recent_hand_limit_per_agent": None if args.all_hands else args.hands,
        },
        "counts": {
            "receipt_records": sum(len(value) for value in receipts_by_agent.values()),
            "unique_tables": len(table_ids),
            "decision_rows": row_count,
            "teacher_eligible_rows": eligible_count,
            "timeout_rows": timeout_count,
            "skipped_decisions": len(skipped),
            "represented_agents": len(represented),
        },
        "agents": [
            {
                key: agent.get(key)
                for key in ("rank", "id", "name", "totalScore", "totalSubmissions")
            }
            for agent in agents
        ],
        "files": {
            "decision_csv": csv_path.name,
            "decision_csv_sha256": _sha256(csv_path),
            "raw_arenas": "raw/arenas.json",
            "raw_leaderboard": "raw/leaderboard.json",
            "raw_agent_history": (
                "raw/agents/<rank>-<agent_id>/tables-<cursor>.json"
                if args.all_hands
                else "raw/agents/<rank>-<agent_id>/receipts.json"
            ),
            "raw_table_replays": "raw/tables/<table_id>.json",
        },
        "model_features": {
            "count": len(LEARNING_FEATURE_NAMES),
            "columns": feature_columns,
            "equity_trials": 100,
            "equity_seed": 7,
        },
        "validation": {
            "missing_agents": missing_agents,
            "all_model_features_finite": True,
        },
        "limitations": [
            (
                "The public table-history pagination was exhausted for each selected agent."
                if args.all_hands
                else "The public receipt endpoint exposes at most 50 recent hands per agent."
            ),
            "Rank is a collection-time snapshot and can change while the season is active.",
            "Repeated hand reward is an outcome label, not proof the chosen action caused it.",
            "Only rows with teacher_eligible=true should be used for initial imitation labels.",
            "Raw replay text fields are preserved in JSON but excluded from model CSV columns.",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    print(
        f"decision rows: {row_count}; eligible: "
        f"{manifest['counts']['teacher_eligible_rows']}; skipped: {len(skipped)}",
        flush=True,
    )
    print(output.resolve(), flush=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", default="poker-playground")
    parser.add_argument("--season", type=int, default=13)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--hands", type=int, default=50)
    parser.add_argument(
        "--all-hands",
        action="store_true",
        help="exhaust the public paginated table history instead of recent receipts",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", default="foreign play data")
    parser.add_argument(
        "--output-dir",
        help="exact output directory; existing raw responses are reused",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.top < 1 or not 1 <= args.hands <= 50 or not 1 <= args.workers <= 8:
        raise SystemExit("top must be positive; hands 1-50; workers 1-8")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
