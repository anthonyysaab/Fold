"""Append-only decision and settled-hand telemetry for offline learning."""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
import uuid
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from devfun_poker_playground.decision_engine import DecisionResult
from devfun_poker_playground.game_state import (
    active_opponent_count,
    _cards,
    _hero_and_seats,
    _integer,
    _mapping,
    _sequence,
    effective_stack_chips,
)
from devfun_poker_playground.learning_contract import (
    FEATURE_SCHEMA_VERSION,
    LEARNING_FEATURE_NAMES,
)
from devfun_poker_playground.policy_features import FEATURE_NAMES, LABELS

TELEMETRY_SCHEMA_VERSION = 2
# Versions this process can still READ. Bumping the writer must never orphan a
# journal: the live archive is the only record of how the deployed policy has
# actually played, and a reader that accepts one exact version turns a schema
# bump into silent data loss. Schema 2 adds `state.recent_actions`; every
# field schema 1 carried is still written, so old records stay loadable and
# consumers must treat the new field as optional.
READABLE_TELEMETRY_SCHEMA_VERSIONS = frozenset({1, 2})
CORPUS_SCHEMA_VERSION = 1
MAX_REPLAY_RECEIPTS = 50


class TelemetryError(ValueError):
    """Raised when telemetry or replay data violates its local contract."""


@dataclass(frozen=True, slots=True)
class ReplayReceipt:
    hand_id: str
    table_id: str
    settled_at_ms: int
    chip_delta: int


@dataclass(frozen=True, slots=True)
class TrainingExample:
    table_id: str
    policy_version: str
    features: tuple[float, ...]
    action_family_index: int
    behavior_probabilities: tuple[float, float, float]
    submitted_risk_fraction: float
    purse_bb: float
    reward_bb: float
    counterfactual: bool = False
    opponent_confidence: float = 0.0
    decision_id: str | None = None
    harvest_leg: str | None = None
    inclusion_count: int | None = None
    # Format-2 value branch for counterfactual rows: aggression is valued
    # per executed size, so two branches can share action_family_index 2.
    action_branch: str | None = None
    # Where each candidate branch went once branches were emitted by
    # executability and distinctness: `{candidate_label: surviving_label}`,
    # identity for a branch that survived. Without it a consumer cannot map
    # the action actually taken back onto a branch that carries a label --
    # V(s) would read zero for every decision whose own action was absorbed,
    # and the hybrid gate's reference action would vanish with it.
    branch_absorption: tuple[tuple[str, str], ...] | None = None


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TelemetryError(f"{name} must be a non-empty string")
    return value


def action_family(action: str) -> str | None:
    return {
        "fold": "fold",
        "check": "check_call",
        "call": "check_call",
        "bet": "aggress",
        "raise": "aggress",
        "all-in": "aggress",
    }.get(action)


def _optional_integer(value: object, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _amount_range(value: object, name: str) -> dict[str, int] | None:
    if value is None:
        return None
    amount_range = _mapping(value, name)
    minimum = _integer(amount_range.get("min"), f"{name}.min")
    maximum = _integer(amount_range.get("max"), f"{name}.max")
    if maximum < minimum:
        raise TelemetryError(f"{name}.max cannot be below min")
    return {"min": minimum, "max": maximum}


def action_response_matches(
    response: object,
    *,
    table_id: str,
    competition_id: str,
    agent_id: str,
) -> bool:
    """Confirm a success response belongs to the submitted identity."""

    if not isinstance(response, Mapping):
        return False
    table = response.get("table")
    participant = response.get("participant")
    return (
        isinstance(table, Mapping)
        and isinstance(participant, Mapping)
        and table.get("tableId") == table_id
        and table.get("competitionId") == competition_id
        and participant.get("competitionId") == competition_id
        and participant.get("agentId") == agent_id
    )


def replay_path(agent_id: str, competition_id: str) -> str:
    agent = quote(_identifier(agent_id, "agent_id"), safe="")
    competition = quote(_identifier(competition_id, "competition_id"), safe="")
    return f"/api/arena/agent/{agent}/replays?competitionId={competition}&limit=50"


def parse_replay_receipts(document: object) -> tuple[ReplayReceipt, ...]:
    """Validate and chronologically sort Arena's bounded replay receipts."""

    if not isinstance(document, list):
        raise TelemetryError("replay response must be an array")
    if len(document) > MAX_REPLAY_RECEIPTS:
        raise TelemetryError("replay response exceeds the 50-entry limit")
    receipts: list[ReplayReceipt] = []
    hand_ids: set[str] = set()
    table_ids: set[str] = set()
    for index, value in enumerate(document):
        if not isinstance(value, Mapping):
            raise TelemetryError(f"replay[{index}] must be an object")
        hand_id = _identifier(value.get("handId"), f"replay[{index}].handId")
        table_id = _identifier(value.get("tableId"), f"replay[{index}].tableId")
        settled_at = value.get("settledAt")
        chip_delta = value.get("chipDelta")
        if (
            isinstance(settled_at, bool)
            or not isinstance(settled_at, int)
            or not 0 <= settled_at <= 2**63 - 1
        ):
            raise TelemetryError(
                f"replay[{index}].settledAt must be a non-negative int64"
            )
        if (
            isinstance(chip_delta, bool)
            or not isinstance(chip_delta, int)
            or not -(2**63) <= chip_delta <= 2**63 - 1
        ):
            raise TelemetryError(f"replay[{index}].chipDelta must be an int64")
        if hand_id in hand_ids or table_id in table_ids:
            raise TelemetryError("replay response contains duplicate identities")
        hand_ids.add(hand_id)
        table_ids.add(table_id)
        receipts.append(ReplayReceipt(hand_id, table_id, settled_at, chip_delta))
    return tuple(sorted(receipts, key=lambda item: (item.settled_at_ms, item.hand_id)))


_RECENT_ACTIONS_LIMIT = 60


def _recent_actions(table: Mapping[str, Any]) -> list[dict[str, object]]:
    """Trimmed copy of the snapshot's ``recentEvents`` for the record.

    Defensive rather than strict: by the time a record is being written the
    engine has already parsed this table and decided on it, so a malformed
    entry here is skipped instead of failing the record -- losing one event
    from the log must never lose the decision. Bounded so a hostile or
    bloated snapshot cannot balloon the journal.
    """

    events = table.get("recentEvents") or []
    if not isinstance(events, (list, tuple)):
        return []
    trimmed: list[dict[str, object]] = []
    for raw_event in events[-_RECENT_ACTIONS_LIMIT:]:
        if not isinstance(raw_event, Mapping):
            continue
        summary = raw_event.get("summary")
        if not isinstance(summary, Mapping):
            continue
        action = summary.get("action")
        seat_number = summary.get("seatNumber")
        if not isinstance(action, str) or not action:
            continue
        entry: dict[str, object] = {
            "street": str(raw_event.get("street") or "").casefold(),
            "seat_number": seat_number if isinstance(seat_number, int) else None,
            "action": action.casefold(),
        }
        amount = summary.get("amount")
        if isinstance(amount, int) and not isinstance(amount, bool):
            entry["amount"] = amount
        trimmed.append(entry)
    return trimmed


def make_decision_record(
    *,
    competition_id: str,
    policy_version: str,
    table: Mapping[str, Any],
    payload: Mapping[str, Any],
    decision: DecisionResult | None,
    deadline_budget_s: float,
    fallback_reason: str | None,
    action_status: int,
    identity_verified: bool,
    response_error: str | None = None,
    recorded_at_ms: int | None = None,
) -> dict[str, object]:
    """Build a bounded, credential-free record for one submitted action."""

    table_id = _identifier(table.get("tableId"), "tableId")
    competition_id = _identifier(competition_id, "competition_id")
    policy_version = _identifier(policy_version, "policy_version")
    allowed = _mapping(table.get("allowedActions"), "allowedActions")
    legal_actions = sorted(
        str(value)
        for value in _sequence(allowed.get("availableActions"), "availableActions")
    )
    action = _identifier(payload.get("action"), "action")
    final_family = action_family(action)
    hero, _ = _hero_and_seats(table)
    hole_cards = _cards(hero.get("holeCards"), "hero holeCards", expected=2)
    board_cards = _cards(table.get("boardCards"), "boardCards")
    pot_chips = _integer(table.get("potChips"), "potChips")
    hero_stack = _integer(hero.get("stackChips"), "hero stackChips")
    contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
    total_committed = _integer(
        hero.get("totalCommittedChips", contribution), "hero totalCommittedChips"
    )
    call_chips = _integer(allowed.get("callChips", 0), "callChips")
    amount_to = payload.get("amount")
    if amount_to is not None:
        amount_to = _integer(amount_to, "amount")
    if action == "call":
        new_chips = call_chips
    elif final_family == "aggress" and amount_to is not None:
        new_chips = max(0, amount_to - contribution)
    else:
        new_chips = 0
    effective_stack = max(1, effective_stack_chips(table))
    submitted_risk_fraction = min(1.0, new_chips / effective_stack)

    features = decision.learning_features if decision is not None else None
    if features is not None and len(features) != len(LEARNING_FEATURE_NAMES):
        raise TelemetryError("decision learning features violate the contract")
    temperature = decision.situation_temperature if decision is not None else None
    if decision is not None and decision.deadline_fallback and fallback_reason is None:
        fallback_reason = "deadline"
    accepted = 200 <= action_status < 300
    legal_families = {
        family
        for value in legal_actions
        if (family := action_family(value)) is not None
    }
    proposed_family = decision.family if decision is not None else None
    probabilities = decision.behavior_probabilities if decision is not None else None
    if probabilities is not None:
        if (
            len(probabilities) != len(LABELS)
            or not all(
                math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities
            )
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)
        ):
            raise TelemetryError("behavior probabilities violate the action contract")
    hyper_aggression = decision.hyper_aggression if decision is not None else False
    training_eligible = (
        accepted
        and identity_verified
        and fallback_reason is None
        and features is not None
        and probabilities is not None
        and len(legal_families) > 1
        and proposed_family == final_family
        # Deliberate anti-modeling noise must never become a teacher.
        and not hyper_aggression
    )
    budget = float(deadline_budget_s)
    if not math.isfinite(budget) or budget < 0.0:
        raise TelemetryError("deadline_budget_s must be finite and non-negative")

    temperature_record: dict[str, object] | None = None
    if temperature is not None:
        temperature_record = {
            "value": temperature.temperature,
            "band": temperature.band,
            "factors": {
                name: round(value, 6) for name, value in temperature.factor_risk.items()
            },
        }
    return {
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "event": "decision",
        "event_id": uuid.uuid4().hex,
        "recorded_at_ms": int(time.time() * 1000)
        if recorded_at_ms is None
        else recorded_at_ms,
        "competition_id": competition_id,
        "table_id": table_id,
        "policy_version": policy_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION if features else None,
        "features": list(features) if features else None,
        "street": str(table.get("street") or "").casefold(),
        "big_blind_chips": _integer(
            table.get("bigBlindChips"), "bigBlindChips", minimum=1
        ),
        "equity": decision.equity if decision is not None else None,
        "temperature": temperature_record,
        "opponent_range_width": (
            decision.opponent_range_width if decision is not None else None
        ),
        "opponent_evidence_confidence": (
            decision.opponent_evidence_confidence if decision is not None else 0.0
        ),
        "lead_position": decision.lead_position if decision is not None else None,
        "bluff_kind": decision.bluff_kind if decision is not None else None,
        "hyper_aggression": hyper_aggression,
        "state": {
            # Schema 2: the betting that led to this decision. The snapshot
            # already carries it and the engine already parses it (aggression
            # counters, blind detection) -- it was simply discarded. Without
            # it no live hand can be attributed: "ran into the top of their
            # range" and "was outplayed" are indistinguishable, which is what
            # made the 2026-08-15 drawdown forensics stop at hero's own side.
            "recent_actions": _recent_actions(table),
            "hero_seat_number": _integer(
                table.get("selfSeatNumber"), "selfSeatNumber", minimum=1
            ),
            "hole_cards": list(hole_cards),
            "board_cards": list(board_cards),
            "active_player_count": 1 + active_opponent_count(table),
            "pot_chips": pot_chips,
            "hero_stack_chips": hero_stack,
            "hero_total_committed_chips": total_committed,
            "hero_purse_chips": hero_stack + total_committed,
            "effective_stack_chips": effective_stack,
            "hero_contribution_chips": contribution,
            "current_bet_chips": _integer(table.get("currentBet"), "currentBet"),
            "position": (
                features[FEATURE_NAMES.index("position")] if features else None
            ),
        },
        "legal": {
            "actions": legal_actions,
            "call_chips": call_chips,
            "call_to_amount": _optional_integer(
                allowed.get("callToAmount"), "callToAmount"
            ),
            "bet_range": _amount_range(allowed.get("betRange"), "betRange"),
            "raise_range": _amount_range(allowed.get("raiseRange"), "raiseRange"),
            "all_in_to_amount": _optional_integer(
                allowed.get("allInToAmount"), "allInToAmount"
            ),
            "amount_semantics": str(allowed.get("amountSemantics") or "toAmount"),
        },
        "proposed_family": proposed_family,
        "behavior_probabilities": list(probabilities) if probabilities else None,
        "action": action,
        "amount_to": amount_to,
        "proposed_risk_fraction": (
            decision.proposed_risk_fraction if decision is not None else None
        ),
        "submitted_risk_fraction": round(submitted_risk_fraction, 6),
        "deadline_budget_s": round(budget, 6),
        "fallback_reason": fallback_reason,
        "action_status": action_status,
        "accepted": accepted,
        "identity_verified": identity_verified,
        "training_eligible": training_eligible,
        "response_error": str(response_error)[:200] if response_error else None,
    }


class TelemetryLog:
    """Single-writer JSONL journal joined to settlements by Arena table ID."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._tables: dict[str, tuple[str, int]] = {}
        self._settled: set[str] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._load_index()

    def _load_index(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TelemetryError(
                        f"invalid telemetry JSON on line {line_number}"
                    ) from exc
                if (
                    record.get("telemetry_schema_version")
                    not in READABLE_TELEMETRY_SCHEMA_VERSIONS
                ):
                    raise TelemetryError(
                        f"unsupported telemetry schema on line {line_number}"
                    )
                self._index(record)

    def _index(self, record: Mapping[str, Any]) -> None:
        event = record.get("event")
        table_id = record.get("table_id")
        if not isinstance(table_id, str) or not table_id:
            return
        if (
            event == "decision"
            and record.get("accepted") is True
            and record.get("identity_verified") is True
        ):
            competition_id = record.get("competition_id")
            big_blind = record.get("big_blind_chips")
            if isinstance(competition_id, str) and isinstance(big_blind, int):
                self._tables[table_id] = (competition_id, big_blind)
        elif event == "hand_result":
            self._settled.add(table_id)

    def append(self, record: Mapping[str, Any]) -> None:
        if record.get("telemetry_schema_version") != TELEMETRY_SCHEMA_VERSION:
            raise TelemetryError("record has the wrong telemetry schema")
        line = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._index(record)

    def append_settlements(
        self,
        competition_id: str,
        receipts: tuple[ReplayReceipt, ...],
    ) -> int:
        """Append unseen results for accepted decisions and return their count."""

        added = 0
        for receipt in receipts:
            table = self._tables.get(receipt.table_id)
            if table is None or receipt.table_id in self._settled:
                continue
            recorded_competition, big_blind = table
            if recorded_competition != competition_id:
                continue
            self.append(
                {
                    "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
                    "event": "hand_result",
                    "event_id": uuid.uuid4().hex,
                    "recorded_at_ms": int(time.time() * 1000),
                    "competition_id": competition_id,
                    "table_id": receipt.table_id,
                    "hand_id": receipt.hand_id,
                    "settled_at_ms": receipt.settled_at_ms,
                    "chip_delta_chips": receipt.chip_delta,
                    "big_blind_chips": big_blind,
                    "reward_bb": round(receipt.chip_delta / big_blind, 6),
                }
            )
            added += 1
        return added


def _records(path: str | Path):
    with Path(path).expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TelemetryError(
                    f"invalid telemetry JSON on line {line_number}"
                ) from exc
            if (
                not isinstance(record, Mapping)
                or record.get("telemetry_schema_version")
                not in READABLE_TELEMETRY_SCHEMA_VERSIONS
            ):
                raise TelemetryError(
                    f"unsupported telemetry schema on line {line_number}"
                )
            yield line_number, record


def load_training_examples(path: str | Path) -> tuple[TrainingExample, ...]:
    """Join eligible decisions to terminal rewards from one telemetry journal."""

    rewards: dict[str, tuple[str, float]] = {}
    for line_number, record in _records(path):
        if record.get("event") != "hand_result":
            continue
        table_id = _identifier(record.get("table_id"), f"line {line_number} table_id")
        competition_id = _identifier(
            record.get("competition_id"), f"line {line_number} competition_id"
        )
        reward = record.get("reward_bb")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise TelemetryError(f"line {line_number} reward_bb must be a number")
        reward = float(reward)
        if not math.isfinite(reward):
            raise TelemetryError(f"line {line_number} reward_bb must be finite")
        if table_id in rewards:
            raise TelemetryError(f"duplicate hand result for table {table_id}")
        rewards[table_id] = (competition_id, reward)

    examples: list[TrainingExample] = []
    for line_number, record in _records(path):
        if (
            record.get("event") != "decision"
            or record.get("training_eligible") is not True
        ):
            continue
        table_id = _identifier(record.get("table_id"), f"line {line_number} table_id")
        result = rewards.get(table_id)
        if result is None:
            continue
        competition_id = _identifier(
            record.get("competition_id"), f"line {line_number} competition_id"
        )
        if result[0] != competition_id:
            raise TelemetryError(f"competition mismatch for table {table_id}")
        raw_features = record.get("features")
        if not isinstance(raw_features, list) or len(raw_features) != len(
            LEARNING_FEATURE_NAMES
        ):
            raise TelemetryError(f"line {line_number} features violate the contract")
        features = tuple(float(value) for value in raw_features)
        if not all(math.isfinite(value) for value in features):
            raise TelemetryError(f"line {line_number} features must be finite")
        raw_probabilities = record.get("behavior_probabilities")
        if not isinstance(raw_probabilities, list) or len(raw_probabilities) != len(
            LABELS
        ):
            raise TelemetryError(
                f"line {line_number} behavior probabilities violate the contract"
            )
        probabilities = tuple(float(value) for value in raw_probabilities)
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities
        ) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise TelemetryError(
                f"line {line_number} behavior probabilities violate the contract"
            )
        family = action_family(str(record.get("action") or ""))
        if family is None:
            raise TelemetryError(f"line {line_number} action is unsupported")
        risk_fraction = record.get("submitted_risk_fraction")
        if isinstance(risk_fraction, bool) or not isinstance(
            risk_fraction, (int, float)
        ):
            raise TelemetryError(
                f"line {line_number} submitted risk fraction must be a number"
            )
        risk_fraction = float(risk_fraction)
        if not math.isfinite(risk_fraction) or not 0.0 <= risk_fraction <= 1.0:
            raise TelemetryError(
                f"line {line_number} submitted risk fraction must be between 0 and 1"
            )
        state = record.get("state")
        if not isinstance(state, Mapping):
            raise TelemetryError(f"line {line_number} state must be an object")
        purse_chips = state.get("hero_purse_chips")
        big_blind = record.get("big_blind_chips")
        if (
            isinstance(purse_chips, bool)
            or not isinstance(purse_chips, (int, float))
            or not math.isfinite(float(purse_chips))
            or float(purse_chips) <= 0.0
        ):
            raise TelemetryError(f"line {line_number} hero purse must be positive")
        if (
            isinstance(big_blind, bool)
            or not isinstance(big_blind, (int, float))
            or not math.isfinite(float(big_blind))
            or float(big_blind) <= 0.0
        ):
            raise TelemetryError(f"line {line_number} big blind must be positive")
        examples.append(
            TrainingExample(
                table_id=table_id,
                policy_version=_identifier(
                    record.get("policy_version"), f"line {line_number} policy_version"
                ),
                features=features,
                action_family_index=LABELS.index(family),
                behavior_probabilities=probabilities,
                submitted_risk_fraction=risk_fraction,
                purse_bb=float(purse_chips) / float(big_blind),
                reward_bb=result[1],
                opponent_confidence=float(
                    record.get("opponent_evidence_confidence") or 0.0
                ),
            )
        )
    return tuple(examples)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TelemetryError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise TelemetryError(f"{name} must be finite")
    return number


def _sidecar_path(path: Path) -> Path:
    return Path(str(path) + ".meta.json")


def save_training_corpus(path: str | Path, examples: Sequence[TrainingExample]) -> None:
    """Persist a training corpus as float32 features plus a JSON sidecar.

    The gzip file at ``path`` holds one JSON header line (corpus and feature
    schema versions, feature count, row count) followed by the row-major
    feature matrix as little-endian float32 bytes; ``<path>.meta.json``
    carries every non-feature field per row at full precision. Features
    therefore round-trip through float32.
    """

    resolved = Path(path).expanduser().resolve()
    rows = tuple(examples)
    feature_count = len(rows[0].features) if rows else len(LEARNING_FEATURE_NAMES)
    matrix = array("f")
    for index, example in enumerate(rows):
        if len(example.features) != feature_count:
            raise TelemetryError(
                f"example {index} has {len(example.features)} features, "
                f"expected {feature_count}"
            )
        matrix.extend(example.features)
    if sys.byteorder != "little":
        matrix.byteswap()
    header = {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": feature_count,
        "rows": len(rows),
    }
    sidecar = {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "rows": [
            {
                "table_id": example.table_id,
                "policy_version": example.policy_version,
                "action_family_index": example.action_family_index,
                "behavior_probabilities": list(example.behavior_probabilities),
                "submitted_risk_fraction": example.submitted_risk_fraction,
                "purse_bb": example.purse_bb,
                "reward_bb": example.reward_bb,
                "counterfactual": example.counterfactual,
                "opponent_confidence": example.opponent_confidence,
                "decision_id": example.decision_id,
                "harvest_leg": example.harvest_leg,
                "inclusion_count": example.inclusion_count,
                "action_branch": example.action_branch,
                # Must round-trip. Without it a persisted Option A corpus is
                # indistinguishable from a legacy pre-Option-A one, and every
                # consumer that resolves an intent through the absorption map
                # silently falls back to `min(returns)` -- which under-scores
                # the `always_aggress_*` trivial baselines and makes the
                # BLOCKING promotion gate more permissive. That is the exact
                # class of failure that closed candidate 0016.
                "branch_absorption": (
                    None
                    if example.branch_absorption is None
                    else [list(pair) for pair in example.branch_absorption]
                ),
            }
            for example in rows
        ],
    }
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(resolved, "wb") as handle:
        handle.write(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        handle.write(matrix.tobytes())
    _sidecar_path(resolved).write_text(
        json.dumps(sidecar, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def load_training_corpus(path: str | Path) -> tuple[TrainingExample, ...]:
    """Reconstruct a saved corpus; features carry float32 precision."""

    resolved = Path(path).expanduser().resolve()
    with gzip.open(resolved, "rb") as handle:
        try:
            header = json.loads(handle.readline())
        except json.JSONDecodeError as exc:
            raise TelemetryError("corpus header must be one JSON line") from exc
        if not isinstance(header, Mapping):
            raise TelemetryError("corpus header must be an object")
        if header.get("corpus_schema_version") != CORPUS_SCHEMA_VERSION:
            raise TelemetryError("unsupported corpus schema version")
        if header.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise TelemetryError("corpus feature schema does not match the contract")
        row_count = _integer(header.get("rows"), "corpus rows")
        feature_count = _integer(
            header.get("feature_count"), "corpus feature_count", minimum=1
        )
        payload = handle.read()
    matrix = array("f")
    if len(payload) != row_count * feature_count * matrix.itemsize:
        raise TelemetryError("corpus feature matrix does not match its header")
    matrix.frombytes(payload)
    if sys.byteorder != "little":
        matrix.byteswap()

    sidecar_path = _sidecar_path(resolved)
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TelemetryError(f"corpus sidecar missing: {sidecar_path}") from exc
    except json.JSONDecodeError as exc:
        raise TelemetryError("corpus sidecar must be valid JSON") from exc
    if (
        not isinstance(sidecar, Mapping)
        or sidecar.get("corpus_schema_version") != CORPUS_SCHEMA_VERSION
    ):
        raise TelemetryError("unsupported corpus sidecar schema")
    metadata = sidecar.get("rows")
    if not isinstance(metadata, list) or len(metadata) != row_count:
        raise TelemetryError("corpus sidecar row count does not match the header")

    examples: list[TrainingExample] = []
    for index, row in enumerate(metadata):
        if not isinstance(row, Mapping):
            raise TelemetryError(f"corpus row {index} must be an object")
        name = f"corpus row {index}"
        start = index * feature_count
        raw_probabilities = row.get("behavior_probabilities")
        if not isinstance(raw_probabilities, list) or len(raw_probabilities) != len(
            LABELS
        ):
            raise TelemetryError(f"{name} behavior probabilities violate the contract")
        counterfactual = row.get("counterfactual")
        if not isinstance(counterfactual, bool):
            raise TelemetryError(f"{name} counterfactual must be a boolean")
        decision_id = row.get("decision_id")
        if decision_id is not None:
            decision_id = _identifier(decision_id, f"{name} decision_id")
        harvest_leg = row.get("harvest_leg")
        if harvest_leg is not None:
            harvest_leg = _identifier(harvest_leg, f"{name} harvest_leg")
        action_branch = row.get("action_branch")
        if action_branch is not None:
            action_branch = _identifier(action_branch, f"{name} action_branch")
        raw_absorption = row.get("branch_absorption")
        branch_absorption = None
        if raw_absorption is not None:
            if not isinstance(raw_absorption, list):
                raise TelemetryError(f"{name} branch_absorption must be a list")
            branch_absorption = tuple(
                (
                    _identifier(pair[0], f"{name} branch_absorption key"),
                    _identifier(pair[1], f"{name} branch_absorption value"),
                )
                for pair in raw_absorption
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            )
            if len(branch_absorption) != len(raw_absorption):
                raise TelemetryError(f"{name} branch_absorption entries must be pairs")
        examples.append(
            TrainingExample(
                table_id=_identifier(row.get("table_id"), f"{name} table_id"),
                policy_version=_identifier(
                    row.get("policy_version"), f"{name} policy_version"
                ),
                features=tuple(matrix[start : start + feature_count]),
                action_family_index=_integer(
                    row.get("action_family_index"), f"{name} action_family_index"
                ),
                behavior_probabilities=tuple(
                    _finite_number(value, f"{name} behavior probability")
                    for value in raw_probabilities
                ),
                submitted_risk_fraction=_finite_number(
                    row.get("submitted_risk_fraction"),
                    f"{name} submitted_risk_fraction",
                ),
                purse_bb=_finite_number(row.get("purse_bb"), f"{name} purse_bb"),
                reward_bb=_finite_number(row.get("reward_bb"), f"{name} reward_bb"),
                counterfactual=counterfactual,
                opponent_confidence=_finite_number(
                    row.get("opponent_confidence"), f"{name} opponent_confidence"
                ),
                decision_id=decision_id,
                harvest_leg=harvest_leg,
                inclusion_count=_optional_integer(
                    row.get("inclusion_count"), f"{name} inclusion_count"
                ),
                action_branch=action_branch,
                branch_absorption=branch_absorption,
            )
        )
    return tuple(examples)


__all__ = [
    "action_family",
    "action_response_matches",
    "CORPUS_SCHEMA_VERSION",
    "load_training_corpus",
    "load_training_examples",
    "make_decision_record",
    "MAX_REPLAY_RECEIPTS",
    "parse_replay_receipts",
    "ReplayReceipt",
    "replay_path",
    "save_training_corpus",
    "TELEMETRY_SCHEMA_VERSION",
    "TelemetryError",
    "TelemetryLog",
    "TrainingExample",
]
