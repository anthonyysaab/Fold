"""Strict PHH replay and compact features for behavior cloning.

The extractor deliberately knows only one game (``NT``/no-limit Texas
hold'em) and one prediction target (the next action family).  Features are
captured before the recorded action is applied, so cards or outcomes revealed
later in the hand cannot leak into a training example.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from warnings import catch_warnings, filterwarnings, simplefilter

from pokerkit import Card, HandHistory, State
from pokerkit.notation import parse_action

LABELS: tuple[str, ...] = ("fold", "check_call", "aggress")

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_CARD_CODES = tuple(f"{rank}{suit}" for rank in _RANKS for suit in _SUITS)
_CARD_INDEX = {code: index for index, code in enumerate(_CARD_CODES)}

_SCALAR_FEATURE_NAMES = (
    "street_preflop",
    "street_flop",
    "street_turn",
    "street_river",
    "player_count",
    "active_player_count",
    "position",
    "log_pot_bb",
    "log_stack_bb",
    "log_effective_stack_bb",
    "log_to_call_bb",
    "log_street_contribution_bb",
    "log_current_bet_bb",
    "log_min_raise_to_bb",
    "pot_odds",
    "spr",
    "raises_current_street",
    "legal_fold",
    "legal_check_call",
    "legal_aggress",
    "hole_known_fraction",
)

FEATURE_NAMES: tuple[str, ...] = (
    *(f"hole_{card}" for card in _CARD_CODES),
    *(f"board_{card}" for card in _CARD_CODES),
    *_SCALAR_FEATURE_NAMES,
)

_PLAYER_RE = re.compile(r"p([1-9][0-9]*)\Z")
_ACTION_LABELS = {"f": 0, "cc": 1, "cbr": 2}


@dataclass(frozen=True, slots=True)
class DecisionExample:
    """One decision point from a completely replayed hand."""

    features: tuple[float, ...]
    label: int
    actor_index: int


class InvalidHandHistory(ValueError):
    """Raised when a row cannot be replayed safely and completely."""


def _as_sequence(row: Mapping[str, object], name: str) -> Sequence[object]:
    try:
        value = row[name]
    except KeyError as exc:
        raise InvalidHandHistory(f"missing required field {name!r}") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise InvalidHandHistory(f"field {name!r} must be a sequence")
    return value


def _numbers(row: Mapping[str, object], name: str) -> list[float]:
    values = _as_sequence(row, name)
    parsed: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise InvalidHandHistory(f"field {name!r} item {index} is not a number")
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise InvalidHandHistory(f"field {name!r} item {index} is not a number") from exc
        if not math.isfinite(number):
            raise InvalidHandHistory(f"field {name!r} item {index} is not finite")
        parsed.append(number)
    return parsed


def _number(row: Mapping[str, object], name: str) -> float:
    try:
        value = row[name]
    except KeyError as exc:
        raise InvalidHandHistory(f"missing required field {name!r}") from exc
    if isinstance(value, bool):
        raise InvalidHandHistory(f"field {name!r} is not a number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise InvalidHandHistory(f"field {name!r} is not a number") from exc
    if not math.isfinite(number):
        raise InvalidHandHistory(f"field {name!r} is not finite")
    return number


def _actions(row: Mapping[str, object]) -> list[str]:
    values = _as_sequence(row, "actions")
    actions: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise InvalidHandHistory(f"action {index} must be a non-empty string")
        actions.append(value)
    return actions


def _hand_history(row: Mapping[str, object]) -> HandHistory:
    variant = row.get("variant")
    if variant != "NT":
        raise InvalidHandHistory(f"unsupported variant {variant!r}; expected 'NT'")

    ante_trimming_status = row.get("ante_trimming_status", False)
    if not isinstance(ante_trimming_status, bool):
        raise InvalidHandHistory("field 'ante_trimming_status' must be a boolean")

    min_bet = _number(row, "min_bet")
    if min_bet <= 0:
        raise InvalidHandHistory("field 'min_bet' must be positive")

    try:
        return HandHistory(
            variant="NT",
            ante_trimming_status=ante_trimming_status,
            antes=_numbers(row, "antes"),
            blinds_or_straddles=_numbers(row, "blinds_or_straddles"),
            min_bet=min_bet,
            starting_stacks=_numbers(row, "starting_stacks"),
            actions=_actions(row),
            parse_value=float,
        )
    except InvalidHandHistory:
        raise
    except Exception as exc:
        raise InvalidHandHistory(f"invalid PHH fields: {exc}") from exc


def _words(action: str) -> list[str]:
    words = action.split()
    if "#" in words:
        words = words[: words.index("#")]
    return words


def _decision(action: str) -> tuple[int, int] | None:
    words = _words(action)
    if len(words) < 2 or words[1] not in _ACTION_LABELS:
        return None
    match = _PLAYER_RE.fullmatch(words[0])
    if match is None:
        raise InvalidHandHistory(f"invalid actor token {words[0]!r} in action {action!r}")
    return int(match.group(1)) - 1, _ACTION_LABELS[words[1]]


def _known_card(card: Card) -> bool:
    return card.rank.value != "?" and card.suit.value != "?"


def _card_bits(cards: Sequence[Card]) -> tuple[float, ...]:
    bits = [0.0] * len(_CARD_CODES)
    for card in cards:
        if not _known_card(card):
            continue
        code = f"{card.rank.value}{card.suit.value}"
        try:
            bits[_CARD_INDEX[code]] = 1.0
        except KeyError as exc:  # Defensive: NT should only contain the standard 52-card deck.
            raise InvalidHandHistory(f"unexpected card {code!r} in an NT hand") from exc
    return tuple(bits)


def _log_bb(amount: float, big_blind: float) -> float:
    return math.log1p(max(0.0, amount) / big_blind)


def _features(state: State, actor_index: int, big_blind: float) -> tuple[float, ...]:
    if state.actor_index != actor_index:
        raise InvalidHandHistory(
            f"recorded actor p{actor_index + 1} does not match "
            f"PokerKit actor {state.actor_index!r}"
        )
    if state.street_index not in range(4):
        raise InvalidHandHistory(f"invalid NT street index {state.street_index!r}")

    hole_cards = tuple(state.hole_cards[actor_index])
    if len(hole_cards) != 2:
        raise InvalidHandHistory(
            f"actor p{actor_index + 1} has {len(hole_cards)} hole cards; expected two"
        )
    known_hole_count = sum(_known_card(card) for card in hole_cards)
    board_cards = tuple(state.get_board_cards(0))

    player_count = len(state.stacks)
    active_player_count = sum(state.statuses)
    position = actor_index / max(1, player_count - 1)
    pot = float(state.total_pot_amount)
    stack = float(state.stacks[actor_index])
    effective_stack = float(state.get_effective_stack(actor_index))
    to_call = float(state.checking_or_calling_amount or 0.0)
    street_contribution = float(state.bets[actor_index])
    current_bet = float(max(state.bets, default=0.0))
    min_raise_to = float(state.min_completion_betting_or_raising_to_amount or 0.0)

    with catch_warnings():
        # Cash-game PokerKit warns, but permits, a fold when checking is available.
        simplefilter("ignore", UserWarning)
        legal_fold = state.can_fold()
    legal_check_call = state.can_check_or_call()
    legal_aggress = state.can_complete_bet_or_raise_to()
    if not (legal_fold or legal_check_call or legal_aggress):
        raise InvalidHandHistory("PokerKit reports no legal action for the recorded actor")

    street = tuple(float(index == state.street_index) for index in range(4))
    scalars = (
        *street,
        float(player_count),
        float(active_player_count),
        float(position),
        _log_bb(pot, big_blind),
        _log_bb(stack, big_blind),
        _log_bb(effective_stack, big_blind),
        _log_bb(to_call, big_blind),
        _log_bb(street_contribution, big_blind),
        _log_bb(current_bet, big_blind),
        _log_bb(min_raise_to, big_blind),
        0.0 if to_call <= 0 else to_call / (pot + to_call),
        0.0 if pot <= 0 else effective_stack / pot,
        float(state.completion_betting_or_raising_count),
        float(legal_fold),
        float(legal_check_call),
        float(legal_aggress),
        known_hole_count / 2.0,
    )
    features = (*_card_bits(hole_cards), *_card_bits(board_cards), *scalars)
    if len(features) != len(FEATURE_NAMES):  # pragma: no cover - module invariant.
        raise AssertionError("feature names and values are out of sync")
    if not all(math.isfinite(value) for value in features):
        raise InvalidHandHistory("non-finite feature produced during replay")
    return features


def extract_decisions(
    row: Mapping[str, object],
    *,
    require_known_hole_cards: bool = True,
) -> list[DecisionExample]:
    """Replay one PHH row and return leakage-safe, pre-action examples.

    Examples are accumulated privately and returned only after every action has
    replayed and the hand has reached a terminal state.  An unknown actor hand
    is skipped by default; setting ``require_known_hole_cards=False`` includes
    it with zeroes for unknown cards and an explicit known-card fraction.
    """

    hand_history = _hand_history(row)
    examples: list[DecisionExample] = []

    try:
        state = hand_history.create_state()
        for action_index, action in enumerate(hand_history.actions):
            words = _words(action)
            if words[:2] == ["d", "db"] and state.can_burn_card():
                # PHH records board deals but represents burns through automation.
                state.burn_card("??")

            decision = _decision(action)
            if decision is not None:
                actor_index, label = decision
                if state.actor_index != actor_index:
                    raise InvalidHandHistory(
                        f"action {action_index} {action!r}: recorded actor "
                        f"p{actor_index + 1} does not match PokerKit actor "
                        f"{state.actor_index!r}"
                    )
                features = _features(state, actor_index, float(hand_history.min_bet))
                hole_known_fraction = features[FEATURE_NAMES.index("hole_known_fraction")]
                if not require_known_hole_cards or hole_known_fraction == 1.0:
                    examples.append(DecisionExample(features, label, actor_index))

            try:
                with catch_warnings():
                    filterwarnings(
                        "ignore",
                        message="There is no reason for this player to fold\\.",
                        category=UserWarning,
                    )
                    parse_action(state, action, float)
            except Exception as exc:
                raise InvalidHandHistory(
                    f"action {action_index} {action!r} failed to replay: {exc}"
                ) from exc

        if state.status:
            raise InvalidHandHistory("hand ended before PokerKit reached a terminal state")
    except InvalidHandHistory:
        raise
    except Exception as exc:
        raise InvalidHandHistory(f"hand failed to replay: {exc}") from exc

    return examples


def _source_group(row: Mapping[str, object]) -> str:
    source_file = row.get("source_file")
    if not isinstance(source_file, str) or not source_file.strip():
        raise InvalidHandHistory("field 'source_file' must be a non-empty string")
    source_file = source_file.strip()
    if "pluribus" not in source_file.lower():
        return source_file

    normalized = source_file.replace("\\", "/").rstrip("/")
    if "/" not in normalized:
        raise InvalidHandHistory("a Pluribus source_file must include its match directory")
    return normalized.rsplit("/", 1)[0]


def split_for_row(row: Mapping[str, object], seed: int = 7) -> str:
    """Assign a source group to a deterministic 80/10/10 split."""

    group = _source_group(row)
    digest = hashlib.sha256(f"{seed}\0{group}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "val"
    return "test"


__all__ = [
    "FEATURE_NAMES",
    "LABELS",
    "DecisionExample",
    "InvalidHandHistory",
    "extract_decisions",
    "split_for_row",
]
