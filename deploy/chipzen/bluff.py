"""Standalone poker bluff advisor.

This script does not import or modify the live agent. It reads one decision
situation and answers whether a bluff is advisable, which kind, and at what
size, with every input to that answer exposed for inspection. The decision
engine will consume :func:`evaluate_bluff` later; until then the module is a
manual gauge like ``risk_temperature.py``.

The advisor encodes the standard bluffing playbook:

* Fold-equity price: a bluff risking ``b`` into ``pot`` profits only when
  everyone folds more often than the breakeven ``b / (pot + b)``; equity when
  called (a draw) lowers that requirement, which is why semi-bluffs come
  first.
* Bluff mostly heads-up: each extra caller multiplies the chance someone
  continues, so multiway bluffs are gated off.
* Board texture: dry, disconnected boards fold more; wet boards hit the
  caller's range.
* Blockers: holding the card the nuts needs (the bare ace on a three-flush
  board) removes opponents' strongest continues and licenses thin river
  bluffs.
* Credibility: a bluff must finish a believable story -- continuation bets
  and barrels by the aggressor work; bluff-raising into strength rarely does.
* Discipline: capped barrels, capped stack risk, no bluffing stacks that are
  already pot-committed, and a deterministic mixed-strategy roll so the same
  spot is not bluffed every time.
* Unpredictability: the roll is a salted hash of the situation (and any
  ``mix_key`` such as a table id), so opponents cannot pattern the mixing
  while every decision stays reproducible for telemetry and training.
* Table standing: ``bluff_density`` scales every frequency cap, and
  ``lead_density_gain`` couples it to the ``lead_position.py`` gradient, so
  training can learn whether the chip leader or the short stack should
  bluff more. The neutral default assumes neither until the data says so.

Every tuning field of :class:`BluffSettings` is a future learned parameter.

Example:
    python bluff.py --hole Ah Kh --board 7h 2h 9c --street flop \
        --pot 100 --stack 1000 --opponents 1 --hero-aggressions 1
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_RANK_VALUES = {rank: index + 2 for index, rank in enumerate(_RANKS)}
_BOARD_SIZES = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
_FLUSH_BLOCKER_SCORES = {14: 0.8, 13: 0.5, 12: 0.3}
MIN_OPPONENTS = 1
MAX_OPPONENTS = 5
MAX_DRAW_OUTS = 15.0
BLUFF_KINDS = ("steal", "continuation", "barrel", "probe", "raise_bluff")


def _bounded(name: str, value: float, low: float, high: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}")
    return number


def _int_input(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class BluffSettings:
    """Bluffing knobs, each with a documented range and a learnable meaning.

    Probabilities and frequencies live in ``[0, 1]``; pot fractions in
    ``(0, 2]``; weights in ``[0, 0.5]``. ``validate`` runs on construction.
    """

    # Hard discipline gates.
    max_opponents: int = 2  # never bluff into more than this many players
    max_barrels: int = 2  # aggressive actions this hand before giving up
    max_risk_fraction: float = 0.33  # of the remaining stack per bluff
    min_stack_to_pot: float = 0.75  # below this SPR opponents are committed
    max_raise_bluff_facing: float = 0.60  # largest bet/pot worth bluff-raising
    min_river_blocker: float = 0.40  # river pure bluffs need this or a story
    min_margin: float = 0.02  # estimated folds must beat breakeven by this

    # Per-opponent fold-probability model. Calibrated 2026-08-12 from the
    # foreign top-15 corpus (facing-a-bet responses, shrunk toward the old
    # theory priors by sample size): preflop 67.1% of 3,633, flop 52.8% of
    # 299, turn 42.4% of 132, river 65.6% of 61 (heavily shrunk).
    preflop_fold: float = 0.67  # was 0.55
    flop_fold: float = 0.52  # was 0.45
    turn_fold: float = 0.42
    river_fold: float = 0.55  # was 0.38
    dryness_weight: float = 0.20  # dry boards fold more, wet boards less
    blocker_weight: float = 0.10  # blockers remove continuing combos
    size_fold_bonus: float = 0.06  # extra folds per pot-fraction above half
    aggressor_stickiness: float = 0.15  # a bettor rarely folds to a raise
    position_bonus: float = 0.04  # in position folds up, out of position down
    wildness_fold_penalty: float = 0.35  # observed maniacs fold less

    # Equity when called (the semi-bluff rebate).
    out_equity_flop: float = 0.035  # equity per out with two streets behind
    out_equity_turn: float = 0.020  # equity per out with one street behind
    overcard_out_weight: float = 0.5  # overcards are discounted outs
    backdoor_outs: float = 1.0  # flop-only three-flush credit
    big_draw_outs: float = 8.0  # at least this many outs makes a semi-bluff
    preflop_equity_base: float = 0.25
    preflop_equity_per_chen: float = 0.015
    min_steal_chen: float = 5.0  # weaker hands do not open-steal
    value_chen: float = 9.0  # stronger hands raise for value, not as bluffs

    # Mixed-strategy frequency caps by bluff kind.
    steal_frequency: float = 0.70
    continuation_frequency: float = 0.65
    semi_bluff_frequency: float = 0.80
    barrel_frequency: float = 0.50
    probe_frequency: float = 0.40
    raise_bluff_frequency: float = 0.30
    river_frequency: float = 0.35

    # Global bluffing density and its coupling to the lead gradient.
    # Effective density is ``bluff_density * (1 + lead_density_gain * lead)``
    # with lead normalized to [-1, +1]; a positive gain bluffs the chip
    # leader more, a negative gain bluffs the trailer more. Both are the
    # primary learned parameters for that question.
    bluff_density: float = 1.0
    lead_density_gain: float = 0.0

    # Proposed sizes as fractions of the pot being attacked.
    steal_pot_fraction: float = 1.00
    continuation_pot_fraction: float = 0.40
    semi_bluff_pot_fraction: float = 0.60
    barrel_pot_fraction: float = 0.66
    probe_pot_fraction: float = 0.50
    raise_bluff_pot_fraction: float = 1.00
    river_pot_fraction: float = 0.75

    salt: str = "fold-ver-4-bluff"

    def __post_init__(self) -> None:
        for name in (
            "preflop_fold",
            "flop_fold",
            "turn_fold",
            "river_fold",
            "min_margin",
            "min_river_blocker",
            "max_raise_bluff_facing",
            "steal_frequency",
            "continuation_frequency",
            "semi_bluff_frequency",
            "barrel_frequency",
            "probe_frequency",
            "raise_bluff_frequency",
            "river_frequency",
        ):
            _bounded(name, getattr(self, name), 0.0, 1.0)
        for name in (
            "dryness_weight",
            "blocker_weight",
            "size_fold_bonus",
            "aggressor_stickiness",
            "position_bonus",
            "wildness_fold_penalty",
        ):
            _bounded(name, getattr(self, name), 0.0, 0.5)
        for name in (
            "steal_pot_fraction",
            "continuation_pot_fraction",
            "semi_bluff_pot_fraction",
            "barrel_pot_fraction",
            "probe_pot_fraction",
            "raise_bluff_pot_fraction",
            "river_pot_fraction",
        ):
            value = _bounded(name, getattr(self, name), 0.0, 2.0)
            if value == 0.0:
                raise ValueError(f"{name} must be greater than zero")
        _bounded("max_risk_fraction", self.max_risk_fraction, 0.01, 1.0)
        _bounded("min_stack_to_pot", self.min_stack_to_pot, 0.0, 10.0)
        _bounded("out_equity_flop", self.out_equity_flop, 0.005, 0.10)
        _bounded("out_equity_turn", self.out_equity_turn, 0.005, 0.10)
        _bounded("overcard_out_weight", self.overcard_out_weight, 0.0, 1.0)
        _bounded("backdoor_outs", self.backdoor_outs, 0.0, 4.0)
        _bounded("big_draw_outs", self.big_draw_outs, 0.0, MAX_DRAW_OUTS)
        _bounded("preflop_equity_base", self.preflop_equity_base, 0.0, 0.5)
        _bounded("preflop_equity_per_chen", self.preflop_equity_per_chen, 0.0, 0.05)
        _bounded("min_steal_chen", self.min_steal_chen, -2.0, 20.0)
        _bounded("value_chen", self.value_chen, -2.0, 22.0)
        _bounded("bluff_density", self.bluff_density, 0.0, 2.0)
        _bounded("lead_density_gain", self.lead_density_gain, -1.0, 1.0)
        if self.value_chen <= self.min_steal_chen:
            raise ValueError("value_chen must exceed min_steal_chen")
        for name in ("max_opponents", "max_barrels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_opponents > MAX_OPPONENTS:
            raise ValueError(f"max_opponents cannot exceed {MAX_OPPONENTS}")
        if not isinstance(self.salt, str) or not self.salt:
            raise ValueError("salt must be a non-empty string")

    def to_mapping(self) -> dict[str, object]:
        """JSON-ready form for a future learned-artifact manifest."""

        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, mapping: "dict[str, object]") -> "BluffSettings":
        """Rebuild validated settings from artifact JSON; unknown keys fail."""

        return cls(**dict(mapping))


DEFAULT_BLUFF_SETTINGS = BluffSettings()


@dataclasses.dataclass(frozen=True, slots=True)
class BluffAdvice:
    """One bluff reading plus every number needed to explain it."""

    bluff: bool
    kind: str
    action: str | None
    pot_fraction: float | None
    risk_chips: int
    bluff_score: float
    semi_bluff: bool
    outs: float
    semi_bluff_equity: float
    estimated_fold_probability: float
    required_fold_probability: float
    margin: float
    expected_value_chips: float
    frequency_cap: float
    frequency_roll: float
    factors: dict[str, float]
    reasons: tuple[str, ...]
    inputs: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "bluff": self.bluff,
            "kind": self.kind,
            "action": self.action,
            "sizing": {
                "pot_fraction": self.pot_fraction,
                "risk_chips": self.risk_chips,
            },
            "score": self.bluff_score,
            "math": {
                "outs": round(self.outs, 2),
                "semi_bluff": self.semi_bluff,
                "semi_bluff_equity": round(self.semi_bluff_equity, 4),
                "estimated_fold_probability": round(
                    self.estimated_fold_probability, 4
                ),
                "required_fold_probability": round(
                    self.required_fold_probability, 4
                ),
                "margin": round(self.margin, 4),
                "expected_value_chips": round(self.expected_value_chips, 2),
            },
            "mixing": {
                "frequency_cap": round(self.frequency_cap, 4),
                "roll": round(self.frequency_roll, 6),
            },
            "factors": {name: round(value, 4) for name, value in self.factors.items()},
            "reasons": list(self.reasons),
            "inputs": dict(self.inputs),
        }


def _normalize_card(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("cards must be strings")
    token = value.strip()
    if len(token) == 3 and token[:2] == "10":
        token = f"T{token[2]}"
    if len(token) != 2:
        raise ValueError(f"invalid card {value!r}")
    card = f"{token[0].upper()}{token[1].lower()}"
    if card[0] not in _RANKS or card[1] not in _SUITS:
        raise ValueError(f"invalid card {value!r}")
    return card


def _with_wheel_alias(values: set[int]) -> set[int]:
    return values | ({1} if 14 in values else set())


def _straight_windows() -> tuple[frozenset[int], ...]:
    return tuple(frozenset(range(start, start + 5)) for start in range(1, 11))


def _made_flush(hole: tuple[str, str], board: tuple[str, ...]) -> bool:
    for suit in _SUITS:
        hero = sum(card[1] == suit for card in hole)
        total = hero + sum(card[1] == suit for card in board)
        if total >= 5 and hero >= 1:
            return True
    return False


def _made_straight(hole: tuple[str, str], board: tuple[str, ...]) -> bool:
    present = _with_wheel_alias(
        {_RANK_VALUES[card[0]] for card in (*hole, *board)}
    )
    board_values = _with_wheel_alias({_RANK_VALUES[card[0]] for card in board})
    hero_only = present - board_values
    return any(
        window <= present and window & hero_only
        for window in _straight_windows()
    )


def _flush_draw_outs(hole: tuple[str, str], board: tuple[str, ...]) -> float:
    for suit in _SUITS:
        hero = sum(card[1] == suit for card in hole)
        if hero >= 1 and hero + sum(card[1] == suit for card in board) == 4:
            return 9.0
    return 0.0


def _backdoor_flush_outs(
    hole: tuple[str, str], board: tuple[str, ...], settings: BluffSettings
) -> float:
    if len(board) != 3:
        return 0.0
    for suit in _SUITS:
        hero = [card for card in hole if card[1] == suit]
        total = len(hero) + sum(card[1] == suit for card in board)
        if total != 3 or not hero:
            continue
        if len(hero) == 2:
            return settings.backdoor_outs
        if hero[0][0] == "A":
            return 0.5 * settings.backdoor_outs
    return 0.0


def _straight_draw_outs(hole: tuple[str, str], board: tuple[str, ...]) -> float:
    present = _with_wheel_alias(
        {_RANK_VALUES[card[0]] for card in (*hole, *board)}
    )
    board_values = _with_wheel_alias({_RANK_VALUES[card[0]] for card in board})
    hero_only = present - board_values
    completing_ranks = set()
    for rank in range(2, 15):
        if rank in present:
            continue
        added = {rank} | ({1} if rank == 14 else set())
        augmented = present | added
        for window in _straight_windows():
            if window <= augmented and window & added and window & hero_only:
                completing_ranks.add(rank)
                break
    return 4.0 * len(completing_ranks)


def _overcard_outs(
    hole: tuple[str, str], board: tuple[str, ...], settings: BluffSettings
) -> float:
    if not board:
        return 0.0
    board_top = max(_RANK_VALUES[card[0]] for card in board)
    values = sorted((_RANK_VALUES[card[0]] for card in hole), reverse=True)
    overs = [value for value in values if value > board_top]
    if len(overs) == 2 and values[0] != values[1]:
        return 6.0 * settings.overcard_out_weight
    if len(overs) == 1 and overs[0] >= 13:
        return 1.5 * settings.overcard_out_weight
    return 0.0


def _draw_outs(
    hole: tuple[str, str],
    board: tuple[str, ...],
    street: str,
    settings: BluffSettings,
) -> float:
    if street in ("preflop", "river"):
        return 0.0
    outs = (
        _flush_draw_outs(hole, board)
        + _straight_draw_outs(hole, board)
        + _overcard_outs(hole, board, settings)
        + _backdoor_flush_outs(hole, board, settings)
    )
    return min(MAX_DRAW_OUTS, outs)


def _has_showdown_value(hole: tuple[str, str], board: tuple[str, ...]) -> bool:
    """Pairs, made flushes, and made straights are not bluffing hands."""

    if hole[0][0] == hole[1][0]:
        return True
    board_ranks = {card[0] for card in board}
    if any(card[0] in board_ranks for card in hole):
        return True
    return _made_flush(hole, board) or _made_straight(hole, board)


def _blocker_score(hole: tuple[str, str], board: tuple[str, ...]) -> float:
    score = 0.0
    for suit in _SUITS:
        if sum(card[1] == suit for card in board) < 3:
            continue
        for card in hole:
            if card[1] == suit:
                value = _FLUSH_BLOCKER_SCORES.get(_RANK_VALUES[card[0]], 0.0)
                score = max(score, value)
    board_values = _with_wheel_alias({_RANK_VALUES[card[0]] for card in board})
    hero_values = _with_wheel_alias({_RANK_VALUES[card[0]] for card in hole})
    for window in _straight_windows():
        board_in = len(window & board_values)
        hero_in = len(window & (hero_values - board_values))
        if board_in == 3 and hero_in == 1:
            score = max(score, 0.3)
    return min(1.0, score)


def _board_dryness(board: tuple[str, ...]) -> float:
    """1.0 is bone dry; 0.0 is soaking wet. Empty boards read neutral."""

    if not board:
        return 0.5
    wetness = 0.0
    suit_max = max(sum(card[1] == suit for card in board) for suit in _SUITS)
    if suit_max >= 3:
        wetness += 0.40
    elif suit_max == 2:
        wetness += 0.20
    values = _with_wheel_alias({_RANK_VALUES[card[0]] for card in board})
    connected = max(len(window & values) for window in _straight_windows())
    if connected >= 4:
        wetness += 0.40
    elif connected == 3:
        wetness += 0.25
    ranks = [card[0] for card in board]
    if len(set(ranks)) != len(ranks):
        wetness -= 0.10
    if max(_RANK_VALUES[rank] for rank in ranks) >= 12:
        wetness -= 0.10
    return min(1.0, max(0.0, 1.0 - wetness))


def _chen_score(hole: tuple[str, str]) -> float:
    """Simplified Chen preflop score; higher is stronger."""

    high, low = sorted((_RANK_VALUES[card[0]] for card in hole), reverse=True)
    points = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}.get(high, high / 2.0)
    if high == low:
        return max(5.0, 2.0 * points)
    if hole[0][1] == hole[1][1]:
        points += 2.0
    gap = high - low - 1
    points -= {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}.get(gap, 5.0)
    if gap <= 1 and high < 12:
        points += 1.0
    return math.ceil(points) if points % 1 else points


def _classify(street: str, to_call: int, hero_aggressions: int) -> str:
    if street == "preflop":
        return "steal"
    if to_call > 0:
        return "raise_bluff"
    if hero_aggressions >= 1:
        return "continuation" if street == "flop" else "barrel"
    return "probe"


def _frequency_and_fraction(
    kind: str, street: str, semi_bluff: bool, settings: BluffSettings
) -> tuple[float, float]:
    frequency = getattr(settings, f"{kind}_frequency")
    fraction = getattr(settings, f"{kind}_pot_fraction")
    if semi_bluff:
        frequency = max(frequency, settings.semi_bluff_frequency)
        fraction = max(fraction, settings.semi_bluff_pot_fraction)
    if street == "river":
        frequency = min(frequency, settings.river_frequency)
        fraction = settings.river_pot_fraction
    return frequency, fraction


def _mix_roll(salt: str, key: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def evaluate_bluff(
    *,
    hole_cards: tuple[str, str] | list[str],
    board_cards: tuple[str, ...] | list[str] = (),
    street: str,
    pot: int,
    to_call: int = 0,
    stack: int,
    opponents: int,
    hero_aggressions: int = 0,
    opponent_aggressions: int = 0,
    in_position: bool | None = None,
    lead_position: float | None = None,
    opponent_wildness: float | None = None,
    mix_key: str | None = None,
    settings: BluffSettings = DEFAULT_BLUFF_SETTINGS,
) -> BluffAdvice:
    """Judge one bluffing opportunity without touching Arena or the engine.

    ``pot`` is the pot being attacked, including any bet in front of us;
    ``to_call`` is that bet's unpaid part; ``stack`` is the effective stack
    still behind. ``hero_aggressions`` and ``opponent_aggressions`` count
    aggressive actions this hand. ``lead_position`` is the -100..+100
    reading from ``lead_position.py``; ``None`` applies no density coupling.
    ``opponent_wildness`` is the observed aggression frequency (0-1) of the
    stickiest live opponent from the session opponent model; wild players
    fold less, so it shrinks the estimated folds. The advice never exceeds
    server legality because it proposes only a pot fraction; the engine
    clamps later.

    Pass a per-hand ``mix_key`` (the Arena table id) in live play so the
    same holding mixes differently every hand; without one, the roll still
    varies with the cards, board, street, pot, price, and opponent count.
    """

    street = str(street).casefold()
    if street not in _BOARD_SIZES:
        raise ValueError(f"street must be one of: {', '.join(_BOARD_SIZES)}")
    hole = tuple(_normalize_card(card) for card in hole_cards)
    if len(hole) != 2:
        raise ValueError("hole_cards must contain exactly 2 cards")
    board = tuple(_normalize_card(card) for card in board_cards)
    if len(board) != _BOARD_SIZES[street]:
        raise ValueError(
            f"{street} requires exactly {_BOARD_SIZES[street]} board cards"
        )
    if len({*hole, *board}) != len(hole) + len(board):
        raise ValueError("hole and board cards contain duplicates")
    pot = _int_input("pot", pot, minimum=1)
    to_call = _int_input("to_call", to_call)
    stack = _int_input("stack", stack, minimum=1)
    opponents = _int_input("opponents", opponents, minimum=MIN_OPPONENTS)
    if opponents > MAX_OPPONENTS:
        raise ValueError(f"opponents cannot exceed {MAX_OPPONENTS}")
    hero_aggressions = _int_input("hero_aggressions", hero_aggressions)
    opponent_aggressions = _int_input("opponent_aggressions", opponent_aggressions)
    if in_position is not None and not isinstance(in_position, bool):
        raise ValueError("in_position must be True, False, or None")
    if lead_position is None:
        lead = 0.0
    else:
        lead = _bounded("lead_position", lead_position, -100.0, 100.0) / 100.0
    if opponent_wildness is None:
        wildness = 0.0
    else:
        wildness = _bounded("opponent_wildness", opponent_wildness, 0.0, 1.0)

    kind = _classify(street, to_call, hero_aggressions)
    outs = _draw_outs(hole, board, street, settings)
    semi_bluff = outs >= settings.big_draw_outs
    chen = _chen_score(hole)
    if street == "preflop":
        equity_when_called = min(
            0.5,
            settings.preflop_equity_base
            + settings.preflop_equity_per_chen * max(0.0, chen),
        )
    else:
        per_out = (
            settings.out_equity_flop
            if street == "flop"
            else settings.out_equity_turn
        )
        equity_when_called = min(0.55, outs * per_out)

    dryness = _board_dryness(board)
    blockers = _blocker_score(hole, board)

    frequency, fraction = _frequency_and_fraction(kind, street, semi_bluff, settings)
    # The learned density scales how often every kind of bluff fires, and
    # the lead gradient tilts it toward the leader or the trailer.
    density = settings.bluff_density * (1.0 + settings.lead_density_gain * lead)
    frequency = min(1.0, max(0.0, frequency * density))
    if to_call > 0:
        risk_chips = to_call + round(fraction * (pot + to_call))
    else:
        risk_chips = max(1, round(fraction * pot))

    base_fold = getattr(settings, f"{street}_fold")
    fold_single = base_fold + settings.dryness_weight * (dryness - 0.5)
    fold_single += settings.blocker_weight * blockers
    fold_single += settings.size_fold_bonus * min(1.0, max(0.0, fraction - 0.5))
    if to_call > 0 and opponent_aggressions >= 1:
        fold_single -= settings.aggressor_stickiness
    if in_position is True:
        fold_single += settings.position_bonus
    elif in_position is False:
        fold_single -= settings.position_bonus
    fold_single -= settings.wildness_fold_penalty * wildness
    fold_single = min(0.95, max(0.05, fold_single))
    fold_all = fold_single**opponents

    called_pot = pot + 2 * risk_chips
    called_ev = equity_when_called * called_pot - risk_chips
    if called_ev >= 0:
        required_fold = 0.0
    else:
        required_fold = -called_ev / (pot - called_ev)
    margin = fold_all - required_fold
    expected_value = fold_all * pot + (1.0 - fold_all) * called_ev

    key = mix_key or (
        f"{'.'.join(sorted(hole))}|{'.'.join(board)}|{street}"
        f"|{pot}|{to_call}|{opponents}"
    )
    roll = _mix_roll(settings.salt, key)

    reasons: list[str] = []
    if opponents > settings.max_opponents:
        reasons.append(
            f"{opponents} opponents; bluffs are capped at {settings.max_opponents}"
        )
    if street != "preflop" and _has_showdown_value(hole, board):
        reasons.append("hand has showdown value; bet it for value or check")
    if street == "preflop" and chen >= settings.value_chen:
        reasons.append("premium hand: raise it for value, not as a bluff")
    if street == "preflop" and chen < settings.min_steal_chen:
        reasons.append("hand too weak to steal profitably")
    if hero_aggressions >= settings.max_barrels and not semi_bluff:
        reasons.append("barrel cap reached without a big draw")
    if stack < settings.min_stack_to_pot * pot:
        reasons.append("stack too shallow for fold equity; opponents are committed")
    if risk_chips > settings.max_risk_fraction * stack:
        reasons.append("bluff would risk too much of the remaining stack")
    if kind == "raise_bluff" and to_call > settings.max_raise_bluff_facing * pot:
        reasons.append("facing bet too large to bluff-raise")
    if (
        street == "river"
        and blockers < settings.min_river_blocker
        and hero_aggressions == 0
    ):
        reasons.append("river bluff needs blockers or an earlier betting story")
    structural = bool(reasons)
    if margin < settings.min_margin:
        reasons.append("estimated folds do not beat the breakeven price")
    if not reasons and roll >= frequency:
        reasons.append("mixed strategy withholds this combo this time")

    bluff = not reasons
    if structural:
        score = 0.0
    else:
        score = round(100.0 * min(1.0, max(0.0, 0.5 + 2.5 * margin)), 1)
    if bluff:
        reasons.append(
            f"{'semi-bluff' if semi_bluff else 'bluff'} {kind}: "
            f"estimated folds {fold_all:.0%} beat breakeven {required_fold:.0%}"
        )

    return BluffAdvice(
        bluff=bluff,
        kind=kind,
        action=("raise" if to_call > 0 else "bet") if bluff else None,
        pot_fraction=fraction if bluff else None,
        risk_chips=risk_chips,
        bluff_score=score,
        semi_bluff=semi_bluff,
        outs=outs,
        semi_bluff_equity=equity_when_called,
        estimated_fold_probability=fold_all,
        required_fold_probability=required_fold,
        margin=margin,
        expected_value_chips=expected_value,
        frequency_cap=frequency,
        frequency_roll=roll,
        factors={
            "fold_probability_single": fold_single,
            "dryness": dryness,
            "blockers": blockers,
            "chen": chen,
            "outs": outs,
            "lead": lead,
            "bluff_density": density,
            "opponent_wildness": wildness,
        },
        reasons=tuple(reasons),
        inputs={
            "hole_cards": list(hole),
            "board_cards": list(board),
            "street": street,
            "pot_chips": pot,
            "to_call_chips": to_call,
            "stack_chips": stack,
            "opponents": opponents,
            "hero_aggressions": hero_aggressions,
            "opponent_aggressions": opponent_aggressions,
            "in_position": in_position,
            "lead_position": lead_position,
            "opponent_wildness": opponent_wildness,
        },
    )


def render(advice: BluffAdvice) -> str:
    width = 20
    filled = round(advice.bluff_score / 100.0 * width)
    gauge = "#" * filled + "." * (width - filled)
    verdict = "BLUFF" if advice.bluff else "NO BLUFF"
    lines = [
        f"Bluff advice: {verdict} ({advice.kind})",
        f"Score: {advice.bluff_score:.1f}/100 [{gauge}]",
    ]
    if advice.bluff:
        lines.append(
            f"Action: {advice.action} {advice.pot_fraction:.0%} of pot "
            f"(~{advice.risk_chips} chips at risk)"
        )
    lines.extend(
        (
            "Math:",
            f"  estimated folds    {advice.estimated_fold_probability:6.1%}",
            f"  breakeven folds    {advice.required_fold_probability:6.1%}",
            f"  margin             {advice.margin:+6.1%}",
            f"  outs / equity      {advice.outs:.1f} / "
            f"{advice.semi_bluff_equity:.1%} when called",
            f"  EV if taken        {advice.expected_value_chips:+.1f} chips",
            f"  mix roll           {advice.frequency_roll:.3f} vs "
            f"cap {advice.frequency_cap:.2f}",
            "Reasons:",
        )
    )
    lines.extend(f"  - {reason}" for reason in advice.reasons)
    return "\n".join(lines)


def self_test() -> None:
    always = dataclasses.replace(
        DEFAULT_BLUFF_SETTINGS,
        steal_frequency=1.0,
        continuation_frequency=1.0,
        semi_bluff_frequency=1.0,
        barrel_frequency=1.0,
        probe_frequency=1.0,
        raise_bluff_frequency=1.0,
        river_frequency=1.0,
    )

    # A heads-up combo-draw continuation bet is the model semi-bluff.
    combo = evaluate_bluff(
        hole_cards=("Ah", "Kh"),
        board_cards=("7h", "2h", "9c"),
        street="flop",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=1,
        in_position=True,
        settings=always,
    )
    assert combo.bluff and combo.semi_bluff and combo.kind == "continuation"
    assert combo.required_fold_probability == 0.0  # the draw bets for itself
    assert round(combo.estimated_fold_probability, 3) == 0.626
    assert combo.action == "bet" and combo.pot_fraction == 0.60

    # The same spot three-way is disqualified outright.
    multiway = evaluate_bluff(
        hole_cards=("Ah", "Kh"),
        board_cards=("7h", "2h", "9c"),
        street="flop",
        pot=100,
        stack=1_000,
        opponents=3,
        hero_aggressions=1,
        in_position=True,
        settings=always,
    )
    assert not multiway.bluff and multiway.bluff_score == 0.0
    assert any("opponents" in reason for reason in multiway.reasons)

    # Two opponents fold less often together than one.
    two_way = evaluate_bluff(
        hole_cards=("Ah", "Kh"),
        board_cards=("7h", "2h", "9c"),
        street="flop",
        pot=100,
        stack=1_000,
        opponents=2,
        hero_aggressions=1,
        in_position=True,
        settings=always,
    )
    assert two_way.estimated_fold_probability < combo.estimated_fold_probability

    # On the river the bare ace of the flush suit licenses the bluff...
    blocker = evaluate_bluff(
        hole_cards=("Ah", "Qd"),
        board_cards=("7h", "2h", "9h", "Jc", "3s"),
        street="river",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=1,
        in_position=True,
        settings=always,
    )
    assert blocker.bluff and blocker.factors["blockers"] == 0.8

    # ...a bare hand with the same story still prices in against arena
    # over-folding, but with less margin than the blocker gives...
    bare = evaluate_bluff(
        hole_cards=("Qd", "4c"),
        board_cards=("7h", "2h", "9h", "Jc", "3s"),
        street="river",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=1,
        in_position=True,
        settings=always,
    )
    assert bare.bluff and blocker.margin > bare.margin

    # ...without blockers or a story the river is abandoned...
    no_story = evaluate_bluff(
        hole_cards=("Qd", "4c"),
        board_cards=("7h", "2h", "9h", "Jc", "3s"),
        street="river",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=0,
        in_position=True,
        settings=always,
    )
    assert not no_story.bluff
    assert any("betting story" in reason for reason in no_story.reasons)

    # ...and a proven maniac kills even the blocker bluff.
    vs_maniac = evaluate_bluff(
        hole_cards=("Ah", "Qd"),
        board_cards=("7h", "2h", "9h", "Jc", "3s"),
        street="river",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=1,
        in_position=True,
        opponent_wildness=0.8,
        settings=always,
    )
    assert not vs_maniac.bluff
    assert any("breakeven" in reason for reason in vs_maniac.reasons)

    # Hands with showdown value never bluff.
    pair = evaluate_bluff(
        hole_cards=("9d", "9s"),
        board_cards=("7h", "2h", "Kc"),
        street="flop",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=1,
        settings=always,
    )
    assert not pair.bluff
    assert any("showdown value" in reason for reason in pair.reasons)

    # Preflop: medium hands steal, premiums and trash do not.
    steal = evaluate_bluff(
        hole_cards=("Kh", "Th"),
        street="preflop",
        pot=150,
        to_call=100,
        stack=2_000,
        opponents=2,
        settings=always,
    )
    assert steal.bluff and steal.kind == "steal" and steal.action == "raise"
    premium = evaluate_bluff(
        hole_cards=("As", "Ad"),
        street="preflop",
        pot=150,
        to_call=100,
        stack=2_000,
        opponents=2,
        settings=always,
    )
    assert not premium.bluff
    assert any("value" in reason for reason in premium.reasons)
    trash = evaluate_bluff(
        hole_cards=("7c", "2d"),
        street="preflop",
        pot=150,
        to_call=100,
        stack=2_000,
        opponents=2,
        settings=always,
    )
    assert not trash.bluff

    # Discipline gates: barrel caps and committed stacks stop the spew.
    capped = evaluate_bluff(
        hole_cards=("Qd", "4c"),
        board_cards=("7h", "2h", "9h", "Jc", "3s"),
        street="river",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=2,
        settings=always,
    )
    assert not capped.bluff
    assert any("barrel cap" in reason for reason in capped.reasons)
    shallow = evaluate_bluff(
        hole_cards=("Ah", "Kh"),
        board_cards=("7h", "2h", "9c"),
        street="flop",
        pot=1_000,
        stack=500,
        opponents=1,
        hero_aggressions=1,
        settings=always,
    )
    assert not shallow.bluff
    assert any("shallow" in reason for reason in shallow.reasons)

    # The mixed strategy is deterministic and honors a zero cap.
    again = evaluate_bluff(
        hole_cards=("Ah", "Kh"),
        board_cards=("7h", "2h", "9c"),
        street="flop",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=1,
        in_position=True,
        settings=always,
    )
    assert again.to_dict() == combo.to_dict()
    never = dataclasses.replace(
        DEFAULT_BLUFF_SETTINGS,
        continuation_frequency=0.0,
        semi_bluff_frequency=0.0,
    )
    withheld = evaluate_bluff(
        hole_cards=("Ah", "Kh"),
        board_cards=("7h", "2h", "9c"),
        street="flop",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=1,
        in_position=True,
        settings=never,
    )
    assert not withheld.bluff and withheld.bluff_score > 0.0
    assert any("mixed strategy" in reason for reason in withheld.reasons)

    # Bluff density scales every frequency cap, and the lead gradient tilts
    # it: with a positive gain the leader bluffs more than the trailer.
    tilted = dataclasses.replace(DEFAULT_BLUFF_SETTINGS, lead_density_gain=0.5)
    leading = evaluate_bluff(
        hole_cards=("Ah", "Kh"),
        board_cards=("7h", "2h", "9c"),
        street="flop",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=1,
        lead_position=100.0,
        settings=tilted,
    )
    trailing = evaluate_bluff(
        hole_cards=("Ah", "Kh"),
        board_cards=("7h", "2h", "9c"),
        street="flop",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=1,
        lead_position=-100.0,
        settings=tilted,
    )
    assert leading.frequency_cap == 1.0  # 0.80 cap x density 1.5, clamped
    assert trailing.frequency_cap == 0.4  # 0.80 cap x density 0.5
    silent = dataclasses.replace(DEFAULT_BLUFF_SETTINGS, bluff_density=0.0)
    muted = evaluate_bluff(
        hole_cards=("Ah", "Kh"),
        board_cards=("7h", "2h", "9c"),
        street="flop",
        pot=100,
        stack=1_000,
        opponents=1,
        hero_aggressions=1,
        settings=silent,
    )
    assert not muted.bluff and muted.frequency_cap == 0.0

    # Malformed situations raise instead of guessing.
    for bad in (
        {"street": "flop", "board_cards": ()},
        {"street": "preflop", "opponents": 9},
        {"street": "preflop", "pot": 0},
        {"street": "preflop", "lead_position": 150.0},
    ):
        try:
            evaluate_bluff(
                hole_cards=("Ah", "Kh"),
                pot=bad.get("pot", 100),
                stack=1_000,
                opponents=bad.get("opponents", 1),
                street=bad["street"],
                board_cards=bad.get("board_cards", ()),
                lead_position=bad.get("lead_position"),
            )
        except ValueError:
            pass
        else:  # pragma: no cover - self-test guard
            raise AssertionError(f"expected ValueError for {bad}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone poker bluff advisor.",
        epilog=(
            "Example: python bluff.py --hole Ah Kh --board 7h 2h 9c "
            "--street flop --pot 100 --stack 1000 --opponents 1 "
            "--hero-aggressions 1"
        ),
    )
    parser.add_argument("--hole", nargs=2, help="two hole cards, e.g. Ah Kh")
    parser.add_argument(
        "--board", nargs="*", default=[], help="board cards for the street"
    )
    parser.add_argument("--street", choices=tuple(_BOARD_SIZES), help="street")
    parser.add_argument("--pot", type=int, help="pot being attacked, in chips")
    parser.add_argument(
        "--to-call", type=int, default=0, help="unpaid bet in front of us"
    )
    parser.add_argument("--stack", type=int, help="effective stack behind")
    parser.add_argument(
        "--opponents", type=int, help=f"active opponents, {MIN_OPPONENTS}-{MAX_OPPONENTS}"
    )
    parser.add_argument(
        "--hero-aggressions",
        type=int,
        default=0,
        help="hero bets/raises so far this hand",
    )
    parser.add_argument(
        "--opponent-aggressions",
        type=int,
        default=0,
        help="opponent bets/raises so far this hand",
    )
    position = parser.add_mutually_exclusive_group()
    position.add_argument(
        "--in-position", action="store_true", help="hero acts last postflop"
    )
    position.add_argument(
        "--out-of-position", action="store_true", help="hero acts first postflop"
    )
    parser.add_argument(
        "--lead",
        type=float,
        help="lead-position reading from lead_position.py, -100 to +100",
    )
    parser.add_argument(
        "--mix-key", help="stable key for the mixed-strategy roll, e.g. a table id"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--self-test", action="store_true", help="run built-in checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        print("self-test: ok")
        return 0

    required = ("hole", "street", "pot", "stack", "opponents")
    missing = [f"--{name}" for name in required if getattr(args, name) is None]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))

    in_position: bool | None = None
    if args.in_position:
        in_position = True
    elif args.out_of_position:
        in_position = False
    try:
        advice = evaluate_bluff(
            hole_cards=tuple(args.hole),
            board_cards=tuple(args.board),
            street=args.street,
            pot=args.pot,
            to_call=args.to_call,
            stack=args.stack,
            opponents=args.opponents,
            hero_aggressions=args.hero_aggressions,
            opponent_aggressions=args.opponent_aggressions,
            in_position=in_position,
            lead_position=args.lead,
            mix_key=args.mix_key,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(advice.to_dict(), indent=2, sort_keys=True))
    else:
        print(render(advice))
    return 0


__all__ = [
    "BLUFF_KINDS",
    "BluffAdvice",
    "BluffSettings",
    "DEFAULT_BLUFF_SETTINGS",
    "evaluate_bluff",
    "render",
]


if __name__ == "__main__":
    raise SystemExit(main())
