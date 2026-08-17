"""Schema-3 feature assembler: one Arena table snapshot to the v8 vector.

``extract_features_v8`` turns a live table snapshot (the same untrusted
mapping every other consumer reads: ``potChips``, ``street``, ``seats``,
``holeCards``, ``boardCards``, ``allowedActions``, ``recentEvents``) into
the full ``schema3.FEATURE_NAMES_V8`` vector. It is an assembler, not a
second implementation: every quantity that already has a canonical producer
is imported and reused, so the definitions cannot drift —

- the kept schema-2 scalars are read out of
  ``game_state.features_from_table`` (the literal schema-2 computation:
  same log base, same normalisations, same legality rules);
- ``lead_position_unit`` mirrors ``decision_engine._lead_position`` exactly
  (``measure_lead_position`` over ``max(1, stack + street bet)`` per active
  player, the engine's ``(lead + 100) / 200`` unit, and its 0.5 fallback
  when the reading raises);
- strength comes from ``strength_metric``, potential from
  ``hand_potential``, board tiers from ``hand_strength.board_improvement``,
  reveal expense from ``game_state.card_reveal_expense``, the history block
  from ``action_history.encode_action_history`` fed by this module's own
  ``_hand_events`` — the telemetry writer's per-record defensive rules,
  uncapped (see its docstring for the two deliberate deviations) — and
  belief buckets from a ``belief_provider.BeliefProvider`` (neutral
  uniform by default), gated through ``require_buckets``.

Values are emitted raw; z-scoring the context block is the trainer's
transform (as it was for v7), never the extractor's.

Branch costs and executability are the **feature-time approximation of the
label-time emission rule** (V8_DESIGN §4 / E6): the branch's total wager
``min(call + pot_fraction * (pot + call), stack_fraction * effective
stack)`` is converted to a to-amount and clamped into the legal bet/raise
range. The pot arm is the engine's realized wager for a pot-fraction size
(``decision_engine._sized_action``: base ``callToAmount``, increment
``round(fraction * (pot + call))``), which preserves the E6 property that
deep spots are byte-identical to today's half-pot/pot sizing. Documented
deviations from the engine: no big-blind minimum increment, no integer
rounding. The label-time rule ("a branch is emitted only where the engine
can execute it and where it is distinct") is approximated as: a legal
bet/raise range exists, and — for the large branch — the two clamped
to-amounts differ.

Failure posture: a decision cannot exist without hole cards, a big blind, a
known street, or legal actions, so those fail loud (``ArenaSnapshotError``
from the shared validators, or a sibling module's own error). Genuinely
optional quantities degrade to 0.0 instead: empty ``recentEvents`` reads an
all-zero history/summary block, an opponent seat with a missing or
malformed stack is skipped from the stack summary, a street with no
aggressor reads ``last_aggressor_position_unit`` 0.0, and a table with no
scorable opponent stack reads all three stack summaries 0.0.

Determinism: all randomness is ``random.Random(seed)`` constructed inside
the call (one instance drives both range-equity estimates in a fixed
order; ``hand_potential`` builds its own from the same seed). Identical
snapshot and arguments give identical vectors.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from functools import lru_cache

from lead_position import measure_lead_position

from devfun_poker_playground import schema3
from devfun_poker_playground._vendor.treys import Card, Evaluator
from devfun_poker_playground.action_history import (
    _actor_unit,
    _seat_ordinals,
    encode_action_history,
)
from devfun_poker_playground.belief_provider import (
    BeliefProvider,
    NeutralBeliefProvider,
    require_buckets,
)
from devfun_poker_playground.game_state import (
    _active_seats,
    _cards,
    _hero_and_seats,
    _integer,
    _mapping,
    _position,
    _sequence,
    card_reveal_expense,
    effective_stack_chips,
    features_from_table,
)
from devfun_poker_playground.hand_potential import hand_potential
from devfun_poker_playground.hand_strength import board_improvement
from devfun_poker_playground.policy_features import FEATURE_NAMES
from devfun_poker_playground.preflop_percentiles import PREFLOP_PERCENTILES
from devfun_poker_playground.strength_metric import (
    canonical_class,
    strength_percentile,
)
from devfun_poker_playground.training_telemetry import action_family

__all__ = ["FeatureExtractError", "extract_features_v8"]

# Range-equity sample size per range. Ablatable: worst-case (p = 0.5)
# binomial standard error is sqrt(0.25 / 200) ~= 0.035, small against the
# strong-wide/dead-narrow swings the slope feature exists to expose
# (V8_DESIGN §3: 0.75 vs a wide range, dead vs the shoving range).
_EQUITY_TRIALS = 200

# Derived from the feature names, not tuned: "top X%" of the preflop
# ordering is the set of canonical classes with percentile >= 1 - X/100.
_TOP20_FLOOR = 0.80
_TOP5_FLOOR = 0.95

# E6 branch targets (V8_DESIGN §4, owner approved 2026-08-15): total wager
# = min(call + pot_fraction * (pot + call), stack_fraction * effective
# stack) — the pot arm realized exactly as the engine sizes a pot
# fraction. All four numbers sit in the §6 ablation battery rather than
# being trusted hand-authored constants.
_BRANCH_SMALL = (0.50, 0.20)
_BRANCH_LARGE = (1.00, 0.45)

# Engine fallback when the lead reading raises: the exact None case of
# decision_engine._learning_features — the midpoint of (lead + 100) / 200.
_NEUTRAL_LEAD_UNIT = 0.5


class FeatureExtractError(ValueError):
    """Raised when an argument to the extractor violates its contract."""


_EVALUATOR: Evaluator | None = None


def _evaluator() -> Evaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = Evaluator()
    return _EVALUATOR


@lru_cache(maxsize=1)
def _treys_by_code() -> dict[str, int]:
    return {code: Card.new(code) for code in schema3.CARD_CODES}


@lru_cache(maxsize=1)
def _combo_percentiles() -> tuple[tuple[str, str, float], ...]:
    """All 1326 concrete holdings with their preflop class percentile."""

    codes = schema3.CARD_CODES
    return tuple(
        (codes[i], codes[j], PREFLOP_PERCENTILES[canonical_class((codes[i], codes[j]))])
        for i in range(len(codes))
        for j in range(i + 1, len(codes))
    )


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _equity_vs_range(
    hole: Sequence[str],
    board: Sequence[str],
    percentile_floor: float,
    rng: random.Random,
) -> float:
    """Hero showdown equity vs a uniformly sampled top-of-range holding.

    Opponent holdings are drawn uniformly from the concrete combos whose
    canonical class sits at or above ``percentile_floor`` in the preflop
    ordering, excluding dead cards; each trial completes the board from the
    remaining unseen cards and scores the showdown (win 1, tie 0.5).
    """

    by_code = _treys_by_code()
    dead = set(hole) | set(board)
    combos = [
        (first, second)
        for first, second, percentile in _combo_percentiles()
        if percentile >= percentile_floor and first not in dead and second not in dead
    ]
    if not combos:
        # Unreachable with at most 7 dead cards (every top range keeps
        # combos); kept so a degenerate input degrades instead of dividing
        # by zero.
        return 0.0
    unseen = [code for code in schema3.CARD_CODES if code not in dead]
    hero_ints = [by_code[card] for card in hole]
    board_ints = [by_code[card] for card in board]
    missing = 5 - len(board)
    evaluator = _evaluator()
    favorable = 0.0
    for _ in range(_EQUITY_TRIALS):
        first, second = combos[rng.randrange(len(combos))]
        opponent_ints = [by_code[first], by_code[second]]
        if missing:
            pool = [code for code in unseen if code != first and code != second]
            runout = board_ints + [
                by_code[code] for code in rng.sample(pool, missing)
            ]
        else:
            runout = board_ints
        hero_rank = evaluator.evaluate(runout, hero_ints)
        opponent_rank = evaluator.evaluate(runout, opponent_ints)
        if hero_rank < opponent_rank:  # treys: lower rank is stronger
            favorable += 1.0
        elif hero_rank == opponent_rank:
            favorable += 0.5
    return favorable / _EQUITY_TRIALS


def _clean_amount(value: object) -> int | None:
    """The value as an int wager, or None (bools and non-ints rejected)."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _wager_amount(summary: Mapping[str, object], action: str) -> int | None:
    """The price signal a raw event actually carries, per action.

    Measured on raw replays and the live journal (2026-08-16): aggressive
    events state their price in ``toAmount`` (``amount`` is null), calls
    state theirs in ``amount``. Each family prefers its own field and
    falls back to the other; fold and check carry no wager. Familyless
    bookkeeping records keep the telemetry writer's plain-``amount`` rule
    (they never occupy a history slot either way).
    """

    amount = _clean_amount(summary.get("amount"))
    to_amount = _clean_amount(summary.get("toAmount"))
    family = action_family(action)
    if family == "aggress":
        return to_amount if to_amount is not None else amount
    if action == "call":
        return amount if amount is not None else to_amount
    if family is not None:  # fold / check: no wager to record
        return None
    return amount


def _hand_events(table: Mapping[str, object]) -> list[dict[str, object]]:
    """Every action record of the hand in ``recentEvents``, uncapped.

    Same defensive per-record rules as ``training_telemetry.
    _recent_actions`` (skip malformed entries, casefold street and action,
    tolerate a missing seat, integer amounts only), with two deliberate
    deviations:

    - **No 60-record cap.** The journal writer bounds its stored copy so a
      hostile snapshot cannot balloon a record; the extractor reads one
      hand's worth of events, whose boundedness comes from the game
      itself, and a cap here would silently drop the oldest actions —
      falsifying the schema-3 rule that truncation is recorded, never
      silent. Journal consumers replaying stored (60-capped)
      ``recent_actions`` may therefore still under-report a hand's early
      actions; the live path does not.
    - **The wager field is normalised per action** (``_wager_amount``):
      the record's single ``amount`` — the history size channel's input —
      is the raise-to amount for bet/raise/all-in and the call wager for
      calls, both being the price the raw event actually carries. So
      ``size = log1p(chips / bb)`` where chips is the call wager or the
      raise-to amount. Journal-stored schema-2 ``recent_actions`` dropped
      ``toAmount``; offline consumers of those records read 0.0 sizes for
      aggressive actions, the live path and raw replays do not.
    """

    events = table.get("recentEvents") or []
    if not isinstance(events, (list, tuple)):
        return []
    records: list[dict[str, object]] = []
    for raw_event in events:
        if not isinstance(raw_event, Mapping):
            continue
        summary = raw_event.get("summary")
        if not isinstance(summary, Mapping):
            continue
        action = summary.get("action")
        if not isinstance(action, str) or not action:
            continue
        action = action.casefold()
        seat_number = summary.get("seatNumber")
        entry: dict[str, object] = {
            "street": str(raw_event.get("street") or "").casefold(),
            "seat_number": seat_number if isinstance(seat_number, int) else None,
            "action": action,
        }
        amount = _wager_amount(summary, action)
        if amount is not None:
            entry["amount"] = amount
        records.append(entry)
    return records


def _lead_position_unit(
    table: Mapping[str, object],
    hero: Mapping[str, object],
    seats: list[Mapping[str, object]],
) -> float:
    """``decision_engine._lead_position`` folded to its learning unit."""

    opponents = [seat for seat in _active_seats(seats) if seat is not hero]

    def _owned(seat: Mapping[str, object]) -> int:
        stack = _integer(seat.get("stackChips"), "stackChips")
        committed = _integer(seat.get("currentBetChips"), "currentBetChips")
        return max(1, stack + committed)

    try:
        reading = measure_lead_position(
            hero_stack=_owned(hero),
            opponent_stacks=tuple(_owned(seat) for seat in opponents),
            position=_position(table, seats, hero),
        )
    except ValueError:
        return _NEUTRAL_LEAD_UNIT
    return (reading.lead + 100.0) / 200.0


def _aggressive_range(
    allowed: Mapping[str, object], available: set[str]
) -> tuple[int, int] | None:
    """The legal bet/raise to-amount range, or None when none is stated.

    Bet is preferred over raise exactly as ``decision_engine._sized_action``
    selects its range by action name; a stated range with max below min is
    treated as absent (feature-time degrade — the telemetry validator is
    where malformed ranges fail loud).
    """

    for action, range_name in (("bet", "betRange"), ("raise", "raiseRange")):
        if action not in available:
            continue
        value = allowed.get(range_name)
        if value is None:
            continue
        amount_range = _mapping(value, range_name)
        low = _integer(amount_range.get("min"), f"{range_name}.min")
        high = _integer(amount_range.get("max"), f"{range_name}.max")
        if high >= low:
            return low, high
    return None


def _branch_block(
    *,
    pot: int,
    effective_stack: int,
    contribution: int,
    to_call: int,
    allowed: Mapping[str, object],
    available: set[str],
) -> dict[str, float]:
    """E6 branch costs and executability (feature-time approximation)."""

    eff = max(1, effective_stack)  # the engine's own denominator guard

    def _target(spec: tuple[float, float]) -> float:
        # The pot arm is the engine's realized wager for a pot-fraction
        # size (_sized_action: base = callToAmount, increment =
        # round(fraction * (pot + call))), so the branch's total wager is
        # call + fraction * (pot + call) — not fraction * pot. The stack
        # arm is the E6 stack bound. Feature-time approximation: no
        # big-blind minimum increment, no integer rounding.
        pot_fraction, stack_fraction = spec
        return min(
            to_call + pot_fraction * (pot + to_call),
            stack_fraction * eff,
        )

    small_target = _target(_BRANCH_SMALL)
    large_target = _target(_BRANCH_LARGE)
    legal_range = _aggressive_range(allowed, available)
    if legal_range is None:
        # No stated bet/raise range: the branches are not executable; their
        # costs stay the nominal branch price so the cost axis still ranks
        # the branches (the executability flags carry the legality).
        small_cost = _clip01(small_target / eff)
        large_cost = _clip01(large_target / eff)
        small_executable = large_executable = 0.0
    else:
        low, high = legal_range
        small_to = min(high, max(low, contribution + small_target))
        large_to = min(high, max(low, contribution + large_target))
        small_cost = _clip01((small_to - contribution) / eff)
        large_cost = _clip01((large_to - contribution) / eff)
        small_executable = 1.0
        large_executable = 1.0 if large_to != small_to else 0.0

    if "all-in" in available:
        allin_cost = 1.0
    else:
        commitments = [float(to_call)]
        if legal_range is not None:
            commitments.append(float(legal_range[1] - contribution))
        allin_to = allowed.get("allInToAmount")
        if allin_to is not None:
            commitments.append(
                float(_integer(allin_to, "allInToAmount") - contribution)
            )
        allin_cost = _clip01(max(commitments) / eff)

    return {
        "cost_call_eff": _clip01(to_call / eff),
        "cost_aggress_small_eff": small_cost,
        "cost_aggress_large_eff": large_cost,
        "cost_allin_eff": allin_cost,
        "branch_small_executable": small_executable,
        "branch_large_executable": large_executable,
    }


def _summary_block(
    records: Sequence[Mapping[str, object]],
    street: str,
    hero_seat: int,
    seat_ordinals: Mapping[int, int],
) -> dict[str, float]:
    """Opponent and hero evidence from the parsed action records.

    Scope (V8_DESIGN §3, the schema-2 "kept" semantics):
    ``opp_aggressive_actions``, ``opp_calls`` and ``hero_aggression_count``
    are **whole-hand** counts — schema 2's ``hero_aggression_count`` was
    ``decision_engine._aggressive_events`` with ``street=None``, and the
    opponent counts mirror that scope. ``callers_of_current_bet`` and
    ``last_aggressor_position_unit`` are **current-street** by definition
    (documented): they describe the bet hero is facing right now.

    ``callers_of_current_bet`` counts opponent call records after the last
    aggress-family record on the street — hero's own calls never count
    (hero calling adds no opponent to beat); with no aggressor on the
    street every opponent call is counted (preflop limps call the posted
    blind, and postflop a call cannot exist without a bet). A street with
    no aggressor reads ``last_aggressor_position_unit`` 0.0 (documented
    degrade — the actor unit's own hero value, which the occurred-style
    counts disambiguate).
    """

    def _is_hero(record: Mapping[str, object]) -> bool:
        seat_number = record.get("seat_number")
        return isinstance(seat_number, int) and seat_number == hero_seat

    opp_aggressive = 0
    opp_calls = 0
    hero_aggressive = 0
    for record in records:  # whole hand, matching _aggressive_events
        action = record.get("action")
        family = action_family(action) if isinstance(action, str) else None
        if family == "aggress":
            if _is_hero(record):
                hero_aggressive += 1
            else:
                opp_aggressive += 1
        elif action == "call" and not _is_hero(record):
            opp_calls += 1

    street_records = [
        record for record in records if record.get("street") == street
    ]
    last_aggress_index = -1
    last_aggressor_seat: object = None
    for index, record in enumerate(street_records):
        action = record.get("action")
        family = action_family(action) if isinstance(action, str) else None
        if family == "aggress":
            last_aggress_index = index
            last_aggressor_seat = record.get("seat_number")
    callers = sum(
        1
        for index, record in enumerate(street_records)
        if index > last_aggress_index
        and record.get("action") == "call"
        and not _is_hero(record)
    )
    if last_aggress_index < 0:
        last_aggressor_unit = 0.0
    else:
        last_aggressor_unit = _actor_unit(
            last_aggressor_seat,
            hero_seat,
            max(1, len(seat_ordinals)),
            seat_ordinals,
        )
    return {
        "opp_aggressive_actions": float(opp_aggressive),
        "opp_calls": float(opp_calls),
        "callers_of_current_bet": float(callers),
        "last_aggressor_position_unit": last_aggressor_unit,
        "hero_aggression_count": float(hero_aggressive),
    }


def _seat_committed(seat: Mapping[str, object]) -> int:
    """Defensive whole-hand commitment for an opponent seat (0 on absence)."""

    value = seat.get("totalCommittedChips", seat.get("currentBetChips"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _table_block(
    hero: Mapping[str, object],
    seats: list[Mapping[str, object]],
    big_blind: int,
) -> dict[str, float]:
    """Opponent stack summary plus hero's whole-hand commitment."""

    allin_count = 0
    stack_logs: list[float] = []
    for seat in _active_seats(seats):
        if seat is hero:
            continue
        stack = seat.get("stackChips")
        if isinstance(stack, bool) or not isinstance(stack, int) or stack < 0:
            continue  # documented degrade: absent/malformed stacks are skipped
        committed = _seat_committed(seat)
        if stack == 0 and committed == 0:
            continue  # busted/empty seat: not in the hand at all
        if stack == 0:
            allin_count += 1
        stack_logs.append(math.log1p(stack / big_blind))
    if stack_logs:
        minimum = min(stack_logs)
        maximum = max(stack_logs)
        # statistics.median is exactly "mean of the middle two" for an even
        # count, and log1p is monotone so min/max commute with the log.
        median = statistics.median(stack_logs)
    else:
        minimum = median = maximum = 0.0  # documented degrade: no opponents

    contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
    total_committed = _integer(
        hero.get("totalCommittedChips", contribution), "hero totalCommittedChips"
    )
    stack = _integer(hero.get("stackChips"), "hero stackChips")
    owned = total_committed + stack
    return {
        "log_opp_stack_min_bb": minimum,
        "log_opp_stack_median_bb": median,
        "log_opp_stack_max_bb": maximum,
        "opp_allin_count": float(allin_count),
        "log_total_committed_bb": math.log1p(total_committed / big_blind),
        "committed_stack_fraction": (
            total_committed / owned if owned > 0 else 0.0
        ),
    }


def extract_features_v8(
    table: Mapping[str, object],
    *,
    equity: float | None = None,
    belief_provider: BeliefProvider | None = None,
    potential_trials: int = 400,
    seed: int = 7,
) -> tuple[float, ...]:
    """Assemble the full schema-3 vector from one live table snapshot.

    ``equity`` is accepted for call-site compatibility with the decision
    engine, which has already computed its multiway Monte-Carlo equity when
    features are built. No schema-3 feature consumes it — the canonical
    ``strength_percentile`` replaced schema-2's equity input, and the two
    scales must never be conflated (V8_DESIGN §2) — so it is validated
    (finite, in [0, 1]) and otherwise ignored.

    ``potential_trials`` defaults to 400: ablatable — binomial standard
    error <= sqrt(0.25 / 400) ~= 0.025, chosen under ``hand_potential``'s
    offline default 1000 for the live-path latency budget. ``seed``
    defaults to 7, matching ``hand_potential``'s fixed default; callers
    doing statistics must vary it themselves.

    Returns ``schema3.FEATURE_NAMES_V8``-ordered floats, validated by
    ``schema3.require_vector`` before returning.
    """

    if equity is not None:
        value = float(equity)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise FeatureExtractError(
                f"equity must be finite and within [0, 1], got {equity!r}"
            )

    # features_from_table is the fail-loud snapshot validation: hole cards,
    # big blind, street, and legal actions all raise ArenaSnapshotError
    # here, and its output IS the kept schema-2 block.
    base_by_name = dict(zip(FEATURE_NAMES, features_from_table(table)))

    hero, seats = _hero_and_seats(table)
    allowed = _mapping(table.get("allowedActions"), "allowedActions")
    available = {
        str(value)
        for value in _sequence(allowed.get("availableActions"), "availableActions")
    }
    street = str(table.get("street") or "").casefold()
    big_blind = _integer(table.get("bigBlindChips"), "bigBlindChips", minimum=1)
    pot = _integer(table.get("potChips"), "potChips")
    contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
    to_call = _integer(allowed.get("callChips", 0), "allowedActions.callChips")
    hero_seat = _integer(table.get("selfSeatNumber"), "selfSeatNumber", minimum=1)
    hole = _cards(hero.get("holeCards"), "hero holeCards", expected=2)
    board = _cards(table.get("boardCards"), "boardCards")
    effective_stack = effective_stack_chips(table)
    records = _hand_events(table)
    # The snapshot's seat-number set — the same seats list the table block
    # reads — resolves actor units ordinally so sparse seat numbers cannot
    # alias (F1); malformed entries are dropped defensively.
    seat_numbers = tuple(
        number
        for number in (seat.get("seatNumber") for seat in seats)
        if isinstance(number, int) and not isinstance(number, bool)
    )
    seat_ordinals = _seat_ordinals(seat_numbers)

    values = {name: 0.0 for name in schema3.FEATURE_NAMES_V8}

    # Card block: hole plane plus one plane per board street (deal order).
    for card in hole:
        values[f"hole_{card}"] = 1.0
    for card in board[:3]:
        values[f"flop_{card}"] = 1.0
    for card in board[3:4]:
        values[f"turn_{card}"] = 1.0
    for card in board[4:5]:
        values[f"river_{card}"] = 1.0

    # Kept schema-2 block, read straight out of the schema-2 computation.
    for name in schema3.KEPT_FEATURE_NAMES:
        if name == "lead_position_unit":
            values[name] = _lead_position_unit(table, hero, seats)
        else:
            values[name] = base_by_name[name]

    # Strength block.
    rng = random.Random(seed)
    equity_top20 = _equity_vs_range(hole, board, _TOP20_FLOOR, rng)
    equity_top5 = _equity_vs_range(hole, board, _TOP5_FLOOR, rng)
    ppot, npot = hand_potential(hole, board, trials=potential_trials, seed=seed)
    values["strength_percentile"] = strength_percentile(hole, board)
    values["equity_vs_top20"] = equity_top20
    values["equity_vs_top5"] = equity_top5
    values["equity_range_slope"] = equity_top20 - equity_top5
    values["hand_ppot"] = ppot
    values["hand_npot"] = npot
    values["card_reveal_expense"] = card_reveal_expense(table, to_call)
    values.update(
        _branch_block(
            pot=pot,
            effective_stack=effective_stack,
            contribution=contribution,
            to_call=to_call,
            allowed=allowed,
            available=available,
        )
    )

    # Summary block: whole-hand counts plus current-street bet evidence.
    values.update(_summary_block(records, street, hero_seat, seat_ordinals))

    # Table block: opponent stack summary and hero commitment.
    values.update(_table_block(hero, seats, big_blind))

    # Board tiers: exactly one of the three, from the canonical classifier.
    tier = board_improvement((hole[0], hole[1]), tuple(board))
    values[f"board_tier_{tier}"] = 1.0

    # History block: the uncapped hand parse through the schema-3 encoder,
    # with the snapshot's seat set resolving actor units ordinally.
    history = encode_action_history(
        records,
        hero_seat_number=hero_seat,
        player_count=len(seats),
        big_blind_chips=big_blind,
        seat_numbers=seat_numbers,
    )
    for name, value in zip(schema3.HISTORY_FEATURE_NAMES, history):
        values[name] = value

    # Belief block, always through the contract gate.
    provider = belief_provider if belief_provider is not None else (
        NeutralBeliefProvider()
    )
    buckets = require_buckets(provider.continuing_range_buckets(table, records))
    for name, value in zip(schema3.BELIEF_FEATURE_NAMES, buckets):
        values[name] = float(value)

    features = tuple(values[name] for name in schema3.FEATURE_NAMES_V8)
    if not all(math.isfinite(value) for value in features):
        raise FeatureExtractError("snapshot produced non-finite v8 features")
    schema3.require_vector(features)
    return features
