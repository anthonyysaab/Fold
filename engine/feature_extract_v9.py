"""Schema-4 assembler: one Arena snapshot to the v9 vector.

``extract_features_v9`` WRAPS ``extract_features_v8`` rather than
restating it: every feature the two schemas share is read out of the v8
assembler by name, so the shared values are bit-identical by
construction and cannot drift. Only schema 4's seven-name delta is
computed here (the plan wrote this as "v9 paths in feature_extract_v8";
a sibling module is the same content with the v8 assembler left
byte-identical, per the additive-only discipline):

- **Legality quartet** — straight from
  ``branch_contract_v9.legal_branches(available, to_call)``; this module
  adds no masking logic of its own.
- **Lane costs through the composed sizing** —
  ``cost_active_eff`` is the call price at a priced spot and g's
  composed bet wager at a free one; ``cost_aggressive_eff`` is the
  composed escalation to-amount, clamped into the legal raise range
  (the v8 branch-block's feature-time approximation, kept: no big-blind
  minimum increment, no integer rounding). At a free spot the aggressive
  lane is masked by the contract, so its cost is 0.0 and
  ``legal_aggressive`` carries the mask.
- **``branch_aggressive_executable``** — whether the clamped composed
  to-amount is a genuine escalation (strictly above the call-to
  amount) within an existing legal range. Mirrors the v8 executability
  convention (legal range only; the serve-side risk cap and its
  near-nut release are equity-dependent and stay serve-side).

The boldness read for the composed sizing is g's own depth-invariant
read at feature time, fed by the UNCONDITIONED multiway Monte-Carlo
equity (``estimate_equity`` at ``_EQUITY_TRIALS`` with this call's
``seed``): deterministic, player-count-aware, and independent of the
engine's facing-aggression range conditioning, which belongs to the
gates. Corpus rows record the read they used (``10·T``), so trainer
parity never re-derives it.

Default parameters are the shipped defaults — every rules dial OFF, so
the costs describe bare g. A harvester that enables dials must pass the
same ``sizing``/``rules`` blocks it records in its corpus header.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from engine import schema3, schema4
from engine.aggression_sizing import (
    DEFAULT_SIZING_PARAMETERS,
    SizingParameters,
    table_boldness,
)
from engine.branch_contract_v9 import BRANCH_LABELS_V9, legal_branches
from engine.belief_provider import BeliefProvider
from engine.feature_extract_v8 import (
    _EQUITY_TRIALS,
    FeatureExtractError,
    _clip01,
    extract_features_v8,
)
from engine.game_state import (
    _cards,
    _hero_and_seats,
    _integer,
    _mapping,
    _sequence,
    active_opponent_count,
    effective_stack_chips,
)
from engine.hand_strength import estimate_equity
from engine.rules.composition import (
    DEFAULT_RULE_LAYER,
    RuleLayerParams,
    compose_active_wager,
    compose_aggressive_target,
)
from engine.rules.ruin_damper import table_exposure

__all__ = ["extract_features_v9"]


def _lane_range(
    allowed: Mapping[str, object], key: str
) -> tuple[int, int] | None:
    """One lane's stated legal range, validated; None when absent."""

    raw = allowed.get(key)
    if raw is None:
        return None
    block = _mapping(raw, key)
    low = _integer(block.get("min"), f"{key}.min")
    high = _integer(block.get("max"), f"{key}.max")
    if high < low:
        return None
    return low, high


def _covered_allin_to_amounts(
    hero: Mapping[str, object], seats: Sequence[Mapping[str, object]]
) -> tuple[int, ...]:
    """All-in to-amounts of active opponents hero covers (for C3A)."""

    hero_stack = _integer(hero.get("stackChips"), "hero stackChips")
    hero_total = hero_stack + _integer(
        hero.get("currentBetChips"), "hero currentBetChips"
    )
    amounts: list[int] = []
    for seat in seats:
        if seat is hero or str(seat.get("status") or "").casefold() != "active":
            continue
        stack = seat.get("stackChips")
        committed = seat.get("currentBetChips") or 0
        if isinstance(stack, bool) or not isinstance(stack, (int, float)):
            continue
        if isinstance(committed, bool) or not isinstance(committed, (int, float)):
            committed = 0
        allin_to = int(stack) + int(committed)
        if allin_to > 0 and hero_total >= allin_to:
            amounts.append(allin_to)
    return tuple(amounts)


def extract_features_v9(
    table: Mapping[str, object],
    *,
    belief_provider: BeliefProvider | None = None,
    potential_trials: int = 400,
    seed: int = 7,
    sizing: SizingParameters = DEFAULT_SIZING_PARAMETERS,
    rules: RuleLayerParams = DEFAULT_RULE_LAYER,
) -> tuple[float, ...]:
    """Assemble the full schema-4 vector from one live table snapshot.

    Returns ``schema4.FEATURE_NAMES_V9``-ordered floats, validated by
    ``schema4.require_vector_v9`` before returning. Shared names are the
    v8 assembler's own values, bit for bit.
    """

    shared = dict(
        zip(
            schema3.FEATURE_NAMES_V8,
            extract_features_v8(
                table,
                belief_provider=belief_provider,
                potential_trials=potential_trials,
                seed=seed,
            ),
        )
    )

    hero, seats = _hero_and_seats(table)
    allowed = _mapping(table.get("allowedActions"), "allowedActions")
    available = {
        str(value)
        for value in _sequence(allowed.get("availableActions"), "availableActions")
    }
    street = str(table.get("street") or "").casefold()
    pot = _integer(table.get("potChips"), "potChips")
    contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
    to_call = _integer(allowed.get("callChips", 0), "allowedActions.callChips")
    hole = _cards(hero.get("holeCards"), "hero holeCards", expected=2)
    board = _cards(table.get("boardCards"), "boardCards")
    eff = max(1, effective_stack_chips(table))
    hero_stack = _integer(hero.get("stackChips"), "hero stackChips", minimum=1)

    # legal_branches returns slot INDICES; resolve to labels before the
    # membership test (a string-in-indices test is silently always false).
    legal_labels = {
        BRANCH_LABELS_V9[index] for index in legal_branches(available, to_call)
    }
    new_values: dict[str, float] = {
        f"legal_{branch}": 1.0 if branch in legal_labels else 0.0
        for branch in BRANCH_LABELS_V9
    }

    # g's read at feature time: unconditioned multiway equity, this
    # call's seed, through the depth-invariant table read.
    opponents = active_opponent_count(table)
    equity_read = (
        1.0
        if opponents < 1
        else estimate_equity(
            (hole[0], hole[1]),
            tuple(board),
            opponents,
            trials=_EQUITY_TRIALS,
            seed=seed,
        )
    )
    boldness = table_boldness(table, allowed, equity_read, sizing)
    exposure = table_exposure(table)
    covered = _covered_allin_to_amounts(hero, seats)

    if to_call > 0:
        new_values["cost_active_eff"] = _clip01(to_call / eff)
        composed = compose_aggressive_target(
            boldness=boldness,
            pot=pot,
            to_call=to_call,
            effective_stack=eff,
            contribution=contribution,
            street=street,
            bankroll=hero_stack,
            exposure=exposure,
            covered_allin_to_amounts=covered,
            sizing=sizing,
            geometric=rules.geometric,
            snap=rules.snap,
            damper=rules.damper,
        )
        lane_range = _lane_range(allowed, "raiseRange")
        if lane_range is None:
            # No stated range: the escalation is not executable; the cost
            # keeps the nominal composed price so the axis still ranks.
            new_values["cost_aggressive_eff"] = _clip01(composed.target / eff)
            new_values["branch_aggressive_executable"] = 0.0
        else:
            low, high = lane_range
            to_amount = min(high, max(low, composed.to_amount))
            call_to = allowed.get("callToAmount")
            call_to_amount = (
                _integer(call_to, "callToAmount")
                if call_to is not None
                else contribution + to_call
            )
            new_values["cost_aggressive_eff"] = _clip01(
                (to_amount - contribution) / eff
            )
            new_values["branch_aggressive_executable"] = (
                1.0 if to_amount > call_to_amount else 0.0
            )
    else:
        composed = compose_active_wager(
            boldness=boldness,
            pot=pot,
            effective_stack=eff,
            contribution=contribution,
            street=street,
            bankroll=hero_stack,
            exposure=exposure,
            covered_allin_to_amounts=covered,
            sizing=sizing,
            geometric=rules.geometric,
            snap=rules.snap,
            damper=rules.damper,
        )
        lane_range = _lane_range(allowed, "betRange")
        if lane_range is None:
            new_values["cost_active_eff"] = _clip01(composed.target / eff)
        else:
            low, high = lane_range
            to_amount = min(high, max(low, composed.to_amount))
            new_values["cost_active_eff"] = _clip01((to_amount - contribution) / eff)
        # The aggressive lane does not exist at a free spot under the v9
        # contract; legal_aggressive carries the mask, the cost is zero.
        new_values["cost_aggressive_eff"] = 0.0
        new_values["branch_aggressive_executable"] = 0.0

    features = tuple(
        new_values[name] if name in new_values else shared[name]
        for name in schema4.FEATURE_NAMES_V9
    )
    if not all(math.isfinite(value) for value in features):
        raise FeatureExtractError("snapshot produced non-finite v9 features")
    schema4.require_vector_v9(features)
    return features
