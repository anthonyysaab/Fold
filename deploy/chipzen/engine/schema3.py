"""Schema 3 — the v8 feature contract. Names, order, blocks, nothing else.

This module is the single coordination point for the v8 feature vector
(V8_DESIGN.md §3, extended 2026-08-16 by the owner-approved research pass:
Deep CFR-style action-history slots, AlphaHoldem-style street-split board
planes, opponent stack summary, commitment, NPOT, branch executability, and
the Tier-2 belief-bucket inputs). Every producer and consumer of a v8
feature vector imports the layout from here; none may restate it.

Schema 2 (142 inputs, ``learning_contract.py``) remains untouched — v7
artifacts keep loading against it. Schema 3 is additive-only in the repo:
a new contract for a new model family.

Layout, in order:

- **Card block (208, binary one-hots, not z-scored)** — ``hole`` 52,
  ``flop`` 52, ``turn`` 52, ``river`` 52. Street-split so the trunk can see
  *which card arrived when*; a merged board cannot express "the flush came
  on the river after the turn went bet/call".
- **Context block (205, z-scored)** — see the grouped tuples below.

History slots follow Deep CFR's encoding (per betting position: an
occurred flag plus the wager) extended for multiway play with an action
family one-hot and the actor's position unit. Six slots per street, most
recent last (slot 5 is the latest action); a street that saw more than six
actions keeps the last six and raises its ``overflow`` flag — truncation is
recorded, never silent. Sizes are ``log1p(chips / big_blind)`` — where
chips is the wager the raw event actually carries: the raise-to amount for
aggressive actions (``toAmount``), the call wager for calls (``amount``;
see ``feature_extract_v8._hand_events``) — because the record does not
carry the pot at the moment of a past action; this is a documented
deviation from Deep CFR's pot-fraction sizes.
"""

from __future__ import annotations

from typing import Iterator

SCHEMA_VERSION = 3

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
CARD_CODES: tuple[str, ...] = tuple(
    f"{rank}{suit}" for rank in _RANKS for suit in _SUITS
)

# --- Card block: hole + one plane per board street -------------------------

CARD_PLANES: tuple[str, ...] = ("hole", "flop", "turn", "river")
CARD_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{plane}_{code}" for plane in CARD_PLANES for code in CARD_CODES
)

# --- Context block ---------------------------------------------------------

# Kept from schema 2 unchanged (22).
KEPT_FEATURE_NAMES: tuple[str, ...] = (
    "street_preflop",
    "street_flop",
    "street_turn",
    "street_river",
    "player_count",
    "active_player_count",
    "position",
    "lead_position_unit",
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

# Strength, range sensitivity, branch costs, potential (V8_DESIGN §3; E4/E5
# owner decisions 2026-08-15). `hand_ppot` is the renamed
# `board_improvement_potential`; `hand_npot` is its new mirror (probability
# the current best hand is outdrawn), the missing half of the Loki/Poki EHS
# decomposition.
STRENGTH_FEATURE_NAMES: tuple[str, ...] = (
    "strength_percentile",
    "equity_vs_top20",
    "equity_vs_top5",
    "equity_range_slope",
    "hand_ppot",
    "hand_npot",
    "cost_call_eff",
    "cost_aggress_small_eff",
    "cost_aggress_large_eff",
    "cost_allin_eff",
    "card_reveal_expense",
    "branch_small_executable",
    "branch_large_executable",
)

# Within-hand opponent evidence summaries (kept alongside the slot history:
# cheap, and the ablation battery can prune the redundancy either way).
SUMMARY_FEATURE_NAMES: tuple[str, ...] = (
    "opp_aggressive_actions",
    "opp_calls",
    "callers_of_current_bet",
    "last_aggressor_position_unit",
    "hero_aggression_count",
)

# Multiway table state the big-network survey found missing: who can cover
# whom, and how committed hero already is across the whole hand.
TABLE_FEATURE_NAMES: tuple[str, ...] = (
    "log_opp_stack_min_bb",
    "log_opp_stack_median_bb",
    "log_opp_stack_max_bb",
    "opp_allin_count",
    "log_total_committed_bb",
    "committed_stack_fraction",
)

# Heuristic board texture, ablatable as a group (V8_DESIGN §6).
BOARD_TIER_FEATURE_NAMES: tuple[str, ...] = (
    "board_tier_fresh",
    "board_tier_thin",
    "board_tier_kicker",
)

# Action-history slots.
HISTORY_STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")
HISTORY_SLOTS = 6
HISTORY_CHANNELS: tuple[str, ...] = (
    "occurred",
    "fold",
    "check_call",
    "aggress",
    "size",
    "actor",
)


def _history_names() -> Iterator[str]:
    for street in HISTORY_STREETS:
        for slot in range(HISTORY_SLOTS):
            for channel in HISTORY_CHANNELS:
                yield f"hist_{street}{slot}_{channel}"
        yield f"hist_{street}_overflow"


HISTORY_FEATURE_NAMES: tuple[str, ...] = tuple(_history_names())

# Tier 2: price-conditioned continuing-range distribution from the fitted
# P3 opponent model. Until a fitted provider exists, these are the neutral
# uniform prior (1/8 each) — the plumbing ships now so P3 plugs in without
# a schema change.
BELIEF_BUCKETS = 8
BELIEF_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"belief_bucket_{index}" for index in range(BELIEF_BUCKETS)
)

# --- Dormant: format-gated, reserved for reactivation ----------------------
#
# Nine schema-2 features are absent from the active v8 contract because they
# are dead in *this* arena format -- NOT because they are uninformative.
# Parking them here rather than deleting them keeps that distinction, and
# keeps reactivation a one-flag change instead of an archaeology exercise.
# The engine still computes all nine (`decision_engine`), and schema 2 still
# declares them, so no code needs recovering -- only re-enabling.

# Cross-hand opponent image. Dormant because the Playground reseats every
# hand: 158 distinct tables for 157 hands, zero revisits, so the tracker
# holds at most one hand of evidence per opponent and `opponent_range_width`
# reads 1.00 on 53.3% of live decisions. **The measurement is a property of
# the format, not of the features.** In a format where the same opponent
# recurs across hands -- `headsup-ladder`, `headsup-sandbox-closed-beta`,
# `poker-tournament` (final tables especially) -- an image built across hands
# is exactly the information these carry, and they should be reconsidered
# before anything new is invented to replace them.
DORMANT_CROSS_HAND_FEATURE_NAMES: tuple[str, ...] = (
    "opponent_range_width",
    "opponent_max_wildness",
    "opponent_max_stickiness",
    "opponent_aggression_count",
)

# Heuristic risk blends. Dormant because each is a hand-authored combination
# of quantities the active contract already supplies raw (strength, pot
# geometry, street, player count), and hand-authored constants have this
# project's worst track record -- six found wrong or inert in a single day.
# Dropping them removes the constants, not the information. Reconsider them
# if an ablation ever shows the network failing to recover a blend it needs,
# which is the honest test of whether the hand-authoring was earning its keep.
DORMANT_RISK_BLEND_FEATURE_NAMES: tuple[str, ...] = (
    "risk_temperature",
    "risk_weakness",
    "risk_bet_pressure",
    "risk_distance_from_river",
    "risk_player_pressure",
)

DORMANT_FEATURE_NAMES: tuple[str, ...] = (
    *DORMANT_CROSS_HAND_FEATURE_NAMES,
    *DORMANT_RISK_BLEND_FEATURE_NAMES,
)

CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    *KEPT_FEATURE_NAMES,
    *STRENGTH_FEATURE_NAMES,
    *SUMMARY_FEATURE_NAMES,
    *TABLE_FEATURE_NAMES,
    *BOARD_TIER_FEATURE_NAMES,
    *HISTORY_FEATURE_NAMES,
    *BELIEF_FEATURE_NAMES,
)

FEATURE_NAMES_V8: tuple[str, ...] = (*CARD_FEATURE_NAMES, *CONTEXT_FEATURE_NAMES)
INPUT_SIZE_V8 = len(FEATURE_NAMES_V8)

# The extended contract, for a format that reactivates the dormant block.
# Dormant features are **appended after every active feature**, deliberately:
# appending leaves indices 0..INPUT_SIZE_V8-1 byte-for-byte unchanged, so
# enabling them is a strict extension rather than a re-partition. Inserting
# them among the context features instead would shift every later index and
# silently invalidate every existing artifact and corpus.
#
# Enabling is still a **schema event** -- fresh corpus, new candidate, and
# normalization recomputed -- because the network's input width changes. What
# appending buys is that it cannot corrupt anything that already exists.
FEATURE_NAMES_V8_EXTENDED: tuple[str, ...] = (
    *FEATURE_NAMES_V8,
    *DORMANT_FEATURE_NAMES,
)
INPUT_SIZE_V8_EXTENDED = len(FEATURE_NAMES_V8_EXTENDED)


def feature_names(*, include_dormant: bool = False) -> tuple[str, ...]:
    """The active contract, or the extended one with the dormant block.

    Pass ``include_dormant=True`` only in a format where the same opponent
    recurs across hands (heads-up ladder, tournament final tables); in the
    reseat-every-hand Playground the cross-hand features are measured dead.
    """

    return FEATURE_NAMES_V8_EXTENDED if include_dormant else FEATURE_NAMES_V8

CARD_BLOCK_SIZE = len(CARD_FEATURE_NAMES)
CONTEXT_BLOCK_SIZE = len(CONTEXT_FEATURE_NAMES)

# The two-encoder split is recorded by index, as v7's was, so a schema edit
# cannot silently re-partition the vector.
CARD_INDICES: tuple[int, ...] = tuple(range(CARD_BLOCK_SIZE))
CONTEXT_INDICES: tuple[int, ...] = tuple(
    range(CARD_BLOCK_SIZE, CARD_BLOCK_SIZE + CONTEXT_BLOCK_SIZE)
)


class Schema3Error(ValueError):
    """Raised when a vector or name set violates the schema-3 contract."""


def feature_index(name: str) -> int:
    """Index of ``name`` in the v8 vector, or a contract error."""

    try:
        return _INDEX_BY_NAME[name]
    except KeyError:
        if name in _DORMANT_SET:
            raise Schema3Error(
                f"schema-3 feature {name!r} is DORMANT, not absent: it is "
                "reserved for formats where the same opponent recurs across "
                "hands. Use FEATURE_NAMES_V8_EXTENDED / "
                "feature_names(include_dormant=True), and note that enabling "
                "it is a schema event (fresh corpus, new candidate)"
            ) from None
        raise Schema3Error(f"unknown schema-3 feature {name!r}") from None


def history_slot_index(street: str, slot: int, channel: str) -> int:
    """Index of one history-slot channel, validated against the layout."""

    if street not in HISTORY_STREETS:
        raise Schema3Error(f"unknown history street {street!r}")
    if not 0 <= slot < HISTORY_SLOTS:
        raise Schema3Error(f"history slot {slot} out of range")
    if channel not in HISTORY_CHANNELS:
        raise Schema3Error(f"unknown history channel {channel!r}")
    return feature_index(f"hist_{street}{slot}_{channel}")


def require_vector(features: tuple[float, ...] | list[float]) -> None:
    """Cheap structural validation for a claimed schema-3 vector."""

    if len(features) != INPUT_SIZE_V8:
        raise Schema3Error(
            f"schema-3 vector must have {INPUT_SIZE_V8} entries, "
            f"found {len(features)}"
        )


_INDEX_BY_NAME = {name: index for index, name in enumerate(FEATURE_NAMES_V8)}
_DORMANT_SET = frozenset(DORMANT_FEATURE_NAMES)

if len(_INDEX_BY_NAME) != len(FEATURE_NAMES_V8):  # pragma: no cover
    raise AssertionError("schema-3 feature names must be unique")
if _DORMANT_SET & set(FEATURE_NAMES_V8):  # pragma: no cover
    raise AssertionError("a dormant feature must not also be active")
if len(set(FEATURE_NAMES_V8_EXTENDED)) != len(  # pragma: no cover
    FEATURE_NAMES_V8_EXTENDED
):
    raise AssertionError("extended schema-3 names must be unique")

__all__ = [
    "BELIEF_BUCKETS",
    "BELIEF_FEATURE_NAMES",
    "BOARD_TIER_FEATURE_NAMES",
    "CARD_BLOCK_SIZE",
    "CARD_CODES",
    "CARD_FEATURE_NAMES",
    "CARD_INDICES",
    "CARD_PLANES",
    "CONTEXT_BLOCK_SIZE",
    "CONTEXT_FEATURE_NAMES",
    "CONTEXT_INDICES",
    "DORMANT_CROSS_HAND_FEATURE_NAMES",
    "DORMANT_FEATURE_NAMES",
    "DORMANT_RISK_BLEND_FEATURE_NAMES",
    "FEATURE_NAMES_V8",
    "FEATURE_NAMES_V8_EXTENDED",
    "INPUT_SIZE_V8_EXTENDED",
    "feature_names",
    "HISTORY_CHANNELS",
    "HISTORY_FEATURE_NAMES",
    "HISTORY_SLOTS",
    "HISTORY_STREETS",
    "INPUT_SIZE_V8",
    "KEPT_FEATURE_NAMES",
    "SCHEMA_VERSION",
    "STRENGTH_FEATURE_NAMES",
    "SUMMARY_FEATURE_NAMES",
    "Schema3Error",
    "TABLE_FEATURE_NAMES",
    "feature_index",
    "history_slot_index",
    "require_vector",
]
