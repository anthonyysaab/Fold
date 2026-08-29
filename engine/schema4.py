"""Schema 4 — the v9 feature contract. Names, order, blocks, nothing else.

Sibling of :mod:`engine.schema3`, which stays byte-identical — v8 artifacts
keep loading against it. Every shared block (card planes, history slots,
belief buckets, table/summary/tier context, the dormant reserve) is
IMPORTED from schema 3, never restated; this module declares only what the
v9 branch contract changes:

- **Legality quartet** replaces the schema-3 trio: ``legal_fatal``,
  ``legal_passive``, ``legal_active``, ``legal_aggressive`` — one flag per
  ``branch_contract_v9.BRANCH_LABELS_V9`` lane. The extractor fills them
  from ``branch_contract_v9.legal_branches(available, to_call)`` — the
  contract is the single source of the masking rules (aggressive dark at
  ``to_call == 0``, fatal dark at a free check); this module only names
  the slots, in contract order.
- **Costs re-derived through g** (``engine.aggression_sizing``): the
  schema-3 pair ``cost_aggress_small_eff``/``cost_aggress_large_eff``
  (fixed 0.5/1.0-pot specs) collapses to one ``cost_aggressive_eff`` at
  g's aggressive target, and ``cost_call_eff`` generalizes to
  ``cost_active_eff`` — the active lane's cost is the call price when one
  exists and g's bet wager at a free spot. Costs are STATE-ONLY: derived
  from g's read on the snapshot, never from a model-proposed size (the
  circularity trap). Extraction uses the g parameter block recorded in
  the corpus header; changing g parameters is a corpus event, not a
  schema event.
- **One executability flag**: ``branch_small_executable``/
  ``branch_large_executable`` (a two-size distinctness test) collapse to
  ``branch_aggressive_executable`` — whether g's clamped aggressive
  to-amount survives legalization as a genuine escalation. This encodes
  the measured demotion mode: the sub-near-nut risk cap can push the cap
  below the minimum raise (ordinary facing bets above ~0.455 of the
  effective stack), in which case a chosen escalation demotes to active.

History slots and within-hand summaries keep the OBSERVABLE 3-family
vocabulary (``fold``/``check_call``/``aggress`` channels) verbatim from
schema 3: they describe what opponents were seen to do, which the v9
taxonomy does not change, and reindexing them would silently break every
stored normalization. The hero's own v9 branch identity travels in
telemetry fields, never in these channels.

Like schema 3, this contract is additive-only in the repo: a new contract
for a new model family. Normalization arrays, corpora, and
``ood_guard_indices`` are schema-scoped — regenerated for schema 4, never
carried across from schema 3.
"""

from __future__ import annotations

from engine.schema3 import (
    BELIEF_BUCKETS,
    BELIEF_FEATURE_NAMES,
    BOARD_TIER_FEATURE_NAMES,
    CARD_CODES,
    CARD_FEATURE_NAMES,
    CARD_PLANES,
    DORMANT_FEATURE_NAMES,
    HISTORY_CHANNELS,
    HISTORY_FEATURE_NAMES,
    HISTORY_SLOTS,
    HISTORY_STREETS,
    SUMMARY_FEATURE_NAMES,
    TABLE_FEATURE_NAMES,
)
from engine.branch_contract_v9 import BRANCH_LABELS_V9

SCHEMA_VERSION_V9 = 4

# --- Context block ---------------------------------------------------------

# Schema 3's kept block with the legality trio replaced by the v9 quartet,
# in BRANCH_LABELS_V9 order. Everything else is unchanged and unmoved.
_LEGALITY_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"legal_{branch}" for branch in BRANCH_LABELS_V9
)
KEPT_FEATURE_NAMES_V9: tuple[str, ...] = (
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
    *_LEGALITY_FEATURE_NAMES,
    "hole_known_fraction",
)

# Strength and range sensitivity carry over from schema 3 verbatim; the
# branch-cost axis is re-partitioned onto the two v9 wager lanes.
STRENGTH_FEATURE_NAMES_V9: tuple[str, ...] = (
    "strength_percentile",
    "equity_vs_top20",
    "equity_vs_top5",
    "equity_range_slope",
    "hand_ppot",
    "hand_npot",
    "cost_active_eff",
    "cost_aggressive_eff",
    "cost_allin_eff",
    "card_reveal_expense",
    "branch_aggressive_executable",
)

CONTEXT_FEATURE_NAMES_V9: tuple[str, ...] = (
    *KEPT_FEATURE_NAMES_V9,
    *STRENGTH_FEATURE_NAMES_V9,
    *SUMMARY_FEATURE_NAMES,
    *TABLE_FEATURE_NAMES,
    *BOARD_TIER_FEATURE_NAMES,
    *HISTORY_FEATURE_NAMES,
    *BELIEF_FEATURE_NAMES,
)

FEATURE_NAMES_V9: tuple[str, ...] = (*CARD_FEATURE_NAMES, *CONTEXT_FEATURE_NAMES_V9)
INPUT_SIZE_V9 = len(FEATURE_NAMES_V9)

# The dormant reserve keeps schema 3's append-only discipline: appended
# after every active feature so enabling it is a strict extension, and
# enabling remains a schema event (fresh corpus, new candidate).
FEATURE_NAMES_V9_EXTENDED: tuple[str, ...] = (
    *FEATURE_NAMES_V9,
    *DORMANT_FEATURE_NAMES,
)
INPUT_SIZE_V9_EXTENDED = len(FEATURE_NAMES_V9_EXTENDED)


def feature_names_v9(*, include_dormant: bool = False) -> tuple[str, ...]:
    """The active v9 contract, or the extended one with the dormant block."""

    return FEATURE_NAMES_V9_EXTENDED if include_dormant else FEATURE_NAMES_V9


CARD_BLOCK_SIZE_V9 = len(CARD_FEATURE_NAMES)
CONTEXT_BLOCK_SIZE_V9 = len(CONTEXT_FEATURE_NAMES_V9)

# The two-encoder split is recorded by index, exactly as schemas 2 and 3
# record theirs, so a schema edit cannot silently re-partition the vector.
CARD_INDICES_V9: tuple[int, ...] = tuple(range(CARD_BLOCK_SIZE_V9))
CONTEXT_INDICES_V9: tuple[int, ...] = tuple(
    range(CARD_BLOCK_SIZE_V9, CARD_BLOCK_SIZE_V9 + CONTEXT_BLOCK_SIZE_V9)
)


class Schema4Error(ValueError):
    """Raised when a vector or name set violates the schema-4 contract."""


def feature_index_v9(name: str) -> int:
    """Index of ``name`` in the v9 vector, or a contract error."""

    try:
        return _INDEX_BY_NAME_V9[name]
    except KeyError:
        if name in _DORMANT_SET_V9:
            raise Schema4Error(
                f"schema-4 feature {name!r} is DORMANT, not absent: use "
                "FEATURE_NAMES_V9_EXTENDED / "
                "feature_names_v9(include_dormant=True), and note that "
                "enabling it is a schema event (fresh corpus, new candidate)"
            ) from None
        if name in _SCHEMA3_ONLY:
            raise Schema4Error(
                f"{name!r} is a schema-3 feature the v9 contract replaced: "
                "the legality trio became the four legal_<branch> flags and "
                "the branch-cost pair became cost_active_eff / "
                "cost_aggressive_eff / branch_aggressive_executable"
            ) from None
        raise Schema4Error(f"unknown schema-4 feature {name!r}") from None


def legality_index(branch: str) -> int:
    """Index of one ``legal_<branch>`` flag, validated against the contract."""

    if branch not in BRANCH_LABELS_V9:
        raise Schema4Error(f"unknown v9 branch {branch!r}")
    return feature_index_v9(f"legal_{branch}")


def require_vector_v9(features: tuple[float, ...] | list[float]) -> None:
    """Cheap structural validation for a claimed schema-4 vector."""

    if len(features) != INPUT_SIZE_V9:
        raise Schema4Error(
            f"schema-4 vector must have {INPUT_SIZE_V9} entries, "
            f"found {len(features)}"
        )


_INDEX_BY_NAME_V9 = {name: index for index, name in enumerate(FEATURE_NAMES_V9)}
_DORMANT_SET_V9 = frozenset(DORMANT_FEATURE_NAMES)
_SCHEMA3_ONLY = frozenset(
    {
        "legal_fold",
        "legal_check_call",
        "legal_aggress",
        "cost_call_eff",
        "cost_aggress_small_eff",
        "cost_aggress_large_eff",
        "branch_small_executable",
        "branch_large_executable",
    }
)

if len(_INDEX_BY_NAME_V9) != len(FEATURE_NAMES_V9):  # pragma: no cover
    raise AssertionError("schema-4 feature names must be unique")
if _DORMANT_SET_V9 & set(FEATURE_NAMES_V9):  # pragma: no cover
    raise AssertionError("a dormant feature must not also be active")
if _SCHEMA3_ONLY & set(FEATURE_NAMES_V9):  # pragma: no cover
    raise AssertionError("a replaced schema-3 feature must not appear in v9")
if len(set(FEATURE_NAMES_V9_EXTENDED)) != len(  # pragma: no cover
    FEATURE_NAMES_V9_EXTENDED
):
    raise AssertionError("extended schema-4 names must be unique")

__all__ = [
    "BELIEF_BUCKETS",
    "CARD_BLOCK_SIZE_V9",
    "CARD_CODES",
    "CARD_FEATURE_NAMES",
    "CARD_INDICES_V9",
    "CARD_PLANES",
    "CONTEXT_BLOCK_SIZE_V9",
    "CONTEXT_FEATURE_NAMES_V9",
    "CONTEXT_INDICES_V9",
    "FEATURE_NAMES_V9",
    "FEATURE_NAMES_V9_EXTENDED",
    "HISTORY_CHANNELS",
    "HISTORY_SLOTS",
    "HISTORY_STREETS",
    "INPUT_SIZE_V9",
    "INPUT_SIZE_V9_EXTENDED",
    "KEPT_FEATURE_NAMES_V9",
    "SCHEMA_VERSION_V9",
    "STRENGTH_FEATURE_NAMES_V9",
    "Schema4Error",
    "feature_index_v9",
    "feature_names_v9",
    "legality_index",
    "require_vector_v9",
]
