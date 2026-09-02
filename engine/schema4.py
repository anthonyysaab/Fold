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
- **`equity_multiway`** — the player-count-conditioned strength input
  schema 3 lacked (owner queue item 1): the unconditioned multiway MC
  equity vs the active opponent count, identical by construction to the
  equity feeding g's boldness read at feature time.
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

import hashlib
import math
from collections.abc import Mapping

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
    # Owner queue item 1 (2026-08-29, approved): the schema-3 contract
    # had NO player-count-conditioned strength input — the canonical
    # percentile is deliberately invariant and the range pair is
    # heads-up, so the network had to synthesize multiway shrinkage from
    # percentile x player_count. This is the unconditioned multiway MC
    # equity against the ACTUAL active opponent count — and it is, by
    # construction, the SAME number that feeds g's boldness read at
    # feature time (one computation, two consumers): the network sees
    # exactly the quantity the sizing responds to.
    "equity_multiway",
    # Pre-harvest decision 2 (owner-confirmed 2026-08-31): hero's MC
    # equity where ONE opponent's holding is drawn from the pooled
    # 8-octile P3 posterior (octile by weight, then a combo uniformly
    # inside that octile's slice) and the rest stay random — the
    # correction the unconditioned read needs once opponents have PAID
    # to continue (measured mean correction -0.0999, growing by street
    # to -0.178 on the river). Pinned at 1,000 trials; emitted as
    # ``equity_multiway`` VERBATIM where the posterior is bit-identical
    # to the uniform prior, so the column only differs where there is
    # evidence.
    "equity_vs_posterior",
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

#: sha256 of the joined feature names — the identity a
#: ``feature_normalization`` block must carry to be schema-4's own
#: (pre-harvest decision 5, owner-confirmed 2026-08-31). Computed once
#: from the frozen tuple; a schema edit changes it by construction.
FEATURE_NAME_DIGEST = hashlib.sha256(
    "\n".join(FEATURE_NAMES_V9).encode("utf-8")
).hexdigest()

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


def normalization_stamp() -> dict[str, object]:
    """The stamp every v9 writer puts on its ``feature_normalization`` block."""

    return {
        "feature_schema_version": SCHEMA_VERSION_V9,
        "feature_name_sha": FEATURE_NAME_DIGEST,
    }


def require_normalization_stamp(block: Mapping[str, object]) -> None:
    """Refuse a ``feature_normalization`` block that is not schema-4's own.

    Pre-harvest decision 5, owner-confirmed 2026-08-31: before this
    stamp ``load_policy_v9`` accepted any length-matching schema-3
    array. No grandfather clause — the frozen v7/v8 writers emit no
    stamp, so an unstamped block is by definition not v9-produced.
    Also asserts the card-block identity transform (means == 0, stds ==
    1 on :data:`CARD_INDICES_V9`), which the length checks never
    inspected, and that every entry is a finite number (a poisoned
    context entry used to surface as a raw ``ValueError`` from the
    serve class's float coercion, or as silent NaN).
    """

    if not isinstance(block, Mapping):
        raise Schema4Error("feature_normalization must be an object")
    version = block.get("feature_schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or (
        version != SCHEMA_VERSION_V9
    ):
        raise Schema4Error(
            "feature_normalization is unstamped or foreign: "
            f"feature_schema_version must be the int {SCHEMA_VERSION_V9} "
            "(the frozen v7/v8 writers emit no stamp, so an unstamped "
            "block is by definition not v9-produced)"
        )
    if block.get("feature_name_sha") != FEATURE_NAME_DIGEST:
        raise Schema4Error(
            "feature_normalization feature_name_sha does not match "
            "schema 4's feature names"
        )
    means = block.get("means")
    stds = block.get("stds")
    if not isinstance(means, (list, tuple)) or not isinstance(stds, (list, tuple)):
        raise Schema4Error("feature_normalization means/stds must be arrays")
    if len(means) != INPUT_SIZE_V9 or len(stds) != INPUT_SIZE_V9:
        raise Schema4Error(
            "feature_normalization must cover the full schema-4 vector "
            f"({INPUT_SIZE_V9} entries)"
        )
    card_indices = set(CARD_INDICES_V9)
    for index, (mean, std) in enumerate(
        zip(means, stds, strict=True)
    ):
        if (
            isinstance(mean, bool)
            or isinstance(std, bool)
            or not isinstance(mean, (int, float))
            or not isinstance(std, (int, float))
        ):
            raise Schema4Error("feature_normalization entries must be numbers")
        mean_value, std_value = float(mean), float(std)
        if not math.isfinite(mean_value) or not math.isfinite(std_value):
            raise Schema4Error("feature_normalization entries must be finite")
        if index in card_indices and (mean_value != 0.0 or std_value != 1.0):
            raise Schema4Error(
                "the card block must be stored at identity scales "
                "(means == 0, stds == 1)"
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
    "FEATURE_NAME_DIGEST",
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
    "normalization_stamp",
    "require_normalization_stamp",
    "require_vector_v9",
]
