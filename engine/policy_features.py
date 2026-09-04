"""Feature names and action labels shared by every policy implementation.

These 125 names are the schema-2 BASE BLOCK of the live v8/v9 feature vector,
not a legacy remnant -- despite having been written for a 125-input network
that no longer exists. `game_state.features_from_table` builds the vector from
this exact order on every decision by every policy, and
`feature_extract_v8.assemble` then looks the base values up BY NAME to compose
the wider schema-3/4 vectors. `learning_contract` derives
``LEARNING_FEATURE_NAMES`` and ``LEARNING_INPUT_SIZE`` structurally from it, and
`training_telemetry` indexes into it positionally.

``LABELS`` is the frozen three-family action vocabulary, shared by the decision
engine, the trainers and the simulator.

The three consumers this docstring used to name -- the trained checkpoint, the
PyTorch inference adapter and the pure-Python deployment build -- are all gone:
the first two were archived on 2026-08-16, and P2 deleted the third on
2026-09-04 along with ``LEGALITY_FEATURE_INDEXES``, whose only reader was the
retired network's legality mask.
"""

from __future__ import annotations

LABELS: tuple[str, ...] = ("fold", "check_call", "aggress")

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_CARD_CODES = tuple(f"{rank}{suit}" for rank in _RANKS for suit in _SUITS)
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

__all__ = [
    "FEATURE_NAMES",
    "LABELS",
]
