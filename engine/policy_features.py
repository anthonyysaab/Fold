"""Feature names and action labels shared by every policy implementation.

The trained checkpoint, the PyTorch inference adapter, and the pure-Python
deployment build all validate against these exact names and orders.
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

LEGALITY_FEATURE_INDEXES: tuple[int, ...] = tuple(
    FEATURE_NAMES.index(name)
    for name in ("legal_fold", "legal_check_call", "legal_aggress")
)

__all__ = [
    "FEATURE_NAMES",
    "LABELS",
    "LEGALITY_FEATURE_INDEXES",
]
