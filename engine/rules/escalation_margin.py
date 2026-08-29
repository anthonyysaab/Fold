"""C4 — escalation-priced call margins. Spec: ``engine/rules/README.md``.

Each re-raise multiplicatively filters the opponent's range toward its
top, so the equity a call needs rises with the raise count. Today's
street margins price the third raise of a street like the first bet.

    margin_added = kappa_e · max(0, opponent_raises_this_street − 1)

**kappa_e is ESTIMATED, not authored** — measured 2026-08-29 on 1,903
complete-information aggressive events
(``artifacts/evaluations/escalation-shift-estimate-2026-08-29.json``):
the k-th aggressor's equity against a random holding climbs
0.6436 → 0.7235 → 0.7626 for k = 1, 2, 3+, slope

    kappa_e = +0.0671 equity per extra raise (SE 0.0065, t ≈ 10.3).

The count deliberately EXCLUDES hero's own aggressive actions — the
existing ``raises_current_street`` feature counts them, which makes it
table escalation, not opponent pressure. The wildness composition is the
ladder's, not this module's: the added margin flows into
``neutral_price`` at the wiring point, so tracked maniacs dissolve it
through the one existing blend — no second mechanism.

Failure posture: a negative or malformed count reads as zero raises —
zero margin added, never a negative one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from engine.game_state import _AGGRESSIVE_ACTIONS, _mapping, _sequence

RULE_NAME = "C4-escalation-margin"

#: The measured value and its provenance; the dataclass default mirrors it.
KAPPA_E_ESTIMATED = 0.0671
KAPPA_E_SOURCE = "artifacts/evaluations/escalation-shift-estimate-2026-08-29.json"


@dataclass(frozen=True, slots=True)
class EscalationMarginParams:
    enabled: bool = False
    kappa_e: float = KAPPA_E_ESTIMATED

    def __post_init__(self) -> None:
        if isinstance(self.enabled, int) and not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if (
            isinstance(self.kappa_e, bool)
            or not isinstance(self.kappa_e, (int, float))
            or not math.isfinite(self.kappa_e)
            or not 0.0 <= self.kappa_e <= 0.5
        ):
            raise ValueError("kappa_e must be a finite number in [0, 0.5]")


DEFAULT_ESCALATION_MARGIN = EscalationMarginParams()


@dataclass(frozen=True, slots=True)
class EscalationVerdict:
    rule: str
    fired: bool
    opponent_raises: int
    margin_added: float
    reason: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "fired": self.fired,
            "opponent_raises": self.opponent_raises,
            "margin_added": self.margin_added,
            "reason": self.reason,
        }


def opponent_raises_this_street(
    table: Mapping[str, Any], street: str, hero_seat: int
) -> int:
    """Aggressive actions by OTHERS this street, from ``recentEvents``.

    The same event walk as ``game_state._aggression_count`` (whose
    aggressive-action set is imported, never restated), narrowed by seat.
    """

    count = 0
    street = str(street).casefold()
    for raw_event in _sequence(table.get("recentEvents") or [], "recentEvents"):
        event = _mapping(raw_event, "recentEvent")
        if str(event.get("street") or "").casefold() != street:
            continue
        summary_value = event.get("summary")
        if summary_value is None:
            continue
        summary = _mapping(summary_value, "recentEvent.summary")
        if summary.get("seatNumber") == hero_seat:
            continue
        if str(summary.get("action") or "").casefold() in _AGGRESSIVE_ACTIONS:
            count += 1
    return count


def escalation_margin(
    params: EscalationMarginParams, opponent_raises: int
) -> EscalationVerdict:
    """Extra call margin for facing escalation beyond the first wager."""

    if not params.enabled:
        return EscalationVerdict(RULE_NAME, False, 0, 0.0, "disabled")
    raises = max(0, int(opponent_raises)) if isinstance(opponent_raises, int) else 0
    extra = max(0, raises - 1)
    if extra == 0:
        return EscalationVerdict(
            RULE_NAME, False, raises, 0.0, "at most one opponent wager: base margin"
        )
    added = params.kappa_e * extra
    return EscalationVerdict(
        RULE_NAME,
        True,
        raises,
        added,
        f"{raises} opponent wagers this street: +{added:.4f} equity demanded",
    )


__all__ = [
    "RULE_NAME",
    "KAPPA_E_ESTIMATED",
    "KAPPA_E_SOURCE",
    "EscalationMarginParams",
    "DEFAULT_ESCALATION_MARGIN",
    "EscalationVerdict",
    "opponent_raises_this_street",
    "escalation_margin",
]
