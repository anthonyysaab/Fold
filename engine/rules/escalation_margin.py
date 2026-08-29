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

**The count must match the estimand, and the estimand is the STREET
ORDINAL.** ``estimate_escalation_shift`` indexes each aggressive event by
its position among *all* aggressive actions of its street, hero's
included, and measures how much stronger the k-th aggressor is. So the
applied count is the street's total aggression count — the same quantity
``game_state._aggression_count`` (and the ``raises_current_street``
feature) already computes. An earlier draft excluded hero's own actions
on the reasoning that opponent pressure is the real signal; that is
defensible in the abstract but it silently applies the measured number
to a *different* quantity than it was measured on, under-pricing every
street where hero bet first and was raised — exactly the 3-bet spot the
margin exists for. Fixed 2026-08-29.

**How the margin composes with tracked wildness.** The wiring scales it
once, as ``margin += margin_added * (1 - wildness)``, and it is
deliberately NOT added to ``neutral_price``. The gate blend is
``required = (1 - w)*floor + w*neutral_price``: it slides TOWARD
neutral_price as wildness rises, so a margin placed there would be
*preserved* against a tracked maniac — exactly backwards, since a
maniac's raises carry no range information and the margin exists to
price range narrowing. An earlier draft of this docstring (and of the
spec) said "flows into neutral_price"; that wording was wrong and the
amendment is recorded in ``engine/rules/README.md``.

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
    street_aggressions: int
    margin_raw: float
    wildness: float
    margin_applied: float
    reason: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "fired": self.fired,
            "street_aggressions": self.street_aggressions,
            "margin_raw": self.margin_raw,
            "wildness": self.wildness,
            "margin_applied": self.margin_applied,
            "reason": self.reason,
        }


def street_aggressions(table: Mapping[str, Any], street: str) -> int:
    """Aggressive actions this street, from ``recentEvents``.

    Hero's own INCLUDED -- see the module docstring: this is the quantity
    kappa_e was measured against, and the applied count must be the
    measured one. Same event walk and same aggressive-action set as
    ``game_state._aggression_count`` (imported, never restated).
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
        if str(summary.get("action") or "").casefold() in _AGGRESSIVE_ACTIONS:
            count += 1
    return count


def escalation_margin(
    params: EscalationMarginParams,
    aggressions: int,
    wildness: float = 0.0,
) -> EscalationVerdict:
    """Extra call margin for facing escalation beyond the first wager.

    ``wildness`` is the tracker's reading; ``margin_applied`` is already
    scaled by ``(1 - wildness)``, so the journaled verdict records the
    number that actually moved the gate rather than a pre-scaling one a
    diagnosis would have to re-derive. See the module docstring for why
    the scaling belongs here and not in ``neutral_price``.
    """

    if not params.enabled:
        return EscalationVerdict(RULE_NAME, False, 0, 0.0, 0.0, 0.0, "disabled")
    count = max(0, int(aggressions)) if isinstance(aggressions, int) else 0
    extra = max(0, count - 1)
    w = min(1.0, max(0.0, float(wildness)))
    if extra == 0:
        return EscalationVerdict(
            RULE_NAME, False, count, 0.0, w, 0.0,
            "at most one wager this street: base margin",
        )
    raw = params.kappa_e * extra
    applied = raw * (1.0 - w)
    reason = f"{count} wagers this street: +{applied:.4f} equity demanded"
    if w:
        reason += f" (raw {raw:.4f} dissolved by wildness {w:.2f})"
    return EscalationVerdict(
        RULE_NAME, applied > 0.0, count, raw, w, applied, reason
    )
