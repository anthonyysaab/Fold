"""C4 — escalation-priced call margins. Spec: ``engine/rules/README.md``.

Each re-raise multiplicatively filters the opponent's range toward its
top, so the equity a call needs rises with the wager count. Today's
street margins price the third raise of a street like the first bet::

    margin = ESCALATION_STEPS[min(street_aggressions, MEASURED_MAX_K)]

**The margin is a table of PER-K MEASURED STEPS.** From 1,903
complete-information aggressive events
(``artifacts/evaluations/escalation-shift-estimate-2026-08-29.json``,
``per_k_steps``), the k-th aggressor's equity against a random holding,
and its step from the k = 1 base of 0.6436::

    k = 2  n=231  step +0.0799        k = 5  n=8   step +0.1799
    k = 3  n= 55  step +0.0978        k = 6  n=2   step +0.2289
    k = 4  n= 16  step +0.1306        k = 7  n=2   step +0.2402

Each shipped entry is the step for ITS OWN k. That is the property two
earlier versions lacked, and both failures came from the same place —
``K_CAP``, a constant the estimator documents as a display bucket:

1. The first version fitted a linear **slope** to k capped at 3.
   Refitting the same rows at caps 2 / 3 / 5 / uncapped gives
   0.0904 / 0.0671 / 0.0551 / 0.0490 — a 2.8x SE spread, because the
   relationship is concave and no single slope survives the choice.
2. The second version replaced the slope with a step table whose
   terminal cell was the **pooled k >= 3 mean** (+0.1190) — but that
   cell is read at exactly ``count == 3``, where the measured step is
   +0.0978. It over-priced the modal 3-bet spot by 22% and moved with
   the cap exactly as the slope had. The dependence was relocated, not
   removed.

**Saturation above k = 3 is a POLICY choice, not a claim about the
data.** The relationship keeps rising through k = 7; the table stops
because the support does not (30 rows above k = 3, thinning to n = 2).
Holding the k = 3 step for longer streets deliberately UNDER-prices
them — the safe direction for a margin, since demanding too little on
thin evidence loses EV while demanding too much folds winners. Extend
the table only by measuring, never by extrapolating.

**The count is the STREET ORDINAL, hero's own aggression included.**
``estimate_escalation_shift`` indexes each event by its position among
all aggressive actions of its street, so the applied count must be the
same quantity. An earlier draft excluded hero's actions, which applied
the measured number to a different quantity and under-priced exactly
the bet-then-raised spot the margin exists for.

**How the margin composes with tracked wildness.** ``escalation_margin``
returns ``margin_applied`` already scaled by ``(1 - wildness)``, and the
wiring adds that. It is deliberately NOT added to ``neutral_price``: the
gate blend is ``required = (1 - w)*floor + w*neutral_price``, which
slides TOWARD neutral_price as wildness rises, so a margin placed there
would be *preserved* against a tracked maniac — backwards, since a
maniac's raises carry no range information.

Failure posture: a negative or malformed count reads as zero wagers —
zero margin, never a negative one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from engine.game_state import _AGGRESSIVE_ACTIONS, _mapping, _sequence

RULE_NAME = "C4-escalation-margin"

#: The measured step table: extra equity demanded at k wagers this
#: street, read directly off the per-k means in the artifact below. Index
#: is the wager count; entry 0 is unused, entry 1 is the base case.
#: Extending this table requires re-measuring, never extrapolating.
#: Per-k measured steps (index = wager count). Entry k is the step for
#: THAT k, never a pool over k and above — see the module docstring.
ESCALATION_STEPS: tuple[float, ...] = (0.0, 0.0, 0.0799, 0.0978)
#: The largest wager count this table prices explicitly. Counts above it
#: hold the last entry: a deliberate conservative policy where the
#: support thins, not a claim that the data saturates.
MEASURED_MAX_K = len(ESCALATION_STEPS) - 1
KAPPA_E_SOURCE = "artifacts/evaluations/escalation-shift-estimate-2026-08-29.json"


@dataclass(frozen=True, slots=True)
class EscalationMarginParams:
    enabled: bool = False
    #: Overridable for ablation; must stay non-decreasing and start at 0.
    steps: tuple[float, ...] = ESCALATION_STEPS

    def __post_init__(self) -> None:
        if isinstance(self.enabled, int) and not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        steps = tuple(self.steps)
        if len(steps) < 2 or steps[0] != 0.0 or steps[1] != 0.0:
            raise ValueError(
                "steps must start (0.0, 0.0): one wager is the base case"
            )
        previous = 0.0
        for value in steps:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 0.5
                or value < previous
            ):
                raise ValueError(
                    "each step must be a finite number in [0, 0.5] and"
                    " non-decreasing"
                )
            previous = value
        object.__setattr__(self, "steps", steps)


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

    The margin is read from the measured step table and SATURATES at the
    edge of its support; it is never extrapolated. ``wildness`` is the
    tracker's reading, and ``margin_applied`` is already scaled by
    ``(1 - wildness)`` so the journaled verdict records the number that
    actually moved the gate.
    """

    if not params.enabled:
        return EscalationVerdict(RULE_NAME, False, 0, 0.0, 0.0, 0.0, "disabled")
    count = max(0, int(aggressions)) if isinstance(aggressions, int) else 0
    w = min(1.0, max(0.0, float(wildness)))
    index = min(count, len(params.steps) - 1)
    raw = params.steps[index] if index >= 1 else 0.0
    if raw == 0.0:
        return EscalationVerdict(
            RULE_NAME, False, count, 0.0, w, 0.0,
            "at most one wager this street: base margin",
        )
    applied = raw * (1.0 - w)
    saturated = count > len(params.steps) - 1
    reason = f"{count} wagers this street: +{applied:.4f} equity demanded"
    if saturated:
        reason += f" (measured support ends at {len(params.steps) - 1})"
    if w:
        reason += f" (raw {raw:.4f} dissolved by wildness {w:.2f})"
    return EscalationVerdict(
        RULE_NAME, applied > 0.0, count, raw, w, applied, reason
    )


__all__ = [
    "RULE_NAME",
    "ESCALATION_STEPS",
    "MEASURED_MAX_K",
    "KAPPA_E_SOURCE",
    "EscalationMarginParams",
    "DEFAULT_ESCALATION_MARGIN",
    "EscalationVerdict",
    "escalation_margin",
    "street_aggressions",
]
