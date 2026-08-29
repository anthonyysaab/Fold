"""C4 — escalation-priced call margins. Spec: ``engine/rules/README.md``.

Each re-raise multiplicatively filters the opponent's range toward its
top, so the equity a call needs rises with the wager count. Today's
street margins price the third raise of a street like the first bet::

    margin = ESCALATION_STEPS[min(street_aggressions, MEASURED_MAX_K)]

**The margin is a MEASURED STEP TABLE, not an extrapolated slope.**
Measured 2026-08-29 on 1,903 complete-information aggressive events
(``artifacts/evaluations/escalation-shift-estimate-2026-08-29.json``):
the k-th aggressor's equity against a random holding reads

    k = 1: 0.6436 (n=1587)   k = 2: 0.7235 (n=231)   k >= 3: 0.7626 (n=85)

so the extra equity a call must find is the STEP from k = 1 — +0.0799 at
k = 2, +0.1190 at k >= 3 — read straight off the measurement.

An earlier version fitted a single linear slope (``kappa_e = 0.0671``)
and multiplied it by an unbounded ``count - 1``. That was wrong twice.
(1) **The slope was an artifact of the reporting cap**: refitting the
same 1,903 rows with the k-bucket cap at 2 / 3 / 5 / uncapped gives
0.0904 / 0.0671 / 0.0551 / 0.0490 — a spread of 2.8x the published
standard error, on a constant the module described as a display
bucket. The relationship is concave, so no single slope survives the
choice. (2) **It extrapolated past its support**: k > 3 has 30 rows
total and the slope kept multiplying, so a five-bet street demanded a
margin larger than the parameter validator's own legal ceiling. The
step table has neither failure mode: it is the measurement itself, and
it saturates at the edge of the data by construction.

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
ESCALATION_STEPS: tuple[float, ...] = (0.0, 0.0, 0.0799, 0.1190)
#: The largest wager count the measurement supports. Counts above it read
#: the last measured step — saturation, not extrapolation.
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
