"""C1 — the forward-commitment gate. Spec: ``engine/rules/README.md``.

A call is not priced by this street alone: it creates next-street
geometry. Facing a shove of the remaining stack E' into pot P', the price
is ``E'/(P'+2E') = 1/(2 + 1/SPR')`` — at SPR' = 1 that is 1/3, odds
nearly any hand "has", so below SPR' ~ 1 the call IS a stack-off in
installments.

This module only *decides*; it never applies a floor. When it fires, the
call ladder evaluates the call as if the strictest existing call stack
gate had tripped — same floor (``call_stack_gates[0]``), same reveal
penalty, same wildness slide, the existing code path verbatim. C1 adds a
trigger, never a floor or a blend, and both its constants are derived:
tau = 1 from the shove-price identity, the floor by reuse.

Failure posture: anything malformed, and any state without a positive
price, is NOT FIRED with a reason — the gate can only ever add a check,
never remove one, so inert is the safe direction (and ``equity is None``
deadline handling stays the ladder's, untouched).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

RULE_NAME = "C1-forward-commitment"


@dataclass(frozen=True, slots=True)
class CommitmentGateParams:
    """Dial and derived threshold; both recorded in any manifest block."""

    enabled: bool = False
    #: Derived, not tuned: SPR' at which a next-street shove prices at 1/3.
    spr_threshold: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.enabled, int) and not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if (
            isinstance(self.spr_threshold, bool)
            or not isinstance(self.spr_threshold, (int, float))
            or not math.isfinite(self.spr_threshold)
            or self.spr_threshold <= 0.0
        ):
            raise ValueError("spr_threshold must be a positive finite number")


DEFAULT_COMMITMENT_GATE = CommitmentGateParams()


@dataclass(frozen=True, slots=True)
class CommitmentVerdict:
    rule: str
    fired: bool
    spr_post: float | None
    reason: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "fired": self.fired,
            "spr_post": self.spr_post,
            "reason": self.reason,
        }


def forward_commitment(
    params: CommitmentGateParams,
    *,
    gate_stack: int,
    to_call: int,
    pot: int,
) -> CommitmentVerdict:
    """Should this call be judged as a stack-off?

    ``gate_stack`` is the call ladder's own denominator (`_gate_stack`),
    so C1 inherits every existing denominator dial; ``pot`` is raw
    ``potChips``, the engine convention.
    """

    if not params.enabled:
        return CommitmentVerdict(RULE_NAME, False, None, "disabled")
    if to_call <= 0:
        return CommitmentVerdict(RULE_NAME, False, None, "no price to commit to")
    if pot < 0 or gate_stack < 0:
        return CommitmentVerdict(RULE_NAME, False, None, "malformed state: inert")
    spr_post = (gate_stack - to_call) / (pot + 2 * to_call)
    if spr_post <= params.spr_threshold:
        return CommitmentVerdict(
            RULE_NAME,
            True,
            spr_post,
            f"post-call SPR {spr_post:.3f} <= {params.spr_threshold:g}:"
            " a next-street shove prices at 1/3 or better",
        )
    return CommitmentVerdict(RULE_NAME, False, spr_post, "geometry uncommitted")


__all__ = [
    "RULE_NAME",
    "CommitmentGateParams",
    "DEFAULT_COMMITMENT_GATE",
    "CommitmentVerdict",
    "forward_commitment",
]
