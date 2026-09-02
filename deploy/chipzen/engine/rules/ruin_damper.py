"""C5 — the ruin damper. Spec: ``engine/rules/README.md``.

Kelly (1956): with an absorbing ruin barrier, tolerable risk scales with
bankroll. Every earlier repair removed hero's raw stack from the rules
because it decays gates at a healthy roll; the unhandled direction is
the shrunken roll — the recorded death pattern (1,000 → 0 in 36 hands at
roll ≈ table scale). This module reintroduces the raw stack
DELIBERATELY, as a survival term, in its own dial — never inside g's
depth-invariant read: g asks what the table warrants, C5 asks what the
roll affords.

    exposure = deepest active opponent's total (stack + committed)
    d = min(1, bankroll / (kappa_r · exposure))
    effect: b ← b·d, before the lanes — every size cools continuously

One effect, one dial. Gate-side tightening is out of scope this
iteration.

Parameter provenance: **kappa_r = 8.0, ESTIMATED from the 2026-08-29
frontier sweep** (`tools/ruin_damper_sweep.py`, 32 seeds x grid
{2,3,5,8,10} on the frozen 60bb instrument, subject the noise-floor
champion — the v7 head never consults the damped arm, measured;
artifact `artifacts/evaluations/ruin-damper-sweep-2026-08-29.json`).
kr-8 bought the largest RESOLVED ruin reductions on both ruin-heavy
channels (vs-station −0.39 busts/100, vs-shover −0.03 at mde 0.0247)
with the EV cost resolved only vs-station (−29 BB/100, the classic
survival-vs-extraction price) and unresolved elsewhere. Re-sweep on the
v9 composed lanes before enabling the dial on any v9 artifact.

The composition honors the damper-supremacy invariant by emitting
``min(damped pipeline, undamped pipeline)`` — see ``composition.py`` for
why the blend makes the raw ``b·d`` non-monotone in one regime.

Failure posture: non-positive exposure → d = 1 (nothing at risk, no
damping); non-positive bankroll → d = 0 (nothing to afford, full damp).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

RULE_NAME = "C5-ruin-damper"

KAPPA_R_STATUS = (
    "ESTIMATED 2026-08-29 on the heuristic champion"
    " (ruin-damper-sweep-2026-08-29.json); re-sweep on v9 lanes before"
    " enabling on a v9 artifact"
)


@dataclass(frozen=True, slots=True)
class RuinDamperParams:
    enabled: bool = False
    kappa_r: float = 8.0

    def __post_init__(self) -> None:
        if isinstance(self.enabled, int) and not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if (
            isinstance(self.kappa_r, bool)
            or not isinstance(self.kappa_r, (int, float))
            or not math.isfinite(self.kappa_r)
            or self.kappa_r <= 0.0
        ):
            raise ValueError("kappa_r must be a positive finite number")


DEFAULT_RUIN_DAMPER = RuinDamperParams()


@dataclass(frozen=True, slots=True)
class DamperVerdict:
    rule: str
    fired: bool
    d: float
    bankroll: int
    exposure: int
    reason: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "fired": self.fired,
            "d": self.d,
            "bankroll": self.bankroll,
            "exposure": self.exposure,
            "reason": self.reason,
        }


def damping(
    params: RuinDamperParams, *, bankroll: int, exposure: int
) -> DamperVerdict:
    """The damping factor for this decision; 1.0 means no effect."""

    if not params.enabled:
        return DamperVerdict(RULE_NAME, False, 1.0, bankroll, exposure, "disabled")
    bankroll = int(bankroll)
    exposure = int(exposure)
    if exposure <= 0:
        return DamperVerdict(
            RULE_NAME, False, 1.0, bankroll, exposure, "no exposure: no damping"
        )
    if bankroll <= 0:
        return DamperVerdict(
            RULE_NAME, True, 0.0, bankroll, exposure, "no bankroll: full damp"
        )
    d = min(1.0, bankroll / (params.kappa_r * exposure))
    if d >= 1.0:
        return DamperVerdict(
            RULE_NAME, False, 1.0, bankroll, exposure,
            "roll comfortably covers the exposure",
        )
    return DamperVerdict(
        RULE_NAME,
        True,
        d,
        bankroll,
        exposure,
        f"roll at {d:.3f} of the comfortable multiple: sizes cooled",
    )


def damped_boldness(boldness: float, verdict: DamperVerdict) -> float:
    """The one effect: b ← b·d. Callers use this, never d directly."""

    return boldness * verdict.d


def table_exposure(table: Mapping[str, Any]) -> int:
    """The exposure unit: the deepest active opponent's stack + committed.

    Observable and table-scoped; returns 0 (=> no damping) when no active
    opponent has a readable stack, matching the module's failure posture.
    """

    from engine.game_state import _active_seats, _hero_and_seats

    hero, seats = _hero_and_seats(table)
    deepest = 0
    for seat in _active_seats(seats):
        if seat is hero:
            continue
        stack = seat.get("stackChips")
        committed = seat.get("currentBetChips") or 0
        if isinstance(stack, bool) or not isinstance(stack, (int, float)):
            continue
        if isinstance(committed, bool) or not isinstance(committed, (int, float)):
            committed = 0
        deepest = max(deepest, int(stack) + int(committed))
    return deepest


__all__ = [
    "RULE_NAME",
    "KAPPA_R_STATUS",
    "RuinDamperParams",
    "DEFAULT_RUIN_DAMPER",
    "DamperVerdict",
    "damping",
    "damped_boldness",
]
