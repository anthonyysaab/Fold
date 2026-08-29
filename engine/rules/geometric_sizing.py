"""C2 — geometric-leverage sizing. Spec: ``engine/rules/README.md``.

The classical result: to be all-in by the river betting the same pot
fraction each street, the pot must grow geometrically —
``P·(1+2f)^n = P + 2E`` gives

    f_geo = ((1 + 2·SPR)^(1/n) − 1) / 2

with SPR = eff/pot and n the betting rounds remaining including this one.
Checks: SPR 13 on the flop → f = 1.0 exactly (three pot bets, the
textbook figure); SPR 4 → ~0.54; SPR 1 on the river → 1.0.

The blend carries no new constants: ``w = max(0, b)`` — value-heavy reads
move toward the geometric size, neutral and negative reads keep the lane
band untouched (pressure/bluff lines keep their old shape) —

    f_out = (1 − w)·f_lane + w·f_geo        while f_geo <= lane_top
    f_out = f_lane                          otherwise (rule inert)

**Above the lane top the rule returns the lane fraction untouched and
reports `fired: false`.** The geometric plan is infeasible there — even
the band maximum every street cannot get stacks in — so the rule has no
advice to give. At live SPR (median ~120) that is the normal case and
the candidate self-deactivates, which is the correct degradation. An
earlier version CLAMPED the blend at `lane_top` instead, which did the
opposite: every mildly value-leaning read at live depth jumped to the
band maximum. The formula above is the fix, and this docstring is the
place that previously disagreed with it.

Failure posture: unknown street or non-positive pot/eff → NOT FIRED, the
untouched lane fraction is returned with a reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

RULE_NAME = "C2-geometric-sizing"

#: Betting rounds remaining, INCLUDING the current one.
STREETS_REMAINING = {"preflop": 4, "flop": 3, "turn": 2, "river": 1}


@dataclass(frozen=True, slots=True)
class GeometricSizingParams:
    enabled: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.enabled, int) and not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")


DEFAULT_GEOMETRIC_SIZING = GeometricSizingParams()


@dataclass(frozen=True, slots=True)
class GeometricVerdict:
    rule: str
    fired: bool
    spr: float | None
    streets_remaining: int | None
    f_geo: float | None
    weight: float | None
    f_out: float
    reason: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "fired": self.fired,
            "spr": self.spr,
            "streets_remaining": self.streets_remaining,
            "f_geo": self.f_geo,
            "weight": self.weight,
            "f_out": self.f_out,
            "reason": self.reason,
        }


def geometric_fraction(spr: float, streets_remaining: int) -> float:
    """The closed form; raises only on impossible inputs."""

    if not math.isfinite(spr) or spr < 0.0:
        raise ValueError("spr must be finite and non-negative")
    if streets_remaining < 1:
        raise ValueError("streets_remaining must be at least 1")
    return ((1.0 + 2.0 * spr) ** (1.0 / streets_remaining) - 1.0) / 2.0


def blended_fraction(
    params: GeometricSizingParams,
    *,
    lane_fraction: float,
    boldness: float,
    pot: int,
    effective_stack: int,
    street: str,
    lane_top: float,
) -> GeometricVerdict:
    """Value-weighted blend of the lane fraction toward the geometric size.

    ``lane_fraction`` is g's own output for this state; disabled or inert
    paths return it BIT-IDENTICAL (the zero-diff invariant).
    """

    if not params.enabled:
        return GeometricVerdict(
            RULE_NAME, False, None, None, None, None, lane_fraction, "disabled"
        )
    n = STREETS_REMAINING.get(str(street).casefold())
    if n is None:
        return GeometricVerdict(
            RULE_NAME, False, None, None, None, None, lane_fraction,
            "unknown street: inert",
        )
    if pot <= 0 or effective_stack <= 0:
        return GeometricVerdict(
            RULE_NAME, False, None, n, None, None, lane_fraction,
            "no pot or no stack: inert",
        )
    weight = max(0.0, min(1.0, boldness))
    if weight == 0.0:
        return GeometricVerdict(
            RULE_NAME, False, None, n, None, 0.0, lane_fraction,
            "read not value-heavy: lane band untouched",
        )
    spr = effective_stack / pot
    f_geo = geometric_fraction(spr, n)
    if f_geo > lane_top:
        # SELF-DEACTIVATION, and it must be a return of the LANE value —
        # not a clamp of the blend. Above the lane top the geometric plan
        # is infeasible: even betting the band maximum every street cannot
        # get stacks in, so the rule has no advice and must not move the
        # size. Clamping the blend instead (the first draft) did the
        # opposite — at live SPR (~120) f_geo exceeds every lane top, so
        # every mildly value-leaning read jumped straight to the band
        # MAXIMUM. Found 2026-08-29; the spec was accepted on the
        # self-deactivation reading (README, C2).
        return GeometricVerdict(
            RULE_NAME, False, spr, n, f_geo, weight, lane_fraction,
            f"geometric target {f_geo:.3f} exceeds the lane top "
            f"{lane_top:.3f}: stacks cannot be gotten in, rule inert",
        )
    f_out = (1.0 - weight) * lane_fraction + weight * f_geo
    fired = f_out != lane_fraction
    return GeometricVerdict(
        RULE_NAME,
        fired,
        spr,
        n,
        f_geo,
        weight,
        f_out,
        "geometric blend applied" if fired else "blend landed on the lane value",
    )


__all__ = [
    "RULE_NAME",
    "STREETS_REMAINING",
    "GeometricSizingParams",
    "DEFAULT_GEOMETRIC_SIZING",
    "GeometricVerdict",
    "geometric_fraction",
    "blended_fraction",
]
