"""C3 — coverage targeting, Rule A (snap-to-cover). Spec: ``engine/rules/README.md``.

Against an opponent hero covers, the maximum loss is THEIR stack and the
fold pressure on them is total. A raise that leaves a covered short stack
a few blinds behind buys the same fold decision as their all-in at worse
leverage — so when the composed target lands within the band BELOW a
covered opponent's all-in to-amount, snap to exactly that all-in (the
largest covered one in band: covering it covers the smaller)::

    (1 - band) * allin_j  <=  target_to  <=  allin_j

The band is two-sided by necessity: it closes a small gap UPWARD, and a
one-sided test admits every covered all-in however far below, which lets
``max()`` snap a large wager DOWN onto a tiny stack.

Rule B (damped stack-offs versus opponents who cover hero) is DEFERRED
into the C5 regime work — the archive that was to estimate its slope has
no support (Phase 2 results in the README).

Parameter provenance, stated because the house rule demands it: the snap
band is **OWNER-SET, 0.15, CONFIRMED by the owner 2026-08-29** — not
estimated: the estimation was attempted (``tools/estimate_snap_band.py``,
2026-08-29) and came back UNRESOLVED because 99.3% of field decisions
face bets under 20% of their own stack; the archive cannot see the
region. Re-estimate if a deeper-bet corpus ever exists.

Failure posture: no covered all-ins, malformed amounts, or a target below
every band → NOT FIRED, target returned untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

RULE_NAME = "C3A-snap-to-cover"


@dataclass(frozen=True, slots=True)
class SnapToCoverParams:
    enabled: bool = False
    #: OWNER-SET pending better data; see module docstring. Fraction of a
    #: covered opponent's all-in to-amount within which the snap applies.
    band: float = 0.15

    def __post_init__(self) -> None:
        if isinstance(self.enabled, int) and not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if (
            isinstance(self.band, bool)
            or not isinstance(self.band, (int, float))
            or not math.isfinite(self.band)
            or not 0.0 < self.band <= 0.5
        ):
            raise ValueError("band must be a finite number in (0, 0.5]")


DEFAULT_SNAP_TO_COVER = SnapToCoverParams()


@dataclass(frozen=True, slots=True)
class SnapVerdict:
    rule: str
    fired: bool
    to_amount: float
    snapped_to: float | None
    candidates: int
    reason: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "fired": self.fired,
            "to_amount": self.to_amount,
            "snapped_to": self.snapped_to,
            "candidates": self.candidates,
            "reason": self.reason,
        }


def snap_to_cover(
    params: SnapToCoverParams,
    *,
    to_amount: float,
    covered_allin_to_amounts: Sequence[int],
) -> SnapVerdict:
    """Snap a composed to-amount onto the largest in-band covered all-in.

    ``covered_allin_to_amounts`` holds, for each ACTIVE opponent hero
    covers, their all-in to-amount (``currentBet + stack``); the caller
    (Phase-4 wiring) extracts them from the snapshot. Legalization stays
    the engine's — a snapped amount still passes through
    ``_sized_action``'s clamps.
    """

    if not params.enabled:
        return SnapVerdict(RULE_NAME, False, to_amount, None, 0, "disabled")
    if to_amount <= 0.0 or not math.isfinite(to_amount):
        return SnapVerdict(
            RULE_NAME, False, to_amount, None, 0, "no positive target: inert"
        )
    # The band is TWO-SIDED. The rule exists to close a small gap upward
    # — "a raise that leaves a covered short stack a few blinds behind
    # buys the same fold decision at worse leverage" — so a candidate
    # all-in must sit at or above the composed target and within the band
    # below it. A one-sided test (the first draft) made every covered
    # all-in a candidate no matter how far below, and `max()` then snapped
    # a 5,000 wager DOWN onto a lone 100-chip all-in. Verified 2026-08-29.
    candidates = [
        int(amount)
        for amount in covered_allin_to_amounts
        if not isinstance(amount, bool)
        and isinstance(amount, (int, float))
        and amount > 0
        and (1.0 - params.band) * amount <= to_amount <= amount
    ]
    if not candidates:
        return SnapVerdict(
            RULE_NAME, False, to_amount, None, 0, "no covered all-in in band"
        )
    snapped = max(candidates)
    if snapped == to_amount:
        return SnapVerdict(
            RULE_NAME, False, to_amount, None, len(candidates),
            "target already exactly the covered all-in",
        )
    return SnapVerdict(
        RULE_NAME,
        True,
        float(snapped),
        float(snapped),
        len(candidates),
        f"snapped {to_amount:g} onto the largest in-band covered all-in {snapped}",
    )


__all__ = [
    "RULE_NAME",
    "SnapToCoverParams",
    "DEFAULT_SNAP_TO_COVER",
    "SnapVerdict",
    "snap_to_cover",
]
