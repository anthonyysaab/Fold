"""The composition — the only file that knows more than one rule exists.

Spec: ``engine/rules/README.md``, "Composition". Pipeline order, one
place, never distributed::

    read (g, depth-invariant) -> C5: b <- b·d      (damper first, wins ties)
    -> lane base f(b)         -> C2: value-blend toward f_geo
    -> stack cap s(b)                               (C3B deferred)
    -> target = min(pot arm, cap arm)
    -> C3A: snap-to-cover within band               (snap beats blend)
    -> engine legalization (_sized_action, unchanged sole authority)

Precedence, exhaustively: (1) C5 applies first and cannot be overridden;
(2) C3A's snap overrides C2's blended target inside its band; (3) C1 and
C4 are CALL-side rules and do not appear here — they wire into the call
ladder at Phase 4; (4) no rule adds a second wildness blend.

**Damper supremacy, and why there is a ``min``.** The one C5 effect is
``b <- b·d``, but the C2 blend is not monotone in b when the geometric
size sits BELOW the lane band (low SPR): damping b shrinks the geometric
weight and can raise the blended fraction. The invariant "no emitted size
exceeds the same state's size at d = 1" is therefore enforced by
construction: when the damper is active, the pipeline is evaluated at
both b·d and b, and the SMALLER target is emitted. In the non-monotone
regime the damper then has no effect — correct, because the geometric
blend already sized down further than the damper would.

**Zero-diff invariant.** With every dial off, the pipeline collapses to
exactly g: the same functions, the same argument order, bit-identical
outputs (``tests/test_rules_composition.py`` fuzzes this).

**Attribution.** Every composed wager names its setter — ``C3A`` when
snapped, ``stack-cap`` when the s·eff arm bound, ``C2`` when the blend
moved the pot arm, ``g`` otherwise — with the damper's verdict carried
separately (it modulates the read, not the arm). Exactly one setter per
wager, so a diagnosis never reconstructs precedence from effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from engine.aggression_sizing import (
    DEFAULT_SIZING_PARAMETERS,
    SizingParameters,
    active_bet_fraction,
    active_wager_from_fraction,
    aggressive_arms,
    aggressive_fractions,
)
from engine.rules.commitment_gate import (
    DEFAULT_COMMITMENT_GATE,
    CommitmentGateParams,
)
from engine.rules.coverage_targeting import (
    DEFAULT_SNAP_TO_COVER,
    SnapToCoverParams,
    SnapVerdict,
    snap_to_cover,
)
from engine.rules.escalation_margin import (
    DEFAULT_ESCALATION_MARGIN,
    EscalationMarginParams,
)
from engine.rules.geometric_sizing import (
    DEFAULT_GEOMETRIC_SIZING,
    GeometricSizingParams,
    GeometricVerdict,
    blended_fraction,
)
from engine.rules.ruin_damper import (
    DEFAULT_RUIN_DAMPER,
    DamperVerdict,
    RuinDamperParams,
    damped_boldness,
    damping,
)


@dataclass(frozen=True, slots=True)
class RuleLayerParams:
    """All five dials in one constructor argument — every default OFF.

    The engine takes one of these; a manifest block will build one at
    serve time (Phase-5+). Field order mirrors the pipeline.
    """

    damper: RuinDamperParams = DEFAULT_RUIN_DAMPER
    geometric: GeometricSizingParams = DEFAULT_GEOMETRIC_SIZING
    snap: SnapToCoverParams = DEFAULT_SNAP_TO_COVER
    commitment: "CommitmentGateParams" = None  # type: ignore[assignment]
    escalation: "EscalationMarginParams" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Late defaults avoid importing the call-side modules at class
        # definition time purely for defaults.
        if self.commitment is None:
            object.__setattr__(self, "commitment", DEFAULT_COMMITMENT_GATE)
        if self.escalation is None:
            object.__setattr__(self, "escalation", DEFAULT_ESCALATION_MARGIN)


DEFAULT_RULE_LAYER = RuleLayerParams()


def composed_sizing_record(
    sizing: SizingParameters = DEFAULT_SIZING_PARAMETERS,
    rules: RuleLayerParams = DEFAULT_RULE_LAYER,
) -> dict:
    """The g block plus the rules block, for v9 manifests and corpus headers.

    Lives here rather than in ``aggression_sizing`` so the g module never
    imports rules (the engine imports rules, rules import g — a record
    function in g would close a cycle).
    """

    from dataclasses import asdict

    from engine.aggression_sizing import sizing_record

    record = sizing_record(sizing)
    record["rules"] = {
        "damper": asdict(rules.damper),
        "geometric": asdict(rules.geometric),
        "snap": asdict(rules.snap),
        "commitment": asdict(rules.commitment),
        "escalation": asdict(rules.escalation),
    }
    return record


@dataclass(frozen=True, slots=True)
class ComposedWager:
    """One composed, PRE-LEGALIZATION wager with full attribution."""

    lane: str
    target: float  # chips beyond hero's current contribution
    to_amount: float  # contribution + target, before _sized_action's clamps
    set_by: str  # exactly one of: g, C2, stack-cap, C3A
    damper: DamperVerdict
    geometric: GeometricVerdict
    snap: SnapVerdict
    boldness_in: float
    boldness_used: float

    def verdicts(self) -> list[dict[str, object]]:
        return [
            self.damper.as_mapping(),
            self.geometric.as_mapping(),
            self.snap.as_mapping(),
        ]


def _aggressive_pipeline(
    boldness: float,
    *,
    pot: int,
    to_call: int,
    effective_stack: int,
    street: str,
    sizing: SizingParameters,
    geometric: GeometricSizingParams,
) -> tuple[float, GeometricVerdict, bool]:
    """One evaluation of the aggressive lane at a given boldness.

    Returns (target, C2 verdict, stack_arm_bound).
    """

    fraction, cap = aggressive_fractions(boldness, sizing)
    verdict = blended_fraction(
        geometric,
        lane_fraction=fraction,
        boldness=boldness,
        pot=pot,
        effective_stack=effective_stack,
        street=street,
        lane_top=sizing.aggressive_base + sizing.aggressive_span,
    )
    pot_arm, stack_arm = aggressive_arms(
        pot=pot,
        to_call=to_call,
        effective_stack=effective_stack,
        fraction=verdict.f_out,
        cap=cap,
    )
    target = min(pot_arm, stack_arm)
    return target, verdict, stack_arm < pot_arm


def compose_aggressive_target(
    *,
    boldness: float,
    pot: int,
    to_call: int,
    effective_stack: int,
    contribution: int,
    street: str,
    bankroll: int,
    exposure: int,
    covered_allin_to_amounts: Sequence[int] = (),
    sizing: SizingParameters = DEFAULT_SIZING_PARAMETERS,
    geometric: GeometricSizingParams = DEFAULT_GEOMETRIC_SIZING,
    snap: SnapToCoverParams = DEFAULT_SNAP_TO_COVER,
    damper: RuinDamperParams = DEFAULT_RUIN_DAMPER,
) -> ComposedWager:
    """The full aggressive-lane composition. Refuses ``to_call <= 0``
    exactly as g does — the v9 contract masks the lane at free spots."""

    if to_call <= 0:
        raise ValueError(
            "the aggressive lane is masked at to_call == 0 —"
            " unprovoked wagers belong to the active lane"
        )
    d_verdict = damping(damper, bankroll=bankroll, exposure=exposure)
    b_used = damped_boldness(boldness, d_verdict)
    target, c2_verdict, stack_bound = _aggressive_pipeline(
        b_used,
        pot=pot,
        to_call=to_call,
        effective_stack=effective_stack,
        street=street,
        sizing=sizing,
        geometric=geometric,
    )
    if d_verdict.d < 1.0:
        undamped_target, _, _ = _aggressive_pipeline(
            boldness,
            pot=pot,
            to_call=to_call,
            effective_stack=effective_stack,
            street=street,
            sizing=sizing,
            geometric=geometric,
        )
        target = min(target, undamped_target)

    to_amount = contribution + target
    snap_verdict = snap_to_cover(
        snap,
        to_amount=to_amount,
        covered_allin_to_amounts=covered_allin_to_amounts,
    )
    if snap_verdict.fired:
        to_amount = snap_verdict.to_amount
        target = to_amount - contribution
        set_by = "C3A"
    elif stack_bound:
        set_by = "stack-cap"
    elif c2_verdict.fired:
        set_by = "C2"
    else:
        set_by = "g"
    return ComposedWager(
        lane="aggressive",
        target=target,
        to_amount=to_amount,
        set_by=set_by,
        damper=d_verdict,
        geometric=c2_verdict,
        snap=snap_verdict,
        boldness_in=boldness,
        boldness_used=b_used,
    )


def compose_active_wager(
    *,
    boldness: float,
    pot: int,
    effective_stack: int,
    contribution: int,
    street: str,
    bankroll: int,
    exposure: int,
    covered_allin_to_amounts: Sequence[int] = (),
    sizing: SizingParameters = DEFAULT_SIZING_PARAMETERS,
    geometric: GeometricSizingParams = DEFAULT_GEOMETRIC_SIZING,
    snap: SnapToCoverParams = DEFAULT_SNAP_TO_COVER,
    damper: RuinDamperParams = DEFAULT_RUIN_DAMPER,
) -> ComposedWager:
    """The active-bet composition at ``to_call == 0``."""

    d_verdict = damping(damper, bankroll=bankroll, exposure=exposure)
    b_used = damped_boldness(boldness, d_verdict)

    def pipeline(b: float) -> tuple[float, GeometricVerdict]:
        fraction = active_bet_fraction(b, sizing)
        verdict = blended_fraction(
            geometric,
            lane_fraction=fraction,
            boldness=b,
            pot=pot,
            effective_stack=effective_stack,
            street=street,
            lane_top=sizing.active_base + sizing.active_span,
        )
        return active_wager_from_fraction(pot=pot, fraction=verdict.f_out), verdict

    target, c2_verdict = pipeline(b_used)
    if d_verdict.d < 1.0:
        undamped_target, _ = pipeline(boldness)
        target = min(target, undamped_target)

    to_amount = contribution + target
    snap_verdict = snap_to_cover(
        snap,
        to_amount=to_amount,
        covered_allin_to_amounts=covered_allin_to_amounts,
    )
    if snap_verdict.fired:
        to_amount = snap_verdict.to_amount
        target = to_amount - contribution
        set_by = "C3A"
    elif c2_verdict.fired:
        set_by = "C2"
    else:
        set_by = "g"
    return ComposedWager(
        lane="active",
        target=target,
        to_amount=to_amount,
        set_by=set_by,
        damper=d_verdict,
        geometric=c2_verdict,
        snap=snap_verdict,
        boldness_in=boldness,
        boldness_used=b_used,
    )


__all__ = [
    "ComposedWager",
    "DEFAULT_RULE_LAYER",
    "RuleLayerParams",
    "compose_active_wager",
    "compose_aggressive_target",
    "composed_sizing_record",
]
