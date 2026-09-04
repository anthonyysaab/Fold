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
from typing import Mapping, Sequence

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


def parameters_and_rules_from_record(
    record: Mapping[str, object],
) -> tuple[SizingParameters, RuleLayerParams]:
    """The identity-checking inverse of :func:`composed_sizing_record`.

    The companion to ``aggression_sizing.parameters_from_record``, which
    handles bare g records and REFUSES composed ones — loading a composed
    record through it would drop the dial states and reproduce the wrong
    sizes under the right identity.
    """

    from engine.aggression_sizing import G_IDENTITY

    identity = record.get("identity")
    if identity != G_IDENTITY:
        raise ValueError(
            f"sizing record identity {identity!r} is not {G_IDENTITY!r}"
        )
    parameters = record.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("sizing record must carry a parameters mapping")
    rules = record.get("rules")
    if not isinstance(rules, Mapping):
        raise ValueError(
            "not a composed record: no 'rules' block. Use"
            " engine.aggression_sizing.parameters_from_record"
        )
    return (
        SizingParameters.from_mapping(parameters),
        RuleLayerParams(
            damper=RuinDamperParams(**dict(rules["damper"])),
            geometric=GeometricSizingParams(**dict(rules["geometric"])),
            snap=SnapToCoverParams(**dict(rules["snap"])),
            commitment=CommitmentGateParams(**dict(rules["commitment"])),
            escalation=EscalationMarginParams(**dict(rules["escalation"])),
        ),
    )


@dataclass(frozen=True, slots=True)
class _LaneRun:
    """One evaluation of a lane at one boldness, with what set its target.

    Attribution MUST come from the run that produced the emitted number.
    When the damper's min picks the undamped run, attributing from the
    discarded damped run names the wrong setter and records a verdict
    whose f_out cannot reproduce the emitted target (found 2026-08-29).
    """

    target: float
    verdict: GeometricVerdict
    stack_bound: bool
    boldness: float


def _aggressive_pipeline(
    boldness: float,
    *,
    pot: int,
    to_call: int,
    effective_stack: int,
    street: str,
    sizing: SizingParameters,
    geometric: GeometricSizingParams,
) -> _LaneRun:
    """One evaluation of the aggressive lane at a given boldness."""

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
    return _LaneRun(
        target=min(pot_arm, stack_arm),
        verdict=verdict,
        stack_bound=stack_arm < pot_arm,
        boldness=boldness,
    )


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
    kwargs = dict(
        pot=pot,
        to_call=to_call,
        effective_stack=effective_stack,
        street=street,
        sizing=sizing,
        geometric=geometric,
    )
    run = _aggressive_pipeline(damped_boldness(boldness, d_verdict), **kwargs)
    if d_verdict.d < 1.0:
        undamped = _aggressive_pipeline(boldness, **kwargs)
        if undamped.target < run.target:
            run = undamped

    target = run.target
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
    elif run.stack_bound:
        set_by = "stack-cap"
    elif run.verdict.fired:
        set_by = "C2"
    else:
        set_by = "g"
    return ComposedWager(
        lane="aggressive",
        target=target,
        to_amount=to_amount,
        set_by=set_by,
        damper=d_verdict,
        geometric=run.verdict,
        snap=snap_verdict,
        boldness_in=boldness,
        boldness_used=run.boldness,
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

    def pipeline(b: float) -> _LaneRun:
        verdict = blended_fraction(
            geometric,
            lane_fraction=active_bet_fraction(b, sizing),
            boldness=b,
            pot=pot,
            effective_stack=effective_stack,
            street=street,
            lane_top=sizing.active_base + sizing.active_span,
        )
        return _LaneRun(
            target=active_wager_from_fraction(pot=pot, fraction=verdict.f_out),
            verdict=verdict,
            stack_bound=False,  # the active lane has no stack arm
            boldness=b,
        )

    run = pipeline(damped_boldness(boldness, d_verdict))
    if d_verdict.d < 1.0:
        undamped = pipeline(boldness)
        if undamped.target < run.target:
            run = undamped

    target = run.target
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
    elif run.verdict.fired:
        set_by = "C2"
    else:
        set_by = "g"
    return ComposedWager(
        lane="active",
        target=target,
        to_amount=to_amount,
        set_by=set_by,
        damper=d_verdict,
        geometric=run.verdict,
        snap=snap_verdict,
        boldness_in=boldness,
        boldness_used=run.boldness,
    )


__all__ = [
    "ComposedWager",
    "DEFAULT_RULE_LAYER",
    "RuleLayerParams",
    "compose_active_wager",
    "compose_aggressive_target",
    "composed_sizing_record",
    "parameters_and_rules_from_record",
]
