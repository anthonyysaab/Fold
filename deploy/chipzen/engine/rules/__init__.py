"""The composed rule-layer candidates (C1–C5). Design: ``README.md`` here.

Five default-off dials, one composition. Nothing in this package is
consulted by any serve path until Phase-4 wiring, and every dial ships
OFF — importing this package changes no behavior anywhere.
"""

from engine.rules.commitment_gate import (
    CommitmentGateParams,
    CommitmentVerdict,
    DEFAULT_COMMITMENT_GATE,
    forward_commitment,
)
from engine.rules.composition import (
    ComposedWager,
    compose_active_wager,
    compose_aggressive_target,
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
    EscalationVerdict,
    escalation_margin,
    street_aggressions,
)
from engine.rules.geometric_sizing import (
    DEFAULT_GEOMETRIC_SIZING,
    GeometricSizingParams,
    GeometricVerdict,
    blended_fraction,
    geometric_fraction,
)
from engine.rules.ruin_damper import (
    DEFAULT_RUIN_DAMPER,
    DamperVerdict,
    RuinDamperParams,
    damped_boldness,
    damping,
)

__all__ = [
    "CommitmentGateParams",
    "CommitmentVerdict",
    "ComposedWager",
    "DEFAULT_COMMITMENT_GATE",
    "DEFAULT_ESCALATION_MARGIN",
    "DEFAULT_GEOMETRIC_SIZING",
    "DEFAULT_RUIN_DAMPER",
    "DEFAULT_SNAP_TO_COVER",
    "DamperVerdict",
    "EscalationMarginParams",
    "EscalationVerdict",
    "GeometricSizingParams",
    "GeometricVerdict",
    "RuinDamperParams",
    "SnapToCoverParams",
    "SnapVerdict",
    "blended_fraction",
    "compose_active_wager",
    "compose_aggressive_target",
    "damped_boldness",
    "damping",
    "escalation_margin",
    "forward_commitment",
    "geometric_fraction",
    "street_aggressions",
    "snap_to_cover",
]
