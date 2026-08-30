"""Turn policy proposals into safe, legal poker actions.

:class:`DecisionEngine` holds the safety rails that were originally methods of
the earlier policy: legal-action mapping, bounded raise sizing, equity
fallbacks for short-handed tables, and the sub-two-second deadline path. A
backend can supply a proposal family and override the documented policy-tuning
hooks. The shared engine keeps legal-action mapping and hard safety gates in one
place.

The per-decision risk temperature also shapes normal play through
:class:`TemperatureShaping`: cold readings (strong, cheap, late, short-handed)
loosen the aggression floor, call margins, and sizing within fixed bounds, and
hot readings tighten them. Hard safety gates never shift, and the reading still
never enters an Arena payload.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, replace as dataclass_replace
from typing import Any

from bluff import BluffAdvice, BluffSettings, DEFAULT_BLUFF_SETTINGS, evaluate_bluff
from lead_position import measure_lead_position
from risk_temperature import RiskTemperature, measure_risk_temperature

from engine.hand_strength import board_improvement, estimate_equity
from engine.learning_contract import build_learning_features
from engine.opponent_model import AggressionTracker
from engine.rules.commitment_gate import forward_commitment
from engine.rules.composition import DEFAULT_RULE_LAYER, RuleLayerParams
from engine.rules.escalation_margin import escalation_margin, street_aggressions
from engine.rules.ruin_damper import damped_boldness, damping, table_exposure
from engine.policy_features import LABELS
from engine.game_state import (
    _active_seats,
    _AGGRESSIVE_ACTIONS,
    active_opponent_count,
    ArenaSnapshotError,
    _cards,
    card_reveal_expense,
    contested_stack_chips,
    effective_stack_chips,
    _hero_and_seats,
    _integer,
    _mapping,
    _position,
    _sequence,
    features_from_table,
)


@dataclass(frozen=True, slots=True)
class ArenaAction:
    action: str
    amount: int | None
    message: str
    reasoning: str | None = None

    def to_payload(self) -> dict[str, str | int]:
        payload: dict[str, str | int] = {
            "action": self.action,
            "message": self.message,
        }
        if self.amount is not None:
            payload["amount"] = self.amount
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        return payload


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Arena action plus private diagnostics for logging and future training."""

    action: ArenaAction
    family: str
    equity: float | None
    situation_temperature: RiskTemperature | None
    learning_features: tuple[float, ...] | None = None
    behavior_probabilities: tuple[float, float, float] | None = None
    proposed_risk_fraction: float | None = None
    deadline_fallback: bool = False
    #: engine/rules attributions that FIRED on this decision, in firing
    #: order (as_mapping() dicts). None when no rule fired — which is
    #: every decision while the dials ship OFF, so records stay
    #: byte-identical to the pre-rules era until a dial is turned.
    rule_verdicts: tuple[dict[str, object], ...] | None = None
    temperature_boldness: float | None = None
    opponent_range_width: float | None = None
    opponent_evidence_confidence: float = 0.0
    lead_position: float | None = None
    bluff_kind: str | None = None
    hyper_aggression: bool = False

    def to_payload(self) -> dict[str, str | int]:
        return self.action.to_payload()


@dataclass(frozen=True, slots=True)
class TemperatureShaping:
    """Bounded response of normal play to the situational risk temperature.

    :meth:`boldness` maps a 0-100 reading onto ``[-1.0, +1.0]``: readings
    colder than ``setpoint`` push toward ``+1`` (strong, cheap, late, or
    short-handed situations play looser) and hotter readings push toward
    ``-1`` (weak, expensive, early, or crowded situations play tighter).
    The three shift fields bound how far the normal-play knobs may move:

    * ``aggression_floor_shift`` -- equity the aggression floor gains or
      loses at full boldness.
    * ``call_margin_shift`` -- extra-equity call margin shaved or added at
      full boldness.
    * ``sizing_span`` -- relative change of the half-pot target size at
      full boldness.

    Hard safety gates -- board-contribution floors, the re-raise war floor,
    the sub-0.72-equity risk cap, server legality, and the deadline path --
    never shift.

    Every field is a future learned parameter: an approved artifact may ship
    its own shaping values, and the per-decision reading plus its factors are
    already model inputs (``learning_contract.EXTRA_FEATURE_NAMES``).
    """

    setpoint: float = 45.0
    span: float = 35.0
    aggression_floor_shift: float = 0.078  # was 0.06, softened 30%
    call_margin_shift: float = 0.039  # was 0.03, softened 30%
    sizing_span: float = 0.39  # was 0.30, softened 30%

    def __post_init__(self) -> None:
        if not 0.0 <= self.setpoint <= 100.0:
            raise ValueError("setpoint must be between 0 and 100")
        if not 5.0 <= self.span <= 100.0:
            raise ValueError("span must be between 5 and 100")
        if not 0.0 <= self.aggression_floor_shift <= 0.15:
            raise ValueError("aggression_floor_shift must be between 0 and 0.15")
        if not 0.0 <= self.call_margin_shift <= 0.10:
            raise ValueError("call_margin_shift must be between 0 and 0.10")
        if not 0.0 <= self.sizing_span <= 0.50:
            raise ValueError("sizing_span must be between 0 and 0.5")

    def boldness(self, temperature: float) -> float:
        """Signed, clamped distance below the setpoint, in span units."""

        return max(-1.0, min(1.0, (self.setpoint - temperature) / self.span))

    def to_mapping(self) -> dict[str, object]:
        """JSON-ready form for a future learned-artifact manifest."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "TemperatureShaping":
        """Rebuild validated shaping from artifact JSON; unknown keys fail."""

        return cls(**dict(mapping))


DEFAULT_TEMPERATURE_SHAPING = TemperatureShaping()
NEUTRAL_TEMPERATURE_SHAPING = TemperatureShaping(
    aggression_floor_shift=0.0, call_margin_shift=0.0, sizing_span=0.0
)


# Extra equity demanded over raw pot odds to continue against a bet, by
# street. Replaces the old flat -0.03 loosening that made almost any pair a
# call against any bet size.
_CALL_MARGINS = {"preflop": 0.0, "flop": 0.02, "turn": 0.05, "river": 0.08}

# How much of the aggressor's range to simulate against on discounted board
# tiers. This conditions the equity estimate; it is estimation, not a gate,
# so it lives outside SafetyGates.
_BOARD_DISCOUNT_RANGE_TIGHTEN = {"kicker": 0.60, "thin": 0.75}

# Owner-fixed anti-modeling noise (2026-08-12): on a salted per-decision
# dice roll, one decision plays hyper-aggressively -- the aggression floor
# drops, sizing targets the full pot, and the bluff mixer's roll is forced
# open. Deliberately HARDCODED rather than a learnable parameter, so
# EV-maximizing training can never erode the unpredictability floor; and
# deliberately inside every hard safety gate, so the noise can neither
# stack off through the risk cap nor bluff a paired board. Hyper decisions
# are flagged in diagnostics and excluded from training labels.
#
# Lowered 5% -> 1% on 2026-08-15 (owner instruction), then set to 2% the same
# day as the value to run live while the ablation is pending. The measured
# case: the served policy is already aggressive on ~71% of live decisions, so a
# 5% random-aggression injection is far below the policy's own variability and
# obscures nothing an observer could detect -- noise only hides a signal when
# it is comparable to that signal's natural spread. Against that it cost a
# measured 4.28% of live decisions (58 of 1,355), every one of them excluded
# from training labels. 2% keeps a floor against genuinely adaptive Arena
# opponents while cutting both the EV leak and the data loss well over half.
# The price of the floor has still never been measured; the constructor
# override exists precisely for that ablation, queued at 0/2/5%.
# OWNER 2026-08-30: the v9 serve path defaults this to 0.0 in its own
# constructor (learned_policy_v9 — "it is part of the bluffing behaviour
# anyway"). This constant stays 0.02 for the v7/v8 paths ONLY because the
# frozen instruments' per-seed numbers bake it in and record no chance of
# their own; changing it here breaks every reproduction gate.
HYPER_AGGRESSION_CHANCE = 0.02
_HYPER_FLOOR_DROP = 0.12
_HYPER_POT_FRACTION = 1.0


@dataclass(frozen=True, slots=True)
class SafetyGates:
    """Every bounded numeric safety net of the engine, as one parameter set.

    The board-contribution gates (v3) come from three rated-match stack-offs
    where the five-card hand was mostly the board's: trips-plus-kicker and
    hollow two pair on paired boards kept calling into ranges stuffed with
    boats. Discounted tiers demand a bigger margin, stop stacking off, and
    stop barreling; ``fresh`` hands are untouched.

    Defaults are the 2026-08-12 owner decision to soften every net by 30%
    from its original value: added margins scale by 0.70, risk allowances
    and gate triggers by 1.30, and equity floors keep 70% of their excess
    over coin-flip equity (``0.50 + 0.70 * (floor - 0.50)``), so no floor
    drops below break-even. ``UNSOFTENED_SAFETY_GATES`` preserves the
    originals for comparison runs.

    Server legality, the deadline path, the check-fold-call failure order,
    and owner-gated money stops are not tuning bounds and are not here.

    Every field is a future learned parameter: an approved artifact may
    ship its own gate values through :meth:`from_mapping`, and the
    validation ranges below define the legal search space for training.
    """

    # Extra call margin demanded on discounted board tiers.
    board_margin_kicker: float = 0.084  # was 0.12
    board_margin_thin: float = 0.07  # was 0.10
    # (stack-fraction trigger, equity floor) stack-off gates per tier.
    board_stackoff_kicker: tuple[float, float] = (0.234, 0.675)  # was (0.18, 0.75)
    board_stackoff_thin: tuple[float, float] = (0.39, 0.71)  # was (0.30, 0.80)
    # Equity needed to keep betting a hand that barely improves the board.
    board_aggression_floor_kicker: float = 0.724  # was 0.82
    board_aggression_floor_thin: float = 0.696  # was 0.78
    # Equity at which the re-raise war floor and the sizing risk cap release.
    near_nut_floor: float = 0.654  # was 0.72
    # Largest stack fraction a sub-near-nut bet or raise may put at risk.
    risk_cap_stack_fraction: float = 0.455  # was 0.35
    # Generic (stack-fraction trigger, equity floor) call gates.
    call_stack_gates: tuple[tuple[float, float], ...] = (
        (0.78, 0.626),  # was (0.60, 0.68)
        (0.455, 0.584),  # was (0.35, 0.62)
    )
    # A fold-family hand may still call at this equity above pot odds.
    rescue_call_floor: float = 0.57  # was 0.60
    rescue_call_margin: float = 0.105  # was 0.15
    # Extra equity demanded per unit of card-reveal expense
    # (`game_state.card_reveal_expense`): paying a large share of what you
    # can lose, with cards still to come, is a bet on unseen cards and is
    # priced as one. Zero reproduces the pre-2026-08-15 gates exactly.
    #
    # Reverted to 0.0 on 2026-08-26 after measurement
    # (`gate-decision-2026-08-26.md`: reverting all three was +16.49
    # BB/100 on the card-aware vs-p3 channel, t = 5.34), then **ROLLED
    # BACK to 0.12 the same day** when the deployed revert busted the live
    # bankroll 1,000 -> 0 in 36 hands. See the rollback note on
    # `risk_cap_on_effective_stack` below; this slope travels with the
    # call gates because it raises their floor and never creates a gate.
    reveal_expense_equity_slope: float = 0.12
    # Denominator for the sizing risk cap and for the call stack gates:
    # the EFFECTIVE stack (True, live since 2026-08-15) or hero's own
    # purse (False, the pre-2026-08-15 form). False reproduces the old
    # denominator exactly and exists so each change can be ablated rather
    # than argued about -- see `.handoff/PENDING_EDITS.md`, "Three
    # unmeasured gate changes sit in the live path". Both default to the
    # live behaviour, so constructing gates without naming them changes
    # nothing.
    #
    # They are SEPARATE flags because the two edits are not one decision.
    # Measured on the stored live journal (`gate-binding-audit-2026-08-26`):
    # the cap re-denomination clips 1.35% of this policy's sub-near-nut
    # bets and lands on 33 hands worth -1,193 chips in total, while the
    # call re-denomination fires on 3.31% of calls and lands on 10 hands
    # worth -13,114. One dial for both would force a single verdict on two
    # changes whose evidence points in different directions.
    #
    # Reverted to False on 2026-08-26 after measurement, then **ROLLED
    # BACK to True the same day.** The measurement stands and is not
    # withdrawn -- reverting was +7.58 BB/100 for the cap and +16.02 for
    # the call gates on the card-aware vs-p3 channel, both resolved -- but
    # the deployed revert busted the live bankroll 1,000 -> 0 in 36 hands
    # over 1.6h, and 2 hands supplied -2,224 of the -2,278 in losses
    # against +1,278 across 24 winning hands.
    #
    # The turn call in the -1,043 hand (As Tc drawing dead against a set
    # of tens) is CALL under the hero-purse denominator and FOLD under the
    # effective-stack one -- verified directly. Note *why* it folds: the
    # opponent was all-in, so `effective_stack_chips` collapsed to 1 (the
    # defect logged in PENDING_EDITS 2026-08-26) and tripped the gate. The
    # protection is real but accidental, and it fires exactly where the
    # most chips are at stake. Fixing that collapse properly is what
    # actually settles this, not the True/False of these flags.
    #
    # Standing caution: the batteries run at 60bb and live play is
    # 500-2,900bb deep, so the measurement never covered the regime that
    # busted.
    risk_cap_on_effective_stack: bool = True
    call_gates_on_effective_stack: bool = True
    # Within the effective-stack branch, count an opponent's ALREADY
    # COMMITTED chips in the denominator (`contested_stack_chips`) rather
    # than only the chips behind them.
    #
    # This is the repair for the collapse logged in `PENDING_EDITS.md`
    # (2026-08-26): `effective_stack_chips` counts chips behind, and an
    # all-in seat stays active with 0, so the denominator falls to 0 ->
    # clamped to 1 -> every call gate trips at any positive price, and
    # `card_reveal_expense` saturates on top. It fires on 31 of 4,333
    # decisions for `candidate-v7-0001c` and 5 of 73 for the reverted
    # build.
    #
    # **Ships False.** On the stored journal the repair is provably
    # loosening-only, and a loosening change to a live safety gate does
    # not go out on a default. Turn it on in `tools.gate_ablation` as the
    # `fix-a` arm, measure it, and only then decide. Deciding on a flag
    # rather than on a measurement is what busted the bankroll on
    # 2026-08-26.
    gate_stack_counts_committed_chips: bool = False
    # Price a call on chips hero can actually WIN, not on `potChips`.
    #
    # An opponent who bets more than hero can match gets the excess back;
    # it never joins a pot hero is contesting. `potChips` counts it
    # anyway, so the price reads low exactly when someone has shoved over
    # hero's stack -- the spots where the most is at risk.
    #
    # Measured live 2026-08-26, table `cmtael1m86iff11453wifw45v`: a 2,958
    # shove into hero's 1,181 purse left 1,777 uncallable, and the pot
    # odds read 0.2823 where the real price was 0.4943. Hero called off
    # the whole stack with Ah 4s at an estimated 0.515 and lost 1,181 --
    # half that day's bust. The `(0.78, 0.626)` gate passed it by 0.00279.
    #
    # **Ships False.** Every threshold in the engine adds pot odds to a
    # required equity, so turning this on can only make prices stricter
    # -- but "only stricter" is still a live-path behaviour change, and
    # those get measured here, not defaulted.
    pot_odds_exclude_uncallable: bool = False
    # Extra equity an ESCALATION demands over a continuation, on the v9
    # line only (`_composed_wager_floor`). The engine sees one "aggress"
    # family for two different acts — the active lane's unprovoked bet
    # and the aggressive lane's raise over a live wager — and the v9
    # contract separates them, so their floors separate too.
    #
    # **Ships 0.0, which reproduces the pre-split floor exactly.** A
    # positive value is a live-path tightening and this project measures
    # those rather than defaulting them; the field exists so the
    # measurement has something to move. The invariant that the
    # escalation floor is never BELOW the continuation floor is
    # structural and holds at any value (see `_composed_wager_floor`).
    escalation_floor_premium: float = 0.0
    # Condition the opponent's range on observed aggression even when
    # hero acts FIRST and there is nothing to call.
    #
    # `_call_top_fraction` returns 1.0 -- literally uniform-random -- the
    # moment `callChips <= 0`, before it looks at aggression at all. So
    # whenever hero opens a street the whole opponent-range model is
    # bypassed. Deterministic on the corpus: 1,369 of 4,795 logged
    # decisions have no price and EVERY one carries width exactly 1.0,
    # zero counterexamples. 940 of them had aggression to condition on;
    # `recentEvents` is cumulative, so the evidence was present and
    # discarded. Being out of position is what triggers it.
    #
    # **Ships False, and this one is not merely caution.** Feature 138
    # (`opponent_range_width`) is 1.0 whenever feature 134
    # (`call_effective_stack_fraction`) is 0 in EVERY training row --
    # 0 of 1,730,110 violate it. Turning this on serves the action-value
    # head a feature joint with zero training support on about a third of
    # decisions, and the incumbent cannot detect it: its
    # `hybrid_min_margin_quantile` is null, so the out-of-distribution
    # branch in `learned_policy._equity_family` never runs. Measure it
    # against a policy that can express the joint before enabling.
    condition_range_without_price: bool = False

    def __post_init__(self) -> None:
        for name in ("board_stackoff_kicker", "board_stackoff_thin"):
            trigger, floor = getattr(self, name)
            object.__setattr__(self, name, (float(trigger), float(floor)))
        object.__setattr__(
            self,
            "call_stack_gates",
            tuple(
                (float(trigger), float(floor))
                for trigger, floor in self.call_stack_gates
            ),
        )
        for name in (
            "board_margin_kicker",
            "board_margin_thin",
            "rescue_call_margin",
            "escalation_floor_premium",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 0.30:
                raise ValueError(f"{name} must be between 0 and 0.3")
        for name in (
            "board_aggression_floor_kicker",
            "board_aggression_floor_thin",
            "near_nut_floor",
            "rescue_call_floor",
        ):
            if not 0.50 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be between 0.5 and 1")
        if not 0.01 <= self.risk_cap_stack_fraction <= 1.0:
            raise ValueError("risk_cap_stack_fraction must be between 0.01 and 1")
        if not 0.0 <= self.reveal_expense_equity_slope <= 0.50:
            raise ValueError("reveal_expense_equity_slope must be between 0 and 0.5")
        for name in (
            "risk_cap_on_effective_stack",
            "call_gates_on_effective_stack",
            "gate_stack_counts_committed_chips",
            "pot_odds_exclude_uncallable",
            "condition_range_without_price",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        gates = (
            self.board_stackoff_kicker,
            self.board_stackoff_thin,
            *self.call_stack_gates,
        )
        for trigger, floor in gates:
            if not 0.01 <= trigger <= 1.0:
                raise ValueError("stack-gate triggers must be between 0.01 and 1")
            if not 0.50 <= floor <= 1.0:
                raise ValueError("stack-gate equity floors must be between 0.5 and 1")

    def board_margin(self, tier: str) -> float:
        return {
            "kicker": self.board_margin_kicker,
            "thin": self.board_margin_thin,
        }.get(tier, 0.0)

    def board_stack_gate(self, tier: str) -> tuple[float, float] | None:
        return {
            "kicker": self.board_stackoff_kicker,
            "thin": self.board_stackoff_thin,
        }.get(tier)

    def board_aggression_floor(self, tier: str) -> float | None:
        return {
            "kicker": self.board_aggression_floor_kicker,
            "thin": self.board_aggression_floor_thin,
        }.get(tier)

    def to_mapping(self) -> dict[str, object]:
        """JSON-ready form for a future learned-artifact manifest."""

        data = asdict(self)
        data["board_stackoff_kicker"] = list(self.board_stackoff_kicker)
        data["board_stackoff_thin"] = list(self.board_stackoff_thin)
        data["call_stack_gates"] = [list(gate) for gate in self.call_stack_gates]
        return data

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "SafetyGates":
        """Rebuild validated gates from artifact JSON; unknown keys fail."""

        data = dict(mapping)
        for name in ("board_stackoff_kicker", "board_stackoff_thin"):
            if name in data:
                trigger, floor = data[name]
                data[name] = (float(trigger), float(floor))
        if "call_stack_gates" in data:
            data["call_stack_gates"] = tuple(
                (float(trigger), float(floor))
                for trigger, floor in data["call_stack_gates"]
            )
        return cls(**data)


DEFAULT_SAFETY_GATES = SafetyGates()
UNSOFTENED_SAFETY_GATES = SafetyGates(
    board_margin_kicker=0.12,
    board_margin_thin=0.10,
    board_stackoff_kicker=(0.18, 0.75),
    board_stackoff_thin=(0.30, 0.80),
    board_aggression_floor_kicker=0.82,
    board_aggression_floor_thin=0.78,
    near_nut_floor=0.72,
    risk_cap_stack_fraction=0.35,
    call_stack_gates=((0.60, 0.68), (0.35, 0.62)),
    rescue_call_floor=0.60,
    rescue_call_margin=0.15,
)


def safest_passive_action(available: Collection[str]) -> str | None:
    """Return the least costly legal fallback, if one exists."""

    return next(
        (action for action in ("check", "fold", "call") if action in available), None
    )


class SharedEquityCache(dict):
    """Equity memo that stays one shared object through ``copy.deepcopy``.

    The counterfactual replay deep-copies the seats -- policies included --
    once per branch and rollout. A plain dict would be copied along with its
    engine, giving every replay a private memo and forfeiting exactly the
    cross-rollout duplication the cache exists for (measured: a per-copy
    dict captured 103 of 628 unique keys on a harvest-shaped leg). Returning
    ``self`` from deepcopy keeps one memo per leg. Safe because entries are
    pure deterministic function values, so sharing them across replays
    cannot leak state between branches.
    """

    def __deepcopy__(self, memo: dict) -> "SharedEquityCache":
        return self


class DecisionEngine:
    """Family proposals plus deterministic Arena safety rails."""

    # Subclasses may declare their own default gate set (the aggressive
    # policy does); an explicit ``safety_gates`` argument always wins.
    default_safety_gates: SafetyGates = DEFAULT_SAFETY_GATES

    # True only on serve classes whose SIZING routes through
    # engine.rules.composition (the v9 composed-value path). The base
    # engine's temperature sizer cannot honour the C2/C3A dials, so the
    # constructor guard below refuses them unless a subclass declares
    # this — declaring it is a claim that the composition IS the sizer.
    serves_composed_sizing: bool = False

    def __init__(
        self,
        *,
        equity_trials: int = 100,
        seed: int = 7,
        temperature_shaping: TemperatureShaping | None = None,
        safety_gates: SafetyGates | None = None,
        opponent_tracker: AggressionTracker | None = None,
        bluff_settings: BluffSettings | None = None,
        hyper_aggression_chance: float | None = None,
        equity_cache: SharedEquityCache | None = None,
        rule_layer: RuleLayerParams | None = None,
    ) -> None:
        if equity_trials < 0:
            raise ValueError("equity_trials cannot be negative")
        self.equity_trials = equity_trials
        # The composed rule-layer dials (engine/rules). Every default is
        # OFF and every consulting site below guards on the dial, so a
        # default-constructed engine is byte-identical to the pre-rules
        # engine — the zero-diff invariant, fuzzed in
        # tests/test_rules_composition.py and held by the full suite.
        self.rule_layer = rule_layer if rule_layer is not None else DEFAULT_RULE_LAYER
        # C2 and C3A size through `engine.rules.composition`, which only the
        # v9 composed-value serve path consults (layer 2). This engine's
        # sizer is the v7/v8 temperature path and CANNOT honour them, while
        # `feature_extract_v9` DOES apply them to the branch-cost features.
        # Enabling them here would therefore teach a corpus sizes this
        # engine never plays — a train/serve divergence, silent in every
        # test because the suite runs dial-off. Refuse instead.
        unsupported = [
            name
            for name, dial in (
                ("geometric (C2)", self.rule_layer.geometric),
                ("snap (C3A)", self.rule_layer.snap),
            )
            if dial.enabled and not self.serves_composed_sizing
        ]
        if unsupported:
            raise ValueError(
                f"{', '.join(unsupported)} cannot be served by this engine's"
                " sizer: they act through engine.rules.composition, which"
                " only the v9 composed-value path consults. Enabling them"
                " here would diverge the features from the played sizes."
            )
        # Opt-in memo for the Monte Carlo equity estimate, keyed on every
        # argument the estimate depends on. estimate_equity is deterministic,
        # so a hit returns the bit-identical float and behaviour cannot
        # change; the switch only decides whether compute is repeated. The
        # default stays None so the live serve path is untouched: measured
        # hit rates are 80.1% inside a counterfactual harvest (branches x
        # rollouts revisit the pinned decision state) and 0.0% on the serve
        # path, so only the harvest passes a cache in.
        self.equity_cache = equity_cache
        self.seed = seed
        self.temperature_shaping = temperature_shaping or DEFAULT_TEMPERATURE_SHAPING
        self.safety_gates = safety_gates or self.default_safety_gates
        self.opponent_tracker = opponent_tracker or AggressionTracker()
        self.bluff_settings = bluff_settings or DEFAULT_BLUFF_SETTINGS
        # The production chance is the module constant; the override exists
        # for deterministic tests and for measuring the noise's price, and
        # is deliberately absent from the learnable artifact parameters.
        if hyper_aggression_chance is None:
            hyper_aggression_chance = HYPER_AGGRESSION_CHANCE
        if not 0.0 <= hyper_aggression_chance <= 1.0:
            raise ValueError("hyper_aggression_chance must be between 0 and 1")
        self.hyper_aggression_chance = hyper_aggression_chance
        self._hyper_bluff_settings = dataclass_replace(
            self.bluff_settings,
            steal_frequency=1.0,
            continuation_frequency=1.0,
            semi_bluff_frequency=1.0,
            barrel_frequency=1.0,
            probe_frequency=1.0,
            raise_bluff_frequency=1.0,
            river_frequency=1.0,
        )
        self._hyper_active = False

    def _family(self, features: tuple[float, ...]) -> str:
        raise NotImplementedError("policy backends must implement _family")

    def _equity(
        self, table: Mapping[str, Any], top_fraction: float = 1.0
    ) -> float | None:
        if self.equity_trials == 0:
            return None
        hero, _ = _hero_and_seats(table)
        hole_cards = _cards(hero.get("holeCards"), "holeCards", 2)
        hole = (hole_cards[0], hole_cards[1])
        board = _cards(table.get("boardCards"), "boardCards")
        opponents = active_opponent_count(table)
        table_id = str(table.get("tableId") or table.get("id") or "")
        digest = hashlib.sha256(f"{self.seed}:{table_id}".encode()).digest()
        trial_seed = int.from_bytes(digest[:8], "big")
        key: tuple | None = None
        if self.equity_cache is not None:
            key = (hole, tuple(board), opponents, trial_seed, top_fraction)
            cached = self.equity_cache.get(key)
            if cached is not None:
                return cached
        value = estimate_equity(
            hole,
            board,
            opponents,
            trials=self.equity_trials,
            seed=trial_seed,
            top_fraction=top_fraction,
        )
        if key is not None:
            self.equity_cache[key] = value
        return value

    def _gate_stack(self, table: Mapping[str, Any], *, effective: bool) -> int:
        """Denominator for a stack-fraction gate.

        A stack-off gate has to be denominated in chips that can actually
        be lost, which is why the live form is the effective stack. The
        hero-purse branch is the pre-2026-08-15 form, kept reachable so
        the change can be measured; it is deliberately unclamped so it
        reproduces the old arithmetic exactly.
        """

        if effective:
            # `gate_stack_counts_committed_chips` selects whether an
            # opponent's already-committed chips count. Off (the shipped
            # default) reproduces the pre-2026-08-27 arithmetic bit for
            # bit, clamp included.
            if self.safety_gates.gate_stack_counts_committed_chips:
                return max(1, contested_stack_chips(table))
            return max(1, effective_stack_chips(table))
        hero, _ = _hero_and_seats(table)
        return _integer(hero.get("stackChips"), "hero stackChips")

    @staticmethod
    def _street(table: Mapping[str, Any]) -> str:
        return str(table.get("street") or "").casefold()

    def _situation_temperature(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float,
    ) -> RiskTemperature:
        hero, _ = _hero_and_seats(table)
        stack = _integer(hero.get("stackChips"), "hero stackChips", minimum=1)
        call_chips = _integer(allowed.get("callChips", 0), "callChips")
        return measure_risk_temperature(
            hand_strength=100.0 * equity,
            purse=stack,
            bet=call_chips,
            street=self._street(table),
            players=1 + active_opponent_count(table),
        )

    def _boldness(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float | None,
    ) -> float:
        """This decision's temperature response; neutral without an equity read."""

        if equity is None:
            return 0.0
        reading = self._situation_temperature(table, allowed, equity)
        return self.temperature_shaping.boldness(reading.temperature)

    def _hyper_roll(self, table: Mapping[str, Any]) -> bool:
        """Salted per-decision dice roll for the anti-modeling noise.

        Deterministic given the snapshot, indistinguishable from chance to
        anyone without the seed -- the same construction as the bluff
        mixer's roll.
        """

        if self.hyper_aggression_chance <= 0.0:
            return False
        key = (
            f"hyper:{self.seed}:{table.get('tableId') or table.get('id') or ''}"
            f":{table.get('handId') or ''}:{self._street(table)}"
            f":{table.get('potChips')}:{table.get('selfSeatNumber')}"
        )
        digest = hashlib.sha256(key.encode()).digest()
        roll = int.from_bytes(digest[:8], "big") / 2.0**64
        return roll < self.hyper_aggression_chance

    def _lead_position(self, table: Mapping[str, Any]) -> float | None:
        """The hero's -100..+100 standing, or None when it cannot be read."""

        hero, seats = _hero_and_seats(table)
        opponents = [seat for seat in _active_seats(seats) if seat is not hero]

        def _owned(seat: Mapping[str, Any]) -> int:
            stack = _integer(seat.get("stackChips"), "stackChips")
            committed = _integer(seat.get("currentBetChips"), "currentBetChips")
            return max(1, stack + committed)

        try:
            reading = measure_lead_position(
                hero_stack=_owned(hero),
                opponent_stacks=tuple(_owned(seat) for seat in opponents),
                position=_position(table, seats, hero),
            )
        except ValueError:
            return None
        return reading.lead

    def _consider_bluff(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float | None,
        lead: float | None,
    ) -> BluffAdvice | None:
        """Ask the bluff advisor when a passive spot could be attacked.

        Engine-side hard gates come first: bluffs need an equity read, a
        legal bet or raise, a board the hero genuinely improves (paired
        boards play for everyone, so representing them has no fold equity),
        and no live raising war. The advisor's own discipline, pricing, and
        mixed-strategy roll decide the rest; sizing is clamped by the same
        legality and risk caps as any aggressive action.
        """

        if equity is None:
            return None
        if "bet" not in available and "raise" not in available:
            return None
        if self._board_tier(table) != "fresh":
            return None
        to_call = _integer(allowed.get("callChips", 0), "callChips")
        street = self._street(table)
        if to_call > 0 and self._aggressive_events(table, hero=True, street=street) > 0:
            return None
        pot = _integer(table.get("potChips"), "potChips")
        if pot < 1:
            return None
        hero, seats = _hero_and_seats(table)
        hole_cards = _cards(hero.get("holeCards"), "holeCards", 2)
        board = _cards(table.get("boardCards"), "boardCards")
        try:
            return evaluate_bluff(
                hole_cards=(hole_cards[0], hole_cards[1]),
                board_cards=board,
                street=street,
                pot=pot,
                to_call=to_call,
                stack=max(1, effective_stack_chips(table)),
                opponents=min(5, max(1, active_opponent_count(table))),
                hero_aggressions=self._aggressive_events(table, hero=True),
                opponent_aggressions=self._aggressive_events(table, hero=False),
                in_position=_position(table, seats, hero) > 0.5,
                lead_position=lead,
                opponent_wildness=self.opponent_tracker.max_active_wildness(table),
                mix_key=str(table.get("tableId") or table.get("id") or ""),
                settings=(
                    self._hyper_bluff_settings
                    if self._hyper_active
                    else self.bluff_settings
                ),
            )
        except ValueError:
            # A malformed bluff situation never blocks the main line.
            return None

    def _learning_features(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        base_features: tuple[float, ...],
        equity: float,
        temperature: RiskTemperature,
    ) -> tuple[float, ...]:
        hero, _ = _hero_and_seats(table)
        contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
        call_chips = _integer(allowed.get("callChips", 0), "callChips")
        aggressive_minimums = []
        for range_name in ("betRange", "raiseRange"):
            amount_range = allowed.get(range_name)
            if amount_range is not None:
                aggressive_minimums.append(
                    _integer(
                        _mapping(amount_range, range_name).get("min"),
                        f"{range_name}.min",
                    )
                )
        if not aggressive_minimums and allowed.get("allInToAmount") is not None:
            aggressive_minimums.append(
                _integer(allowed.get("allInToAmount"), "allInToAmount")
            )
        minimum_to = min(aggressive_minimums, default=contribution)
        effective_stack = max(1, effective_stack_chips(table))
        lead = self._lead_position(table)
        opponent_wildness, opponent_stickiness, _ = (
            self.opponent_tracker.pressure_actor_profile(table)
        )
        return build_learning_features(
            base_features,
            hand_strength=equity,
            board_tier=self._board_tier(table),
            risk_temperature=temperature.temperature / 100.0,
            risk_factors=temperature.factor_risk,
            call_effective_stack_fraction=min(1.0, call_chips / effective_stack),
            min_aggressive_effective_stack_fraction=min(
                1.0, max(0, minimum_to - contribution) / effective_stack
            ),
            hero_aggression_count=self._aggressive_events(table, hero=True),
            opponent_aggression_count=self._aggressive_events(table, hero=False),
            opponent_range_width=self._call_top_fraction(table, allowed),
            opponent_max_wildness=opponent_wildness,
            opponent_max_stickiness=opponent_stickiness,
            lead_position_unit=(0.5 if lead is None else (lead + 100.0) / 200.0),
        )

    @staticmethod
    def _board_tier(table: Mapping[str, Any]) -> str:
        """Board-contribution tier of the hero holding (see board_improvement)."""

        hero, _ = _hero_and_seats(table)
        hole_cards = _cards(hero.get("holeCards"), "holeCards", 2)
        board = _cards(table.get("boardCards"), "boardCards")
        return board_improvement((hole_cards[0], hole_cards[1]), board)

    @staticmethod
    def _hero_seat_number(table: Mapping[str, Any]) -> int:
        return _integer(table.get("selfSeatNumber"), "selfSeatNumber", minimum=1)

    @classmethod
    def _aggressive_events(
        cls, table: Mapping[str, Any], *, hero: bool, street: str | None = None
    ) -> int:
        """Count aggressive actions by hero (or by opponents) in recentEvents."""

        hero_seat = cls._hero_seat_number(table)
        count = 0
        for raw_event in _sequence(table.get("recentEvents") or [], "recentEvents"):
            event = _mapping(raw_event, "recentEvent")
            if (
                street is not None
                and str(event.get("street") or "").casefold() != street
            ):
                continue
            summary_value = event.get("summary")
            if summary_value is None:
                continue
            summary = _mapping(summary_value, "recentEvent.summary")
            action = str(summary.get("action") or "").casefold()
            if action not in _AGGRESSIVE_ACTIONS:
                continue
            seat_number = summary.get("seatNumber")
            is_hero = isinstance(seat_number, int) and seat_number == hero_seat
            if is_hero == hero:
                count += 1
        return count

    def _call_top_fraction(
        self, table: Mapping[str, Any], allowed: Mapping[str, Any]
    ) -> float:
        """How much of the opponent's range to consider when facing a bet.

        The more they have raised this hand, and the bigger the bet in
        front of us, the more their range is weighted toward strong made
        hands, so the smaller the fraction of holdings we simulate against.
        No aggression means no conditioning (1.0 = uniformly random).

        The tracker floor below is escalation-decayed per attacker
        (``opponent_model._ESCALATION_DECAY``, anchored on the 2026-08-13
        bust hand where the flat 0.4237 floor overrode four raises of
        escalation and cost 756 BB); the final [0.20, 1.0] clamp is
        applied after that decayed floor, so 0.20 stays the hard minimum.
        """

        to_call = _integer(allowed.get("callChips", 0), "callChips")
        # With no price there is no bet to size the read against, and the
        # size multiplier below self-neutralises (bet_fraction is 0). The
        # escalation exponent does not need to be discarded with it --
        # see `condition_range_without_price`, which is what decides
        # whether it is.
        if to_call <= 0 and not self.safety_gates.condition_range_without_price:
            return 1.0
        opponent_raises = self._aggressive_events(table, hero=False)
        if opponent_raises == 0:
            return 1.0
        fraction = 0.75 * (0.8 ** (opponent_raises - 1))
        pot = _integer(table.get("potChips"), "potChips")
        bet_fraction = to_call / max(pot - to_call, 1)
        if bet_fraction > 1.0:
            fraction *= 0.6
        elif bet_fraction > 0.6:
            fraction *= 0.8
        # A bet into a hand that barely improves the board is aimed at the
        # board itself: weight the aggressor even further toward hands that
        # beat it.
        fraction *= _BOARD_DISCOUNT_RANGE_TIGHTEN.get(self._board_tier(table), 1.0)
        # A range cannot be narrower than how often its owner plays it fast:
        # the observed aggression frequency floors the conditioning, so a
        # permanent shover converges toward a random range instead of being
        # credited with strength forever. The floor itself decays with the
        # attacker's aggressive-action count this hand, so within-hand
        # escalation is credited even from a tracked maniac.
        fraction = max(fraction, self.opponent_tracker.range_floor_for_table(table))
        return min(1.0, max(0.20, fraction))

    def _call_clears_margin(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float | None,
    ) -> bool:
        """Whether calling is justified: pot odds + street margin + stack gate."""

        if equity is None:
            return True
        tier = self._board_tier(table)
        # Hoisted above the price check for the C4 scaling below; the
        # tracker read is pure, so gate behaviour is unchanged.
        wildness = self.opponent_tracker.max_active_wildness(table)
        margin = self._call_margin(table)
        # Cold situations shave the normal margin, hot ones add to it; the
        # board-discount margin below is a hard gate and never shifts.
        margin -= self.temperature_shaping.call_margin_shift * self._boldness(
            table, allowed, equity
        )
        margin += self.safety_gates.board_margin(tier)
        if self.rule_layer.escalation.enabled:
            # C4 — each wager beyond the first this street demands the
            # measured extra equity, scaled by (1 - wildness): the same
            # signal the gate floors blend on, so tracked maniacs whose
            # raises mean nothing dissolve this margin exactly as they
            # dissolve the floors. The count is the STREET ordinal (hero's
            # own aggression included) because that is the quantity
            # kappa_e was measured against. Dial-off skips the walk.
            escalation = escalation_margin(
                self.rule_layer.escalation,
                street_aggressions(table, self._street(table)),
                wildness,
            )
            if escalation.fired:
                self._record_rule_verdict(escalation)
            margin += escalation.margin_applied
        price = self._pot_odds(table, allowed) + margin
        if equity < price:
            return False
        # Cheap calls into tracked hyper-aggression are shove invitations:
        # the entry bar blends toward the softest stack-gate floor so limps
        # only happen with hands that can stand the pressure.
        if wildness > 0.0:
            gate_floors = [floor for _, floor in self._call_stack_gates()]
            if gate_floors and equity < wildness * min(gate_floors):
                return False
        to_call = _integer(allowed.get("callChips", 0), "callChips")
        # These gates ask "is this call a large share of what I can lose?",
        # so they are denominated in the EFFECTIVE stack, not hero's own
        # purse. Keyed on hero's stack they decay to nothing as the bankroll
        # grows past the table: measured live 2026-08-15, a 2,207-chip call
        # against a 2,207 effective stack read as 24% of a 9,143 purse and
        # tripped no gate, and the hand lost 3,768. Same mis-scoping as the
        # sizing risk cap, in the path that governs the larger losses --
        # every one of the biggest live losses is a call, and calls are
        # otherwise ungated. `_gate_stack` carries the ablation dial.
        stack = self._gate_stack(
            table, effective=self.safety_gates.call_gates_on_effective_stack
        )
        # Chips staked before a card is turned are staked on unseen cards.
        reveal_penalty = (
            self.safety_gates.reveal_expense_equity_slope
            * card_reveal_expense(table, to_call)
        )
        # The temperature's defensive margin also assumes bets mean
        # strength, so the wild-opponent blend targets the plain price.
        neutral_price = (
            self._pot_odds(table, allowed)
            + self._call_margin(table)
            + self.safety_gates.board_margin(tier)
        )
        commitment = None
        if self.rule_layer.commitment.enabled:
            # C1 — a call whose post-call SPR is at or under 1 is a
            # stack-off in installments (a next-street shove prices at
            # 1/3), so it is judged by the strictest existing gate: same
            # floor, same reveal penalty, same wildness slide. A new
            # trigger, never a new floor.
            commitment = forward_commitment(
                self.rule_layer.commitment,
                gate_stack=stack,
                to_call=to_call,
                pot=_integer(table.get("potChips"), "potChips"),
            )
            # Recorded only when a gate exists for it to trigger. C1 adds
            # a trigger to the strictest existing gate; with
            # call_stack_gates empty (a legal configuration) the loop
            # below never runs, so a "fired" verdict would journal an
            # enforcement that did not happen.
            if commitment.fired and self._call_stack_gates():
                self._record_rule_verdict(commitment)
        for index, (stack_fraction, equity_floor) in enumerate(
            self._call_stack_gates()
        ):
            equity_floor += reveal_penalty
            if to_call >= stack_fraction * stack or (
                index == 0 and commitment is not None and commitment.fired
            ):
                # The gate's premise is that big bets mean strength. Tracked
                # wildness dissolves that premise proportionally, sliding the
                # requirement from the gate floor back to the plain price.
                required = (1.0 - wildness) * equity_floor + wildness * neutral_price
                if equity < required:
                    return False
        gate = self.safety_gates.board_stack_gate(tier)
        if gate is not None:
            stack_fraction, floor = gate
            if to_call >= stack_fraction * stack and equity < floor + reveal_penalty:
                return False
        return True

    def _call_margin(self, table: Mapping[str, Any]) -> float:
        return _CALL_MARGINS.get(self._street(table), 0.08)

    def _call_stack_gates(self) -> tuple[tuple[float, float], ...]:
        return self.safety_gates.call_stack_gates

    def _aggression_floor(self, table: Mapping[str, Any], opponent_count: int) -> float:
        floor = min(0.72, 0.52 + 0.05 * max(0, opponent_count - 1))
        if self._street(table) == "preflop":
            floor += 0.04
        return floor

    # Set transiently by decide_forced; consulted once where the family
    # proposal lands so a counterfactual branch runs the identical serve
    # path -- bluff conversion, engine sizing, and every safety clamp.
    _forced_proposal: tuple[str, float | None] | None = None

    # Per-decision accumulator for engine/rules verdicts that FIRED.
    # Reset by decide_with_diagnostics; None outside a decision, so a
    # direct unit call of a gated method records nothing. A plain list on
    # purpose: counterfactual probes deepcopy the policy per branch and
    # per-branch isolation is the correct behavior (never the return-self
    # SharedEquityCache pattern).
    _rule_verdicts: list[dict[str, object]] | None = None

    def _record_rule_verdict(self, verdict: Any) -> None:
        if self._rule_verdicts is None:
            return
        mapping = verdict.as_mapping()
        # The call ladder runs more than once per decision (family ladder,
        # rescue, passive fallback); identical re-firings are one fact.
        if mapping not in self._rule_verdicts:
            self._rule_verdicts.append(mapping)

    def _take_rule_verdicts(self) -> tuple[dict[str, object], ...] | None:
        """Drain the accumulator, restoring the "closed outside a decision"
        state its contract promises: leaving the list in place would let a
        later direct call of a gated method append to a finished
        decision's record."""

        collected = self._rule_verdicts
        self._rule_verdicts = None
        return tuple(collected) if collected else None

    def decide(
        self,
        table: Mapping[str, Any],
        deadline_s: float = 10.0,
        research_context: Mapping[str, Any] | None = None,
    ) -> dict[str, str | int]:
        return self.decide_with_diagnostics(
            table,
            deadline_s=deadline_s,
            research_context=research_context,
        ).to_payload()

    def decide_forced(
        self,
        table: Mapping[str, Any],
        *,
        family: str,
        pot_fraction: float | None = None,
        deadline_s: float = 10.0,
    ) -> dict[str, str | int]:
        """Decide with the family proposal pinned to ``family``.

        Counterfactual value branches call this so Q(state, branch)
        measures the action this policy would actually submit: a pinned
        passive family may still be upgraded by the bluff module exactly
        as at serve, and a pinned aggressive family is sized by the engine
        at ``pot_fraction`` under the normal clamps. The pin applies to
        one decision and is always cleared. An unknown family raises.

        **The vocabulary is the FROZEN observable trio.** L5 briefly
        grew distinct literal ``check`` / ``call`` families here and the
        adversarial sweep reverted them: nothing called them (both
        hand-off points — ``ContractForcingRecorder`` and the simulator's
        ``_payload_for_family`` — *reject* those names), and if reached
        they escaped ``policy_features.LABELS``, producing an all-zero
        ``behavior_probabilities`` one-hot that
        ``training_telemetry`` refuses and ``LABELS.index`` raises on.
        The v9 literal-forcing doctrine lives where it is measured, in
        ``tools.build_phase_b_corpus_v9.ContractForcingRecorder``, which
        emits literal payloads without entering this family channel at
        all. Widening the trio is an L6 item (``proposed_branch`` as an
        additive field), not this method's.
        """

        self._forced_proposal = (str(family), pot_fraction)
        try:
            return self.decide(table, deadline_s=deadline_s)
        finally:
            self._forced_proposal = None

    @staticmethod
    def _family_available(family: str, available: set[str]) -> bool:
        """Whether a forced-family pin can execute in this state.

        The vocabulary is the FROZEN observable trio and nothing else —
        widening it here was tried at L5 and reverted (see
        ``decide_forced``). An unknown name RAISES instead of defaulting
        into the aggress arm: that default was one of the two
        silent-corruption channels for version skew, because a v9 BRANCH
        name (``"aggressive"``) handed to the family channel read as
        aggress-availability here and then fell through to the fold
        default below, silently. Branch-to-family projection belongs to
        ``branch_contract_v9.branch_engine_family``, never to this
        method.
        """

        if family == "fold":
            return "fold" in available
        if family == "check_call":
            return "check" in available or "call" in available
        if family == "aggress":
            return "bet" in available or "raise" in available
        raise ArenaSnapshotError(f"unknown forced family {family!r}")

    def decide_with_diagnostics(
        self,
        table: Mapping[str, Any],
        deadline_s: float = 10.0,
        research_context: Mapping[str, Any] | None = None,
    ) -> DecisionResult:
        context = research_context or {}
        cached_position = context.get("position")
        if cached_position is not None:
            try:
                cached_position = float(cached_position)
            except (TypeError, ValueError) as exc:
                raise ArenaSnapshotError(
                    "research_context.position must be a number"
                ) from exc
        features = features_from_table(table, position=cached_position)
        allowed = _mapping(table.get("allowedActions"), "allowedActions")
        available = {
            str(value)
            for value in _sequence(allowed.get("availableActions"), "availableActions")
        }
        # Record this hand's observed actions before deciding, so opponent
        # aggression frequencies stay current even on deadline fallbacks.
        self.opponent_tracker.observe(table)
        self._hyper_active = self._hyper_roll(table)
        self._rule_verdicts = []
        if deadline_s < 2.0:
            action = self._deadline_action(table, allowed, available)
            # The deadline path reaches _sized_action with equity=None,
            # so boldness is 0 and the damped and undamped fractions are
            # identical — C5 cannot journal here today. The carry-out is
            # kept anyway so a future rule that CAN fire on this path is
            # recorded rather than silently dropped; it is None in
            # practice.
            deadline_verdicts = self._take_rule_verdicts()
            return DecisionResult(
                action=self._render(action, table, allowed, equity=None),
                family="deadline",
                equity=None,
                situation_temperature=None,
                deadline_fallback=True,
                rule_verdicts=deadline_verdicts,
            )

        # Facing aggression, estimate equity against the strong part of the
        # opponent's range instead of a uniformly random hand.
        top_fraction = self._call_top_fraction(table, allowed)
        equity = self._equity(table, top_fraction=top_fraction)
        temperature = (
            self._situation_temperature(table, allowed, equity)
            if equity is not None
            else None
        )
        learning_features = (
            self._learning_features(table, allowed, features, equity, temperature)
            if equity is not None and temperature is not None
            else None
        )
        family = (
            self._equity_family(table, allowed, available, equity, features=features)
            if equity is not None
            else self._family(features)
        )
        forced = self._forced_proposal
        forced_pot_fraction: float | None = None
        if forced is not None and self._family_available(forced[0], available):
            family, forced_pot_fraction = forced
        if family == "aggress":
            action = None
            if forced_pot_fraction is not None:
                for name in ("raise", "bet"):
                    if name in available:
                        action = self._sized_action(
                            name,
                            table,
                            allowed,
                            equity,
                            pot_fraction=forced_pot_fraction,
                        )
                        if action is not None:
                            break
            if action is None:
                action = self._aggressive_action(table, allowed, available, equity)
        elif family == "check_call":
            action = self._passive_action(table, allowed, available, equity)
        elif family == "fold":
            action = self._fold_action(table, allowed, available, equity)
        else:
            # The second silent-corruption channel for version skew,
            # hardened (L5): anything unrecognised used to fall into the
            # fold arm. The known families are exhaustive above.
            raise ArenaSnapshotError(f"unknown family proposal {family!r}")

        # A passive spot may still be attacked as a priced, mixed bluff. The
        # submitted family becomes aggress so telemetry labels match the
        # action; the bluff kind is recorded as private diagnostics.
        lead = self._lead_position(table)
        bluff_kind: str | None = None
        if family != "aggress":
            advice = self._consider_bluff(table, allowed, available, equity, lead)
            if advice is not None and advice.bluff and advice.action in available:
                sized = self._sized_action(
                    advice.action,
                    table,
                    allowed,
                    equity,
                    pot_fraction=advice.pot_fraction,
                )
                if sized is not None:
                    action = sized
                    family = "aggress"
                    bluff_kind = advice.kind
        return DecisionResult(
            action=self._render(action, table, allowed, equity),
            family=family,
            equity=equity,
            situation_temperature=temperature,
            learning_features=learning_features,
            behavior_probabilities=tuple(float(label == family) for label in LABELS),
            proposed_risk_fraction=getattr(self, "_proposed_risk_fraction", None),
            temperature_boldness=(
                self.temperature_shaping.boldness(temperature.temperature)
                if temperature is not None
                else None
            ),
            opponent_range_width=top_fraction if equity is not None else None,
            opponent_evidence_confidence=(
                self.opponent_tracker.pressure_actor_profile(table)[2]
            ),
            rule_verdicts=self._take_rule_verdicts(),
            lead_position=lead,
            bluff_kind=bluff_kind,
            hyper_aggression=self._hyper_active,
        )

    def _equity_family(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float,
        features: tuple[float, ...] | None = None,
    ) -> str:
        del features  # Equity thresholds decide here; model backends may use them.
        opponent_count = max(1, active_opponent_count(table))
        aggression_floor = self._aggression_floor(table, opponent_count)
        # Temperature shaping moves the policy-owned floor a bounded amount:
        # cold lowers the bar to aggress, hot raises it.
        aggression_floor -= (
            self.temperature_shaping.aggression_floor_shift
            * self._boldness(table, allowed, equity)
        )
        # Against tracked hyper-aggression a raise buys no folds, so it is
        # dominated except for pure value: the floor blends continuously
        # toward the war floor and marginal hands defend by calling
        # instead. Simulation found the open-then-fold leak this closes
        # (-134 bb/100 vs a permanent shover before, near breakeven after;
        # a looser shove-price entry floor re-tested worse at -15 bb/100).
        wildness = self.opponent_tracker.max_active_wildness(table)
        if wildness > 0.0:
            aggression_floor = (
                1.0 - wildness
            ) * aggression_floor + wildness * self.safety_gates.near_nut_floor
        if self._hyper_active:
            aggression_floor -= _HYPER_FLOOR_DROP
        aggression_floor = min(0.95, max(0.05, aggression_floor))
        if (
            any(action in available for action in ("bet", "raise"))
            and equity >= aggression_floor
        ):
            return "aggress"
        if "check" in available:
            return "check_call"
        if "call" in available and self._call_clears_margin(table, allowed, equity):
            return "check_call"
        return "fold"

    def _fold_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float | None,
    ) -> tuple[str, int | None]:
        if "check" in available:
            return "check", None
        pot_odds = self._pot_odds(table, allowed)
        if (
            "call" in available
            and equity is not None
            and equity
            >= max(
                self.safety_gates.rescue_call_floor,
                pot_odds + self.safety_gates.rescue_call_margin,
            )
            and self._call_clears_margin(table, allowed, equity)
        ):
            return "call", None
        if "fold" in available:
            return "fold", None
        return self._passive_action(table, allowed, available, equity)

    def _passive_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float | None,
    ) -> tuple[str, int | None]:
        if "check" in available:
            return "check", None
        if "call" in available and self._call_clears_margin(table, allowed, equity):
            return "call", None
        if "fold" in available:
            return "fold", None
        if "call" in available:
            return "call", None
        return self._first_legal_aggression(table, allowed, available)

    def _aggressive_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float | None,
    ) -> tuple[str, int | None]:
        hero, _ = _hero_and_seats(table)
        safety_floor = 0.0
        street = self._street(table)
        to_call = _integer(allowed.get("callChips", 0), "callChips")
        if to_call > 0 and self._aggressive_events(table, hero=True, street=street) > 0:
            # We already raised this street and got raised back: continuing
            # the war needs a near-nut hand, not a marginal edge vs random.
            safety_floor = self.safety_gates.near_nut_floor
        if equity is not None:
            # Betting the board's own hand only ever levers out worse board
            # play; hands that beat it never fold. Near-nuts may still bet.
            tier_floor = self.safety_gates.board_aggression_floor(
                self._board_tier(table)
            )
            if tier_floor is not None:
                safety_floor = max(safety_floor, tier_floor)
        if self.serves_composed_sizing and equity is not None:
            safety_floor = self._composed_wager_floor(table, safety_floor, to_call)
        if equity is not None and equity < safety_floor:
            return self._passive_action(table, allowed, available, equity)

        if "bet" in available:
            sized = self._sized_action("bet", table, allowed, equity)
            if sized is not None:
                return sized
        if "raise" in available:
            sized = self._sized_action("raise", table, allowed, equity)
            if sized is not None:
                return sized
        if self.serves_composed_sizing:
            # v9 flows only (L5): a CHOSEN escalation whose bet/raise
            # sizing produced nothing reaches the gated shove lane. The
            # v7/v8 lines never enter this block, so their behaviour
            # stays byte-identical.
            shove = self._gated_shove(table, allowed, available, equity)
            if shove is not None:
                return shove
        # The demotion target for an unsizeable escalation IS this call —
        # `_passive_action` plays the active lane at the current price
        # when the rails allow it and folds when they do not.
        #
        # An earlier L5 draft returned a LITERAL ("call", None) here on
        # v9 flows, reasoning that the composed layer had already ranked
        # aggression above the active call. The sweep measured what that
        # cost: `_sized_action` returns None precisely because the risk
        # cap emptied the raise range, which only happens at SUB-NEAR-NUT
        # equity, so the literal call answered the cap's refusal by
        # calling off anyway — 19,000 of a 20,000 stack at 0.05 equity
        # where this line folds. That is the 2026-08-26 bust class, and
        # the loosening shipped on a default, which `SafetyGates` says
        # never happens. The "never a silent fold" clause in the plan is
        # a HARVEST-labelling requirement, and the purity check drops
        # aggressive->call and aggressive->fold identically, so the
        # literal call bought the corpus nothing either.
        return self._passive_action(table, allowed, available, equity)

    def _composed_wager_floor(
        self, table: Mapping[str, Any], base_floor: float, to_call: int
    ) -> float:
        """Split the chosen-wager floor across the v9 contract's two lanes.

        **v9 flows only** (`serves_composed_sizing`); the legacy lines
        never reach here and keep `base_floor` untouched.

        The engine sees a single ``aggress`` family for two different
        acts: the ACTIVE lane's unprovoked bet (``to_call == 0``) and
        the AGGRESSIVE lane's escalation over a live wager
        (``to_call > 0``). The v9 contract separates those, so their
        floors separate too, and the escalation's floor is never BELOW
        the continuation's — opening a pot may not be harder than
        raising one. That invariant is enforced AFTER the clamp, so it
        holds at every parameter value, and
        ``escalation_floor_premium`` ships 0.0 so this reproduces the
        pre-split floor exactly.

        It also RESTORES the tracked-wildness blend that the v9
        projection silently dropped. The blend lives in
        ``_equity_family``, which ``learned_policy_v9`` overrides
        wholesale for its composed argmax — so on the v9 line a tracked
        maniac had stopped raising hero's bar to aggress at all. The
        blend's rationale is unchanged and is the v9 demotion rule
        stated plainly: against an opponent who does not fold, a raise
        buys no folds and is dominated except for pure value, so
        marginal hands should defend by CALLING instead. The demotion
        that produces is ``_passive_action``'s gated call — never a bare
        one, which is the defect this layer already had to fix once.
        """

        floor = base_floor
        wildness = self.opponent_tracker.max_active_wildness(table)
        if wildness > 0.0:
            floor = (
                1.0 - wildness
            ) * floor + wildness * self.safety_gates.near_nut_floor
        active_floor = min(0.95, max(0.05, floor))
        if to_call <= 0:
            return active_floor
        escalation_floor = min(
            0.95,
            max(0.05, floor + self.safety_gates.escalation_floor_premium),
        )
        return max(escalation_floor, active_floor)

    def _gated_shove(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float | None,
    ) -> tuple[str, int] | None:
        """The L5 shove lane — v9 flows only, and never an open door.

        Reachable ONLY from ``_aggressive_action``'s fallthrough (a
        chosen escalation with no sizeable bet/raise), and it fires on
        ONE condition: near-nut equity, the same release the risk cap
        itself uses. An ungated chosen shove reopens the 2026-08-26 bust
        class. The risk cap never moves, and ``_first_legal_aggression``
        stays the ungated last resort on the failure paths, never a
        chosen one.

        **The plan's second arm — "a clamped max already committing the
        effective stack" — is DEFERRED, not implemented**, because the
        sweep measured the obvious reading of it as a money hole. With
        shipped gates ``_gate_stack`` returns the same quantity the arm
        compared against, so ``risk_cap_to >= contribution + effective``
        reduced algebraically to ``effective_stack <= big_blind`` — no
        equity term at all, firing at 0.05 equity as readily as 0.95.
        Worse, ``effective_stack_chips`` counts chips BEHIND and
        collapses to 0 the moment every live opponent is all-in (the
        defect already filed in PENDING_EDITS), so the arm fired on
        exactly the states where the most chips are contested; and its
        "the Arena refunds the excess, so the shove costs what a call
        costs" defence is heads-up-only — measured multiway with two
        all-ins (1,200 and 800), a 6,000 shove's matched risk is 2,000
        against a 1,200 call price. Re-landing it needs a denominator
        that does not collapse (``contested_stack_chips``), a multiway
        risk definition, and the call ladder's own verdict — its own
        change, with its own measurement.
        """

        if "all-in" not in available or equity is None:
            return None
        if equity < self.safety_gates.near_nut_floor:
            return None
        # A malformed snapshot is not a policy choice: raise exactly as
        # `_first_legal_aggression` does on the identical field, rather
        # than degrading a near-nut shove into whatever follows.
        all_in_to = _integer(
            allowed.get("allInToAmount"), "allInToAmount", minimum=1
        )
        return "all-in", all_in_to

    def _sized_action(
        self,
        action: str,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float | None,
        pot_fraction: float | None = None,
    ) -> tuple[str, int] | None:
        hero, _ = _hero_and_seats(table)
        pot = _integer(table.get("potChips"), "potChips")
        big_blind = _integer(table.get("bigBlindChips"), "bigBlindChips", minimum=1)
        contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
        call_chips = _integer(allowed.get("callChips", 0), "callChips")

        range_name = "betRange" if action == "bet" else "raiseRange"
        amount_range = _mapping(allowed.get(range_name), range_name)
        minimum = _integer(amount_range.get("min"), f"{range_name}.min")
        maximum = _integer(amount_range.get("max"), f"{range_name}.max")
        call_to_amount = allowed.get("callToAmount")
        if action == "bet" or (call_to_amount is None and call_chips == 0):
            # Arena nulls callToAmount whenever nothing is left to call, so
            # a raise from a check spot starts at hero's already-matched
            # street commitment, exactly like a bet.
            base = contribution
        else:
            base = _integer(call_to_amount, "callToAmount")
        # The half-pot target breathes with the temperature: bigger when
        # cold, smaller when hot. A caller (the bluff path) may bring its
        # own pot fraction instead, and a hyper-aggression decision targets
        # the full pot. The risk cap below never shifts.
        pending_damper = None
        undamped_fraction: float | None = None
        if pot_fraction is None:
            if self._hyper_active:
                pot_fraction = _HYPER_POT_FRACTION
            else:
                boldness = self._boldness(table, allowed, equity)
                pot_fraction = 0.5 * (
                    1.0 + self.temperature_shaping.sizing_span * boldness
                )
                if self.rule_layer.damper.enabled:
                    # C5 — the ruin damper cools the read before sizing.
                    #
                    # The emitted fraction is min(damped, undamped), NOT
                    # simply the damped one. This arm is monotone
                    # INCREASING in boldness, so scaling b toward zero
                    # RAISES the fraction whenever b < 0 (a hot read):
                    # measured b=-1, d=0.1 moves 0.3050 -> 0.4805. An
                    # earlier comment here claimed the arm was monotone in
                    # the safe direction and skipped the min — it grew
                    # bets in exactly the hot states the damper exists to
                    # cool. Same construction, same reason, as the rules
                    # composition. The hyper branch above deliberately
                    # bypasses it: the anti-modeling floor is not a tuning
                    # surface.
                    damper_verdict = damping(
                        self.rule_layer.damper,
                        bankroll=_integer(
                            hero.get("stackChips"), "hero stackChips", minimum=1
                        ),
                        exposure=table_exposure(table),
                    )
                    damped_fraction = 0.5 * (
                        1.0
                        + self.temperature_shaping.sizing_span
                        * damped_boldness(boldness, damper_verdict)
                    )
                    # Deferred: the verdict is journaled at the END of this
                    # method, and only if the EMITTED AMOUNT differs. A
                    # fraction comparison is not enough — the big-blind
                    # floor, the integer round and the legal clamp all
                    # absorb small fractional differences, and this method
                    # can still bail with None afterwards, which would
                    # journal an effect for a proposal the engine
                    # abandons. Comparing fractions was the first attempt
                    # and left both holes open.
                    pending_damper = damper_verdict if damper_verdict.fired else None
                    undamped_fraction = pot_fraction
                    pot_fraction = min(pot_fraction, damped_fraction)
        desired = base + max(big_blind, round(pot_fraction * (pot + call_chips)))

        if equity is None or equity < self.safety_gates.near_nut_floor:
            # The cap keys on the EFFECTIVE stack, not hero's own purse.
            #
            # A stack-off gate has to be denominated in chips that can
            # actually be lost. Hero's own stack is the wrong unit: hero
            # brings the whole bankroll to a table of short opponents, so as
            # the bankroll grows a fixed fraction of it stops bounding
            # anything. Measured live on 2026-08-15, the hero-stack form
            # could bind on 58.6% of sub-near-nut sizing decisions at a
            # ~2.6k purse, 30.4% at ~8.7k and 4.3% at ~12k -- the gate
            # decayed to inert exactly as the money it guards grew. In the
            # worst observed case an 11,842 stack bought a 5,388 cap against
            # a 1,133 effective stack, and a flop raise put 87.9% of that
            # effective stack in on 10% equity.
            #
            # min(hero stack, effective stack) is the same number:
            # effective_stack_chips already clamps to hero's purse. So when
            # hero IS the short stack the denominator is hero's stack and
            # the bankroll protection is exactly what it always was; the two
            # definitions only diverge when hero covers the table, which is
            # precisely where the old form failed.
            #
            # Multiway, "effective" is ambiguous. effective_stack_chips takes
            # the DEEPEST active opponent (bounded by hero) -- the most any
            # single opponent can make hero pay -- rather than the shallowest.
            # The shallowest would shrink a legitimate bet whenever one short
            # stack is in the pot, and it is not the bound on hero's risk;
            # hero's commitment is capped directly by this number either way.
            # It is also the definition the learned features, the bluff
            # advisor and the telemetry already use, so the gate and the model
            # agree on what "effective stack" means.
            gate_stack = self._gate_stack(
                table, effective=self.safety_gates.risk_cap_on_effective_stack
            )
            risk_cap = contribution + max(
                big_blind,
                round(self.safety_gates.risk_cap_stack_fraction * gate_stack),
            )
            maximum = min(maximum, risk_cap)
        if maximum < minimum:
            # Abandoned proposal: journal nothing. Recording before this
            # bail attributed a cooling to a wager the engine never made
            # (measured: 1,521 of 3,000 probed states).
            return None
        sized = min(max(desired, minimum), maximum)
        if pending_damper is not None:
            # Journal the damper only if it changed the AMOUNT. Equal
            # amounts mean the legalization absorbed the cooled fraction,
            # and "sizes cooled" would assert an effect that did not reach
            # the table.
            undamped_desired = base + max(
                big_blind, round(undamped_fraction * (pot + call_chips))
            )
            if sized != min(max(undamped_desired, minimum), maximum):
                self._record_rule_verdict(pending_damper)
        return action, sized

    def _first_legal_aggression(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
    ) -> tuple[str, int | None]:
        for action in ("bet", "raise"):
            if action in available:
                sized = self._sized_action(action, table, allowed, equity=None)
                if sized is not None:
                    return sized
        if "all-in" in available:
            return "all-in", _integer(allowed.get("allInToAmount"), "allInToAmount")
        raise ArenaSnapshotError("no legal fallback action")

    def _deadline_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
    ) -> tuple[str, int | None]:
        passive = safest_passive_action(available)
        if passive is not None:
            return passive, None
        return self._first_legal_aggression(table, allowed, available)

    @staticmethod
    def _uncallable_chips(table: Mapping[str, Any], call_chips: int) -> int:
        """Chips in the pot hero cannot win even by calling all-in.

        An opponent who bets more than hero can match gets the excess
        back: it never joins any pot hero is contesting. Public
        information only -- bets and stacks -- so this is equally usable
        as a gate input and as a learned feature.

        Anything unreadable returns 0, which reproduces the old
        arithmetic exactly, so this can only raise the measured price and
        only on snapshots where the overhang is legible.
        """

        try:
            hero, seats = _hero_and_seats(table)
            hero_cap = (
                _integer(hero.get("currentBetChips"), "hero currentBetChips")
                + call_chips
            )
            return sum(
                max(
                    0,
                    _integer(seat.get("currentBetChips"), "currentBetChips")
                    - hero_cap,
                )
                for seat in seats
                if seat is not hero
            )
        except (ArenaSnapshotError, TypeError, ValueError):
            return 0

    def _pot_odds(
        self, table: Mapping[str, Any], allowed: Mapping[str, Any]
    ) -> float:
        """The price of a call, as a share of what calling can win.

        Was a ``@staticmethod``; it reads ``safety_gates`` now so the
        uncallable-overhang correction is ablatable. Every call site
        already went through ``self``.
        """

        pot = _integer(table.get("potChips"), "potChips")
        call_chips = _integer(allowed.get("callChips", 0), "callChips")
        if call_chips == 0:
            return 0.0
        winnable = pot + call_chips
        if self.safety_gates.pot_odds_exclude_uncallable:
            winnable -= self._uncallable_chips(table, call_chips)
        return call_chips / max(winnable, 1)

    def _render(
        self,
        action: tuple[str, int | None],
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float | None,
    ) -> ArenaAction:
        del equity  # Keep private-card estimates out of public table chat.
        action_name, amount = action
        templates = {
            "fold": (
                "the price and line make this a clean release",
                "this branch is too expensive to continue",
            ),
            "check": (
                "keeping the pot controlled on this texture",
                "taking the free card and preserving flexibility",
            ),
            "call": (
                "the price leaves enough room to continue",
                "continuing without inflating the pot",
            ),
            "bet": (
                "a measured size pressures the weaker range",
                "using a controlled size to deny cheap realization",
            ),
            "raise": (
                "applying pressure while keeping stack risk bounded",
                "this line supports a measured pressure raise",
            ),
            "all-in": (
                "stack geometry makes full commitment cleaner than a partial size",
                "the remaining stack works better as one decision",
            ),
        }
        table_id = str(table.get("tableId") or table.get("id") or "")
        digest = hashlib.sha256(f"{table_id}:{action_name}".encode()).digest()
        choices = templates[action_name]
        message = choices[digest[0] % len(choices)]

        reasoning: str | None = None
        if bool(allowed.get("reasoningRequired")):
            pot_odds = round(100 * self._pot_odds(table, allowed))
            fields = [f'ke: "pot odds {pot_odds}%"', 'pp: "risk-controlled line"']
            if action_name in _AGGRESSIVE_ACTIONS:
                fields.append('sr: "bounded pot pressure"')
            reasoning = "{" + ", ".join(fields) + "}"
        return ArenaAction(action_name, amount, message, reasoning)


__all__ = [
    "ArenaAction",
    "DecisionEngine",
    "DecisionResult",
    "DEFAULT_SAFETY_GATES",
    "DEFAULT_TEMPERATURE_SHAPING",
    "NEUTRAL_TEMPERATURE_SHAPING",
    "safest_passive_action",
    "SafetyGates",
    "SharedEquityCache",
    "TemperatureShaping",
    "UNSOFTENED_SAFETY_GATES",
]

# Re-exported for policy construction convenience.
__all__ += ["BluffSettings", "DEFAULT_BLUFF_SETTINGS"]
