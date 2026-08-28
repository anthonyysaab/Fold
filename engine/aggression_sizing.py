"""The v9 sizing function g — one definition, every consumer imports it.

Spec: `.handoff/notes/V9_RESTRUCTURE_PLAN.md`, "The sizing function g"
(accepted 2026-08-28, amended the same day by the six-skeptic adversarial
panel). The v9 contract is `branch_contract_v9.py`; sizes here belong to
its two wager lanes only — ``active`` as an unprovoked bet, ``aggressive``
as an escalation. A second implementation of any formula in this module,
anywhere, is the silent-drift class the Phase-B E6 cross-check existed to
prevent: extractor, composer, trainers, harvester, engine sizer, and
trivial floors must all import from here.

What g deliberately is NOT:

- **Not a legalizer.** g returns unclamped chip targets; ``_sized_action``
  is the sole legalizing authority (big-blind floor, legal range, risk
  cap). Submitting a raw g target is an illegal amount in exactly the
  effective-stack-collapse states.
- **Not the v7 sizer.** The v7 line keeps ``TemperatureShaping.sizing_span``
  and the shared ``_situation_temperature`` untouched. g owns its own
  parameters and its own temperature read.
- **Not consulted on the fatal or passive branches**, and it REFUSES the
  aggressive lane at ``to_call == 0`` — under the v9 contract an
  unprovoked wager is active's, never aggressive's.

The depth-invariant read (the panel's confirmed blocker): the engine's
temperature purse is hero's RAW stack, so at live depth (median 2,875bb)
bet pressure is dead and any parameters priced on the 60bb instrument
would not reproduce live. g therefore reads the same five-factor
temperature through a table-scoped purse::

    purse_g = min(hero stackChips, max(1, callChips, contested_stack_chips))
    bet     = callChips                      # <= purse_g by construction

``contested_stack_chips`` carries the invariant that it is at least
``callChips`` whenever a price exists, and the Arena caps ``callChips`` at
hero's stack (verified on all 4,810 stored rows), so the read satisfies
``measure_risk_temperature``'s ``bet <= purse`` guard structurally. The
active-bet lane is unaffected either way — its bet pressure is
identically zero at any depth.

Parameters are three ADDITIVE ``(base, span)`` pairs::

    fraction = base + span * b,   b = clamp((setpoint - T) / span_T, -1, 1)

The active span is 0.195, NOT the v7 knob's 0.39 — that knob is relative
(``0.5 * (1 + 0.39 * b)``) and the additive equivalent halves it; copying
0.39 would double the active swing. The aggressive endpoints reproduce
the retired v8 branch specs (0.50, 0.20) and (1.00, 0.45) at b = -1/+1.
A neutral arm must zero ALL THREE spans; zeroing only the active one
leaves the aggressive lane temperature-modulated and contaminates the
arm. Parameters ship in a g-owned manifest block (``sizing_record``),
never inside ``TemperatureShaping`` (its ``from_mapping`` fails on
unknown keys and ~20 shipped manifests pin its five fields).

Parity: v9 moves sizes inside the value composition, so the Phase-B
parity replay needs the read per decision. The raw temperature travels in
corpus rows as the integer ``10 * T`` (``read_to_context_int``); T has
exactly one decimal, and ``int / 10.0`` is the correctly-rounded double
of that decimal, so the round-trip is bit-exact. The z-normalized feature
copy of T is NOT a safe source. Only Python ever computes
``round(x, 1)`` — banker's rounding, which torch cannot reproduce.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from risk_temperature import RiskTemperature, measure_risk_temperature

from engine.game_state import (
    _hero_and_seats,
    _integer,
    active_opponent_count,
    contested_stack_chips,
)

#: Recorded in every v9 architecture block, corpus header, and sidecar.
#: Bump ONLY with a new identity string — never change formulas under an
#: existing id; stored corpora re-derive their sizes against this name.
G_IDENTITY = "g-v9-linear-boldness-1"


@dataclass(frozen=True, slots=True)
class SizingParameters:
    """The g parameter block — three additive (base, span) pairs.

    ``setpoint``/``temperature_span`` are the boldness map's constants,
    carried here so a recorded parameter block reproduces b without
    consulting ``TemperatureShaping`` (whose fields belong to the frozen
    v7 line).
    """

    active_base: float = 0.5
    active_span: float = 0.195
    aggressive_base: float = 0.75
    aggressive_span: float = 0.25
    cap_base: float = 0.325
    cap_span: float = 0.125
    setpoint: float = 45.0
    temperature_span: float = 35.0

    def __post_init__(self) -> None:
        for name in (
            "active_base",
            "active_span",
            "aggressive_base",
            "aggressive_span",
            "cap_base",
            "cap_span",
            "setpoint",
            "temperature_span",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 <= self.active_span <= self.active_base:
            raise ValueError("active_span must be in [0, active_base]")
        if not 0.0 < self.active_base + self.active_span <= 1.0:
            raise ValueError("active fraction must stay within (0, 1]")
        if not 0.0 <= self.aggressive_span <= self.aggressive_base:
            raise ValueError("aggressive_span must be in [0, aggressive_base]")
        if not 0.0 < self.aggressive_base + self.aggressive_span <= 1.0:
            raise ValueError("aggressive fraction must stay within (0, 1]")
        if not 0.0 <= self.cap_span <= self.cap_base:
            raise ValueError("cap_span must be in [0, cap_base]")
        # 0.455 mirrors SafetyGates.risk_cap_stack_fraction's DEFAULT. It
        # cannot be imported (the engine sizer will import g — a cycle),
        # so this is a deliberate mirrored constant: revisit if a manifest
        # ever overrides the engine's cap. Below near-nut equity the
        # engine cap binds first at 0.455; at near-nut it releases and g
        # would be the sole stack authority, which is exactly why a block
        # is refused rather than trusted above this line.
        if not 0.0 < self.cap_base + self.cap_span <= 0.455:
            raise ValueError(
                "cap fraction must stay within (0, 0.455], the engine's"
                " default sub-near-nut risk cap"
            )
        if not 0.0 <= self.setpoint <= 100.0:
            raise ValueError("setpoint must be between 0 and 100")
        if not 5.0 <= self.temperature_span <= 100.0:
            raise ValueError("temperature_span must be between 5 and 100")

    def boldness(self, temperature: float) -> float:
        """Signed, clamped distance below the setpoint, in span units.

        The ONLY legitimate b for the sizing functions below. Feeding
        them the engine's ``_boldness`` — numerically identical today —
        reintroduces the raw-stack depth bug the panel blocked, and the
        agreement makes it invisible in shallow tests. Note the
        retired-endpoint property is stated AT b = ±1; a non-default
        ``(setpoint, temperature_span)`` may make those endpoints
        unreachable from real temperatures.
        """

        if not math.isfinite(temperature):
            raise ValueError("temperature must be finite")
        return max(
            -1.0, min(1.0, (self.setpoint - temperature) / self.temperature_span)
        )

    def to_mapping(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "SizingParameters":
        """Rebuild validated parameters from artifact JSON; unknown keys fail."""

        return cls(**dict(mapping))


DEFAULT_SIZING_PARAMETERS = SizingParameters()
#: Every span zeroed — the sizer ignores temperature on ALL lanes. This is
#: the only correct neutral arm; zeroing the active span alone leaves the
#: aggressive fraction and cap moving with b.
NEUTRAL_SIZING_PARAMETERS = SizingParameters(
    active_span=0.0, aggressive_span=0.0, cap_span=0.0
)


def sizing_record(
    parameters: SizingParameters = DEFAULT_SIZING_PARAMETERS,
) -> dict[str, Any]:
    """The g block for a v9 manifest, corpus header, or sidecar."""

    return {"identity": G_IDENTITY, "parameters": parameters.to_mapping()}


def parameters_from_record(record: Mapping[str, object]) -> SizingParameters:
    """The identity-checking inverse of :func:`sizing_record`.

    Every consumer of a stored g block (manifest, corpus header, sidecar)
    goes through here, so the identity comparison lives once: a record
    written under a different g refuses to load instead of silently
    re-deriving sizes under new formulas.
    """

    identity = record.get("identity")
    if identity != G_IDENTITY:
        raise ValueError(
            f"sizing record identity {identity!r} is not {G_IDENTITY!r} —"
            " its sizes were derived under a different g"
        )
    parameters = record.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("sizing record must carry a parameters mapping")
    return SizingParameters.from_mapping(parameters)


def depth_invariant_temperature(
    *,
    equity: float,
    hero_stack: int,
    call_chips: int,
    contested_stack: int,
    street: str,
    players: int,
) -> RiskTemperature:
    """g's own five-factor read, purse scoped to the table, not the bankroll.

    Numeric core — callers with a snapshot use :func:`table_temperature`.
    Validation is ``measure_risk_temperature``'s own and fails closed.
    """

    hero_stack = int(hero_stack)
    if hero_stack < 1:
        raise ValueError("hero_stack must be positive")
    call_chips = int(call_chips)
    if call_chips < 0:
        raise ValueError("call_chips cannot be negative")
    # The 1-floor is an implementation amendment to the spec's literal
    # formula: it can only bind at call_chips == 0, where bet pressure is
    # zero regardless, so T is unchanged wherever both forms read.
    purse_g = min(hero_stack, max(1, call_chips, int(contested_stack)))
    # bet is passed UNCLAMPED on purpose: on Arena-shaped data
    # call_chips <= purse_g holds by the contested-stack invariant, so
    # the bet <= purse guard stays armed and a corrupt row (a parity
    # replay's, say) raises instead of reading maximum pressure.
    return measure_risk_temperature(
        hand_strength=100.0 * equity,
        purse=purse_g,
        bet=call_chips,
        street=street,
        players=players,
    )


def table_temperature(
    table: Mapping[str, Any],
    allowed: Mapping[str, Any],
    equity: float | None,
) -> RiskTemperature | None:
    """The depth-invariant read from an Arena snapshot; None without equity.

    ``None`` means "no read" and every consumer sizes at b = 0 — the same
    neutral convention as the engine's ``_boldness``.
    """

    if equity is None:
        return None
    hero, _ = _hero_and_seats(table)
    return depth_invariant_temperature(
        equity=equity,
        hero_stack=_integer(hero.get("stackChips"), "hero stackChips", minimum=1),
        call_chips=_integer(allowed.get("callChips", 0), "callChips"),
        contested_stack=contested_stack_chips(table),
        street=str(table.get("street") or "").casefold(),
        players=1 + active_opponent_count(table),
    )


def table_boldness(
    table: Mapping[str, Any],
    allowed: Mapping[str, Any],
    equity: float | None,
    parameters: SizingParameters = DEFAULT_SIZING_PARAMETERS,
) -> float:
    """b from a snapshot via g's own read — the fence against the b-source trap.

    Snapshot consumers (engine sizer, extractor, harvester, floors) use
    THIS, never the engine's ``_boldness``: that one reads the raw-stack
    temperature and reproduces the depth bug while agreeing with g at
    shallow depth. Trainers reconstruct b from the recorded read instead:
    ``parameters.boldness(context_int_to_temperature(row))``.
    """

    reading = table_temperature(table, allowed, equity)
    return 0.0 if reading is None else parameters.boldness(reading.temperature)


def active_bet_fraction(
    boldness: float,
    parameters: SizingParameters = DEFAULT_SIZING_PARAMETERS,
) -> float:
    """Pot fraction of an unprovoked active-lane bet.

    ``boldness`` must come from :meth:`SizingParameters.boldness` on g's
    depth-invariant read (:func:`table_boldness` from a snapshot, the
    recorded ``10·T`` context int in a trainer) — never from the engine's
    raw-stack ``_boldness``.
    """

    return parameters.active_base + parameters.active_span * boldness


def active_bet_wager(
    pot: int,
    boldness: float,
    parameters: SizingParameters = DEFAULT_SIZING_PARAMETERS,
) -> float:
    """Unclamped chip wager of an active-lane bet at ``to_call == 0``.

    ``pot`` is RAW ``potChips``, uncallable overhang included — the
    engine's sizing convention. Passing an overhang-corrected pot breaks
    corpus/live parity by up to the overhang. The engine's big-blind
    floor, rounding, and legal clamp are NOT applied here —
    ``_sized_action`` legalizes.
    """

    if pot < 0:
        raise ValueError("pot cannot be negative")
    return active_bet_fraction(boldness, parameters) * pot


def aggressive_fractions(
    boldness: float,
    parameters: SizingParameters = DEFAULT_SIZING_PARAMETERS,
) -> tuple[float, float]:
    """``(f, s)`` for the aggressive lane: pot-target and stack-cap fractions."""

    return (
        parameters.aggressive_base + parameters.aggressive_span * boldness,
        parameters.cap_base + parameters.cap_span * boldness,
    )


def aggressive_target(
    *,
    pot: int,
    to_call: int,
    effective_stack: int,
    boldness: float,
    parameters: SizingParameters = DEFAULT_SIZING_PARAMETERS,
) -> float:
    """Unclamped escalation target in chips beyond hero's contribution.

    ``min(to_call + f·(pot + to_call), s·eff)`` — at b = -1/+1 exactly the
    retired v8 specs (0.50, 0.20)/(1.00, 0.45). ``pot`` is RAW
    ``potChips`` (overhang included, the engine convention); ``boldness``
    must come from g's own read (see :func:`table_boldness`). REFUSES
    ``to_call <= 0``: the aggressive lane does not exist at a free spot
    under the v9 contract, and a caller reaching here with one has a
    masking bug.

    The result can fall below the legal minimum raise (near-certain in
    the effective-stack-collapse states, where ``s·eff`` is under a
    chip); legalization — and the demotion of an unsizeable escalation to
    the active lane — belongs to the engine, never here. A genuine
    collapse arrives as ``effective_stack == 0`` and is floored to 1; a
    NEGATIVE stack is always a caller bug and refused, so a sign error
    cannot masquerade as a collapse.
    """

    if to_call <= 0:
        raise ValueError(
            "the aggressive lane is masked at to_call == 0 —"
            " unprovoked wagers belong to the active lane"
        )
    if pot < 0:
        raise ValueError("pot cannot be negative")
    if int(effective_stack) < 0:
        raise ValueError("effective_stack cannot be negative")
    fraction, cap = aggressive_fractions(boldness, parameters)
    return min(
        to_call + fraction * (pot + to_call),
        cap * max(1, int(effective_stack)),
    )


def read_to_context_int(temperature: float) -> int:
    """Encode a one-decimal temperature for a corpus row's raw-int context.

    ``round(temperature * 10)`` — exact for every value
    ``measure_risk_temperature`` can return (one decimal in [0, 100]).
    """

    encoded = round(temperature * 10)
    if not 0 <= encoded <= 1000:
        raise ValueError("temperature must be a 0-100 reading")
    return encoded


def context_int_to_temperature(encoded: int) -> float:
    """Decode ``read_to_context_int`` bit-exactly.

    ``encoded / 10.0`` is the correctly-rounded double of the one-decimal
    value, which is the same double ``round(x, 1)`` produced live.
    """

    encoded = int(encoded)
    if not 0 <= encoded <= 1000:
        raise ValueError("encoded temperature must be in [0, 1000]")
    return encoded / 10.0


__all__ = [
    "G_IDENTITY",
    "SizingParameters",
    "DEFAULT_SIZING_PARAMETERS",
    "NEUTRAL_SIZING_PARAMETERS",
    "sizing_record",
    "parameters_from_record",
    "depth_invariant_temperature",
    "table_temperature",
    "table_boldness",
    "active_bet_fraction",
    "active_bet_wager",
    "aggressive_fractions",
    "aggressive_target",
    "read_to_context_int",
    "context_int_to_temperature",
]
