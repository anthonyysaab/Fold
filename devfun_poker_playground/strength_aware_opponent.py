"""Precondition **P3** — a strength-aware opponent, fitted from real play.

`PLAYERS.md` ("What is missing, and what to build first") names this the
project's oldest blocker. Every scripted archetype in `table_simulator` is
**card-blind**: its continuing range is random no matter what it is charged,
so it cannot punish indiscriminate aggression, and a policy can therefore win
every battery while betting a random hand into a real field.
:class:`~devfun_poker_playground.table_simulator.TexturedAgent` moved the fold
rate onto the *price* but still reads nothing but public information.

This module closes the gap the only way `DECISIONS.md` §4.4 permits — from
data rather than invention. Public arena replays are complete information
(every seat's hole cards are revealed, folders included), so the field's
conditional fold behaviour is **measurable**:

    P(fold | strength_percentile, price, street, texture, seats, position)

`tools/build_p3_dataset.py` extracts every decision where a player faced
chips to call, labels it with the actor's *true* hand strength on the
canonical metric, fits a logistic regression, and writes
`artifacts/p3/p3-fit.json`. :class:`StrengthAwareAgent` evaluates that fit on
its **own** hole cards and the live price.

What this opponent **is**: a fitted model of the observed field's fold
frequency. What it is **not**: an optimal player, an exploitative player, or
a range-balanced one. It folds more when its hand is worse and when the price
is worse, which is the single property the batteries have never had — nothing
more is claimed for it.

Three contracts this module exists to honour, each of which has cost this
project a candidate or a day:

- ``reads_cards = True``. ``table_simulator.py:262`` excludes ``reads_cards``
  agents from the counterfactual chance salt. A card-aware agent whose hole
  cards get resampled between the decision and the rollout is exactly the
  frozen-runout defect that closed candidate 0016.
- ``_fold_probability`` **draws nothing from the RNG**. Its caller has already
  spent exactly one ``roll.random()`` for the decision; consuming more
  desynchronises counterfactual replay. Every quantity here is a pure
  deterministic function of the snapshot.
- The coefficients are **data, loaded from the artifact** — never literals in
  code (`DECISIONS.md` §4.6: hand-authored constants have a terrible track
  record here). The one surviving hand-set knob, the output clamp band, is
  recorded in the artifact with its derivation and is explicitly ablatable.
  The same goes for the **price-support clamp** (2026-08-16 repair of the
  overbet blind spot): each model's ``price_cap`` is a quantile of its own
  training rows' observed price, derived and recorded by the dataset tool
  (``price_support`` in the artifact), never authored here.

Stdlib-only, like every module on the serve/sim path.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from devfun_poker_playground.game_state import (
    ArenaSnapshotError,
    _hero_and_seats,
    _normalize_card,
    _position,
    active_opponent_count,
)
from devfun_poker_playground.strength_metric import (
    StrengthMetricError,
    strength_percentile,
)
from devfun_poker_playground.table_simulator import ScriptedAgent, board_coordination

_LOGGER = logging.getLogger(__name__)

#: The regressors the logistic is fitted on, in vector order.
#:
#: ``bet_to_pot`` is recorded on every dataset row but is deliberately **not**
#: a regressor: ``pot_odds == bet_to_pot / (1 + bet_to_pot)`` exactly, so the
#: two are the same variable under a monotone reparameterisation. Fitting both
#: is exact collinearity, the two price coefficients can then take opposite
#: signs while the model is unchanged, and the sign invariant that decides
#: this whole fit's validity would become unreadable. One price term, one
#: sign to check.
FIT_FEATURE_NAMES: tuple[str, ...] = (
    "strength_percentile",
    "pot_odds",
    "texture",
    "active_players",
    "position_unit",
)

#: The index of the strength regressor and of the price regressor. The
#: invariant is stated against these two, in one place, so the tool, the
#: artifact and the tests cannot disagree about which column it names.
STRENGTH_INDEX = FIT_FEATURE_NAMES.index("strength_percentile")
PRICE_INDEX = FIT_FEATURE_NAMES.index("pot_odds")

STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")

DEFAULT_FIT_PATH = Path("artifacts") / "p3" / "p3-fit.json"

#: Used when a snapshot's seating cannot be resolved into a position. Neutral
#: by construction (mid-table), and the same value ``foreign_data`` uses for a
#: standing it cannot read.
NEUTRAL_POSITION_UNIT = 0.5


class P3FitError(ValueError):
    """Raised when a P3 fit artifact is missing, malformed, or sign-broken."""


# ---------------------------------------------------------------------------
# Features — one definition, shared by the dataset tool and the agent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class P3Decision:
    """One "facing a wager" decision, in the P3 feature space.

    ``strength_percentile`` is the canonical metric
    (:mod:`devfun_poker_playground.strength_metric`) — the only strength scale
    this project reports on, per ``V8_DESIGN.md`` §2. In the dataset it is the
    actor's revealed hand; at serve it is the agent's own hand. Same function,
    same scale, no drift between fit and use.
    """

    street: str
    strength_percentile: float
    pot_odds: float
    bet_to_pot: float
    texture: float
    active_players: int
    position_unit: float
    to_call: int
    pot: int

    @property
    def vector(self) -> tuple[float, ...]:
        """The regressors in :data:`FIT_FEATURE_NAMES` order."""

        return tuple(float(getattr(self, name)) for name in FIT_FEATURE_NAMES)


@lru_cache(maxsize=200_000)
def _cached_strength(hole: tuple[str, ...], board: tuple[str, ...]) -> float:
    """Memoised canonical strength.

    ``strength_percentile`` is a pure, seedless, deterministic function
    (exact enumeration postflop, a table lookup preflop), so caching it can
    change nothing but the clock. It enumerates ~1,081 opponent holdings per
    postflop call, which a simulated match would otherwise pay for on every
    decision.
    """

    return strength_percentile(hole, board)


def _cards(values: object) -> tuple[str, ...]:
    if values is None:
        raise ArenaSnapshotError("missing cards")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ArenaSnapshotError("cards must be an array")
    return tuple(_normalize_card(value) for value in values)


def hero_hole_cards(table: Mapping[str, Any]) -> tuple[str, ...] | None:
    """The acting seat's hole cards, or ``None`` when the snapshot hides them.

    Returning ``None`` rather than raising is the whole point: the caller
    degrades to the card-blind parent behaviour and records that it happened.
    """

    try:
        hero, _ = _hero_and_seats(table)
    except ArenaSnapshotError:
        return None
    try:
        cards = _cards(hero.get("holeCards"))
    except ArenaSnapshotError:
        return None
    if len(cards) != 2 or len(set(cards)) != 2:
        return None
    return cards


def position_unit(table: Mapping[str, Any]) -> float:
    """Hero's seat order in [0, 1], or the neutral value if unreadable.

    Delegates to ``game_state._position`` so the dataset and the agent share
    the engine's own definition instead of a second implementation that could
    drift from it.
    """

    try:
        hero, seats = _hero_and_seats(table)
        value = _position(table, seats, hero)
    except (ArenaSnapshotError, ValueError, TypeError, KeyError):
        return NEUTRAL_POSITION_UNIT
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return NEUTRAL_POSITION_UNIT
    return float(value)


def decision_features(
    table: Mapping[str, Any],
    allowed: Mapping[str, Any],
    hole_cards: Sequence[str],
) -> P3Decision:
    """Build the P3 feature row for one decision facing chips to call.

    ``hole_cards`` is passed in rather than read here because the two callers
    resolve it from different places — the dataset tool from the replay's
    complete-information reveal, the agent from its own seat — while the
    feature *arithmetic* below must be byte-identical in both. Raises
    :class:`ArenaSnapshotError` / :class:`StrengthMetricError` on a snapshot
    it cannot read; both callers treat that as "degrade, and count it".
    """

    street = str(table.get("street") or "").casefold()
    if street not in STREETS:
        raise ArenaSnapshotError(f"unsupported street {table.get('street')!r}")

    raw_call = allowed.get("callChips")
    if raw_call is None:
        raw_call = allowed.get("callAmount")
    if isinstance(raw_call, bool) or not isinstance(raw_call, (int, float)):
        raise ArenaSnapshotError("allowedActions.callChips must be a number")
    to_call = max(0, int(raw_call))

    raw_pot = table.get("potChips")
    if isinstance(raw_pot, bool) or not isinstance(raw_pot, (int, float)):
        raise ArenaSnapshotError("potChips must be a number")
    pot = max(0, int(raw_pot))

    # The pot already includes the outstanding bet in both sources (the
    # simulator sums hand commitments; the replay payload's ``pot`` counts
    # the posted blinds), so this is the standard price laid.
    total = pot + to_call
    pot_odds = (to_call / total) if total > 0 else 0.0
    bet_to_pot = (to_call / pot) if pot > 0 else 0.0

    board = _cards(table.get("boardCards") or ())
    hole = _cards(hole_cards)
    if len(hole) != 2:
        raise StrengthMetricError("a holding is exactly two cards")
    strength = _cached_strength(tuple(sorted(hole)), board)

    return P3Decision(
        street=street,
        strength_percentile=float(strength),
        pot_odds=float(pot_odds),
        bet_to_pot=float(bet_to_pot),
        texture=float(board_coordination(board)),
        active_players=int(active_opponent_count(table) + 1),
        position_unit=position_unit(table),
        to_call=to_call,
        pot=pot,
    )


# ---------------------------------------------------------------------------
# The fitted model — prediction only; the fitting lives in tools/
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogisticModel:
    """A standardised logistic regression, evaluated on the serve path.

    ``coefficients`` are on the **standardised** scale, paired with the
    ``mean``/``std`` recorded at fit time. Standardising with a positive std
    preserves sign, so the sign invariant reads identically on either scale.

    ``price_cap`` is the **price-support clamp** (repair of the overbet
    blind spot measured 2026-08-16, ``tests/test_p3_audit.py``
    ``OverbetPunishmentTests``): the archive's fields bet at most ~1.1x pot,
    so every overbet price the opponent is asked to fold to is an
    extrapolation, and the positive price slope extrapolated far off its
    support drove the fold probability toward 1.0 — a zero-equity bluff's EV
    *rose* with size. The cap is the fitted-support quantile of the model's
    own training rows' ``pot_odds`` (derived by the dataset tool, stored in
    the artifact's ``price_support`` section, quantile choice ablatable —
    never a hand-authored constant, per `DECISIONS.md` §4.6). The price
    input is clamped **from above only** before the logit is evaluated:
    above the cap the fold probability is flat in bet size, so a bigger
    overbet buys no extra fold equity. ``bet_to_pot`` is capped by
    implication — it is the same variable as ``pot_odds`` under a monotone
    reparameterisation and is not a regressor. ``None`` (e.g. a pre-repair
    artifact, or the audit tests' ad-hoc fits) means no clamp.
    """

    intercept: float
    coefficients: tuple[float, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    feature_names: tuple[str, ...] = FIT_FEATURE_NAMES
    price_cap: float | None = None

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        if not (len(self.coefficients) == len(self.mean) == len(self.std) == width):
            raise P3FitError("model vectors disagree on width")
        if any(value <= 0.0 or not math.isfinite(value) for value in self.std):
            raise P3FitError("standard deviations must be finite and positive")
        if not math.isfinite(self.intercept) or any(
            not math.isfinite(value) for value in self.coefficients
        ):
            raise P3FitError("model contains a non-finite parameter")
        if self.price_cap is not None:
            if not math.isfinite(self.price_cap) or not 0.0 < self.price_cap < 1.0:
                raise P3FitError(
                    f"price_cap {self.price_cap!r} is not a pot-odds value in (0, 1)"
                )
            # The clamp is applied at PRICE_INDEX; a model whose feature
            # order disagrees would silently clamp the wrong regressor.
            if self.feature_names[PRICE_INDEX] != "pot_odds":
                raise P3FitError(
                    "price_cap requires the price regressor at PRICE_INDEX; "
                    f"found {self.feature_names[PRICE_INDEX]!r}"
                )

    def logit(self, values: Sequence[float]) -> float:
        if len(values) != len(self.coefficients):
            raise P3FitError("feature vector width does not match the model")
        if (
            self.price_cap is not None
            and float(values[PRICE_INDEX]) > self.price_cap
        ):
            values = list(values)
            values[PRICE_INDEX] = self.price_cap
        total = self.intercept
        for weight, value, centre, scale in zip(
            self.coefficients, values, self.mean, self.std
        ):
            total += weight * ((float(value) - centre) / scale)
        return total

    def predict(self, values: Sequence[float]) -> float:
        """P(fold), numerically safe at both tails."""

        z = self.logit(values)
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
        exponent = math.exp(max(z, -700.0))
        return exponent / (1.0 + exponent)

    @property
    def strength_coefficient(self) -> float:
        return self.coefficients[STRENGTH_INDEX]

    @property
    def price_coefficient(self) -> float:
        return self.coefficients[PRICE_INDEX]

    def check_sign_invariant(self, label: str = "model") -> None:
        """The check that decides whether this fit is allowed to exist.

        Better hands must fold **less** (negative strength coefficient) and
        worse prices must fold **more** (positive price coefficient). If
        either sign is wrong the fit is broken or the label is inverted, and
        no downstream number from it means anything.
        """

        if not self.strength_coefficient < 0.0:
            raise P3FitError(
                f"{label}: strength_percentile coefficient "
                f"{self.strength_coefficient:+.6f} is not negative — better "
                "hands must fold less; the fit or the label is inverted"
            )
        if not self.price_coefficient > 0.0:
            raise P3FitError(
                f"{label}: pot_odds coefficient {self.price_coefficient:+.6f} "
                "is not positive — worse prices must fold more; the fit or "
                "the label is inverted"
            )

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any], feature_names: Sequence[str]
    ) -> "LogisticModel":
        try:
            raw_cap = document.get("price_cap")
            return cls(
                intercept=float(document["intercept"]),
                coefficients=tuple(float(v) for v in document["coefficients"]),
                mean=tuple(float(v) for v in document["mean"]),
                std=tuple(float(v) for v in document["std"]),
                feature_names=tuple(str(name) for name in feature_names),
                price_cap=None if raw_cap is None else float(raw_cap),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, P3FitError):
                raise
            raise P3FitError(f"malformed model document: {error}") from error

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "mean": list(self.mean),
            "std": list(self.std),
        }
        if self.price_cap is not None:
            document["price_cap"] = self.price_cap
        return document


@dataclass(frozen=True)
class P3Fit:
    """Everything :class:`StrengthAwareAgent` needs, loaded from the artifact."""

    feature_names: tuple[str, ...]
    models: Mapping[str, LogisticModel]
    street_to_model: Mapping[str, str]
    fallback_model: str
    band: tuple[float, float]
    source: str = ""

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != FIT_FEATURE_NAMES:
            raise P3FitError(
                "fit was built against a different feature contract: "
                f"{tuple(self.feature_names)!r} != {FIT_FEATURE_NAMES!r}"
            )
        if self.fallback_model not in self.models:
            raise P3FitError(f"fallback model {self.fallback_model!r} is not present")
        for street, key in self.street_to_model.items():
            if street not in STREETS:
                raise P3FitError(f"unknown street {street!r} in street_to_model")
            if key not in self.models:
                raise P3FitError(f"street {street!r} maps to missing model {key!r}")
        low, high = self.band
        if not (0.0 <= low < high <= 1.0):
            raise P3FitError(f"clamp band {self.band!r} is not 0 <= low < high <= 1")

    def model_for(self, street: str) -> LogisticModel:
        key = self.street_to_model.get(str(street).casefold(), self.fallback_model)
        return self.models[key]

    def check_sign_invariant(self) -> None:
        for key, model in sorted(self.models.items()):
            model.check_sign_invariant(key)

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any], *, source: str = ""
    ) -> "P3Fit":
        try:
            names = tuple(str(name) for name in document["features"])
            models = {
                str(key): LogisticModel.from_document(value, names)
                for key, value in document["models"].items()
            }
            band_doc = document["band"]
            band = (float(band_doc["low"]), float(band_doc["high"]))
            fit = cls(
                feature_names=names,
                models=models,
                street_to_model={
                    str(k): str(v) for k, v in document["street_to_model"].items()
                },
                fallback_model=str(document["fallback_model"]),
                band=band,
                source=source,
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, P3FitError):
                raise
            raise P3FitError(f"malformed P3 fit document: {error}") from error
        # A sign-broken artifact must never reach a battery: it would make
        # the opponent reward bad hands, and every number measured against it
        # would be confidently wrong.
        fit.check_sign_invariant()
        return fit

    @classmethod
    def load(cls, path: str | Path = DEFAULT_FIT_PATH) -> "P3Fit":
        resolved = Path(path)
        try:
            document = json.loads(resolved.read_text(encoding="utf-8"))
        except OSError as error:
            raise P3FitError(f"cannot read P3 fit at {resolved}: {error}") from error
        except ValueError as error:
            raise P3FitError(f"P3 fit at {resolved} is not JSON: {error}") from error
        if not isinstance(document, Mapping):
            raise P3FitError(f"P3 fit at {resolved} is not an object")
        return cls.from_document(document, source=str(resolved))


@lru_cache(maxsize=8)
def load_fit(path: str | Path = DEFAULT_FIT_PATH) -> P3Fit:
    """Load and cache a fit artifact (immutable, so sharing it is safe)."""

    return P3Fit.load(path)


# ---------------------------------------------------------------------------
# The opponent
# ---------------------------------------------------------------------------


@dataclass
class StrengthAwareAgent(ScriptedAgent):
    """Precondition **P3**: an archetype whose folding responds to its hand.

    Everything :class:`ScriptedAgent` does — the seeded mixer, the aggression
    roll, the sizing — is inherited unchanged. The single override is
    :meth:`_fold_probability`, which replaces the constant ``fold_vs_bet``
    with the fitted field model evaluated on this agent's **own** hole cards
    and the price it is being laid.

    Consequences, spelled out because they are load-bearing:

    - ``reads_cards = True``, so ``TableSimulator`` will not resample this
      agent's holding under the counterfactual chance salt. That exclusion is
      mandatory: an agent that acts on its cards and then has them replaced
      produces a betting history the replay cannot reproduce (the defect that
      closed candidate 0016). It also means labels harvested against this
      opponent need the ``V8_DESIGN.md`` §5 salt rule, which is a Phase-B
      concern and not this class's job.
    - :meth:`_fold_probability` consumes **no** randomness. The parent has
      already drawn its single ``roll.random()`` before calling this.
    - ``band`` clamps the output. Its purpose is a rail, not shaping: a model
      extrapolated off its support must never be able to fold always (an
      opponent that always folds is unbeatable-by-folding and would flatter
      any aggressive candidate). It is recorded in the artifact with its
      derivation and is **ablatable** — pass ``band=(lo, hi)`` to move it.
    """

    # table_simulator.py:262 keys the chance-salt exclusion on this flag.
    reads_cards = True

    policy_version = "strength-aware"

    #: Injected in tests / ablations; ``None`` loads :data:`DEFAULT_FIT_PATH`.
    fit: P3Fit | None = None
    fit_path: str | Path = DEFAULT_FIT_PATH
    #: Overrides the artifact's band (the ablation arm).
    band: tuple[float, float] | None = None

    #: How often this agent had to fall back to the card-blind constant.
    #: A battery run that reports a non-zero count is running a *different*
    #: opponent than the one it thinks it is; tests fail loud on it.
    fallback_count: int = field(default=0, compare=False, repr=False)
    last_fallback_reason: str = field(default="", compare=False, repr=False)

    def resolved_fit(self) -> P3Fit:
        if self.fit is None:
            self.fit = load_fit(self.fit_path)
        return self.fit

    def resolved_band(self) -> tuple[float, float]:
        if self.band is not None:
            low, high = self.band
            if not (0.0 <= low < high <= 1.0):
                raise P3FitError(f"clamp band {self.band!r} is not 0 <= low < high <= 1")
            return float(low), float(high)
        return self.resolved_fit().band

    def _degrade(self, reason: str) -> float:
        self.fallback_count += 1
        self.last_fallback_reason = reason
        _LOGGER.warning(
            "%s: strength-aware fold probability unavailable (%s); "
            "degrading to the card-blind constant %.3f (occurrence %d)",
            self.name,
            reason,
            self.fold_vs_bet,
            self.fallback_count,
        )
        return self.fold_vs_bet

    def fold_probability_for(self, decision: P3Decision) -> float:
        """The clamped fitted P(fold) for an already-built feature row.

        Exposed so the monotonicity property the whole opponent exists for
        can be swept directly, without synthesising a snapshot per point.
        """

        model = self.resolved_fit().model_for(decision.street)
        low, high = self.resolved_band()
        return min(high, max(low, model.predict(decision.vector)))

    def _fold_probability(
        self, table: Mapping[str, Any], allowed: Mapping[str, Any]
    ) -> float:
        """Chance of folding when facing a wager — fitted, not constant.

        A pure deterministic function of ``table`` and ``allowed``. **Draws
        nothing from any RNG**: the caller has already spent exactly one
        ``roll.random()`` for this decision and consuming more would
        desynchronise counterfactual replay (see
        :meth:`ScriptedAgent._fold_probability`).
        """

        hole = hero_hole_cards(table)
        if hole is None:
            return self._degrade("hole cards absent from the snapshot")
        try:
            decision = decision_features(table, allowed, hole)
        except (
            ArenaSnapshotError,
            StrengthMetricError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            return self._degrade(f"{type(error).__name__}: {error}")
        try:
            return self.fold_probability_for(decision)
        except P3FitError as error:
            return self._degrade(f"P3FitError: {error}")


def strength_aware_lineup(
    fit: P3Fit | None = None, seed: int = 5
) -> list[tuple[str, StrengthAwareAgent]]:
    """The P3 battery seats: the fitted fold model at three aggression levels.

    ``aggression``/``shove_rate`` are inherited scripted knobs and stay at the
    audited arena values used by ``calibrated_lineup``; only the *folding* is
    fitted here, which is precisely the axis P3 was blocked on. ``fold_vs_bet``
    is the value each seat degrades to if a snapshot ever hides its cards.
    """

    return [
        (
            "p3-median",
            StrengthAwareAgent("p3-median", 0.226, 0.5, 0.0, seed, fit=fit),
        ),
        (
            "p3-passive",
            StrengthAwareAgent("p3-passive", 0.10, 0.5, 0.0, seed + 1, fit=fit),
        ),
        (
            "p3-aggressive",
            StrengthAwareAgent("p3-aggressive", 0.35, 0.5, 0.02, seed + 2, fit=fit),
        ),
    ]


__all__ = [
    "DEFAULT_FIT_PATH",
    "FIT_FEATURE_NAMES",
    "LogisticModel",
    "P3Decision",
    "P3Fit",
    "P3FitError",
    "PRICE_INDEX",
    "STREETS",
    "STRENGTH_INDEX",
    "StrengthAwareAgent",
    "decision_features",
    "hero_hole_cards",
    "load_fit",
    "position_unit",
    "strength_aware_lineup",
]
