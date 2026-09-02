"""The fitted P3 belief provider — block 8 wired at last (owner, 2026-08-30).

The eight ``belief_bucket_*`` inputs were plumbed on 2026-08-16 for a
"price-conditioned continuing-range distribution from the fitted P3
opponent model" and then served the neutral uniform prior on every vector
ever produced — the fitted provider was never written (found in the
2026-08-29 schema review, block 8: eight constant inputs are, after
z-scoring, eight inputs that are identically zero forever). This module
is that provider.

The construction — Bayes over strength octiles, with the fit as the
likelihood:

- A uniformly random holding's canonical strength percentile is uniform
  by construction, so the PRIOR over the eight octiles is exactly the
  neutral provider's 1/8 each.
- Every priced CONTINUE by a currently-active, non-hero opponent this
  hand (a call or a raise recorded in the hand's events) is evidence:
  P3 gives ``P(fold | strength, price, street, texture, seats,
  position)``, so the likelihood of that opponent still being here is
  ``1 − P(fold)``, evaluated at each octile's midpoint strength with the
  price and pot THEY faced at that moment.
- Buckets are the normalized product of those likelihoods. No priced
  continues yet → no evidence → exactly the uniform prior (bit-equal to
  the neutral provider, which keeps the wire-in a strict refinement).

Reuse, not re-derivation: the likelihood is the SAME fitted model the
strength-aware battery opponent serves — ``P3Fit`` / ``P3Decision`` /
``LogisticModel.predict`` (which applies the price-support clamp
internally, so the overbet repair is inherited) with the fit's own clamp
band, and the texture is ``board_coordination``. The fit's sign
invariant (better hands fold less, worse prices fold more) is checked at
load, which is what makes the direction of the tilt trustworthy: mass
moves toward the TOP octiles after an opponent calls, more so for bigger
prices.

Reconstruction from the hand's event records (``_hand_events`` shape:
street / seat_number / action / amount, where amount is the call wager
for calls and the raise-TO amount for aggressive actions), with three
documented approximations, each chosen over a fabricated precision:

- The running pot seeds from the posted blinds (the snapshot's blind
  fields; blind EVENTS carry no action and are absent from the records),
  and per-seat street commitments are tracked from the records alone —
  a blind poster's preflop raise therefore reads as slightly more new
  money than it was.
- The position unit of a past actor is served NEUTRAL (0.5); the fit
  was trained with real positions. Its coefficient is small; the
  approximation is recorded rather than invented around.
- ``active_players`` at a past event counts seats not yet folded in the
  visible record window.

Failure posture: fit problems fail LOUD at construction (a serving
stack must not silently degrade to the constant buckets the corpus was
NOT trained on — that is the exact defect this module closes). A
malformed EVENT is skipped and counted; an unexpected per-decision error
degrades that one decision to the uniform prior with the reason kept on
``last_degrade_reason`` for diagnosis. Deterministic: pure arithmetic,
no RNG, cached fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from engine.belief_provider import BeliefProviderError
from engine.schema3 import BELIEF_BUCKETS
from engine.strength_aware_opponent import (
    DEFAULT_FIT_PATH,
    P3Decision,
    P3Fit,
    board_coordination,
    load_fit,
)

_BOARD_BY_STREET = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
_CONTINUE_ACTIONS = frozenset({"call", "raise", "bet", "all-in"})
#: Neutral acting-order input for past actors — see the module docstring.
_NEUTRAL_POSITION_UNIT = 0.5


class P3BeliefProvider:
    """Continuing-range octile distribution from the fitted P3 model."""

    def __init__(self, fit: P3Fit) -> None:
        if not isinstance(fit, P3Fit):
            raise BeliefProviderError("P3BeliefProvider requires a P3Fit")
        self._fit = fit
        #: Midpoint strength of each octile — the representative holding
        #: the likelihood is evaluated at.
        self._midpoints = tuple(
            (index + 0.5) / BELIEF_BUCKETS for index in range(BELIEF_BUCKETS)
        )
        self.last_degrade_reason: str | None = None

    def clone(self) -> "P3BeliefProvider":
        """A provider copy that isolates the per-decision degrade reason.

        The counterfactual replay clones the serving policy per branch and
        rollout; a clone sharing its provider would overwrite the
        original's ``last_degrade_reason`` with the replay's, corrupting
        the per-decision belief-degrade telemetry. The fitted model is
        immutable (and already shared across seats via
        :func:`strength_aware_opponent.load_fit`), so only the container
        is copied.
        """

        clone = type(self).__new__(type(self))
        clone.__dict__ = self.__dict__.copy()
        return clone

    @classmethod
    def from_artifact(cls, path: str | Path = DEFAULT_FIT_PATH) -> "P3BeliefProvider":
        """Load the fitted artifact — loud on any problem, never a fallback."""

        return cls(load_fit(path))

    @property
    def fit_source(self) -> str:
        """Provenance string for corpus headers and manifests."""

        return self._fit.source

    def continuing_range_buckets(
        self,
        table: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[float, ...]:
        self.last_degrade_reason = None
        try:
            return self._buckets(table, records)
        except Exception as error:  # noqa: BLE001 — degrade ONE decision, keep why
            self.last_degrade_reason = f"{type(error).__name__}: {error}"
            return tuple(1.0 / BELIEF_BUCKETS for _ in range(BELIEF_BUCKETS))

    # -- internals ---------------------------------------------------------

    def _buckets(
        self,
        table: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[float, ...]:
        hero_seat = table.get("selfSeatNumber")
        board = [
            str(card)
            for card in (table.get("boardCards") or ())
            if isinstance(card, str)
        ]
        active_now: set[int] = set()
        seats = table.get("seats") or ()
        for seat in seats if isinstance(seats, (list, tuple)) else ():
            if not isinstance(seat, Mapping):
                continue
            number = seat.get("seatNumber")
            if (
                isinstance(number, int)
                and not isinstance(number, bool)
                and str(seat.get("status") or "").casefold() == "active"
            ):
                active_now.add(number)

        def _blind(key: str) -> int:
            value = table.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return 0
            return max(0, int(value))

        pot = _blind("smallBlindChips") + _blind("bigBlindChips")
        street = "preflop"
        commits: dict[int, int] = {}
        high_bet = _blind("bigBlindChips")
        players = max(2, len(list(seats))) if seats else 2
        folded: set[int] = set()

        likelihood = [1.0] * BELIEF_BUCKETS
        evidence = 0

        for record in records:
            if not isinstance(record, Mapping):
                continue
            action = str(record.get("action") or "")
            event_street = str(record.get("street") or "")
            seat_number = record.get("seat_number")
            if event_street != street:
                # New betting round: commitments and the high bet reset.
                street = event_street
                commits = {}
                high_bet = 0
            if action == "fold":
                if isinstance(seat_number, int) and not isinstance(seat_number, bool):
                    folded.add(seat_number)
                continue
            amount = record.get("amount")
            if isinstance(amount, bool) or not isinstance(amount, int):
                continue

            prior_commit = (
                commits.get(seat_number, 0)
                if isinstance(seat_number, int) and not isinstance(seat_number, bool)
                else 0
            )
            if action == "call":
                to_call = max(0, amount)
                new_money = to_call
                new_commit = prior_commit + to_call
            elif action in ("bet", "raise", "all-in"):
                # amount is the raise-TO for aggressive actions.
                to_call = max(0, high_bet - prior_commit)
                new_money = max(0, amount - prior_commit)
                new_commit = max(prior_commit, amount)
                high_bet = max(high_bet, amount)
            else:
                continue

            pot_before = pot
            pot += new_money
            if isinstance(seat_number, int) and not isinstance(seat_number, bool):
                commits[seat_number] = new_commit

            # Evidence: a PRICED continue by a still-active, non-hero seat.
            if (
                action in _CONTINUE_ACTIONS
                and to_call > 0
                and isinstance(seat_number, int)
                and not isinstance(seat_number, bool)
                and seat_number != hero_seat
                and seat_number in active_now
                and event_street in _BOARD_BY_STREET
            ):
                event_board = tuple(board[: _BOARD_BY_STREET[event_street]])
                active_at_event = max(2, players - len(folded))
                total = pot_before + to_call
                decision = P3Decision(
                    street=event_street,
                    strength_percentile=0.0,  # per-octile below
                    pot_odds=(to_call / total) if total > 0 else 0.0,
                    bet_to_pot=(to_call / pot_before) if pot_before > 0 else 0.0,
                    texture=float(board_coordination(event_board)),
                    active_players=active_at_event,
                    position_unit=_NEUTRAL_POSITION_UNIT,
                    to_call=to_call,
                    pot=pot_before,
                )
                model = self._fit.model_for(event_street)
                low, high = self._fit.band
                for index, midpoint in enumerate(self._midpoints):
                    row = P3Decision(
                        street=decision.street,
                        strength_percentile=midpoint,
                        pot_odds=decision.pot_odds,
                        bet_to_pot=decision.bet_to_pot,
                        texture=decision.texture,
                        active_players=decision.active_players,
                        position_unit=decision.position_unit,
                        to_call=decision.to_call,
                        pot=decision.pot,
                    )
                    p_fold = min(high, max(low, model.predict(row.vector)))
                    likelihood[index] *= 1.0 - p_fold
                evidence += 1

        if evidence == 0:
            return tuple(1.0 / BELIEF_BUCKETS for _ in range(BELIEF_BUCKETS))
        total_mass = sum(likelihood)
        if total_mass <= 0.0:
            return tuple(1.0 / BELIEF_BUCKETS for _ in range(BELIEF_BUCKETS))
        return tuple(value / total_mass for value in likelihood)


__all__ = ["P3BeliefProvider"]
