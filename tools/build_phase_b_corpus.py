"""Phase B: counterfactual value labels against the repaired P3 opponent.

``V8_DESIGN.md`` §5 Phase B, built on the repaired strength-aware opponent
(the 2026-08-16 price-support clamp). Branch labels are generated exactly as
v7's were — **one eligible decision per actor and hand**, ``inclusion_count``
recorded, forced branches through the acting policy's **own serve path**
(``decide_forced``) — but the acting policy is a v8 candidate, the branch set
is the **E6 state-dependent branch set**, and the rollouts run against
lineups that include the fitted :class:`StrengthAwareAgent`.

Everything here is additive: ``table_simulator.py`` is subclassed, never
edited, and its counterfactual machinery (the branch probe, absorption by
distinct executed action, the selection RNG, ``_ForcedFamilyPolicy``) is
reused verbatim wherever a card-aware opponent does not change the physics.

The chance-salt rule for a card-aware opponent (V8_DESIGN §5)
-------------------------------------------------------------

The stock salt resamples a card-blind opponent's holes freely *because* its
play ignores them, and freezes a ``reads_cards`` agent's holes because
resampling them would change the replayed prefix (the frozen-runout defect
that closed candidate 0016 — through the opponent's cards this time). Both
halves are wrong for P3: frozen holes make every rollout share the same
opponent holding, so the label converges on E[chips | that exact holding]
instead of Q(s, a); free resampling contradicts the opponent's own committed
actions. The rule, implemented here:

- Hero's cards and the already-revealed board stay **fixed**; future board
  cards resample per rollout (unchanged).
- The P3 opponent's holes **also resample per rollout — never frozen** —
  and they resample **consistently with its own committed actions in the
  hand prefix**: during the prefix replay the P3 seat plays its original
  cards (so the betting history reproduces byte-for-byte), and at the first
  post-branch event its holes are replaced by a **rejection sample** from
  the unseen deck. A candidate holding is accepted when the joint fitted
  probability of the seat's observed prefix choices — ``P(fold)`` for each
  prefix fold, ``1 - P(fold)`` for each faced-price continue, evaluated by
  the seat's own clamped fit on the candidate holding — is at least
  ``--p3-accept-threshold`` (default ``DEFAULT_ACCEPT_THRESHOLD``,
  documented there, ablatable). If no candidate passes within
  ``--p3-resample-tries`` draws the seat falls back to a **uniform**
  resample and the fallback is **counted**; the fallback rate is reported
  per leg and in the corpus summary.

Documented approximation: the acceptance likelihood reads only the fitted
fold probability. The archetype's card-independent rolls (shove, aggression,
sizing) cancel between candidate holdings, except that a shove that fired
before the fold roll makes the ``1 - P(fold)`` factor conservative; at the
lineup's shove rates (0–2%) this is negligible and it errs toward stronger
holdings, never weaker.

Mechanically, a replay pre-computes the exact card arrangement the stock
salt would deal (original deal RNG + salt RNG, replicated bit-for-bit and
verified per decision against the captured branch-point snapshot), plays it
through ``deck_for_test``, and swaps the P3 holes at the first continuation
event — the swap sees the full arrangement, so the sample provably cannot
collide with the board, hero, or any other seat. With no P3 seat present the
arranged replay is **byte-identical** to the stock salted replay
(``tests/test_build_phase_b_corpus.py`` asserts this — the
impossible-by-construction validation of the replication).

The E6 branch set (V8_DESIGN §4)
--------------------------------

Candidates per decision: ``fold``, ``check_call``, and — only when a legal
bet/raise range is stated, mirroring the serve path's emission rule —
``aggress_small`` / ``aggress_large`` with the **state-dependent** pot
fraction ``(target - call) / (pot + call)`` from the unclamped E6 target
``min(call + pot_fraction*(pot + call), stack_fraction*effective_stack)``
(specs shared with ``feature_extract_v8`` / ``compose_branch_values``, so
label-time sizing and serve-time sizing cannot drift). The engine's own
``decide_forced`` realizes the fraction with every clamp intact, the branch
probe dedups by distinct executed action, and the absorption map is
recorded per decision; the small/large collision rate is reported per leg.

Corpus format — why not ``training_telemetry.save_training_corpus``
-------------------------------------------------------------------

That writer hard-pins ``feature_schema_version`` to the v7 learning
contract (2, 142 features) in its header and its loader refuses anything
else; Phase B rows carry the schema-3 vector (413 floats), so using it
would either lie in the header or be unreadable. Phase B rows also need
per-decision grouping (features shared by the decision's branches, the E6
branch context, per-decision P3 resample stats) that ``TrainingExample``
cannot carry. The corpus is therefore a documented v8-native format under
``artifacts/phase_b/``: gzip JSONL, first line a header object
(``kind: phase-b-corpus``, corpus schema 1, feature schema 3, input size,
branch labels), then **one line per decision** with the 413-float feature
vector, the emitted branches (centered ``reward_bb`` per branch, raw
``outcome_bb``, executed action, E6 sizing), the absorption map, and
``inclusion_count``. ``load_phase_b_corpus`` / ``validate_phase_b_rows``
below are the reference reader and validator. Targets are centered within
the decision (the v7 rule the v8 composition keeps), and the validator
asserts the per-decision centered sum is ~0, every label finite, every
vector ``schema3.require_vector``-clean, and the absorption map coherent.

Money safety: pure offline simulation. No Arena requests, no credentials,
no promotion, no ``artifacts/approved.json``.

Usage (repo root, stdlib interpreter):
    python -m tools.build_phase_b_corpus --dry-run
    python -m tools.build_phase_b_corpus \
        --candidate artifacts/candidates/candidate-v8-0001.manifest.json
    python -m tools.build_phase_b_corpus --validate \
        artifacts/phase_b/candidate-v8-0002.phase-b.jsonl.gz
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gzip
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine import schema3
from engine.decision_engine import SharedEquityCache
from engine.feature_extract_v8 import (
    _aggressive_range,
    _BRANCH_LARGE,
    _BRANCH_SMALL,
    extract_features_v8,
)
from engine.game_state import effective_stack_chips
from engine.learned_policy_v8 import load_policy_v8
from engine.strength_aware_opponent import (
    P3Decision,
    StrengthAwareAgent,
    _cached_strength,
    decision_features,
    hero_hole_cards,
    load_fit,
)
from engine.table_simulator import (
    _DECK,
    _ForcedFamilyPolicy,
    _REVEALED_BOARD,
    MatchResult,
    RecordingPolicy,
    ScriptedAgent,
    SimulationError,
    TableSimulator,
    TexturedAgent,
)
from engine.v8_trainer import BRANCH_LABELS_V8

CORPUS_KIND = "phase-b-corpus"
CORPUS_SCHEMA_VERSION = 1
DEFAULT_CORPUS_NAME = "candidate-v8-0002"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "phase_b"
DEFAULT_CANDIDATE = (
    Path("artifacts") / "candidates" / "candidate-v8-0001.manifest.json"
)

#: Acceptance threshold for the conditional P3 hole resample: the joint
#: fitted probability of the seat's observed prefix choices, evaluated on
#: the candidate holding, must reach this value. 0.35 keeps a holding the
#: fit would continue about one time in three per faced bet — genuine
#: narrowing toward the continuing range without starving multi-continue
#: prefixes (three continues need per-decision continue probability
#: >= 0.70, which strong holdings clear). Ablatable: ``--p3-accept-threshold``
#: (0 accepts the first draw — a pure uniform resample; near 1 rejects
#: almost everything and the reported fallback rate approaches 1).
DEFAULT_ACCEPT_THRESHOLD = 0.35

#: Draws before the uniform fallback. Ablatable: ``--p3-resample-tries``.
DEFAULT_RESAMPLE_TRIES = 40

#: E6 aggress branch specs, shared with the serve path.
_E6_BRANCH_SPECS: dict[str, tuple[float, float]] = {
    "aggress_small": _BRANCH_SMALL,
    "aggress_large": _BRANCH_LARGE,
}

#: Dedup precedence for the E6 branch set. Semantics inherited from the
#: stock ``_BRANCH_DEDUP_ORDER``: when two candidates execute the same
#: action the surviving label is the one naming it honestly — a fold the
#: engine turns into a free check is a check, and a merged aggression size
#: is the smaller wager's. The mirror collision — a forced ``check_call``
#: the engine's hard gates veto into a fold — also keeps the
#: ``check_call`` label, deliberately: in the v8 composition ``V(fold)``
#: is a constant 0 with no learnable parameters, while ``check_call`` is
#: a slot the serve-time argmax can always choose, and in a gate-veto
#: state choosing it *realizes* the fold outcome (the same gates fire at
#: serve). The label therefore teaches the choosable slot its realized
#: consequence; the row's ``executed`` field records the veto verbatim.
_PHASE_B_DEDUP_ORDER = ("check_call", "fold", "aggress_small", "aggress_large")

_CENTERING_TOLERANCE_BB = 1e-6


class PhaseBError(RuntimeError):
    """Raised when the pipeline or a corpus violates its own contract."""


# ---------------------------------------------------------------------------
# Seat wrappers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class P3PrefixRecord:
    """One faced-price decision a P3 seat committed to in the hand prefix.

    ``decision`` is the full fitted feature row built on the seat's actual
    holding; every field except ``strength_percentile`` is hole-independent,
    so evaluating a candidate holding only replaces the strength (recomputed
    on ``board``, the decision-time board).
    """

    decision: P3Decision
    board: tuple[str, ...]
    folded: bool


class P3SeatWrapper:
    """Transparent recorder around a :class:`StrengthAwareAgent` seat.

    Delegates every decision unchanged and records the seat's faced-price
    decisions for the current table, which is exactly the prefix constraint
    set the conditional resample needs: inside a counterfactual replay the
    (deep-copied) wrapper has, at the moment of the hole swap, recorded
    precisely the prefix decisions and nothing else. Consumes no RNG and
    exposes no ``last_diagnostics``, so the simulator never records
    examples for it; ``reads_cards`` stays ``True`` so the stock salt keeps
    its hands off the seat's holes (the swap machinery owns them instead).
    """

    reads_cards = True
    policy_version = "strength-aware"

    def __init__(self, agent: StrengthAwareAgent) -> None:
        self.agent = agent
        # Resolve the fit eagerly so replays never hit the loader.
        self.agent.resolved_fit()
        self._table_id: str | None = None
        self.prefix: list[P3PrefixRecord] = []
        self.record_failures = 0

    def records_for(self, table_id: str) -> list[P3PrefixRecord]:
        if table_id != self._table_id:
            return []
        return list(self.prefix)

    def decide(self, table: Mapping[str, Any]) -> dict:
        table_id = str(table.get("tableId"))
        if table_id != self._table_id:
            self._table_id = table_id
            self.prefix = []
        payload = self.agent.decide(table)
        allowed = table.get("allowedActions") or {}
        to_call = int(allowed.get("callChips") or 0)
        if to_call > 0:
            hole = hero_hole_cards(table)
            if hole is None:
                self.record_failures += 1
            else:
                try:
                    decision = decision_features(table, allowed, hole)
                except Exception:
                    self.record_failures += 1
                else:
                    self.prefix.append(
                        P3PrefixRecord(
                            decision=decision,
                            board=tuple(table.get("boardCards") or ()),
                            folded=str(payload.get("action")) == "fold",
                        )
                    )
        return payload


class HeroRecorder(RecordingPolicy):
    """RecordingPolicy that also captures each decision's snapshot.

    The counterfactual pass needs the branch-point snapshot (schema-3
    feature extraction, E6 sizing context) but ``_CounterfactualPoint``
    does not carry it; snapshots are fresh dicts per decision and are never
    mutated afterwards, so keeping references is safe. Only the current
    table's captures are retained.
    """

    def __init__(self, policy: Any) -> None:
        super().__init__(policy, record_examples=True)
        self._table_id: str | None = None
        self._ordinal = 0
        self._captures: dict[int, Mapping[str, Any]] = {}

    def decide(self, table: Mapping[str, Any]) -> dict:
        table_id = str(table.get("tableId"))
        if table_id != self._table_id:
            self._table_id = table_id
            self._ordinal = 0
            self._captures = {}
        self._captures[self._ordinal] = table
        self._ordinal += 1
        return super().decide(table)

    def capture_for(self, table_id: str, ordinal: int) -> Mapping[str, Any]:
        if table_id != self._table_id or ordinal not in self._captures:
            raise PhaseBError(
                f"no captured snapshot for {table_id!r} ordinal {ordinal}"
            )
        return self._captures[ordinal]


# ---------------------------------------------------------------------------
# The conditional hole swap
# ---------------------------------------------------------------------------


@dataclass
class P3SwapStats:
    """Counters for the conditional resample, shared across replays."""

    seats_resampled: int = 0
    tries: int = 0
    accepted: int = 0
    fallbacks: int = 0
    swaps_applied: int = 0
    record_likelihoods: bool = False
    accepted_likelihoods: list[float] = field(default_factory=list)

    def snapshot(self) -> tuple[int, int, int, int, int]:
        return (
            self.seats_resampled,
            self.tries,
            self.accepted,
            self.fallbacks,
            self.swaps_applied,
        )


def _joint_plausibility(
    agent: StrengthAwareAgent,
    records: Sequence[P3PrefixRecord],
    hole: Sequence[str],
) -> float:
    """Joint fitted probability of the observed prefix choices on ``hole``."""

    joint = 1.0
    key = tuple(sorted(hole))
    for record in records:
        strength = _cached_strength(key, record.board)
        decision = dataclasses.replace(
            record.decision, strength_percentile=float(strength)
        )
        p_fold = agent.fold_probability_for(decision)
        joint *= p_fold if record.folded else (1.0 - p_fold)
    return joint


@dataclass
class P3HoleSwap:
    """One rollout's pending conditional resample of every P3 seat's holes.

    ``pool`` is the unseen deck for this rollout — every card not held by
    the hero, a card-blind seat, or the board, which by construction
    includes the P3 seats' original holes (unseen to hero, so legitimate
    candidates). Applying the swap draws per seat, in seat order, without
    replacement, so no rollout can double-deal a card.
    """

    table_id: str
    pool: list[str]
    p3_agent_ids: frozenset[str]
    rng_key: str
    threshold: float
    max_tries: int
    stats: P3SwapStats
    armed: bool = False
    done: bool = False

    def apply(self, seats: Sequence[Any]) -> None:
        self.done = True
        rng = random.Random(self.rng_key)
        pool = list(self.pool)
        for seat in seats:
            if seat.agent_id not in self.p3_agent_ids:
                continue
            wrapper = seat.agent
            if not isinstance(wrapper, P3SeatWrapper):
                raise PhaseBError(
                    f"P3 swap targeted seat {seat.agent_id!r} whose agent is "
                    f"{type(wrapper).__name__}, not P3SeatWrapper"
                )
            records = wrapper.records_for(self.table_id)
            chosen: list[str] | None = None
            for _ in range(self.max_tries):
                sample = rng.sample(pool, 2)
                self.stats.tries += 1
                joint = _joint_plausibility(wrapper.agent, records, sample)
                if joint >= self.threshold:
                    chosen = sample
                    self.stats.accepted += 1
                    if self.stats.record_likelihoods:
                        self.stats.accepted_likelihoods.append(joint)
                    break
            if chosen is None:
                chosen = rng.sample(pool, 2)
                self.stats.fallbacks += 1
            self.stats.seats_resampled += 1
            pool.remove(chosen[0])
            pool.remove(chosen[1])
            seat.hole_cards = (chosen[0], chosen[1])
        self.stats.swaps_applied += 1


class PhaseBReplaySimulator(TableSimulator):
    """Replay simulator that installs the conditional P3 holes.

    Driven entirely by ``deck_for_test`` (the pre-computed arrangement), so
    the stock chance salt never runs here. The swap fires at the first
    event that can only belong to the continuation: the first snapshot
    built with a ``continuation_rollout``, or settlement when the forced
    action ended all decisions. The branch probe unwinds before either, so
    probes never consume swap randomness or stats.
    """

    def __init__(self, *, swap: P3HoleSwap | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._swap = swap

    def _decide(self, seat: Any, snapshot: dict, result: MatchResult) -> dict:
        payload = super()._decide(seat, snapshot, result)
        if self._swap is not None and payload.get("_counterfactual_rollout") is not None:
            self._swap.armed = True
        return payload

    def _snapshot(
        self,
        actor: Any,
        seats: list,
        street: str,
        board: list,
        current_bet: int,
        last_raise: int,
        table_id: str,
        hand_index: int,
        events: list,
        continuation_rollout: int | None = None,
    ) -> dict:
        if (
            continuation_rollout is not None
            and self._swap is not None
            and not self._swap.done
        ):
            self._swap.apply(seats)
        return super()._snapshot(
            actor,
            seats,
            street,
            board,
            current_bet,
            last_raise,
            table_id,
            hand_index,
            events,
            continuation_rollout,
        )

    def _settle(self, seats: list, board: list, events: list):
        if self._swap is not None and self._swap.armed and not self._swap.done:
            self._swap.apply(seats)
        return super()._settle(seats, board, events)


# ---------------------------------------------------------------------------
# The harvest simulator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Arrangement:
    """One rollout's full card layout, replicated from the stock salt."""

    deck_for_test: tuple[str, ...]
    holes: Mapping[str, tuple[str, str]]
    board: tuple[str, ...]
    unseen: tuple[str, ...]


class PhaseBHarvestSimulator(TableSimulator):
    """Harvest simulator emitting Phase-B decision rows.

    Reuses the stock counterfactual machinery — point collection, one
    selected decision per actor and hand (same selection RNG),
    ``inclusion_count``, the branch probe, absorption by distinct executed
    action — and replaces exactly three things: the branch candidates (E6,
    state-dependent), the rollout replay (arranged deck + conditional P3
    hole swap), and the emitted rows (schema-3 features, per-decision
    grouping, into ``phase_b_rows`` instead of ``MatchResult.examples``).
    """

    #: Overridable in tests to observe the replay (e.g. recording each
    #: seat's holes at settlement).
    replay_class: type[PhaseBReplaySimulator] = PhaseBReplaySimulator

    def __init__(
        self,
        *,
        hero_id: str = "hero",
        hero_recorder: HeroRecorder | None = None,
        leg_name: str = "",
        accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
        resample_tries: int = DEFAULT_RESAMPLE_TRIES,
        potential_trials: int = 400,
        feature_seed: int = 7,
        force_arranged: bool = False,
        record_likelihoods: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= accept_threshold <= 1.0:
            raise PhaseBError("accept_threshold must be within [0, 1]")
        if resample_tries < 1:
            raise PhaseBError("resample_tries must be positive")
        self.hero_id = hero_id
        self.hero_recorder = hero_recorder
        self.leg_name = leg_name
        self.accept_threshold = float(accept_threshold)
        self.resample_tries = int(resample_tries)
        self.potential_trials = int(potential_trials)
        self.feature_seed = int(feature_seed)
        self.force_arranged = force_arranged
        self.phase_b_rows: list[dict[str, Any]] = []
        self.p3_stats = P3SwapStats(record_likelihoods=record_likelihoods)
        self.decisions_emitted = 0
        self.decisions_selected = 0

    # -- arrangement ---------------------------------------------------

    def _p3_opponent_ids(
        self, initial_seats: Sequence[Any], owner: str
    ) -> frozenset[str]:
        ids = set()
        for seat in initial_seats:
            if seat.agent_id == owner:
                continue
            if isinstance(seat.agent, P3SeatWrapper):
                ids.add(seat.agent_id)
            elif getattr(seat.agent, "reads_cards", True) is not False:
                # A card-aware opponent this pipeline has no conditional
                # sampler for would silently reintroduce the frozen-holes
                # defect; refuse rather than mislabel.
                raise PhaseBError(
                    f"seat {seat.agent_id!r} reads cards but is not a "
                    "P3SeatWrapper; Phase B cannot salt it honestly"
                )
        return frozenset(ids)

    def _arranged(
        self,
        initial_seats: Sequence[Any],
        hand_index: int,
        point: Any,
        rollout: int,
    ) -> _Arrangement:
        """Replicate the deal plus the stock chance salt for one rollout.

        Bit-for-bit the arithmetic of ``TableSimulator._play_hand``: the
        deal RNG is ``Random(f"{seed}:{hand_index}")``, the salt RNG
        ``Random(f"{seed}:{hand_index}:chance:{rollout}")``, the salt pool
        is future board + card-blind non-owner holes + remaining deck, and
        reassignment order matches. Verified two ways: per decision against
        the captured branch-point snapshot (hero holes and revealed board
        must match), and by the parity test that plays arranged replays
        against stock salted replays with no P3 seat and demands identical
        outcomes.
        """

        rng = random.Random(f"{self.seed}:{hand_index}")
        deck = list(_DECK)
        rng.shuffle(deck)
        holes: dict[str, tuple[str, str]] = {}
        for seat in initial_seats:
            holes[seat.agent_id] = (deck.pop(0), deck.pop(0))
        board = [deck.pop(0) for _ in range(5)]

        revealed = _REVEALED_BOARD.get(point.street, 0)
        salt_rng = random.Random(f"{self.seed}:{hand_index}:chance:{rollout}")
        pool = board[revealed:]
        resampled = [
            seat
            for seat in initial_seats
            if seat.agent_id != point.agent_id
            and getattr(seat.agent, "reads_cards", True) is False
        ]
        for seat in resampled:
            pool.extend(holes[seat.agent_id])
        pool.extend(deck)
        salt_rng.shuffle(pool)
        for seat in resampled:
            holes[seat.agent_id] = (pool.pop(0), pool.pop(0))
        board[revealed:] = [pool.pop(0) for _ in range(5 - revealed)]
        deck = pool

        assigned = set(board)
        assigned.update(holes[point.agent_id])
        for seat in resampled:
            assigned.update(holes[seat.agent_id])
        unseen = tuple(card for card in _DECK if card not in assigned)

        deck_for_test = tuple(
            card for seat in initial_seats for card in holes[seat.agent_id]
        ) + tuple(board) + tuple(deck)
        return _Arrangement(
            deck_for_test=deck_for_test,
            holes=holes,
            board=tuple(board),
            unseen=unseen,
        )

    def _verify_arrangement(
        self,
        arrangement: _Arrangement,
        point: Any,
        capture: Mapping[str, Any],
    ) -> None:
        """The branch-point snapshot must agree with the replicated deal."""

        hero_seat = next(
            seat
            for seat in capture["seats"]
            if seat["seatNumber"] == capture["selfSeatNumber"]
        )
        expected_holes = list(arrangement.holes[point.agent_id])
        if list(hero_seat["holeCards"] or ()) != expected_holes:
            raise PhaseBError(
                "replicated deal disagrees with the captured snapshot's hero "
                f"holes: {hero_seat['holeCards']!r} != {expected_holes!r}"
            )
        revealed = _REVEALED_BOARD.get(point.street, 0)
        if tuple(capture.get("boardCards") or ()) != arrangement.board[:revealed]:
            raise PhaseBError(
                "replicated deal disagrees with the captured snapshot's "
                f"board: {capture.get('boardCards')!r} != "
                f"{arrangement.board[:revealed]!r}"
            )

    # -- replays -------------------------------------------------------

    def _counterfactual_outcome(
        self,
        initial_seats: list,
        button_index: int,
        hand_index: int,
        point: Any,
        family: str,
        pot_fraction: float | None,
        rollout: int,
        deck_for_test: Sequence[str] | None,
    ) -> tuple[int, float, tuple[str, int | None] | None]:
        if deck_for_test is not None:
            # A pinned test deck freezes the stock salt too; nothing to do.
            return super()._counterfactual_outcome(
                initial_seats,
                button_index,
                hand_index,
                point,
                family,
                pot_fraction,
                rollout,
                deck_for_test,
            )
        p3_ids = self._p3_opponent_ids(initial_seats, point.agent_id)
        if not p3_ids and not self.force_arranged:
            return super()._counterfactual_outcome(
                initial_seats,
                button_index,
                hand_index,
                point,
                family,
                pot_fraction,
                rollout,
                deck_for_test,
            )
        arrangement = self._arranged(initial_seats, hand_index, point, rollout)
        seats = copy.deepcopy(initial_seats)
        target = next(seat for seat in seats if seat.agent_id == point.agent_id)
        forced = _ForcedFamilyPolicy(
            target.agent,
            decision_ordinal=point.decision_ordinal,
            family=family,
            risk_fraction=point.proposed_risk_fraction,
            rollout=rollout,
            pot_fraction=pot_fraction,
        )
        target.agent = forced
        swap = None
        if p3_ids:
            swap = P3HoleSwap(
                table_id=f"sim-{self.seed}-{hand_index}",
                pool=list(arrangement.unseen),
                p3_agent_ids=p3_ids,
                rng_key=f"{self.seed}:{hand_index}:p3holes:{rollout}",
                threshold=self.accept_threshold,
                max_tries=self.resample_tries,
                stats=self.p3_stats,
            )
        replay = self.replay_class(
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            starting_stack=self.starting_stack,
            seed=self.seed,
            swap=swap,
        )
        result = MatchResult(
            hands=0,
            big_blind=self.big_blind,
            chip_deltas={seat.agent_id: 0 for seat in seats},
            decisions={seat.agent_id: 0 for seat in seats},
        )
        hand = replay._play_hand(
            seats,
            button_index=button_index,
            hand_index=hand_index,
            result=result,
            deck_for_test=list(arrangement.deck_for_test),
            chance_salt=None,
        )
        if not forced.forced or forced.submitted_risk_fraction is None:
            raise SimulationError(
                "counterfactual replay did not reach its target decision"
            )
        return (
            hand.chip_deltas[point.agent_id],
            forced.submitted_risk_fraction,
            forced.executed_action,
        )

    # -- branch set ----------------------------------------------------

    def _branch_candidates(
        self, point: Any, context: Mapping[str, Any]
    ) -> list[tuple[str, str, float | None]]:
        """The E6 candidate set for one decision, sized from its snapshot."""

        candidates: list[tuple[str, str, float | None]] = []
        for family in point.legal_families:
            if family == "fold":
                candidates.append(("fold", "fold", None))
            elif family == "check_call":
                candidates.append(("check_call", "check_call", None))
            elif family == "aggress":
                if context["legal_range"] is None:
                    # Serve-path emission rule: no stated bet/raise range,
                    # no aggress branch (all-in-only states are not sized
                    # branches the composed argmax can choose).
                    continue
                pot = context["pot"]
                to_call = context["to_call"]
                eff = max(1, context["effective_stack"])
                for label in ("aggress_small", "aggress_large"):
                    pot_fraction, stack_fraction = _E6_BRANCH_SPECS[label]
                    target = min(
                        to_call + pot_fraction * (pot + to_call),
                        stack_fraction * eff,
                    )
                    fraction = (target - to_call) / max(1, pot + to_call)
                    candidates.append((label, "aggress", fraction))
        return candidates

    def _emitted_branches_v8(
        self,
        initial_seats: list,
        button_index: int,
        hand_index: int,
        point: Any,
        candidates: Sequence[tuple[str, str, float | None]],
        deck_for_test: Sequence[str] | None,
    ) -> tuple[
        list[tuple[str, str, float | None]],
        dict[str, str],
        dict[str, tuple[str, int | None]],
    ]:
        """Stock dedup-by-executed-action, ranked for the E6 labels."""

        executed = self._probe_branch_set(
            initial_seats, button_index, hand_index, point, candidates, deck_for_test
        )
        rank = {label: index for index, label in enumerate(_PHASE_B_DEDUP_ORDER)}
        survivors: dict[tuple[str, int | None], str] = {}
        for label, _, _ in sorted(candidates, key=lambda entry: rank[entry[0]]):
            survivors.setdefault(executed[label], label)
        absorption = {label: survivors[executed[label]] for label, _, _ in candidates}
        keep = set(survivors.values())
        branches = [entry for entry in candidates if entry[0] in keep]
        self.emitted_branch_counts[len(branches)] = (
            self.emitted_branch_counts.get(len(branches), 0) + 1
        )
        return branches, absorption, executed

    # -- rows ----------------------------------------------------------

    @staticmethod
    def _decision_context(capture: Mapping[str, Any]) -> dict[str, Any]:
        allowed = capture["allowedActions"]
        available = {str(value) for value in allowed["availableActions"]}
        hero = next(
            seat
            for seat in capture["seats"]
            if seat["seatNumber"] == capture["selfSeatNumber"]
        )
        contribution = int(hero["currentBetChips"])
        stack = int(hero["stackChips"])
        total_committed = int(hero.get("totalCommittedChips", contribution))
        return {
            "pot": int(capture["potChips"]),
            "to_call": int(allowed.get("callChips") or 0),
            "contribution": contribution,
            "effective_stack": int(effective_stack_chips(capture)),
            "purse": stack + total_committed,
            "legal_range": _aggressive_range(allowed, available),
            "big_blind": int(capture["bigBlindChips"]),
        }

    def _counterfactual_examples(
        self,
        initial_seats: list,
        button_index: int,
        hand_index: int,
        points: list,
        deck_for_test: Sequence[str] | None,
    ) -> list:
        """Emit Phase-B decision rows; returns no v7 TrainingExamples."""

        selected = []
        for agent_id in sorted({point.agent_id for point in points}):
            choices = [point for point in points if point.agent_id == agent_id]
            selected.append(
                random.Random(
                    f"{self.seed}:{hand_index}:{agent_id}:counterfactual"
                ).choice(choices)
            )
        inclusion_counts = {
            point.agent_id: len(
                [entry for entry in points if entry.agent_id == point.agent_id]
            )
            for point in selected
        }
        table_id = f"sim-{self.seed}-{hand_index}"
        for point in selected:
            self.decisions_selected += 1
            if point.agent_id != self.hero_id or self.hero_recorder is None:
                raise PhaseBError(
                    f"counterfactual point for unexpected actor "
                    f"{point.agent_id!r}; only the hero seat is recorded"
                )
            capture = self.hero_recorder.capture_for(
                table_id, point.decision_ordinal
            )
            if deck_for_test is None:
                self._verify_arrangement(
                    self._arranged(initial_seats, hand_index, point, 0),
                    point,
                    capture,
                )
            context = self._decision_context(capture)
            candidates = self._branch_candidates(point, context)
            if len(candidates) < 2:
                self.single_branch_groups += 1
                continue
            branches, absorption, executed = self._emitted_branches_v8(
                initial_seats,
                button_index,
                hand_index,
                point,
                candidates,
                deck_for_test,
            )
            if len(branches) < 2:
                self.single_branch_groups += 1
                continue
            stats_before = self.p3_stats.snapshot()
            outcomes: dict[str, float] = {}
            risks: dict[str, float] = {}
            for label, family, pot_fraction in branches:
                samples = [
                    self._counterfactual_outcome(
                        initial_seats,
                        button_index,
                        hand_index,
                        point,
                        family,
                        pot_fraction,
                        rollout,
                        deck_for_test,
                    )
                    for rollout in range(self.counterfactual_rollouts)
                ]
                outcomes[label] = sum(sample[0] for sample in samples) / len(samples)
                risks[label] = sum(sample[1] for sample in samples) / len(samples)
            baseline = sum(outcomes.values()) / len(outcomes)
            rewards = {
                label: (outcomes[label] - baseline) / self.big_blind
                for label in outcomes
            }
            if not all(math.isfinite(value) for value in rewards.values()):
                raise PhaseBError(f"non-finite branch reward at {table_id}")
            if abs(sum(rewards.values())) > _CENTERING_TOLERANCE_BB:
                raise PhaseBError(
                    f"centered rewards do not cancel at {table_id}: "
                    f"{rewards!r}"
                )
            features = extract_features_v8(
                capture,
                equity=None,
                belief_provider=None,
                potential_trials=self.potential_trials,
                seed=self.feature_seed,
            )
            seats_delta = tuple(
                after - before
                for after, before in zip(self.p3_stats.snapshot(), stats_before)
            )
            eff = max(1, context["effective_stack"])
            branch_rows = []
            for label, family, pot_fraction in branches:
                entry: dict[str, Any] = {
                    "branch": label,
                    "family": family,
                    "pot_fraction": pot_fraction,
                    "reward_bb": rewards[label],
                    "outcome_bb": outcomes[label] / self.big_blind,
                    "risk_fraction": risks[label],
                    "executed": [executed[label][0], executed[label][1]],
                }
                if family == "aggress":
                    spec = _E6_BRANCH_SPECS[label]
                    target = min(
                        context["to_call"] + spec[0] * (context["pot"] + context["to_call"]),
                        spec[1] * eff,
                    )
                    low, high = context["legal_range"]
                    entry["e6_target"] = target
                    entry["e6_to_amount"] = min(
                        high, max(low, context["contribution"] + target)
                    )
                branch_rows.append(entry)
            self.phase_b_rows.append(
                {
                    "decision_id": (
                        f"{table_id}:{point.agent_id}:{point.decision_ordinal}"
                    ),
                    "table_id": table_id,
                    "harvest_leg": self.leg_name,
                    "policy_version": point.example.policy_version,
                    "street": point.street,
                    "big_blind": self.big_blind,
                    "purse_bb": context["purse"] / self.big_blind,
                    "inclusion_count": inclusion_counts[point.agent_id],
                    "rollouts": self.counterfactual_rollouts,
                    "context": {
                        "pot": context["pot"],
                        "to_call": context["to_call"],
                        "contribution": context["contribution"],
                        "effective_stack": context["effective_stack"],
                        "purse": context["purse"],
                        "legal_range": (
                            None
                            if context["legal_range"] is None
                            else list(context["legal_range"])
                        ),
                    },
                    "features": [float(value) for value in features],
                    "branches": branch_rows,
                    "branch_absorption": dict(sorted(absorption.items())),
                    "p3": {
                        "seats_resampled": seats_delta[0],
                        "tries": seats_delta[1],
                        "accepted": seats_delta[2],
                        "fallbacks": seats_delta[3],
                        "swaps_applied": seats_delta[4],
                    },
                }
            )
            self.decisions_emitted += 1
        return []


# ---------------------------------------------------------------------------
# Corpus IO and validation
# ---------------------------------------------------------------------------


def corpus_header() -> dict[str, Any]:
    return {
        "kind": CORPUS_KIND,
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "feature_schema_version": schema3.SCHEMA_VERSION,
        "input_size": schema3.INPUT_SIZE_V8,
        "branch_labels": list(BRANCH_LABELS_V8),
    }


def write_phase_b_corpus(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "wb") as raw:
        # mtime pinned to zero so byte-identical reruns are byte-identical
        # outputs (the Phase-A convention).
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(
                json.dumps(
                    corpus_header(), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                + b"\n"
            )
            for row in rows:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, allow_nan=False, sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )


def load_phase_b_corpus(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with gzip.open(resolved, "rt", encoding="utf-8") as handle:
        header_line = handle.readline()
        try:
            header = json.loads(header_line)
        except json.JSONDecodeError as error:
            raise PhaseBError(f"corpus header is not JSON: {error}") from error
        if not isinstance(header, Mapping) or header.get("kind") != CORPUS_KIND:
            raise PhaseBError("corpus header kind is not phase-b-corpus")
        if header.get("corpus_schema_version") != CORPUS_SCHEMA_VERSION:
            raise PhaseBError("unsupported phase-b corpus schema version")
        if header.get("feature_schema_version") != schema3.SCHEMA_VERSION:
            raise PhaseBError("corpus feature schema is not schema 3")
        if header.get("input_size") != schema3.INPUT_SIZE_V8:
            raise PhaseBError("corpus input size does not match schema 3")
        for line_number, line in enumerate(handle, 2):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise PhaseBError(
                    f"corpus line {line_number} is not JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise PhaseBError(f"corpus line {line_number} is not an object")
            rows.append(row)
    return rows


def validate_phase_b_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Contract checks over decision rows; returns aggregate statistics.

    Enforced per decision: schema-3 vector shape and finiteness, at least
    two emitted branches, finite labels, centered targets summing to ~0
    within the decision, a coherent absorption map (every candidate maps to
    an emitted survivor, survivors map to themselves), positive purse,
    known street, unique decision ids, positive inclusion counts.
    """

    if not rows:
        raise PhaseBError("corpus has no rows")
    streets = ("preflop", "flop", "turn", "river")
    seen_ids: set[str] = set()
    branch_rows = 0
    per_street: dict[str, int] = {}
    per_leg: dict[str, int] = {}
    branch_label_counts: dict[str, int] = {}
    emitted_sizes: dict[int, int] = {}
    p3_totals = {"seats_resampled": 0, "tries": 0, "accepted": 0, "fallbacks": 0}
    for index, row in enumerate(rows):
        name = f"row {index} ({row.get('decision_id')!r})"
        decision_id = row.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            raise PhaseBError(f"{name}: missing decision_id")
        if decision_id in seen_ids:
            raise PhaseBError(f"{name}: duplicate decision_id")
        seen_ids.add(decision_id)
        street = row.get("street")
        if street not in streets:
            raise PhaseBError(f"{name}: unknown street {street!r}")
        features = row.get("features")
        if not isinstance(features, list):
            raise PhaseBError(f"{name}: features must be a list")
        vector = tuple(float(value) for value in features)
        if not all(math.isfinite(value) for value in vector):
            raise PhaseBError(f"{name}: non-finite feature")
        schema3.require_vector(vector)
        purse_bb = row.get("purse_bb")
        if (
            not isinstance(purse_bb, (int, float))
            or isinstance(purse_bb, bool)
            or not math.isfinite(float(purse_bb))
            or float(purse_bb) <= 0.0
        ):
            raise PhaseBError(f"{name}: purse_bb must be positive and finite")
        inclusion = row.get("inclusion_count")
        if not isinstance(inclusion, int) or isinstance(inclusion, bool) or inclusion < 1:
            raise PhaseBError(f"{name}: inclusion_count must be a positive int")
        branches = row.get("branches")
        if not isinstance(branches, list) or len(branches) < 2:
            raise PhaseBError(f"{name}: needs at least two emitted branches")
        labels = [entry.get("branch") for entry in branches]
        if len(set(labels)) != len(labels):
            raise PhaseBError(f"{name}: duplicate branch labels")
        for label in labels:
            if label not in BRANCH_LABELS_V8:
                raise PhaseBError(f"{name}: unknown branch label {label!r}")
        executed_pairs = [tuple(entry.get("executed") or ()) for entry in branches]
        if len(set(executed_pairs)) != len(executed_pairs):
            # Emission is by distinct executed action; two emitted branches
            # sharing one action means the dedup did not run.
            raise PhaseBError(f"{name}: emitted branches share an executed action")
        total = 0.0
        for entry in branches:
            reward = entry.get("reward_bb")
            if (
                not isinstance(reward, (int, float))
                or isinstance(reward, bool)
                or not math.isfinite(float(reward))
            ):
                raise PhaseBError(f"{name}: non-finite reward_bb")
            total += float(reward)
            risk = entry.get("risk_fraction")
            if (
                not isinstance(risk, (int, float))
                or isinstance(risk, bool)
                or not math.isfinite(float(risk))
                or not 0.0 <= float(risk) <= 1.0
            ):
                raise PhaseBError(f"{name}: risk_fraction outside [0, 1]")
            branch_label_counts[str(entry.get("branch"))] = (
                branch_label_counts.get(str(entry.get("branch")), 0) + 1
            )
        if abs(total) > 1e-4:
            raise PhaseBError(
                f"{name}: centered rewards sum to {total!r}, not ~0"
            )
        absorption = row.get("branch_absorption")
        if not isinstance(absorption, Mapping) or not absorption:
            raise PhaseBError(f"{name}: missing branch_absorption")
        for source, survivor in absorption.items():
            if survivor not in labels:
                raise PhaseBError(
                    f"{name}: {source!r} absorbed into non-emitted {survivor!r}"
                )
        for label in labels:
            if absorption.get(label) != label:
                raise PhaseBError(
                    f"{name}: emitted branch {label!r} does not survive its "
                    "own absorption entry"
                )
        branch_rows += len(branches)
        per_street[street] = per_street.get(street, 0) + len(branches)
        leg = str(row.get("harvest_leg") or "")
        per_leg[leg] = per_leg.get(leg, 0) + len(branches)
        emitted_sizes[len(branches)] = emitted_sizes.get(len(branches), 0) + 1
        p3 = row.get("p3") or {}
        for key in p3_totals:
            value = p3.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                p3_totals[key] += value
    resamples = p3_totals["seats_resampled"]
    return {
        "decisions": len(rows),
        "branch_rows": branch_rows,
        "branch_rows_per_street": dict(sorted(per_street.items())),
        "branch_rows_per_leg": dict(sorted(per_leg.items())),
        "branch_label_counts": dict(sorted(branch_label_counts.items())),
        "emitted_branch_sizes": {
            str(key): value for key, value in sorted(emitted_sizes.items())
        },
        "p3_resample_totals": p3_totals,
        "p3_fallback_rate": (
            p3_totals["fallbacks"] / resamples if resamples else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Harvest legs
# ---------------------------------------------------------------------------

#: (kind, aggression, fold_vs_bet, shove_rate). ``p3`` seats take the
#: audited scripted knobs of ``strength_aware_lineup``; only the folding is
#: fitted. Card-blind seats reuse the calibrated archetypes.
#:
#: NO LEARNED SEAT BELONGS HERE, and one was tried and reverted on
#: 2026-09-02. The 2026-09-02 diagnosis (section 4) is right that the
#: roster carries no v6/v7/v8/v9 policy and that this makes the batteries
#: in-distribution — but the fix does not live in the harvest. Phase B's
#: counterfactual replay resamples every card-reading opponent's holes
#: CONDITIONAL on the prefix of decisions that opponent already made;
#: ``P3SeatWrapper`` exists to record that prefix, and
#: ``_p3_opponent_ids`` refuses any other card-reading seat rather than
#: reintroduce the frozen-holes defect. A learned policy would need a
#: conditional hole sampler of its own — i.e. inverting the network —
#: which is a research problem, not a seat entry. The reverted attempt
#: justified skipping the wrapper with "an engine policy reads its own
#: hole cards so the chance salt leaves it alone anyway"; the salt
#: leaving it alone is precisely the defect the guard names.
#:
#: Measuring against the real incumbent belongs in EVALUATION, where the
#: gauntlet already duels ``candidate-v7-0001c`` and no hole resampling
#: is involved.
_OPPONENT_KINDS: dict[str, tuple[str, float, float, float]] = {
    "p3-median": ("p3", 0.226, 0.5, 0.0),
    "p3-passive": ("p3", 0.10, 0.5, 0.0),
    "p3-aggressive": ("p3", 0.35, 0.5, 0.02),
    "median-bot": ("scripted", 0.226, 0.5, 0.0),
    "station-bot": ("scripted", 0.15, 0.05, 0.0),
    "tight-bot": ("scripted", 0.10, 0.75, 0.0),
    "wild-bot": ("scripted", 0.35, 0.30, 0.02),
    "textured-bot": ("textured", 0.226, 0.5, 0.0),
}


@dataclass(frozen=True, slots=True)
class LegSpec:
    """One harvest leg, described by value so worker processes can build it."""

    name: str
    opponents: tuple[str, ...]
    hands: int
    seed: int
    session_hands: int
    candidate: str
    equity_trials: int
    potential_trials: int
    feature_seed: int
    counterfactual_rollouts: int
    accept_threshold: float
    resample_tries: int
    starting_stack: int = 6_000
    small_blind: int = 50
    big_blind: int = 100
    # v9-only (pre-harvest supplemental harvests): the postflop selection
    # mode and its per-leg street quotas. The v8 harvester never reads
    # these; defaults preserve its bytes.
    postflop_selection: bool = False
    street_quotas: dict[str, int] | None = None


def default_leg_specs(args: argparse.Namespace) -> list[LegSpec]:
    """The fixed leg set: every lineup includes at least one P3 seat."""

    common = {
        "session_hands": args.session_hands,
        "candidate": args.candidate,
        "equity_trials": args.equity_trials,
        "potential_trials": args.potential_trials,
        "feature_seed": args.feature_seed,
        "counterfactual_rollouts": args.counterfactual_rollouts,
        "accept_threshold": args.p3_accept_threshold,
        "resample_tries": args.p3_resample_tries,
        "starting_stack": args.starting_stack,
    }
    scale = args.hands_scale

    def hands(value: int) -> int:
        return max(0, round(value * scale))

    legs = [
        LegSpec(
            name="five-max-p3-a",
            opponents=("p3-median", "p3-passive", "p3-aggressive", "median-bot"),
            hands=hands(args.five_max_hands),
            seed=args.seed,
            **common,
        ),
        LegSpec(
            name="five-max-p3-b",
            opponents=("p3-median", "p3-passive", "p3-aggressive", "station-bot"),
            hands=hands(args.five_max_hands),
            seed=args.seed + 40,
            **common,
        ),
        LegSpec(
            name="hu-p3-median",
            opponents=("p3-median",),
            hands=hands(args.hu_hands),
            seed=args.seed + 41,
            **common,
        ),
        LegSpec(
            name="hu-p3-aggressive",
            opponents=("p3-aggressive",),
            hands=hands(args.hu_hands),
            seed=args.seed + 42,
            **common,
        ),
        LegSpec(
            name="mixed-p3-texture",
            opponents=("p3-median", "textured-bot", "station-bot"),
            hands=hands(args.mixed_hands),
            seed=args.seed + 43,
            **common,
        ),
        LegSpec(
            name="mixed-p3-wild",
            opponents=("p3-passive", "wild-bot", "tight-bot"),
            hands=hands(args.mixed_hands),
            seed=args.seed + 44,
            **common,
        ),
    ]
    return [leg for leg in legs if leg.hands > 0]


def _build_opponents(spec: LegSpec) -> list[tuple[str, Any]]:
    fit = load_fit()
    seats: list[tuple[str, Any]] = []
    for offset, label in enumerate(spec.opponents):
        kind, aggression, fold_vs_bet, shove_rate = _OPPONENT_KINDS[label]
        seed = spec.seed + 100 + offset
        if kind == "p3":
            agent: Any = P3SeatWrapper(
                StrengthAwareAgent(
                    label, aggression, fold_vs_bet, shove_rate, seed, fit=fit
                )
            )
        elif kind == "textured":
            agent = TexturedAgent(label, aggression, fold_vs_bet, shove_rate, seed)
        else:
            agent = ScriptedAgent(label, aggression, fold_vs_bet, shove_rate, seed)
        seats.append((label, agent))
    return seats


def _build_hero(spec: LegSpec) -> HeroRecorder:
    policy = load_policy_v8(
        spec.candidate,
        equity_trials=spec.equity_trials,
        equity_cache=SharedEquityCache(),
        # The anti-modeling roll is off for harvesting, and only for
        # harvesting, for the same measured reason as self_play_cycle: no
        # harvest opponent models us, so the noise is all cost.
        hyper_aggression_chance=0.0,
        potential_trials=spec.potential_trials,
        feature_seed=spec.feature_seed,
    )
    return HeroRecorder(policy)


def run_leg(spec: LegSpec) -> dict[str, Any]:
    """Run one leg's carry-over sessions; returns rows plus diagnostics.

    Sessions mirror ``run_sessions``: stacks carry within a session, a new
    session starts fresh policy instances on a derived seed, so results do
    not depend on scheduling. Each session is capped at ``session_hands``
    so a busted hero cannot burn a long tail of unrecorded hands.
    """

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    totals = {
        "hands": 0,
        "sessions": 0,
        "hero_decisions": 0,
        "decisions_selected": 0,
        "decisions_emitted": 0,
        "single_branch_groups": 0,
        "hero_chip_delta": 0,
        "hero_hands": 0,
    }
    emitted_branch_counts: dict[int, int] = {}
    p3_totals = {
        "seats_resampled": 0,
        "tries": 0,
        "accepted": 0,
        "fallbacks": 0,
        "swaps_applied": 0,
    }
    opponents = _build_opponents(spec)
    p3_wrappers = [agent for _, agent in opponents if isinstance(agent, P3SeatWrapper)]
    session = 0
    while totals["hands"] < spec.hands:
        chunk = min(spec.session_hands, spec.hands - totals["hands"])
        recorder = _build_hero(spec)
        simulator = PhaseBHarvestSimulator(
            small_blind=spec.small_blind,
            big_blind=spec.big_blind,
            starting_stack=spec.starting_stack,
            seed=spec.seed + 7_919 * session,
            collect_counterfactuals=True,
            counterfactual_rollouts=spec.counterfactual_rollouts,
            hero_id="hero",
            hero_recorder=recorder,
            leg_name=spec.name,
            accept_threshold=spec.accept_threshold,
            resample_tries=spec.resample_tries,
            potential_trials=spec.potential_trials,
            feature_seed=spec.feature_seed,
        )
        agents = [("hero", recorder)] + opponents
        result = simulator.play_match(agents, hands=chunk, reset_stacks=False)
        if result.hands == 0:
            break
        rows.extend(simulator.phase_b_rows)
        totals["hands"] += result.hands
        totals["sessions"] += 1
        totals["hero_decisions"] += result.decisions.get("hero", 0)
        totals["decisions_selected"] += simulator.decisions_selected
        totals["decisions_emitted"] += simulator.decisions_emitted
        totals["single_branch_groups"] += simulator.single_branch_groups
        totals["hero_chip_delta"] += result.chip_deltas.get("hero", 0)
        totals["hero_hands"] += result.hands_by_agent.get("hero", result.hands)
        for size, count in simulator.emitted_branch_counts.items():
            emitted_branch_counts[size] = emitted_branch_counts.get(size, 0) + count
        for key in p3_totals:
            p3_totals[key] += getattr(simulator.p3_stats, key)
        session += 1
    resamples = p3_totals["seats_resampled"]
    hero_bb_per_100 = (
        100.0 * totals["hero_chip_delta"] / (spec.big_blind * totals["hero_hands"])
        if totals["hero_hands"]
        else 0.0
    )
    return {
        "name": spec.name,
        "opponents": list(spec.opponents),
        "seed": spec.seed,
        "rows": rows,
        "hands": totals["hands"],
        "sessions": totals["sessions"],
        "hero_decisions": totals["hero_decisions"],
        "decisions_selected": totals["decisions_selected"],
        "decisions_emitted": totals["decisions_emitted"],
        "single_branch_groups": totals["single_branch_groups"],
        "branch_rows": sum(len(row["branches"]) for row in rows),
        "emitted_branch_counts": {
            str(key): value for key, value in sorted(emitted_branch_counts.items())
        },
        "hero_bb_per_100": round(hero_bb_per_100, 3),
        "p3_resample": dict(p3_totals),
        "p3_fallback_rate": (
            p3_totals["fallbacks"] / resamples if resamples else 0.0
        ),
        # A non-zero degrade count means a different opponent played than
        # the one the leg thinks it measured; must be zero.
        "p3_agent_fallbacks": sum(
            wrapper.agent.fallback_count for wrapper in p3_wrappers
        ),
        "p3_record_failures": sum(
            wrapper.record_failures for wrapper in p3_wrappers
        ),
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _harvest_workers(requested: int, legs: int) -> int:
    if requested == 1 or legs <= 1:
        return 1
    if requested > 1:
        return min(requested, legs)
    return max(1, min(legs, (os.cpu_count() or 2) - 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--candidate",
        default=str(DEFAULT_CANDIDATE),
        help="v8 candidate manifest used as the acting (hero) policy",
    )
    parser.add_argument("--corpus-name", default=DEFAULT_CORPUS_NAME)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory for <corpus-name>.phase-b.jsonl.gz and its summary",
    )
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--five-max-hands", type=int, default=1_600)
    parser.add_argument("--hu-hands", type=int, default=1_200)
    parser.add_argument("--mixed-hands", type=int, default=1_400)
    parser.add_argument(
        "--hands-scale",
        type=float,
        default=1.0,
        help="multiply every leg's hand count (budget knob)",
    )
    parser.add_argument("--session-hands", type=int, default=250)
    parser.add_argument("--equity-trials", type=int, default=80)
    parser.add_argument(
        "--potential-trials",
        type=int,
        default=400,
        help="hand_potential trials for schema-3 extraction (serve default)",
    )
    parser.add_argument("--feature-seed", type=int, default=7)
    parser.add_argument("--counterfactual-rollouts", type=int, default=2)
    parser.add_argument(
        "--p3-accept-threshold",
        type=float,
        default=DEFAULT_ACCEPT_THRESHOLD,
        help="joint prefix-consistency probability a sampled P3 holding "
        "must reach; 0 degrades to a uniform resample (ablation arm)",
    )
    parser.add_argument(
        "--p3-resample-tries",
        type=int,
        default=DEFAULT_RESAMPLE_TRIES,
        help="draws before the counted uniform fallback",
    )
    parser.add_argument("--starting-stack", type=int, default=6_000)
    parser.add_argument(
        "--harvest-workers",
        type=int,
        default=0,
        help="0 = one process per leg within the core count; 1 = sequential",
    )
    parser.add_argument(
        "--legs",
        default="",
        help="comma-separated leg-name substrings to keep (default: all)",
    )
    parser.add_argument(
        "--validate",
        metavar="CORPUS",
        help="validate an existing corpus file and print its statistics",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.validate:
        rows = load_phase_b_corpus(args.validate)
        report = validate_phase_b_rows(rows)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.counterfactual_rollouts < 1:
        parser.error("--counterfactual-rollouts must be positive")
    if not 0.0 <= args.p3_accept_threshold <= 1.0:
        parser.error("--p3-accept-threshold must be within [0, 1]")
    if args.p3_resample_tries < 1:
        parser.error("--p3-resample-tries must be positive")
    if args.session_hands < 1:
        parser.error("--session-hands must be positive")

    specs = default_leg_specs(args)
    if args.legs:
        wanted = [part.strip() for part in args.legs.split(",") if part.strip()]
        specs = [
            spec
            for spec in specs
            if any(fragment in spec.name for fragment in wanted)
        ]
    if not specs:
        parser.error("no legs selected")

    output_dir = Path(args.output_dir).expanduser().resolve()
    corpus_path = output_dir / f"{args.corpus_name}.phase-b.jsonl.gz"
    summary_path = output_dir / f"{args.corpus_name}.phase-b.summary.json"

    plan = {
        "candidate": args.candidate,
        "corpus": str(corpus_path),
        "summary": str(summary_path),
        "legs": [
            {"name": spec.name, "opponents": list(spec.opponents), "hands": spec.hands}
            for spec in specs
        ],
        "total_hands": sum(spec.hands for spec in specs),
        "counterfactual_rollouts": args.counterfactual_rollouts,
        "p3_accept_threshold": args.p3_accept_threshold,
        "p3_resample_tries": args.p3_resample_tries,
        "seed": args.seed,
    }
    if args.dry_run:
        # Fail loud on a bad candidate before anyone waits on a harvest.
        load_policy_v8(args.candidate)
        load_fit()
        print(json.dumps({"mode": "dry-run", **plan}, indent=2, sort_keys=True))
        return 0

    started = time.monotonic()
    workers = _harvest_workers(args.harvest_workers, len(specs))
    print(f"harvesting {len(specs)} legs across {workers} worker processes")
    if workers == 1:
        leg_results = [run_leg(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            leg_results = list(pool.map(run_leg, specs))

    rows: list[dict[str, Any]] = []
    for leg in leg_results:
        rows.extend(leg.pop("rows"))
        print(
            f"{leg['name']}: {leg['branch_rows']} branch rows from "
            f"{leg['decisions_emitted']} decisions over {leg['hands']} hands "
            f"(hero {leg['hero_bb_per_100']:+.1f} bb/100, "
            f"p3 fallback rate {leg['p3_fallback_rate']:.4f}, "
            f"{leg['wall_seconds']}s)"
        )
        if leg["p3_agent_fallbacks"]:
            raise PhaseBError(
                f"leg {leg['name']} saw {leg['p3_agent_fallbacks']} P3 "
                "card-blind degrades; the opponent measured is not the one "
                "configured"
            )

    report = validate_phase_b_rows(rows)
    write_phase_b_corpus(corpus_path, rows)
    reloaded = load_phase_b_corpus(corpus_path)
    if len(reloaded) != len(rows):
        raise PhaseBError("corpus did not round-trip its row count")
    for original, loaded in zip(rows, reloaded):
        if dict(original["branch_absorption"]) != dict(loaded["branch_absorption"]):
            raise PhaseBError("branch_absorption did not round-trip")

    summary = {
        "kind": "phase-b-corpus-summary",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": plan,
        "legs": leg_results,
        "validation": report,
        "wall_seconds": round(time.monotonic() - started, 1),
        "salt_rule": (
            "hero cards and revealed board fixed; future board resampled per "
            "rollout; card-blind holes resampled freely; P3 holes resampled "
            "per rollout by rejection against the seat's own prefix "
            "(threshold {t}, tries {n}), uniform fallback counted".format(
                t=args.p3_accept_threshold, n=args.p3_resample_tries
            )
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"wrote {report['branch_rows']} branch rows "
        f"({report['decisions']} decisions) to {corpus_path}"
    )
    print(
        "p3 fallback rate: "
        f"{report['p3_fallback_rate']:.4f} over "
        f"{report['p3_resample_totals']['seats_resampled']} seat resamples"
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
