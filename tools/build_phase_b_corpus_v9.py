"""Phase-B v9 harvester: contract-driven branches, schema-2 corpus (L4).

Sibling of ``tools.build_phase_b_corpus`` — that module and its stored
v8 corpora stay byte-untouched, and every piece of harvest physics that
the v9 contract does not change is IMPORTED from it verbatim: the
arranged-deck replay, the conditional P3 hole swap and its statistics,
the hero snapshot recorder, the leg/lineup/session machinery, and the
one-selected-decision-per-actor-and-hand rule with its RNG. What changes
is exactly the branch layer and the corpus format, to the contracts
pinned in `.handoff/notes/V9_RESTRUCTURE_PLAN.md` ("L3/L4 DATA
CONTRACTS" plus its "L3 LANDED" block, which the landed trainer loaders
enforce):

- **Candidates come from the contract on the snapshot.** The emitted set
  is ``branch_contract_v9.legal_branch_labels(available, to_call)`` in
  slot order — ``{passive, active}`` at a free spot, ``{fatal, active,
  aggressive}`` (aggressive only when a raise/all-in exists) at a priced
  one. There is no dedup order and no absorption map: no two v9
  branches execute the same action by construction, and the probe
  asserts that instead of assuming it (below). ``_E6_BRANCH_SPECS``
  dissolves into g — wager sizes are the COMPOSED sizing
  (``engine.rules.composition``: g plus whatever dials the header
  records; every dial ships OFF) at the decision's recorded read.
- **One read per decision, recorded and consumed.** The read is the
  extractor's own convention (schema-frozen: unconditioned multiway
  Monte-Carlo at ``feature_extract_v8._EQUITY_TRIALS`` with this
  harvest's ``feature_seed``), pushed through g's depth-invariant
  ``table_temperature`` and stored as the raw int ``10·T``. The
  composed wagers, the forced pot fractions, and the row's
  ``sizing_target``/``sizing_to_amount`` all derive from the DECODED
  stored int — bit-identical to what ``load_phase_b_corpus_v9``
  re-derives through frozen g, which is the trainer-side cross-check
  this recording exists for. The feature vector's own ``equity_multiway``
  slot is asserted equal to the read's equity per decision (one
  computation, two consumers — checked, not assumed). The hero POLICY's
  serve read is a separate draw at ``equity_trials`` (its gate/serve
  precision, pinned in the header as ``harvest == serve, one number``);
  the schema's feature convention stays 200 regardless.
- **The purity check replaces the dedup.** ``_probe_branch_set`` replays
  the deterministic prefix once and reports the action each forced
  branch actually executes; :func:`expected_executions` accepts only the
  contract's own — fatal->fold, passive->check, active->call at a
  price, active->bet/raise/all-in at a free spot,
  aggressive->raise/all-in. The v9 slots are SEMANTIC (each carries a
  fixed value formula: a call's value is eq·(pot+tc) − tc, a fold's is
  0), so an execution outside the branch's own set poisons its slot's
  arithmetic, and the v9 corpus has no absorption channel to record the
  substitution. The engine's rails would retarget literal-intent
  branches constantly — measured on the first smoke harvest, the
  call-margin gate folded 45% of forced calls and the bluff mixer
  raised forced folds and bet forced checks, 40 of 73 selected
  decisions dropped, biased against exactly the negative-EV call states
  the model needs fold-beats-call contrast on. So
  :class:`ContractForcingRecorder` executes fatal / passive / priced
  active as LITERAL fold / check / call payloads (the rails stay
  serve-side overrides ABOVE the composed layer — the L2 doctrine,
  tested there), while the two sized wager lanes still run through the
  policy's own ``decide_forced`` (the g hand-off needs the engine's
  floors, rounding and risk cap), and an all-in-only escalation is the
  literal shove. The purity check remains the ASSERTION of all this:
  literal lanes are clean by construction, and the one droppable class
  left is a genuine wager demotion (a risk-cap-collapsed escalation
  executing as a call/check — the state the L5 demotion rule will own),
  dropped and counted per class, reported per leg and in the summary.
- **Leg diagnostics are complete (defect 18h).** The emitted-set-size
  histogram (``emitted_branch_counts``, the field the v8 harvester
  serialises per leg and the simulator already counts) is counted at
  the purity-clean emission and serialised per leg, and the leg print
  carries every counter: emitted sizes, single-branch groups, purity
  drops with the action/size-mismatch and collision breakdowns, and
  belief degrades.
- **Rows are the pinned schema-2 shape.** ``decision.context`` is
  exactly ``compose_branch_values_v9``'s argument list (raw ints; the
  read as ``10·T``; ``legal_labels`` in slot order; the serve path's
  lane ranges — ``bet_range`` falls back to ``raiseRange`` at
  blind-option free spots exactly as serve does); branch entries carry
  centered ``reward_bb`` keyed by branch label, and the two wager
  executions (aggressive always, active only as a bet) carry
  ``sizing_target`` (the unclamped composed target) and
  ``sizing_to_amount`` (the composed to-amount clamped into the
  recorded lane range). The header is ``corpus_schema_version: 2`` with
  the composed sizing record, ``belief_fit_source`` (the P3 fit the
  belief buckets were computed from — the same provider serves the hero
  and the extractor), ``equity_trials``, and the frozen-instrument
  fields. After writing, the corpus is reloaded through the REAL
  trainer loader (``training.v9_trainer_phase_b.load_phase_b_corpus_v9``),
  so a corpus this tool blesses is one the trainer provably accepts,
  frozen-g re-derivation and all.

Money safety: pure offline simulation. No Arena requests, no
credentials, no promotion, no ``artifacts/approved.json``.

Usage (repo root, stdlib interpreter):
    python -m tools.build_phase_b_corpus_v9 --dry-run
    python -m tools.build_phase_b_corpus_v9 \
        --candidate artifacts/candidates/candidate-v9-0001.manifest.json
    python -m tools.build_phase_b_corpus_v9 --validate \
        artifacts/phase_b_v9/<corpus>.phase-b.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from engine import schema4
from engine.aggression_sizing import (
    DEFAULT_SIZING_PARAMETERS,
    SizingParameters,
    context_int_to_temperature,
    read_to_context_int,
    table_temperature,
)
from engine.branch_contract_v9 import BRANCH_LABELS_V9, legal_branch_labels
from engine.decision_engine import SharedEquityCache
from engine.feature_extract_v8 import _EQUITY_TRIALS
from engine.feature_extract_v9 import (
    _covered_allin_to_amounts,
    _lane_range,
    extract_features_v9,
)
from engine.game_state import (
    _hero_and_seats,
    _integer,
    active_opponent_count,
    effective_stack_chips,
)
from engine.hand_strength import estimate_equity
from engine.learned_policy_v9 import load_policy_v9
from engine.p3_belief_provider import P3BeliefProvider
from engine.rules.composition import (
    DEFAULT_RULE_LAYER,
    RuleLayerParams,
    compose_active_wager,
    compose_aggressive_target,
    composed_sizing_record,
)
from engine.rules.ruin_damper import table_exposure
from training.v9_trainer_phase_b import load_phase_b_corpus_v9
from tools.build_phase_b_corpus import (
    DEFAULT_ACCEPT_THRESHOLD,
    DEFAULT_RESAMPLE_TRIES,
    HeroRecorder,
    LegSpec,
    P3SeatWrapper,
    PhaseBError,
    PhaseBHarvestSimulator,
    PhaseBReplaySimulator,
    _build_opponents,
    _harvest_workers,
    default_leg_specs,
)

CORPUS_KIND_V9 = "phase-b-corpus"
CORPUS_SCHEMA_VERSION_V9 = 2
DEFAULT_CORPUS_NAME_V9 = "candidate-v9-phase-b"
DEFAULT_OUTPUT_DIR_V9 = Path("artifacts") / "phase_b_v9"
DEFAULT_CANDIDATE_V9 = (
    Path("artifacts") / "candidates" / "candidate-v9-0001.manifest.json"
)

_CENTERING_TOLERANCE_BB = 1e-6
_EQUITY_MULTIWAY_INDEX = schema4.feature_index_v9("equity_multiway")


def expected_executions(branch: str, to_call_zero: bool) -> frozenset[str]:
    """The action names a forced v9 branch is allowed to execute.

    The contract's own table, with the engine's two legitimate
    renderings folded in: a stack-reaching wager renders as ``all-in``
    (the contract calls that a realization, not a fifth branch), and a
    blind-option free-spot wager renders as ``raise`` because the Arena
    names it so. Everything else — a rescue-called fold, a gate-folded
    call, a bluff-upgraded check, a demoted escalation — is a rail
    retargeting the branch, and the decision is dropped rather than
    recorded under a lying label.
    """

    if branch == "fatal":
        return frozenset({"fold"})
    if branch == "passive":
        return frozenset({"check"})
    if branch == "active":
        return (
            frozenset({"bet", "raise", "all-in"})
            if to_call_zero
            else frozenset({"call"})
        )
    if branch == "aggressive":
        return frozenset({"raise", "all-in"})
    raise PhaseBError(f"unknown v9 branch {branch!r}")


class ContractForcingRecorder(HeroRecorder):
    """HeroRecorder whose FORCED branches execute the contract's actions.

    The hero's own behavior decisions (``decide``) pass through the
    wrapped policy untouched — only the counterfactual forcing channel
    changes. ``fold`` and ``check_call`` become literal payloads (the
    contract's fold / check / call for the state), so the rescue rail,
    the call-margin gates, and the bluff mixer — serve-side rails that
    sit ABOVE the composed layer by design — cannot retarget a branch
    whose value formula is fixed. ``aggress`` still routes through the
    policy's own ``decide_forced`` so g's fraction is realized with the
    engine's floors, rounding, and risk cap intact — except at an
    all-in-only state (no stated bet/raise range), where the escalation
    is the literal shove: its size is the stack, and the engine cannot
    choose an optional all-in until the L5 shove lane lands.
    """

    def decide_forced(
        self,
        table: Mapping[str, Any],
        *,
        family: str,
        pot_fraction: float | None = None,
    ) -> dict:
        allowed = table.get("allowedActions") or {}
        available = {
            str(value) for value in allowed.get("availableActions") or ()
        }
        if family == "fold":
            return {"action": "fold", "message": "contract branch"}
        if family == "check_call":
            if "check" in available:
                return {"action": "check", "message": "contract branch"}
            return {"action": "call", "message": "contract branch"}
        if family == "aggress":
            has_range = (
                allowed.get("betRange") is not None
                or allowed.get("raiseRange") is not None
            )
            if not has_range:
                if "all-in" not in available:
                    raise PhaseBError(
                        "an aggressive branch was forced with no wager "
                        "action available"
                    )
                return {
                    "action": "all-in",
                    "amount": int(allowed.get("allInToAmount") or 0),
                    "message": "contract branch",
                }
            return super().decide_forced(
                table, family=family, pot_fraction=pot_fraction
            )
        raise PhaseBError(f"unknown forced family {family!r}")


class _ArenaShapedReplaySimulator(PhaseBReplaySimulator):
    """The v9 rollout replay: call events carry the Arena's increment.

    Pre-harvest decision 3 (owner-confirmed 2026-08-31): the rollout
    policy must see the same history shape the serve path will — the
    frozen v8 replay keeps the legacy street-total record.
    """

    def __init__(self, *, swap=None, **kwargs: Any) -> None:
        super().__init__(swap=swap, arena_shaped_call_amounts=True, **kwargs)


class PhaseBHarvestSimulatorV9(PhaseBHarvestSimulator):
    """Harvest simulator emitting v9 (schema-2) Phase-B decision rows.

    Inherits the arranged replay, the conditional P3 hole swap, the
    selection RNG and the probe from the v8 harvester; overrides only
    the branch layer (contract candidates, composed sizing, the purity
    check), the emitted rows, and the call-event size encoding
    (pre-harvest decision 3: the Arena's increment, so Phase-B simulator
    rows agree with Phase-A and live).
    """

    def __init__(
        self,
        *,
        belief_provider: P3BeliefProvider,
        sizing: SizingParameters = DEFAULT_SIZING_PARAMETERS,
        rules: RuleLayerParams = DEFAULT_RULE_LAYER,
        arena_shaped_call_amounts: bool = True,
        postflop_selection: bool = False,
        street_quotas: Mapping[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(arena_shaped_call_amounts=arena_shaped_call_amounts, **kwargs)
        # The rollout replays must carry the SAME event shape as the main
        # play and the probe — a mixed flag re-creates the skew on the
        # replay side instead.
        self.replay_class = (
            _ArenaShapedReplaySimulator
            if arena_shaped_call_amounts
            else PhaseBReplaySimulator
        )
        # Street-targeted selection for supplemental postflop harvests:
        # one point per reached postflop street instead of one per hand,
        # capped per street so the corpus meets deliberate quotas. The
        # default (False) is the uniform one-per-hand rule, verbatim.
        self.postflop_selection = bool(postflop_selection)
        self._street_quota_remaining: dict[str, int] = dict(street_quotas or {})
        self.belief_provider = belief_provider
        self.sizing = sizing
        self.rules = rules
        #: Decisions dropped because a forced branch executed an action
        #: outside its contract set; keyed ``branch->action`` for the
        #: diagnosis (`fatal->call` is the rescue rail, `aggressive->call`
        #: the pre-L5 all-in-only demotion, and so on).
        self.probe_action_mismatches: Counter[str] = Counter()
        #: Decisions dropped because two branches executed one action.
        self.probe_collisions = 0
        #: Decisions dropped because a wager branch executed at an amount
        #: other than the one the row would record (see _purity_verdict).
        self.probe_size_mismatches: Counter[str] = Counter()
        #: Rows whose belief buckets silently degraded to the uniform
        #: prior (the provider swallows its own errors per decision —
        #: found by the 2026-08-30 range-note audit: nothing outside the
        #: Phase-A builder counted these). The row still ships (uniform
        #: buckets are the neutral convention), but the rate must be
        #: visible: a material rate means the corpus trained on buckets
        #: the serve path would not reproduce.
        self.belief_degrades = 0

    # -- the decision's read -------------------------------------------

    def _decision_read(
        self, capture: Mapping[str, Any], allowed: Mapping[str, Any]
    ) -> tuple[float, int, float]:
        """(equity, 10·T, boldness) — the extractor's own convention.

        Same estimator, trials, and seed as ``extract_features_v9``'s
        internal read, so the recorded read and the vector's baked
        costs describe one computation (asserted per decision against
        the vector's ``equity_multiway`` slot). The boldness handed to
        the composed sizing comes from the DECODED stored int, so the
        harvest derivation and the trainer's re-derivation are the same
        arithmetic on the same bits.
        """

        hero, _ = _hero_and_seats(capture)
        hole = [str(card) for card in hero.get("holeCards") or ()]
        if len(hole) != 2:
            raise PhaseBError("captured snapshot lacks hero hole cards")
        board = tuple(str(card) for card in capture.get("boardCards") or ())
        opponents = active_opponent_count(capture)
        equity = (
            1.0
            if opponents < 1
            else estimate_equity(
                (hole[0], hole[1]),
                board,
                opponents,
                trials=_EQUITY_TRIALS,
                seed=self.feature_seed,
            )
        )
        reading = table_temperature(capture, allowed, equity)
        if reading is None:
            raise PhaseBError("temperature read failed on a captured snapshot")
        encoded = read_to_context_int(reading.temperature)
        boldness = self.sizing.boldness(context_int_to_temperature(encoded))
        return equity, encoded, boldness

    # -- context and candidates ----------------------------------------

    def _context_v9(self, capture: Mapping[str, Any]) -> dict[str, Any]:
        """The pinned raw-int decision context, minus the read (added by
        the caller once computed)."""

        allowed = capture["allowedActions"]
        available = {str(value) for value in allowed["availableActions"]}
        hero, seats = _hero_and_seats(capture)
        contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
        stack = _integer(hero.get("stackChips"), "hero stackChips", minimum=1)
        total_committed = _integer(
            hero.get("totalCommittedChips", contribution), "hero totalCommittedChips"
        )
        to_call = _integer(allowed.get("callChips", 0), "callChips")
        legal = legal_branch_labels(available, to_call)
        return {
            "pot": _integer(capture.get("potChips"), "potChips"),
            "to_call": to_call,
            "contribution": contribution,
            "effective_stack": int(effective_stack_chips(capture)),
            "purse": stack + total_committed,
            "street": str(capture.get("street") or "").casefold(),
            "bankroll": stack,
            "exposure": table_exposure(capture),
            "covered_allin_to_amounts": list(
                _covered_allin_to_amounts(hero, seats)
            ),
            "legal_labels": [
                label for label in BRANCH_LABELS_V9 if label in legal
            ],
            # The serve path's lane resolution, mirrored exactly: at
            # blind-option free spots the Arena names the unprovoked
            # wager "raise" and leaves betRange null.
            "bet_range": (
                _lane_range(allowed, "betRange")
                or _lane_range(allowed, "raiseRange")
            ),
            "raise_range": _lane_range(allowed, "raiseRange"),
        }

    def _candidates_v9(
        self, context: Mapping[str, Any], boldness: float
    ) -> tuple[
        list[tuple[str, str, float | None]],
        dict[str, tuple[float, float]],
    ]:
        """Contract candidates plus each wager execution's sizing pair.

        Returns ``(candidates, sizing_fields)`` where candidates are the
        ``(branch, forced_family, pot_fraction)`` tuples the stock probe
        and rollout machinery consume, and ``sizing_fields`` maps each
        wager-making branch to ``(sizing_target, sizing_to_amount)`` —
        the unclamped composed target and the lane-range-clamped
        absolute amount, exactly what the trainer loader re-derives.
        The pot fractions are the serve path's own derivation from the
        UNCLAMPED target (the E6 discipline), so the engine realizes the
        composed size with every clamp intact.
        """

        to_call = int(context["to_call"])
        pot = int(context["pot"])
        shared = dict(
            boldness=boldness,
            pot=pot,
            effective_stack=int(context["effective_stack"]),
            contribution=int(context["contribution"]),
            street=str(context["street"]),
            bankroll=int(context["bankroll"]),
            exposure=int(context["exposure"]),
            covered_allin_to_amounts=tuple(context["covered_allin_to_amounts"]),
            sizing=self.sizing,
            geometric=self.rules.geometric,
            snap=self.rules.snap,
            damper=self.rules.damper,
        )

        def clamped(to_amount: float, lane_range) -> float:
            if lane_range is None:
                return to_amount
            low, high = lane_range
            return min(high, max(low, to_amount))

        candidates: list[tuple[str, str, float | None]] = []
        sizing_fields: dict[str, tuple[float, float]] = {}
        for label in context["legal_labels"]:
            if label == "fatal":
                candidates.append(("fatal", "fold", None))
            elif label == "passive":
                candidates.append(("passive", "check_call", None))
            elif label == "active":
                if to_call > 0:
                    candidates.append(("active", "check_call", None))
                else:
                    composed = compose_active_wager(**shared)
                    fraction = composed.target / max(1, pot)
                    sizing_fields["active"] = (
                        composed.target,
                        clamped(composed.to_amount, context["bet_range"]),
                    )
                    candidates.append(("active", "aggress", fraction))
            elif label == "aggressive":
                composed = compose_aggressive_target(to_call=to_call, **shared)
                fraction = (composed.target - to_call) / max(1, pot + to_call)
                sizing_fields["aggressive"] = (
                    composed.target,
                    clamped(composed.to_amount, context["raise_range"]),
                )
                candidates.append(("aggressive", "aggress", fraction))
            else:  # pragma: no cover — legal_branch_labels cannot emit it
                raise PhaseBError(f"unknown legal label {label!r}")
        return candidates, sizing_fields

    def _purity_verdict(
        self,
        candidates: Sequence[tuple[str, str, float | None]],
        executed: Mapping[str, tuple[str, int | None]],
        to_call_zero: bool,
        sizing_fields: Mapping[str, tuple[float, float]],
    ) -> str | None:
        """None when every branch executed its own contract action
        distinctly AT THE SIZE THE ROW WILL RECORD; else the drop reason.

        The size check is not decoration. A wager branch's value formula
        is priced with ``sizing_to_amount`` as its wager, so a row whose
        rollout measured a DIFFERENT amount teaches the formula a reward
        earned at another price. The sweep found this live: a near-nut
        escalation whose raise the risk cap refused now executes as a
        shove, which the action-name check admits (``{raise, all-in}``)
        while the recorded constants still describe the small raise —
        biasing ``equity_called[aggressive]`` upward at a cheaper price.
        The loader cannot catch it: it re-derives the same composed
        number through frozen g, so its cross-check passes tautologically
        and ``executed`` is read by nothing. Checked here, at the only
        place that holds both numbers.
        """

        for label, _, _ in candidates:
            action, amount = executed[label]
            if action not in expected_executions(label, to_call_zero):
                self.probe_action_mismatches[f"{label}->{action}"] += 1
                return f"{label} executed {action}"
            if label in sizing_fields:
                _, to_amount = sizing_fields[label]
                # Tolerance is ONE BIG BLIND plus a chip, and that is a
                # substantive choice, not slack. The recorded
                # `sizing_to_amount` is the range-clamped FLOAT, because
                # the value formula prices exactly that at BOTH train
                # and serve time (`learned_policy_v9._clamped_wager` —
                # the v8 approximation, no big-blind floor, no integer
                # rounding). The engine then legalizes with both. So a
                # sub-big-blind difference IS the approximation the two
                # sides already share, and rejecting it would drop good
                # rows for agreeing with the design — measured, it threw
                # away 4 of 140 decisions on a production-settings leg.
                # What the check is FOR is the category error the sweep
                # found: a cap-refused escalation shoving 6,000 while the
                # row prices 400. That survives this tolerance intact.
                if (
                    amount is None
                    or abs(float(amount) - to_amount) > self.big_blind + 1
                ):
                    self.probe_size_mismatches[f"{label}->{action}"] += 1
                    return (
                        f"{label} executed {action} to {amount!r}, but the "
                        f"row would record sizing_to_amount {to_amount!r}"
                    )
        pairs = [executed[label] for label, _, _ in candidates]
        if len(set(pairs)) != len(pairs):
            self.probe_collisions += 1
            return "two branches share an executed action"
        return None

    # -- rows ----------------------------------------------------------

    def _select_points(self, hand_index: int, points: list) -> list:
        """The point-selection rule for one hand.

        Uniform (the default, and the v8 rule verbatim): one point per
        agent, the same RNG key the frozen v8 harvester uses. Postflop
        (supplemental harvests): one point per reached postflop street,
        each capped by that street's remaining quota — the street-balanced
        sampler NEXT.md item 1 demanded, run as a supplement instead of a
        full re-harvest.
        """

        if not self.postflop_selection:
            selected = []
            for agent_id in sorted({point.agent_id for point in points}):
                choices = [point for point in points if point.agent_id == agent_id]
                selected.append(
                    random.Random(
                        f"{self.seed}:{hand_index}:{agent_id}:counterfactual"
                    ).choice(choices)
                )
            return selected
        selected = []
        for street in ("flop", "turn", "river"):
            remaining = self._street_quota_remaining.get(street, 0)
            if remaining <= 0:
                continue
            choices = [
                point
                for point in points
                if point.agent_id == self.hero_id and point.street == street
            ]
            if not choices:
                continue
            selected.append(
                random.Random(
                    f"{self.seed}:{hand_index}:{self.hero_id}:postflop:{street}"
                ).choice(choices)
            )
            self._street_quota_remaining[street] = remaining - 1
        return selected

    def _counterfactual_examples(
        self,
        initial_seats: list,
        button_index: int,
        hand_index: int,
        points: list,
        deck_for_test: Sequence[str] | None,
    ) -> list:
        """Emit v9 Phase-B decision rows; returns no TrainingExamples.

        The selection rule, the arrangement verification, the rollout
        loop and the centering check are the v8 harvester's own; the
        branch set, the sizing, the purity check and the row shape are
        the v9 contract's.
        """

        selected = self._select_points(hand_index, points)
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
            context = self._context_v9(capture)
            if context["street"] != point.street:
                raise PhaseBError(
                    f"captured street {context['street']!r} does not match "
                    f"the point's {point.street!r}"
                )
            equity_read, encoded, boldness = self._decision_read(
                capture, capture["allowedActions"]
            )
            context["read_temperature_x10"] = encoded
            candidates, sizing_fields = self._candidates_v9(context, boldness)
            if len(candidates) < 2:  # pragma: no cover — contract emits >= 2
                self.single_branch_groups += 1
                continue
            executed = self._probe_branch_set(
                initial_seats,
                button_index,
                hand_index,
                point,
                candidates,
                deck_for_test,
            )
            to_call_zero = context["to_call"] == 0
            if self._purity_verdict(
                candidates, executed, to_call_zero, sizing_fields
            ):
                continue
            stats_before = self.p3_stats.snapshot()
            outcomes: dict[str, float] = {}
            risks: dict[str, float] = {}
            for label, family, pot_fraction in candidates:
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
                    f"centered rewards do not cancel at {table_id}: {rewards!r}"
                )
            features = extract_features_v9(
                capture,
                belief_provider=self.belief_provider,
                potential_trials=self.potential_trials,
                seed=self.feature_seed,
                sizing=self.sizing,
                rules=self.rules,
            )
            if features[_EQUITY_MULTIWAY_INDEX] != equity_read:
                raise PhaseBError(
                    f"{table_id}: the vector's equity_multiway "
                    f"{features[_EQUITY_MULTIWAY_INDEX]!r} is not the "
                    f"recorded read's equity {equity_read!r} — one read, "
                    "two consumers is broken"
                )
            # Direct attribute access on purpose: a getattr default would
            # report 0 forever after a provider swap or a rename — the
            # very silence this counter exists to break.
            if self.belief_provider.last_degrade_reason is not None:
                self.belief_degrades += 1
            seats_delta = tuple(
                after - before
                for after, before in zip(self.p3_stats.snapshot(), stats_before)
            )
            branch_rows = []
            for label, _, _ in candidates:
                entry: dict[str, Any] = {
                    "branch": label,
                    "reward_bb": rewards[label],
                    "outcome_bb": outcomes[label] / self.big_blind,
                    "risk_fraction": risks[label],
                    "executed": [executed[label][0], executed[label][1]],
                }
                if label in sizing_fields:
                    target, to_amount = sizing_fields[label]
                    entry["sizing_target"] = target
                    entry["sizing_to_amount"] = to_amount
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
                    "context": dict(context),
                    "features": [float(value) for value in features],
                    "branches": branch_rows,
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
            # Defect 18h: the emitted-set-size histogram the v8 harvester
            # serialises per leg was dropped on the v9 line. This is the
            # one place the emitted set is final — the purity verdict has
            # passed and every candidate becomes one branch row — so the
            # count is exact here, per decision.
            self.emitted_branch_counts[len(candidates)] = (
                self.emitted_branch_counts.get(len(candidates), 0) + 1
            )
        return []


# ---------------------------------------------------------------------------
# Corpus IO
# ---------------------------------------------------------------------------


def corpus_header_v9(
    *,
    sizing_record: Mapping[str, Any],
    belief_fit_source: str,
    equity_trials: int,
    starting_stack: int,
    big_blind: int,
    seeds: Sequence[int],
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The schema-2 header, canonicalized the way the trainer reads it."""

    payload: dict[str, Any] = {
        "kind": CORPUS_KIND_V9,
        "corpus_schema_version": CORPUS_SCHEMA_VERSION_V9,
        "feature_schema_version": schema4.SCHEMA_VERSION_V9,
        "input_size": schema4.INPUT_SIZE_V9,
        "branch_labels": list(BRANCH_LABELS_V9),
        "sizing": dict(sizing_record),
        "belief_fit_source": belief_fit_source,
        "equity_trials": int(equity_trials),
        "starting_stack": int(starting_stack),
        "big_blind": int(big_blind),
        "seeds": [int(seed) for seed in seeds],
    }
    # The sampling provenance: a merged corpus is a UNION of sampling
    # schemes and must say so, or its statistics lie.
    if selection is not None:
        payload["selection"] = json.loads(
            json.dumps(selection, sort_keys=True, allow_nan=False)
        )
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def write_phase_b_corpus_v9(
    path: str | Path,
    header: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "wb") as raw:
        # mtime pinned to zero so byte-identical reruns are byte-identical
        # outputs (the Phase-A convention, kept).
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(
                json.dumps(
                    header, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
                + b"\n"
            )
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )


def corpus_statistics(corpus) -> dict[str, Any]:
    """Aggregates over a corpus already loaded by the TRAINER's loader.

    The v9 contract deliberately has one reader: everything structural
    was already enforced by ``load_phase_b_corpus_v9`` (the version
    gate, the frozen-g sizing cross-check, emission == legality); this
    reports what a harvest operator needs to see.
    """

    decisions = corpus.decisions
    per_street: dict[str, int] = {}
    branch_counts: dict[str, int] = {}
    emitted_sizes: dict[str, int] = {}
    free_spots = 0
    for decision in decisions:
        per_street[decision.street] = per_street.get(decision.street, 0) + 1
        emitted_sizes[str(len(decision.emitted))] = (
            emitted_sizes.get(str(len(decision.emitted)), 0) + 1
        )
        if decision.to_call_zero:
            free_spots += 1
        for label in decision.emitted:
            branch_counts[label] = branch_counts.get(label, 0) + 1
    return {
        "decisions": len(decisions),
        "branch_rows": sum(len(decision.emitted) for decision in decisions),
        "branch_label_counts": dict(sorted(branch_counts.items())),
        "decisions_per_street": dict(sorted(per_street.items())),
        "emitted_branch_sizes": dict(sorted(emitted_sizes.items())),
        "free_spot_decisions": free_spots,
        "equity_trials": corpus.equity_trials,
        "belief_fit_source": corpus.belief_fit_source,
        "instrument": {
            "starting_stack": corpus.starting_stack,
            "big_blind": corpus.big_blind,
            "seeds": list(corpus.seeds),
        },
    }


# ---------------------------------------------------------------------------
# Harvest legs
# ---------------------------------------------------------------------------


def _build_hero_v9(spec: LegSpec, provider: P3BeliefProvider) -> HeroRecorder:
    policy = load_policy_v9(
        spec.candidate,
        equity_trials=spec.equity_trials,
        equity_cache=SharedEquityCache(),
        belief_provider=provider,
        potential_trials=spec.potential_trials,
        feature_seed=spec.feature_seed,
        # hyper_aggression_chance stays None: the v9 line's default is
        # already 0.0 (owner decision 2026-08-30).
    )
    return ContractForcingRecorder(policy)


def run_leg_v9(spec: LegSpec) -> dict[str, Any]:
    """Run one leg's carry-over sessions; returns rows plus diagnostics.

    Session mechanics are the v8 harvester's: stacks carry within a
    session, each session starts fresh policy instances on a derived
    seed, and a session is capped so a busted hero cannot burn a long
    unrecorded tail. One P3 belief provider is built per leg and serves
    BOTH the hero policy and the harvester's feature extraction — one
    fit, one provenance string.
    """

    started = time.monotonic()
    provider = P3BeliefProvider.from_artifact()
    sizing = DEFAULT_SIZING_PARAMETERS
    rules = DEFAULT_RULE_LAYER
    rows: list[dict[str, Any]] = []
    totals = {
        "hands": 0,
        "sessions": 0,
        "hero_decisions": 0,
        "decisions_selected": 0,
        "decisions_emitted": 0,
        "single_branch_groups": 0,
        "probe_collisions": 0,
        "belief_degrades": 0,
        "hero_chip_delta": 0,
        "hero_hands": 0,
    }
    # Defect 18h: the emitted-set-size histogram the simulator counts
    # per session; each session builds a fresh simulator, so without
    # this the histogram is computed and thrown away once per session
    # (the v8 harvester's run_leg does the same accumulation).
    emitted_branch_counts: dict[int, int] = {}
    probe_action_mismatches: Counter[str] = Counter()
    probe_size_mismatches: Counter[str] = Counter()
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
        recorder = _build_hero_v9(spec, provider)
        simulator = PhaseBHarvestSimulatorV9(
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
            belief_provider=provider,
            sizing=sizing,
            rules=rules,
            postflop_selection=spec.postflop_selection,
            street_quotas=spec.street_quotas,
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
        totals["probe_collisions"] += simulator.probe_collisions
        totals["belief_degrades"] += simulator.belief_degrades
        probe_action_mismatches.update(simulator.probe_action_mismatches)
        probe_size_mismatches.update(simulator.probe_size_mismatches)
        for size, count in simulator.emitted_branch_counts.items():
            emitted_branch_counts[size] = (
                emitted_branch_counts.get(size, 0) + count
            )
        totals["hero_chip_delta"] += result.chip_deltas.get("hero", 0)
        totals["hero_hands"] += result.hands_by_agent.get("hero", result.hands)
        for key in p3_totals:
            p3_totals[key] += getattr(simulator.p3_stats, key)
        session += 1
    resamples = p3_totals["seats_resampled"]
    hero_bb_per_100 = (
        100.0 * totals["hero_chip_delta"] / (spec.big_blind * totals["hero_hands"])
        if totals["hero_hands"]
        else 0.0
    )
    dropped = (
        sum(probe_action_mismatches.values())
        + sum(probe_size_mismatches.values())
        + totals["probe_collisions"]
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
        "purity_dropped_decisions": dropped,
        "purity_drop_rate": (
            dropped / totals["decisions_selected"]
            if totals["decisions_selected"]
            else 0.0
        ),
        "probe_action_mismatches": dict(sorted(probe_action_mismatches.items())),
        "probe_size_mismatches": dict(sorted(probe_size_mismatches.items())),
        "probe_collisions": totals["probe_collisions"],
        "belief_degrades": totals["belief_degrades"],
        "branch_rows": sum(len(row["branches"]) for row in rows),
        "emitted_branch_counts": {
            str(key): value for key, value in sorted(emitted_branch_counts.items())
        },
        "hero_bb_per_100": round(hero_bb_per_100, 3),
        "belief_fit_source": provider.fit_source,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--candidate",
        default=str(DEFAULT_CANDIDATE_V9),
        help="v9 candidate manifest used as the acting (hero) policy",
    )
    parser.add_argument("--corpus-name", default=DEFAULT_CORPUS_NAME_V9)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR_V9),
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
    parser.add_argument(
        "--equity-trials",
        type=int,
        # OWNER DECISION 2026-08-30 (settled at L4 close): harvest ==
        # serve at 1,000 trials. Halves the gate-read noise (sigma(E)
        # 0.035 -> 0.016; the 2026-08-26 bust hand's 521 BB call cleared
        # its gate by 0.032 — a coin flip at 200-trial noise), costs
        # ~27-54 ms per read at serve against the 10 s deadline, and
        # ~+40 min on a 50k-decision harvest. Scoped to the v9 line:
        # the v7/v8 serve default (DEFAULT_SERVE_EQUITY_TRIALS = 200)
        # is untouched — frozen instruments bake it.
        default=1_000,
        help="the hero policy's gate/serve equity precision; recorded in "
        "the header and pinned at serve.equity_trials by the trainer "
        "(harvest == serve, one number — owner-settled at 1,000 on "
        "2026-08-30). The feature/read equity stays the schema-frozen "
        "200-trial convention regardless.",
    )
    parser.add_argument(
        "--potential-trials",
        type=int,
        default=400,
        help="hand_potential trials for schema-4 extraction (serve default)",
    )
    parser.add_argument("--feature-seed", type=int, default=7)
    parser.add_argument("--counterfactual-rollouts", type=int, default=2)
    parser.add_argument(
        "--p3-accept-threshold", type=float, default=DEFAULT_ACCEPT_THRESHOLD
    )
    parser.add_argument(
        "--p3-resample-tries", type=int, default=DEFAULT_RESAMPLE_TRIES
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
        help="load an existing corpus through the TRAINER's loader and "
        "print its statistics",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--postflop",
        action="store_true",
        help="street-targeted selection: one point per reached postflop "
        "street, capped by --street-targets — the supplemental-harvest "
        "mode for street balance (default: the uniform one-per-hand rule)",
    )
    parser.add_argument(
        "--street-targets",
        default="15000,10000,6000",
        help="total per-street decision targets for --postflop, as "
        "FLOP,TURN,RIVER (divided across legs; default 15000,10000,6000)",
    )
    parser.add_argument(
        "--merge",
        nargs=2,
        metavar=("BASE", "EXTRA"),
        default=None,
        help="merge two v9 corpora (base then extra) into --corpus-name in "
        "--output-dir; refuses incompatible headers",
    )
    return parser


def _raw_corpus_parts(path: Path) -> tuple[dict[str, Any], list[str]]:
    """(header, raw row lines) — read, never re-serialized."""

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(stream.readline())
        rows = [line for line in stream if line.strip()]
    return header, rows


#: Header fields two corpora must agree on to be mergeable — everything
#: the trainer validates per-row against the header, plus anything that
#: would make a mixed corpus dishonest.
_MERGE_COMPAT_KEYS = (
    "corpus_schema_version",
    "feature_schema_version",
    "input_size",
    "branch_labels",
    "sizing",
    "belief_fit_source",
    "equity_trials",
    "starting_stack",
    "big_blind",
)


def merge_corpora_v9(
    base: str | Path, extra: str | Path, output: str | Path
) -> dict[str, Any]:
    """Merge two v9 corpora (base first) into one, refusing incompatibles.

    The merged header records the union of seeds and the list of both
    harvests' ``selection`` provenance blocks, so a union of sampling
    schemes is stated, never disguised. Both inputs are validated by the
    trainer's own loader before anything is written, and the output is
    validated the same way before it is blessed.
    """

    base_path = Path(base).expanduser().resolve()
    extra_path = Path(extra).expanduser().resolve()
    if base_path == extra_path:
        raise PhaseBError("cannot merge a corpus with itself")
    load_phase_b_corpus_v9(base_path)
    load_phase_b_corpus_v9(extra_path)
    base_header, base_rows = _raw_corpus_parts(base_path)
    extra_header, extra_rows = _raw_corpus_parts(extra_path)
    for key in _MERGE_COMPAT_KEYS:
        if base_header.get(key) != extra_header.get(key):
            raise PhaseBError(
                f"corpora disagree on header {key!r}; merging them would "
                "produce a corpus the trainer's per-row checks reject"
            )
    # decision ids are ``sim-<seed>-<hand>:hero:<ordinal>``, so shared
    # leg seeds collide PROBABILISTICALLY — refuse the overlap outright
    # rather than hoping no hand index happens to match.
    overlap = set(base_header["seeds"]) & set(extra_header["seeds"])
    if overlap:
        raise PhaseBError(
            f"corpora share leg seeds {sorted(overlap)}; decision ids "
            "would collide. Harvest the supplement with a disjoint "
            "--seed base and re-merge"
        )
    selections = [
        selection
        for selection in (
            base_header.get("selection"),
            extra_header.get("selection"),
        )
        if selection is not None
    ]
    merged = dict(base_header)
    merged["seeds"] = sorted(set(base_header["seeds"]) | set(extra_header["seeds"]))
    if selections:
        merged["selection"] = selections
    else:
        merged.pop("selection", None)
    rows = [json.loads(line) for line in base_rows + extra_rows]
    write_phase_b_corpus_v9(output, merged, rows)
    corpus = load_phase_b_corpus_v9(output)
    return {
        "base": str(base_path),
        "extra": str(extra_path),
        "output": str(Path(output).expanduser().resolve()),
        "base_rows": len(base_rows),
        "extra_rows": len(extra_rows),
        "merged_rows": len(rows),
        "decisions": len(corpus.decisions),
        "seeds": merged["seeds"],
    }


def merge_corpora_v9_cli(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output = output_dir / f"{args.corpus_name}.phase-b.jsonl.gz"
    report = merge_corpora_v9(args.merge[0], args.merge[1], output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.validate:
        corpus = load_phase_b_corpus_v9(args.validate)
        print(json.dumps(corpus_statistics(corpus), indent=2, sort_keys=True))
        return 0

    if args.merge:
        return merge_corpora_v9_cli(args)

    if args.counterfactual_rollouts < 1:
        parser.error("--counterfactual-rollouts must be positive")
    if not 0.0 <= args.p3_accept_threshold <= 1.0:
        parser.error("--p3-accept-threshold must be within [0, 1]")
    if args.p3_resample_tries < 1:
        parser.error("--p3-resample-tries must be positive")
    if args.session_hands < 1:
        parser.error("--session-hands must be positive")
    if args.equity_trials < 1:
        parser.error("--equity-trials must be positive")

    street_targets: dict[str, int] | None = None
    if args.postflop:
        parts = [part.strip() for part in args.street_targets.split(",")]
        if len(parts) != 3:
            parser.error("--street-targets must be FLOP,TURN,RIVER")
        try:
            values = [int(part) for part in parts]
        except ValueError:
            parser.error("--street-targets values must be integers")
        if any(value < 0 for value in values):
            parser.error("--street-targets values must be non-negative")
        street_targets = {
            street: value for street, value in zip(("flop", "turn", "river"), values)
        }

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
    if street_targets is not None:
        leg_count = len(specs)
        # Ceiling so shortfalls are impossible-by-construction; a quota
        # is a cap, and availability may bind first (documented).
        per_leg = {
            street: math.ceil(value / leg_count)
            for street, value in street_targets.items()
        }
        specs = [
            replace(spec, postflop_selection=True, street_quotas=dict(per_leg))
            for spec in specs
        ]
    stacks = {spec.starting_stack for spec in specs}
    blinds = {spec.big_blind for spec in specs}
    if len(stacks) != 1 or len(blinds) != 1:
        parser.error(
            "the corpus header records ONE instrument; legs disagree on "
            f"starting_stack {sorted(stacks)} or big_blind {sorted(blinds)}"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    corpus_path = output_dir / f"{args.corpus_name}.phase-b.jsonl.gz"
    summary_path = output_dir / f"{args.corpus_name}.phase-b.summary.json"

    selection_provenance: dict[str, Any] | None = None
    if street_targets is not None:
        selection_provenance = {
            "mode": "postflop",
            "street_targets": street_targets,
        }

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
        "equity_trials": args.equity_trials,
        "p3_accept_threshold": args.p3_accept_threshold,
        "p3_resample_tries": args.p3_resample_tries,
        "seed": args.seed,
        "selection": selection_provenance,
    }
    if args.dry_run:
        # Fail loud on a bad candidate or missing fit before anyone
        # waits on a harvest.
        load_policy_v9(args.candidate)
        provider = P3BeliefProvider.from_artifact()
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "belief_fit_source": provider.fit_source,
                    "sizing": composed_sizing_record(),
                    **plan,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    started = time.monotonic()
    workers = _harvest_workers(args.harvest_workers, len(specs))
    print(f"harvesting {len(specs)} legs across {workers} worker processes")
    if workers == 1:
        leg_results = [run_leg_v9(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            leg_results = list(pool.map(run_leg_v9, specs))

    rows: list[dict[str, Any]] = []
    fit_sources = set()
    mismatch_totals: Counter[str] = Counter()
    size_mismatch_totals: Counter[str] = Counter()
    for leg in leg_results:
        rows.extend(leg.pop("rows"))
        fit_sources.add(leg["belief_fit_source"])
        mismatch_totals.update(leg["probe_action_mismatches"])
        size_mismatch_totals.update(leg["probe_size_mismatches"])
        print(
            f"{leg['name']}: {leg['branch_rows']} branch rows from "
            f"{leg['decisions_emitted']} decisions over {leg['hands']} hands "
            f"(hero {leg['hero_bb_per_100']:+.1f} bb/100, "
            f"emitted sizes {leg['emitted_branch_counts']}, "
            f"single-branch groups {leg['single_branch_groups']}, "
            f"purity drops {leg['purity_dropped_decisions']} "
            f"(rate {leg['purity_drop_rate']:.4f}), "
            f"probe action mismatches {leg['probe_action_mismatches']}, "
            f"probe size mismatches {leg['probe_size_mismatches']}, "
            f"probe collisions {leg['probe_collisions']}, "
            f"belief degrades {leg['belief_degrades']}, "
            f"p3 fallback rate {leg['p3_fallback_rate']:.4f}, "
            f"{leg['wall_seconds']}s)"
        )
        if leg["p3_agent_fallbacks"]:
            raise PhaseBError(
                f"leg {leg['name']} saw {leg['p3_agent_fallbacks']} P3 "
                "card-blind degrades; the opponent measured is not the one "
                "configured"
            )
    if len(fit_sources) != 1:
        raise PhaseBError(
            f"legs disagree on the P3 fit source: {sorted(fit_sources)}"
        )

    header = corpus_header_v9(
        sizing_record=composed_sizing_record(),
        belief_fit_source=fit_sources.pop(),
        equity_trials=args.equity_trials,
        starting_stack=specs[0].starting_stack,
        big_blind=specs[0].big_blind,
        seeds=[spec.seed for spec in specs],
        selection=selection_provenance,
    )
    write_phase_b_corpus_v9(corpus_path, header, rows)
    # The proof, not a formality: the corpus is reloaded through the
    # TRAINER's own loader — the schema-2 gate, the frozen-g sizing
    # re-derivation, and emission-equals-legality all run here, so a
    # corpus this tool blesses is one the trainer provably accepts.
    corpus = load_phase_b_corpus_v9(corpus_path)
    if len(corpus.decisions) != len(rows):
        raise PhaseBError("corpus did not round-trip its decision count")
    report = corpus_statistics(corpus)

    summary = {
        "kind": "phase-b-corpus-v9-summary",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": plan,
        "header": header,
        "legs": leg_results,
        "validation": report,
        "purity_drops": {
            "total": sum(leg["purity_dropped_decisions"] for leg in leg_results),
            "action_mismatches": dict(sorted(mismatch_totals.items())),
            "size_mismatches": dict(sorted(size_mismatch_totals.items())),
            "collisions": sum(leg["probe_collisions"] for leg in leg_results),
            "belief_degrades": sum(leg["belief_degrades"] for leg in leg_results),
        },
        "wall_seconds": round(time.monotonic() - started, 1),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {report['branch_rows']} branch rows "
        f"({report['decisions']} decisions) to {corpus_path}"
    )
    print(f"trainer loader accepted the corpus ({report['decisions']} decisions)")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
