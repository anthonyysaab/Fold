"""Seeded no-limit hold'em table simulator emitting Arena-shaped snapshots.

The simulator is the counterfactual half of the learning plan: it deals
2-6-player hands, asks each agent for a decision through the same snapshot
dictionaries the live Arena sends (``toAmount`` semantics, ``allowedActions``,
``recentEvents``, per-seat ``agentId``), and settles exact chip deltas with
layered side pots. Live policies therefore run in simulation unchanged, and
scripted archetypes provide opponent coverage the teacher data lacks
(calibrated medians through permanent shovers).

Everything is deterministic from the match seed. Simplifications versus a
full cash-game engine, chosen for evaluation stability: stacks reset every
hand by default (clean BB/100), any raise reopens betting, split-pot
remainders go to the lowest seat, and there are no antes or straddles.

No Arena requests are made and no credentials are touched.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from devfun_poker_playground.hand_strength import _shared_evaluator, _treys_card
from devfun_poker_playground.policy_features import LABELS
from devfun_poker_playground.training_telemetry import TrainingExample

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_DECK = tuple(f"{rank}{suit}" for rank in _RANKS for suit in _SUITS)
_STREETS = ("preflop", "flop", "turn", "river")
_BOARD_REVEAL = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


@dataclass
class SimSeat:
    """One player's live state within a simulated hand."""

    seat_number: int
    agent_id: str
    agent: Any
    stack: int
    hole_cards: tuple[str, str] = ("2c", "3c")
    street_commit: int = 0
    hand_commit: int = 0
    folded: bool = False
    all_in: bool = False

    @property
    def active(self) -> bool:
        return not self.folded


@dataclass
class HandResult:
    """Settled outcome of one simulated hand."""

    hand_id: str
    chip_deltas: dict[str, int]
    showdown: bool
    board: tuple[str, ...]
    decisions: int


@dataclass
class MatchResult:
    """Aggregate outcome of a seeded multi-hand match."""

    hands: int
    big_blind: int
    chip_deltas: dict[str, int]
    decisions: dict[str, int]
    examples: list[TrainingExample] = field(default_factory=list)
    sessions: int = 1
    busts: dict[str, int] = field(default_factory=dict)
    # Hands each agent was actually seated for. A busted agent's frozen
    # chip delta must not be divided by hands it never played; in multiway
    # that subsidized ruin by up to a third.
    hands_by_agent: dict[str, int] = field(default_factory=dict)

    def bb_per_100(self, agent_id: str) -> float:
        hands = self.hands_by_agent.get(agent_id, self.hands)
        if not hands:
            return 0.0
        return 100.0 * self.chip_deltas.get(agent_id, 0) / (self.big_blind * hands)


class SimulationError(RuntimeError):
    """Raised when an agent submits an action the table cannot accept."""


@dataclass(frozen=True, slots=True)
class _CounterfactualPoint:
    agent_id: str
    decision_ordinal: int
    example: TrainingExample
    legal_families: tuple[str, ...]
    proposed_risk_fraction: float
    street: str = "preflop"


# Board cards visible on each street: the chance salt must not move any
# card the branch-point decision could already see.
_REVEALED_BOARD = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

# One value branch per legal family, except aggression, which is valued at
# the two sizes the serve path can actually pin. This is the sizing-defect
# repair: Q(state, aggress) is undefined without a size.
_FAMILY_BRANCHES = {
    "fold": (("fold", "fold", None),),
    "check_call": (("check_call", "check_call", None),),
    "aggress": (
        ("aggress_half_pot", "aggress", 0.5),
        ("aggress_pot", "aggress", 1.0),
    ),
}


class TableSimulator:
    """Deal seeded hands between agents that speak the Arena snapshot dialect."""

    def __init__(
        self,
        *,
        small_blind: int = 50,
        big_blind: int = 100,
        starting_stack: int = 1_000,
        seed: int = 1,
        collect_examples: bool = False,
        collect_counterfactuals: bool = False,
        counterfactual_rollouts: int = 1,
    ) -> None:
        if small_blind < 1 or big_blind < small_blind:
            raise ValueError("blinds must be positive and ordered")
        if counterfactual_rollouts < 1:
            raise ValueError("counterfactual_rollouts must be positive")
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stack = starting_stack
        self.seed = seed
        self.collect_examples = collect_examples or collect_counterfactuals
        self.collect_counterfactuals = collect_counterfactuals
        self.counterfactual_rollouts = counterfactual_rollouts
        self._evaluator = _shared_evaluator()

    # ------------------------------------------------------------------
    # Match level
    # ------------------------------------------------------------------

    def play_match(
        self,
        agents: Sequence[tuple[str, Any]],
        hands: int,
        *,
        reset_stacks: bool = True,
        deck_for_test: Sequence[str] | None = None,
    ) -> MatchResult:
        """Play ``hands`` seeded hands, rotating the button every hand."""

        if not 2 <= len(agents) <= 6:
            raise ValueError("simulate between 2 and 6 agents")
        stacks = {agent_id: self.starting_stack for agent_id, _ in agents}
        result = MatchResult(
            hands=0,
            big_blind=self.big_blind,
            chip_deltas={agent_id: 0 for agent_id, _ in agents},
            decisions={agent_id: 0 for agent_id, _ in agents},
        )
        for hand_index in range(hands):
            if reset_stacks:
                stacks = {agent_id: self.starting_stack for agent_id, _ in agents}
            seats = [
                SimSeat(
                    seat_number=index + 1,
                    agent_id=agent_id,
                    agent=agent,
                    stack=stacks[agent_id],
                )
                for index, (agent_id, agent) in enumerate(agents)
                if stacks[agent_id] > 0
            ]
            if len(seats) < 2:
                break
            button = hand_index % len(seats)
            hand = self._play_hand(
                seats,
                button_index=button,
                hand_index=hand_index,
                result=result,
                deck_for_test=deck_for_test,
            )
            for agent_id, delta in hand.chip_deltas.items():
                result.chip_deltas[agent_id] += delta
                stacks[agent_id] += delta
            for seat in seats:
                result.hands_by_agent[seat.agent_id] = (
                    result.hands_by_agent.get(seat.agent_id, 0) + 1
                )
            result.hands += 1
        return result

    # ------------------------------------------------------------------
    # Hand level
    # ------------------------------------------------------------------

    def _play_hand(
        self,
        seats: list[SimSeat],
        *,
        button_index: int,
        hand_index: int,
        result: MatchResult,
        deck_for_test: Sequence[str] | None = None,
        chance_salt: tuple[int, int, str] | None = None,
    ) -> HandResult:
        initial_seats = copy.deepcopy(seats) if self.collect_counterfactuals else None
        rng = random.Random(f"{self.seed}:{hand_index}")
        deck = list(deck_for_test) if deck_for_test else list(_DECK)
        if deck_for_test is None:
            rng.shuffle(deck)
        for seat in seats:
            seat.hole_cards = (deck.pop(0), deck.pop(0))
        board = [deck.pop(0) for _ in range(5)]
        if chance_salt is not None and deck_for_test is None:
            # Counterfactual continuation: resample every chance outcome the
            # branch-point decision could not see. The old replay froze the
            # entire deal, so averaging rollouts converged on E[chips | this
            # exact runout] instead of Q(s, a) -- measured as 67-100% of
            # rollouts returning bit-identical chips. The decision owner's
            # hole cards and the already-revealed board stay fixed; future
            # board cards always resample, and a card-blind opponent's hole
            # cards resample too because its actions cannot depend on them.
            rollout, revealed, owner = chance_salt
            salt_rng = random.Random(f"{self.seed}:{hand_index}:chance:{rollout}")
            pool = board[revealed:]
            resampled_seats = [
                seat
                for seat in seats
                if seat.agent_id != owner
                and getattr(seat.agent, "reads_cards", True) is False
            ]
            for seat in resampled_seats:
                pool.extend(seat.hole_cards)
            pool.extend(deck)
            salt_rng.shuffle(pool)
            for seat in resampled_seats:
                seat.hole_cards = (pool.pop(0), pool.pop(0))
            board[revealed:] = [pool.pop(0) for _ in range(5 - revealed)]
            deck = pool

        table_id = f"sim-{self.seed}-{hand_index}"
        continuation_rollout: int | None = None
        events: list[dict] = []
        order = seats[button_index + 1 :] + seats[: button_index + 1]
        # Heads-up: the button posts the small blind and acts first preflop.
        if len(seats) == 2:
            small_seat, big_seat = seats[button_index], order[0]
        else:
            small_seat, big_seat = order[0], order[1]
        self._post_blind(small_seat, self.small_blind, "smallBlind", events)
        self._post_blind(big_seat, self.big_blind, "bigBlind", events)

        pending: list[TrainingExample | None] = []
        counterfactual_points: list[_CounterfactualPoint] = []
        decision_ordinals: dict[str, int] = {}
        showdown = False
        decisions = 0

        for street in _STREETS:
            revealed = board[: _BOARD_REVEAL[street]]
            if street == "preflop":
                current_bet = self.big_blind
                last_raise = self.big_blind
                if len(seats) == 2:
                    actors = [small_seat, big_seat]
                else:
                    actors = order[2:] + order[:2]
            else:
                current_bet = 0
                last_raise = self.big_blind
                for seat in seats:
                    seat.street_commit = 0
                if len(seats) == 2:
                    actors = [big_seat, small_seat]
                else:
                    actors = [seat for seat in order if seat.active]
            live = [seat for seat in seats if seat.active]
            if len(live) < 2 or all(seat.all_in for seat in live):
                continue

            to_act = [seat for seat in actors if seat.active and not seat.all_in]
            acted_since_raise: set[int] = set()
            while to_act:
                seat = to_act.pop(0)
                if not seat.active or seat.all_in:
                    continue
                live_now = [other for other in seats if other.active]
                if len(live_now) < 2:
                    break
                snapshot = self._snapshot(
                    seat,
                    seats,
                    street,
                    revealed,
                    current_bet,
                    last_raise,
                    table_id,
                    hand_index,
                    events,
                    continuation_rollout,
                )
                payload = self._decide(seat, snapshot, result)
                ordinal = decision_ordinals.get(seat.agent_id, 0)
                decision_ordinals[seat.agent_id] = ordinal + 1
                decisions += 1
                action, amount = self._apply(
                    seat, payload, snapshot, current_bet, last_raise, events, street
                )
                rollout = payload.get("_counterfactual_rollout")
                if rollout is not None:
                    continuation_rollout = int(rollout)
                if self.collect_examples:
                    example = self._pending_example(seat, snapshot, payload)
                    if example is not None:
                        pending.append(example)
                        if self.collect_counterfactuals:
                            families = tuple(
                                family
                                for family in LABELS
                                if family
                                in {
                                    value
                                    for action_name in snapshot["allowedActions"][
                                        "availableActions"
                                    ]
                                    if (value := _family_of(action_name)) is not None
                                }
                            )
                            diagnostics = seat.agent.last_diagnostics
                            counterfactual_points.append(
                                _CounterfactualPoint(
                                    agent_id=seat.agent_id,
                                    decision_ordinal=ordinal,
                                    example=example,
                                    legal_families=families,
                                    proposed_risk_fraction=float(
                                        diagnostics.proposed_risk_fraction
                                        if diagnostics.proposed_risk_fraction
                                        is not None
                                        else 0.25
                                    ),
                                    street=str(
                                        snapshot.get("street") or "preflop"
                                    ).casefold(),
                                )
                            )
                if action in ("bet", "raise", "all-in") and amount > current_bet:
                    last_raise = max(last_raise, amount - current_bet)
                    current_bet = amount
                    acted_since_raise = {seat.seat_number}
                    to_act = [
                        other
                        for other in self._rotate_from(actors, seat)
                        if other.active and not other.all_in
                    ]
                else:
                    acted_since_raise.add(seat.seat_number)
            if sum(1 for seat in seats if seat.active) < 2:
                break

        deltas, showdown = self._settle(seats, board, events)
        if self.collect_counterfactuals:
            assert initial_seats is not None
            result.examples.extend(self._finalize_examples(pending, deltas))
            result.examples.extend(
                self._counterfactual_examples(
                    initial_seats,
                    button_index,
                    hand_index,
                    counterfactual_points,
                    deck_for_test,
                )
            )
        elif self.collect_examples:
            result.examples.extend(self._finalize_examples(pending, deltas))
        return HandResult(
            hand_id=f"{table_id}-hand",
            chip_deltas=deltas,
            showdown=showdown,
            board=tuple(board),
            decisions=decisions,
        )

    # ------------------------------------------------------------------
    # Betting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_from(actors: list[SimSeat], seat: SimSeat) -> list[SimSeat]:
        index = actors.index(seat)
        return actors[index + 1 :] + actors[:index]

    def _post_blind(
        self, seat: SimSeat, blind: int, label: str, events: list[dict]
    ) -> None:
        amount = min(blind, seat.stack)
        seat.stack -= amount
        seat.street_commit += amount
        seat.hand_commit += amount
        if seat.stack == 0:
            seat.all_in = True
        events.append(
            {
                "type": "BlindPosted",
                "street": "preflop",
                "summary": {
                    "seatNumber": seat.seat_number,
                    "action": label,
                    "amount": amount,
                },
            }
        )

    def _decide(self, seat: SimSeat, snapshot: dict, result: MatchResult) -> dict:
        decide = getattr(seat.agent, "decide", None)
        if decide is None:
            payload = seat.agent(snapshot)
        else:
            payload = decide(snapshot)
        result.decisions[seat.agent_id] = result.decisions.get(seat.agent_id, 0) + 1
        if not isinstance(payload, Mapping) or "action" not in payload:
            raise SimulationError(f"{seat.agent_id} returned an invalid payload")
        return dict(payload)

    def _apply(
        self,
        seat: SimSeat,
        payload: dict,
        snapshot: dict,
        current_bet: int,
        last_raise: int,
        events: list[dict],
        street: str,
    ) -> tuple[str, int]:
        allowed = snapshot["allowedActions"]
        available = set(allowed["availableActions"])
        action = str(payload.get("action"))
        if action not in available:
            raise SimulationError(
                f"{seat.agent_id} chose illegal action {action!r} "
                f"(legal: {sorted(available)})"
            )
        if action == "fold":
            seat.folded = True
            self._record(events, street, seat, "fold", None)
            return action, current_bet
        if action == "check":
            self._record(events, street, seat, "check", None)
            return action, current_bet
        if action == "call":
            call_chips = int(allowed["callChips"])
            paid = min(call_chips, seat.stack)
            self._commit(seat, paid)
            self._record(events, street, seat, "call", seat.street_commit)
            return action, current_bet
        # bet / raise / all-in arrive as total street commitments.
        if action == "all-in":
            target = int(allowed["allInToAmount"])
        else:
            target = int(payload.get("amount", 0))
            amount_range = allowed["betRange" if action == "bet" else "raiseRange"]
            if amount_range is None or not (
                int(amount_range["min"]) <= target <= int(amount_range["max"])
            ):
                raise SimulationError(
                    f"{seat.agent_id} sized {action} to {target} outside {amount_range}"
                )
        added = target - seat.street_commit
        if added > seat.stack:
            raise SimulationError(f"{seat.agent_id} bet more than its stack")
        self._commit(seat, added)
        self._record(events, street, seat, action, target)
        return action, max(target, current_bet)

    @staticmethod
    def _commit(seat: SimSeat, chips: int) -> None:
        seat.stack -= chips
        seat.street_commit += chips
        seat.hand_commit += chips
        if seat.stack == 0:
            seat.all_in = True

    @staticmethod
    def _record(
        events: list[dict], street: str, seat: SimSeat, action: str, amount: int | None
    ) -> None:
        summary: dict = {"seatNumber": seat.seat_number, "action": action}
        if amount is not None:
            summary["amount"] = amount
        events.append({"type": "ActionTaken", "street": street, "summary": summary})

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _snapshot(
        self,
        actor: SimSeat,
        seats: list[SimSeat],
        street: str,
        board: list[str],
        current_bet: int,
        last_raise: int,
        table_id: str,
        hand_index: int,
        events: list[dict],
        continuation_rollout: int | None = None,
    ) -> dict:
        call_chips = min(max(0, current_bet - actor.street_commit), actor.stack)
        can_check = call_chips == 0
        all_in_to = actor.street_commit + actor.stack
        min_raise_to = current_bet + last_raise
        can_bet = current_bet == 0 and actor.stack > 0
        can_raise = (
            current_bet > 0 and actor.stack > call_chips and all_in_to >= min_raise_to
        )
        available = ["fold"]
        if can_check:
            available.append("check")
        if call_chips > 0:
            available.append("call")
        if can_bet:
            available.append("bet")
        if can_raise:
            available.append("raise")
        if actor.stack > 0:
            available.append("all-in")
        pot = sum(seat.hand_commit for seat in seats)
        return {
            "id": table_id,
            "tableId": table_id,
            "handId": f"{table_id}-h{hand_index}",
            "street": street,
            "potChips": pot,
            "currentBet": current_bet,
            "boardCards": list(board),
            "smallBlindChips": self.small_blind,
            "bigBlindChips": self.big_blind,
            "selfSeatNumber": actor.seat_number,
            "seats": [
                {
                    "seatNumber": seat.seat_number,
                    "agentId": seat.agent_id,
                    "status": "Folded" if seat.folded else "Active",
                    "stackChips": seat.stack,
                    "currentBetChips": seat.street_commit,
                    "totalCommittedChips": seat.hand_commit,
                    "holeCards": list(seat.hole_cards) if seat is actor else None,
                }
                for seat in seats
            ],
            "allowedActions": {
                "canFold": True,
                "canCheck": can_check,
                "canCall": call_chips > 0,
                "canBet": can_bet,
                "canRaise": can_raise,
                "canAllIn": actor.stack > 0,
                "callAmount": call_chips,
                "callChips": call_chips,
                "callToAmount": current_bet,
                "minBet": self.big_blind if can_bet else None,
                "minRaiseTo": min_raise_to if can_raise else None,
                "betRange": (
                    {"min": min(self.big_blind, all_in_to), "max": all_in_to}
                    if can_bet
                    else None
                ),
                "raiseRange": (
                    {"min": min_raise_to, "max": all_in_to} if can_raise else None
                ),
                "allInToAmount": all_in_to if actor.stack > 0 else None,
                "availableActions": available,
                "amountSemantics": "toAmount",
                "reasoningRequired": False,
            },
            "recentEvents": list(events),
            "simulationRollout": continuation_rollout,
        }

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def _settle(
        self, seats: list[SimSeat], board: list[str], events: list[dict]
    ) -> tuple[dict[str, int], bool]:
        live = [seat for seat in seats if seat.active]
        deltas = {seat.agent_id: -seat.hand_commit for seat in seats}

        # Refund the uncalled portion of the largest commitment.
        top_seat = max(seats, key=lambda seat: seat.hand_commit)
        second = max(
            (seat.hand_commit for seat in seats if seat is not top_seat),
            default=0,
        )
        refund = top_seat.hand_commit - second
        if refund > 0:
            top_seat.hand_commit -= refund
            deltas[top_seat.agent_id] += refund

        if len(live) == 1:
            winner = live[0]
            pot = sum(seat.hand_commit for seat in seats)
            deltas[winner.agent_id] += pot
            return deltas, False

        # Layered side pots: each commitment level pays the best eligible hand.
        scores = {
            seat.seat_number: self._evaluator.evaluate(
                [_treys_card(card) for card in seat.hole_cards],
                [_treys_card(card) for card in board],
            )
            for seat in live
        }
        levels = sorted({seat.hand_commit for seat in seats if seat.hand_commit > 0})
        previous = 0
        for level in levels:
            layer = 0
            for seat in seats:
                layer += max(0, min(seat.hand_commit, level) - previous)
            eligible = [seat for seat in live if seat.hand_commit >= level]
            if not eligible:
                break
            best = min(scores[seat.seat_number] for seat in eligible)
            winners = sorted(
                (seat for seat in eligible if scores[seat.seat_number] == best),
                key=lambda seat: seat.seat_number,
            )
            share, remainder = divmod(layer, len(winners))
            for index, seat in enumerate(winners):
                deltas[seat.agent_id] += share + (remainder if index == 0 else 0)
            previous = level
        return deltas, True

    # ------------------------------------------------------------------
    # Self-play training capture
    # ------------------------------------------------------------------

    def _pending_example(
        self, seat: SimSeat, snapshot: dict, payload: dict
    ) -> TrainingExample | None:
        diagnostics = getattr(seat.agent, "last_diagnostics", None)
        if diagnostics is None:
            return None
        decision = diagnostics
        if (
            decision.learning_features is None
            or decision.behavior_probabilities is None
            or decision.deadline_fallback
            or decision.hyper_aggression  # noise is never a teacher
        ):
            return None
        allowed = snapshot["allowedActions"]
        families = {
            family
            for name in allowed["availableActions"]
            if (family := _family_of(name)) is not None
        }
        if len(families) < 2:
            return None
        submitted_family = _family_of(str(payload.get("action")))
        if submitted_family != decision.family:
            return None
        amount = payload.get("amount")
        contribution = snapshot["seats"][snapshot["selfSeatNumber"] - 1][
            "currentBetChips"
        ]
        hero = snapshot["seats"][snapshot["selfSeatNumber"] - 1]
        if str(payload.get("action")) == "call":
            new_chips = int(allowed["callChips"])
        elif submitted_family == "aggress" and amount is not None:
            new_chips = max(0, int(amount) - contribution)
        else:
            new_chips = 0
        effective = max(
            1,
            min(
                (
                    seat_info["stackChips"] + seat_info["currentBetChips"]
                    for seat_info in snapshot["seats"]
                    if seat_info["status"] == "Active"
                ),
                default=1,
            ),
        )
        return TrainingExample(
            table_id=f"{snapshot['tableId']}|{seat.agent_id}",
            policy_version=f"sim-{getattr(seat.agent, 'policy_version', 'agent')}",
            features=decision.learning_features,
            action_family_index=LABELS.index(decision.family),
            behavior_probabilities=decision.behavior_probabilities,
            submitted_risk_fraction=min(1.0, new_chips / effective),
            purse_bb=(hero["stackChips"] + hero["totalCommittedChips"])
            / self.big_blind,
            reward_bb=0.0,
            opponent_confidence=decision.opponent_evidence_confidence,
        )

    def _finalize_examples(
        self,
        pending: list[TrainingExample | None],
        deltas: dict[str, int],
    ) -> list[TrainingExample]:
        finished = []
        for example in pending:
            if example is None:
                continue
            table_id, agent_id = example.table_id.rsplit("|", 1)
            reward = deltas.get(agent_id, 0) / self.big_blind
            finished.append(
                TrainingExample(
                    table_id=table_id,
                    policy_version=example.policy_version,
                    features=example.features,
                    action_family_index=example.action_family_index,
                    behavior_probabilities=example.behavior_probabilities,
                    submitted_risk_fraction=example.submitted_risk_fraction,
                    purse_bb=example.purse_bb,
                    reward_bb=reward,
                    counterfactual=False,
                    opponent_confidence=example.opponent_confidence,
                )
            )
        return finished

    def _counterfactual_examples(
        self,
        initial_seats: list[SimSeat],
        button_index: int,
        hand_index: int,
        points: list[_CounterfactualPoint],
        deck_for_test: Sequence[str] | None,
    ) -> list[TrainingExample]:
        """Average repeated legal-family continuations for one state per actor."""

        selected: list[_CounterfactualPoint] = []
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
        examples: list[TrainingExample] = []
        for point in selected:
            decision_id = (
                f"sim-{self.seed}-{hand_index}:{point.agent_id}:"
                f"{point.decision_ordinal}"
            )
            branches = [
                branch
                for family in point.legal_families
                for branch in _FAMILY_BRANCHES[family]
            ]
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
                outcomes[label] = sum(outcome for outcome, _ in samples) / len(samples)
                risks[label] = sum(risk for _, risk in samples) / len(samples)
            baseline = sum(outcomes.values()) / len(outcomes)
            for label, family, _ in branches:
                examples.append(
                    TrainingExample(
                        table_id=f"sim-{self.seed}-{hand_index}",
                        policy_version=point.example.policy_version,
                        features=point.example.features,
                        action_family_index=LABELS.index(family),
                        behavior_probabilities=point.example.behavior_probabilities,
                        submitted_risk_fraction=risks[label],
                        purse_bb=point.example.purse_bb,
                        reward_bb=(outcomes[label] - baseline) / self.big_blind,
                        counterfactual=True,
                        opponent_confidence=point.example.opponent_confidence,
                        decision_id=decision_id,
                        inclusion_count=inclusion_counts[point.agent_id],
                        action_branch=label,
                    )
                )
        return examples

    def _counterfactual_outcome(
        self,
        initial_seats: list[SimSeat],
        button_index: int,
        hand_index: int,
        point: _CounterfactualPoint,
        family: str,
        pot_fraction: float | None,
        rollout: int,
        deck_for_test: Sequence[str] | None,
    ) -> tuple[int, float]:
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
        replay = TableSimulator(
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            starting_stack=self.starting_stack,
            seed=self.seed,
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
            deck_for_test=deck_for_test,
            chance_salt=(
                rollout,
                _REVEALED_BOARD.get(point.street, 0),
                point.agent_id,
            ),
        )
        if not forced.forced or forced.submitted_risk_fraction is None:
            raise SimulationError(
                "counterfactual replay did not reach its target decision"
            )
        return hand.chip_deltas[point.agent_id], forced.submitted_risk_fraction


def run_sessions(
    agent_factories: Sequence[tuple[str, Any]],
    *,
    target_hands: int,
    seed: int = 1,
    starting_stack: int = 6_000,
    small_blind: int = 50,
    big_blind: int = 100,
    collect_examples: bool = False,
    collect_counterfactuals: bool = False,
    counterfactual_rollouts: int = 1,
) -> MatchResult:
    """Play carry-over sessions until roughly ``target_hands`` accumulate.

    Stacks are never reset inside a session: they swing, shorten, and bust
    exactly as they would live, which is also what feeds the lead gauge
    real asymmetries. A session ends when fewer than two players have
    chips; the next session starts fresh with new policy instances (a new
    sitting, so session opponent models start cold) and a derived seed.
    Each factory entry is ``(agent_id, factory)`` where ``factory()``
    builds the agent; pass a lambda returning a shared instance for
    stateless scripted archetypes.
    """

    total = MatchResult(
        hands=0,
        big_blind=big_blind,
        chip_deltas={agent_id: 0 for agent_id, _ in agent_factories},
        decisions={agent_id: 0 for agent_id, _ in agent_factories},
        sessions=0,
        busts={agent_id: 0 for agent_id, _ in agent_factories},
    )
    session = 0
    while total.hands < target_hands:
        simulator = TableSimulator(
            small_blind=small_blind,
            big_blind=big_blind,
            starting_stack=starting_stack,
            seed=seed + 7_919 * session,
            collect_examples=collect_examples,
            collect_counterfactuals=collect_counterfactuals,
            counterfactual_rollouts=counterfactual_rollouts,
        )
        agents = [(agent_id, factory()) for agent_id, factory in agent_factories]
        result = simulator.play_match(
            agents, hands=target_hands - total.hands, reset_stacks=False
        )
        if result.hands == 0:
            break
        total.hands += result.hands
        total.sessions += 1
        for agent_id, delta in result.chip_deltas.items():
            total.chip_deltas[agent_id] += delta
            if delta <= -starting_stack:
                total.busts[agent_id] += 1
        for agent_id, count in result.decisions.items():
            total.decisions[agent_id] += count
        for agent_id, count in result.hands_by_agent.items():
            total.hands_by_agent[agent_id] = (
                total.hands_by_agent.get(agent_id, 0) + count
            )
        total.examples.extend(result.examples)
        session += 1
    return total


def _family_of(action: str) -> str | None:
    if action in ("bet", "raise", "all-in"):
        return "aggress"
    if action in ("check", "call"):
        return "check_call"
    if action == "fold":
        return "fold"
    return None


class RecordingPolicy:
    """Wrap a DecisionEngine policy so the simulator can capture diagnostics."""

    def __init__(self, policy: Any, *, record_examples: bool = True) -> None:
        self.policy = policy
        self.policy_version = getattr(policy, "policy_version", "policy")
        self.record_examples = record_examples
        self.last_diagnostics = None

    def decide(self, table: Mapping[str, Any]) -> dict:
        decision = self.policy.decide_with_diagnostics(table)
        self.last_diagnostics = decision if self.record_examples else None
        return decision.to_payload()

    def decide_forced(
        self,
        table: Mapping[str, Any],
        *,
        family: str,
        pot_fraction: float | None = None,
    ) -> dict:
        # Forced branches are measurements, never teacher demonstrations,
        # so they bypass diagnostics recording entirely.
        return self.policy.decide_forced(
            table, family=family, pot_fraction=pot_fraction
        )


class _ForcedFamilyPolicy:
    """Replay one policy trajectory while replacing exactly one decision."""

    def __init__(
        self,
        policy: Any,
        *,
        decision_ordinal: int,
        family: str,
        risk_fraction: float,
        rollout: int,
        pot_fraction: float | None = None,
    ) -> None:
        self.policy = policy
        self.decision_ordinal = decision_ordinal
        self.family = family
        self.risk_fraction = risk_fraction
        self.rollout = rollout
        self.pot_fraction = pot_fraction
        self.calls = 0
        self.forced = False
        self.submitted_risk_fraction: float | None = None

    def decide(self, table: Mapping[str, Any]) -> dict:
        # Calling the original policy keeps its tracker/RNG state aligned for
        # continuation after the forced branch.
        original = self.policy.decide(table)
        if self.calls != self.decision_ordinal:
            self.calls += 1
            return original
        self.calls += 1
        self.forced = True
        # Route the forced branch through the acting policy's own serve
        # path when it has one: the branch value then measures the action
        # the policy actually plays -- engine sizing at the branch's pot
        # fraction, bluff conversion on pinned passive families, and every
        # safety clamp. Fabricated payloads remain only for scripted agents
        # with no engine.
        if hasattr(self.policy, "decide_forced"):
            payload = dict(
                self.policy.decide_forced(
                    table, family=self.family, pot_fraction=self.pot_fraction
                )
            )
        else:
            payload = _payload_for_family(table, self.family, self.risk_fraction)
        payload["_counterfactual_rollout"] = self.rollout
        self.submitted_risk_fraction = _payload_risk_fraction(table, payload)
        return payload


def _payload_for_family(
    table: Mapping[str, Any], family: str, risk_fraction: float
) -> dict:
    allowed = table["allowedActions"]
    available = set(allowed["availableActions"])
    if family == "fold" and "fold" in available:
        return {"action": "fold", "message": "counterfactual"}
    if family == "check_call":
        if "check" in available:
            return {"action": "check", "message": "counterfactual"}
        if "call" in available:
            return {"action": "call", "message": "counterfactual"}
    if family == "aggress":
        hero = next(
            seat
            for seat in table["seats"]
            if seat["seatNumber"] == table["selfSeatNumber"]
        )
        active_purses = [
            seat["stackChips"] + seat["currentBetChips"]
            for seat in table["seats"]
            if seat["status"] == "Active"
        ]
        effective = max(1, min(active_purses))
        contribution = int(hero["currentBetChips"])
        desired = contribution + round(max(0.05, min(1.0, risk_fraction)) * effective)
        for action, range_name in (("raise", "raiseRange"), ("bet", "betRange")):
            amount_range = allowed.get(range_name)
            if action in available and amount_range is not None:
                target = min(
                    int(amount_range["max"]),
                    max(int(amount_range["min"]), desired),
                )
                return {
                    "action": action,
                    "amount": target,
                    "message": "counterfactual",
                }
        if "all-in" in available:
            return {
                "action": "all-in",
                "amount": int(allowed["allInToAmount"]),
                "message": "counterfactual",
            }
    raise SimulationError(f"no legal payload for counterfactual family {family}")


def _payload_risk_fraction(
    table: Mapping[str, Any], payload: Mapping[str, Any]
) -> float:
    allowed = table["allowedActions"]
    hero = next(
        seat for seat in table["seats"] if seat["seatNumber"] == table["selfSeatNumber"]
    )
    active_purses = [
        seat["stackChips"] + seat["currentBetChips"]
        for seat in table["seats"]
        if seat["status"] == "Active"
    ]
    effective = max(1, min(active_purses))
    action = str(payload["action"])
    if action == "call":
        new_chips = int(allowed["callChips"])
    elif action in ("bet", "raise", "all-in"):
        new_chips = max(0, int(payload.get("amount", 0)) - int(hero["currentBetChips"]))
    else:
        new_chips = 0
    return min(1.0, new_chips / effective)


@dataclass
class ScriptedAgent:
    """Parameterized opponent archetype with a seeded, deterministic mixer.

    ``aggression`` is the chance of choosing a bet or raise when one is
    legal; ``fold_vs_bet`` the chance of folding when facing chips to call;
    ``shove_rate`` the chance of open-shoving any decision (the permanent
    all-in attacker at 1.0). Sizes are minimum-raise or pot-fraction mixes.
    """

    # Scripted archetypes never read hole cards or the board, which lets a
    # counterfactual chance salt resample their holes without perturbing
    # the replayed betting history.
    reads_cards = False

    name: str
    aggression: float = 0.226  # audited arena median
    fold_vs_bet: float = 0.5
    shove_rate: float = 0.0
    seed: int = 97

    policy_version = "scripted"

    def decide(self, table: Mapping[str, Any]) -> dict:
        allowed = table["allowedActions"]
        available = set(allowed["availableActions"])
        rollout = table.get("simulationRollout")
        key = (
            f"{self.seed}:{self.name}:{table['tableId']}:{table['street']}"
            f":{table['potChips']}:{table['selfSeatNumber']}"
        )
        if rollout is not None:
            key += f":rollout-{rollout}"
        roll = random.Random(key)
        if self.shove_rate > 0 and roll.random() < self.shove_rate:
            if "all-in" in available:
                return {
                    "action": "all-in",
                    "amount": allowed["allInToAmount"],
                    "message": "pressure",
                }
            if "raise" in available:
                return {
                    "action": "raise",
                    "amount": allowed["raiseRange"]["max"],
                    "message": "pressure",
                }
        facing = int(allowed.get("callChips") or 0) > 0
        if facing and roll.random() < self.fold_vs_bet and "fold" in available:
            return {"action": "fold", "message": "release"}
        if roll.random() < self.aggression:
            for action, range_name in (("raise", "raiseRange"), ("bet", "betRange")):
                if action in available and allowed[range_name] is not None:
                    low = int(allowed[range_name]["min"])
                    high = int(allowed[range_name]["max"])
                    target = min(
                        high,
                        max(low, round(low + 0.5 * (high - low) * roll.random())),
                    )
                    return {"action": action, "amount": target, "message": "value"}
        if "check" in available:
            return {"action": "check", "message": "wait"}
        if "call" in available:
            return {"action": "call", "message": "along"}
        return {"action": "fold", "message": "done"}


def calibrated_lineup(seed: int = 5) -> list[tuple[str, ScriptedAgent]]:
    """Opponent set spanning the audited arena distribution plus extremes."""

    return [
        ("median-bot", ScriptedAgent("median-bot", 0.226, 0.5, 0.0, seed)),
        ("tight-bot", ScriptedAgent("tight-bot", 0.10, 0.75, 0.0, seed + 1)),
        ("wild-bot", ScriptedAgent("wild-bot", 0.35, 0.30, 0.02, seed + 2)),
        ("station-bot", ScriptedAgent("station-bot", 0.15, 0.05, 0.0, seed + 3)),
    ]


__all__ = [
    "calibrated_lineup",
    "HandResult",
    "MatchResult",
    "RecordingPolicy",
    "ScriptedAgent",
    "SimulationError",
    "TableSimulator",
]
