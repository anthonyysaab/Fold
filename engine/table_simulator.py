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

from engine.hand_strength import _shared_evaluator, _treys_card
from engine.policy_features import LABELS
from engine.training_telemetry import TrainingExample

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
    # Branch-set diagnostics, summed across sessions: how many decisions
    # collapsed to a single executable action and were dropped, and the
    # histogram of emitted branch-set sizes.
    single_branch_groups: int = 0
    emitted_branch_counts: dict[int, int] = field(default_factory=dict)

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

# These are candidates, not the emitted set: a branch survives only if the
# engine executes it as an action no earlier branch already executes
# (`_probe_branch_set`). When two collide the surviving label is the one that
# names the executed action honestly -- a "fold" the engine turns into a free
# check is a check, and the smaller wager is the size the engine actually
# sized -- so this order decides which head slot carries the label.
_BRANCH_DEDUP_ORDER = ("check_call", "fold", "aggress_half_pot", "aggress_pot")


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
        # Branch-set diagnostics: group size stops being constant once
        # branches are emitted by executability and distinctness, and the
        # emitted sizes are the measurement that says how much of the old
        # label degeneracy was definitional.
        self.single_branch_groups = 0
        self.emitted_branch_counts: dict[int, int] = {}

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
            candidates = [
                branch
                for family in point.legal_families
                for branch in _FAMILY_BRANCHES[family]
            ]
            branches, absorption = self._emitted_branches(
                initial_seats,
                button_index,
                hand_index,
                point,
                candidates,
                deck_for_test,
            )
            if len(branches) < 2:
                # No choice exists, so the group carries no preference to
                # learn: every centred target is exactly zero by
                # construction. Dropping it here keeps it out of feature
                # normalization and out of the rollout budget.
                self.single_branch_groups += 1
                continue
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
                        branch_absorption=tuple(sorted(absorption.items())),
                    )
                )
        return examples

    def _emitted_branches(
        self,
        initial_seats: list[SimSeat],
        button_index: int,
        hand_index: int,
        point: _CounterfactualPoint,
        candidates: Sequence[tuple[str, str, float | None]],
        deck_for_test: Sequence[str] | None,
    ) -> tuple[list[tuple[str, str, float | None]], dict[str, str]]:
        """Keep the candidate branches that name distinct executed actions.

        Deduplication is on the **executed** action, never on the nominal
        family: the whole defect this closes is that the two disagree. One
        mechanism covers all three cases -- a `fold` the engine turns into a
        free check collides with `check_call`; an `aggress` with no legal
        bet or raise falls through to a check and collides the same way; and
        the two aggression sizes collide once the risk cap or the
        `raiseRange` clamp merges them.

        Returns the surviving branches and the absorption map -- every
        candidate label pointing at the survivor that carries its executed
        action. Consumers need that map to find the branch holding the value
        of an action whose own label was dropped.
        """

        executed = self._probe_branch_set(
            initial_seats, button_index, hand_index, point, candidates, deck_for_test
        )
        rank = {label: index for index, label in enumerate(_BRANCH_DEDUP_ORDER)}
        survivors: dict[tuple[str, int | None], str] = {}
        for label, _, _ in sorted(candidates, key=lambda entry: rank[entry[0]]):
            survivors.setdefault(executed[label], label)
        absorption = {label: survivors[executed[label]] for label, _, _ in candidates}
        keep = set(survivors.values())
        branches = [entry for entry in candidates if entry[0] in keep]
        self.emitted_branch_counts[len(branches)] = (
            self.emitted_branch_counts.get(len(branches), 0) + 1
        )
        return branches, absorption

    def _probe_branch_set(
        self,
        initial_seats: list[SimSeat],
        button_index: int,
        hand_index: int,
        point: _CounterfactualPoint,
        candidates: Sequence[tuple[str, str, float | None]],
        deck_for_test: Sequence[str] | None,
    ) -> dict[str, tuple[str, int | None]]:
        """Executed ``(action, amount)`` per candidate branch, in one replay.

        ``decide_forced`` is a pure function of the branch-point table, and
        that table is identical in every rollout: the chance salt resamples
        only the continuation, and the replayed prefix is deterministic. So
        the branch set is settled by a single prefix replay rather than by
        one full rollout per candidate, which is what saves the duplicate
        aggression rollouts. That invariant is load-bearing, so it is
        checked against reality rather than assumed: see
        ``test_table_simulator.BranchSetTests.test_probe_matches_every_rollout``,
        which asserts the probe's answer equals what each rollout submits.
        """

        seats = copy.deepcopy(initial_seats)
        target = next(seat for seat in seats if seat.agent_id == point.agent_id)
        probe = _BranchProbePolicy(
            target.agent,
            decision_ordinal=point.decision_ordinal,
            candidates=candidates,
            risk_fraction=point.proposed_risk_fraction,
        )
        target.agent = probe
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
        try:
            replay._play_hand(
                seats,
                button_index=button_index,
                hand_index=hand_index,
                result=result,
                deck_for_test=deck_for_test,
                chance_salt=(
                    0,
                    _REVEALED_BOARD.get(point.street, 0),
                    point.agent_id,
                ),
            )
        except _BranchProbeSignal as signal:
            return signal.executed
        raise SimulationError("branch probe did not reach its target decision")

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
    ) -> tuple[int, float, tuple[str, int | None] | None]:
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
        return (
            hand.chip_deltas[point.agent_id],
            forced.submitted_risk_fraction,
            forced.executed_action,
        )


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
        # Branch-set diagnostics live on the simulator, not the match result,
        # and each session builds a fresh simulator -- so without this the
        # drop rate the design requires reporting is computed and thrown away
        # once per session.
        total.single_branch_groups += simulator.single_branch_groups
        for size, count in simulator.emitted_branch_counts.items():
            total.emitted_branch_counts[size] = (
                total.emitted_branch_counts.get(size, 0) + count
            )
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


class _BranchProbeSignal(Exception):
    """Carries the executed action of every candidate branch, then unwinds.

    The probe only needs the hand's deterministic prefix, so it stops the
    replay at the branch point instead of playing a continuation whose
    result it would discard.
    """

    def __init__(self, executed: dict[str, tuple[str, int | None]]) -> None:
        super().__init__("branch probe reached its target decision")
        self.executed = executed


class _BranchProbePolicy:
    """Replay to one decision, price every candidate branch, and stop."""

    def __init__(
        self,
        policy: Any,
        *,
        decision_ordinal: int,
        candidates: Sequence[tuple[str, str, float | None]],
        risk_fraction: float,
    ) -> None:
        self.policy = policy
        self.decision_ordinal = decision_ordinal
        self.candidates = candidates
        self.risk_fraction = risk_fraction
        self.calls = 0

    def decide(self, table: Mapping[str, Any]) -> dict:
        original = self.policy.decide(table)
        if self.calls != self.decision_ordinal:
            self.calls += 1
            return original
        executed: dict[str, tuple[str, int | None]] = {}
        for label, family, pot_fraction in self.candidates:
            # Each candidate decides from its own copy of the policy state
            # the real rollout will hold at this point -- after the original
            # decide, before any forcing -- so one probe cannot leak tracker
            # or RNG movement into the next.
            probe = copy.deepcopy(self.policy)
            if hasattr(probe, "decide_forced"):
                payload = dict(
                    probe.decide_forced(table, family=family, pot_fraction=pot_fraction)
                )
            else:
                payload = _payload_for_family(table, family, self.risk_fraction)
            amount = payload.get("amount")
            executed[label] = (
                str(payload.get("action")),
                None if amount is None else int(amount),
            )
        raise _BranchProbeSignal(executed)


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
        # What the engine actually did with this branch, for the probe's
        # agreement check.
        self.executed_action: tuple[str, int | None] | None = None

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
        amount = payload.get("amount")
        self.executed_action = (
            str(payload.get("action")),
            None if amount is None else int(amount),
        )
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

    def _fold_probability(
        self, table: Mapping[str, Any], allowed: Mapping[str, Any]
    ) -> float:
        """Chance of folding when facing a wager.

        Constant here, and that constant is precisely the defect precondition
        P3 names: a fold frequency that ignores the size of the bet makes
        expected value linear in that size, so the largest legal wager
        dominates and any model trained against this archetype learns to
        overbet. :class:`TexturedAgent` overrides this.

        Implementations must not draw from the caller's RNG. The caller has
        already spent exactly one ``roll.random()`` for this decision, and
        consuming more would desynchronise counterfactual replay.
        """

        return self.fold_vs_bet

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
        if (
            facing
            and roll.random() < self._fold_probability(table, allowed)
            and "fold" in available
        ):
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


_RANK_ORDER = "23456789TJQKA"

# Half pot is the reference price: a villain laid these odds folds at its base
# frequency, and the size response is measured as a deviation from it.
_REFERENCE_PRICE = 1.0 / 3.0


def board_coordination(board_cards: Sequence[str]) -> float:
    """How connected a revealed board is, on a 0..1 scale.

    Reads only the public board, never a hole card, so an archetype keyed on
    this stays ``reads_cards = False`` and the counterfactual chance salt can
    still resample its holdings. The already-revealed board is held fixed
    across rollouts by that same salt, so this value is stable exactly where
    replay consistency requires it and moves only on genuinely new cards.
    """

    cards = [card for card in board_cards if len(card) >= 2]
    if not cards:
        return 0.0

    suits: dict[str, int] = {}
    ranks: list[int] = []
    for card in cards:
        suits[card[-1]] = suits.get(card[-1], 0) + 1
        position = _RANK_ORDER.find(card[0].upper())
        if position >= 0:
            ranks.append(position)

    flush = max(suits.values())
    suited = 0.0 if flush < 2 else 0.35 if flush == 2 else 0.8 if flush == 3 else 1.0

    distinct = sorted(set(ranks))
    paired = 1.0 if len(distinct) < len(ranks) else 0.0
    if len(distinct) < 2:
        connected = 0.0
    else:
        # Tightest window spanned by any two ranks; adjacent ranks score 1.0
        # and anything five apart or wider scores 0.
        span = min(
            distinct[index + 1] - distinct[index] for index in range(len(distinct) - 1)
        )
        connected = max(0.0, 1.0 - (span - 1) / 4.0)

    score = 0.45 * suited + 0.35 * connected + 0.20 * paired
    return min(1.0, max(0.0, score))


@dataclass
class TexturedAgent(ScriptedAgent):
    """Opponent that prices a call instead of folding at a fixed rate.

    This is precondition **P3**, phase one. :class:`ScriptedAgent` folds with a
    constant probability no matter what it is charged, which makes expected
    value linear in bet size: a larger wager buys the same fold equity at
    strictly more profit, and the villain's range never narrows, so the EV
    optimum is always the corner of the legal range. That is the measured
    cause of candidate v7-0001's preference for maximum-size aggression, and
    it is why no conclusion about sizing can be drawn against these opponents.

    Here the fold frequency rises with the price being laid and with how
    connected the board is, which puts a cost on overbetting and moves the
    optimum into the interior of the size range.

    Both inputs -- the wager and the revealed board -- are **public**, so
    ``reads_cards`` stays ``False`` and the chance salt still resamples this
    agent's hole cards. The label therefore remains a true ``Q(s, a)``. An
    archetype that read its own hole cards would have to be excluded from
    resampling, which would reintroduce the frozen-runout defect that closed
    candidate 0016; that is a separate, later problem and it is not this one.

    ``size_response`` and ``texture_response`` set how hard the archetype
    bites. At the defaults a villain facing a base rate of 0.5 folds 0.50 at
    half pot, about 0.60 at pot, and about 0.70 at two times pot.
    """

    size_response: float = 0.6
    texture_response: float = 0.15
    min_fold: float = 0.02
    max_fold: float = 0.95

    policy_version = "textured"

    def _fold_probability(
        self, table: Mapping[str, Any], allowed: Mapping[str, Any]
    ) -> float:
        call_chips = int(allowed.get("callChips") or 0)
        pot = int(table.get("potChips") or 0)
        total = pot + call_chips
        # Pot odds being laid: half pot prices at 1/3, pot at 1/2, two times
        # pot at 2/3. Larger wagers are worse prices and get folded on more.
        price = (call_chips / total) if total > 0 else _REFERENCE_PRICE

        texture = board_coordination(table.get("boardCards") or ())

        probability = (
            self.fold_vs_bet
            + self.size_response * (price - _REFERENCE_PRICE)
            + self.texture_response * texture
        )
        return min(self.max_fold, max(self.min_fold, probability))


def calibrated_lineup(seed: int = 5) -> list[tuple[str, ScriptedAgent]]:
    """Opponent set spanning the audited arena distribution plus extremes."""

    return [
        ("median-bot", ScriptedAgent("median-bot", 0.226, 0.5, 0.0, seed)),
        ("tight-bot", ScriptedAgent("tight-bot", 0.10, 0.75, 0.0, seed + 1)),
        ("wild-bot", ScriptedAgent("wild-bot", 0.35, 0.30, 0.02, seed + 2)),
        ("station-bot", ScriptedAgent("station-bot", 0.15, 0.05, 0.0, seed + 3)),
    ]


__all__ = [
    "board_coordination",
    "calibrated_lineup",
    "HandResult",
    "MatchResult",
    "RecordingPolicy",
    "ScriptedAgent",
    "SimulationError",
    "TableSimulator",
    "TexturedAgent",
]
