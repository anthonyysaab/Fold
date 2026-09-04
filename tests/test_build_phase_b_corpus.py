"""Tests for ``tools/build_phase_b_corpus.py`` — the Phase B label pipeline.

The load-bearing validation is ``ArrangedReplayParityTests``: the pipeline's
whole honesty argument rests on ``_arranged`` replicating the stock deal +
chance-salt arithmetic bit-for-bit, so an arranged replay with no P3 seat is
required to be **byte-identical** to the stock salted replay across points,
branches, and rollouts (impossible by construction unless the replication is
exact). Everything else — the conditional hole swap, the E6 candidate set,
the corpus contract — is tested on top of that foundation.

Money safety: pure offline simulation; no Arena requests, no credentials.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from engine.poker_policy import AggressivePokerPolicy
from engine.schema3 import INPUT_SIZE_V8
from engine.strength_aware_opponent import (
    P3Decision,
    StrengthAwareAgent,
    load_fit,
)
from engine.table_simulator import (
    _FAMILY_BRANCHES,
    RecordingPolicy,
    ScriptedAgent,
    SimSeat,
    TableSimulator,
)
from tools.build_phase_b_corpus import (
    DEFAULT_ACCEPT_THRESHOLD,
    HeroRecorder,
    P3HoleSwap,
    P3PrefixRecord,
    P3SeatWrapper,
    P3SwapStats,
    PhaseBError,
    PhaseBHarvestSimulator,
    PhaseBReplaySimulator,
    _joint_plausibility,
    corpus_header,
    load_phase_b_corpus,
    validate_phase_b_rows,
    write_phase_b_corpus,
)


class _CapturingHarvest(PhaseBHarvestSimulator):
    """Keeps the counterfactual pass's arguments for post-hoc re-probing."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.captured: list[tuple] = []

    def _counterfactual_examples(
        self, initial_seats, button_index, hand_index, points, deck_for_test
    ):
        self.captured.append(
            (
                copy.deepcopy(initial_seats),
                button_index,
                hand_index,
                list(points),
                deck_for_test,
            )
        )
        return super()._counterfactual_examples(
            initial_seats, button_index, hand_index, points, deck_for_test
        )


class _SettlementRecordingReplay(PhaseBReplaySimulator):
    """Records every seat's holes and the full board at settlement.

    Settlement happens after the swap has fired on every path (snapshot or
    settle), so the recorded holes are the ones the rollout actually scored.
    """

    log: list[tuple[dict[str, tuple[str, str]], tuple[str, ...]]] = []

    def _settle(self, seats, board, events):
        result = super()._settle(seats, board, events)
        type(self).log.append(
            (
                {seat.agent_id: tuple(seat.hole_cards) for seat in seats},
                tuple(board),
            )
        )
        return result


def _hero_recorder() -> HeroRecorder:
    return HeroRecorder(
        AggressivePokerPolicy(equity_trials=20)
    )


_CARDBLIND_HARVEST: _CapturingHarvest | None = None
_P3_HARVEST: _CapturingHarvest | None = None


def _cardblind_harvest() -> _CapturingHarvest:
    """Shared no-P3 harvest with ``force_arranged`` — the parity fixture."""

    global _CARDBLIND_HARVEST
    if _CARDBLIND_HARVEST is None:
        recorder = _hero_recorder()
        simulator = _CapturingHarvest(
            small_blind=50,
            big_blind=100,
            starting_stack=6_000,
            seed=8,
            collect_counterfactuals=True,
            counterfactual_rollouts=2,
            hero_id="hero",
            hero_recorder=recorder,
            leg_name="parity",
            potential_trials=40,
            force_arranged=True,
        )
        simulator.play_match(
            [
                ("hero", recorder),
                ("median", ScriptedAgent("median", 0.226, 0.5, 0.0, seed=6)),
            ],
            hands=8,
        )
        _CARDBLIND_HARVEST = simulator
    return _CARDBLIND_HARVEST


def _p3_harvest() -> _CapturingHarvest:
    """Shared 3-max harvest with one P3 seat and one card-blind seat."""

    global _P3_HARVEST
    if _P3_HARVEST is None:
        recorder = _hero_recorder()
        wrapper = P3SeatWrapper(
            StrengthAwareAgent("p3", 0.226, 0.5, 0.0, 11, fit=load_fit())
        )
        simulator = _CapturingHarvest(
            small_blind=50,
            big_blind=100,
            starting_stack=6_000,
            seed=9,
            collect_counterfactuals=True,
            counterfactual_rollouts=2,
            hero_id="hero",
            hero_recorder=recorder,
            leg_name="p3-mini",
            potential_trials=40,
            record_likelihoods=True,
        )
        simulator.p3_wrapper = wrapper  # type: ignore[attr-defined]
        simulator.play_match(
            [
                ("hero", recorder),
                ("p3", wrapper),
                ("median", ScriptedAgent("median", 0.226, 0.5, 0.0, seed=6)),
            ],
            hands=10,
        )
        _P3_HARVEST = simulator
    return _P3_HARVEST


class ArrangedReplayParityTests(unittest.TestCase):
    """`_arranged` must replicate the stock deal + salt bit-for-bit."""

    def test_arranged_replay_is_byte_identical_to_stock_salt(self) -> None:
        """No-P3 arranged replays must equal stock salted replays exactly.

        Chips, submitted risk fraction, and the executed action are compared
        for every captured point, every legal family branch, and three
        rollouts. Any deviation in the replicated deal RNG, salt RNG, pool
        order, or reassignment order would flip at least one outcome.
        """

        simulator = _cardblind_harvest()
        self.assertTrue(simulator.captured)
        stock = TableSimulator(
            small_blind=50, big_blind=100, starting_stack=6_000, seed=8
        )
        compared = 0
        streets = set()
        for seats, button, hand_index, points, deck in simulator.captured:
            self.assertIsNone(deck)
            for point in points:
                streets.add(point.street)
                candidates = [
                    branch
                    for family in point.legal_families
                    for branch in _FAMILY_BRANCHES[family]
                ]
                for _, family, pot_fraction in candidates:
                    for rollout in range(3):
                        expected = stock._counterfactual_outcome(
                            seats, button, hand_index, point,
                            family, pot_fraction, rollout, None,
                        )
                        actual = simulator._counterfactual_outcome(
                            seats, button, hand_index, point,
                            family, pot_fraction, rollout, None,
                        )
                        self.assertEqual(
                            actual,
                            expected,
                            f"arranged replay diverged from stock salt at "
                            f"hand {hand_index} {family} rollout {rollout}",
                        )
                        compared += 1
        self.assertGreater(compared, 60, "parity sample too small to mean much")
        self.assertIn("preflop", streets)

    def test_arranged_layout_fixes_the_seen_and_moves_the_unseen(self) -> None:
        """Hero holes and revealed board fixed; future chance varies."""

        simulator = _cardblind_harvest()
        checked = 0
        for seats, _, hand_index, points, _ in simulator.captured:
            for point in points:
                arrangements = [
                    simulator._arranged(seats, hand_index, point, rollout)
                    for rollout in range(6)
                ]
                revealed = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[
                    point.street
                ]
                first = arrangements[0]
                for arrangement in arrangements[1:]:
                    self.assertEqual(
                        arrangement.holes[point.agent_id],
                        first.holes[point.agent_id],
                        "hero holes moved between rollouts",
                    )
                    self.assertEqual(
                        arrangement.board[:revealed],
                        first.board[:revealed],
                        "revealed board moved between rollouts",
                    )
                if revealed < 5:
                    self.assertGreater(
                        len({arr.board[revealed:] for arr in arrangements}),
                        1,
                        "future board frozen across rollouts",
                    )
                for arrangement in arrangements:
                    dealt = [
                        card
                        for hole in arrangement.holes.values()
                        for card in hole
                    ] + list(arrangement.board)
                    self.assertEqual(
                        len(dealt), len(set(dealt)), "card dealt twice"
                    )
                    # The unseen pool excludes everything dealt this rollout
                    # except the P3-style holes it exists to replace -- with
                    # no P3 seat, that is everything dealt.
                    self.assertFalse(set(arrangement.unseen) & set(dealt))
                checked += 1
        self.assertGreater(checked, 0)

    def test_deck_replication_is_verified_per_decision(self) -> None:
        """A wrong hand_index must fail the snapshot agreement check."""

        simulator = _cardblind_harvest()
        seats, _, hand_index, point, _ = _first_resolvable_point(simulator)
        # The recorder only retains the last table's captures, so rebuild the
        # verification input from the arrangement instead: tampering with the
        # hand index changes the deal, and the check must notice.
        arrangement = simulator._arranged(seats, hand_index, point, 0)
        fake_capture = {
            "selfSeatNumber": next(
                seat.seat_number for seat in seats if seat.agent_id == point.agent_id
            ),
            "seats": [
                {
                    "seatNumber": seat.seat_number,
                    "holeCards": list(arrangement.holes[seat.agent_id])
                    if seat.agent_id == point.agent_id
                    else None,
                }
                for seat in seats
            ],
            "boardCards": list(
                arrangement.board[: {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[
                    point.street
                ]]
            ),
        }
        simulator._verify_arrangement(arrangement, point, fake_capture)
        wrong = simulator._arranged(seats, hand_index + 1, point, 0)
        if wrong.holes[point.agent_id] != arrangement.holes[point.agent_id]:
            with self.assertRaises(PhaseBError):
                simulator._verify_arrangement(wrong, point, fake_capture)


def _first_resolvable_point(simulator: _CapturingHarvest):
    """The first captured decision whose continuation still has chance in it.

    A river decision's board is fully revealed, so "the future board varies"
    is vacuous there; every harvest here reliably captures a preflop or flop
    decision, and the guard keeps the test honest rather than lucky.
    """

    for seats, button, hand_index, points, deck in simulator.captured:
        for point in points:
            if point.street != "river":
                return seats, button, hand_index, point, deck
    raise AssertionError("no pre-river decision captured")


class ConditionalSwapTests(unittest.TestCase):
    """The V8_DESIGN §5 chance-salt rule for the card-aware P3 seat."""

    def test_p3_holes_resample_per_rollout_and_never_collide(self) -> None:
        simulator = _p3_harvest()
        self.assertTrue(simulator.captured)
        seats, button, hand_index, point, deck = _first_resolvable_point(simulator)
        self.assertIsNone(deck)
        simulator.replay_class = _SettlementRecordingReplay
        try:
            records = []
            for rollout in range(8):
                _SettlementRecordingReplay.log = []
                simulator._counterfactual_outcome(
                    seats, button, hand_index, point,
                    "check_call", None, rollout, None,
                )
                self.assertEqual(len(_SettlementRecordingReplay.log), 1)
                records.append(
                    (
                        *_SettlementRecordingReplay.log[0],
                        simulator._arranged(seats, hand_index, point, rollout),
                    )
                )
        finally:
            del simulator.replay_class
            _SettlementRecordingReplay.log = []
        hero_holes = {holes["hero"] for holes, _, _ in records}
        self.assertEqual(
            hero_holes,
            {records[0][2].holes["hero"]},
            "hero holes must stay fixed across rollouts",
        )
        p3_holes = {holes["p3"] for holes, _, _ in records}
        self.assertGreater(
            len(p3_holes), 1, "P3 holes frozen across rollouts (the 0016 defect)"
        )
        boards = {board for _, board, _ in records}
        self.assertGreater(len(boards), 1, "future board frozen across rollouts")
        for holes, board, arrangement in records:
            cards = [card for hole in holes.values() for card in hole]
            cards.extend(board)
            self.assertEqual(len(cards), len(set(cards)), "card dealt twice")
            # This rollout's swap can only have drawn from this rollout's
            # unseen pool -- the cards the hero could not see, minus what
            # this rollout dealt elsewhere.
            self.assertTrue(set(holes["p3"]) <= set(arrangement.unseen))
            self.assertEqual(tuple(board), arrangement.board)

    def test_rollouts_are_deterministic(self) -> None:
        simulator = _p3_harvest()
        seats, button, hand_index, point, _ = _first_resolvable_point(simulator)
        first = simulator._counterfactual_outcome(
            seats, button, hand_index, point, "check_call", None, 3, None
        )
        second = simulator._counterfactual_outcome(
            seats, button, hand_index, point, "check_call", None, 3, None
        )
        self.assertEqual(first, second)

    def test_rows_carry_per_decision_resample_stats(self) -> None:
        simulator = _p3_harvest()
        self.assertTrue(simulator.phase_b_rows)
        for row in simulator.phase_b_rows:
            stats = row["p3"]
            expected = len(row["branches"]) * row["rollouts"]  # one P3 seat
            self.assertEqual(stats["seats_resampled"], expected)
            self.assertEqual(stats["swaps_applied"], expected)
            self.assertEqual(
                stats["accepted"] + stats["fallbacks"], stats["seats_resampled"]
            )
            self.assertGreaterEqual(stats["tries"], stats["accepted"])

    def test_accepted_samples_meet_the_threshold(self) -> None:
        simulator = _p3_harvest()
        stats = simulator.p3_stats
        self.assertGreater(stats.accepted, 0)
        self.assertEqual(len(stats.accepted_likelihoods), stats.accepted)
        for joint in stats.accepted_likelihoods:
            self.assertGreaterEqual(joint, simulator.accept_threshold)
            self.assertLessEqual(joint, 1.0)

    def test_agent_never_degraded_and_prefix_recording_held(self) -> None:
        simulator = _p3_harvest()
        wrapper = simulator.p3_wrapper  # type: ignore[attr-defined]
        self.assertEqual(wrapper.agent.fallback_count, 0)
        self.assertEqual(wrapper.record_failures, 0)

    def test_organic_rows_pass_the_full_corpus_contract(self) -> None:
        simulator = _p3_harvest()
        report = validate_phase_b_rows(simulator.phase_b_rows)
        self.assertEqual(report["decisions"], len(simulator.phase_b_rows))
        self.assertGreater(report["branch_rows"], 0)
        for row in simulator.phase_b_rows:
            legal = row["context"]["legal_range"]
            for entry in row["branches"]:
                if entry["family"] == "aggress":
                    self.assertIsNotNone(legal)
                    self.assertGreaterEqual(entry["e6_to_amount"], legal[0])
                    self.assertLessEqual(entry["e6_to_amount"], legal[1])


class SwapUnitTests(unittest.TestCase):
    """Direct tests of the rejection sampler, off the simulator."""

    BOARD = ("7c", "8d", "Kh")

    @staticmethod
    def _decision(folded_price: float = 0.25) -> P3Decision:
        return P3Decision(
            street="flop",
            strength_percentile=0.5,  # replaced per candidate holding
            pot_odds=folded_price,
            bet_to_pot=folded_price / (1.0 - folded_price),
            texture=0.3,
            active_players=3,
            position_unit=0.5,
            to_call=100,
            pot=300,
        )

    def _wrapper(self, *, folded: bool) -> P3SeatWrapper:
        wrapper = P3SeatWrapper(
            StrengthAwareAgent("p3", 0.226, 0.5, 0.0, 3, fit=load_fit())
        )
        wrapper._table_id = "t1"
        wrapper.prefix = [
            P3PrefixRecord(
                decision=self._decision(), board=self.BOARD, folded=folded
            )
        ]
        return wrapper

    def _pool(self) -> list[str]:
        ranks = "23456789TJQA"
        pool = [f"{rank}{suit}" for rank in ranks for suit in "sh"]
        return [card for card in pool if card not in self.BOARD]

    def _swap(self, wrapper: P3SeatWrapper, threshold: float, tries: int = 6):
        stats = P3SwapStats(record_likelihoods=True)
        seat = SimSeat(seat_number=2, agent_id="p3", agent=wrapper, stack=6_000)
        swap = P3HoleSwap(
            table_id="t1",
            pool=self._pool(),
            p3_agent_ids=frozenset({"p3"}),
            rng_key="unit-key",
            threshold=threshold,
            max_tries=tries,
            stats=stats,
        )
        swap.apply([seat])
        return seat, stats

    def test_threshold_zero_accepts_the_first_draw(self) -> None:
        seat, stats = self._swap(self._wrapper(folded=False), threshold=0.0)
        self.assertEqual(stats.tries, 1)
        self.assertEqual(stats.accepted, 1)
        self.assertEqual(stats.fallbacks, 0)
        self.assertEqual(stats.seats_resampled, 1)
        self.assertEqual(len(set(seat.hole_cards)), 2)
        self.assertTrue(set(seat.hole_cards) <= set(self._pool()))

    def test_threshold_one_falls_back_and_counts_it(self) -> None:
        # The fitted fold probability is banded strictly inside (0, 1), so a
        # non-empty prefix makes every joint < 1; threshold 1.0 must reject
        # every draw, exhaust the tries, and count exactly one fallback.
        seat, stats = self._swap(self._wrapper(folded=False), threshold=1.0, tries=5)
        self.assertEqual(stats.tries, 5)
        self.assertEqual(stats.accepted, 0)
        self.assertEqual(stats.fallbacks, 1)
        self.assertEqual(stats.seats_resampled, 1)
        self.assertEqual(len(set(seat.hole_cards)), 2)

    def test_swap_refuses_a_seat_that_is_not_wrapped(self) -> None:
        stats = P3SwapStats()
        seat = SimSeat(
            seat_number=2,
            agent_id="p3",
            agent=ScriptedAgent("imposter", 0.2, 0.5, 0.0, 1),
            stack=6_000,
        )
        swap = P3HoleSwap(
            table_id="t1",
            pool=self._pool(),
            p3_agent_ids=frozenset({"p3"}),
            rng_key="k",
            threshold=0.5,
            max_tries=3,
            stats=stats,
        )
        with self.assertRaises(PhaseBError):
            swap.apply([seat])

    def test_joint_plausibility_narrows_toward_consistent_holdings(self) -> None:
        agent = StrengthAwareAgent("p3", 0.226, 0.5, 0.0, 3, fit=load_fit())
        continue_record = [
            P3PrefixRecord(
                decision=self._decision(), board=self.BOARD, folded=False
            )
        ]
        fold_record = [
            P3PrefixRecord(
                decision=self._decision(), board=self.BOARD, folded=True
            )
        ]
        strong, weak = ("Ks", "Kd"), ("2s", "3h")
        strong_continue = _joint_plausibility(agent, continue_record, strong)
        weak_continue = _joint_plausibility(agent, continue_record, weak)
        self.assertGreater(
            strong_continue,
            weak_continue,
            "a continue must make strong holdings more plausible",
        )
        strong_fold = _joint_plausibility(agent, fold_record, strong)
        weak_fold = _joint_plausibility(agent, fold_record, weak)
        self.assertGreater(
            weak_fold,
            strong_fold,
            "a fold must make weak holdings more plausible",
        )
        for value in (strong_continue, weak_continue, strong_fold, weak_fold):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertEqual(_joint_plausibility(agent, [], strong), 1.0)

    def test_p3_opponent_ids_refuses_unhandled_card_readers(self) -> None:
        simulator = PhaseBHarvestSimulator(seed=1)
        hero = SimSeat(seat_number=1, agent_id="hero", agent=object(), stack=100)
        wrapped = SimSeat(
            seat_number=2,
            agent_id="p3",
            agent=P3SeatWrapper(
                StrengthAwareAgent("p3", 0.226, 0.5, 0.0, 3, fit=load_fit())
            ),
            stack=100,
        )
        blind = SimSeat(
            seat_number=3,
            agent_id="median",
            agent=ScriptedAgent("median", 0.2, 0.5, 0.0, 1),
            stack=100,
        )
        self.assertEqual(
            simulator._p3_opponent_ids([hero, wrapped, blind], "hero"),
            frozenset({"p3"}),
        )
        bare = SimSeat(
            seat_number=4,
            agent_id="bare",
            agent=StrengthAwareAgent("bare", 0.226, 0.5, 0.0, 3, fit=load_fit()),
            stack=100,
        )
        with self.assertRaises(PhaseBError):
            simulator._p3_opponent_ids([hero, wrapped, blind, bare], "hero")


class E6BranchCandidateTests(unittest.TestCase):
    """State-dependent E6 sizing at the label site (V8_DESIGN §4)."""

    @staticmethod
    def _context(effective_stack: int, legal=(200, 6_000)) -> dict:
        return {
            "pot": 300,
            "to_call": 100,
            "contribution": 0,
            "effective_stack": effective_stack,
            "purse": 6_000,
            "legal_range": legal,
            "big_blind": 100,
        }

    def _candidates(self, context, families=("fold", "check_call", "aggress")):
        simulator = PhaseBHarvestSimulator(seed=1)
        point = SimpleNamespace(legal_families=families)
        return simulator._branch_candidates(point, context)

    def test_deep_stacks_reproduce_half_pot_and_pot(self) -> None:
        candidates = self._candidates(self._context(effective_stack=6_000))
        fractions = {
            label: fraction for label, _, fraction in candidates if fraction
        }
        self.assertAlmostEqual(fractions["aggress_small"], 0.5)
        self.assertAlmostEqual(fractions["aggress_large"], 1.0)

    def test_shallow_stacks_bind_on_the_stack_target(self) -> None:
        candidates = self._candidates(self._context(effective_stack=800))
        fractions = {
            label: fraction for label, _, fraction in candidates if fraction
        }
        # small: min(100 + 0.5*400, 0.20*800) = 160 -> (160-100)/400
        self.assertAlmostEqual(fractions["aggress_small"], 0.15)
        # large: min(100 + 1.0*400, 0.45*800) = 360 -> (360-100)/400
        self.assertAlmostEqual(fractions["aggress_large"], 0.65)

    def test_no_stated_range_emits_no_aggress_branch(self) -> None:
        candidates = self._candidates(self._context(6_000, legal=None))
        self.assertEqual(
            [label for label, _, _ in candidates], ["fold", "check_call"]
        )

    def test_families_limit_the_candidate_set(self) -> None:
        candidates = self._candidates(
            self._context(6_000), families=("fold", "check_call")
        )
        self.assertEqual(
            [label for label, _, _ in candidates], ["fold", "check_call"]
        )


def _synthetic_row(
    decision_id: str = "sim-1-0:hero:0",
    street: str = "flop",
    rewards: tuple[float, ...] = (1.5, -1.5),
    labels: tuple[str, ...] = ("check_call", "fold"),
    absorption: dict | None = None,
    p3: dict | None = None,
) -> dict:
    executed_by_label = {
        "fold": ["fold", None],
        "check_call": ["check", None],
        "aggress_small": ["raise", 300],
        "aggress_large": ["raise", 500],
    }
    branches = []
    for label, reward in zip(labels, rewards):
        entry = {
            "branch": label,
            "family": "aggress" if label.startswith("aggress") else label,
            "pot_fraction": 0.5 if label.startswith("aggress") else None,
            "reward_bb": reward,
            "outcome_bb": reward + 10.0,
            "risk_fraction": 0.2,
            "executed": executed_by_label.get(label, ["check", None]),
        }
        branches.append(entry)
    return {
        "decision_id": decision_id,
        "table_id": "sim-1-0",
        "harvest_leg": "unit",
        "policy_version": "sim-test",
        "street": street,
        "big_blind": 100,
        "purse_bb": 60.0,
        "inclusion_count": 2,
        "rollouts": 2,
        "context": {
            "pot": 300,
            "to_call": 100,
            "contribution": 0,
            "effective_stack": 5_900,
            "purse": 6_000,
            "legal_range": [200, 6_000],
        },
        "features": [0.0] * INPUT_SIZE_V8,
        "branches": branches,
        "branch_absorption": absorption
        if absorption is not None
        else {label: label for label in labels},
        "p3": p3
        if p3 is not None
        else {
            "seats_resampled": 4,
            "tries": 6,
            "accepted": 3,
            "fallbacks": 1,
            "swaps_applied": 4,
        },
    }


class CorpusContractTests(unittest.TestCase):
    """The v8-native corpus format: IO round-trip and the validator."""

    def test_round_trip_and_byte_determinism(self) -> None:
        rows = [
            _synthetic_row(),
            _synthetic_row(decision_id="sim-1-1:hero:2", street="turn"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unit.phase-b.jsonl.gz"
            write_phase_b_corpus(path, rows)
            first_bytes = path.read_bytes()
            reloaded = load_phase_b_corpus(path)
            self.assertEqual(reloaded, rows)
            write_phase_b_corpus(path, rows)
            self.assertEqual(
                path.read_bytes(), first_bytes, "rewrite is not byte-identical"
            )
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                header = json.loads(handle.readline())
            self.assertEqual(header, corpus_header())

    def test_loader_rejects_foreign_headers(self) -> None:
        cases = [
            {"kind": "not-phase-b"},
            {**corpus_header(), "corpus_schema_version": 999},
            {**corpus_header(), "feature_schema_version": 2},
            {**corpus_header(), "input_size": 142},
        ]
        for header in cases:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bad.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    handle.write(json.dumps(header) + "\n")
                with self.assertRaises(PhaseBError):
                    load_phase_b_corpus(path)

    def test_validator_reports_aggregates(self) -> None:
        rows = [
            _synthetic_row(),
            _synthetic_row(decision_id="sim-1-1:hero:2", street="turn"),
        ]
        report = validate_phase_b_rows(rows)
        self.assertEqual(report["decisions"], 2)
        self.assertEqual(report["branch_rows"], 4)
        self.assertEqual(
            report["branch_rows_per_street"], {"flop": 2, "turn": 2}
        )
        self.assertEqual(
            report["branch_label_counts"], {"check_call": 2, "fold": 2}
        )
        self.assertEqual(report["p3_resample_totals"]["fallbacks"], 2)
        self.assertAlmostEqual(report["p3_fallback_rate"], 2 / 8)

    def test_validator_rejects_contract_violations(self) -> None:
        breakages = {
            "empty corpus": [],
            "duplicate decision ids": [_synthetic_row(), _synthetic_row()],
            "uncentered rewards": [_synthetic_row(rewards=(1.0, -0.5))],
            "non-finite reward": [_synthetic_row(rewards=(math.inf, -1.5))],
            "unknown street": [_synthetic_row(street="6th")],
            "single branch": [
                _synthetic_row(rewards=(0.0,), labels=("fold",), absorption={"fold": "fold"})
            ],
            "unknown branch label": [
                _synthetic_row(labels=("check_call", "limp"))
            ],
            "absorption to non-emitted": [
                _synthetic_row(
                    absorption={
                        "check_call": "check_call",
                        "fold": "aggress_small",
                    }
                )
            ],
            "emitted branch not fixed under absorption": [
                _synthetic_row(
                    absorption={"check_call": "check_call", "fold": "check_call"}
                )
            ],
            "missing absorption": [_synthetic_row(absorption={})],
        }
        for name, rows in breakages.items():
            with self.assertRaises(PhaseBError, msg=name):
                validate_phase_b_rows(rows)
        short = _synthetic_row()
        short["features"] = [0.0] * (INPUT_SIZE_V8 - 1)
        with self.assertRaises(Exception, msg="short feature vector"):
            validate_phase_b_rows([short])
        bad_risk = _synthetic_row()
        bad_risk["branches"][0]["risk_fraction"] = 1.5
        with self.assertRaises(PhaseBError, msg="risk fraction outside [0,1]"):
            validate_phase_b_rows([bad_risk])
        shared_action = _synthetic_row()
        shared_action["branches"][1]["executed"] = list(
            shared_action["branches"][0]["executed"]
        )
        with self.assertRaises(PhaseBError, msg="shared executed action"):
            validate_phase_b_rows([shared_action])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
