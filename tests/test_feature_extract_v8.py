"""Behavioural, wiring, and invariant checks for the schema-3 assembler.

Expected values here are derived independently of the assembler wherever
possible: card-plane and history positions come from the public schema3
index API, kept scalars are compared against ``features_from_table``'s own
output (the assembler must forward, not recompute), strength/potential are
compared against direct sibling-module calls with identical arguments, and
the cost block is hand-computed from the E6 definitions.

The fuzz battery carries the impossible-by-construction invariants (house
rule: verify the probe): whatever random snapshot is fed in, the vector is
exactly ``schema3.INPUT_SIZE_V8`` long and finite, every card entry is
exactly 0.0 or 1.0 with plane sums fixed by the board length, the street
and board-tier one-hots each sum to exactly 1.0, every probability-like
feature sits in [0, 1], the equity slope equals its defining difference
exactly, and the neutral belief block sums to exactly 1.0. None of these
can fail unless the code is wrong.
"""

from __future__ import annotations

import math
import random
import statistics
import unittest
from collections.abc import Sequence

from devfun_poker_playground import schema3
from devfun_poker_playground.action_history import encode_action_history
from devfun_poker_playground.belief_provider import BeliefProviderError
from devfun_poker_playground.feature_extract_v8 import (
    FeatureExtractError,
    _hand_events,
    extract_features_v8,
)
from devfun_poker_playground.game_state import features_from_table
from devfun_poker_playground.hand_potential import hand_potential
from devfun_poker_playground.policy_features import FEATURE_NAMES
from devfun_poker_playground.preflop_percentiles import PREFLOP_PERCENTILES
from devfun_poker_playground.strength_metric import (
    canonical_class,
    strength_percentile,
)

# Fuzz constants, per the house rule on authored numbers: the seed is
# arbitrary but fixed (determinism rule); the case count and the potential
# trial count are ablatable — they buy runtime only, because the asserted
# invariants hold at any sample size by construction.
_FUZZ_SEED = 0x5C43A3
_FUZZ_CASES = 10
_FUZZ_POTENTIAL_TRIALS = 32

_STREET_BY_BOARD_LEN = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}

# Features whose values are probabilities, unit fractions, or binary flags
# — each must sit in [0, 1] for every conceivable input.
_UNIT_INTERVAL_FEATURES = (
    "street_preflop",
    "street_flop",
    "street_turn",
    "street_river",
    "position",
    "lead_position_unit",
    "pot_odds",
    "legal_fold",
    "legal_check_call",
    "legal_aggress",
    "hole_known_fraction",
    "strength_percentile",
    "equity_vs_top20",
    "equity_vs_top5",
    "hand_ppot",
    "hand_npot",
    "cost_call_eff",
    "cost_aggress_small_eff",
    "cost_aggress_large_eff",
    "cost_allin_eff",
    "card_reveal_expense",
    "branch_small_executable",
    "branch_large_executable",
    "last_aggressor_position_unit",
    "committed_stack_fraction",
    *schema3.BELIEF_FEATURE_NAMES,
)

_NON_NEGATIVE_FEATURES = (
    "player_count",
    "active_player_count",
    "spr",
    "raises_current_street",
    "opp_aggressive_actions",
    "opp_calls",
    "callers_of_current_bet",
    "hero_aggression_count",
    "opp_allin_count",
    "log_pot_bb",
    "log_stack_bb",
    "log_effective_stack_bb",
    "log_to_call_bb",
    "log_street_contribution_bb",
    "log_current_bet_bb",
    "log_min_raise_to_bb",
    "log_opp_stack_min_bb",
    "log_opp_stack_median_bb",
    "log_opp_stack_max_bb",
    "log_total_committed_bb",
)


def _event(
    street: str,
    seat: int | None,
    action: str,
    amount: int | None = None,
    *,
    to_amount: int | None = None,
) -> dict:
    summary: dict = {"seatNumber": seat, "action": action}
    if amount is not None:
        summary["amount"] = amount
    if to_amount is not None:
        summary["toAmount"] = to_amount
    return {"type": "ActionTaken", "street": street, "summary": summary}


def _blind(seat: int, amount: int) -> dict:
    return {
        "type": "BlindPosted",
        "street": "preflop",
        "summary": {"seatNumber": seat, "amount": amount},
    }


def _main_events() -> list[dict]:
    return [
        _blind(2, 50),
        _blind(3, 100),
        _event("preflop", 1, "call", 100),
        _event("preflop", 2, "call", 50),
        _event("preflop", 3, "check"),
        _event("flop", 2, "bet", 200),
        _event("flop", 1, "call", 200),
        _event("flop", 3, "fold"),
        _event("turn", 2, "bet", 200),
    ]


def _make_table(
    *,
    street: str = "turn",
    board: Sequence[str] = ("2c", "7d", "9h", "Qs"),
    hole: Sequence[str] = ("Ah", "Kd"),
    pot: int = 600,
    to_call: int = 200,
    hero_stack: int = 2000,
    hero_contribution: int = 0,
    hero_total_committed: int | None = 300,
    opp_stack: int = 1800,
    opp_committed: int = 400,
    opp_status: str = "Active",
    available: Sequence[str] = ("fold", "call", "raise"),
    bet_range: tuple[int, int] | None = None,
    raise_range: tuple[int, int] | None = (400, 900),
    all_in_to: int | None = None,
    recent_events: list[dict] | None = None,
    extra_seats: list[dict] | None = None,
) -> dict:
    hero_seat: dict = {
        "seatNumber": 1,
        "status": "Active",
        "stackChips": hero_stack,
        "currentBetChips": hero_contribution,
        "holeCards": list(hole),
    }
    if hero_total_committed is not None:
        hero_seat["totalCommittedChips"] = hero_total_committed
    seats = [
        hero_seat,
        {
            "seatNumber": 2,
            "status": opp_status,
            "stackChips": opp_stack,
            "currentBetChips": to_call,
            "totalCommittedChips": opp_committed,
            "holeCards": None,
        },
        {
            "seatNumber": 3,
            "status": "Folded",
            "stackChips": 500,
            "currentBetChips": 0,
            "holeCards": None,
        },
    ]
    if extra_seats:
        seats.extend(extra_seats)
    return {
        "id": "feature-extract-v8-test",
        "tableId": "feature-extract-v8-test",
        "street": street,
        "potChips": pot,
        "currentBet": hero_contribution + to_call,
        "boardCards": list(board),
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": 1,
        "seats": seats,
        "allowedActions": {
            "canFold": "fold" in available,
            "canCheck": "check" in available,
            "canCall": "call" in available,
            "canBet": "bet" in available,
            "canRaise": "raise" in available,
            "canAllIn": "all-in" in available,
            "callAmount": to_call,
            "callChips": to_call,
            "callToAmount": (hero_contribution + to_call) if to_call else None,
            "minBet": bet_range[0] if bet_range else None,
            "minRaiseTo": raise_range[0] if raise_range else None,
            "betRange": (
                {"min": bet_range[0], "max": bet_range[1]} if bet_range else None
            ),
            "raiseRange": (
                {"min": raise_range[0], "max": raise_range[1]}
                if raise_range
                else None
            ),
            "allInToAmount": all_in_to,
            "availableActions": list(available),
            "amountSemantics": "toAmount",
            "reasoningRequired": False,
        },
        "recentEvents": _main_events() if recent_events is None else recent_events,
    }


def _by_name(vector: tuple[float, ...]) -> dict[str, float]:
    return dict(zip(schema3.FEATURE_NAMES_V8, vector))


class MainFixtureTests(unittest.TestCase):
    """The turn fixture: 4-card board, a bet of 200 into hero, eff 1800."""

    table: dict
    vector: tuple[float, ...]
    values: dict[str, float]

    @classmethod
    def setUpClass(cls) -> None:
        cls.table = _make_table()
        cls.vector = extract_features_v8(cls.table)
        cls.values = _by_name(cls.vector)

    def test_vector_survives_require_vector(self) -> None:
        schema3.require_vector(self.vector)
        self.assertEqual(len(self.vector), schema3.INPUT_SIZE_V8)

    def test_card_planes_have_exactly_the_dealt_bits(self) -> None:
        expected_set = {
            "hole_Ah",
            "hole_Kd",
            "flop_2c",
            "flop_7d",
            "flop_9h",
            "turn_Qs",
        }
        for name in expected_set:
            self.assertEqual(self.values[name], 1.0, name)
        card_total = sum(
            self.values[f"{plane}_{code}"]
            for plane in schema3.CARD_PLANES
            for code in schema3.CARD_CODES
        )
        # Exactly the six dealt bits are set, so every other entry is 0.0.
        self.assertEqual(card_total, 6.0)
        turn_total = sum(
            self.values[f"turn_{code}"] for code in schema3.CARD_CODES
        )
        self.assertEqual(turn_total, 1.0)

    def test_kept_block_is_the_schema2_computation(self) -> None:
        base = dict(zip(FEATURE_NAMES, features_from_table(self.table)))
        for name in schema3.KEPT_FEATURE_NAMES:
            if name == "lead_position_unit":
                continue  # schema-2 never carried it in the base vector
            self.assertEqual(self.values[name], base[name], name)

    def test_hand_computed_scalars(self) -> None:
        self.assertEqual(self.values["pot_odds"], 200 / (600 + 200))
        self.assertEqual(self.values["spr"], 1800 / 600)
        self.assertEqual(self.values["street_turn"], 1.0)
        self.assertEqual(self.values["street_river"], 0.0)
        self.assertEqual(self.values["raises_current_street"], 1.0)
        # Balanced owned stacks and lead 0.0 -> the exact midpoint unit.
        self.assertEqual(self.values["lead_position_unit"], 0.5)

    def test_cost_block_hand_computed(self) -> None:
        # eff = min(2000, 1800) = 1800. The pot arm is the engine's
        # realized pot-fraction wager, call + frac * (pot + call):
        # small = min(200 + 0.5 * 800, 0.20 * 1800 = 360) = 360, to-amount
        # 360 clamped up to the 400 raise minimum; large = min(200 + 800,
        # 0.45 * 1800 = 810) = 810, inside [400, 900].
        self.assertEqual(self.values["cost_call_eff"], 200 / 1800)
        self.assertEqual(self.values["cost_aggress_small_eff"], 400 / 1800)
        self.assertEqual(self.values["cost_aggress_large_eff"], 810 / 1800)
        # No all-in offered: the deepest legal commitment is the 900 raise cap.
        self.assertEqual(self.values["cost_allin_eff"], 900 / 1800)
        self.assertEqual(self.values["branch_small_executable"], 1.0)
        self.assertEqual(self.values["branch_large_executable"], 1.0)

    def test_card_reveal_expense_hand_computed(self) -> None:
        # share 200/1800, one reveal remaining on the turn -> x 1/3.
        self.assertEqual(
            self.values["card_reveal_expense"], (200 / 1800) * (1 / 3)
        )

    def test_summary_block_hand_computed(self) -> None:
        # Whole-hand counts (F4): seat 2 bet the flop and the turn; seat 2
        # also called preflop. Hero (seat 1) never aggressed.
        self.assertEqual(self.values["opp_aggressive_actions"], 2.0)
        self.assertEqual(self.values["opp_calls"], 1.0)
        self.assertEqual(self.values["callers_of_current_bet"], 0.0)
        # Seat 2 aggressed last on the turn: ordinal distance 1 of 2 in
        # the sorted seat set [1, 2, 3] with hero at 1.
        self.assertEqual(self.values["last_aggressor_position_unit"], 0.5)
        self.assertEqual(self.values["hero_aggression_count"], 0.0)

    def test_table_block_hand_computed(self) -> None:
        # Only seat 2 is an active opponent (seat 3 folded).
        expected = math.log1p(1800 / 100)
        self.assertEqual(self.values["log_opp_stack_min_bb"], expected)
        self.assertEqual(self.values["log_opp_stack_median_bb"], expected)
        self.assertEqual(self.values["log_opp_stack_max_bb"], expected)
        self.assertEqual(self.values["opp_allin_count"], 0.0)
        self.assertEqual(self.values["log_total_committed_bb"], math.log1p(3.0))
        self.assertEqual(
            self.values["committed_stack_fraction"], 300 / (300 + 2000)
        )

    def test_strength_block_reuses_the_siblings(self) -> None:
        hole, board = ("Ah", "Kd"), ("2c", "7d", "9h", "Qs")
        self.assertEqual(
            self.values["strength_percentile"], strength_percentile(hole, board)
        )
        ppot, npot = hand_potential(hole, board, trials=400, seed=7)
        self.assertEqual(self.values["hand_ppot"], ppot)
        self.assertEqual(self.values["hand_npot"], npot)
        # Unpaired board, big-card hole: the fresh tier, one-hot.
        self.assertEqual(self.values["board_tier_fresh"], 1.0)
        self.assertEqual(self.values["board_tier_thin"], 0.0)
        self.assertEqual(self.values["board_tier_kicker"], 0.0)

    def test_equity_slope_is_the_exact_difference(self) -> None:
        self.assertEqual(
            self.values["equity_range_slope"],
            self.values["equity_vs_top20"] - self.values["equity_vs_top5"],
        )
        for name in ("equity_vs_top20", "equity_vs_top5"):
            self.assertGreaterEqual(self.values[name], 0.0)
            self.assertLessEqual(self.values[name], 1.0)

    def test_history_block_matches_the_encoder(self) -> None:
        base = schema3.feature_index(schema3.HISTORY_FEATURE_NAMES[0])
        block = self.vector[base : base + len(schema3.HISTORY_FEATURE_NAMES)]
        expected = encode_action_history(
            _hand_events(self.table),
            hero_seat_number=1,
            player_count=3,
            big_blind_chips=100,
            seat_numbers=(1, 2, 3),
        )
        self.assertEqual(block, expected)
        # Spot checks through the public index API.
        self.assertEqual(self.values["hist_turn5_occurred"], 1.0)
        self.assertEqual(self.values["hist_turn5_aggress"], 1.0)
        self.assertEqual(self.values["hist_turn5_size"], math.log1p(2.0))
        self.assertEqual(self.values["hist_turn5_actor"], 0.5)
        self.assertEqual(self.values["hist_turn4_occurred"], 0.0)
        self.assertEqual(self.values["hist_turn_overflow"], 0.0)

    def test_belief_block_defaults_to_the_uniform_prior(self) -> None:
        uniform = 1.0 / schema3.BELIEF_BUCKETS
        for name in schema3.BELIEF_FEATURE_NAMES:
            self.assertEqual(self.values[name], uniform)

    def test_two_calls_are_identical(self) -> None:
        self.assertEqual(extract_features_v8(self.table), self.vector)


class CostBranchTests(unittest.TestCase):
    def test_contribution_shifts_the_to_amounts(self) -> None:
        table = _make_table(hero_contribution=150)
        values = _by_name(extract_features_v8(table))
        # small target min(200 + 0.5 * 800, 360) = 360: to 150 + 360 = 510
        # in [400, 900] -> commit 360; large target min(200 + 800, 810) =
        # 810: to 960 clamps to the 900 ceiling -> commit 750.
        self.assertEqual(values["cost_aggress_small_eff"], 360 / 1800)
        self.assertEqual(values["cost_aggress_large_eff"], 750 / 1800)
        self.assertEqual(values["branch_small_executable"], 1.0)
        self.assertEqual(values["branch_large_executable"], 1.0)

    def test_no_legal_aggression_reads_nominal_costs(self) -> None:
        table = _make_table(available=("fold", "call"), raise_range=None)
        values = _by_name(extract_features_v8(table))
        self.assertEqual(values["branch_small_executable"], 0.0)
        self.assertEqual(values["branch_large_executable"], 0.0)
        # Nominal branch prices: min(200 + 0.5 * 800, 360) = 360 and
        # min(200 + 800, 810) = 810 over the 1800 effective stack.
        self.assertEqual(values["cost_aggress_small_eff"], 360 / 1800)
        self.assertEqual(values["cost_aggress_large_eff"], 810 / 1800)
        # Calling is the deepest legal commitment left.
        self.assertEqual(values["cost_allin_eff"], 200 / 1800)

    def test_legal_allin_reads_the_full_stack(self) -> None:
        table = _make_table(
            available=("fold", "call", "raise", "all-in"), all_in_to=2000
        )
        values = _by_name(extract_features_v8(table))
        self.assertEqual(values["cost_allin_eff"], 1.0)

    def test_colliding_branches_mark_large_not_executable(self) -> None:
        # Both targets (360 and 810) clamp to the same 300 range ceiling:
        # distinctness fails and only the small branch stays executable.
        table = _make_table(raise_range=(200, 300))
        values = _by_name(extract_features_v8(table))
        self.assertEqual(values["branch_small_executable"], 1.0)
        self.assertEqual(values["branch_large_executable"], 0.0)
        self.assertEqual(values["cost_aggress_small_eff"], 300 / 1800)
        self.assertEqual(values["cost_aggress_large_eff"], 300 / 1800)


class HistoryParseTests(unittest.TestCase):
    """The extractor's own uncapped hand parse (F3 + wager normalisation)."""

    def test_early_actions_survive_past_sixty_events(self) -> None:
        # F3: the telemetry journal parse caps at the last 60 raw events;
        # the extractor must not — the oldest actions of a long hand would
        # silently vanish, falsifying "truncation is recorded, never
        # silent". The two preflop records here are the OLDEST events and
        # a 60-capped parse would drop both.
        early = [
            _event("preflop", 2, "raise", to_amount=300),
            _event("preflop", 1, "call", 300),
        ]
        filler = [
            _event("flop", 2 if index % 2 else 3, "check")
            for index in range(70)
        ]
        events = [_blind(2, 50), _blind(3, 100), *early, *filler]
        self.assertGreater(len(events), 60)
        table = _make_table(
            street="flop",
            board=("2c", "7d", "9h"),
            to_call=0,
            available=("check", "bet"),
            bet_range=(100, 1500),
            raise_range=None,
            recent_events=events,
        )
        values = _by_name(extract_features_v8(table))
        self.assertEqual(values["hist_preflop4_occurred"], 1.0)
        self.assertEqual(values["hist_preflop4_aggress"], 1.0)
        self.assertEqual(values["hist_preflop4_size"], math.log1p(3.0))
        self.assertEqual(values["hist_preflop5_check_call"], 1.0)
        # The whole-hand summaries see the early raise too.
        self.assertEqual(values["opp_aggressive_actions"], 1.0)
        # Slot truncation stays the loud kind: 70 flop checks overflow.
        self.assertEqual(values["hist_flop_overflow"], 1.0)

    def test_wager_normalisation_prefers_the_carried_field(self) -> None:
        # Raw events carry aggressive prices in toAmount (amount is null)
        # and call prices in amount; each family prefers its own field and
        # falls back to the other; checks carry no wager at all.
        events = [
            _blind(2, 50),
            _blind(3, 100),
            _event("preflop", 2, "raise", to_amount=300),
            _event("preflop", 3, "call", 250),
            _event("preflop", 2, "raise", 250, to_amount=300),
            _event("preflop", 3, "call", 250, to_amount=300),
            _event("preflop", 1, "check", 400),
        ]
        table = _make_table(
            street="preflop",
            board=(),
            pot=150,
            to_call=100,
            raise_range=(200, 2000),
            recent_events=events,
        )
        values = _by_name(extract_features_v8(table))
        self.assertEqual(values["hist_preflop1_size"], math.log1p(3.0))
        self.assertEqual(values["hist_preflop2_size"], math.log1p(2.5))
        self.assertEqual(values["hist_preflop3_size"], math.log1p(3.0))
        self.assertEqual(values["hist_preflop4_size"], math.log1p(2.5))
        self.assertEqual(values["hist_preflop5_check_call"], 1.0)
        self.assertEqual(values["hist_preflop5_size"], 0.0)


class SummaryScopeTests(unittest.TestCase):
    """F4 whole-hand scope and the F10 hero-call exclusion."""

    def test_hero_aggression_counts_the_whole_hand(self) -> None:
        # F4: hero 3-bets preflop and faces a flop decision — schema 2's
        # hero_aggression_count was whole-hand (decision_engine.
        # _aggressive_events with street=None), and the kept feature must
        # read 1 on the flop, not 0.
        events = [
            _blind(2, 50),
            _blind(3, 100),
            _event("preflop", 2, "raise", to_amount=300),
            _event("preflop", 1, "raise", to_amount=900),
            _event("preflop", 2, "call", 600),
            _event("flop", 2, "check"),
        ]
        table = _make_table(
            street="flop",
            board=("2c", "7d", "9h"),
            to_call=0,
            available=("check", "bet"),
            bet_range=(100, 1500),
            raise_range=None,
            recent_events=events,
        )
        values = _by_name(extract_features_v8(table))
        self.assertEqual(values["hero_aggression_count"], 1.0)
        self.assertEqual(values["opp_aggressive_actions"], 1.0)
        self.assertEqual(values["opp_calls"], 1.0)
        # Current-street quantities stay current-street (documented): the
        # flop has no aggressor and nobody has called a flop bet.
        self.assertEqual(values["last_aggressor_position_unit"], 0.0)
        self.assertEqual(values["callers_of_current_bet"], 0.0)

    def test_callers_of_current_bet_excludes_hero(self) -> None:
        # F10: hero limps preflop, one opponent calls behind, no aggressor
        # — every call answers the posted blind, but hero's own call must
        # not count as a caller hero has to beat: 1, not 2.
        events = [
            _blind(2, 50),
            _blind(3, 100),
            _event("preflop", 1, "call", 100),
            _event("preflop", 2, "call", 50),
        ]
        table = _make_table(
            street="preflop",
            board=(),
            pot=300,
            to_call=0,
            available=("check", "bet"),
            bet_range=(100, 2000),
            raise_range=None,
            recent_events=events,
        )
        values = _by_name(extract_features_v8(table))
        self.assertEqual(values["callers_of_current_bet"], 1.0)
        self.assertEqual(values["opp_calls"], 1.0)


class SparseSeatTests(unittest.TestCase):
    def test_sparse_seats_resolve_actor_units_ordinally(self) -> None:
        # F1 mirror: the review's probe at the extractor level. Seats
        # {1, 4, 5} with hero in seat 4; seat 1 bets the turn. The old
        # modular unit read (1 - 4) % 3 == 0 — hero's own 0.0 — for both
        # the history actor channel and last_aggressor_position_unit.
        table = _make_table()
        table["seats"][0]["seatNumber"] = 4
        table["seats"][1]["seatNumber"] = 5
        table["seats"][2]["seatNumber"] = 1
        table["seats"][2]["status"] = "Active"
        table["selfSeatNumber"] = 4
        table["recentEvents"] = [
            _blind(5, 50),
            _blind(1, 100),
            _event("turn", 1, "bet", to_amount=200),
        ]
        values = _by_name(extract_features_v8(table))
        # sorted seat set [1, 4, 5], hero 4: seat 1 sits ordinal distance
        # 2 of a maximum 2 -> 1.0.
        self.assertEqual(values["last_aggressor_position_unit"], 1.0)
        self.assertEqual(values["hist_turn5_actor"], 1.0)


class TableBlockTests(unittest.TestCase):
    def test_allin_and_busted_opponents(self) -> None:
        extra = [
            {
                # All-in earlier in the hand: stack 0, chips committed.
                "seatNumber": 4,
                "status": "Active",
                "stackChips": 0,
                "currentBetChips": 0,
                "totalCommittedChips": 600,
                "holeCards": None,
            },
            {
                # Busted/empty: stack 0 and nothing committed -> excluded.
                "seatNumber": 5,
                "status": "Active",
                "stackChips": 0,
                "currentBetChips": 0,
                "totalCommittedChips": 0,
                "holeCards": None,
            },
        ]
        table = _make_table(extra_seats=extra)
        values = _by_name(extract_features_v8(table))
        logs = [math.log1p(0.0), math.log1p(1800 / 100)]
        self.assertEqual(values["opp_allin_count"], 1.0)
        self.assertEqual(values["log_opp_stack_min_bb"], min(logs))
        self.assertEqual(values["log_opp_stack_max_bb"], max(logs))
        # Even count: the median is the mean of the middle two.
        self.assertEqual(
            values["log_opp_stack_median_bb"], statistics.median(logs)
        )

    def test_lead_unit_falls_back_to_the_midpoint(self) -> None:
        # Every opponent folded: the lead reading raises, the engine's
        # documented fallback is the 0.5 midpoint.
        table = _make_table(
            opp_status="Folded",
            to_call=0,
            available=("check", "bet"),
            bet_range=(100, 1500),
            raise_range=None,
        )
        values = _by_name(extract_features_v8(table))
        self.assertEqual(values["lead_position_unit"], 0.5)


class StreetFixtureTests(unittest.TestCase):
    def test_preflop_fixture(self) -> None:
        table = _make_table(
            street="preflop",
            board=(),
            hole=("Ah", "Ad"),
            pot=150,
            to_call=100,
            raise_range=(200, 2000),
            recent_events=[_blind(2, 50), _blind(3, 100)],
        )
        values = _by_name(extract_features_v8(table))
        for plane in ("flop", "turn", "river"):
            plane_sum = sum(
                values[f"{plane}_{code}"] for code in schema3.CARD_CODES
            )
            self.assertEqual(plane_sum, 0.0, plane)
        self.assertEqual(
            values["strength_percentile"],
            PREFLOP_PERCENTILES[canonical_class(("Ah", "Ad"))],
        )
        self.assertEqual(values["hand_ppot"], 0.0)
        self.assertEqual(values["hand_npot"], 0.0)
        self.assertEqual(values["street_preflop"], 1.0)

    def test_river_fixture_plane_sums(self) -> None:
        table = _make_table(
            street="river",
            board=("2c", "7d", "9h", "Qs", "3s"),
            to_call=0,
            available=("check", "bet"),
            bet_range=(100, 1500),
            raise_range=None,
            recent_events=[],
        )
        values = _by_name(extract_features_v8(table))
        sums = tuple(
            sum(values[f"{plane}_{code}"] for code in schema3.CARD_CODES)
            for plane in schema3.CARD_PLANES
        )
        self.assertEqual(sums, (2.0, 3.0, 1.0, 1.0))
        # No future cards: potential is identically zero.
        self.assertEqual(values["hand_ppot"], 0.0)
        self.assertEqual(values["hand_npot"], 0.0)


class _SpikedProvider:
    """A valid non-uniform provider: mass split across the top two octiles."""

    def continuing_range_buckets(self, table, recent_actions):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5)


class _BrokenProvider:
    """Mass 2.0: must be rejected by the require_buckets gate."""

    def continuing_range_buckets(self, table, recent_actions):
        return (0.25,) * 8


class ProviderAndArgumentTests(unittest.TestCase):
    def test_custom_provider_lands_in_the_belief_slots(self) -> None:
        values = _by_name(
            extract_features_v8(_make_table(), belief_provider=_SpikedProvider())
        )
        buckets = tuple(
            values[name] for name in schema3.BELIEF_FEATURE_NAMES
        )
        self.assertEqual(buckets, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5))

    def test_broken_provider_fails_loud(self) -> None:
        with self.assertRaises(BeliefProviderError):
            extract_features_v8(_make_table(), belief_provider=_BrokenProvider())

    def test_equity_is_validated_then_ignored(self) -> None:
        table = _make_table()
        self.assertEqual(
            extract_features_v8(table, equity=0.7), extract_features_v8(table)
        )
        for bad in (1.5, -0.1, float("nan"), float("inf")):
            with self.assertRaises(FeatureExtractError):
                extract_features_v8(table, equity=bad)

    def test_missing_hole_cards_fail_loud(self) -> None:
        table = _make_table()
        table["seats"][0]["holeCards"] = None
        with self.assertRaises(ValueError):
            extract_features_v8(table)

    def test_missing_big_blind_fails_loud(self) -> None:
        table = _make_table()
        del table["bigBlindChips"]
        with self.assertRaises(ValueError):
            extract_features_v8(table)


class FuzzInvariantTests(unittest.TestCase):
    """Impossible-by-construction properties over random snapshots."""

    def _assert_invariants(self, vector: tuple[float, ...], board_len: int) -> None:
        schema3.require_vector(vector)
        self.assertEqual(len(vector), schema3.INPUT_SIZE_V8)
        self.assertTrue(all(math.isfinite(value) for value in vector))
        values = _by_name(vector)

        plane_expected = {
            "hole": 2.0,
            "flop": 3.0 if board_len >= 3 else 0.0,
            "turn": 1.0 if board_len >= 4 else 0.0,
            "river": 1.0 if board_len >= 5 else 0.0,
        }
        for plane in schema3.CARD_PLANES:
            entries = [values[f"{plane}_{code}"] for code in schema3.CARD_CODES]
            for entry in entries:
                self.assertIn(entry, (0.0, 1.0))
            self.assertEqual(sum(entries), plane_expected[plane], plane)

        street_sum = sum(
            values[f"street_{street}"] for street in schema3.HISTORY_STREETS
        )
        self.assertEqual(street_sum, 1.0)
        tier_sum = sum(values[name] for name in schema3.BOARD_TIER_FEATURE_NAMES)
        self.assertEqual(tier_sum, 1.0)

        for name in _UNIT_INTERVAL_FEATURES:
            self.assertGreaterEqual(values[name], 0.0, name)
            self.assertLessEqual(values[name], 1.0, name)
        for name in _NON_NEGATIVE_FEATURES:
            self.assertGreaterEqual(values[name], 0.0, name)

        self.assertEqual(
            values["equity_range_slope"],
            values["equity_vs_top20"] - values["equity_vs_top5"],
        )
        # The neutral prior is exact in binary floating point.
        self.assertEqual(
            math.fsum(values[name] for name in schema3.BELIEF_FEATURE_NAMES),
            1.0,
        )

        for street in schema3.HISTORY_STREETS:
            self.assertIn(values[f"hist_{street}_overflow"], (0.0, 1.0))
            for slot in range(schema3.HISTORY_SLOTS):
                prefix = f"hist_{street}{slot}"
                for channel in ("occurred", "fold", "check_call", "aggress"):
                    self.assertIn(values[f"{prefix}_{channel}"], (0.0, 1.0))
                self.assertGreaterEqual(values[f"{prefix}_size"], 0.0)
                self.assertGreaterEqual(values[f"{prefix}_actor"], 0.0)
                self.assertLessEqual(values[f"{prefix}_actor"], 1.0)

    def test_fuzzed_snapshots_hold_every_invariant(self) -> None:
        rng = random.Random(_FUZZ_SEED)
        streets = tuple(_STREET_BY_BOARD_LEN.values())
        actions = ("bet", "raise", "all-in", "call", "check", "fold")
        for case in range(_FUZZ_CASES):
            drawn = rng.sample(schema3.CARD_CODES, 7)
            board_len = rng.choice((0, 3, 4, 5))
            hole = tuple(drawn[:2])
            board = tuple(drawn[2 : 2 + board_len])
            to_call = rng.choice((0, 150))
            events = [
                _event(
                    rng.choice(streets),
                    rng.choice((1, 2, 3, None)),
                    rng.choice(actions),
                    rng.choice((None, rng.randrange(0, 500))),
                )
                for _ in range(rng.randrange(0, 5))
            ]
            extra_seats = []
            if rng.random() < 0.5:
                extra_seats.append(
                    {
                        "seatNumber": 4,
                        "status": "Active",
                        "stackChips": 0,
                        "currentBetChips": 0,
                        "totalCommittedChips": 600,
                        "holeCards": None,
                    }
                )
            table = _make_table(
                street=_STREET_BY_BOARD_LEN[board_len],
                board=board,
                hole=hole,
                pot=rng.randrange(100, 3000),
                to_call=to_call,
                hero_stack=rng.randrange(200, 5000),
                hero_total_committed=rng.randrange(0, 500),
                opp_stack=rng.randrange(200, 5000),
                opp_committed=rng.randrange(0, 500),
                available=(
                    ("fold", "call", "raise")
                    if to_call
                    else ("check", "bet")
                ),
                bet_range=None if to_call else (100, 2000),
                raise_range=(to_call + 100, 2500) if to_call else None,
                recent_events=events,
                extra_seats=extra_seats,
            )
            vector = extract_features_v8(
                table, potential_trials=_FUZZ_POTENTIAL_TRIALS
            )
            with self.subTest(case=case, board_len=board_len):
                self._assert_invariants(vector, board_len)


if __name__ == "__main__":
    unittest.main()
