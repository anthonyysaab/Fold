"""Field strength-separation benchmark: the instrument before the result.

`V8_DESIGN.md` §2 requires the frozen field benchmark to be recomputed on
``strength_metric.strength_percentile``. These tests measure the measuring
tool first (house rule), against quantities that are known by construction
rather than by running it:

- **The decision-time board.** A miniature replay whose street-closing
  action carries a post-action snapshot that already shows the next
  street's cards. The tool must score the actor on the board it could
  actually see; if it read the snapshot it would score a flop decision on
  a turn board, and the assertion below pins the exact strength either
  way. This is the defect that motivated rebuilding the board from
  ``StreetDealt``.
- **Structural invariants that cannot hold in a real deal**: a card dealt
  twice, a holding intersecting the board, and a street whose board is
  the wrong length. Each must raise, never be swallowed as a skip.
- **The control gate discriminates.** A synthetic archive whose control
  strengths carry a large aggress/fold gap must FAIL the control gate,
  and one whose control strengths are balanced must PASS it. A gate that
  cannot fail is not a gate.
- **The reproduction gate discriminates**, in both directions, at the
  documented 2pp tolerance.
- **Known-answer arithmetic**: the pot fraction on the exact S14 payload
  quoted in the tool's docstring, percentile interpolation, Spearman on
  perfectly ordered and perfectly reversed inputs, and a separation whose
  value is fixed by the numbers that went in.
- **Seeded determinism**: identical seeds reproduce control holdings and
  bootstrap intervals exactly; a different seed moves the controls.

One real replay from the S13 archive is parsed end to end so the
miniatures cannot drift from the archive's actual shape.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from engine.strength_metric import strength_percentile
from tools.collect_foreign_play_data import _read_json, _unwrap_rpc
from tools.measure_field_separation import (
    Decision,
    FAMILIES,
    GATE_TOLERANCE,
    MIN_CI_HANDS,
    MeasurementInvariantError,
    PUBLISHED_FIELD_RATES,
    STREET_BOARD_SIZE,
    STREETS,
    _percentile,
    _spearman,
    bootstrap_cells,
    build_report,
    collection_alias,
    leaderboard_top,
    pot_fraction,
    render_markdown,
    replay_decisions,
    separation_of,
)

_S13 = Path("foreign play data") / "20260812T082057Z_poker-playground_s13_top15"
_REAL_REPLAY = _S13 / "raw" / "tables" / "cmspqme3qt6ct14cawl3trl69.json"

_FAST = 64


# ---------------------------------------------------------------------------
# miniature replay construction
# ---------------------------------------------------------------------------


def _action(
    sequence: int,
    street: str,
    seat: int,
    action: str,
    *,
    name: str,
    agent_id: str,
    snapshot_board: list[str],
    pot: int = 100,
    call: int = 0,
    to_amount: int | None = None,
    before: int = 0,
) -> dict:
    payload: dict = {
        "seatNumber": seat,
        "action": action,
        "agentName": name,
        "pot": pot,
        "callAmount": call,
        "actorCurrentBetBefore": before,
        "allowedActions": {"callChips": call},
    }
    if to_amount is not None:
        payload["toAmount"] = to_amount
    return {
        "type": "ActionTaken",
        "sequence": sequence,
        "street": street,
        "agentId": agent_id,
        "payload": payload,
        "snapshot": {"boardCards": list(snapshot_board)},
    }


def _replay(
    table_id: str,
    holes: dict[int, list[str]],
    events: list[dict],
    *,
    final_board: list[str] | None = None,
) -> dict:
    hole_event = {
        "type": "HoleCardsDealt",
        "sequence": 0,
        "street": "PreDeal",
        "payload": {
            "seats": [
                {"seatNumber": seat, "holeCards": cards}
                for seat, cards in sorted(holes.items())
            ]
        },
    }
    return {
        "table": {"id": table_id, "boardCards": final_board or [], "winners": []},
        "events": [hole_event, *events],
    }


def _street_dealt(sequence: int, street: str, board: list[str]) -> dict:
    return {
        "type": "StreetDealt",
        "sequence": sequence,
        "street": street,
        "payload": {"boardCards": list(board)},
    }


class PotFractionTests(unittest.TestCase):
    """The size formula the frozen 0.60x / 0.50x medians are on."""

    def test_reproduces_the_documented_s14_payload(self) -> None:
        # The exact opening payload quoted in the tool docstring: blinds
        # 1/2, pot 3, call 2, raise to 4 from a zero contribution. New
        # chips beyond the call = 2, pot + call = 5.
        payload = {
            "action": "raise",
            "pot": 3,
            "callAmount": 2,
            "toAmount": 4,
            "actorCurrentBetBefore": 0,
        }
        self.assertAlmostEqual(pot_fraction(payload), 0.4)

    def test_half_pot_and_pot_sized_bets_are_exact(self) -> None:
        half = {
            "action": "bet",
            "pot": 100,
            "callAmount": 0,
            "toAmount": 50,
            "actorCurrentBetBefore": 0,
        }
        full = dict(half, toAmount=100)
        self.assertAlmostEqual(pot_fraction(half), 0.5)
        self.assertAlmostEqual(pot_fraction(full), 1.0)

    def test_a_raise_over_a_bet_prices_only_the_new_chips(self) -> None:
        # Pot 100 already includes the opponent's 40 bet; hero raises to
        # 120 having posted nothing. New chips 120, call 40, wager 80,
        # denominator 140.
        payload = {
            "action": "raise",
            "pot": 100,
            "callAmount": 40,
            "toAmount": 120,
            "actorCurrentBetBefore": 0,
        }
        self.assertAlmostEqual(pot_fraction(payload), 80 / 140)

    def test_call_chips_fill_in_for_a_missing_call_amount(self) -> None:
        payload = {
            "action": "raise",
            "pot": 100,
            "toAmount": 120,
            "actorCurrentBetBefore": 0,
            "allowedActions": {"callChips": 40},
        }
        self.assertAlmostEqual(pot_fraction(payload), 80 / 140)

    def test_non_aggressive_and_unsized_actions_have_no_fraction(self) -> None:
        for action in ("fold", "check", "call"):
            self.assertIsNone(pot_fraction({"action": action, "pot": 100}))
        self.assertIsNone(pot_fraction({"action": "bet", "pot": 0, "toAmount": 5}))
        self.assertIsNone(pot_fraction({"action": "raise", "pot": 100}))

    def test_all_in_falls_back_to_the_allowed_target(self) -> None:
        payload = {
            "action": "all-in",
            "pot": 100,
            "callAmount": 0,
            "actorCurrentBetBefore": 0,
            "allowedActions": {"allInToAmount": 200, "callChips": 0},
        }
        self.assertAlmostEqual(pot_fraction(payload), 2.0)


class DecisionTimeBoardTests(unittest.TestCase):
    """The leak the tool exists to avoid, pinned to an exact number."""

    def _leaky_replay(self) -> dict:
        holes = {1: ["Ah", "Kh"], 2: ["2c", "3d"]}
        flop = ["Qh", "Jh", "7s"]
        turn = [*flop, "Th"]
        events = [
            _street_dealt(1, "Flop", flop),
            # The street-closing bet: its snapshot already shows the turn.
            _action(
                2,
                "Flop",
                1,
                "bet",
                name="hero",
                agent_id="a1",
                snapshot_board=turn,
                pot=100,
                to_amount=50,
            ),
            _action(
                3,
                "Flop",
                2,
                "fold",
                name="villain",
                agent_id="a2",
                snapshot_board=turn,
                pot=150,
            ),
            _street_dealt(4, "Turn", turn),
        ]
        return _replay("t-leak", holes, events, final_board=turn)

    def test_strength_uses_the_street_board_not_the_snapshot(self) -> None:
        rows, stats = replay_decisions(
            self._leaky_replay(), collection="mini", seed=1
        )
        self.assertEqual(len(rows), 2)
        flop = ["Qh", "Jh", "7s"]
        turn = [*flop, "Th"]
        hero = rows[0]
        self.assertEqual(hero.street, "flop")
        # AhKh is a big draw on the flop and the nut flush on the turn:
        # the two boards give very different strengths, so this assertion
        # cannot pass for the wrong reason.
        expected = strength_percentile(["Ah", "Kh"], flop)
        leaked = strength_percentile(["Ah", "Kh"], turn)
        self.assertAlmostEqual(hero.strength, expected)
        self.assertNotAlmostEqual(expected, leaked, places=2)
        self.assertEqual(stats["snapshot_board_leaked_forward"], 2)

    def test_a_snapshot_that_is_not_an_extension_is_an_invariant_error(
        self,
    ) -> None:
        replay = self._leaky_replay()
        replay["events"][2]["snapshot"]["boardCards"] = ["2h", "3h", "4h", "5h"]
        with self.assertRaises(MeasurementInvariantError):
            replay_decisions(replay, collection="mini", seed=1)


class StructuralInvariantTests(unittest.TestCase):
    """Impossible-by-construction states must raise, never skip."""

    def test_a_card_dealt_twice_raises(self) -> None:
        holes = {1: ["Ah", "Kh"], 2: ["Ah", "3d"]}
        replay = _replay(
            "t-dup",
            holes,
            [_action(1, "Preflop", 1, "fold", name="a", agent_id="a1",
                     snapshot_board=[])],
        )
        with self.assertRaises(MeasurementInvariantError):
            replay_decisions(replay, collection="mini", seed=1)

    def test_a_holding_on_the_board_raises(self) -> None:
        holes = {1: ["Ah", "Kh"], 2: ["2c", "3d"]}
        board = ["Ah", "Jh", "7s"]
        replay = _replay(
            "t-clash",
            holes,
            [
                _street_dealt(1, "Flop", board),
                _action(2, "Flop", 1, "bet", name="a", agent_id="a1",
                        snapshot_board=board, to_amount=50),
            ],
            final_board=board,
        )
        with self.assertRaises(MeasurementInvariantError):
            replay_decisions(replay, collection="mini", seed=1)

    def test_a_board_of_the_wrong_length_for_its_street_raises(self) -> None:
        holes = {1: ["Ah", "Kh"], 2: ["2c", "3d"]}
        board = ["Qh", "Jh", "7s"]
        replay = _replay(
            "t-len",
            holes,
            [
                _street_dealt(1, "Flop", board),
                # Labelled turn, but only the flop has been dealt.
                _action(2, "Turn", 1, "bet", name="a", agent_id="a1",
                        snapshot_board=board, to_amount=50),
            ],
            final_board=board,
        )
        with self.assertRaises(MeasurementInvariantError):
            replay_decisions(replay, collection="mini", seed=1)


class ParsingConventionTests(unittest.TestCase):
    def _mixed_replay(self) -> dict:
        holes = {1: ["Ah", "Ad"], 2: ["7c", "2d"], 3: ["Ks", "Kd"]}
        return _replay(
            "t-mix",
            holes,
            [
                _action(1, "Preflop", 1, "raise", name="alpha", agent_id="a1",
                        snapshot_board=[], pot=3, call=2, to_amount=6),
                {
                    "type": "TimeoutAction",
                    "sequence": 2,
                    "street": "Preflop",
                    "agentId": "a2",
                    "payload": {"seatNumber": 2, "action": "fold",
                                "agentName": "beta"},
                    "snapshot": {"boardCards": []},
                },
                _action(3, "Preflop", 3, "call", name="gamma", agent_id="a3",
                        snapshot_board=[], pot=9, call=4),
            ],
        )

    def test_timeouts_are_counted_but_never_scored(self) -> None:
        rows, stats = replay_decisions(
            self._mixed_replay(), collection="mini", seed=5
        )
        self.assertEqual(stats["timeout_actions"], 1)
        self.assertEqual(stats["action_taken"], 2)
        self.assertEqual([row.agent_name for row in rows], ["alpha", "gamma"])
        self.assertEqual([row.family for row in rows], ["aggress", "check_call"])

    def test_preflop_strength_is_the_committed_percentile_table(self) -> None:
        rows, _ = replay_decisions(
            self._mixed_replay(), collection="mini", seed=5
        )
        self.assertAlmostEqual(
            rows[0].strength, strength_percentile(["Ah", "Ad"], [])
        )

    def test_control_holdings_are_seeded_and_legal(self) -> None:
        replay = self._mixed_replay()
        first, _ = replay_decisions(replay, collection="mini", seed=5)
        again, _ = replay_decisions(replay, collection="mini", seed=5)
        other, _ = replay_decisions(replay, collection="mini", seed=6)
        self.assertEqual(
            [row.control_strength for row in first],
            [row.control_strength for row in again],
        )
        self.assertNotEqual(
            [row.control_strength for row in first],
            [row.control_strength for row in other],
        )
        for row in first:
            self.assertGreaterEqual(row.control_strength, 0.0)
            self.assertLessEqual(row.control_strength, 1.0)


class RealArchiveTests(unittest.TestCase):
    """The miniatures must not drift from the archive's real shape."""

    @classmethod
    def setUpClass(cls) -> None:
        # The Arena archive was quarantined on 2026-09-03 (DATA.md section 1.1);
        # like every other archive test, skip rather than fail without it.
        if not _REAL_REPLAY.exists():
            raise unittest.SkipTest(f"archive replay not present: {_REAL_REPLAY}")

    def test_one_real_replay_parses_end_to_end(self) -> None:
        replay = _unwrap_rpc(_read_json(_REAL_REPLAY))
        rows, stats = replay_decisions(replay, collection="s13", seed=3)
        self.assertGreater(len(rows), 0)
        self.assertEqual(stats["scored"], len(rows))
        for row in rows:
            self.assertIn(row.family, FAMILIES)
            self.assertIn(row.street, STREETS)
            self.assertGreaterEqual(row.strength, 0.0)
            self.assertLessEqual(row.strength, 1.0)
            if row.pot_fraction is not None:
                self.assertEqual(row.family, "aggress")

    def test_the_leaderboard_rank_field_ties_as_documented(self) -> None:
        top = leaderboard_top(_S13, 15)
        ranks = [row["rank"] for row in top]
        self.assertTrue(all(rank <= 15 for rank in ranks))
        # Documented in ``leaderboard_top``: dense ranks tie, so rank<=15
        # selects more than 15 agents. If the archive ever stops tying,
        # the docstring's "18 in S13" needs revisiting.
        self.assertGreater(len(top), 15)
        self.assertEqual(ranks, sorted(ranks))


class ArithmeticTests(unittest.TestCase):
    def test_separation_is_the_difference_of_means(self) -> None:
        self.assertAlmostEqual(separation_of(3.0, 4, 1.0, 4), 0.5)
        self.assertIsNone(separation_of(3.0, 4, 0.0, 0))
        self.assertIsNone(separation_of(0.0, 0, 1.0, 4))

    def test_percentile_interpolates(self) -> None:
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(_percentile(values, 0.0), 0.0)
        self.assertAlmostEqual(_percentile(values, 1.0), 4.0)
        self.assertAlmostEqual(_percentile(values, 0.5), 2.0)
        self.assertAlmostEqual(_percentile(values, 0.25), 1.0)
        self.assertAlmostEqual(_percentile([7.0], 0.3), 7.0)

    def test_spearman_known_answers(self) -> None:
        rising = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(_spearman(rising, rising), 1.0)
        self.assertAlmostEqual(_spearman(rising, rising[::-1]), -1.0)
        self.assertIsNone(_spearman(rising, [1.0, 1.0, 1.0, 1.0]))
        self.assertIsNone(_spearman([1.0, 2.0], [1.0, 2.0]))

    def test_bootstrap_is_seeded_and_brackets_the_point_estimate(self) -> None:
        cells = {
            "x|overall": [
                (index, 0.8, 1.0, 0.2, 1.0, 0.0) for index in range(40)
            ]
        }
        first = bootstrap_cells(cells, 40, resamples=200, seed=11)
        again = bootstrap_cells(cells, 40, resamples=200, seed=11)
        self.assertEqual(first, again)
        entry = first["x|overall"]
        self.assertAlmostEqual(entry["separation"], 0.6)
        low, high = entry["separation_ci95"]
        self.assertLessEqual(low, entry["separation"])
        self.assertLessEqual(entry["separation"], high)
        self.assertEqual(entry["decisions"], 80)
        self.assertAlmostEqual(entry["fold_rate"], 0.5)
        self.assertAlmostEqual(entry["aggression_rate"], 0.5)

    def test_a_cell_with_no_folds_has_no_separation(self) -> None:
        cells = {"y|overall": [(0, 0.8, 1.0, 0.0, 0.0, 0.0)]}
        entry = bootstrap_cells(cells, 1, resamples=50, seed=3)["y|overall"]
        self.assertIsNone(entry["separation"])
        self.assertIsNone(entry["separation_ci95"])
        self.assertEqual(entry["bootstrap"]["undefined_separation_resamples"], 50)

    def test_a_cell_living_in_too_few_hands_has_its_interval_withheld(
        self,
    ) -> None:
        # Every aggress and every fold sits in the same three hands. The
        # multiplicity cancels in every resample, so a naive bootstrap
        # returns a point-tight interval that means nothing. This is the
        # `us|turn` shape seen on the real archive.
        cells = {
            "z|turn": [(index, 1.8, 2.0, 0.3, 1.0, 0.0) for index in range(3)]
        }
        entry = bootstrap_cells(cells, 200, resamples=200, seed=7)["z|turn"]
        self.assertAlmostEqual(entry["separation"], 0.6)
        self.assertIsNone(entry["separation_ci95"])
        self.assertEqual(entry["hands"], 3)
        self.assertEqual(entry["hands_with_aggress"], 3)
        self.assertEqual(entry["hands_with_fold"], 3)
        self.assertIn("separation", entry["bootstrap"]["ci_withheld"])
        self.assertLess(entry["hands"], MIN_CI_HANDS)

    def test_a_cell_with_enough_hands_keeps_its_interval(self) -> None:
        cells = {
            "z|turn": [
                (index, 1.8, 2.0, 0.3, 1.0, 0.0) for index in range(MIN_CI_HANDS)
            ]
        }
        entry = bootstrap_cells(cells, 200, resamples=200, seed=7)["z|turn"]
        self.assertIsNotNone(entry["separation_ci95"])
        self.assertEqual(entry["bootstrap"]["ci_withheld"], {})


# ---------------------------------------------------------------------------
# the two gates
# ---------------------------------------------------------------------------


def _synthetic(
    *,
    n_hands: int,
    fold_rate: float,
    aggression_rate: float,
    bet_fraction: float,
    control_gap: float,
    collection: str = "20260815T210237Z_poker-playground_s14_top15",
    agent: tuple[str, str] = ("agent-1", "alpha"),
    per_hand: int = 100,
) -> list[Decision]:
    """Hands whose rates and control gap are fixed by construction.

    ``per_hand`` is 100 so a rate like the published 0.558 survives the
    integer split to within 0.002 — well inside the 2pp gate tolerance
    the tests below are probing.
    """

    n_fold = round(fold_rate * per_hand)
    n_aggress = round(aggression_rate * per_hand)
    n_call = per_hand - n_fold - n_aggress
    rows: list[Decision] = []
    agent_id, agent_name = agent
    for hand in range(n_hands):
        sequence = 0
        plan = (
            [("fold", 0.2, 0.5 - control_gap / 2)] * n_fold
            + [("aggress", 0.8, 0.5 + control_gap / 2)] * n_aggress
            + [("check_call", 0.5, 0.5)] * n_call
        )
        for family, strength, control in plan:
            sequence += 1
            rows.append(
                Decision(
                    collection=collection,
                    table_id=f"hand-{hand}",
                    sequence=sequence,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    street="preflop" if sequence % 2 else "flop",
                    family=family,
                    # A little seeded-free variation so the bootstrap has
                    # something to resample; the means stay exact.
                    strength=strength + (0.05 if hand % 2 else -0.05),
                    control_strength=control + (0.05 if hand % 2 else -0.05),
                    pot_fraction=bet_fraction if family == "aggress" else None,
                )
            )
    return rows


def _leaderboards(
    collection: str, agent_id: str, name: str
) -> dict[str, list[dict]]:
    return {
        collection: [
            {
                "agent_id": agent_id,
                "name": name,
                "rank": 1,
                "total_score": 100,
                "adjusted_bb100": 10.0,
            }
        ]
    }


class ValidationGateTests(unittest.TestCase):
    COLLECTION = "20260815T210237Z_poker-playground_s14_top15"

    def _report(self, **kwargs) -> dict:
        rows = _synthetic(**kwargs)
        return build_report(
            rows,
            {self.COLLECTION: __import__("collections").Counter()},
            _leaderboards(self.COLLECTION, "agent-1", "alpha"),
            seed=17,
            resamples=_FAST,
            roots=[Path("nowhere")],
        )

    def test_reproduction_gate_passes_on_the_published_rates(self) -> None:
        report = self._report(
            n_hands=20,
            fold_rate=PUBLISHED_FIELD_RATES["fold_rate"],
            aggression_rate=PUBLISHED_FIELD_RATES["aggression_rate"],
            bet_fraction=PUBLISHED_FIELD_RATES["median_bet_pot_fraction"],
            control_gap=0.0,
        )
        gate = report["validation"]["reproduction_gate"]
        self.assertTrue(gate["pass"], gate)
        self.assertTrue(report["validation"]["pass"])

    def test_reproduction_gate_fails_when_the_fold_rate_drifts(self) -> None:
        report = self._report(
            n_hands=20,
            fold_rate=0.3,
            aggression_rate=PUBLISHED_FIELD_RATES["aggression_rate"],
            bet_fraction=PUBLISHED_FIELD_RATES["median_bet_pot_fraction"],
            control_gap=0.0,
        )
        checks = report["validation"]["reproduction_gate"]["collections"][
            self.COLLECTION
        ]["checks"]
        self.assertFalse(checks["fold_rate"]["pass"])
        self.assertFalse(report["validation"]["pass"])

    def test_reproduction_gate_fails_when_the_bet_size_drifts(self) -> None:
        report = self._report(
            n_hands=20,
            fold_rate=PUBLISHED_FIELD_RATES["fold_rate"],
            aggression_rate=PUBLISHED_FIELD_RATES["aggression_rate"],
            bet_fraction=1.25,
            control_gap=0.0,
        )
        checks = report["validation"]["reproduction_gate"]["collections"][
            self.COLLECTION
        ]["checks"]
        self.assertFalse(checks["median_bet_pot_fraction"]["pass"])

    def test_the_tolerance_is_the_documented_two_points(self) -> None:
        report = self._report(
            n_hands=20,
            fold_rate=PUBLISHED_FIELD_RATES["fold_rate"] + 0.015,
            aggression_rate=PUBLISHED_FIELD_RATES["aggression_rate"],
            bet_fraction=PUBLISHED_FIELD_RATES["median_bet_pot_fraction"],
            control_gap=0.0,
        )
        checks = report["validation"]["reproduction_gate"]["collections"][
            self.COLLECTION
        ]["checks"]
        self.assertLessEqual(abs(checks["fold_rate"]["delta"]), GATE_TOLERANCE)
        self.assertTrue(checks["fold_rate"]["pass"])

    def test_control_gate_fails_on_a_manufactured_control_gap(self) -> None:
        report = self._report(
            n_hands=20,
            fold_rate=PUBLISHED_FIELD_RATES["fold_rate"],
            aggression_rate=PUBLISHED_FIELD_RATES["aggression_rate"],
            bet_fraction=PUBLISHED_FIELD_RATES["median_bet_pot_fraction"],
            control_gap=0.4,
        )
        control = report["validation"]["control_gate"]
        self.assertFalse(control["pass"], control)
        self.assertFalse(report["validation"]["pass"])

    def test_an_unresolvable_control_cell_is_not_a_failure(self) -> None:
        # Our own 98 real decisions live in 9 fold-hands, so the `us`
        # control cell has no resolvable interval. That is UNRESOLVED, not
        # evidence of a broken instrument: the gate binds on the field
        # cell, which is the one the benchmark rests on.
        field = _synthetic(
            n_hands=20,
            fold_rate=PUBLISHED_FIELD_RATES["fold_rate"],
            aggression_rate=PUBLISHED_FIELD_RATES["aggression_rate"],
            bet_fraction=PUBLISHED_FIELD_RATES["median_bet_pot_fraction"],
            control_gap=0.0,
        )
        ours = _synthetic(
            n_hands=2,
            fold_rate=0.5,
            aggression_rate=0.5,
            bet_fraction=0.5,
            control_gap=0.0,
            agent=("agent-us", "Fold-ver-4"),
            per_hand=4,
        )
        report = build_report(
            field + ours,
            {self.COLLECTION: __import__("collections").Counter()},
            _leaderboards(self.COLLECTION, "agent-1", "alpha"),
            seed=31,
            resamples=_FAST,
            roots=[Path("nowhere")],
        )
        control = report["validation"]["control_gate"]
        self.assertEqual(
            control["checks"]["control_us|overall"]["status"], "unresolved"
        )
        self.assertEqual(
            control["checks"]["control_field|overall"]["status"], "pass"
        )
        self.assertTrue(control["pass"], control)

    def test_control_gate_passes_when_the_control_carries_no_signal(
        self,
    ) -> None:
        report = self._report(
            n_hands=20,
            fold_rate=PUBLISHED_FIELD_RATES["fold_rate"],
            aggression_rate=PUBLISHED_FIELD_RATES["aggression_rate"],
            bet_fraction=PUBLISHED_FIELD_RATES["median_bet_pot_fraction"],
            control_gap=0.0,
        )
        control = report["validation"]["control_gate"]
        self.assertTrue(control["pass"], control)


class ReportShapeTests(unittest.TestCase):
    COLLECTION = "20260815T210237Z_poker-playground_s14_top15"

    def _rows(self) -> list[Decision]:
        field = _synthetic(
            n_hands=20,
            fold_rate=0.6,
            aggression_rate=0.2,
            bet_fraction=0.6,
            control_gap=0.0,
            collection=self.COLLECTION,
            agent=("agent-1", "alpha"),
        )
        ours = _synthetic(
            n_hands=20,
            fold_rate=0.1,
            aggression_rate=0.7,
            bet_fraction=0.5,
            control_gap=0.0,
            collection=self.COLLECTION,
            agent=("agent-us", "Fold-ver-4"),
        )
        return field + ours

    def _report(self) -> dict:
        # Our agent really is on the S14 top-15 board (rank 10), so the
        # per-agent block must be able to carry it alongside the field.
        leaderboards = _leaderboards(self.COLLECTION, "agent-1", "alpha")
        leaderboards[self.COLLECTION].append(
            {
                "agent_id": "agent-us",
                "name": "Fold-ver-4",
                "rank": 2,
                "total_score": 50,
                "adjusted_bb100": 5.0,
            }
        )
        return build_report(
            self._rows(),
            {self.COLLECTION: __import__("collections").Counter()},
            leaderboards,
            seed=23,
            resamples=_FAST,
            roots=[Path("nowhere")],
        )

    def test_our_decisions_are_excluded_from_the_field(self) -> None:
        report = self._report()
        self.assertAlmostEqual(report["field"]["overall"]["fold_rate"], 0.6)
        self.assertAlmostEqual(report["us"]["overall"]["fold_rate"], 0.1)
        self.assertAlmostEqual(
            report["all_agents"]["overall"]["fold_rate"], 0.35
        )
        self.assertEqual(report["counts"]["our_agent_ids"], ["agent-us"])

    def test_every_street_and_scope_block_is_present(self) -> None:
        report = self._report()
        for scope in ("field", "us", "all_agents"):
            self.assertIn("overall", report[scope])
            for street in STREETS:
                self.assertIn(street, report[scope]["per_street"])
        self.assertEqual(
            set(STREET_BOARD_SIZE), set(STREETS)
        )

    def test_per_collection_slices_match_the_pooled_block(self) -> None:
        # A single-collection archive: the S14 slice must be identical to
        # the pooled field figure, or the alias routing is wrong.
        report = self._report()
        pooled = report["field"]["overall"]
        sliced = report["field_by_collection"]["s14"]["overall"]
        self.assertEqual(pooled["decisions"], sliced["decisions"])
        self.assertAlmostEqual(pooled["separation"], sliced["separation"])
        self.assertEqual(list(report["us_by_collection"]), ["s14"])

    def test_collection_alias_extracts_the_season(self) -> None:
        self.assertEqual(collection_alias(self.COLLECTION), "s14")
        self.assertEqual(
            collection_alias("20260812T082057Z_poker-playground_s13_top15"),
            "s13",
        )
        self.assertEqual(collection_alias("weird-name"), "weird-name")

    def test_markdown_renders_without_raising(self) -> None:
        text = render_markdown(self._report())
        self.assertIn("VALIDATION GATE", text)
        self.assertIn("Fold-ver-4", text)
        self.assertIn("separation", text)

    def test_the_report_is_deterministic_for_a_fixed_seed(self) -> None:
        self.assertEqual(self._report(), self._report())


if __name__ == "__main__":
    unittest.main()
