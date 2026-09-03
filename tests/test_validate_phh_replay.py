"""Tests for the PHH replay validator (``tools.validate_phh_replay``).

The instrument must be able to fail before its results are believed
(DECISIONS §3.5), and it must fail in the ONE place it is allowed to
turn an inequality into a pass. So the cases here are, in order:

* a clean hand passes every invariant with class ``equal``;
* a corrupted finishing stack fails invariant 1 with class ``mismatch``
  — corrupted on a DIFFERENT hand and by a different shape from the
  tool's own inline self-check, so this is a test, not a re-run of the
  implementation;
* the real half-chip-split hand ``pluribus/102/0`` classifies
  ``half_chip_split`` with the exact ±0.5 seat deltas, and the same
  replay with the odd chip's seat removed from the winners fails
  ``mismatch`` — the clause that refuses an odd chip paid to the wrong
  seat;
* the class the tool used to accept — a file whose stacks sum ONE SHORT
  of the starting sum, with the replay a whole chip up on one seat — is
  now a ``mismatch`` and a chip-conservation failure. No file in the
  Pluribus corpus has that shape; the old classifier manufactured it by
  casting ``Decimal('10112.5')`` to ``int``;
* invariant 7 fails when a row goes missing (rows, per-street rows and
  free-spot rows are asserted, not merely printed);
* the sample is a seeded random draw, not a head slice;
* ``--output`` is required and refuses to overwrite a frozen record, and
  the ``.md`` sibling is written from the same report.

Every test that needs pokerkit is skipped when it is not installed, and
``HalfChipSplitTests`` skips when the PHH clone is not on disk (it reads
``pluribus/102/0.phh`` rather than an inline copy, so it measures the
corpus the gate actually runs on).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

from tools.validate_phh_replay import (
    _label_availability,
    _selected,
    check_hand,
    main,
    validate_roots,
)

_POKERKIT_AVAILABLE = importlib.util.find_spec("pokerkit") is not None

_PLURIBUS_ROOT = Path("phh-dataset/data/pluribus")
#: The half-chip split the corpus actually contains: the file records
#: 10112.5 on both winners (p1, p5) and its stacks sum to exactly 60000.
_HALF_CHIP_FILE = _PLURIBUS_ROOT / "102" / "0.phh"

_RECORDED_STARTING = [10000, 10000, 10000, 10000, 10000, 10000]
_RECORDED_FINISHING = [10310, 9900, 10000, 9790, 10000, 10000]

_SIDE_POT_STARTING = [10000, 10000, 500]
_SIDE_POT_FINISHING = [15000, 5500, 0]


def _load_half_chip_hand():
    """The real half-chip hand: (replay, starting, finishing) from disk."""

    import pokerkit

    from tools.phh_replay import replays_from_path

    with _HALF_CHIP_FILE.open("rb") as stream:
        history = pokerkit.HandHistory.load(stream)
    pairs = list(replays_from_path(_HALF_CHIP_FILE))
    return (
        pairs[0][1],
        list(history.starting_stacks),
        list(history.finishing_stacks),
    )


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class CleanHandTests(unittest.TestCase):
    """A clean inline hand passes every invariant."""

    @classmethod
    def setUpClass(cls) -> None:
        from test_phh_replay import PLURIBUS_HAND, _convert

        cls.pairs, cls.counter = _convert(PLURIBUS_HAND)
        cls.table_id, cls.replay = cls.pairs[0]
        cls.finding = check_hand(
            cls.table_id,
            cls.replay,
            _RECORDED_STARTING,
            _RECORDED_FINISHING,
        )

    def test_no_refusals_and_one_replay(self) -> None:
        self.assertEqual(len(self.pairs), 1)
        self.assertEqual(self.counter.total, 0)

    def test_every_invariant_passes(self) -> None:
        self.assertEqual(self.finding["invariant1"]["verdict"], "pass")
        self.assertEqual(self.finding["invariant1"]["class"], "equal")
        self.assertTrue(self.finding["invariant1"]["exactly_equal"])
        for name in ("invariant2", "invariant3", "invariant4",
                     "invariant5"):
            self.assertEqual(self.finding[name]["verdict"], "pass", name)
        self.assertEqual(self.finding["actions"], 12)
        self.assertEqual(self.finding["players"], 6)
        self.assertEqual(self.finding["invariant5"]["conservation_delta"], 0)
        self.assertEqual(
            self.finding["invariant5"]["file_conservation_delta"], 0
        )


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class CorruptedStackTests(unittest.TestCase):
    """A corrupted finishing stack fails invariant 1.

    Deliberately NOT the tool's own self-check hand or its ``+7`` on
    seat 0: the side-pot hand, corrupted by ``-3`` on the last seat.
    """

    def test_corrupted_finishing_stack_fails_invariant_1(self) -> None:
        from test_phh_replay import SIDE_POT_HAND, _convert

        pairs, _ = _convert(SIDE_POT_HAND)
        replay = pairs[0][1]
        clean = check_hand(
            "clean", replay, _SIDE_POT_STARTING, _SIDE_POT_FINISHING
        )
        self.assertEqual(clean["invariant1"]["verdict"], "pass")
        self.assertEqual(clean["invariant1"]["class"], "equal")
        corrupted = copy.deepcopy(replay)
        corrupted["table"]["seats"][1]["stackChips"] = (
            _SIDE_POT_FINISHING[1] - 3
        )
        finding = check_hand(
            "corrupted",
            corrupted,
            _SIDE_POT_STARTING,
            _SIDE_POT_FINISHING,
        )
        self.assertEqual(finding["invariant1"]["verdict"], "fail")
        self.assertEqual(finding["invariant1"]["class"], "mismatch")
        self.assertFalse(finding["invariant1"]["exactly_equal"])
        self.assertEqual(finding["invariant1"]["deltas"], {"2": "-3"})
        self.assertEqual(
            finding["invariant5"]["conservation_delta"], -3
        )
        self.assertEqual(finding["invariant5"]["verdict"], "fail")


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class HalfChipSplitTests(unittest.TestCase):
    """The one accepted inequality, and every clause that bounds it."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _HALF_CHIP_FILE.is_file():
            raise unittest.SkipTest("PHH Pluribus subset not on disk")
        cls.replay, cls.starting, cls.recorded = _load_half_chip_hand()

    def test_the_file_is_fractional_and_conserves(self) -> None:
        """The premise: the file does NOT drop a chip, it halves one."""

        self.assertEqual(
            [str(value) for value in self.recorded],
            ["10112.5", "9775.0", "10000.0", "10000.0", "10112.5",
             "10000.0"],
        )
        self.assertEqual(
            sum(Decimal(str(value)) for value in self.recorded),
            sum(Decimal(str(value)) for value in self.starting),
        )

    def test_real_half_chip_hand_is_classified_not_failed(self) -> None:
        finding = check_hand(
            "phh/pluribus/102/0", self.replay, self.starting, self.recorded
        )
        invariant1 = finding["invariant1"]
        self.assertEqual(invariant1["class"], "half_chip_split")
        self.assertEqual(invariant1["verdict"], "pass")
        self.assertFalse(invariant1["exactly_equal"])
        self.assertEqual(invariant1["deltas"], {"1": "+0.5", "5": "-0.5"})
        self.assertEqual(invariant1["winner_seats"], [1, 5])
        self.assertEqual(finding["invariant5"]["verdict"], "pass")
        self.assertEqual(finding["invariant5"]["conservation_delta"], 0)
        self.assertEqual(
            finding["invariant5"]["file_conservation_delta"], 0
        )

    def test_odd_chip_on_a_non_winner_is_a_mismatch(self) -> None:
        """The clause pokerkit's known defect needs (wrong-seat payout)."""

        forged = copy.deepcopy(self.replay)
        forged["table"]["winners"] = forged["table"]["winners"][:1]
        finding = check_hand(
            "wrong-seat", forged, self.starting, self.recorded
        )
        self.assertEqual(finding["invariant1"]["class"], "mismatch")
        self.assertEqual(finding["invariant1"]["verdict"], "fail")
        self.assertIn(
            "seat5_is_not_a_winner", finding["invariant1"]["reasons"]
        )

    def test_no_winners_at_all_is_a_mismatch(self) -> None:
        forged = copy.deepcopy(self.replay)
        forged["table"]["winners"] = []
        finding = check_hand(
            "no-winners", forged, self.starting, self.recorded
        )
        self.assertEqual(finding["invariant1"]["class"], "mismatch")
        self.assertEqual(
            sorted(finding["invariant1"]["reasons"]),
            ["seat1_is_not_a_winner", "seat5_is_not_a_winner"],
        )


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class WholeChipShortfallTests(unittest.TestCase):
    """The class the tool used to accept is now a failure.

    Before this repair the classifier accepted ANY hand whose recorded
    stacks summed to ``start - 1`` with exactly one replay seat a whole
    chip up, and called it ``odd_chip_defect``. That shape does not
    occur in the Pluribus corpus at all — it was manufactured by casting
    the file's ``Decimal('10112.5')`` to ``int``. It must fail.
    """

    def test_file_one_chip_short_fails_invariant_1_and_5(self) -> None:
        from test_phh_replay import PLURIBUS_HAND, _convert

        pairs, _ = _convert(PLURIBUS_HAND)
        replay = pairs[0][1]
        short = list(_RECORDED_FINISHING)
        short[0] -= 1  # the file, one chip light overall
        finding = check_hand(
            "one-short", replay, _RECORDED_STARTING, short
        )
        self.assertEqual(finding["invariant1"]["class"], "mismatch")
        self.assertEqual(finding["invariant1"]["verdict"], "fail")
        self.assertIn(
            "seat1_delta_is_a_whole_chip_or_more",
            finding["invariant1"]["reasons"],
        )
        self.assertIn(
            "seat1_file_stack_is_whole_chips",
            finding["invariant1"]["reasons"],
        )
        self.assertEqual(finding["invariant5"]["verdict"], "fail")
        self.assertEqual(
            finding["invariant5"]["file_conservation_delta"], -1
        )

    def test_half_chip_delta_on_a_non_winner_is_a_mismatch(self) -> None:
        """A sub-chip delta is accepted only on a WINNER's seat.

        The file is made fractional on seats 1 and 2 and the totals still
        agree, so only the winner clause can refuse it. Seat 1 won this
        hand; seat 2 folded preflop.
        """

        from test_phh_replay import PLURIBUS_HAND, _convert

        pairs, _ = _convert(PLURIBUS_HAND)
        replay = pairs[0][1]
        self.assertEqual(replay["table"]["winners"], [{"agentId": "p1"}])
        recorded = [Decimal(value) for value in _RECORDED_FINISHING]
        recorded[0] -= Decimal("0.5")
        recorded[1] += Decimal("0.5")
        finding = check_hand(
            "non-winner-half", replay, _RECORDED_STARTING, recorded
        )
        self.assertEqual(finding["invariant1"]["class"], "mismatch")
        self.assertEqual(
            finding["invariant1"]["reasons"], ["seat2_is_not_a_winner"]
        )
        self.assertEqual(
            finding["invariant1"]["deltas"], {"1": "+0.5", "2": "-0.5"}
        )

    def test_sub_chip_delta_where_the_file_is_whole_is_a_mismatch(
        self,
    ) -> None:
        """A sub-chip delta is accepted only where the FILE was fractional.

        Forges fractional stacks into the replay (the adapter cannot emit
        them) so the clause is exercised on its own: seat 1 is a winner
        and the totals agree, and it is still refused because the file
        recorded whole chips there.
        """

        from test_phh_replay import PLURIBUS_HAND, _convert

        pairs, _ = _convert(PLURIBUS_HAND)
        replay = copy.deepcopy(pairs[0][1])
        replay["table"]["seats"][0]["stackChips"] = Decimal("10309.5")
        replay["table"]["seats"][1]["stackChips"] = Decimal("9900.5")
        finding = check_hand(
            "whole-chip-seat", replay, _RECORDED_STARTING,
            _RECORDED_FINISHING,
        )
        self.assertEqual(finding["invariant1"]["class"], "mismatch")
        self.assertIn(
            "seat1_file_stack_is_whole_chips",
            finding["invariant1"]["reasons"],
        )
        self.assertEqual(
            finding["invariant1"]["deltas"], {"1": "-0.5", "2": "+0.5"}
        )
        self.assertEqual(finding["invariant5"]["conservation_delta"], 0)


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class SampleSelectionTests(unittest.TestCase):
    """The invariant-7 sample is a seeded draw, not the first N hands."""

    IDS = [f"phh/pluribus/{index // 110}/{index % 110}" for index in
           range(1000)]

    def test_selection_is_deterministic_for_a_seed(self) -> None:
        first = [_selected(name, 7, 0.05) for name in self.IDS]
        second = [_selected(name, 7, 0.05) for name in self.IDS]
        self.assertEqual(first, second)

    def test_a_different_seed_draws_a_different_sample(self) -> None:
        seven = {name for name in self.IDS if _selected(name, 7, 0.05)}
        eight = {name for name in self.IDS if _selected(name, 8, 0.05)}
        self.assertNotEqual(seven, eight)

    def test_the_draw_is_spread_and_roughly_the_target_size(self) -> None:
        chosen = [
            index
            for index, name in enumerate(self.IDS)
            if _selected(name, 7, 0.05)
        ]
        self.assertGreater(len(chosen), 20)
        self.assertLess(len(chosen), 90)
        # Not a head slice: the draw reaches the far end of the walk.
        self.assertGreater(max(chosen), 900)
        self.assertNotEqual(chosen, list(range(len(chosen))))

    def test_probability_extremes(self) -> None:
        self.assertTrue(_selected("phh/x/1", 7, 1.0))
        self.assertFalse(_selected("phh/x/1", 7, 0.0))


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class LabelAvailabilityTests(unittest.TestCase):
    """Invariant 7's per-street and free-spot counters have a partner."""

    @classmethod
    def setUpClass(cls) -> None:
        from test_phh_replay import PLURIBUS_HAND, _convert
        from tools.build_phase_a_dataset_v9 import replay_rows_v9

        pairs, _ = _convert(PLURIBUS_HAND)
        cls.replay = pairs[0][1]
        cls.rows, _ = replay_rows_v9(
            cls.replay, seed=7, potential_trials=32, equity_trials=64
        )

    def test_rows_match_decisions_street_for_street(self) -> None:
        label = _label_availability(self.replay, self.rows)
        self.assertEqual(label["rows_missing"], 0)
        self.assertEqual(label["per_street"], label["per_street_decisions"])
        self.assertEqual(
            label["per_street"],
            {"preflop": 6, "flop": 2, "turn": 2, "river": 2},
        )
        self.assertEqual(
            label["free_spot_rows"], label["free_spot_decisions"]
        )
        self.assertEqual(label["free_spot_rows"], 5)

    def test_a_dropped_river_row_shows_up_on_both_counters(self) -> None:
        kept = [row for row in self.rows if row["street"] != "river"]
        label = _label_availability(self.replay, kept)
        self.assertEqual(label["rows_missing"], 2)
        self.assertNotEqual(
            label["per_street"], label["per_street_decisions"]
        )
        self.assertNotIn("river", label["per_street"])
        self.assertEqual(label["per_street_decisions"]["river"], 2)


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class ReportSchemaTests(unittest.TestCase):
    """``validate_roots`` over a temp root: the report schema."""

    @staticmethod
    def _root(raw: str) -> Path:
        from test_phh_replay import PLURIBUS_HAND, SIDE_POT_HAND

        root = Path(raw) / "pluribus"
        root.mkdir(parents=True)
        (root / "0.phh").write_text(PLURIBUS_HAND, encoding="utf-8")
        (root / "1.phh").write_text(SIDE_POT_HAND, encoding="utf-8")
        return root

    def test_report_schema_over_two_hands(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._root(raw)
            report = validate_roots([root], sample=2, seed=7)

        self.assertEqual(report["tool"], "tools.validate_phh_replay")
        self.assertTrue(report["adapter_version"])
        self.assertTrue(
            report["dataset_commit"] == "unknown"
            or re.fullmatch(r"[0-9a-f]{40}", report["dataset_commit"]),
            report["dataset_commit"],
        )
        self.assertEqual(report["roots"], [str(root)])
        self.assertEqual(report["counts"]["hands"], 2)
        self.assertEqual(report["counts"]["files"], 2)
        self.assertGreater(report["counts"]["actions"], 0)
        self.assertEqual(report["self_check"], {
            "clean_passed": True,
            "corrupted_failed_invariant_1": True,
            "half_chip_split_classified": True,
            "wrong_seat_odd_chip_failed": True,
        })
        names = {
            "1_finishing_stacks",
            "2_action_legality",
            "3_pot_equals_committed",
            "4_board_length_by_street",
            "5_chip_conservation",
            "6_refusals_and_seven_plus",
            "7_fast_sample_labels",
        }
        self.assertEqual(set(report["invariants"]), names)
        for invariant in report["invariants"].values():
            self.assertIsInstance(invariant["passed"], bool)
            self.assertTrue(invariant["statement"])
        inv1 = report["invariants"]["1_finishing_stacks"]
        self.assertEqual(inv1["hands_exactly_equal"], 2)
        self.assertEqual(inv1["hands_mismatch"], 0)
        self.assertEqual(inv1["hands_half_chip_split"], 0)
        self.assertTrue(inv1["exact_equality_holds_on_every_hand"])
        self.assertEqual(
            report["invariants"]["6_refusals_and_seven_plus"][
                "refusal_total"
            ],
            0,
        )
        self.assertEqual(
            report["invariants"]["6_refusals_and_seven_plus"][
                "max_players"
            ],
            6,
        )
        inv7 = report["invariants"]["7_fast_sample_labels"]
        self.assertEqual(inv7["hands_sampled"], 2)
        self.assertEqual(inv7["failures"], [])
        self.assertGreater(inv7["rows"], 0)
        self.assertEqual(inv7["skipped_decisions"], 0)
        self.assertEqual(inv7["rows"], inv7["decisions_sampled"])
        self.assertEqual(
            inv7["per_street_rows"], inv7["per_street_decisions"]
        )
        self.assertEqual(
            inv7["free_spot_rows"], inv7["free_spot_decisions"]
        )
        self.assertGreater(inv7["fold_through_active_rows"], 0)
        self.assertGreater(inv7["fold_through_aggressive_rows"], 0)
        self.assertEqual(report["sampling"]["probability"], 1.0)
        self.assertEqual(report["sampling"]["hands_selected"], 2)
        self.assertTrue(report["caveats"])
        self.assertTrue(report["limitations"])
        self.assertEqual(report["overall"], "pass")

    def test_invariant_7_fails_when_a_row_goes_missing(self) -> None:
        """The counters are asserted, not printed: drop one row."""

        from tools.build_phase_a_dataset_v9 import replay_rows_v9

        def _short(replay, **kwargs):
            rows, stats = replay_rows_v9(replay, **kwargs)
            return rows[:-1], stats

        with tempfile.TemporaryDirectory() as raw:
            root = self._root(raw)
            with mock.patch(
                "tools.validate_phh_replay.replay_rows_v9", _short
            ):
                report = validate_roots([root], sample=2, seed=7)

        inv7 = report["invariants"]["7_fast_sample_labels"]
        self.assertFalse(inv7["passed"])
        self.assertEqual(inv7["rows_missing"], 2)
        joined = " | ".join(inv7["failures"])
        self.assertIn("rows_missing 2", joined)
        self.assertIn("!= sampled decisions", joined)
        self.assertEqual(report["overall"], "fail")

    def test_a_misbehaving_self_check_refuses_to_report(self) -> None:
        """The refusal gate itself, not just its happy path."""

        broken = {
            "clean_passed": True,
            "corrupted_failed_invariant_1": False,
            "half_chip_split_classified": True,
            "wrong_seat_odd_chip_failed": True,
        }
        with tempfile.TemporaryDirectory() as raw:
            root = self._root(raw)
            with mock.patch(
                "tools.validate_phh_replay._self_check", return_value=broken
            ):
                with self.assertRaises(SystemExit) as caught:
                    validate_roots([root], sample=2, seed=7)
        self.assertIn("refusing to report", str(caught.exception))


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class OutputGuardTests(unittest.TestCase):
    """``--output`` is required, named on purpose, and never clobbers."""

    def _root(self, raw: str) -> Path:
        return ReportSchemaTests._root(raw)

    def test_output_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._root(raw)
            with self.assertRaises(SystemExit):
                main(["--roots", str(root), "--sample", "2"])

    def test_writes_the_json_and_its_markdown_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._root(raw)
            output = Path(raw) / "record.json"
            code = main(
                ["--roots", str(root), "--sample", "2", "--output",
                 str(output)]
            )
            self.assertEqual(code, 0)
            markdown = output.with_suffix(".md")
            self.assertTrue(output.is_file())
            self.assertTrue(markdown.is_file())
            report = json.loads(output.read_text(encoding="utf-8"))
            text = markdown.read_text(encoding="utf-8")
            inv1 = report["invariants"]["1_finishing_stacks"]
            inv7 = report["invariants"]["7_fast_sample_labels"]
            self.assertIn(str(inv1["hands_exactly_equal"]), text)
            self.assertIn(
                f"invariant 7 ran on {inv7['hands_sampled']} of "
                f"{inv7['hands_total']}",
                text,
            )
            self.assertIn(report["dataset_commit"], text)
            for caveat in report["caveats"]:
                self.assertIn(caveat, text)
            for limitation in report["limitations"]:
                self.assertIn(limitation, text)
            # An empty example list must not point at a per-hand table
            # the renderer never writes.
            self.assertEqual(inv1["hands_mismatch"], 0)
            self.assertEqual(inv1["hands_half_chip_split"], 0)
            self.assertIn("| `mismatch_examples` | none |", text)
            self.assertIn("| `half_chip_split_hands` | none |", text)
            self.assertNotIn("0 listed", text)
            self.assertNotIn("hand by hand below", text)

    def test_refuses_to_overwrite_an_existing_record(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._root(raw)
            output = Path(raw) / "record.json"
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                main(
                    ["--roots", str(root), "--sample", "2", "--output",
                     str(output)]
                )
            self.assertIn("refusing to overwrite", str(caught.exception))
            self.assertEqual(output.read_text(encoding="utf-8"), "{}")

    def test_refuses_when_only_the_markdown_sibling_exists(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._root(raw)
            output = Path(raw) / "record.json"
            output.with_suffix(".md").write_text("frozen", encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                main(
                    ["--roots", str(root), "--sample", "2", "--output",
                     str(output)]
                )
            self.assertIn("refusing to overwrite", str(caught.exception))
            self.assertFalse(output.exists())

    def test_overwrite_regenerates_both_files_together(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._root(raw)
            output = Path(raw) / "record.json"
            main(
                ["--roots", str(root), "--sample", "2", "--output",
                 str(output)]
            )
            first = output.read_text(encoding="utf-8")
            output.with_suffix(".md").write_text("stale", encoding="utf-8")
            main(
                ["--roots", str(root), "--sample", "2", "--output",
                 str(output), "--overwrite"]
            )
            second = output.read_text(encoding="utf-8")
            markdown = output.with_suffix(".md").read_text(encoding="utf-8")
            self.assertNotEqual(markdown, "stale")
            self.assertIn("PHH adapter validation", markdown)
            self.assertNotEqual(first, second)  # run_at moved on

    def test_the_half_chip_table_is_rendered_when_there_is_one(self) -> None:
        """The other side of the pointer: a real half-chip hand.

        Copies ``pluribus/102/0.phh`` into a temp root so the record is
        rendered end to end on the one inequality the gate accepts —
        the caveat, the ``exact_equality_holds_on_every_hand`` false,
        and the seat-by-seat table.
        """

        if not _HALF_CHIP_FILE.is_file():
            self.skipTest("PHH Pluribus subset not on disk")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "pluribus"
            root.mkdir(parents=True)
            (root / "0.phh").write_text(
                _HALF_CHIP_FILE.read_text(encoding="utf-8"), encoding="utf-8"
            )
            output = Path(raw) / "record.json"
            code = main(
                ["--roots", str(root), "--sample", "1", "--output",
                 str(output)]
            )
            self.assertEqual(code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            text = output.with_suffix(".md").read_text(encoding="utf-8")

        inv1 = report["invariants"]["1_finishing_stacks"]
        self.assertTrue(inv1["passed"])
        self.assertEqual(inv1["hands_exactly_equal"], 0)
        self.assertEqual(inv1["hands_half_chip_split"], 1)
        self.assertEqual(inv1["hands_mismatch"], 0)
        self.assertEqual(inv1["hands_not_exactly_equal"], 1)
        self.assertFalse(inv1["exact_equality_holds_on_every_hand"])
        self.assertEqual(
            inv1["half_chip_split_hands"],
            [{
                "table_id": "phh/pluribus/0",
                "deltas": {"1": "+0.5", "5": "-0.5"},
                "winner_seats": [1, 5],
            }],
        )
        self.assertEqual(
            report["invariants"]["5_chip_conservation"][
                "hands_file_conservation_violations"
            ],
            0,
        )
        self.assertIn("| `half_chip_split_hands` | 1 listed", text)
        self.assertIn("The half-chip-split hands, seat by seat", text)
        self.assertIn("seat 1 +0.5, seat 5 -0.5", text)
        self.assertIn("| `mismatch_examples` | none |", text)
        self.assertIn(
            "exact equality does NOT hold on 1 of 1 hands", text
        )

    def test_output_must_be_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._root(raw)
            with self.assertRaises(SystemExit) as caught:
                main(
                    ["--roots", str(root), "--sample", "2", "--output",
                     str(Path(raw) / "record.txt")]
                )
            self.assertIn("must end in .json", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
