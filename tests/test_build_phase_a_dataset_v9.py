"""Tests for the v9 Phase-A dataset builder.

The v8 suite's hand-built miniatures are reused verbatim (their
outcomes are known by construction) and re-asserted under the v9
contract: the preflop raise that folds the table supervises the
AGGRESSIVE lane (priced escalation), the unprovoked river/turn bets
supervise the ACTIVE lane (free-spot bet execution), folds and calls
supervise neither, and the deleted midpoint rule leaves a recorded
realized size instead. The build path is proven end to end: the written
dataset loads through the TRAINER's own ``load_phase_a_dataset_v9``
(the builder itself enforces this), the trainer's
``resolve_sizing_record`` reads the sidecar's composed record, and
reruns are byte-identical.
"""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from engine import schema4
from engine.v9_trainer import load_phase_a_dataset_v9, resolve_sizing_record
from tools.build_phase_a_dataset_v9 import (
    build_dataset_v9,
    replay_rows_v9,
    wager_lane,
)

#: Small trial counts keep the suite fast; every exact assertion below
#: is independent of both numbers by construction.
_FAST = {"potential_trials": 32, "equity_trials": 64}


def _fold_through_rows():
    from test_phase_a_dataset import _fold_through_replay

    return replay_rows_v9(_fold_through_replay(), **_FAST)


def _showdown_replay():
    from test_phase_a_dataset import _two_seat_replay

    return _two_seat_replay(
        "mini-sd",
        {1: ["As", "2s"], 2: ["Jh", "Th"]},
        flop=["Ah", "Kh", "Qh"],
        turn="2c",
        river="2d",
        river_bet_to=10,
        winners=["e2"],
    )


class WagerLaneTests(unittest.TestCase):
    """The pinned gate, hand-written: raw wager actions, lane by state."""

    def test_the_lane_table(self) -> None:
        cases = [
            (("bet", 0, 50, 0), "fold_through_active"),
            (("all-in", 0, 600, 0), "fold_through_active"),
            (("raise", 100, 400, 100), "fold_through_aggressive"),
            (("all-in", 100, 101, 100), "fold_through_aggressive"),
            # The call-for-less all-in: nobody faces new chips, no
            # fold-through to observe.
            (("all-in", 100, 90, 100), None),
            (("all-in", 100, 100, 100), None),
            # Non-wager actions and unreadable sizes never supervise.
            (("call", 100, 100, 100), None),
            (("check", 0, None, 0), None),
            (("fold", 100, None, 100), None),
            (("bet", 0, None, 0), None),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(wager_lane(*arguments), expected)


class FoldThroughHandV9Tests(unittest.TestCase):
    """mini-ft: the preflop raise (a priced escalation) folds the table."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.stats = _fold_through_rows()

    def test_row_count_and_timeout_handling(self) -> None:
        self.assertEqual(len(self.rows), 3)
        self.assertEqual(self.stats["timeout_actions"], 1)
        self.assertEqual(self.stats["skipped_decisions"], 0)

    def test_the_raise_supervises_the_aggressive_lane(self) -> None:
        row = next(row for row in self.rows if row["sequence"] == 6)
        self.assertFalse(row["to_call_zero"])
        self.assertEqual(row["masks"]["fold_through_aggressive"], 1)
        self.assertEqual(row["masks"]["fold_through_active"], 0)
        self.assertEqual(row["labels"]["fold_through_aggressive"], 1.0)
        # The midpoint rule is gone; the realized size is recorded raw.
        self.assertEqual(row["realized"]["action"], "raise")
        self.assertEqual(row["realized"]["to_amount"], 6)
        self.assertTrue(0 <= row["read_temperature_x10"] <= 1000)
        # No opponent continued: range and equity stay undefined.
        self.assertEqual(row["masks"]["range_bucket"], 0)
        self.assertEqual(row["masks"]["equity_called"], 0)

    def test_folds_supervise_no_lane_and_record_no_size(self) -> None:
        for sequence in (7, 8):
            row = next(row for row in self.rows if row["sequence"] == sequence)
            self.assertEqual(row["masks"]["fold_through_active"], 0)
            self.assertEqual(row["masks"]["fold_through_aggressive"], 0)
            self.assertNotIn("realized", row)

    def test_vectors_are_schema_4(self) -> None:
        for row in self.rows:
            self.assertEqual(len(row["features"]), schema4.INPUT_SIZE_V9)


class ShowdownHandV9Tests(unittest.TestCase):
    """mini-sd: an unprovoked river bet (the active lane's execution)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.stats = replay_rows_v9(_showdown_replay(), **_FAST)

    def test_the_bet_supervises_the_active_lane_exactly(self) -> None:
        bet = next(
            row
            for row in self.rows
            if row["street"] == "river" and row["seat"] == 2
        )
        self.assertTrue(bet["to_call_zero"])
        self.assertEqual(bet["masks"]["fold_through_active"], 1)
        self.assertEqual(bet["masks"]["fold_through_aggressive"], 0)
        self.assertEqual(bet["labels"]["fold_through_active"], 0.0)  # called
        # The royal flush against the caller: exactly 1.0, no sampling.
        self.assertEqual(bet["masks"]["equity_called"], 1)
        self.assertEqual(bet["labels"]["equity_called"], 1.0)
        self.assertEqual(bet["realized"]["action"], "bet")
        self.assertEqual(bet["realized"]["to_amount"], 10)

    def test_the_call_supervises_no_lane_and_is_drawing_dead(self) -> None:
        call = next(
            row
            for row in self.rows
            if row["street"] == "river" and row["seat"] == 1
        )
        self.assertFalse(call["to_call_zero"])
        self.assertEqual(call["masks"]["fold_through_active"], 0)
        self.assertEqual(call["masks"]["fold_through_aggressive"], 0)
        self.assertEqual(call["labels"]["equity_called"], 0.0)

    def test_checks_are_free_spot_rows_without_lanes(self) -> None:
        checks = [
            row
            for row in self.rows
            if row["to_call_zero"] and "realized" not in row
        ]
        self.assertTrue(checks)
        for row in checks:
            self.assertEqual(row["masks"]["fold_through_active"], 0)
            self.assertEqual(row["masks"]["fold_through_aggressive"], 0)


class BoardLookAheadRepairTests(unittest.TestCase):
    """The shared reconstruction leaks the next street's cards; the v9
    builder must not.

    The assertion is derived from the SCHEMA, not from the builder: the
    card block is four 52-code planes (hole / flop / turn / river), so a
    preflop decision must have zero lit flop planes, a flop decision
    zero lit turn planes, and so on. A row that violates that was scored
    on cards its actor had not seen.
    """

    _REAL_REPLAY = (
        Path("foreign play data")
        / "20260812T082057Z_poker-playground_s13_top15"
        / "raw"
        / "tables"
        # Chosen because it exercises the defect on three streets at
        # once (preflop, flop and turn each close a round here); a
        # replay without a street-closing decision would pass the repair
        # tests vacuously.
        / "cmspqjsy7t1h814ca6v9fihch.json"
    )

    @classmethod
    def setUpClass(cls) -> None:
        from tools.collect_foreign_play_data import _read_json, _unwrap_rpc

        if not cls._REAL_REPLAY.is_file():
            raise unittest.SkipTest("archive replay not present")
        cls.replay = _unwrap_rpc(_read_json(cls._REAL_REPLAY))

    @staticmethod
    def _plane_indices(plane: str) -> list[int]:
        from engine import schema4

        return [
            index
            for index, name in enumerate(schema4.FEATURE_NAMES_V9)
            if name.startswith(f"{plane}_")
        ]

    def test_the_defect_is_real_in_the_shared_reconstruction(self) -> None:
        """Documents WHY the repair exists — if this ever stops finding a
        leak, the upstream reconstruction changed and the v9 repair
        should be re-examined rather than silently kept."""

        from tools.collect_foreign_play_data import _reconstruct_state

        expected = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
        leaks = 0
        for event in self.replay.get("events") or []:
            if not isinstance(event, dict) or event.get("type") != "ActionTaken":
                continue
            street = str(event.get("street") or "").casefold()
            if street not in expected:
                continue
            state = _reconstruct_state(dict(self.replay), dict(event))
            board = list(state.get("boardCards") or [])
            if len(board) != expected[street]:
                leaks += 1
        self.assertGreater(
            leaks, 0, "no street-closing decision in this replay to exercise"
        )

    def test_built_rows_never_see_a_future_street(self) -> None:
        from engine import schema4

        rows, stats = replay_rows_v9(self.replay, **_FAST)
        self.assertTrue(rows)
        # The repair must have fired on this replay.
        self.assertGreater(stats["board_corrected"], 0)

        street_flag = {
            street: schema4.feature_index_v9(f"street_{street}")
            for street in ("preflop", "flop", "turn", "river")
        }
        planes = {
            plane: self._plane_indices(plane)
            for plane in ("flop", "turn", "river")
        }
        # Cards visible on each street, by definition of the planes.
        visible = {
            "preflop": set(),
            "flop": {"flop"},
            "turn": {"flop", "turn"},
            "river": {"flop", "turn", "river"},
        }
        for row in rows:
            vector = row["features"]
            street = row["street"]
            self.assertEqual(vector[street_flag[street]], 1.0)
            for plane, indices in planes.items():
                lit = sum(1 for index in indices if vector[index] != 0.0)
                if plane in visible[street]:
                    self.assertGreater(
                        lit, 0, f"{street} row has an empty {plane} plane"
                    )
                else:
                    self.assertEqual(
                        lit,
                        0,
                        f"{street} row sees {lit} {plane} card(s) it "
                        "could not have seen",
                    )


class BuildPathV9Tests(unittest.TestCase):
    """The full build over the miniatures, proven through the trainer."""

    @staticmethod
    def _write_roots(directory: Path) -> list[Path]:
        from test_phase_a_dataset import _fold_through_replay

        roots = []
        for name, replay in (
            ("mini-ft-root", _fold_through_replay()),
            ("mini-sd-root", _showdown_replay()),
        ):
            root = directory / name
            tables = root / "raw" / "tables"
            tables.mkdir(parents=True)
            # The shared fixtures carry one table id; the builder now
            # deduplicates by table id (the widened archive holds 536
            # byte-identical duplicate tables), so each root gets its own
            # identity. The reconstructed state prefers the event
            # SNAPSHOT's tableId over the table metadata, so both are
            # stamped. Distinct ids also mean the per-decision seeds
            # differ, which the pinned expectations do not depend on.
            replay["table"]["tableId"] = f"{name}-table"
            replay["table"]["id"] = f"{name}-table"
            for event in replay.get("events") or []:
                snapshot = event.get("snapshot")
                if isinstance(snapshot, dict):
                    snapshot["tableId"] = f"{name}-table"
            (tables / f"{name}.json").write_text(
                json.dumps({"result": {"data": {"json": replay}}}),
                encoding="utf-8",
            )
            roots.append(root)
        return roots

    def test_build_loads_through_the_trainer_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            roots = self._write_roots(directory)
            output = directory / "out" / "phase-a-dataset-v9.jsonl.gz"
            sidecar_document = build_dataset_v9(
                roots, output, seed=7, workers=1, **_FAST
            )
            first_bytes = output.read_bytes()
            rows = load_phase_a_dataset_v9(output)
            self.assertEqual(len(rows), sidecar_document["counts"]["rows"])

            # Hand-picked slot expectations on known rows (the pinned
            # equity order: passive=0, active=1, aggressive=2).
            raise_row = next(
                row
                for row in rows
                if row.fold_through_mask == (0, 1)
            )
            self.assertEqual(raise_row.equity_slot, 2)
            bet_row = next(
                row for row in rows if row.fold_through_mask == (1, 0)
            )
            self.assertEqual(bet_row.equity_slot, 1)
            check_row = next(
                row
                for row in rows
                if row.to_call_zero and row.fold_through_mask == (0, 0)
            )
            self.assertEqual(check_row.equity_slot, 0)
            priced_no_wager = next(
                row
                for row in rows
                if not row.to_call_zero and row.fold_through_mask == (0, 0)
            )
            self.assertEqual(priced_no_wager.equity_slot, 1)

            # The sidecar's sizing record is what the trainer resolves.
            self.assertEqual(
                resolve_sizing_record(None, sidecar_document),
                sidecar_document["sizing"],
            )
            self.assertEqual(
                sidecar_document["schema_version"], schema4.SCHEMA_VERSION_V9
            )
            self.assertEqual(
                sidecar_document["generator"]["read_equity_trials"], 200
            )
            self.assertTrue(sidecar_document["generator"]["belief_fit_source"])

            # Byte-identical rerun.
            second = directory / "out2" / "phase-a-dataset-v9.jsonl.gz"
            build_dataset_v9(roots, second, seed=7, workers=1, **_FAST)
            self.assertEqual(first_bytes, second.read_bytes())

    def test_v8_loader_refuses_the_organic_v9_dataset(self) -> None:
        from engine.v8_trainer import load_phase_a_dataset

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            roots = self._write_roots(directory)
            output = directory / "out" / "phase-a-dataset-v9.jsonl.gz"
            build_dataset_v9(roots, output, seed=7, workers=1, **_FAST)
            with self.assertRaises(ValueError):
                load_phase_a_dataset(output)

    def test_dataset_rows_are_compact_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            roots = self._write_roots(directory)
            output = directory / "out" / "phase-a-dataset-v9.jsonl.gz"
            build_dataset_v9(roots, output, seed=7, workers=1, **_FAST)
            with gzip.open(output, "rt", encoding="utf-8") as stream:
                first = json.loads(stream.readline())
        self.assertIn("to_call_zero", first)
        self.assertIn("read_temperature_x10", first)
        self.assertNotIn("fold_through_small", first["masks"])


if __name__ == "__main__":
    unittest.main()
