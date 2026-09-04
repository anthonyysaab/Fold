"""Tests for the PHH Phase-A dataset builder.

Integration over the REAL adapter (``tools.phh_replay``, pokerkit):
three inline ``.phh`` hands in a temp ``pluribus`` root — two accepted
(9 + 3 rows through ``replay_rows_v9``), one refused for antes. The
hand strings are pinned pokerkit-valid histories; the refusal reason
comes from the adapter's own ``REASON_ANTES`` vocabulary. Asserts: row
count, table ids (the adapter's root-scoped ``phh/pluribus/<file>``
form), sidecar keys (source phh, dataset commit, roots, adapter
version, refusal counters, skip/timeout counters, label coverage), the
trainer self-load, byte-identical rerun, and that the refused hand is
counted and absent.

``ShardingTests`` pins the property the per-file ``--workers`` shard
rests on: the dataset is byte-identical at any worker count, because
ids come from ``tools.phh_replay.root_table_base`` rather than from the
shard boundary. Before 2026-09-03 the builder made one work item per
root, so ``--workers`` was a no-op for the documented one-root run; a
test that only ever built at ``workers=1`` could not have seen either
the no-op or a sharding regression.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from engine.v9_trainer import load_phase_a_dataset_v9
from tools.build_phase_a_dataset_phh import build_phase_a_dataset_phh

try:
    import pokerkit  # noqa: F401

    _POKERKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _POKERKIT_AVAILABLE = False

#: Small trial counts keep the suite fast; every exact assertion below
#: is independent of both numbers by construction.
_FAST = {"potential_trials": 32, "equity_trials": 64}

#: Inline .phh hands, valid for pokerkit 0.7.4 (probed). Row counts:
#: hand1 = 9, hand2 = 3, hand3 refused for antes.
_HAND_1 = (
    "variant = 'NT'\n"
    "antes = [0, 0, 0]\n"
    "blinds_or_straddles = [50, 100, 0]\n"
    "min_bet = 100\n"
    "starting_stacks = [10000, 10000, 10000]\n"
    "actions = ['d dh p1 As2s', 'd dh p2 JhTh', 'd dh p3 9c3d', 'p3 f', "
    "'p1 cbr 200', 'p2 cc', 'd db AhKhQh', 'p1 cc', 'p2 cc', "
    "'d db 2c', 'p1 cc', 'p2 cc', 'd db 2d', 'p1 cbr 300', 'p2 cc']\n"
    "hand = 0\n"
    "players = ['p1', 'p2', 'p3']\n"
    "finishing_stacks = [9500, 10500, 10000]\n"
)
_HAND_2 = (
    "variant = 'NT'\n"
    "antes = [0, 0, 0]\n"
    "blinds_or_straddles = [50, 100, 0]\n"
    "min_bet = 100\n"
    "starting_stacks = [10000, 10000, 10000]\n"
    "actions = ['d dh p1 TcQc', 'd dh p2 8s4c', 'd dh p3 9c3d', 'p3 f', "
    "'p1 cbr 210', 'p2 f']\n"
    "hand = 0\n"
    "players = ['p1', 'p2', 'p3']\n"
    "finishing_stacks = [10100, 9950, 10000]\n"
)
#: An ante game: refused, counted, and absent from the rows.
_HAND_3 = (
    "variant = 'NT'\n"
    "antes = [10, 10, 10]\n"
    "blinds_or_straddles = [50, 100, 0]\n"
    "min_bet = 100\n"
    "starting_stacks = [10000, 10000, 10000]\n"
    "actions = ['d dh p1 TcQc', 'd dh p2 8s4c', 'd dh p3 9c3d', 'p3 f', "
    "'p1 cbr 210', 'p2 f']\n"
    "hand = 0\n"
    "players = ['p1', 'p2', 'p3']\n"
    "finishing_stacks = [10100, 9950, 10000]\n"
)

_TABLE_1 = "phh/pluribus/hand1"
_TABLE_2 = "phh/pluribus/hand2"


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit is not available")
class PhhBuildPathTests(unittest.TestCase):
    """The full PHH build over three inline hands, through the trainer."""

    @classmethod
    def setUpClass(cls) -> None:
        from tools.phh_replay import PHH_REPLAY_VERSION

        cls.directory = tempfile.TemporaryDirectory()
        raw = Path(cls.directory.name)
        cls.root = raw / "pluribus"
        cls.root.mkdir(parents=True)
        for name, text in (
            ("hand1.phh", _HAND_1),
            ("hand2.phh", _HAND_2),
            ("hand3.phh", _HAND_3),
        ):
            (cls.root / name).write_text(text, encoding="utf-8")
        cls.adapter_version = PHH_REPLAY_VERSION

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def _build(self, output: Path) -> dict:
        return build_phase_a_dataset_phh(
            [self.root], output, seed=7, workers=1, **_FAST
        )

    def test_build_sidecar_and_self_load(self) -> None:
        output = (
            Path(self.directory.name)
            / "out"
            / "phase-a-dataset-v9-pluribus.jsonl.gz"
        )
        sidecar = self._build(output)
        rows = load_phase_a_dataset_v9(output)
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            Counter(row.table_id for row in rows),
            Counter({_TABLE_1: 9, _TABLE_2: 3}),
        )

        generator = sidecar["generator"]
        self.assertEqual(generator["source"], "phh")
        self.assertEqual(generator["tool"], "tools.build_phase_a_dataset_phh")
        self.assertEqual(generator["roots"], [str(self.root)])
        self.assertEqual(generator["phh_replay_version"], self.adapter_version)
        self.assertEqual(generator["refusals"], {"antes": 1})
        self.assertEqual(generator["skipped_decisions"], 0)
        self.assertEqual(generator["timeout_actions"], 0)
        self.assertEqual(
            generator["label_coverage"], sidecar["counts"]["label_coverage"]
        )
        self.assertEqual(generator["label_coverage"]["rows"], 12)
        self.assertEqual(generator["read_equity_trials"], 200)
        self.assertTrue(generator["belief_fit_source"])
        commit = generator["dataset_commit"]
        self.assertTrue(
            commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", commit),
            commit,
        )

        counts = sidecar["counts"]
        self.assertEqual(counts["files"], 3)
        self.assertEqual(counts["hands"], 2)
        self.assertEqual(counts["tables_with_rows"], 2)
        self.assertEqual(counts["rows"], 12)
        self.assertEqual(counts["duplicate_table_rows_dropped"], 0)
        self.assertEqual(sidecar["schema_version"], 4)

    def test_refused_hand_is_counted_and_absent(self) -> None:
        output = (
            Path(self.directory.name)
            / "out"
            / "phase-a-dataset-v9-pluribus.jsonl.gz"
        )
        sidecar = self._build(output)
        rows = load_phase_a_dataset_v9(output)
        table_ids = {row.table_id for row in rows}
        self.assertNotIn("phh/pluribus/hand3", table_ids)
        self.assertIn("antes", sidecar["generator"]["refusals"])
        self.assertEqual(sidecar["generator"]["refusals"]["antes"], 1)

    def test_byte_identical_rerun(self) -> None:
        output_one = (
            Path(self.directory.name) / "out1" / "dataset.jsonl.gz"
        )
        output_two = (
            Path(self.directory.name) / "out2" / "dataset.jsonl.gz"
        )
        self._build(output_one)
        self._build(output_two)
        self.assertEqual(output_one.read_bytes(), output_two.read_bytes())


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit is not available")
class ShardingTests(unittest.TestCase):
    """``--workers`` shards per file and moves no dataset byte."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name) / "pluribus"
        cls.root.mkdir(parents=True)
        for name, text in (
            ("hand1.phh", _HAND_1),
            ("hand2.phh", _HAND_2),
            ("hand3.phh", _HAND_3),
        ):
            (cls.root / name).write_text(text, encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_the_parent_walk_matches_the_adapters(self) -> None:
        """``_phh_files`` must stay equal to the adapter's own walk.

        The parent builds the per-file work list without importing the
        adapter (and so without pokerkit); if the two walks diverge, a
        file is silently dropped from or added to the dataset.
        """

        from tools.build_phase_a_dataset_phh import _phh_files
        from tools.phh_replay import root_files

        self.assertEqual(_phh_files(self.root), root_files(self.root))
        self.assertEqual(
            [path.name for path in _phh_files(self.root)],
            ["hand1.phh", "hand2.phh", "hand3.phh"],
        )

    def test_the_shard_reproduces_the_root_walks_ids(self) -> None:
        """Sharding per file yields the root walk's ids, in its order.

        This is what makes the per-decision seed
        ``sha256(seed:table_id:sequence)`` invariant to the shard.
        """

        from tools.phh_replay import (
            replays_from_file_in_root,
            replays_from_root,
            root_files,
        )

        walked = [table_id for table_id, _ in replays_from_root(self.root)]
        sharded = [
            table_id
            for path in root_files(self.root)
            for table_id, _ in replays_from_file_in_root(self.root, path)
        ]
        self.assertEqual(walked, sharded)
        self.assertTrue(
            all(table_id.startswith("phh/pluribus/") for table_id in walked),
            walked,
        )

    def test_worker_count_does_not_change_the_dataset(self) -> None:
        """The bytes are identical at ``workers=1`` and ``workers=3``.

        Fails on the pre-2026-09-03 builder only in spirit -- that one
        could not differ because it never sharded -- so this is a
        regression pin on the new shard, not a defect reproduction, and
        it is the check that must run before any dataset is trusted.
        """

        one = Path(self.directory.name) / "w1" / "dataset.jsonl.gz"
        many = Path(self.directory.name) / "w3" / "dataset.jsonl.gz"
        summary_one = build_phase_a_dataset_phh(
            [self.root], one, seed=7, workers=1, **_FAST
        )
        summary_many = build_phase_a_dataset_phh(
            [self.root], many, seed=7, workers=3, **_FAST
        )
        self.assertEqual(one.read_bytes(), many.read_bytes())
        self.assertEqual(
            summary_one["counts"]["rows"], summary_many["counts"]["rows"]
        )
        self.assertEqual(
            summary_one["counts"]["hands"], summary_many["counts"]["hands"]
        )
        # The shard must leave no trace in the record either: coverage
        # stays keyed by ROOT, one entry, at any worker count -- the same
        # shape the --limit fallback produces below.
        self.assertEqual(
            sorted(summary_one["per_collection"]),
            sorted(summary_many["per_collection"]),
        )
        self.assertEqual(len(summary_one["per_collection"]), 1)
        self.assertEqual(
            summary_one["per_collection"], summary_many["per_collection"]
        )

    def test_limit_still_caps_hands_per_root(self) -> None:
        """``--limit`` keeps the per-root path and its hand semantics.

        A per-file shard cannot cap hands per root without coordinating
        workers, so the builder falls back; this pins that the fallback
        still counts HANDS, not files.
        """

        output = Path(self.directory.name) / "limited" / "dataset.jsonl.gz"
        summary = build_phase_a_dataset_phh(
            [self.root], output, seed=7, workers=1, limit=1, **_FAST
        )
        self.assertEqual(summary["counts"]["hands"], 1)
        self.assertEqual(summary["generator"]["limit"], 1)
        # The fallback is the per-ROOT path, so coverage is keyed by
        # root -- one entry, not one per file.
        self.assertEqual(len(summary["per_collection"]), 1)


if __name__ == "__main__":
    unittest.main()
