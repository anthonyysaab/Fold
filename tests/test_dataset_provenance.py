"""Tests for the retired-corpus gate.

The defect these pin (2026-09-04): ``training.v9_trainer --dataset`` and
``training.v9_trainer_phase_b --phase-a-dataset`` both DEFAULTED to
``artifacts/phase_a_v9/phase-a-dataset-v9.jsonl.gz`` -- the Arena-built
corpus the 2026-09-03 PHH switch retired. A bare invocation therefore
trained on quarantined data and said nothing, because the file exists
and loads cleanly. The quarantine was a directory move and a note; the
gate is ``training.dataset_provenance``.

Which of these fail on the unfixed code, and how that was established
(``.handoff/DECISIONS.md`` section 3.5):

* ``TrainerDatasetArgumentTests`` -- BOTH fail on the unfixed code.
  Established by restoring the old ``default=`` line in a scratch copy
  of each trainer outside the working tree and re-running them: with the
  default in place ``main`` gets past argparse and no ``SystemExit`` is
  raised, so both assertions fail.
* ``ProvenanceGateTests`` and ``RealDatasetTests`` -- these exercise a
  module that did not exist before, so "fails on the unfixed code" is
  vacuous for them. They are regression pins, not defect reproductions,
  and are recorded as such rather than counted as the fix's evidence.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

from training.dataset_provenance import (
    LIVE_SOURCES,
    RetiredDatasetError,
    describe,
    read_provenance,
    require_live_dataset,
    sidecar_path,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PHASE_A = _REPO_ROOT / "artifacts" / "phase_a_v9"

#: The shape the Arena sidecar actually has on disk: no ``source`` key
#: at all, and roots pointing into what is now the quarantined archive.
_ARENA_GENERATOR = {
    "tool": "tools.build_phase_a_dataset_v9",
    "seed": 7,
    "roots": [
        "foreign play data\\20260812T082057Z_poker-playground_s13_top15",
        "foreign play data\\20260815T210237Z_poker-playground_s14_top15",
    ],
}
_PHH_GENERATOR = {
    "tool": "tools.build_phase_a_dataset_phh",
    "source": "phh",
    "seed": 7,
    "roots": ["phh-dataset\\data\\pluribus"],
    "dataset_commit": "e2ec038d31a1a46a82d147db4bbfdb0910459705",
}


def _write_dataset(directory: Path, name: str, generator: dict | None) -> Path:
    """A dataset file and, when given, the sidecar beside it."""

    dataset = directory / f"{name}.jsonl.gz"
    dataset.write_bytes(b"not really gzip; the gate never opens it")
    if generator is not None:
        sidecar_path(dataset).write_text(
            json.dumps({"schema_version": 4, "generator": generator}),
            encoding="utf-8",
        )
    return dataset


class ProvenanceGateTests(unittest.TestCase):
    """A dataset is live iff its sidecar declares a live source."""

    def test_a_phh_dataset_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = _write_dataset(Path(raw), "phh", _PHH_GENERATOR)
            self.assertEqual(require_live_dataset(dataset), "phh")
            self.assertIn("phh", LIVE_SOURCES)

    def test_the_arena_shape_is_refused(self) -> None:
        """No ``source`` key is the retired shape, not a missing detail."""

        with tempfile.TemporaryDirectory() as raw:
            dataset = _write_dataset(Path(raw), "arena", _ARENA_GENERATOR)
            with self.assertRaises(RetiredDatasetError) as caught:
                require_live_dataset(dataset)
            message = str(caught.exception)
            self.assertIn("no generator.source", message)
            self.assertIn("foreign play data", message)
            # The refusal must carry its own recovery.
            self.assertIn("build_phase_a_dataset_phh", message)
            self.assertIn("--allow-retired-dataset", message)

    def test_an_unknown_source_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = _write_dataset(
                Path(raw), "other", {"source": "handhq", "roots": []}
            )
            with self.assertRaises(RetiredDatasetError) as caught:
                require_live_dataset(dataset)
            self.assertIn("'handhq'", str(caught.exception))

    def test_a_missing_sidecar_fails_closed(self) -> None:
        """Unknown provenance is refused, never waved through."""

        with tempfile.TemporaryDirectory() as raw:
            dataset = _write_dataset(Path(raw), "bare", None)
            with self.assertRaises(RetiredDatasetError) as caught:
                require_live_dataset(dataset)
            self.assertIn("no readable sidecar", str(caught.exception))
            self.assertIsNone(read_provenance(dataset))

    def test_an_unparsable_sidecar_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = _write_dataset(Path(raw), "broken", None)
            sidecar_path(dataset).write_text("{ not json", encoding="utf-8")
            with self.assertRaises(RetiredDatasetError):
                require_live_dataset(dataset)

    def test_allow_retired_permits_but_still_names_it(self) -> None:
        """The escape hatch never returns a live source."""

        with tempfile.TemporaryDirectory() as raw:
            dataset = _write_dataset(Path(raw), "arena", _ARENA_GENERATOR)
            source = require_live_dataset(dataset, allow_retired=True)
            self.assertTrue(source.startswith("retired:"), source)
            self.assertNotIn(source, LIVE_SOURCES)

    def test_sidecar_path_mirrors_the_sink(self) -> None:
        self.assertEqual(
            sidecar_path("a/b/phase-a-dataset-v9-pluribus-2026-09-04.jsonl.gz"),
            Path("a/b/phase-a-dataset-v9-pluribus-2026-09-04.summary.json"),
        )

    def test_describe_names_the_source_and_roots(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = _write_dataset(Path(raw), "phh", _PHH_GENERATOR)
            text = describe(dataset)
            self.assertIn("source phh", text)
            self.assertIn("pluribus", text)
            self.assertIn("e2ec038d", text)

    def test_describe_does_not_raise_without_a_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = _write_dataset(Path(raw), "bare", None)
            self.assertIn("provenance unknown", describe(dataset))


class TrainerDatasetArgumentTests(unittest.TestCase):
    """Neither trainer may carry a dataset default.

    Both assertions fail on the unfixed code: it defaulted to the
    retired Arena corpus, so ``main`` got past argparse instead of
    exiting. argparse raises before any torch import, so these run on
    the stdlib interpreter.
    """

    def test_phase_a_trainer_requires_an_explicit_dataset(self) -> None:
        from training import v9_trainer

        with self.assertRaises(SystemExit):
            v9_trainer.main(["--model-version", "unused"])

    def test_phase_b_trainer_requires_an_explicit_dataset(self) -> None:
        from training import v9_trainer_phase_b

        with self.assertRaises(SystemExit):
            v9_trainer_phase_b.main(
                ["--model-version", "unused", "--phase-b-corpus", "unused"]
            )

    def test_no_module_still_exports_the_retired_default(self) -> None:
        """The retired path must not survive as an importable constant."""

        from training import v9_trainer_phase_b

        self.assertFalse(
            hasattr(v9_trainer_phase_b, "DEFAULT_PHASE_A_DATASET_V9"),
            "the retired dataset default is still importable",
        )


class PhaseBTrainerCliTests(unittest.TestCase):
    """The Phase-B trainer's CLI must carry the flag its main() reads.

    FAILS ON THE UNFIXED CODE (verified 2026-09-04): ``main`` called
    ``args.allow_retired_dataset`` while its parser never defined
    ``--allow-retired-dataset``, so EVERY invocation raised
    ``AttributeError`` immediately after loading the Phase-B corpus — the
    trainer could not run at all. The existing pins all stop at argparse
    (they assert ``SystemExit`` on a missing required argument), so none of
    them reached the line, and the 2026-09-03 integration passed a green
    suite with the trainer unusable. These tests get PAST the corpus load
    by stubbing it, which is the only way to reach the guard.
    """

    @staticmethod
    def _stub_corpus():
        return SimpleNamespace(decisions=(), equity_trials=1000)

    def _argv(self, dataset: Path, extra: list[str]) -> list[str]:
        return [
            "--phase-b-corpus", "unused-the-loader-is-stubbed",
            "--phase-a-dataset", str(dataset),
            "--model-version", "candidate-test",
            *extra,
        ]

    def test_main_reaches_the_provenance_guard(self) -> None:
        """Not AttributeError: the guard itself must be what refuses."""

        from training import v9_trainer_phase_b

        with tempfile.TemporaryDirectory() as raw:
            dataset = _write_dataset(Path(raw), "arena", _ARENA_GENERATOR)
            with mock.patch.object(
                v9_trainer_phase_b,
                "load_phase_b_corpus_v9",
                return_value=self._stub_corpus(),
            ):
                with self.assertRaises(RetiredDatasetError):
                    v9_trainer_phase_b.main(self._argv(dataset, []))

    def test_the_flag_reaches_require_live_dataset(self) -> None:
        """The escape hatch is wired through, not merely declared."""

        from training import v9_trainer_phase_b

        with tempfile.TemporaryDirectory() as raw:
            dataset = _write_dataset(Path(raw), "arena", _ARENA_GENERATOR)
            with mock.patch.object(
                v9_trainer_phase_b,
                "load_phase_b_corpus_v9",
                return_value=self._stub_corpus(),
            ), mock.patch.object(
                v9_trainer_phase_b, "require_live_dataset"
            ) as guard:
                guard.side_effect = RuntimeError("stop after the guard")
                with self.assertRaises(RuntimeError):
                    v9_trainer_phase_b.main(
                        self._argv(dataset, ["--allow-retired-dataset"])
                    )
            self.assertTrue(guard.call_args.kwargs["allow_retired"])


    def test_the_phase_a_trainer_carries_the_same_flag(self) -> None:
        """The two trainers' escape hatches must not drift apart again.

        The Phase-A trainer has the identical construct — ``main`` reads
        ``args.allow_retired_dataset`` — and it happens to be correct today
        only because that file also defines the option. Nothing pinned it,
        which is how the Phase-B half shipped broken.
        """

        from training import v9_trainer

        with tempfile.TemporaryDirectory() as raw:
            dataset = _write_dataset(Path(raw), "arena", _ARENA_GENERATOR)
            with mock.patch.object(
                v9_trainer, "require_live_dataset"
            ) as guard:
                guard.side_effect = RuntimeError("stop after the guard")
                with self.assertRaises(RuntimeError):
                    v9_trainer.main(
                        [
                            "--dataset", str(dataset),
                            "--model-version", "candidate-test",
                            "--allow-retired-dataset",
                        ]
                    )
            self.assertTrue(guard.call_args.kwargs["allow_retired"])


class RealDatasetTests(unittest.TestCase):
    """The gate against the corpora actually on this machine."""

    @unittest.skipUnless(
        (_PHASE_A / "phase-a-dataset-v9.summary.json").is_file(),
        "the Arena sidecar is not on disk",
    )
    def test_the_real_arena_dataset_is_refused(self) -> None:
        with self.assertRaises(RetiredDatasetError):
            require_live_dataset(_PHASE_A / "phase-a-dataset-v9.jsonl.gz")

    @unittest.skipUnless(
        (_PHASE_A / "phase-a-dataset-v9-pluribus-2026-09-04.summary.json").is_file(),
        "the PHH sidecar is not on disk",
    )
    def test_the_real_phh_dataset_is_accepted(self) -> None:
        dataset = _PHASE_A / "phase-a-dataset-v9-pluribus-2026-09-04.jsonl.gz"
        self.assertEqual(require_live_dataset(dataset), "phh")


if __name__ == "__main__":
    unittest.main()
