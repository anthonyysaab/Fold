"""Checks for the learned runtime, promotion, and rollback path."""

from __future__ import annotations

import gzip
import json
import hashlib
import random
import shutil
import tempfile
import pathlib
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from engine.decision_engine import DEFAULT_SAFETY_GATES
from engine.learned_policy import (
    DEFAULT_SERVE_EQUITY_TRIALS,
    approved_fingerprint,
    LearnedPolicyError,
    load_approved,
    load_policy,
)
from engine.learning_contract import LEARNING_INPUT_SIZE
from engine.offline_trainer import TrainingConfig, train_candidate
from engine.training_telemetry import TrainingExample
from tools.promote_candidate import main as promote_main


def _examples() -> tuple[TrainingExample, ...]:
    rows = []
    for index in range(30):
        aggress = index % 3 == 0
        features = [0.1] * LEARNING_INPUT_SIZE
        features[7] = 3.0 if aggress else -1.0
        rows.append(
            TrainingExample(
                table_id=f"table-{index}",
                policy_version="sim-test",
                features=tuple(features),
                action_family_index=2 if aggress else 1,
                behavior_probabilities=(0.0, 0.0, 1.0) if aggress else (0.0, 1.0, 0.0),
                submitted_risk_fraction=0.25 if aggress else 0.0,
                purse_bb=10.0,
                reward_bb=1.0 if aggress else 0.0,
                counterfactual=True,
                opponent_confidence=1.0,
            )
        )
    return tuple(rows)


def _table(available: list[str], call_chips: int = 0) -> dict:
    can_raise = "raise" in available
    can_bet = "bet" in available
    return {
        "id": "learned-test",
        "tableId": "learned-test",
        "street": "flop",
        "potChips": 400,
        "currentBet": call_chips,
        "boardCards": ["2c", "7d", "9h"],
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": 1,
        "seats": [
            {
                "seatNumber": number,
                "status": "Active",
                "stackChips": 1_000,
                "currentBetChips": 0,
                "holeCards": ["Ah", "Ad"] if number == 1 else None,
            }
            for number in (1, 2)
        ],
        "allowedActions": {
            "canFold": "fold" in available,
            "canCheck": "check" in available,
            "canCall": "call" in available,
            "canBet": can_bet,
            "canRaise": can_raise,
            "canAllIn": False,
            "callAmount": call_chips,
            "callChips": call_chips,
            "callToAmount": call_chips,
            "minBet": 100 if can_bet else None,
            "minRaiseTo": 100 if can_raise else None,
            "betRange": {"min": 100, "max": 1_000} if can_bet else None,
            "raiseRange": {"min": 100, "max": 1_000} if can_raise else None,
            "allInToAmount": None,
            "availableActions": available,
            "amountSemantics": "toAmount",
            "reasoningRequired": False,
        },
        "recentEvents": [],
    }


#: The schema-4 features the OLS gate's decisive ``branch-strength`` arm
#: fits. Only these need to vary for the fit to be non-singular.
_GATE_FEATURE_NAMES = (
    "equity_vs_posterior",
    "equity_multiway",
    "strength_percentile",
    "pot_odds",
    "spr",
)

#: The split seed the synthetic gate fixture records in its manifest; the
#: gate reads it from there and fixes the validation fraction at 0.1.
_GATE_SPLIT_SEED = 17


def _write_ols_gate_inputs(directory: Path) -> tuple[Path, Path]:
    """Write a Phase-B corpus and a Phase-A dataset the real OLS gate scores.

    ``tools.promote_candidate._ols_gate`` recomputes both baselines FRESH
    from the manifest's own recorded training inputs -- by design, so a
    candidate can neither inherit nor hand-write a passing score -- which
    means a test that wants a genuine gate verdict has to supply genuine
    files. Patching the gate away instead proves only that an exception
    raised before the writes leaves the files alone, which is true of any
    ordering and so tests nothing about defect 26.

    Both requirements are constructed, not hoped for: the table ids are
    chosen by the trainer's own split rule so neither side of the 10%
    split is empty, and the five gate features vary per decision so the
    ``branch-strength`` fit is not singular.
    """

    from engine import schema4
    from tools.ols_baseline import _split_value

    train: list[str] = []
    validation: list[str] = []
    index = 0
    while len(train) < 24 or len(validation) < 4:
        table_id = f"ols-gate-{index}"
        index += 1
        if _split_value(_GATE_SPLIT_SEED, table_id) < 0.1:
            if len(validation) < 4:
                validation.append(table_id)
        elif len(train) < 24:
            train.append(table_id)

    columns = [schema4.feature_index_v9(name) for name in _GATE_FEATURE_NAMES]
    rng = random.Random(20260903)
    corpus = directory / "gate-fixture.phase-b.jsonl.gz"
    dataset = directory / "gate-fixture.phase-a.jsonl.gz"
    with (
        gzip.open(corpus, "wt", encoding="utf-8") as phase_b,
        gzip.open(dataset, "wt", encoding="utf-8") as phase_a,
    ):
        phase_b.write(json.dumps({"schema": 2, "note": "gate fixture"}) + "\n")
        for table_id in (*train, *validation):
            features = [0.0] * schema4.INPUT_SIZE_V9
            for column in columns:
                features[column] = rng.random()
            reward = rng.uniform(-3.0, 3.0)
            phase_b.write(
                json.dumps(
                    {
                        "table_id": table_id,
                        "features": features,
                        "purse_bb": 10.0,
                        # Per-decision centered, as a harvest emits them.
                        "branches": [
                            {"branch": "passive", "reward_bb": reward},
                            {"branch": "aggressive", "reward_bb": -reward},
                        ],
                    }
                )
                + "\n"
            )
            phase_a.write(
                json.dumps(
                    {
                        "table_id": table_id,
                        "features": features,
                        "masks": {"equity_called": True},
                        "labels": {"equity_called": 0.2 + 0.6 * rng.random()},
                    }
                )
                + "\n"
            )
    return corpus, dataset


class LearnedPolicyTests(unittest.TestCase):
    def _train(self, directory: str) -> Path:
        summary = train_candidate(
            _examples(),
            directory,
            TrainingConfig(epochs=3, validation_fraction=0.2, model_version="lp-test"),
        )
        return summary.manifest_path

    def test_loads_verifies_and_plays_within_legal_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._train(directory)
            policy = load_policy(manifest_path, equity_trials=1)

            self.assertEqual(policy.policy_version, "lp-test")
            # The trainer stamped default engine parameters; loading must
            # rebuild them.
            self.assertEqual(policy.safety_gates, DEFAULT_SAFETY_GATES)

            for available in (["check", "bet"], ["fold", "call"], ["fold"]):
                with patch.object(policy, "_equity", return_value=0.5):
                    payload = policy.decide(
                        _table(available, 300 if "call" in available else 0)
                    )
                self.assertIn(payload["action"], available)

    def test_corrupted_weights_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._train(directory)
            weights_path = manifest_path.parent / "lp-test.weights.json"
            corrupted = weights_path.read_bytes().replace(b"0.1", b"0.2", 1)
            weights_path.write_bytes(corrupted)
            with self.assertRaises(LearnedPolicyError):
                load_policy(manifest_path)

    def test_v6_defaults_to_heuristic_sizing_but_supports_sizing_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._train(directory)
            default = load_policy(manifest_path, equity_trials=1)
            heuristic = load_policy(
                manifest_path, equity_trials=1, use_learned_sizing=False
            )
            learned = load_policy(
                manifest_path, equity_trials=1, use_learned_sizing=True
            )
            table = _table(["check", "bet"])
            allowed = table["allowedActions"]
            for policy in (default, heuristic, learned):
                policy._proposed_risk_fraction = 1.0

            default_action = default._sized_action("bet", table, allowed, 0.8)
            heuristic_action = heuristic._sized_action("bet", table, allowed, 0.8)
            learned_action = learned._sized_action("bet", table, allowed, 0.8)

            self.assertEqual(default_action, heuristic_action)
            self.assertGreater(learned_action[1], heuristic_action[1])

    def test_hybrid_only_overrides_heuristic_for_confident_in_distribution_value(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._train(directory)
            table = _table(["fold", "call", "raise"], 100)
            allowed = table["allowedActions"]
            available = set(allowed["availableActions"])
            output = {
                "action_logits": [0.11, 0.0, 0.10],
                "action_probabilities": [0.34, 0.32, 0.34],
                "risk_fraction": 0.5,
            }

            conservative = load_policy(
                manifest_path,
                equity_trials=1,
                hybrid_min_value_advantage=0.2,
            )
            with (
                patch.object(
                    conservative,
                    "_learning_features",
                    return_value=tuple(conservative._means),
                ),
                patch(
                    "engine.learned_policy._forward",
                    return_value=output,
                ),
            ):
                family = conservative._equity_family(
                    table, allowed, available, 0.8, features=(0.0,)
                )
            self.assertEqual(family, "aggress")

            permissive = load_policy(
                manifest_path,
                equity_trials=1,
                hybrid_min_value_advantage=0.0,
            )
            with (
                patch.object(
                    permissive,
                    "_learning_features",
                    return_value=tuple(permissive._means),
                ),
                patch(
                    "engine.learned_policy._forward",
                    return_value=output,
                ),
            ):
                family = permissive._equity_family(
                    table, allowed, available, 0.8, features=(0.0,)
                )
            self.assertEqual(family, "fold")

            out_of_distribution = list(permissive._means)
            out_of_distribution[0] += 6.0 * permissive._stds[0]
            with (
                patch.object(
                    permissive,
                    "_learning_features",
                    return_value=tuple(out_of_distribution),
                ),
                patch(
                    "engine.learned_policy._forward",
                    return_value=output,
                ),
            ):
                family = permissive._equity_family(
                    table, allowed, available, 0.8, features=(0.0,)
                )
            self.assertEqual(family, "aggress")

    def test_promotion_writes_atomic_pointer_and_rollback_restores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            (artifacts / "candidates").mkdir()
            manifest_path = self._train(str(artifacts / "candidates"))

            self.assertIsNone(approved_fingerprint(artifacts))
            code = promote_main(
                [
                    str(manifest_path),
                    "--artifacts-dir",
                    str(artifacts),
                    "--reason",
                    "test promotion",
                ]
            )
            self.assertEqual(code, 0)
            first_token = approved_fingerprint(artifacts)
            self.assertIsNotNone(first_token)

            policy = load_approved(artifacts, equity_trials=1)
            self.assertEqual(policy.policy_version, "lp-test")

            # Second candidate becomes the new approval; rollback restores.
            second_manifest = train_candidate(
                _examples(),
                str(artifacts / "candidates"),
                TrainingConfig(
                    epochs=2, validation_fraction=0.2, model_version="lp-test-2"
                ),
            ).manifest_path
            promote_main(
                [
                    str(second_manifest),
                    "--artifacts-dir",
                    str(artifacts),
                    "--reason",
                    "second",
                ]
            )
            self.assertEqual(
                load_approved(artifacts, equity_trials=1).policy_version, "lp-test-2"
            )
            promote_main(["--rollback", "--artifacts-dir", str(artifacts)])
            self.assertEqual(
                load_approved(artifacts, equity_trials=1).policy_version, "lp-test"
            )
            pointer = json.loads((artifacts / "approved.json").read_text())
            self.assertEqual(pointer["model_version"], "lp-test")

    def _v9_candidate(
        self,
        directory: Path,
        model_version: str = "v9-gate-test",
        *,
        phase_b_corpus: Path | None = None,
        phase_a_dataset: Path | None = None,
        validation_losses: dict | None = None,
    ) -> Path:
        """A valid format-4 candidate with matching weights, for gate tests.

        With ``phase_b_corpus``/``phase_a_dataset`` the manifest records
        everything ``_ols_gate`` needs to recompute both baselines from
        the artifact's own inputs, so the gate runs end to end;
        ``validation_losses`` are the network numbers it compares against.
        Without them the manifest is the same shape the gate refuses for
        missing metadata.
        """

        from engine import schema4
        from engine.branch_contract_v9 import BRANCH_LABELS_V9, MODEL_FORMAT_VERSION_V9
        from engine.v8_trainer import default_v9_architecture

        weights = directory / "weights.json"
        weights.write_text('{"_": 1}\n', encoding="utf-8")
        window: dict = {"hand_count": 0, "phase_b_decisions": 1}
        if phase_b_corpus is not None:
            window["phase_b_corpus"] = str(phase_b_corpus)
        if phase_a_dataset is not None:
            window["phase_a_dataset"] = str(phase_a_dataset)
        manifest = {
            "format": "fold-multihead-policy",
            "format_version": MODEL_FORMAT_VERSION_V9,
            "model_version": model_version,
            "state": "candidate",
            "parent_version": None,
            "created_at": "2026-09-02T12:00:00Z",
            "feature_schema_version": schema4.SCHEMA_VERSION_V9,
            "input_size": schema4.INPUT_SIZE_V9,
            "feature_names": list(schema4.FEATURE_NAMES_V9),
            "action_labels": list(BRANCH_LABELS_V9),
            "architecture": default_v9_architecture(),
            "serve": {"deployable": True, "equity_trials": 1000},
            "weights_file": weights.name,
            "weights_sha256": hashlib.sha256(
                weights.read_bytes().rstrip(b"\n")
            ).hexdigest(),
            "training": {"split": {"split_seed": _GATE_SPLIT_SEED}},
            "training_window": window,
            "evaluation": (
                {"validation_losses": dict(validation_losses)}
                if validation_losses is not None
                else {}
            ),
        }
        path = directory / f"{model_version}.manifest.json"
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    def _sentinels(self, artifacts: Path, stem: str) -> tuple[Path, Path]:
        """Put both promotion files on disk with known bytes.

        Defect 26 is a REWRITE, not a creation: the approved manifest and
        the pointer already existed when the gate review overwrote them.
        A test that only asserts the files are absent afterwards cannot
        see that failure, so every case here starts from files that exist
        and compares bytes.
        """

        pointer = artifacts / "approved.json"
        pointer.write_text('{"approved_at": "sentinel"}\n', encoding="utf-8")
        approved = artifacts / "candidates" / f"{stem}.approved.manifest.json"
        approved.write_text(
            '{"state": "approved", "sentinel": true}\n', encoding="utf-8"
        )
        return pointer, approved

    def test_a_failed_gate_writes_nothing(self) -> None:
        """Defect 26: a gate that really runs and really FAILS leaves the
        approved manifest and the pointer byte-identical.

        The gate is NOT patched out. The candidate records a real corpus
        and a real Phase-A dataset, and validation losses that lose both
        arms, so ``_ols_gate`` recomputes both baselines and reaches its
        verdict before refusing. Patching the gate away instead asserts
        only that an exception raised before the writes leaves the files
        alone -- true of any ordering, and so no test of this property.
        """

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            (artifacts / "candidates").mkdir()
            corpus, dataset = _write_ols_gate_inputs(artifacts)
            manifest_path = self._v9_candidate(
                artifacts / "candidates",
                phase_b_corpus=corpus,
                phase_a_dataset=dataset,
                validation_losses={"value_normalized": 99.0, "equity_called": 99.0},
            )
            pointer, approved = self._sentinels(artifacts, "v9-gate-test")
            before_pointer = pointer.read_bytes()
            before_approved = approved.read_bytes()

            printed = StringIO()
            with redirect_stdout(printed), self.assertRaises(SystemExit) as refusal:
                promote_main(
                    [
                        str(manifest_path),
                        "--artifacts-dir",
                        str(artifacts),
                        "--reason",
                        "must not promote",
                    ]
                )

            # The refusal must be the gate's VERDICT, not a missing input:
            # a gate that never compared anything would leave the files
            # alone too, and would prove nothing about the write path.
            output = printed.getvalue()
            self.assertIn("OLS gate: FAIL (enforce)", output)
            self.assertIn("phase-b value head", output)
            self.assertIn("phase-a equity_called", output)
            self.assertIn("refusing to promote", str(refusal.exception))
            self.assertEqual(pointer.read_bytes(), before_pointer)
            self.assertEqual(approved.read_bytes(), before_approved)

    def test_a_pointer_read_failure_cannot_half_stamp_a_promotion(self) -> None:
        """Defect 26's other half: the approved manifest is written only
        once the pointer payload is in hand.

        The defect's signature is a manifest and a pointer stamped by
        different promotions. Writing the manifest first makes that one
        step away at all times: anything that goes wrong while reading
        the previous pointer -- here a corrupt ``approved.json`` -- leaves
        a manifest claiming a promotion the pointer never recorded. This
        is the case that fails on the unfixed ordering.
        """

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            (artifacts / "candidates").mkdir()
            manifest_path = self._train(str(artifacts / "candidates"))
            pointer, approved = self._sentinels(artifacts, "lp-test")
            pointer.write_text("{ this pointer is not json", encoding="utf-8")
            before_pointer = pointer.read_bytes()
            before_approved = approved.read_bytes()

            with self.assertRaises(json.JSONDecodeError):
                promote_main(
                    [
                        str(manifest_path),
                        "--artifacts-dir",
                        str(artifacts),
                        "--reason",
                        "must not half-promote",
                    ]
                )

            self.assertEqual(pointer.read_bytes(), before_pointer)
            self.assertEqual(approved.read_bytes(), before_approved)

    def test_dry_run_runs_the_checks_and_writes_nothing(self) -> None:
        """Defect 26: the gate review is its own verb.

        One manifest, one directory, two runs. The dry run must leave
        both promotion files byte-identical; the promotion that follows
        must change both. The second half is what makes the first mean
        anything -- a dry run that wrote nothing because nothing would
        have been written is not a repair.
        """

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            (artifacts / "candidates").mkdir()
            manifest_path = self._train(str(artifacts / "candidates"))
            pointer, approved = self._sentinels(artifacts, "lp-test")
            before_pointer = pointer.read_bytes()
            before_approved = approved.read_bytes()

            printed = StringIO()
            with redirect_stdout(printed):
                code = promote_main(
                    [
                        str(manifest_path),
                        "--artifacts-dir",
                        str(artifacts),
                        "--dry-run",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertIn("no files were written", printed.getvalue())
            self.assertEqual(pointer.read_bytes(), before_pointer)
            self.assertEqual(approved.read_bytes(), before_approved)

            with redirect_stdout(StringIO()):
                code = promote_main(
                    [
                        str(manifest_path),
                        "--artifacts-dir",
                        str(artifacts),
                        "--reason",
                        "the same inputs, promoted for real",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertNotEqual(pointer.read_bytes(), before_pointer)
            self.assertNotEqual(approved.read_bytes(), before_approved)

    def test_dry_run_runs_the_real_gate_and_still_writes_nothing(self) -> None:
        """A dry run runs the OLS gate itself, unpatched, and a FAIL is a
        refusal that has still written nothing.

        This is the operator act defect 26 came from -- someone wanting
        the gate's verdict and nothing else -- so the verdict has to be
        real here, not a mock's return value.
        """

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            (artifacts / "candidates").mkdir()
            corpus, dataset = _write_ols_gate_inputs(artifacts)
            manifest_path = self._v9_candidate(
                artifacts / "candidates",
                phase_b_corpus=corpus,
                phase_a_dataset=dataset,
                validation_losses={"value_normalized": 99.0, "equity_called": 99.0},
            )
            pointer, approved = self._sentinels(artifacts, "v9-gate-test")
            before_pointer = pointer.read_bytes()
            before_approved = approved.read_bytes()

            printed = StringIO()
            with redirect_stdout(printed), self.assertRaises(SystemExit):
                promote_main(
                    [
                        str(manifest_path),
                        "--artifacts-dir",
                        str(artifacts),
                        "--dry-run",
                    ]
                )

            self.assertIn("OLS gate: FAIL (enforce)", printed.getvalue())
            self.assertEqual(pointer.read_bytes(), before_pointer)
            self.assertEqual(approved.read_bytes(), before_approved)

    def test_promotion_stamps_both_files_with_one_timestamp(self) -> None:
        """Defect 26: both stamps come from ONE reading of the clock.

        Equal stamps prove nothing against a clock that can return the
        same value twice -- on Windows ``datetime.now`` is coarse enough
        for two calls to land on one microsecond. So the clock is
        replaced by one that never repeats, and the call COUNT is
        asserted as well: a second ``now()`` anywhere on the promotion
        path would then stamp the two files differently.
        """

        class _TickingClock:
            """A ``datetime`` stand-in whose ``now`` never repeats itself."""

            calls = 0

            @classmethod
            def now(cls, tz=None) -> datetime:
                cls.calls += 1
                return datetime(2026, 9, 3, 12, 0, cls.calls, tzinfo=tz or UTC)

        # The clock really does move; without this the assertions below
        # could be satisfied by a stand-in that returns a constant.
        self.assertNotEqual(_TickingClock.now(UTC), _TickingClock.now(UTC))
        _TickingClock.calls = 0

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            (artifacts / "candidates").mkdir()
            manifest_path = self._train(str(artifacts / "candidates"))
            with (
                patch("tools.promote_candidate.datetime", _TickingClock),
                redirect_stdout(StringIO()),
            ):
                code = promote_main(
                    [
                        str(manifest_path),
                        "--artifacts-dir",
                        str(artifacts),
                        "--reason",
                        "test promotion",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(_TickingClock.calls, 1)
            pointer = json.loads((artifacts / "approved.json").read_text())
            approved = json.loads(
                (
                    artifacts / "candidates" / "lp-test.approved.manifest.json"
                ).read_text()
            )
            self.assertEqual(approved["promotion"]["approved_at"], "2026-09-03T12:00:01Z")
            self.assertEqual(
                pointer["approved_at"], approved["promotion"]["approved_at"]
            )

    def test_runner_flag_conflicts_are_rejected(self) -> None:
        from contextlib import redirect_stderr
        from io import StringIO

        from run_agent import parse_args

        self.assertTrue(parse_args(["comp", "--learned"]).learned)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(["comp", "--learned", "--aggressive"])



class ServeEquityTrialsPinTests(unittest.TestCase):
    """An artifact may pin the precision of its own equity estimate.

    `equity_trials` decides the precision of the number every safety gate
    compares against, and until 2026-08-27 no approved artifact could
    record it -- it lived only as a Python default. That is the same
    unpinned-serve-parameter hole that let three unmeasured gate changes
    ship under an approved manifest.

    Precedence: explicit caller argument > `serve.equity_trials` in the
    manifest > `DEFAULT_SERVE_EQUITY_TRIALS`.
    """

    MANIFEST = "artifacts/candidates/candidate-v7-0001c.approved.manifest.json"

    def _with_serve(self, **serve):
        source = pathlib.Path(self.MANIFEST)
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["serve"] = dict(manifest.get("serve") or {}, **serve)
        target = source.parent / f"tmp-serve-pin-{self._counter()}.manifest.json"
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.addCleanup(lambda: target.unlink(missing_ok=True))
        return target

    _seq = 0

    @classmethod
    def _counter(cls) -> int:
        cls._seq += 1
        return cls._seq

    def test_an_unpinned_manifest_keeps_the_module_default(self) -> None:
        policy = load_policy(self.MANIFEST)
        self.assertEqual(policy.equity_trials, DEFAULT_SERVE_EQUITY_TRIALS)

    def test_a_pinned_manifest_wins_over_the_default(self) -> None:
        policy = load_policy(self._with_serve(equity_trials=1_234))
        self.assertEqual(policy.equity_trials, 1_234)

    def test_an_explicit_argument_wins_over_the_pin(self) -> None:
        policy = load_policy(self._with_serve(equity_trials=1_234), equity_trials=7)
        self.assertEqual(policy.equity_trials, 7)

    def test_a_nonsense_pin_is_refused_rather_than_served(self) -> None:
        for bad in (0, -5, 2.5, True, "many"):
            with self.subTest(bad=bad):
                with self.assertRaises(LearnedPolicyError):
                    load_policy(self._with_serve(equity_trials=bad))


if __name__ == "__main__":
    unittest.main()


class ApprovedPointerPortabilityTests(unittest.TestCase):
    """The pointer is promoted on Windows and read by the Linux live host."""

    def test_promotion_writes_posix_separators(self) -> None:
        pointer = json.loads(
            (Path("artifacts") / "approved.json").read_text(encoding="utf-8")
        )

        self.assertNotIn("\\", pointer["manifest_file"])

    def test_load_approved_accepts_a_windows_written_pointer(self) -> None:
        source = Path("artifacts")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(source / "candidates", root / "candidates")
            pointer = json.loads((source / "approved.json").read_text(encoding="utf-8"))
            # Re-introduce the legacy separator a Windows promotion produced.
            pointer["manifest_file"] = pointer["manifest_file"].replace("/", "\\")
            (root / "approved.json").write_text(json.dumps(pointer), encoding="utf-8")

            policy = load_approved(root, equity_trials=1)

        self.assertEqual(policy.policy_version, pointer["model_version"])


class ApprovedFormatDispatchTests(unittest.TestCase):
    """``load_approved`` routes by declared format, and refuses rather than
    falling back.

    Each promotable format has its own loader, its own input schema and
    its own branch meanings, so the failure this guards is an artifact of
    one shape being served under another's. These cases all refuse before
    any weights are read, which is why they need no artifact on disk.
    """

    def _pointer(self, root: Path, manifest: dict) -> None:
        candidates = root / "candidates"
        candidates.mkdir(parents=True, exist_ok=True)
        (candidates / "c.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (root / "approved.json").write_text(
            json.dumps(
                {
                    "approved_at": "2026-09-02T00:00:00Z",
                    "manifest_file": "candidates/c.manifest.json",
                    "model_version": "c",
                    "previous": None,
                    "weights_sha256": None,
                }
            ),
            encoding="utf-8",
        )

    def test_a_format_with_no_loader_is_refused_not_fallen_back_on(self) -> None:
        # Format 3 was never promotable. Silently handing it to the v7
        # loader would read a 413-input artifact against the 142-input
        # contract, so the refusal must be explicit.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._pointer(root, {"format_version": 3, "model_version": "c"})
            with self.assertRaises(LearnedPolicyError) as caught:
                load_approved(root, equity_trials=1)
            self.assertIn("no serve path", str(caught.exception))

    def test_a_contract_failure_surfaces_as_a_learned_policy_error(self) -> None:
        # run_agent's between-hands refresh catches LearnedPolicyError
        # only. LearningContractError is a sibling of it, not a subclass,
        # so letting one escape here would take a live session down
        # instead of keeping the policy already in hand. The exception
        # TYPE is the assertion; the refusal itself is incidental.
        from engine import schema4
        from engine.branch_contract_v9 import (
            BRANCH_LABELS_V9,
            MODEL_FORMAT_VERSION_V9,
        )
        from engine.v8_trainer import default_v9_architecture

        undeclared = {
            "format": "fold-multihead-policy",
            "format_version": MODEL_FORMAT_VERSION_V9,
            "model_version": "c",
            "state": "candidate",
            "parent_version": None,
            "created_at": "2026-09-02T12:00:00Z",
            "feature_schema_version": schema4.SCHEMA_VERSION_V9,
            "input_size": schema4.INPUT_SIZE_V9,
            "feature_names": list(schema4.FEATURE_NAMES_V9),
            "action_labels": list(BRANCH_LABELS_V9),
            "architecture": default_v9_architecture(),
            "serve": {},
            "weights_file": "weights.json",
            "weights_sha256": "b" * 64,
            "training_window": {"hand_count": 0},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._pointer(root, undeclared)
            with self.assertRaises(LearnedPolicyError):
                load_approved(root, equity_trials=1)

    def test_an_unreadable_manifest_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._pointer(root, {"format_version": 1})
            (root / "candidates" / "c.manifest.json").write_text(
                "not json", encoding="utf-8"
            )
            with self.assertRaises(LearnedPolicyError):
                load_approved(root, equity_trials=1)
