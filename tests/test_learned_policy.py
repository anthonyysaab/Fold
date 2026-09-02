"""Checks for the learned runtime, promotion, and rollback path."""

from __future__ import annotations

import json
import shutil
import tempfile
import pathlib
import unittest
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
