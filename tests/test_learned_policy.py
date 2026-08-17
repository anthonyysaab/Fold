"""Checks for the learned runtime, promotion, and rollback path."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from devfun_poker_playground.decision_engine import DEFAULT_SAFETY_GATES
from devfun_poker_playground.learned_policy import (
    approved_fingerprint,
    LearnedPolicyError,
    load_approved,
    load_policy,
)
from devfun_poker_playground.learning_contract import LEARNING_INPUT_SIZE
from devfun_poker_playground.offline_trainer import TrainingConfig, train_candidate
from devfun_poker_playground.training_telemetry import TrainingExample
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
                    "devfun_poker_playground.learned_policy._forward",
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
                    "devfun_poker_playground.learned_policy._forward",
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
                    "devfun_poker_playground.learned_policy._forward",
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
