"""CUDA smoke check for the vectorized candidate trainer."""

from __future__ import annotations

import json
import tempfile
import unittest

from devfun_poker_playground.learning_contract import LEARNING_INPUT_SIZE
from devfun_poker_playground.offline_trainer import TrainingConfig, train_candidate
from devfun_poker_playground.training_telemetry import TrainingExample

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(), "CUDA PyTorch is unavailable"
)
class CudaTrainerTests(unittest.TestCase):
    def test_cuda_backend_writes_a_candidate(self) -> None:
        examples = tuple(
            TrainingExample(
                table_id=f"hand-{index}",
                policy_version="cuda-smoke",
                features=tuple(
                    float(index % 3) if feature == 0 else 0.1
                    for feature in range(LEARNING_INPUT_SIZE)
                ),
                action_family_index=index % 3,
                behavior_probabilities=(1 / 3, 1 / 3, 1 / 3),
                submitted_risk_fraction=0.2 if index % 3 == 2 else 0.0,
                purse_bb=60.0,
                reward_bb=(-1.0, 0.5, 2.0)[index % 3],
                counterfactual=True,
                opponent_confidence=1.0,
            )
            for index in range(12)
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = train_candidate(
                examples,
                directory,
                TrainingConfig(
                    epochs=1,
                    behavior_warmup_epochs=1,
                    validation_fraction=0.25,
                    model_version="cuda-smoke",
                    device="cuda",
                    batch_size=4,
                ),
            )
            manifest = json.loads(summary.manifest_path.read_text())

        self.assertEqual(manifest["training"]["backend"], "pytorch")
        self.assertEqual(manifest["training"]["device"], "cuda")
        self.assertIn("Quadro RTX 3000", manifest["training"]["device_name"])
        self.assertEqual(manifest["training"]["batch_size"], 4)


if __name__ == "__main__":
    unittest.main()
