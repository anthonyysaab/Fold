"""CPU checks for the Phase-A v8 trainer.

The dataset/split/normalization contract runs everywhere (the module
imports torch function-locally); the training checks skip when torch is
absent, mirroring ``tests/test_cuda_trainer.py``'s idiom. Everything here
is tiny and CPU-only — no CUDA, no real dataset, no artifacts outside
temporary directories.
"""

from __future__ import annotations

import gzip
import json
import random
import tempfile
import unittest
from pathlib import Path

from engine import schema3
from engine.v8_trainer import (
    CARD_ENCODER_WIDTH,
    CONTEXT_ENCODER_WIDTH,
    CONTEXT_STD_FLOOR,
    MODEL_FAMILY_V8,
    MODEL_FORMAT_VERSION_V8,
    V8TrainingConfig,
    V8_HEAD_SIZES,
    check_v8_config,
    context_normalization,
    fit_phase_a,
    load_phase_a_dataset,
    split_rows,
    table_split_value,
    train_phase_a_candidate,
    validate_v8_manifest,
)

try:
    import torch
except ImportError:
    torch = None


def _features(rng: random.Random) -> list[float]:
    """A plausible schema-3 vector: binary card block, bounded context."""

    card = [0.0] * schema3.CARD_BLOCK_SIZE
    for index in rng.sample(range(schema3.CARD_BLOCK_SIZE), 5):
        card[index] = 1.0
    context = [rng.uniform(-1.0, 1.0) for _ in range(schema3.CONTEXT_BLOCK_SIZE)]
    return card + context


def _row_document(
    rng: random.Random,
    table_id: str,
    sequence: int,
    *,
    kind: str = "call",
    fold_through: float = 0.0,
    bucket: int = 3,
    equity: float = 0.4,
) -> dict[str, object]:
    """One raw dataset row. ``kind``: call | aggress_small | aggress_large."""

    masks = {
        "fold_through_small": 1 if kind == "aggress_small" else 0,
        "fold_through_large": 1 if kind == "aggress_large" else 0,
        "range_bucket": 0 if fold_through == 1.0 else 1,
        "equity_called": 0 if fold_through == 1.0 else 1,
    }
    labels = {
        "fold_through_small": fold_through if kind != "call" else 0.0,
        "fold_through_large": fold_through if kind != "call" else 0.0,
        "range_bucket": bucket,
        "equity_called": equity,
    }
    return {
        "table_id": table_id,
        "seat": 1,
        "street": rng.choice(("preflop", "flop", "turn", "river")),
        "actor_agent": "synthetic",
        "sequence": sequence,
        "features": _features(rng),
        "labels": labels,
        "masks": masks,
    }


def _write_dataset(path: Path, documents: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for document in documents:
            stream.write(json.dumps(document) + "\n")


def _synthetic_documents(
    tables: int = 16, rows_per_table: int = 3, seed: int = 5
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    kinds = ("call", "aggress_small", "aggress_large")
    documents = []
    for table in range(tables):
        table_id = f"table-{table:03d}"
        for sequence in range(rows_per_table):
            kind = kinds[(table + sequence) % len(kinds)]
            fold_through = 0.0
            if kind != "call" and (table + sequence) % 2 == 0:
                fold_through = 1.0
            documents.append(
                _row_document(
                    rng,
                    table_id,
                    sequence,
                    kind=kind,
                    fold_through=fold_through,
                    bucket=(table + sequence) % schema3.BELIEF_BUCKETS,
                    equity=rng.uniform(0.0, 1.0),
                )
            )
    return documents


def _load_synthetic(directory: str, **kwargs: object):
    path = Path(directory) / "tiny.jsonl.gz"
    _write_dataset(path, _synthetic_documents(**kwargs))  # type: ignore[arg-type]
    return load_phase_a_dataset(path)


_TINY_CONFIG = V8TrainingConfig(
    epochs=3,
    learning_rate=1e-3,
    warmup_steps=4,
    early_stop_patience=5,
    batch_size=16,
    validation_fraction=0.25,
    split_seed=17,
    init_seed=11,
    device="cpu",
)


class PhaseADatasetTests(unittest.TestCase):
    """Loader, split, and normalization contracts — no torch required."""

    def test_load_round_trip_and_equity_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
        self.assertEqual(len(rows), 48)
        for row in rows:
            self.assertEqual(len(row.features), schema3.INPUT_SIZE_V8)
            small, large = row.fold_through_mask
            self.assertFalse(small and large)
            expected_slot = 0 if small else 1 if large else 2
            self.assertEqual(row.equity_slot, expected_slot)
            if row.fold_through_label == 1.0:
                # Everyone folded: no continuing range or equity labels.
                self.assertEqual(row.range_mask, 0)
                self.assertEqual(row.equity_mask, 0)

    def test_load_rejects_wrong_feature_length(self) -> None:
        rng = random.Random(0)
        document = _row_document(rng, "t", 0)
        document["features"] = [0.0] * (schema3.INPUT_SIZE_V8 - 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl.gz"
            _write_dataset(path, [document])
            with self.assertRaisesRegex(ValueError, "413"):
                load_phase_a_dataset(path)

    def test_load_rejects_non_finite_feature(self) -> None:
        rng = random.Random(0)
        document = _row_document(rng, "t", 0)
        document["features"][100] = float("nan")  # type: ignore[index]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl.gz"
            _write_dataset(path, [document])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_phase_a_dataset(path)

    def test_load_rejects_both_branch_masks(self) -> None:
        rng = random.Random(0)
        document = _row_document(rng, "t", 0, kind="aggress_small")
        document["masks"]["fold_through_large"] = 1  # type: ignore[index]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl.gz"
            _write_dataset(path, [document])
            with self.assertRaisesRegex(ValueError, "both fold_through"):
                load_phase_a_dataset(path)

    def test_load_rejects_out_of_range_bucket(self) -> None:
        rng = random.Random(0)
        document = _row_document(rng, "t", 0, bucket=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl.gz"
            _write_dataset(path, [document])
            with self.assertRaisesRegex(ValueError, "range_bucket"):
                load_phase_a_dataset(path)

    def test_split_is_deterministic_disjoint_and_whole_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
        first_train, first_validation = split_rows(rows, _TINY_CONFIG)
        second_train, second_validation = split_rows(rows, _TINY_CONFIG)
        self.assertEqual(first_train, second_train)
        self.assertEqual(first_validation, second_validation)
        self.assertEqual(len(first_train) + len(first_validation), len(rows))
        train_tables = {row.table_id for row in first_train}
        validation_tables = {row.table_id for row in first_validation}
        self.assertTrue(train_tables)
        self.assertTrue(validation_tables)
        self.assertFalse(train_tables & validation_tables)
        for table_id in validation_tables:
            self.assertLess(
                table_split_value(_TINY_CONFIG.split_seed, table_id),
                _TINY_CONFIG.validation_fraction,
            )

    def test_context_normalization_card_identity_and_std_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
        constant_index = schema3.CONTEXT_INDICES[0]
        pinned = [
            type(row)(
                **{
                    **{
                        name: getattr(row, name)
                        for name in row.__dataclass_fields__
                    },
                    "features": tuple(
                        3.25 if index == constant_index else value
                        for index, value in enumerate(row.features)
                    ),
                }
            )
            for row in rows
        ]
        means, stds = context_normalization(pinned)
        self.assertEqual(len(means), schema3.INPUT_SIZE_V8)
        self.assertEqual(len(stds), schema3.INPUT_SIZE_V8)
        for index in schema3.CARD_INDICES:
            self.assertEqual(means[index], 0.0)
            self.assertEqual(stds[index], 1.0)
        self.assertAlmostEqual(means[constant_index], 3.25)
        self.assertEqual(stds[constant_index], CONTEXT_STD_FLOOR)
        for index in schema3.CONTEXT_INDICES:
            self.assertGreaterEqual(stds[index], CONTEXT_STD_FLOOR)

    def test_config_validation_fails_closed(self) -> None:
        for broken in (
            V8TrainingConfig(epochs=0),
            V8TrainingConfig(dropout=1.0),
            V8TrainingConfig(learning_rate=0.0),
            V8TrainingConfig(validation_fraction=0.0),
            V8TrainingConfig(device="tpu"),
        ):
            with self.assertRaises(ValueError):
                check_v8_config(broken)


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class V8TrainerTorchTests(unittest.TestCase):
    """Tiny CPU fits; skipped wherever torch is not installed."""

    def test_trains_and_exports_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
            dataset_path = Path(directory) / "tiny.jsonl.gz"
            config = V8TrainingConfig(
                **{
                    **{
                        name: getattr(_TINY_CONFIG, name)
                        for name in _TINY_CONFIG.__dataclass_fields__
                    },
                    "model_version": "candidate-v8-test",
                }
            )
            summary = train_phase_a_candidate(
                rows,
                directory,
                config,
                init_seeds=(11, 12),
                dataset_path=dataset_path,
            )
            manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
            weights_bytes = summary.weights_path.read_bytes()

        validate_v8_manifest(manifest)
        self.assertEqual(manifest["format_version"], MODEL_FORMAT_VERSION_V8)
        self.assertEqual(manifest["architecture"]["family"], MODEL_FAMILY_V8)
        self.assertEqual(manifest["state"], "candidate")
        self.assertIsNone(manifest["promotion"])
        self.assertEqual(manifest["input_size"], schema3.INPUT_SIZE_V8)
        self.assertEqual(
            manifest["feature_names"], list(schema3.FEATURE_NAMES_V8)
        )
        self.assertEqual(
            len(manifest["training"]["init_seeds_evaluated"]), 2
        )
        self.assertIn(
            manifest["training"]["init_seed"], (11, 12)
        )
        self.assertEqual(summary.selected_init_seed, manifest["training"]["init_seed"])

        import hashlib

        self.assertEqual(
            hashlib.sha256(weights_bytes.rstrip(b"\n")).hexdigest(),
            manifest["weights_sha256"],
        )
        document = json.loads(weights_bytes)
        weights = document["weights"]
        self.assertEqual(
            len(weights["card_encoder"]["w"]), CARD_ENCODER_WIDTH
        )
        self.assertEqual(
            len(weights["card_encoder"]["w"][0]), schema3.CARD_BLOCK_SIZE
        )
        self.assertEqual(
            len(weights["context_encoder"]["w"]), CONTEXT_ENCODER_WIDTH
        )
        self.assertEqual(
            len(weights["context_encoder"]["w"][0]), schema3.CONTEXT_BLOCK_SIZE
        )
        self.assertNotIn("ln_g", weights["trunk"][-1])
        # The residual head ships zero-initialized and untrained.
        residual = weights["heads"]["residual"]
        self.assertTrue(
            all(value == 0.0 for row in residual["out_w"] for value in row)
        )
        self.assertTrue(all(value == 0.0 for value in residual["out_b"]))
        # The supervised heads did train away from the zero init.
        for name in ("fold_through", "range", "equity_called"):
            out_w = weights["heads"][name]["out_w"]
            self.assertEqual(len(out_w), V8_HEAD_SIZES[name])
            self.assertTrue(any(value != 0.0 for row in out_w for value in row))
        # Card block identity scales, context block z-scored.
        normalization = document["feature_normalization"]
        for index in schema3.CARD_INDICES:
            self.assertEqual(normalization["means"][index], 0.0)
            self.assertEqual(normalization["stds"][index], 1.0)
        for index in schema3.CONTEXT_INDICES:
            self.assertGreaterEqual(
                normalization["stds"][index], CONTEXT_STD_FLOOR - 1e-9
            )

    def test_same_seed_is_deterministic_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
        first = fit_phase_a(rows, _TINY_CONFIG)
        second = fit_phase_a(rows, _TINY_CONFIG)
        self.assertEqual(
            json.dumps(first["weights"], sort_keys=True),
            json.dumps(second["weights"], sort_keys=True),
        )
        self.assertEqual(first["validation_losses"], second["validation_losses"])

    def test_fully_masked_out_head_stays_at_zero_init(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.jsonl.gz"
            documents = _synthetic_documents()
            for document in documents:
                document["masks"]["range_bucket"] = 0  # type: ignore[index]
            _write_dataset(path, documents)
            rows = load_phase_a_dataset(path)
        result = fit_phase_a(rows, _TINY_CONFIG)
        range_head = result["weights"]["heads"]["range"]  # type: ignore[index]
        self.assertTrue(
            all(value == 0.0 for row in range_head["out_w"] for value in row)
        )
        fold_through = result["weights"]["heads"]["fold_through"]  # type: ignore[index]
        self.assertTrue(
            any(value != 0.0 for row in fold_through["out_w"] for value in row)
        )
        self.assertIsNone(result["validation_losses"]["range"])  # type: ignore[index]

    def test_per_head_losses_and_calibration_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
        result = fit_phase_a(rows, _TINY_CONFIG)
        for split in ("train_losses", "validation_losses"):
            losses = result[split]
            for head in ("fold_through", "range", "equity_called", "total"):
                self.assertIn(head, losses)  # type: ignore[operator]
        calibration = result["calibration"]
        self.assertIn("fold_through_deciles", calibration)  # type: ignore[operator]
        self.assertIn("range_buckets", calibration)  # type: ignore[operator]
        buckets = calibration["range_buckets"]["buckets"]  # type: ignore[index]
        self.assertEqual(len(buckets), schema3.BELIEF_BUCKETS)
        empirical_total = sum(entry["empirical"] for entry in buckets)
        self.assertAlmostEqual(empirical_total, 1.0, places=3)

    def test_export_refuses_to_overwrite_an_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
            config = V8TrainingConfig(
                **{
                    **{
                        name: getattr(_TINY_CONFIG, name)
                        for name in _TINY_CONFIG.__dataclass_fields__
                    },
                    "model_version": "candidate-v8-test",
                }
            )
            (Path(directory) / "candidate-v8-test.weights.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                train_phase_a_candidate(rows, directory, config, init_seeds=(11,))


if __name__ == "__main__":
    unittest.main()
