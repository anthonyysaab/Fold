"""CPU checks for the Phase-A v9 trainer.

The dataset/split/normalization contract runs everywhere (the module
imports torch function-locally); the training checks skip when torch is
absent, mirroring ``tests/test_v8_trainer.py``'s idiom. Expectations are
derived independently of the implementation (the standing rule): equity
slots are asserted as hand-derived literal indices of the contract's
pinned ``(passive, active, aggressive)`` order, and the version-guard
tests prove the pinned property itself — a v8 file fails in the v9
loader and a v9 file fails in the v8 loader, in BOTH directions,
regardless of vector length.
"""

from __future__ import annotations

import gzip
import json
import random
import tempfile
import unittest
from pathlib import Path

from engine import schema4
from engine.branch_contract_v9 import MODEL_FORMAT_VERSION_V9
from engine.rules.composition import composed_sizing_record
from engine.v8_trainer import (
    CONTEXT_STD_FLOOR,
    V8TrainingConfig,
    default_v9_architecture,
    load_phase_a_dataset,
)
from engine.v9_trainer import (
    context_normalization_v9,
    fit_phase_a_v9,
    load_phase_a_dataset_v9,
    resolve_sizing_record,
    train_phase_a_candidate_v9,
    validate_v9_manifest,
)

try:
    import torch
except ImportError:
    torch = None

#: The four row kinds and their hand-derived slot in the contract's
#: pinned equity order (passive=0, active=1, aggressive=2): a check
#: observes the checked-through conditional; a call (and a fold) the
#: continuing set at the existing price; a sized wager its own lane.
_KIND_EXPECTATIONS = {
    "check": {"to_call_zero": True, "ft": None, "slot": 0},
    "call": {"to_call_zero": False, "ft": None, "slot": 1},
    "active_bet": {"to_call_zero": True, "ft": "fold_through_active", "slot": 1},
    "aggressive": {"to_call_zero": False, "ft": "fold_through_aggressive", "slot": 2},
}


def _features(rng: random.Random) -> list[float]:
    """A plausible schema-4 vector: binary card block, bounded context."""

    card = [0.0] * schema4.CARD_BLOCK_SIZE_V9
    for index in rng.sample(range(schema4.CARD_BLOCK_SIZE_V9), 5):
        card[index] = 1.0
    context = [
        rng.uniform(-1.0, 1.0) for _ in range(schema4.CONTEXT_BLOCK_SIZE_V9)
    ]
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
    read_x10: int = 450,
) -> dict[str, object]:
    """One raw v9 dataset row. ``kind`` is a `_KIND_EXPECTATIONS` key."""

    expectation = _KIND_EXPECTATIONS[kind]
    ft_key = expectation["ft"]
    masks = {
        "fold_through_active": 1 if ft_key == "fold_through_active" else 0,
        "fold_through_aggressive": 1 if ft_key == "fold_through_aggressive" else 0,
        "range_bucket": 0 if fold_through == 1.0 else 1,
        "equity_called": 0 if fold_through == 1.0 else 1,
    }
    labels = {
        "fold_through_active": fold_through if ft_key else 0.0,
        "fold_through_aggressive": fold_through if ft_key else 0.0,
        "range_bucket": bucket,
        "equity_called": equity,
    }
    return {
        "table_id": table_id,
        "seat": 1,
        "street": rng.choice(("preflop", "flop", "turn", "river")),
        "actor_agent": "synthetic",
        "sequence": sequence,
        "to_call_zero": expectation["to_call_zero"],
        "read_temperature_x10": read_x10,
        "features": _features(rng),
        "labels": labels,
        "masks": masks,
    }


def _write_dataset(path: Path, documents: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for document in documents:
            stream.write(json.dumps(document) + "\n")


def _synthetic_documents(
    tables: int = 16, rows_per_table: int = 4, seed: int = 5
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    kinds = tuple(_KIND_EXPECTATIONS)
    documents = []
    for table in range(tables):
        table_id = f"table-{table:03d}"
        for sequence in range(rows_per_table):
            kind = kinds[(table + sequence) % len(kinds)]
            fold_through = 0.0
            if _KIND_EXPECTATIONS[kind]["ft"] and (table + sequence) % 2 == 0:
                fold_through = 1.0
            documents.append(
                _row_document(
                    rng,
                    table_id,
                    sequence,
                    kind=kind,
                    fold_through=fold_through,
                    bucket=(table + sequence) % schema4.BELIEF_BUCKETS,
                    equity=rng.uniform(0.0, 1.0),
                    read_x10=rng.randrange(0, 1001),
                )
            )
    return documents


def _load_synthetic(directory: str, **kwargs: object):
    path = Path(directory) / "tiny-v9.jsonl.gz"
    _write_dataset(path, _synthetic_documents(**kwargs))  # type: ignore[arg-type]
    return load_phase_a_dataset_v9(path)


def _loader_rejects(test: unittest.TestCase, document: dict, pattern: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.jsonl.gz"
        _write_dataset(path, [document])
        with test.assertRaisesRegex(ValueError, pattern):
            load_phase_a_dataset_v9(path)


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


class PhaseADatasetV9Tests(unittest.TestCase):
    """Loader and normalization contracts — no torch required."""

    def test_load_round_trip_and_equity_slots(self) -> None:
        rng = random.Random(3)
        documents = []
        for index, kind in enumerate(_KIND_EXPECTATIONS):
            documents.append(
                _row_document(rng, f"t-{index}", index, kind=kind, read_x10=7)
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kinds.jsonl.gz"
            _write_dataset(path, documents)
            rows = load_phase_a_dataset_v9(path)
        self.assertEqual(len(rows), len(_KIND_EXPECTATIONS))
        for row, kind in zip(rows, _KIND_EXPECTATIONS):
            expectation = _KIND_EXPECTATIONS[kind]
            self.assertEqual(len(row.features), schema4.INPUT_SIZE_V9)
            self.assertEqual(row.to_call_zero, expectation["to_call_zero"])
            self.assertEqual(row.equity_slot, expectation["slot"])
            self.assertEqual(row.read_temperature_x10, 7)
            self.assertLessEqual(sum(row.fold_through_mask), 1)

    def test_load_full_synthetic_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
        self.assertEqual(len(rows), 64)
        for row in rows:
            if row.fold_through_label == 1.0:
                # Everyone folded: no continuing range or equity labels.
                self.assertEqual(row.range_mask, 0)
                self.assertEqual(row.equity_mask, 0)

    def test_rejects_v8_key_names_with_guidance(self) -> None:
        """The renamed keys ARE the version guard (pinned): a v8 row must
        fail loudly regardless of vector length."""

        rng = random.Random(0)
        document = _row_document(rng, "t", 0, kind="call")
        masks = dict(document["masks"])  # type: ignore[arg-type]
        del masks["fold_through_active"], masks["fold_through_aggressive"]
        masks["fold_through_small"] = 0
        masks["fold_through_large"] = 0
        document["masks"] = masks
        _loader_rejects(self, document, "v8")

    def test_v8_loader_refuses_a_v9_row(self) -> None:
        """The guard works in the other direction too."""

        rng = random.Random(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v9.jsonl.gz"
            _write_dataset(path, [_row_document(rng, "t", 0)])
            with self.assertRaises(ValueError):
                load_phase_a_dataset(path)

    def test_rejects_both_branch_masks(self) -> None:
        rng = random.Random(0)
        document = _row_document(rng, "t", 0, kind="active_bet")
        document["masks"]["fold_through_aggressive"] = 1  # type: ignore[index]
        _loader_rejects(self, document, "both fold_through")

    def test_rejects_active_supervision_on_a_priced_row(self) -> None:
        rng = random.Random(0)
        document = _row_document(rng, "t", 0, kind="active_bet")
        document["to_call_zero"] = False
        _loader_rejects(self, document, "fold_through_active")

    def test_rejects_aggressive_supervision_on_a_free_row(self) -> None:
        rng = random.Random(0)
        document = _row_document(rng, "t", 0, kind="aggressive")
        document["to_call_zero"] = True
        _loader_rejects(self, document, "fold_through_aggressive")

    def test_rejects_missing_or_non_bool_to_call_zero(self) -> None:
        rng = random.Random(0)
        for broken in (None, 1, "true"):
            document = _row_document(rng, "t", 0)
            if broken is None:
                del document["to_call_zero"]
            else:
                document["to_call_zero"] = broken
            _loader_rejects(self, document, "to_call_zero")

    def test_rejects_bad_temperature_reads(self) -> None:
        rng = random.Random(0)
        for broken in (-1, 1001, True, 45.0, None):
            document = _row_document(rng, "t", 0)
            document["read_temperature_x10"] = broken
            _loader_rejects(self, document, "read_temperature_x10")

    def test_rejects_wrong_feature_length(self) -> None:
        rng = random.Random(0)
        document = _row_document(rng, "t", 0)
        document["features"] = [0.0] * (schema4.INPUT_SIZE_V9 - 1)
        _loader_rejects(self, document, str(schema4.INPUT_SIZE_V9))

    def test_rejects_non_finite_feature(self) -> None:
        rng = random.Random(0)
        document = _row_document(rng, "t", 0)
        document["features"][100] = float("nan")  # type: ignore[index]
        _loader_rejects(self, document, "non-finite")

    def test_rejects_out_of_range_bucket(self) -> None:
        rng = random.Random(0)
        document = _row_document(
            rng, "t", 0, bucket=schema4.BELIEF_BUCKETS
        )
        _loader_rejects(self, document, "range_bucket")

    def test_context_normalization_card_identity_and_std_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
        constant_index = schema4.CONTEXT_INDICES_V9[0]
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
        means, stds = context_normalization_v9(pinned)
        self.assertEqual(len(means), schema4.INPUT_SIZE_V9)
        self.assertEqual(len(stds), schema4.INPUT_SIZE_V9)
        for index in schema4.CARD_INDICES_V9:
            self.assertEqual(means[index], 0.0)
            self.assertEqual(stds[index], 1.0)
        self.assertAlmostEqual(means[constant_index], 3.25)
        self.assertEqual(stds[constant_index], CONTEXT_STD_FLOOR)
        for index in schema4.CONTEXT_INDICES_V9:
            self.assertGreaterEqual(stds[index], CONTEXT_STD_FLOOR)


def _canonical(record: dict) -> dict:
    """The on-disk form of a sizing record (tuples become lists)."""

    return json.loads(json.dumps(record))


class SizingRecordResolutionTests(unittest.TestCase):
    def test_default_is_the_composed_record_with_dials_off(self) -> None:
        self.assertEqual(
            resolve_sizing_record(None, None), _canonical(composed_sizing_record())
        )

    def test_sidecar_record_wins_when_no_explicit_record(self) -> None:
        record = composed_sizing_record()
        resolved = resolve_sizing_record(None, {"sizing": record})
        self.assertEqual(resolved, _canonical(record))

    def test_explicit_record_must_agree_with_the_sidecar(self) -> None:
        record = composed_sizing_record()
        disagreeing = json.loads(json.dumps(record))
        disagreeing["parameters"]["active_base"] = 0.45
        with self.assertRaisesRegex(ValueError, "disagrees"):
            resolve_sizing_record(disagreeing, {"sizing": record})

    def test_foreign_identity_is_refused(self) -> None:
        record = composed_sizing_record()
        record["identity"] = "g-v9-other"
        with self.assertRaisesRegex(ValueError, "identity"):
            resolve_sizing_record(record, None)

    def test_bare_g_record_without_rules_is_refused(self) -> None:
        record = composed_sizing_record()
        del record["rules"]
        with self.assertRaisesRegex(ValueError, "rules"):
            resolve_sizing_record(record, None)


def _minimal_manifest() -> dict[str, object]:
    """A structurally valid format-4 manifest for validator tests."""

    return {
        "format": "fold-multihead-policy",
        "format_version": MODEL_FORMAT_VERSION_V9,
        "model_version": "candidate-v9-validator-test",
        "state": "candidate",
        "parent_version": None,
        "created_at": "2026-08-30T00:00:00Z",
        "feature_schema_version": schema4.SCHEMA_VERSION_V9,
        "input_size": schema4.INPUT_SIZE_V9,
        "feature_names": list(schema4.FEATURE_NAMES_V9),
        "action_labels": ["fatal", "passive", "active", "aggressive"],
        "architecture": default_v9_architecture(),
        "sizing": composed_sizing_record(),
        "weights_file": "candidate-v9-validator-test.weights.json",
        "weights_sha256": "0" * 64,
        "training_window": {},
        "engine_parameters": {},
        "serve": {"ood_guard_indices": list(schema4.CONTEXT_INDICES_V9)},
        "training": {},
        "evaluation": {},
        "promotion": None,
    }


_DELETE = object()


class ManifestValidatorTests(unittest.TestCase):
    def test_accepts_a_well_formed_candidate(self) -> None:
        validate_v9_manifest(_minimal_manifest())

    def test_refusals(self) -> None:
        cases = {
            "v8 format version": {"format_version": 3},
            "missing sizing": {"sizing": _DELETE},
            "foreign sizing identity": {
                "sizing": {**composed_sizing_record(), "identity": "g-x"}
            },
            "v8 labels": {
                "action_labels": [
                    "fold",
                    "check_call",
                    "aggress_small",
                    "aggress_large",
                ]
            },
            "hybrid quantiles": {"serve": {"margin_quantiles": {"p90": 0.02}}},
            "non-candidate state": {"state": "approved"},
            "promotion record": {"promotion": {"who": "nobody"}},
            "wrong input size": {"input_size": 412},
        }
        for reason, mutation in cases.items():
            manifest = _minimal_manifest()
            for key, value in mutation.items():
                if value is _DELETE:
                    del manifest[key]
                else:
                    manifest[key] = value
            with self.assertRaises(ValueError, msg=reason):
                validate_v9_manifest(manifest)


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class V9TrainerTorchTests(unittest.TestCase):
    """Tiny CPU fits; skipped wherever torch is not installed."""

    def test_trains_and_exports_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
            dataset_path = Path(directory) / "tiny-v9.jsonl.gz"
            config = V8TrainingConfig(
                **{
                    **{
                        name: getattr(_TINY_CONFIG, name)
                        for name in _TINY_CONFIG.__dataclass_fields__
                    },
                    "model_version": "candidate-v9-test",
                }
            )
            summary = train_phase_a_candidate_v9(
                rows,
                directory,
                config,
                init_seeds=(11, 12),
                dataset_path=dataset_path,
            )
            manifest = json.loads(
                Path(summary["manifest_path"]).read_text(encoding="utf-8")
            )
            weights_bytes = Path(summary["weights_path"]).read_bytes()

        validate_v9_manifest(manifest)
        self.assertEqual(manifest["format_version"], MODEL_FORMAT_VERSION_V9)
        self.assertEqual(manifest["architecture"]["family"], "v9-composed-value")
        self.assertEqual(manifest["state"], "candidate")
        self.assertIsNone(manifest["promotion"])
        self.assertEqual(manifest["input_size"], schema4.INPUT_SIZE_V9)
        self.assertEqual(manifest["feature_names"], list(schema4.FEATURE_NAMES_V9))
        self.assertEqual(
            manifest["action_labels"],
            ["fatal", "passive", "active", "aggressive"],
        )
        # No explicit record, no sidecar: the module default ships (in
        # its canonical JSON form — tuples become lists on disk).
        self.assertEqual(
            manifest["sizing"],
            json.loads(json.dumps(composed_sizing_record())),
        )
        coverage = manifest["training_window"]["label_coverage"]
        self.assertIn("fold_through_active", coverage)
        self.assertIn("fold_through_aggressive", coverage)
        self.assertIn("free_spot_rows", coverage)
        self.assertGreater(coverage["fold_through_active"], 0)
        self.assertGreater(coverage["fold_through_aggressive"], 0)

        import hashlib

        self.assertEqual(
            hashlib.sha256(weights_bytes.rstrip(b"\n")).hexdigest(),
            manifest["weights_sha256"],
        )
        # Pre-harvest decision 5: the trainer's normalization block must
        # carry the stamp, and the loader must accept exactly that block.
        document = json.loads(weights_bytes.decode("utf-8"))
        normalization = document["feature_normalization"]
        self.assertEqual(
            {key: normalization[key] for key in schema4.normalization_stamp()},
            schema4.normalization_stamp(),
        )
        schema4.require_normalization_stamp(normalization)  # must not raise
        document = json.loads(weights_bytes)
        weights = document["weights"]
        self.assertEqual(
            len(weights["card_encoder"]["w"][0]), schema4.CARD_BLOCK_SIZE_V9
        )
        self.assertEqual(
            len(weights["context_encoder"]["w"][0]),
            schema4.CONTEXT_BLOCK_SIZE_V9,
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
            self.assertTrue(any(value != 0.0 for row in out_w for value in row))
        # Card block identity scales, context block z-scored.
        normalization = document["feature_normalization"]
        for index in schema4.CARD_INDICES_V9:
            self.assertEqual(normalization["means"][index], 0.0)
            self.assertEqual(normalization["stds"][index], 1.0)
        for index in schema4.CONTEXT_INDICES_V9:
            self.assertGreaterEqual(
                normalization["stds"][index], CONTEXT_STD_FLOOR - 1e-9
            )

    def test_same_seed_is_deterministic_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
        first = fit_phase_a_v9(rows, _TINY_CONFIG)
        second = fit_phase_a_v9(rows, _TINY_CONFIG)
        self.assertEqual(
            json.dumps(first["weights"], sort_keys=True),
            json.dumps(second["weights"], sort_keys=True),
        )
        self.assertEqual(first["validation_losses"], second["validation_losses"])

    def test_fully_masked_out_head_stays_at_zero_init(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny-v9.jsonl.gz"
            documents = _synthetic_documents()
            for document in documents:
                document["masks"]["range_bucket"] = 0  # type: ignore[index]
            _write_dataset(path, documents)
            rows = load_phase_a_dataset_v9(path)
        result = fit_phase_a_v9(rows, _TINY_CONFIG)
        range_head = result["weights"]["heads"]["range"]  # type: ignore[index]
        self.assertTrue(
            all(value == 0.0 for row in range_head["out_w"] for value in row)
        )
        fold_through = result["weights"]["heads"]["fold_through"]  # type: ignore[index]
        self.assertTrue(
            any(value != 0.0 for row in fold_through["out_w"] for value in row)
        )
        self.assertIsNone(result["validation_losses"]["range"])  # type: ignore[index]

    def test_export_refuses_to_overwrite_an_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = _load_synthetic(directory)
            config = V8TrainingConfig(
                **{
                    **{
                        name: getattr(_TINY_CONFIG, name)
                        for name in _TINY_CONFIG.__dataclass_fields__
                    },
                    "model_version": "candidate-v9-test",
                }
            )
            (Path(directory) / "candidate-v9-test.weights.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                train_phase_a_candidate_v9(rows, directory, config, init_seeds=(11,))


if __name__ == "__main__":
    unittest.main()
