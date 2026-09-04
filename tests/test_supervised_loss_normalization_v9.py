"""Tests for the v9 supervised-loss normalization (Phase 1, 2026-09-02).

Stdlib everywhere: the constant-predictor baselines against HAND-DERIVED
numbers, the bias-only-optimum invariant (the marginal is the minimizer
of the trainer's own masked loss, so any perturbation of it scores
worse — a check on the closed forms that does not mirror them), the
floors, and the config refusals. The torch section pins the two facts
the pre-registration leans on: ``raw`` mode is the shipped objective
bit-for-bit, and ``constant-predictor`` mode's ``total`` is exactly the
weighted normalized sum the manifest describes. Skipped without torch.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from engine import schema4
from training.supervised_loss_normalization_v9 import (
    BASELINE_FLOOR,
    SUPERVISED_HEADS_V9,
    SupervisedLossConfigV9,
    check_supervised_loss_config,
    constant_predictor_baselines,
    supervised_head_scales,
)
from training.v9_trainer import PhaseARowV9

try:
    import torch
except ImportError:
    torch = None


def _row(
    table: str,
    *,
    ft_label: float = 0.0,
    ft_mask: tuple[int, ...] = (0, 0),
    bucket: int = 0,
    range_mask: int = 0,
    equity: float = 0.0,
    equity_mask: int = 0,
    slot: int = 0,
) -> PhaseARowV9:
    return PhaseARowV9(
        table_id=table,
        street="flop",
        features=(0.0,) * schema4.INPUT_SIZE_V9,
        fold_through_label=ft_label,
        fold_through_mask=ft_mask,
        range_bucket=bucket,
        range_mask=range_mask,
        equity_called=equity,
        equity_mask=equity_mask,
        equity_slot=slot,
        to_call_zero=False,
        read_temperature_x10=450,
    )


def _hand_rows() -> list[PhaseARowV9]:
    """Lane 0 labels 1,0,0,0 (p = 1/4); lane 1 labels 1,1 (p = 1);
    buckets 0,0,1,2; equity slot 0 = {0.2, 0.4}, slot 1 = {0.6}."""

    return [
        _row("a", ft_label=1.0, ft_mask=(1, 0), bucket=0, range_mask=1, equity=0.2, equity_mask=1, slot=0),
        _row("b", ft_label=0.0, ft_mask=(1, 0), bucket=0, range_mask=1, equity=0.4, equity_mask=1, slot=0),
        _row("c", ft_label=0.0, ft_mask=(1, 0), bucket=1, range_mask=1, equity=0.6, equity_mask=1, slot=1),
        _row("d", ft_label=0.0, ft_mask=(1, 0), bucket=2, range_mask=1),
        _row("e", ft_label=1.0, ft_mask=(0, 1)),
        _row("f", ft_label=1.0, ft_mask=(0, 1)),
        _row("g"),  # carries no label at all
    ]


class HandDerivedBaselineTest(unittest.TestCase):
    def test_hand_derived_numbers(self) -> None:
        result = constant_predictor_baselines(_hand_rows())
        # Lane 0: H(1/4) = -(1/4 ln 1/4 + 3/4 ln 3/4) = 0.562335; lane 1:
        # H(1) = 0. Lane-count weighted: (4 * 0.562335 + 2 * 0) / 6.
        self.assertAlmostEqual(result.measured["fold_through"], 0.374890, places=6)
        # Buckets (1/2, 1/4, 1/4): 1/2 ln 2 + 2 * 1/4 ln 4 = 1.039721.
        self.assertAlmostEqual(result.measured["range"], 1.039721, places=6)
        # Slot 0 mean 0.3, deviations 0.01 + 0.01; slot 1 deviation 0; / 3.
        self.assertAlmostEqual(result.measured["equity_called"], 0.02 / 3, places=9)
        self.assertEqual(
            result.masked_rows,
            {"fold_through": 6, "range": 4, "equity_called": 3},
        )
        # None of these reach the floor.
        self.assertEqual(result.baselines, result.measured)
        record = result.as_record()
        self.assertEqual(tuple(record), SUPERVISED_HEADS_V9)
        self.assertEqual(record["range"]["masked_rows"], 4)

    def test_no_labels_floors_every_head(self) -> None:
        result = constant_predictor_baselines([_row("x"), _row("y")])
        self.assertEqual(result.measured, {name: 0.0 for name in SUPERVISED_HEADS_V9})
        self.assertEqual(
            result.baselines, {name: BASELINE_FLOOR for name in SUPERVISED_HEADS_V9}
        )
        self.assertEqual(result.masked_rows, {name: 0 for name in SUPERVISED_HEADS_V9})

    def test_a_constant_label_floors_only_its_head(self) -> None:
        rows = [
            _row("x", ft_label=1.0, ft_mask=(1, 1), bucket=3, range_mask=1, equity=0.5, equity_mask=1),
            _row("y", ft_label=1.0, ft_mask=(1, 1), bucket=3, range_mask=1, equity=0.7, equity_mask=1),
        ]
        result = constant_predictor_baselines(rows)
        self.assertEqual(result.measured["fold_through"], 0.0)
        self.assertEqual(result.measured["range"], 0.0)
        self.assertEqual(result.baselines["fold_through"], BASELINE_FLOOR)
        self.assertEqual(result.baselines["range"], BASELINE_FLOOR)
        self.assertAlmostEqual(result.baselines["equity_called"], 0.01, places=12)


class BiasOnlyOptimumInvariantTest(unittest.TestCase):
    """The baseline must be the MINIMUM of the trainer's masked loss over
    constant predictions. Computed here directly, row by row, from
    perturbed marginals — a perturbation that scored lower would mean
    the closed form is not the constant predictor's loss."""

    @staticmethod
    def _random_rows(seed: int) -> list[PhaseARowV9]:
        rng = random.Random(seed)
        rows = []
        for index in range(300):
            lane = rng.randrange(2)
            mask = (1, 0) if lane == 0 else (0, 1)
            rows.append(
                _row(
                    f"t{index}",
                    ft_label=float(rng.random() < (0.3 if lane == 0 else 0.8)),
                    ft_mask=mask if rng.random() < 0.7 else (0, 0),
                    bucket=min(7, int(rng.expovariate(0.6))),
                    range_mask=int(rng.random() < 0.8),
                    equity=rng.betavariate(2, 3),
                    equity_mask=int(rng.random() < 0.6),
                    slot=rng.randrange(3),
                )
            )
        return rows

    @staticmethod
    def _direct_losses(rows, lane_p, bucket_p, slot_mean) -> dict[str, float]:
        ft_sum = ft_n = 0.0
        range_sum = range_n = 0.0
        eq_sum = eq_n = 0.0
        for row in rows:
            for lane, flag in enumerate(row.fold_through_mask):
                if flag:
                    p = min(max(lane_p[lane], 1e-12), 1 - 1e-12)
                    y = row.fold_through_label
                    ft_sum -= y * math.log(p) + (1 - y) * math.log(1 - p)
                    ft_n += 1
            if row.range_mask:
                range_sum -= math.log(max(bucket_p[row.range_bucket], 1e-12))
                range_n += 1
            if row.equity_mask:
                eq_sum += (row.equity_called - slot_mean[row.equity_slot]) ** 2
                eq_n += 1
        return {
            "fold_through": ft_sum / ft_n,
            "range": range_sum / range_n,
            "equity_called": eq_sum / eq_n,
        }

    def test_marginals_minimize_the_direct_masked_loss(self) -> None:
        rows = self._random_rows(7)
        result = constant_predictor_baselines(rows)
        lane_n = [0.0, 0.0]
        lane_pos = [0.0, 0.0]
        bucket_n = [0.0] * 8
        slot_sum = [0.0] * 3
        slot_n = [0.0] * 3
        for row in rows:
            for lane, flag in enumerate(row.fold_through_mask):
                if flag:
                    lane_n[lane] += 1
                    lane_pos[lane] += row.fold_through_label
            if row.range_mask:
                bucket_n[row.range_bucket] += 1
            if row.equity_mask:
                slot_n[row.equity_slot] += 1
                slot_sum[row.equity_slot] += row.equity_called
        lane_p = [lane_pos[i] / lane_n[i] for i in range(2)]
        bucket_p = [n / sum(bucket_n) for n in bucket_n]
        slot_mean = [slot_sum[i] / slot_n[i] for i in range(3)]

        at_marginal = self._direct_losses(rows, lane_p, bucket_p, slot_mean)
        for name in SUPERVISED_HEADS_V9:
            self.assertAlmostEqual(at_marginal[name], result.measured[name], places=9)

        for delta in (0.05, -0.05):
            worse_ft = self._direct_losses(
                rows, [min(max(p + delta, 0.01), 0.99) for p in lane_p], bucket_p, slot_mean
            )
            self.assertGreater(worse_ft["fold_through"], result.measured["fold_through"])
            mixed = [0.8 * p + 0.2 / 8 for p in bucket_p]
            worse_range = self._direct_losses(rows, lane_p, mixed, slot_mean)
            self.assertGreater(worse_range["range"], result.measured["range"])
            worse_eq = self._direct_losses(
                rows, lane_p, bucket_p, [m + delta for m in slot_mean]
            )
            self.assertGreater(worse_eq["equity_called"], result.measured["equity_called"])


class ConfigAndScalesTest(unittest.TestCase):
    def test_default_is_raw_with_unit_weights(self) -> None:
        config = SupervisedLossConfigV9()
        self.assertEqual(config.normalization, "raw")
        self.assertEqual(config.head_weights(), {name: 1.0 for name in SUPERVISED_HEADS_V9})
        check_supervised_loss_config(config)

    def test_raw_scales_ignore_baselines(self) -> None:
        config = SupervisedLossConfigV9(range_weight=0.25)
        scales = supervised_head_scales(
            config, {"fold_through": 0.7, "range": 1.6, "equity_called": 0.05}
        )
        self.assertEqual(
            scales, {"fold_through": 1.0, "range": 0.25, "equity_called": 1.0}
        )

    def test_constant_predictor_scales_divide_by_baselines(self) -> None:
        config = SupervisedLossConfigV9(
            normalization="constant-predictor", range_weight=0.25
        )
        scales = supervised_head_scales(
            config, {"fold_through": 0.7, "range": 1.6, "equity_called": 0.05}
        )
        self.assertAlmostEqual(scales["fold_through"], 1 / 0.7)
        self.assertAlmostEqual(scales["range"], 0.25 / 1.6)
        self.assertAlmostEqual(scales["equity_called"], 20.0)

    def test_refusals(self) -> None:
        for bad in (
            SupervisedLossConfigV9(normalization="variance"),
            SupervisedLossConfigV9(range_weight=-0.1),
            SupervisedLossConfigV9(equity_called_weight=math.nan),
            SupervisedLossConfigV9(fold_through_weight=math.inf),
            SupervisedLossConfigV9(fold_through_weight=True),  # type: ignore[arg-type]
            SupervisedLossConfigV9(
                fold_through_weight=0.0, range_weight=0.0, equity_called_weight=0.0
            ),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    check_supervised_loss_config(bad)
        # A single positive head is fine (an ablation, not a mistake).
        check_supervised_loss_config(
            SupervisedLossConfigV9(fold_through_weight=0.0, range_weight=0.0)
        )


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class PhaseBObjectiveTorchTests(unittest.TestCase):
    """Tiny CPU fits on the Phase-B test fixtures."""

    def _fixtures(self, directory: Path):
        from test_v9_trainer import _synthetic_documents, _write_dataset
        from test_v9_trainer_phase_b import _synthetic_corpus, _write_corpus

        from training.v9_trainer import load_phase_a_dataset_v9
        from training.v9_trainer_phase_b import load_phase_b_corpus_v9

        corpus_path = directory / "tiny.phase-b-v9.jsonl.gz"
        _write_corpus(corpus_path, _synthetic_corpus())
        dataset_path = directory / "tiny-v9.jsonl.gz"
        _write_dataset(dataset_path, _synthetic_documents())
        return (
            load_phase_b_corpus_v9(corpus_path),
            load_phase_a_dataset_v9(dataset_path),
            corpus_path,
            dataset_path,
        )

    def test_raw_mode_is_the_shipped_objective_bit_for_bit(self) -> None:
        from test_v9_trainer_phase_b import _TINY_CONFIG

        from training.v9_trainer_phase_b import fit_phase_b_v9

        with tempfile.TemporaryDirectory() as raw:
            corpus, rows, _, _ = self._fixtures(Path(raw))
        implicit = fit_phase_b_v9(corpus, rows, _TINY_CONFIG)
        explicit = fit_phase_b_v9(corpus, rows, _TINY_CONFIG, SupervisedLossConfigV9())
        self.assertEqual(
            json.dumps(implicit["weights"], sort_keys=True),
            json.dumps(explicit["weights"], sort_keys=True),
        )
        losses = implicit["validation_losses"]
        self.assertEqual(losses["supervised_weighted"], losses["supervised_total"])
        self.assertEqual(
            losses["total"], losses["value_normalized"] + losses["supervised_total"]
        )
        self.assertEqual(implicit["supervised_normalization"], "raw")
        self.assertEqual(
            implicit["supervised_head_scales"],
            {name: 1.0 for name in SUPERVISED_HEADS_V9},
        )
        # The baselines are measured and reported even in raw mode.
        for name in SUPERVISED_HEADS_V9:
            record = implicit["supervised_baselines"][name]
            self.assertGreaterEqual(record["baseline"], BASELINE_FLOOR)
            self.assertAlmostEqual(
                losses[f"{name}_normalized"], losses[name] / record["baseline"]
            )

    def test_constant_predictor_total_is_the_weighted_normalized_sum(self) -> None:
        from test_v9_trainer_phase_b import _TINY_CONFIG

        from training.v9_trainer_phase_b import fit_phase_b_v9

        with tempfile.TemporaryDirectory() as raw:
            corpus, rows, _, _ = self._fixtures(Path(raw))
        config = SupervisedLossConfigV9(
            normalization="constant-predictor", range_weight=0.25
        )
        result = fit_phase_b_v9(corpus, rows, _TINY_CONFIG, config)
        baseline = fit_phase_b_v9(corpus, rows, _TINY_CONFIG)
        self.assertEqual(result["supervised_normalization"], "constant-predictor")
        # The knob reaches the gradient: a different objective, different weights.
        self.assertNotEqual(
            json.dumps(result["weights"], sort_keys=True),
            json.dumps(baseline["weights"], sort_keys=True),
        )
        for split in ("train_losses", "validation_losses"):
            losses = result[split]
            expected = sum(
                config.head_weights()[name]
                * losses[name]
                / result["supervised_baselines"][name]["baseline"]
                for name in SUPERVISED_HEADS_V9
            )
            self.assertAlmostEqual(losses["supervised_weighted"], expected, places=9)
            self.assertAlmostEqual(
                losses["total"], losses["value_normalized"] + expected, places=9
            )
        scales = result["supervised_head_scales"]
        self.assertAlmostEqual(
            scales["range"], 0.25 / result["supervised_baselines"]["range"]["baseline"]
        )

    def test_manifest_stamps_the_normalization(self) -> None:
        from test_v9_trainer_phase_b import _TINY_CONFIG

        from training.v8_trainer import V8TrainingConfig
        from training.v8_trainer_phase_b import PhaseBTrainingConfig
        from training.v9_trainer import validate_v9_manifest
        from training.v9_trainer_phase_b import train_phase_b_candidate_v9

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            corpus, rows, corpus_path, dataset_path = self._fixtures(directory)
            config = PhaseBTrainingConfig(
                base=V8TrainingConfig(
                    **{
                        **{
                            name: getattr(_TINY_CONFIG.base, name)
                            for name in _TINY_CONFIG.base.__dataclass_fields__
                        },
                        "model_version": "candidate-v9-rebalance-test",
                    }
                ),
                parity_sample=_TINY_CONFIG.parity_sample,
            )
            summary = train_phase_b_candidate_v9(
                corpus,
                rows,
                directory,
                config,
                init_seeds=(11,),
                corpus_path=corpus_path,
                dataset_path=dataset_path,
                supervised=SupervisedLossConfigV9(
                    normalization="constant-predictor", range_weight=0.0
                ),
            )
            manifest = json.loads(
                Path(summary["manifest_path"]).read_text(encoding="utf-8")
            )
        validate_v9_manifest(manifest)
        stamp = manifest["training"]["loss"]["supervised_normalization"]
        self.assertEqual(stamp["mode"], "constant-predictor")
        self.assertEqual(
            stamp["head_weights"],
            {"fold_through": 1.0, "range": 0.0, "equity_called": 1.0},
        )
        self.assertEqual(stamp["effective_scales"]["range"], 0.0)
        for name in SUPERVISED_HEADS_V9:
            self.assertIn("baseline", stamp["baselines"][name])
            self.assertIn("masked_rows", stamp["baselines"][name])
        validation = manifest["evaluation"]["validation_losses"]
        # RAW per-head numbers survive beside the normalized readouts.
        self.assertIn("equity_called", validation)
        self.assertIn("equity_called_normalized", validation)
        self.assertIn("supervised_weighted", validation)


if __name__ == "__main__":
    unittest.main()
