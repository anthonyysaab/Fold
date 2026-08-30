"""Tests for the Phase-B v9 composed-value trainer.

Everything importable on the stdlib interpreter runs everywhere: the
fail-closed corpus loader with its frozen-g re-derivation, the header
version gate (version 1 — the stored v8 corpus — refused by number),
the shared wager-column helper, and the load-bearing stdlib parity —
the trainer's constants-based composition against the serve path's
``compose_branch_values_v9`` reconstructed from the recorded context.
The torch fitting path runs in the CUDA venv and is skipped elsewhere.

Fixture sizes are HAND-DERIVED from the g spec, not computed by calling
g (the standing rule: a test that mirrors the implementation passes its
bug). At ``read_temperature_x10 = 450``, T = 45 and b = 0 exactly, so
with every dial off the aggressive lane is f = 0.75, s = 0.325:
target = min(100 + 0.75·(300+100), 0.325·5000) = 400. At
``read_temperature_x10 = 100``, T = 10 and b = (45−10)/35 = 1 exactly,
so the active bet is f = 0.5 + 0.195 = 0.695: wager = 0.695·500 =
347.5.
"""

from __future__ import annotations

import gzip
import json
import random
import tempfile
import unittest
from pathlib import Path

from engine import schema4
from engine.aggression_sizing import context_int_to_temperature
from engine.learned_policy_v8 import RESIDUAL_CAP_POT_FRACTION
from engine.learned_policy_v9 import compose_branch_values_v9
from engine.rules.composition import composed_sizing_record
from engine.v8_trainer import V8TrainingConfig
from engine.v8_trainer_phase_b import (
    RESIDUAL_CAP_POT_FRACTION_DEFAULT,
    PhaseBTrainingConfig,
)
from engine.v9_trainer_phase_b import (
    compose_from_constants_v9,
    fit_phase_b_v9,
    load_phase_b_corpus_v9,
    train_phase_b_candidate_v9,
    wager_column_slice,
)

try:
    import torch
except ImportError:
    torch = None


def _header(**overrides) -> dict:
    header = {
        "kind": "phase-b-corpus",
        "corpus_schema_version": 2,
        "feature_schema_version": schema4.SCHEMA_VERSION_V9,
        "input_size": schema4.INPUT_SIZE_V9,
        "branch_labels": ["fatal", "passive", "active", "aggressive"],
        "sizing": composed_sizing_record(),
        "belief_fit_source": "artifacts/p3/p3-fit-test.json",
        "equity_trials": 200,
        "starting_stack": 6_000,
        "big_blind": 100,
        "seeds": [11, 12],
    }
    header.update(overrides)
    return header


def _priced_row(table: str, ordinal: int = 0, *, with_aggressive: bool = True) -> dict:
    """A priced decision at b = 0 (read 450): aggressive target 400."""

    context = {
        "pot": 300,
        "to_call": 100,
        "contribution": 0,
        "effective_stack": 5_000,
        "purse": 6_000,
        "read_temperature_x10": 450,
        "street": "flop",
        "bankroll": 6_000,
        "exposure": 800,
        "covered_allin_to_amounts": [],
        "legal_labels": (
            ["fatal", "active", "aggressive"]
            if with_aggressive
            else ["fatal", "active"]
        ),
        "bet_range": None,
        "raise_range": [200, 6_000] if with_aggressive else None,
    }
    branches = [
        {"branch": "fatal", "reward_bb": -1.0 if with_aggressive else -0.75},
        {"branch": "active", "reward_bb": 0.25 if with_aggressive else 0.75},
    ]
    if with_aggressive:
        branches.append(
            {
                "branch": "aggressive",
                "reward_bb": 0.75,
                # Hand-derived at b = 0, dials off (module docstring).
                "sizing_target": 400.0,
                "sizing_to_amount": 400.0,
            }
        )
    return {
        "decision_id": f"{table}:hero:{ordinal}",
        "table_id": table,
        "street": "flop",
        "big_blind": 100,
        "purse_bb": 60.0,
        "context": context,
        "features": [0.0] * schema4.INPUT_SIZE_V9,
        "branches": branches,
    }


def _free_row(table: str, ordinal: int = 1) -> dict:
    """A free-spot decision at b = 1 (read 100): active bet 347.5."""

    context = {
        "pot": 500,
        "to_call": 0,
        "contribution": 0,
        "effective_stack": 5_000,
        "purse": 6_000,
        "read_temperature_x10": 100,
        "street": "turn",
        "bankroll": 6_000,
        "exposure": 800,
        "covered_allin_to_amounts": [],
        "legal_labels": ["passive", "active"],
        "bet_range": [100, 6_000],
        "raise_range": None,
    }
    return {
        "decision_id": f"{table}:hero:{ordinal}",
        "table_id": table,
        "street": "turn",
        "big_blind": 100,
        "purse_bb": 60.0,
        "context": context,
        "features": [0.0] * schema4.INPUT_SIZE_V9,
        "branches": [
            {"branch": "passive", "reward_bb": -0.5},
            {
                "branch": "active",
                "reward_bb": 0.5,
                # Hand-derived at b = 1, dials off (module docstring).
                "sizing_target": 347.5,
                "sizing_to_amount": 347.5,
            },
        ],
    }


def _write_corpus(path: Path, rows: list[dict], header: dict | None = None) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(header if header is not None else _header()) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _load(rows: list[dict], header: dict | None = None):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.phase-b-v9.jsonl.gz"
        _write_corpus(path, rows, header)
        return load_phase_b_corpus_v9(path)


class HeaderGateTest(unittest.TestCase):
    def test_loads_a_well_formed_corpus_and_header_fields(self) -> None:
        corpus = _load([_priced_row("sim-1-0"), _free_row("sim-1-0")])
        self.assertEqual(len(corpus.decisions), 2)
        self.assertEqual(corpus.equity_trials, 200)
        self.assertEqual(corpus.starting_stack, 6_000)
        self.assertEqual(corpus.big_blind, 100)
        self.assertEqual(corpus.seeds, (11, 12))
        self.assertEqual(
            corpus.belief_fit_source, "artifacts/p3/p3-fit-test.json"
        )

    def test_refuses_the_v8_corpus_by_version_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.phase-b-v9.jsonl.gz"
            _write_corpus(
                path,
                [_priced_row("sim-1-0")],
                _header(corpus_schema_version=1),
            )
            with self.assertRaisesRegex(ValueError, "v8 corpus"):
                load_phase_b_corpus_v9(path)

    def test_header_refusals(self) -> None:
        cases = {
            "future version": ({"corpus_schema_version": 3}, "unsupported"),
            "v8 labels": (
                {
                    "branch_labels": [
                        "fold",
                        "check_call",
                        "aggress_small",
                        "aggress_large",
                    ]
                },
                "not the v9",
            ),
            "wrong schema": ({"feature_schema_version": 3}, "schema 4"),
            "wrong input size": ({"input_size": 412}, "input size"),
            "missing sizing": ({"sizing": None}, "sizing"),
            "foreign sizing identity": (
                {"sizing": {**composed_sizing_record(), "identity": "g-x"}},
                "sizing record is invalid",
            ),
            "missing belief fit": (
                {"belief_fit_source": None},
                "belief_fit_source",
            ),
            "empty belief fit": ({"belief_fit_source": ""}, "belief_fit_source"),
            "missing equity trials": ({"equity_trials": None}, "equity_trials"),
            "zero equity trials": ({"equity_trials": 0}, "equity_trials"),
            "missing seeds": ({"seeds": None}, "seeds"),
            "empty seeds": ({"seeds": []}, "seeds"),
            "missing starting stack": (
                {"starting_stack": None},
                "starting_stack",
            ),
        }
        for reason, (overrides, pattern) in cases.items():
            with self.subTest(reason):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "c.phase-b-v9.jsonl.gz"
                    _write_corpus(path, [_priced_row("sim-1-0")], _header(**overrides))
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_phase_b_corpus_v9(path)


class LoaderDerivationTest(unittest.TestCase):
    def test_derives_the_hand_computed_constants(self) -> None:
        corpus = _load(
            [
                _priced_row("sim-1-0"),
                _free_row("sim-1-0"),
                _priced_row("sim-1-1", with_aggressive=False),
            ]
        )
        priced, free, call_only = corpus.decisions
        # WAGER_LANES order is (active, aggressive).
        self.assertEqual(priced.emitted, ("fatal", "active", "aggressive"))
        self.assertFalse(priced.to_call_zero)
        self.assertIsNone(priced.wager_unit[0])  # a call has no size
        self.assertAlmostEqual(priced.wager_unit[1], 400 / 6_000)
        # pot_if_called = 300 + 2*400 - 100 = 1000.
        self.assertAlmostEqual(priced.pot_if_called_unit[1], 1_000 / 6_000)
        self.assertAlmostEqual(priced.targets["aggressive"], 0.75 / 60.0)
        self.assertAlmostEqual(priced.cc_pot_unit, 400 / 6_000)
        self.assertAlmostEqual(priced.cc_cost_unit, 100 / 6_000)

        self.assertEqual(free.emitted, ("passive", "active"))
        self.assertTrue(free.to_call_zero)
        self.assertAlmostEqual(free.wager_unit[0], 347.5 / 6_000)
        # pot_if_called = 500 + 2*347.5 - 0 = 1195.
        self.assertAlmostEqual(free.pot_if_called_unit[0], 1_195 / 6_000)
        self.assertIsNone(free.wager_unit[1])  # aggressive not emitted

        self.assertEqual(call_only.emitted, ("fatal", "active"))
        self.assertEqual(call_only.wager_unit, (None, None))

    def test_row_refusals(self) -> None:
        def corrupted_target() -> dict:
            row = _priced_row("sim-1-0")
            row["branches"][2]["sizing_target"] = 401.0
            return row

        def corrupted_to_amount() -> dict:
            row = _priced_row("sim-1-0")
            row["branches"][2]["sizing_to_amount"] = 399.0
            return row

        def v8_sizing_keys() -> dict:
            row = _priced_row("sim-1-0")
            entry = row["branches"][2]
            entry["e6_target"] = entry.pop("sizing_target")
            entry["e6_to_amount"] = entry.pop("sizing_to_amount")
            return row

        def missing_sizing() -> dict:
            row = _priced_row("sim-1-0")
            del row["branches"][2]["sizing_target"]
            return row

        def sizing_on_a_call() -> dict:
            row = _priced_row("sim-1-0")
            row["branches"][1]["sizing_target"] = 100.0
            row["branches"][1]["sizing_to_amount"] = 100.0
            return row

        def out_of_slot_order() -> dict:
            row = _priced_row("sim-1-0")
            row["context"]["legal_labels"] = ["active", "fatal", "aggressive"]
            row["branches"] = [
                row["branches"][1],
                row["branches"][0],
                row["branches"][2],
            ]
            return row

        def fatal_at_a_free_check() -> dict:
            row = _free_row("sim-1-0")
            row["context"]["legal_labels"] = ["fatal", "passive", "active"]
            return row

        def emitted_not_legal() -> dict:
            row = _priced_row("sim-1-0")
            row["branches"] = row["branches"][1:]  # drop the fatal entry
            return row

        def uncentered() -> dict:
            row = _priced_row("sim-1-0")
            row["branches"][0]["reward_bb"] += 0.5
            return row

        def purse_mismatch() -> dict:
            row = _priced_row("sim-1-0")
            row["purse_bb"] = 61.0
            return row

        def foreign_big_blind() -> dict:
            row = _priced_row("sim-1-0")
            row["big_blind"] = 50
            row["purse_bb"] = 120.0
            return row

        def float_context() -> dict:
            row = _priced_row("sim-1-0")
            row["context"]["pot"] = 300.0
            return row

        def read_out_of_range() -> dict:
            row = _priced_row("sim-1-0")
            row["context"]["read_temperature_x10"] = 1_001
            return row

        def street_mismatch() -> dict:
            row = _priced_row("sim-1-0")
            row["context"]["street"] = "turn"
            return row

        cases = {
            "corrupted sizing target": (corrupted_target, "sizing target"),
            "corrupted to-amount": (corrupted_to_amount, "to-amount"),
            "v8 sizing keys": (v8_sizing_keys, "e6_"),
            "missing sizing fields": (missing_sizing, "sizing_target"),
            "sizing on a call": (sizing_on_a_call, "must not carry sizing"),
            "labels out of slot order": (out_of_slot_order, "slot order"),
            "fatal at a free check": (fatal_at_a_free_check, "dominated"),
            "emitted != legal": (emitted_not_legal, "legal_labels"),
            "uncentered rewards": (uncentered, "centered rewards"),
            "purse mismatch": (purse_mismatch, "purse"),
            "foreign big blind": (foreign_big_blind, "header"),
            "float context int": (float_context, "raw integer"),
            "read out of range": (read_out_of_range, r"\[0, 1000\]"),
            "context street mismatch": (street_mismatch, "street"),
        }
        for reason, (build, pattern) in cases.items():
            with self.subTest(reason):
                with self.assertRaisesRegex(ValueError, pattern):
                    _load([build()])

    def test_rejects_duplicate_decision_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate decision_id"):
            _load([_priced_row("sim-1-0"), _priced_row("sim-1-0")])


class CompositionParityTest(unittest.TestCase):
    """The trainer's constants must reproduce the serve composition.

    ``compose_from_constants_v9`` runs over the LOADER-DERIVED wager
    constants; ``compose_branch_values_v9`` re-runs the composed sizing
    live from the recorded context. Agreement proves the loader's
    derivation and the serve path's sizing are one arithmetic — the
    impossible-by-construction invariant this contract was designed
    around.
    """

    def test_matches_compose_branch_values_v9(self) -> None:
        corpus = _load(
            [
                _priced_row("sim-1-0"),
                _free_row("sim-1-0"),
                _priced_row("sim-1-1", with_aggressive=False),
            ]
        )
        rng = random.Random(11)
        for decision in corpus.decisions:
            for _ in range(25):
                outputs = {
                    "fold_through": [rng.uniform(-4, 4) for _ in range(2)],
                    "range": [rng.uniform(-1, 1) for _ in range(8)],
                    "equity_called": [rng.uniform(-0.3, 1.3) for _ in range(3)],
                    "residual": [rng.uniform(-0.5, 0.5) for _ in range(4)],
                }
                mine = compose_from_constants_v9(outputs, decision)
                context = decision.context
                boldness = corpus.sizing.boldness(
                    context_int_to_temperature(context["read_temperature_x10"])
                )
                serve, _ = compose_branch_values_v9(
                    outputs,
                    pot=context["pot"],
                    to_call=context["to_call"],
                    contribution=context["contribution"],
                    effective_stack=context["effective_stack"],
                    purse=context["purse"],
                    boldness=boldness,
                    street=context["street"],
                    bankroll=context["bankroll"],
                    exposure=context["exposure"],
                    covered_allin_to_amounts=tuple(
                        context["covered_allin_to_amounts"]
                    ),
                    legal_labels=frozenset(decision.emitted),
                    bet_range=context["bet_range"],
                    raise_range=context["raise_range"],
                    sizing=corpus.sizing,
                    rules=corpus.rules,
                )
                self.assertEqual(set(mine), set(serve))
                self.assertEqual(set(mine), set(decision.emitted))
                for label in decision.emitted:
                    self.assertLess(abs(mine[label] - serve[label]), 1e-12)

    def test_residual_cap_constant_matches_serve(self) -> None:
        self.assertEqual(
            RESIDUAL_CAP_POT_FRACTION, RESIDUAL_CAP_POT_FRACTION_DEFAULT
        )

    def test_wager_column_slice_is_the_ledger_slice(self) -> None:
        # The version ledger pins the wager-column slice as [:, 2:4].
        self.assertEqual(wager_column_slice(), slice(2, 4))


class ServeEquityTrialsPinTest(unittest.TestCase):
    """load_policy_v9 honours the manifest's serve.equity_trials pin —
    the mechanism that makes the corpus contract's 'harvest == serve,
    one number' true at serve time."""

    def test_pin_precedence(self) -> None:
        from test_learned_policy_v9 import _write_artifact

        from engine.learned_policy_v9 import load_policy_v9

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            pinned = _write_artifact(
                directory,
                mutate=lambda m: m.update(
                    serve={"equity_trials": 512}
                ),
            )
            self.assertEqual(load_policy_v9(pinned).equity_trials, 512)
            # An explicit caller argument wins over the pin.
            self.assertEqual(
                load_policy_v9(pinned, equity_trials=64).equity_trials, 64
            )
        with tempfile.TemporaryDirectory() as raw:
            unpinned = _write_artifact(Path(raw))
            self.assertEqual(load_policy_v9(unpinned).equity_trials, 200)

    def test_invalid_pin_is_refused(self) -> None:
        from test_learned_policy_v9 import _write_artifact

        from engine.learned_policy_v9 import (
            LearnedPolicyV9Error,
            load_policy_v9,
        )

        with tempfile.TemporaryDirectory() as raw:
            path = _write_artifact(
                Path(raw),
                mutate=lambda m: m.update(serve={"equity_trials": 0}),
            )
            with self.assertRaises(LearnedPolicyV9Error):
                load_policy_v9(path)


def _synthetic_corpus(tables: int = 40):
    rows = []
    for index in range(tables):
        table = f"sim-{index}-0"
        rows.append(_priced_row(table, 0))
        rows.append(_free_row(table, 1))
    return rows


_TINY_CONFIG = PhaseBTrainingConfig(
    base=V8TrainingConfig(
        epochs=3,
        learning_rate=1e-3,
        warmup_steps=4,
        early_stop_patience=5,
        batch_size=32,
        validation_fraction=0.25,
        split_seed=17,
        init_seed=11,
        device="cpu",
    ),
    parity_sample=16,
)


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class V9PhaseBTorchTests(unittest.TestCase):
    """Tiny CPU fits; skipped wherever torch is not installed."""

    def _fixtures(self, directory: Path):
        from test_v9_trainer import _synthetic_documents, _write_dataset

        from engine.v9_trainer import load_phase_a_dataset_v9

        corpus_path = directory / "tiny.phase-b-v9.jsonl.gz"
        _write_corpus(corpus_path, _synthetic_corpus())
        corpus = load_phase_b_corpus_v9(corpus_path)
        dataset_path = directory / "tiny-v9.jsonl.gz"
        _write_dataset(dataset_path, _synthetic_documents())
        rows = load_phase_a_dataset_v9(dataset_path)
        return corpus, rows, corpus_path, dataset_path

    def test_trains_exports_and_passes_parity(self) -> None:
        from engine.v9_trainer import validate_v9_manifest

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
                        "model_version": "candidate-v9-phase-b-test",
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
            )
            manifest = json.loads(
                Path(summary["manifest_path"]).read_text(encoding="utf-8")
            )

        validate_v9_manifest(manifest)
        self.assertEqual(
            manifest["action_labels"],
            ["fatal", "passive", "active", "aggressive"],
        )
        # Harvest == serve: the manifest ships the corpus header's own
        # record and pins the header's equity_trials for the loader.
        self.assertEqual(
            manifest["sizing"], json.loads(json.dumps(composed_sizing_record()))
        )
        self.assertEqual(manifest["serve"]["equity_trials"], 200)
        window = manifest["training_window"]
        self.assertEqual(
            window["belief_fit_source"], "artifacts/p3/p3-fit-test.json"
        )
        self.assertEqual(window["equity_trials"], 200)
        self.assertEqual(
            window["instrument"],
            {"starting_stack": 6_000, "big_blind": 100, "seeds": [11, 12]},
        )
        self.assertEqual(window["phase_b_free_spot_decisions"], 40)
        parity = manifest["evaluation"]["parity_check"]
        self.assertGreater(parity["decisions_checked"], 0)
        self.assertLessEqual(parity["max_abs_value_diff"], parity["tolerance"])
        # The composed objective reached the residual head (its own
        # decay group), while Phase A alone would have left it at zero.
        share = manifest["evaluation"]["residual_share"]
        self.assertGreater(share["wager_executions"], 0)

    def test_same_seed_is_deterministic_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            corpus, rows, _, _ = self._fixtures(Path(raw))
        first = fit_phase_b_v9(corpus, rows, _TINY_CONFIG)
        second = fit_phase_b_v9(corpus, rows, _TINY_CONFIG)
        self.assertEqual(
            json.dumps(first["weights"], sort_keys=True),
            json.dumps(second["weights"], sort_keys=True),
        )
        self.assertEqual(first["validation_losses"], second["validation_losses"])


if __name__ == "__main__":
    unittest.main()
