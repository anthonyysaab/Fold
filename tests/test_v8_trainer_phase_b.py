"""Stdlib-only tests for the Phase-B composed-value trainer.

The torch fitting path runs only in the CUDA venv, so these tests cover
everything importable on the stdlib interpreter: the fail-closed corpus
loader with its E6 cross-check, the split rule, the estimated value-loss
normalizer, and — the load-bearing one — parity between the trainer's
constants-based composition and the serve path's
``learned_policy_v8.compose_branch_values`` on the same head outputs.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from devfun_poker_playground import schema3
from devfun_poker_playground.learned_policy_v8 import (
    RESIDUAL_CAP_POT_FRACTION,
    compose_branch_values,
)
from devfun_poker_playground.v8_trainer import V8TrainingConfig
from devfun_poker_playground.v8_trainer_phase_b import (
    RESIDUAL_CAP_POT_FRACTION_DEFAULT,
    PhaseBTrainingConfig,
    check_phase_b_config,
    compose_from_constants,
    load_phase_b_decisions,
    split_decisions,
    value_target_variance,
)


def _header() -> dict:
    return {
        "kind": "phase-b-corpus",
        "corpus_schema_version": 1,
        "feature_schema_version": schema3.SCHEMA_VERSION,
        "input_size": schema3.INPUT_SIZE_V8,
        "branch_labels": ["fold", "check_call", "aggress_small", "aggress_large"],
    }


def _aggress_entry(branch: str, context: dict, reward: float) -> dict:
    pot = context["pot"]
    to_call = context["to_call"]
    eff = max(1, context["effective_stack"])
    fraction, stack_fraction = (
        (0.50, 0.20) if branch == "aggress_small" else (1.00, 0.45)
    )
    target = min(to_call + fraction * (pot + to_call), stack_fraction * eff)
    low, high = context["legal_range"]
    to_amount = min(high, max(low, context["contribution"] + target))
    return {
        "branch": branch,
        "family": "aggress",
        "pot_fraction": (target - to_call) / max(1, pot + to_call),
        "reward_bb": reward,
        "outcome_bb": reward,
        "risk_fraction": 0.01,
        "executed": ["raise", to_amount],
        "e6_target": target,
        "e6_to_amount": to_amount,
    }


def _row(
    table: str,
    ordinal: int = 0,
    *,
    with_aggress: bool = True,
    street: str = "flop",
) -> dict:
    context = {
        "pot": 300,
        "to_call": 100,
        "contribution": 0,
        "effective_stack": 5_000,
        "purse": 6_000,
        "legal_range": [200, 6_000] if with_aggress else None,
    }
    if with_aggress:
        rewards = {
            "fold": -1.0,
            "check_call": 0.25,
            "aggress_small": 0.25,
            "aggress_large": 0.5,
        }
        branches = [
            {
                "branch": "fold",
                "family": "fold",
                "pot_fraction": None,
                "reward_bb": rewards["fold"],
                "outcome_bb": 0.0,
                "risk_fraction": 0.0,
                "executed": ["fold", None],
            },
            {
                "branch": "check_call",
                "family": "check_call",
                "pot_fraction": None,
                "reward_bb": rewards["check_call"],
                "outcome_bb": 1.0,
                "risk_fraction": 0.0,
                "executed": ["call", 100],
            },
            _aggress_entry("aggress_small", context, rewards["aggress_small"]),
            _aggress_entry("aggress_large", context, rewards["aggress_large"]),
        ]
    else:
        branches = [
            {
                "branch": "fold",
                "family": "fold",
                "pot_fraction": None,
                "reward_bb": -0.75,
                "outcome_bb": 0.0,
                "risk_fraction": 0.0,
                "executed": ["fold", None],
            },
            {
                "branch": "check_call",
                "family": "check_call",
                "pot_fraction": None,
                "reward_bb": 0.75,
                "outcome_bb": 1.5,
                "risk_fraction": 0.0,
                "executed": ["call", 100],
            },
        ]
    return {
        "decision_id": f"{table}:hero:{ordinal}",
        "table_id": table,
        "harvest_leg": "test-leg",
        "policy_version": "test",
        "street": street,
        "big_blind": 100,
        "purse_bb": 60.0,
        "inclusion_count": 1,
        "rollouts": 2,
        "context": context,
        "features": [0.0] * schema3.INPUT_SIZE_V8,
        "branches": branches,
        "branch_absorption": {entry["branch"]: entry["branch"] for entry in branches},
        "p3": {
            "seats_resampled": 0,
            "tries": 0,
            "accepted": 0,
            "fallbacks": 0,
            "swaps_applied": 0,
        },
    }


def _write_corpus(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(_header()) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class LoaderTest(unittest.TestCase):
    def test_loads_and_derives_constants(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.phase-b.jsonl.gz"
            _write_corpus(
                path,
                [_row("sim-1-0"), _row("sim-1-1", with_aggress=False)],
            )
            decisions = load_phase_b_decisions(path)
        self.assertEqual(len(decisions), 2)
        first = decisions[0]
        self.assertEqual(
            first.emitted, ("fold", "check_call", "aggress_small", "aggress_large")
        )
        # Purse units: reward_bb / purse_bb.
        self.assertAlmostEqual(first.targets["aggress_large"], 0.5 / 60.0)
        # Derived sizing: small target = 100 + 0.5*400 = 300 -> wager 300.
        self.assertAlmostEqual(first.wager_unit[0], 300 / 6_000)
        # pot_if_called = 300 + 2*300 - 100 = 800.
        self.assertAlmostEqual(first.pot_if_called_unit[0], 800 / 6_000)
        second = decisions[1]
        self.assertEqual(second.emitted, ("fold", "check_call"))
        self.assertIsNone(second.wager_unit[0])

    def test_rejects_corrupted_e6_sizing(self) -> None:
        row = _row("sim-1-0")
        row["branches"][2]["e6_target"] += 1.0
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.phase-b.jsonl.gz"
            _write_corpus(path, [row])
            with self.assertRaisesRegex(ValueError, "E6 target"):
                load_phase_b_decisions(path)

    def test_rejects_uncentered_rewards(self) -> None:
        row = _row("sim-1-0")
        row["branches"][0]["reward_bb"] += 0.5
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.phase-b.jsonl.gz"
            _write_corpus(path, [row])
            with self.assertRaisesRegex(ValueError, "centered rewards"):
                load_phase_b_decisions(path)

    def test_rejects_duplicate_decision_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.phase-b.jsonl.gz"
            _write_corpus(path, [_row("sim-1-0"), _row("sim-1-0")])
            with self.assertRaisesRegex(ValueError, "duplicate decision_id"):
                load_phase_b_decisions(path)

    def test_rejects_purse_mismatch(self) -> None:
        row = _row("sim-1-0")
        row["purse_bb"] = 61.0  # context purse stays 6000 at bb 100
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.phase-b.jsonl.gz"
            _write_corpus(path, [row])
            with self.assertRaisesRegex(ValueError, "purse"):
                load_phase_b_decisions(path)

    def test_rejects_wrong_schema_header(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.phase-b.jsonl.gz"
            header = _header()
            header["input_size"] = 142
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(header) + "\n")
                handle.write(json.dumps(_row("sim-1-0")) + "\n")
            with self.assertRaisesRegex(ValueError, "input size"):
                load_phase_b_decisions(path)


class CompositionParityTest(unittest.TestCase):
    """The trainer's composition must be the serve path's composition."""

    def test_matches_compose_branch_values(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.phase-b.jsonl.gz"
            _write_corpus(
                path,
                [_row("sim-1-0"), _row("sim-1-1", with_aggress=False)],
            )
            decisions = load_phase_b_decisions(path)
        rng = random.Random(11)
        for decision in decisions:
            for _ in range(25):
                outputs = {
                    "fold_through": [rng.uniform(-4, 4) for _ in range(2)],
                    "range": [rng.uniform(-1, 1) for _ in range(8)],
                    "equity_called": [rng.uniform(-0.3, 1.3) for _ in range(3)],
                    "residual": [rng.uniform(-0.5, 0.5) for _ in range(4)],
                }
                mine = compose_from_constants(outputs, decision)
                serve, _ = compose_branch_values(
                    outputs,
                    pot=decision.context["pot"],
                    to_call=decision.context["to_call"],
                    contribution=decision.context["contribution"],
                    effective_stack=decision.context["effective_stack"],
                    purse=decision.context["purse"],
                    legal_range=decision.context["legal_range"],
                )
                for label in decision.emitted:
                    self.assertIn(label, serve)
                    self.assertLess(abs(mine[label] - serve[label]), 1e-12)

    def test_residual_cap_constant_matches_serve(self) -> None:
        self.assertEqual(
            RESIDUAL_CAP_POT_FRACTION, RESIDUAL_CAP_POT_FRACTION_DEFAULT
        )


class SplitAndVarianceTest(unittest.TestCase):
    def _decisions(self):
        rows = [_row(f"sim-{index}-0", 0) for index in range(40)]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.phase-b.jsonl.gz"
            _write_corpus(path, rows)
            return load_phase_b_decisions(path)

    def test_split_is_by_table_and_disjoint(self) -> None:
        decisions = self._decisions()
        config = V8TrainingConfig()
        train, validation = split_decisions(decisions, config)
        self.assertEqual(len(train) + len(validation), len(decisions))
        self.assertFalse(
            {d.table_id for d in train} & {d.table_id for d in validation}
        )
        # Same rule and seed as the Phase-A split: deterministic.
        train2, validation2 = split_decisions(decisions, config)
        self.assertEqual(
            [d.decision_id for d in train], [d.decision_id for d in train2]
        )

    def test_value_target_variance_is_positive_and_estimated(self) -> None:
        decisions = self._decisions()
        variance = value_target_variance(decisions)
        values = [t for d in decisions for t in d.targets.values()]
        mean = sum(values) / len(values)
        expected = sum((v - mean) ** 2 for v in values) / len(values)
        self.assertAlmostEqual(variance, max(expected, 1e-8))
        self.assertGreater(variance, 0.0)


class ConfigTest(unittest.TestCase):
    def test_rejects_negative_weights(self) -> None:
        base = V8TrainingConfig()
        with self.assertRaises(ValueError):
            check_phase_b_config(
                PhaseBTrainingConfig(base=base, value_loss_weight=-1.0)
            )
        with self.assertRaises(ValueError):
            check_phase_b_config(
                PhaseBTrainingConfig(base=base, residual_weight_decay=-0.1)
            )
        with self.assertRaises(ValueError):
            check_phase_b_config(
                PhaseBTrainingConfig(
                    base=base,
                    value_loss_weight=0.0,
                    supervised_loss_weight=0.0,
                )
            )

    def test_accepts_defaults(self) -> None:
        check_phase_b_config(PhaseBTrainingConfig(base=V8TrainingConfig()))


if __name__ == "__main__":
    unittest.main()
