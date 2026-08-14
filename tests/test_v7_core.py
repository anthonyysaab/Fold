from __future__ import annotations

import json
import math
import random
import unittest

from devfun_poker_playground.learning_contract import (
    BRANCH_FAMILIES,
    BRANCH_LABELS,
    BRANCH_POT_FRACTIONS,
    card_feature_indices,
    context_feature_indices,
    default_v7_architecture,
    LEARNING_INPUT_SIZE,
    LearningContractError,
    validate_v7_architecture,
)
from devfun_poker_playground.offline_trainer import (
    _baseline_floor_v7,
    _branch_metrics_v7,
    _forward_v2,
    _layer_norm,
    _margin_quantiles_v7,
    _round9,
)
from devfun_poker_playground.training_telemetry import TrainingExample


def _random_v7_weights(architecture: dict, seed: int = 5) -> dict:
    rng = random.Random(seed)

    def matrix(rows: int, cols: int) -> list[list[float]]:
        return [[rng.uniform(-0.2, 0.2) for _ in range(cols)] for _ in range(rows)]

    def vector(size: int) -> list[float]:
        return [rng.uniform(-0.1, 0.1) for _ in range(size)]

    encoder = architecture["encoder_width"]
    trunk_widths = architecture["trunk_widths"]

    def with_norm(rows: int, cols: int) -> dict:
        return {
            "w": matrix(rows, cols),
            "b": vector(rows),
            "ln_g": [1.0] * rows,
            "ln_b": [0.0] * rows,
        }

    trunk = []
    previous = 2 * encoder
    for index, width in enumerate(trunk_widths):
        block = {"w": matrix(width, previous), "b": vector(width)}
        if index < len(trunk_widths) - 1:
            block["ln_g"] = [1.0] * width
            block["ln_b"] = [0.0] * width
        trunk.append(block)
        previous = width
    heads = {}
    for name, size in architecture["heads"].items():
        tower = architecture["head_towers"][name]
        heads[name] = {
            "tower_w": matrix(tower, trunk_widths[-1]),
            "tower_b": vector(tower),
            "out_w": matrix(size, tower),
            "out_b": vector(size),
        }
    return {
        "card_encoder": with_norm(encoder, len(architecture["card_indices"])),
        "context_encoder": with_norm(encoder, len(architecture["context_indices"])),
        "trunk": trunk,
        "heads": heads,
    }


def _example(
    decision: str,
    branch: str,
    reward_bb: float,
    *,
    behavior=(0.0, 1.0, 0.0),
) -> TrainingExample:
    family = BRANCH_FAMILIES[BRANCH_LABELS.index(branch)]
    labels = ("fold", "check_call", "aggress")
    example = TrainingExample(
        table_id=f"table-{decision}",
        policy_version="test",
        features=tuple([0.0] * LEARNING_INPUT_SIZE),
        action_family_index=labels.index(family),
        behavior_probabilities=behavior,
        submitted_risk_fraction=0.1,
        purse_bb=100.0,
        reward_bb=reward_bb,
        counterfactual=True,
        opponent_confidence=1.0,
        decision_id=decision,
    )
    try:
        from dataclasses import replace

        return replace(example, action_branch=branch)
    except TypeError:
        return example


class ContractV7Tests(unittest.TestCase):
    def test_card_and_context_indices_partition_the_vector(self) -> None:
        cards = card_feature_indices()
        context = context_feature_indices()

        self.assertEqual(len(cards), 104)
        self.assertEqual(len(context), 38)
        self.assertEqual(sorted([*cards, *context]), list(range(LEARNING_INPUT_SIZE)))

    def test_board_tier_and_legality_stay_in_the_context_block(self) -> None:
        from devfun_poker_playground.learning_contract import (
            LEARNING_FEATURE_NAMES,
        )

        context_names = {
            LEARNING_FEATURE_NAMES[index] for index in context_feature_indices()
        }
        for name in (
            "board_tier_fresh",
            "hole_known_fraction",
            "legal_fold",
            "spr",
            "opponent_max_wildness",
        ):
            self.assertIn(name, context_names)

    def test_default_architecture_validates(self) -> None:
        validate_v7_architecture(default_v7_architecture())

    def test_partition_gaps_are_rejected(self) -> None:
        architecture = default_v7_architecture()
        architecture["card_indices"] = architecture["card_indices"][1:]

        with self.assertRaises(LearningContractError):
            validate_v7_architecture(architecture)

    def test_branch_tables_agree(self) -> None:
        self.assertEqual(len(BRANCH_LABELS), len(BRANCH_FAMILIES))
        self.assertEqual(len(BRANCH_LABELS), len(BRANCH_POT_FRACTIONS))
        self.assertEqual(BRANCH_POT_FRACTIONS[2], 0.5)
        self.assertEqual(BRANCH_POT_FRACTIONS[3], 1.0)


class ForwardV2Tests(unittest.TestCase):
    def test_layer_norm_matches_manual_computation(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        gamma = [2.0, 2.0, 2.0, 2.0]
        beta = [0.5, 0.5, 0.5, 0.5]

        result = _layer_norm(values, gamma, beta)

        mean = 2.5
        variance = sum((v - mean) ** 2 for v in values) / 4
        expected = [(v - mean) / math.sqrt(variance + 1e-5) * 2.0 + 0.5 for v in values]
        for got, want in zip(result, expected):
            self.assertAlmostEqual(got, want, places=12)

    def test_forward_v2_produces_one_value_per_branch(self) -> None:
        architecture = default_v7_architecture()
        weights = _random_v7_weights(architecture)
        features = [random.Random(3).uniform(-1, 1) for _ in range(LEARNING_INPUT_SIZE)]

        outputs = _forward_v2(architecture, weights, features)

        self.assertEqual(set(outputs), {"action_value"})
        self.assertEqual(len(outputs["action_value"]), len(BRANCH_LABELS))
        self.assertTrue(all(math.isfinite(value) for value in outputs["action_value"]))

    def test_forward_v2_is_deterministic(self) -> None:
        architecture = default_v7_architecture()
        weights = _random_v7_weights(architecture)
        features = [0.1] * LEARNING_INPUT_SIZE

        first = _forward_v2(architecture, weights, features)["action_value"]
        second = _forward_v2(architecture, weights, features)["action_value"]

        self.assertEqual(first, second)

    def test_diagnostic_heads_are_computed_only_on_request(self) -> None:
        architecture = default_v7_architecture()
        weights = _random_v7_weights(architecture)
        features = [0.0] * LEARNING_INPUT_SIZE

        outputs = _forward_v2(
            architecture, weights, features, heads=("action_value", "behavior_prior")
        )

        self.assertEqual(set(outputs), {"action_value", "behavior_prior"})
        self.assertEqual(len(outputs["behavior_prior"]), 3)


class Round9Tests(unittest.TestCase):
    def test_round9_shortens_but_preserves_float32_values(self) -> None:
        import struct

        original = 0.123456789012345678
        rounded = _round9(original)
        as_float32 = struct.unpack("f", struct.pack("f", original))[0]
        rounded32 = struct.unpack("f", struct.pack("f", rounded))[0]

        self.assertEqual(as_float32, rounded32)
        self.assertLessEqual(len(repr(rounded)), len(repr(original)))

    def test_round9_walks_nested_structures(self) -> None:
        document = {"a": [1.00000000001, {"b": 2.5}], "c": "text"}

        rounded = _round9(document)

        self.assertEqual(rounded["c"], "text")
        self.assertEqual(rounded["a"][1]["b"], 2.5)
        self.assertEqual(
            json.dumps(rounded), json.dumps(json.loads(json.dumps(rounded)))
        )


class BranchMetricsTests(unittest.TestCase):
    def test_metrics_score_branches_not_families(self) -> None:
        examples = [
            _example("d1", "fold", 0.0),
            _example("d1", "check_call", -1.0),
            _example("d1", "aggress_half_pot", 3.0),
            _example("d1", "aggress_pot", -2.0),
        ]
        # Model ranks the half-pot branch highest: zero regret.
        scores = [[0.0, -0.1, 0.9, -0.5]] * 4

        accuracy, regret, _ = _branch_metrics_v7(examples, scores)

        self.assertEqual(accuracy, 1.0)
        self.assertEqual(regret, 0.0)

    def test_regret_measures_the_gap_to_the_best_branch(self) -> None:
        examples = [
            _example("d1", "fold", 0.0),
            _example("d1", "aggress_pot", 5.0),
        ]
        scores = [[0.9, 0.0, 0.0, 0.1]] * 2

        accuracy, regret, _ = _branch_metrics_v7(examples, scores)

        self.assertEqual(accuracy, 0.0)
        self.assertAlmostEqual(regret, 100.0 * (5.0 / 100.0), places=9)

    def test_baseline_floor_reports_every_trivial_policy(self) -> None:
        examples = [
            _example("d1", "fold", 0.0),
            _example("d1", "aggress_pot", 5.0),
            _example("d2", "fold", 0.0),
            _example("d2", "check_call", 2.0),
        ]

        table = _baseline_floor_v7(examples, seed=1)

        self.assertEqual(
            set(table),
            {
                "always_fold",
                "always_check_call",
                "always_aggress_half_pot",
                "always_aggress_pot",
                "uniform_random_legal",
            },
        )
        self.assertEqual(table["always_fold"]["best_action_accuracy"], 0.0)

    def test_margin_quantiles_are_monotone(self) -> None:
        examples = []
        scores = []
        rng = random.Random(9)
        for index in range(50):
            decision = f"d{index}"
            examples += [
                _example(decision, "fold", 0.0),
                _example(decision, "check_call", rng.uniform(-2, 2)),
            ]
            row = [rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0, 0.0]
            scores += [row, row]

        quantiles = _margin_quantiles_v7(examples, scores)

        self.assertLessEqual(quantiles["p50"], quantiles["p90"])
        self.assertLessEqual(quantiles["p90"], quantiles["p99"])


if __name__ == "__main__":
    unittest.main()
