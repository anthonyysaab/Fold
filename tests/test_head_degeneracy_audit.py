"""Checks for the head-degeneracy detector.

The detector this replaced was circular: it tested "are all tower
pre-activations <= 0", which stops being measurable the moment a
LayerNorm sits in front of the ReLU -- the very fix it was meant to
verify -- and it read 0.00% on a control that must read 100%. It would
have passed a repair that did nothing.

These tests pin the controls, because the controls are the only reason
to believe the number.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tools.head_degeneracy_audit import (
    DEFAULT_MANIFEST,
    _forced_live_tower,
    _zeroed_out_w,
    constancy,
    default_head,
    head_outputs,
    load_artifact,
    load_rows,
    stage_instrument,
)

HEAD = "action_value"


def _fixture(limit: int = 60):
    architecture, weights, means, stds = load_artifact(Path(DEFAULT_MANIFEST))
    rows = load_rows(Path(".arena-training.jsonl"), len(means))[:limit]
    return architecture, weights, rows, means, stds


class ConstancyPredicateTests(unittest.TestCase):
    def test_a_bias_only_head_reads_fully_constant(self) -> None:
        """Impossible by construction: zero out_w can only emit out_b."""

        architecture, weights, rows, means, stds = _fixture()
        out_b = weights["heads"][HEAD]["out_b"]
        outputs = head_outputs(
            architecture, _zeroed_out_w(weights, HEAD), rows, means, stds, HEAD
        )
        self.assertEqual(constancy(outputs, out_b)["constant_pct"], 100.0)

    def test_a_forced_live_tower_reads_fully_variable(self) -> None:
        """Every tower unit active on every row cannot produce the bias."""

        architecture, weights, rows, means, stds = _fixture()
        out_b = weights["heads"][HEAD]["out_b"]
        outputs = head_outputs(
            architecture, _forced_live_tower(weights, HEAD), rows, means, stds, HEAD
        )
        self.assertEqual(constancy(outputs, out_b)["constant_pct"], 0.0)

    def test_the_controls_do_not_mutate_the_artifact(self) -> None:
        """A control that edited the real weights would poison the result."""

        _, weights, _, _, _ = _fixture(limit=1)
        before = [list(row) for row in weights["heads"][HEAD]["out_w"]]
        _zeroed_out_w(weights, HEAD)
        _forced_live_tower(weights, HEAD)
        self.assertEqual(weights["heads"][HEAD]["out_w"], before)

    def test_constancy_is_bit_identical_not_approximate(self) -> None:
        """A head that lands NEAR its bias is still discriminating."""

        out_b = [0.1, 0.2]
        nearly = [[0.1, 0.2 + 1e-12]]
        self.assertEqual(constancy(nearly, out_b)["at_bias"], 0)
        self.assertEqual(constancy([[0.1, 0.2]], out_b)["at_bias"], 1)


class InstrumentGateTests(unittest.TestCase):
    def test_the_live_artifact_passes_every_control(self) -> None:
        architecture, weights, rows, means, stds = _fixture(limit=40)
        report = stage_instrument(architecture, weights, rows, means, stds, HEAD)
        self.assertTrue(report["all_passed"])

    def test_a_broken_detector_would_be_caught(self) -> None:
        """The degenerate control must fail if constancy is mis-defined.

        Comparing against the wrong bias is exactly the class of error
        the controls exist to catch.
        """

        architecture, weights, rows, means, stds = _fixture(limit=40)
        wrong_bias = [value + 1.0 for value in weights["heads"][HEAD]["out_b"]]
        outputs = head_outputs(
            architecture, _zeroed_out_w(weights, HEAD), rows, means, stds, HEAD
        )
        self.assertEqual(constancy(outputs, wrong_bias)["constant_pct"], 0.0)


class DefaultHeadTests(unittest.TestCase):
    def test_action_value_wins_when_present(self) -> None:
        _, weights, _, _, _ = _fixture(limit=1)
        self.assertEqual(default_head(weights), "action_value")

    def test_foreign_architectures_fall_back_to_the_widest_head(self) -> None:
        # No action_value head (the v9 shape): the widest head is the
        # closest analogue of a value head.
        weights = {
            "heads": {
                "fold_through": {"out_b": [0.0, 0.0]},
                "range": {"out_b": [0.0] * 8},
                "equity_called": {"out_b": [0.0, 0.0, 0.0]},
                "residual": {"out_b": [0.0, 0.0, 0.0, 0.0]},
            }
        }
        self.assertEqual(default_head(weights), "range")


if __name__ == "__main__":
    unittest.main()
