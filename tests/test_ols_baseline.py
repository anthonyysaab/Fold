"""Checks for the OLS promotion gate.

``tools/ols_baseline.py`` decides whether a format-4 candidate may reach
``artifacts/approved.json``, so its verdict function needs the same
scrutiny as anything else on the promotion path. The cases that matter
most are the refusals: a gate that can be switched off by starving it of
input is not a gate.
"""

from __future__ import annotations

import unittest

from tools.ols_baseline import (
    GATE_FEATURE_SET,
    GATE_MARGIN_NORMALIZED,
    ols_gate_verdict,
)


def _phase_b(value_normalized: float, verdict: str = "ok") -> dict:
    return {
        "feature_sets": {
            GATE_FEATURE_SET: {
                "value_normalized": value_normalized,
                "verdict": verdict,
            }
        }
    }


def _phase_a(r2: float, variance: float = 0.05, verdict: str = "ok") -> dict:
    return {
        "r2_validation": r2,
        "validation_target_variance": variance,
        "verdict": verdict,
    }


class GateVerdictTests(unittest.TestCase):
    def test_a_network_beating_both_arms_passes(self) -> None:
        # Phase B: lower normalized value is better, so the network must
        # sit below the OLS by the margin. Phase A: higher R2 is better.
        passes, failures = ols_gate_verdict(
            _phase_b(1.045),
            _phase_a(0.198),
            network_value_normalized=0.90,
            # R2 = 1 - 0.030/0.05 = 0.40, comfortably above 0.198.
            network_equity_called_mse=0.030,
        )
        self.assertTrue(passes, failures)
        self.assertEqual(failures, [])

    def test_the_shipped_0003b_shape_fails_the_phase_a_arm(self) -> None:
        # The 2026-09-02 finding: the network wins on its own objective
        # and loses badly on the equity_called label.
        passes, failures = ols_gate_verdict(
            _phase_b(1.045),
            _phase_a(0.198),
            network_value_normalized=0.94368,
            # R2 = 1 - 0.045997/0.05 = 0.080, below 0.198 - 0.02.
            network_equity_called_mse=0.045997,
        )
        self.assertFalse(passes)
        self.assertTrue(any("phase-a equity_called" in f for f in failures))
        self.assertFalse(any("phase-b value head" in f for f in failures))

    def test_a_network_losing_the_value_arm_fails(self) -> None:
        passes, failures = ols_gate_verdict(
            _phase_b(1.045),
            _phase_a(0.198),
            network_value_normalized=1.045 - GATE_MARGIN_NORMALIZED,
            network_equity_called_mse=0.030,
        )
        self.assertFalse(passes)
        self.assertTrue(any("phase-b value head" in f for f in failures))


class GateFailsClosedTests(unittest.TestCase):
    """Every unevaluable arm must refuse, never wave the candidate through.

    Before 2026-09-02 the phase-a arm was skipped when its baseline was
    missing or singular, so the harder of the two gates could be switched
    off by making it uncomputable.
    """

    def test_a_missing_phase_a_baseline_refuses(self) -> None:
        passes, failures = ols_gate_verdict(
            _phase_b(1.045),
            None,
            network_value_normalized=0.90,
            network_equity_called_mse=0.030,
        )
        self.assertFalse(passes)
        self.assertTrue(any("cannot be evaluated" in f for f in failures))

    def test_a_singular_phase_a_baseline_refuses(self) -> None:
        passes, failures = ols_gate_verdict(
            _phase_b(1.045),
            _phase_a(0.198, verdict="singular"),
            network_value_normalized=0.90,
            network_equity_called_mse=0.030,
        )
        self.assertFalse(passes)
        self.assertTrue(any("singular" in f for f in failures))

    def test_a_singular_phase_b_baseline_refuses(self) -> None:
        passes, failures = ols_gate_verdict(
            _phase_b(1.045, verdict="singular"),
            _phase_a(0.198),
            network_value_normalized=0.90,
            network_equity_called_mse=0.030,
        )
        self.assertFalse(passes)
        self.assertTrue(any("singular" in f for f in failures))

    def test_missing_network_numbers_refuse(self) -> None:
        for value, mse, expected in (
            (None, 0.030, "no validation value_normalized"),
            (0.90, None, "no validation equity_called MSE"),
        ):
            with self.subTest(value=value, mse=mse):
                passes, failures = ols_gate_verdict(
                    _phase_b(1.045),
                    _phase_a(0.198),
                    network_value_normalized=value,
                    network_equity_called_mse=mse,
                )
                self.assertFalse(passes)
                self.assertTrue(any(expected in f for f in failures))


if __name__ == "__main__":
    unittest.main()
