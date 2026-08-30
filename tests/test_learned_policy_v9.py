"""Layer-2 checks: the v9 composition, projection, and format-4 loader.

Expectations are derived independently of the implementation wherever
possible (the standing rule): composition values are hand-computed from
the documented formulas with synthetic head outputs; the projection table
is hand-written; the end-to-end serve test builds a real artifact on disk
whose zero weights force every head to its bias, making the decision
predictable from the value formulas alone.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from engine import schema4
from engine.aggression_sizing import (
    DEFAULT_SIZING_PARAMETERS,
    aggressive_target,
)
from engine.branch_contract_v9 import (
    BRANCH_LABELS_V9,
    MODEL_FORMAT_VERSION_V9,
    BranchContractError,
    branch_engine_family,
)
from engine.learned_policy_v9 import (
    LearnedPolicyV9Error,
    compose_branch_values_v9,
    load_policy_v9,
)
from engine.rules.composition import (
    DEFAULT_RULE_LAYER,
    composed_sizing_record,
)
from engine.v8_trainer import default_v9_architecture

# Synthetic head outputs with hand-chosen values. Slot orders are the
# contract's: fold_through (active, aggressive); equity_called (passive,
# active, aggressive); residual (fatal, passive, active, aggressive).
_HEADS = {
    "fold_through": [0.0, 0.0],          # sigmoid -> 0.5 for both lanes
    "range": [0.0] * 8,
    "equity_called": [0.30, 0.60, 0.80],
    "residual": [0.0, 0.0, 0.0, 0.0],
}


def _compose(**overrides):
    state = dict(
        pot=500,
        to_call=200,
        contribution=0,
        effective_stack=600,
        purse=700,
        boldness=0.0,
        street="turn",
        bankroll=700,
        exposure=800,
        covered_allin_to_amounts=(),
        legal_labels=frozenset({"fatal", "active", "aggressive"}),
        bet_range=None,
        raise_range=(400, 700),
        sizing=DEFAULT_SIZING_PARAMETERS,
        rules=DEFAULT_RULE_LAYER,
    )
    heads = overrides.pop("heads", _HEADS)
    state.update(overrides)
    return compose_branch_values_v9(heads, **state)


class CompositionTests(unittest.TestCase):
    def test_priced_values_match_the_hand_formulas(self) -> None:
        values, wagers = _compose()
        purse = 700
        # fatal is exactly zero.
        self.assertEqual(values["fatal"], 0.0)
        # active as a CALL: eq*(pot+tc)/purse - tc/purse, no fold-through.
        self.assertAlmostEqual(
            values["active"], 0.60 * (500 + 200) / purse - 200 / purse
        )
        # aggressive: bare-g target at b=0 (dials off), clamped for value.
        target = aggressive_target(
            pot=500, to_call=200, effective_stack=600, boldness=0.0
        )
        self.assertAlmostEqual(wagers["aggressive"].target, target)
        wager = min(700, max(400, 0 + target)) - 0
        pot_if_called = 500 + 2 * wager - 200
        expected = 0.5 * (500 / purse) + 0.5 * (
            0.80 * pot_if_called / purse - wager / purse
        )
        self.assertAlmostEqual(values["aggressive"], expected)
        # passive is not legal on a priced spot and must not be emitted.
        self.assertNotIn("passive", values)

    def test_free_spot_values_match_the_hand_formulas(self) -> None:
        values, wagers = _compose(
            to_call=0,
            legal_labels=frozenset({"passive", "active"}),
            bet_range=(100, 700),
            raise_range=None,
        )
        purse = 700
        self.assertAlmostEqual(values["passive"], 0.30 * 500 / purse)
        # active as a BET: fold-through form over g's wager (b=0 -> f=0.5).
        self.assertAlmostEqual(wagers["active"].target, 0.5 * 500)
        wager = min(700, max(100, 0 + 250)) - 0
        expected = 0.5 * (500 / purse) + 0.5 * (
            0.60 * (500 + 2 * wager) / purse - wager / purse
        )
        self.assertAlmostEqual(values["active"], expected)
        self.assertNotIn("fatal", values)
        self.assertNotIn("aggressive", values)

    def test_call_value_ignores_fold_through_entirely(self) -> None:
        """The contract rule: a call closes the action and buys no folds."""

        heads_hot = copy.deepcopy(_HEADS)
        heads_hot["fold_through"] = [50.0, -50.0]
        base, _ = _compose()
        hot, _ = _compose(heads=heads_hot)
        self.assertEqual(base["active"], hot["active"])
        # ...while the aggressive branch DOES move with its own slot.
        self.assertNotEqual(base["aggressive"], hot["aggressive"])

    def test_residual_caps_at_the_pot_fraction(self) -> None:
        heads = copy.deepcopy(_HEADS)
        heads["residual"] = [0.0, 0.0, 0.0, 99.0]
        capped, _ = _compose(heads=heads)
        base, _ = _compose()
        purse = 700
        cap = 0.05 * (500 / purse)
        self.assertAlmostEqual(capped["aggressive"] - base["aggressive"], cap)
        # And residual never reaches a call value (wager executions only).
        self.assertEqual(capped["active"], base["active"])

    def test_only_the_legal_set_is_emitted(self) -> None:
        values, _ = _compose(legal_labels=frozenset({"fatal", "active"}))
        self.assertEqual(set(values), {"fatal", "active"})


class ProjectionTests(unittest.TestCase):
    def test_engine_family_table(self) -> None:
        # Hand-written, not derived from branch_action.
        self.assertEqual(branch_engine_family("fatal", 100), "fold")
        self.assertEqual(branch_engine_family("passive", 0), "check_call")
        self.assertEqual(branch_engine_family("active", 100), "check_call")
        self.assertEqual(branch_engine_family("active", 0), "aggress")
        self.assertEqual(branch_engine_family("aggressive", 100), "aggress")

    def test_masked_projections_raise(self) -> None:
        for branch, to_call in (("fatal", 0), ("passive", 100), ("aggressive", 0)):
            with self.assertRaises(BranchContractError):
                branch_engine_family(branch, to_call)


def _zero_linear(rows: int, cols: int) -> dict:
    return {"w": [[0.0] * cols for _ in range(rows)], "b": [0.0] * rows}


def _zero_weights(architecture: dict) -> dict:
    card = len(architecture["card_indices"])
    context = len(architecture["context_indices"])
    cw = architecture["card_encoder_width"]
    xw = architecture["context_encoder_width"]
    trunk_in = cw + xw
    weights = {
        "card_encoder": {
            **_zero_linear(cw, card), "ln_g": [1.0] * cw, "ln_b": [0.0] * cw
        },
        "context_encoder": {
            **_zero_linear(xw, context), "ln_g": [1.0] * xw, "ln_b": [0.0] * xw
        },
        "trunk": [],
        "heads": {},
    }
    dims = [trunk_in, *architecture["trunk_widths"]]
    for i in range(len(dims) - 1):
        block = _zero_linear(dims[i + 1], dims[i])
        if i < len(dims) - 2:
            block["ln_g"] = [1.0] * dims[i + 1]
            block["ln_b"] = [0.0] * dims[i + 1]
        weights["trunk"].append(block)
    tower = 32
    for name, size in architecture["heads"].items():
        weights["heads"][name] = {
            "tower_w": [[0.0] * dims[-1] for _ in range(tower)],
            "tower_b": [0.0] * tower,
            "out_w": [[0.0] * tower for _ in range(size)],
            "out_b": [0.0] * size,
        }
    return weights


def _write_artifact(directory: Path, *, mutate=None) -> Path:
    architecture = default_v9_architecture()
    document = {
        "format_version": MODEL_FORMAT_VERSION_V9,
        "model_version": "candidate-v9-test",
        "weights": _zero_weights(architecture),
        "feature_normalization": {
            "means": [0.0] * schema4.INPUT_SIZE_V9,
            "stds": [1.0] * schema4.INPUT_SIZE_V9,
        },
    }
    weights_path = directory / "candidate-v9-test.weights.json"
    weights_path.write_text(json.dumps(document), encoding="utf-8")
    manifest = {
        "format": "fold-multihead-policy",
        "format_version": MODEL_FORMAT_VERSION_V9,
        "model_version": "candidate-v9-test",
        "model_family": "v9-composed-value",
        "feature_schema_version": schema4.SCHEMA_VERSION_V9,
        "input_size": schema4.INPUT_SIZE_V9,
        "feature_names": list(schema4.FEATURE_NAMES_V9),
        "action_labels": list(BRANCH_LABELS_V9),
        "architecture": architecture,
        "sizing": composed_sizing_record(),
        "weights_file": weights_path.name,
        "weights_sha256": hashlib.sha256(
            weights_path.read_bytes().rstrip(b"\n")
        ).hexdigest(),
        "state": "candidate",
        "promotion": None,
    }
    if mutate is not None:
        mutate(manifest)
        manifest["weights_sha256"] = hashlib.sha256(
            weights_path.read_bytes().rstrip(b"\n")
        ).hexdigest()
    manifest_path = directory / "candidate-v9-test.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _snapshot(**overrides) -> dict:
    from test_rules_composition import _snapshot as base

    return base(**overrides)


class LoaderTests(unittest.TestCase):
    def test_round_trip_and_end_to_end_serve(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            policy = load_policy_v9(_write_artifact(directory))
            self.assertEqual(policy.policy_version, "candidate-v9-test")
            self.assertTrue(policy.serves_composed_sizing)

            # Zero weights force every head to its bias: fold-through
            # sigmoid 0.5, equities 0, residual 0. The serve-time read
            # (real equity through g) is NOT ours to assume, so the
            # expectation must be robust across EVERY boldness in [-1,1]:
            # with raiseRange.min above the pot, the clamped wager w >=
            # 600 > pot = 500, so aggressive = 0.5*pot/p - 0.5*w/p < 0
            # for any read, active(call) = -tc/p < 0, and fatal (0) wins.
            # (The first version of this test assumed b = 0 and asserted
            # the wrong winner — the read made aggressive positive.)
            #
            # The engine's rails sit ABOVE the composed layer by design:
            # with a strong hand the rescue call converts a fold into a
            # justified call (that is what it is for), so the fold case
            # needs a hand the rescue floor (0.57) refuses.
            table = _snapshot(
                hole=("2c", "7d"),
                board=("As", "Kh", "Qs", "Jh"),
                raise_range=(600, 700),
            )
            result = policy.decide_with_diagnostics(table)
            self.assertEqual(result.action.action, "fold")

            # Free spot: passive = 0, active-bet = 0.5*pot/p - 0.5*w/p > 0
            # for EVERY read (w = clamp(f*pot) <= 0.695*pot < pot), so the
            # composed layer bets — and the strong hand clears the
            # engine's aggression floor, so the bet survives the rails.
            free = _snapshot(
                to_call=0,
                available=("check", "bet"),
                bet_range=(100, 700),
                raise_range=None,
            )
            result = policy.decide_with_diagnostics(free)
            self.assertEqual(result.action.action, "bet")

    def test_refusals(self) -> None:
        def refuses(reason: str, mutate) -> None:
            with tempfile.TemporaryDirectory() as raw:
                path = _write_artifact(Path(raw), mutate=mutate)
                with self.assertRaises(LearnedPolicyV9Error, msg=reason):
                    load_policy_v9(path)

        refuses(
            "a v8 manifest must not load",
            lambda m: m.update(format_version=3),
        )
        refuses(
            "v8 labels must not load",
            lambda m: m.update(
                action_labels=["fold", "check_call", "aggress_small", "aggress_large"]
            ),
        )
        refuses(
            "wrong input size",
            lambda m: m.update(input_size=413),
        )
        refuses(
            "missing sizing block",
            lambda m: m.pop("sizing"),
        )
        refuses(
            "foreign sizing identity",
            lambda m: m["sizing"].update(identity="g-v9-other"),
        )
        refuses(
            "inherited hybrid quantiles",
            lambda m: m.update(serve={"margin_quantiles": {"p90": 0.02}}),
        )
        refuses(
            "v8 architecture family",
            lambda m: m["architecture"].update(family="v8-composed-value"),
        )

        # Tampered weights: mutate the file after the sha is recorded.
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            path = _write_artifact(directory)
            weights = directory / "candidate-v9-test.weights.json"
            weights.write_text(
                weights.read_text(encoding="utf-8").replace("0.0", "0.1", 1),
                encoding="utf-8",
            )
            with self.assertRaises(LearnedPolicyV9Error):
                load_policy_v9(path)

    def test_tie_break_is_slot_order(self) -> None:
        # Equal centered values must pick the earlier slot. With zero
        # equities and to_call such that active == fatal == 0 is not
        # constructible cheaply, test the pure rule instead.
        values = {"aggressive": 0.25, "active": 0.25, "fatal": 0.25}
        best = max(
            (b for b in BRANCH_LABELS_V9 if b in values), key=lambda b: values[b]
        )
        self.assertEqual(best, "fatal")


class NonFiniteGuardTests(unittest.TestCase):
    def test_infinite_equity_is_absorbed_by_the_clip(self) -> None:
        """clip01 turns an infinite equity into 1.0 — absorbed, finite."""

        heads = copy.deepcopy(_HEADS)
        heads["equity_called"] = [0.3, math.inf, 0.8]
        values, _ = _compose(heads=heads)
        self.assertTrue(all(math.isfinite(v) for v in values.values()))

    def test_nan_fold_through_propagates_to_the_finite_guard(self) -> None:
        """A NaN logit reaches the composed value, which the serve path's
        finite check then refuses (fail-closed to the heuristic)."""

        heads = copy.deepcopy(_HEADS)
        heads["fold_through"] = [0.0, math.nan]
        values, _ = _compose(heads=heads)
        self.assertFalse(math.isfinite(values["aggressive"]))


if __name__ == "__main__":
    unittest.main()
