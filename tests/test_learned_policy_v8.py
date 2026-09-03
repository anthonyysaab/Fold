"""Checks for the v8 composed-value runtime: loading, forward, composition.

Four concerns, per the serve-path contract (V8_DESIGN §4 amended):

- ``_forward_v3`` against hand-computed fixtures (closed-form LayerNorm
  arithmetic, ReLU clamps, the card/context index partition) and against
  the real torch network run on the trained ``candidate-v8-0001`` artifact
  in the CUDA venv (subprocess; skipped when that interpreter or the
  artifact is absent).
- ``compose_branch_values`` hand-checked on one fully worked state,
  including the residual cap, its ablation, and the E6 emission and
  pinned-fraction rules.
- Legality masking: the argmax only ever proposes families the snapshot
  allows, and the serve path fails closed to the heuristic engine.
- A full ``decide()`` smoke on a fixture table returning a legal action
  deterministically, plus the loader's fail-loud validation battery.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine import schema3
from engine.decision_engine import DEFAULT_SAFETY_GATES
from engine.game_state import features_from_table
from engine.learned_policy import LearnedPolicyError
from engine.learned_policy_v8 import (
    LearnedPokerPolicyV8,
    LearnedPolicyV8Error,
    _forward_v3,
    compose_branch_values,
    load_policy_v8,
)
from engine.learning_contract import MODEL_FORMAT
from engine.v8_trainer import (
    BRANCH_LABELS_V8,
    CARD_ENCODER_WIDTH,
    CONTEXT_ENCODER_WIDTH,
    HEAD_TOWER_WIDTH,
    MODEL_FORMAT_VERSION_V8,
    TRUNK_WIDTHS,
    V8_HEAD_SIZES,
    default_v8_architecture,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAINED_MANIFEST = (
    _REPO_ROOT / "artifacts" / "candidates" / "candidate-v8-0001.manifest.json"
)
#: The training interpreter moved on 2026-09-03 and now lives INSIDE the
#: repo (owner decision, `.handoff/DECISIONS.md` section 6), so this is
#: repo-relative rather than absolute. It was
#: ``C:/Users/user/poker-nn-training/.venv/...`` until then, a path that no
#: longer exists -- which meant the two tests guarded on it skipped even
#: when run on the CUDA interpreter, silently and forever.
_CUDA_VENV_PYTHON = (
    _REPO_ROOT / "neural network training" / ".venv" / "Scripts" / "python.exe"
)

_LN_EPS = 1e-5
_IDENTITY_NORMALIZATION = {
    "means": [0.0] * schema3.INPUT_SIZE_V8,
    "stds": [1.0] * schema3.INPUT_SIZE_V8,
}


# ---------------------------------------------------------------------------
# Fixture weights
# ---------------------------------------------------------------------------


def _matrix(rows: int, cols: int, fill: float = 0.0) -> list[list[float]]:
    return [[fill] * cols for _ in range(rows)]


def _vector(size: int, fill: float = 0.0) -> list[float]:
    return [fill] * size


def _zero_weights() -> dict:
    """A structurally valid format-3 weights block, everything 0.0."""

    trunk_input = CARD_ENCODER_WIDTH + CONTEXT_ENCODER_WIDTH
    return {
        "card_encoder": {
            "w": _matrix(CARD_ENCODER_WIDTH, schema3.CARD_BLOCK_SIZE),
            "b": _vector(CARD_ENCODER_WIDTH),
            "ln_g": _vector(CARD_ENCODER_WIDTH),
            "ln_b": _vector(CARD_ENCODER_WIDTH),
        },
        "context_encoder": {
            "w": _matrix(CONTEXT_ENCODER_WIDTH, schema3.CONTEXT_BLOCK_SIZE),
            "b": _vector(CONTEXT_ENCODER_WIDTH),
            "ln_g": _vector(CONTEXT_ENCODER_WIDTH),
            "ln_b": _vector(CONTEXT_ENCODER_WIDTH),
        },
        "trunk": [
            {
                "w": _matrix(TRUNK_WIDTHS[0], trunk_input),
                "b": _vector(TRUNK_WIDTHS[0]),
                "ln_g": _vector(TRUNK_WIDTHS[0]),
                "ln_b": _vector(TRUNK_WIDTHS[0]),
            },
            {
                "w": _matrix(TRUNK_WIDTHS[1], TRUNK_WIDTHS[0]),
                "b": _vector(TRUNK_WIDTHS[1]),
            },
        ],
        "heads": {
            name: {
                "tower_w": _matrix(HEAD_TOWER_WIDTH, TRUNK_WIDTHS[-1]),
                "tower_b": _vector(HEAD_TOWER_WIDTH),
                "out_w": _matrix(size, HEAD_TOWER_WIDTH),
                "out_b": _vector(size),
            }
            for name, size in V8_HEAD_SIZES.items()
        },
    }


def _bias_weights(out_biases: dict[str, list[float]]) -> dict:
    """Zero weights whose head outputs equal the given biases exactly."""

    weights = _zero_weights()
    for name, biases in out_biases.items():
        weights["heads"][name]["out_b"] = list(biases)
    return weights


def _architecture() -> dict:
    return default_v8_architecture()


# ---------------------------------------------------------------------------
# Fixture tables (self-contained; same snapshot dialect as test_v8_parity)
# ---------------------------------------------------------------------------


def _seat(
    number: int,
    stack: int,
    *,
    status: str = "Active",
    bet: int = 0,
    committed: int | None = None,
    hole: tuple[str, str] | None = None,
) -> dict:
    seat: dict = {
        "seatNumber": number,
        "status": status,
        "stackChips": stack,
        "currentBetChips": bet,
        "holeCards": list(hole) if hole else None,
    }
    if committed is not None:
        seat["totalCommittedChips"] = committed
    return seat


def _event(street: str, seat: int, action: str, **amounts: int) -> dict:
    summary: dict = {"seatNumber": seat, "action": action}
    summary.update(amounts)
    return {"type": "ActionTaken", "street": street, "summary": summary}


def _turn_table(
    available: tuple[str, ...] = ("fold", "call", "raise"),
    raise_range: tuple[int, int] | None = (400, 900),
) -> dict:
    """Hero bet the flop, faces a 200 turn bet into a 1300 pot."""

    return {
        "id": "v8-serve-test",
        "tableId": "v8-serve-test",
        "street": "turn",
        "potChips": 1300,
        "currentBet": 200,
        "boardCards": ["2c", "7d", "9h", "Qs"],
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": 1,
        "seats": [
            _seat(1, 2600, hole=("Ah", "Kd"), committed=500),
            _seat(2, 1800, bet=200, committed=700),
            _seat(3, 900, committed=100, status="Folded"),
        ],
        "allowedActions": {
            "canFold": "fold" in available,
            "canCheck": "check" in available,
            "canCall": "call" in available,
            "canBet": "bet" in available,
            "canRaise": "raise" in available,
            "canAllIn": False,
            "callAmount": 200,
            "callChips": 200,
            "callToAmount": 200,
            "betRange": None,
            "raiseRange": (
                {"min": raise_range[0], "max": raise_range[1]}
                if raise_range
                else None
            ),
            "allInToAmount": None,
            "availableActions": list(available),
            "amountSemantics": "toAmount",
        },
        "recentEvents": [
            _event("preflop", 1, "call", amount=100),
            _event("preflop", 2, "check"),
            _event("flop", 1, "bet", toAmount=200),
            _event("flop", 2, "call", amount=200),
            _event("turn", 2, "bet", toAmount=200),
        ],
    }


def _policy(weights: dict, **overrides) -> LearnedPokerPolicyV8:
    keywords = {
        "model_version": "v8-fixture",
        "architecture": _architecture(),
        "weights": weights,
        "normalization": _IDENTITY_NORMALIZATION,
        "equity_trials": 4,
        "potential_trials": 16,
    }
    keywords.update(overrides)
    return LearnedPokerPolicyV8(**keywords)


# ---------------------------------------------------------------------------
# _forward_v3 against hand-computed fixtures
# ---------------------------------------------------------------------------


class ForwardV3HandFixtureTests(unittest.TestCase):
    def test_zero_network_outputs_exactly_the_head_biases(self) -> None:
        # All weights zero: every LayerNorm sees a constant (zero) vector,
        # so its output is exactly ln_b (also zero), and each head output
        # is exactly its out_b — hand-computable with no arithmetic at all.
        biases = {
            "fold_through": [0.25, -0.5],
            "range": [0.125 * index for index in range(8)],
            "equity_called": [0.1, 0.2, 0.3],
            "residual": [1.0, 2.0, 3.0, 4.0],
        }
        outputs = _forward_v3(
            _architecture(),
            _bias_weights(biases),
            [0.7] * schema3.INPUT_SIZE_V8,
        )
        self.assertEqual(set(outputs), set(V8_HEAD_SIZES))
        for name, expected in biases.items():
            self.assertEqual(outputs[name], expected)

    def test_constant_channel_arithmetic_is_exact(self) -> None:
        # Constant pre-activation vectors make every LayerNorm output its
        # beta exactly ((v - mean) is exactly 0), and 1/128 and 1/32 are
        # dyadic so every partial sum is exact: the whole expected output
        # is derivable on paper. The residual tower's negative
        # pre-activation additionally pins the tower ReLU: without it the
        # residual head would read -1.0 + 0.5 instead of its bias.
        weights = _zero_weights()
        weights["card_encoder"]["ln_g"] = _vector(CARD_ENCODER_WIDTH, 1.0)
        weights["card_encoder"]["ln_b"] = _vector(CARD_ENCODER_WIDTH, 0.5)
        weights["context_encoder"]["ln_g"] = _vector(CONTEXT_ENCODER_WIDTH, 1.0)
        weights["context_encoder"]["ln_b"] = _vector(CONTEXT_ENCODER_WIDTH, 0.25)
        trunk_input = CARD_ENCODER_WIDTH + CONTEXT_ENCODER_WIDTH
        weights["trunk"][0]["w"] = _matrix(TRUNK_WIDTHS[0], trunk_input, 1.0)
        weights["trunk"][0]["ln_g"] = _vector(TRUNK_WIDTHS[0], 1.0)
        weights["trunk"][0]["ln_b"] = _vector(TRUNK_WIDTHS[0], 1.0)
        weights["trunk"][1]["w"] = _matrix(
            TRUNK_WIDTHS[1], TRUNK_WIDTHS[0], 1.0 / 128.0
        )
        for name in V8_HEAD_SIZES:
            fill = -1.0 / 128.0 if name == "residual" else 1.0 / 128.0
            weights["heads"][name]["tower_w"] = _matrix(
                HEAD_TOWER_WIDTH, TRUNK_WIDTHS[-1], fill
            )
            weights["heads"][name]["out_w"] = _matrix(
                V8_HEAD_SIZES[name], HEAD_TOWER_WIDTH, 1.0 / 32.0
            )
        weights["heads"]["fold_through"]["out_b"] = [0.1, -0.2]
        weights["heads"]["equity_called"]["out_b"] = [0.01, 0.02, 0.03]
        weights["heads"]["residual"]["out_b"] = _vector(4, 0.5)

        # Hand derivation: encoders emit their betas (0.5 / 0.25), so
        # trunk[0] sees 64*0.5 + 48*0.25 = 44 on every unit, LayerNorm
        # collapses the constant to 1.0, trunk[1] emits 128*(1/128) = 1.0,
        # the three positive towers emit 1.0, and each output is
        # 32*(1/32)*1.0 + out_b = 1.0 + out_b. The residual tower emits
        # relu(-1.0) = 0, so residual outputs are exactly out_b.
        outputs = _forward_v3(
            _architecture(), weights, [0.0] * schema3.INPUT_SIZE_V8
        )
        for value, expected in zip(
            outputs["fold_through"] + outputs["range"] + outputs["equity_called"],
            [1.1, 0.8] + [1.0] * 8 + [1.01, 1.02, 1.03],
        ):
            self.assertAlmostEqual(value, expected, places=12)
        for value in outputs["residual"]:
            self.assertAlmostEqual(value, 0.5, places=12)

    def test_single_hot_card_feature_routes_and_normalizes(self) -> None:
        # One card feature drives one encoder unit; the closed form for
        # LayerNorm of (h, 0, ..., 0) over n units is hand-derivable:
        # mean = h/n, var = ((h - mean)^2 + (n-1) mean^2)/n. The signal
        # must ride the card partition (a context feature must be inert)
        # and a negative input must vanish at the encoder ReLU.
        weights = _zero_weights()
        weights["card_encoder"]["w"][0][0] = 1.0
        weights["card_encoder"]["ln_g"] = _vector(CARD_ENCODER_WIDTH, 1.0)
        weights["context_encoder"]["ln_g"] = _vector(CONTEXT_ENCODER_WIDTH, 1.0)
        weights["trunk"][0]["w"][0][0] = 1.0
        weights["trunk"][0]["ln_g"] = _vector(TRUNK_WIDTHS[0], 1.0)
        for row in weights["trunk"][1]["w"]:
            row[0] = 1.0
        for name in V8_HEAD_SIZES:
            for row in weights["heads"][name]["tower_w"]:
                row[0] = 1.0
            for row in weights["heads"][name]["out_w"]:
                row[0] = 1.0

        def expected_hot(h: float, n: int) -> float:
            mean = h / n
            variance = ((h - mean) ** 2 + (n - 1) * mean * mean) / n
            return (h - mean) / math.sqrt(variance + _LN_EPS)

        h1 = expected_hot(3.0, CARD_ENCODER_WIDTH)
        h2 = expected_hot(h1, TRUNK_WIDTHS[0])

        features = [0.0] * schema3.INPUT_SIZE_V8
        features[0] = 3.0
        outputs = _forward_v3(_architecture(), weights, features)
        for name in V8_HEAD_SIZES:
            for value in outputs[name]:
                self.assertAlmostEqual(value, h2, places=9)

        # A context feature must not reach the card encoder.
        with_context = list(features)
        with_context[schema3.CONTEXT_INDICES[0]] = 7.0
        self.assertEqual(
            _forward_v3(_architecture(), weights, with_context), outputs
        )

        # A negative card input is clamped by the encoder ReLU, so every
        # layer sees zeros and every head reads exactly its (zero) bias.
        negative = [0.0] * schema3.INPUT_SIZE_V8
        negative[0] = -3.0
        for name, values in _forward_v3(
            _architecture(), weights, negative
        ).items():
            self.assertEqual(values, [0.0] * V8_HEAD_SIZES[name])


# ---------------------------------------------------------------------------
# Composition arithmetic
# ---------------------------------------------------------------------------


_HAND_STATE = {
    "pot": 1000,
    "to_call": 200,
    "contribution": 100,
    "effective_stack": 2000,
    "purse": 5000,
    "legal_range": (400, 2600),
}
_HAND_OUTPUTS = {
    "fold_through": [0.0, math.log(3.0)],  # sigmoid -> 0.5, 0.75
    "range": [0.125] * 8,  # never read by the composition
    "equity_called": [0.6, 0.55, 0.45],  # small, large, check_call
    # Fold/check_call residual entries are deliberately wild: §4 reads
    # only the aggress entries, so they must never leak into a value.
    "residual": [7.0, -7.0, 0.5, -1.0],
}


class CompositionArithmeticTests(unittest.TestCase):
    def test_hand_checked_state(self) -> None:
        # Worked by hand (purse 5000, pot 1000, call 200):
        #   pot_u = 0.2, cap = 0.05 * 0.2 = 0.01
        #   V(check_call) = 0.45*1200/5000 - 200/5000 = 0.108-0.040 = 0.068
        #   small: target = min(200+0.5*1200, 0.2*2000) = 400 (stack arm),
        #     to = clamp(100+400, 400, 2600) = 500, wager 400,
        #     pot' = 1000 + 800 - 200 = 1600,
        #     V = 0.5*0.2 + 0.5*(0.6*1600/5000 - 400/5000) + cap(0.5)
        #       = 0.1 + 0.5*(0.192-0.080) + 0.01 = 0.166
        #   large: target = min(200+1200, 0.45*2000) = 900 (stack arm),
        #     to = clamp(100+900, 400, 2600) = 1000, wager 900,
        #     pot' = 1000 + 1800 - 200 = 2600,
        #     V = 0.75*0.2 + 0.25*(0.55*2600/5000 - 900/5000) - cap(1.0)
        #       = 0.15 + 0.25*(0.286-0.180) - 0.01 = 0.1665
        values, fractions = compose_branch_values(_HAND_OUTPUTS, **_HAND_STATE)
        self.assertEqual(values["fold"], 0.0)
        self.assertAlmostEqual(values["check_call"], 0.068, places=12)
        self.assertAlmostEqual(values["aggress_small"], 0.166, places=12)
        self.assertAlmostEqual(values["aggress_large"], 0.1665, places=12)
        # Pinned fractions come from the unclamped stack-arm targets:
        # (400-200)/1200 and (900-200)/1200.
        self.assertAlmostEqual(fractions["aggress_small"], 200.0 / 1200.0, places=12)
        self.assertAlmostEqual(fractions["aggress_large"], 700.0 / 1200.0, places=12)

    def test_residual_is_capped_and_ablatable(self) -> None:
        values, _ = compose_branch_values(_HAND_OUTPUTS, **_HAND_STATE)
        bare, _ = compose_branch_values(
            _HAND_OUTPUTS, use_residual=False, **_HAND_STATE
        )
        # The raw corrections are +-0.5/-1.0 but the cap is 0.01: exactly
        # +-cap survives, in the sign of the correction.
        self.assertAlmostEqual(values["aggress_small"] - bare["aggress_small"], 0.01)
        self.assertAlmostEqual(values["aggress_large"] - bare["aggress_large"], -0.01)
        self.assertAlmostEqual(bare["aggress_small"], 0.156, places=12)
        self.assertAlmostEqual(bare["aggress_large"], 0.1765, places=12)
        # Passive branches never read the residual head at all.
        self.assertEqual(values["fold"], bare["fold"])
        self.assertEqual(values["check_call"], bare["check_call"])

    def test_emission_rules(self) -> None:
        # No stated bet/raise range: no aggress branch is emitted.
        state = dict(_HAND_STATE, legal_range=None)
        values, fractions = compose_branch_values(_HAND_OUTPUTS, **state)
        self.assertEqual(set(values), {"fold", "check_call"})
        self.assertEqual(fractions, {})
        # A collapsed legal range absorbs the large branch into the small
        # one (both clamp to the same to-amount).
        state = dict(_HAND_STATE, legal_range=(400, 450))
        values, fractions = compose_branch_values(_HAND_OUTPUTS, **state)
        self.assertIn("aggress_small", values)
        self.assertNotIn("aggress_large", values)
        self.assertEqual(set(fractions), {"aggress_small"})

    def test_deep_stacks_recover_the_v7_pot_fractions_exactly(self) -> None:
        # When the pot arm binds (E6's deep-spot property) the pinned
        # fractions are byte-identical to v7's static half-pot/full-pot.
        _, fractions = compose_branch_values(
            _HAND_OUTPUTS,
            pot=1000,
            to_call=0,
            contribution=0,
            effective_stack=100_000,
            purse=100_000,
            legal_range=(1, 1_000_000),
        )
        self.assertEqual(fractions["aggress_small"], 0.5)
        self.assertEqual(fractions["aggress_large"], 1.0)


# ---------------------------------------------------------------------------
# Legality masking and fail-closed serving
# ---------------------------------------------------------------------------


_AGGRESSIVE_BIASES = {
    "fold_through": [6.0, 6.0],
    "equity_called": [0.95, 0.95, 0.2],
}


class LegalityMaskingTests(unittest.TestCase):
    def _family(self, policy: LearnedPokerPolicyV8, table: dict) -> str:
        allowed = table["allowedActions"]
        available = set(allowed["availableActions"])
        return policy._equity_family(
            table, allowed, available, 0.5, features=features_from_table(table)
        )

    def test_aggression_needs_an_executable_branch(self) -> None:
        policy = _policy(_bias_weights(_AGGRESSIVE_BIASES))
        table = _turn_table(available=("fold", "call", "raise"))
        self.assertEqual(self._family(policy, table), "aggress")
        self.assertIsNotNone(policy._branch_pot_fraction)

        # Same rigged network, no legal aggression: the aggress branches
        # are not emitted and the (positive-value) call is chosen instead.
        passive = _turn_table(available=("fold", "call"), raise_range=None)
        self.assertEqual(self._family(policy, passive), "check_call")
        self.assertIsNone(policy._branch_pot_fraction)

    def test_negative_call_value_folds(self) -> None:
        biases = {
            "fold_through": [-9.0, -9.0],
            # check_call slot clips to 0.0: V(check_call) < 0 < V(fold).
            "equity_called": [0.0, 0.0, -5.0],
        }
        policy = _policy(_bias_weights(biases))
        table = _turn_table(available=("fold", "call"), raise_range=None)
        self.assertEqual(self._family(policy, table), "fold")

    def test_feature_or_forward_failure_falls_back_to_the_heuristic(self) -> None:
        policy = _policy(_bias_weights(_AGGRESSIVE_BIASES))
        table = _turn_table()
        available = list(table["allowedActions"]["availableActions"])
        with patch(
            "engine.learned_policy_v8.extract_features_v8",
            side_effect=RuntimeError("boom"),
        ):
            payload = policy.decide(table)
        self.assertIn(payload["action"], available)

        nan_outputs = {
            "fold_through": [math.nan, math.nan],
            "range": [0.0] * 8,
            "equity_called": [math.nan] * 3,
            "residual": [0.0] * 4,
        }
        with patch(
            "engine.learned_policy_v8._forward_v3",
            return_value=nan_outputs,
        ):
            payload = policy.decide(table)
        self.assertIn(payload["action"], available)


# ---------------------------------------------------------------------------
# decide() smoke
# ---------------------------------------------------------------------------


class DecideSmokeTests(unittest.TestCase):
    def test_decide_returns_a_legal_action_deterministically(self) -> None:
        table = _turn_table()
        available = list(table["allowedActions"]["availableActions"])
        payloads = []
        for _ in range(2):  # fresh instances: no shared tracker state
            policy = _policy(_bias_weights(_AGGRESSIVE_BIASES))
            payloads.append(policy.decide(_turn_table()))
        self.assertEqual(payloads[0], payloads[1])
        self.assertIn(payloads[0]["action"], available)
        # The rigged network picks an aggress branch; the raise executes
        # through the engine's sizing path inside the stated legal range.
        self.assertEqual(payloads[0]["action"], "raise")
        self.assertGreaterEqual(payloads[0]["amount"], 400)
        self.assertLessEqual(payloads[0]["amount"], 900)

    @unittest.skipUnless(
        _TRAINED_MANIFEST.is_file(), "trained v8 artifact is absent"
    )
    def test_trained_artifact_loads_and_decides(self) -> None:
        table = _turn_table()
        available = list(table["allowedActions"]["availableActions"])
        payloads = []
        for _ in range(2):
            policy = load_policy_v8(
                _TRAINED_MANIFEST, equity_trials=8, potential_trials=32
            )
            payloads.append(policy.decide(_turn_table()))
        self.assertEqual(payloads[0], payloads[1])
        self.assertIn(payloads[0]["action"], available)


# ---------------------------------------------------------------------------
# Loader validation
# ---------------------------------------------------------------------------


def _write_artifact(
    directory: str,
    *,
    model_version: str = "v8-loader-fixture",
    weights: dict | None = None,
    means: list[float] | None = None,
    mutate_manifest=None,
) -> Path:
    document = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION_V8,
        "model_version": model_version,
        "feature_normalization": {
            "means": (
                means
                if means is not None
                else list(_IDENTITY_NORMALIZATION["means"])
            ),
            "stds": list(_IDENTITY_NORMALIZATION["stds"]),
        },
        "weights": weights if weights is not None else _zero_weights(),
    }
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    manifest = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION_V8,
        "model_version": model_version,
        "feature_schema_version": schema3.SCHEMA_VERSION,
        "input_size": schema3.INPUT_SIZE_V8,
        "feature_names": list(schema3.FEATURE_NAMES_V8),
        "action_labels": list(BRANCH_LABELS_V8),
        "architecture": default_v8_architecture(),
        "weights_file": f"{model_version}.weights.json",
        "weights_sha256": hashlib.sha256(encoded).hexdigest(),
        "serve": {
            "ood_guard_indices": list(schema3.CONTEXT_INDICES),
            "temperature": None,
        },
        "engine_parameters": {
            "safety_gates": DEFAULT_SAFETY_GATES.to_mapping()
        },
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    root = Path(directory)
    (root / f"{model_version}.weights.json").write_bytes(encoded + b"\n")
    manifest_path = root / f"{model_version}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


class LoadPolicyV8Tests(unittest.TestCase):
    def test_loads_verifies_and_rebuilds_engine_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = load_policy_v8(_write_artifact(directory))
            self.assertIsInstance(policy, LearnedPokerPolicyV8)
            self.assertEqual(policy.policy_version, "v8-loader-fixture")
            self.assertEqual(policy.safety_gates, DEFAULT_SAFETY_GATES)

    def test_corrupted_weights_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_artifact(directory)
            weights_path = manifest_path.parent / "v8-loader-fixture.weights.json"
            weights_path.write_bytes(
                weights_path.read_bytes().replace(b"0.0", b"0.1", 1)
            )
            with self.assertRaises(LearnedPolicyV8Error):
                load_policy_v8(manifest_path)

    def test_contract_violations_are_refused(self) -> None:
        def wrong_format_version(manifest: dict) -> None:
            manifest["format_version"] = 2

        def wrong_schema(manifest: dict) -> None:
            manifest["feature_schema_version"] = 2

        def wrong_input_size(manifest: dict) -> None:
            manifest["input_size"] = 142

        def wrong_names(manifest: dict) -> None:
            manifest["feature_names"] = manifest["feature_names"][:-1]

        def wrong_labels(manifest: dict) -> None:
            manifest["action_labels"] = ["fold", "check_call", "aggress"]

        def wrong_architecture(manifest: dict) -> None:
            manifest["architecture"]["heads"] = {"action_value": 4}

        def missing_key(manifest: dict) -> None:
            del manifest["weights_sha256"]

        mutations = {
            "format_version": wrong_format_version,
            "schema_version": wrong_schema,
            "input_size": wrong_input_size,
            "feature_names": wrong_names,
            "action_labels": wrong_labels,
            "architecture": wrong_architecture,
            "missing_key": missing_key,
        }
        for name, mutate in mutations.items():
            with self.subTest(violation=name):
                with tempfile.TemporaryDirectory() as directory:
                    manifest_path = _write_artifact(
                        directory, mutate_manifest=mutate
                    )
                    with self.assertRaises(LearnedPolicyV8Error):
                        load_policy_v8(manifest_path)

    def test_truncated_normalization_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_artifact(directory, means=[0.0] * 142)
            with self.assertRaises(LearnedPolicyError):
                load_policy_v8(manifest_path)

    def test_malformed_weight_shapes_are_refused(self) -> None:
        weights = _zero_weights()
        weights["heads"]["range"]["out_b"] = [0.0] * 6  # 8 expected
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_artifact(directory, weights=weights)
            with self.assertRaises(LearnedPolicyError):
                load_policy_v8(manifest_path)


# ---------------------------------------------------------------------------
# Torch parity on the trained artifact (CUDA venv subprocess)
# ---------------------------------------------------------------------------


_TORCH_PARITY_SCRIPT = r"""
import json
import sys

import torch
from torch import nn

weights_path, inputs_path, output_path = sys.argv[1:4]
with open(weights_path, encoding="utf-8") as stream:
    weights = json.load(stream)["weights"]
with open(inputs_path, encoding="utf-8") as stream:
    payload = json.load(stream)


def linear(block):
    layer = nn.Linear(len(block["w"][0]), len(block["w"])).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(block["w"], dtype=torch.float64))
        layer.bias.copy_(torch.tensor(block["b"], dtype=torch.float64))
    return layer


def norm(block):
    layer = nn.LayerNorm(len(block["ln_g"])).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(block["ln_g"], dtype=torch.float64))
        layer.bias.copy_(torch.tensor(block["ln_b"], dtype=torch.float64))
    return layer


card_enc, card_ln = linear(weights["card_encoder"]), norm(weights["card_encoder"])
ctx_enc, ctx_ln = (
    linear(weights["context_encoder"]),
    norm(weights["context_encoder"]),
)
trunk = [linear(block) for block in weights["trunk"]]
trunk_ln = [norm(block) for block in weights["trunk"][:-1]]
towers = {
    name: linear({"w": block["tower_w"], "b": block["tower_b"]})
    for name, block in weights["heads"].items()
}
outs = {
    name: linear({"w": block["out_w"], "b": block["out_b"]})
    for name, block in weights["heads"].items()
}

features = torch.tensor(payload["vectors"], dtype=torch.float64)
with torch.no_grad():
    left = card_ln(torch.relu(card_enc(features[:, payload["card_indices"]])))
    right = ctx_ln(torch.relu(ctx_enc(features[:, payload["context_indices"]])))
    hidden = torch.cat([left, right], dim=1)
    for index, layer in enumerate(trunk):
        hidden = torch.relu(layer(hidden))
        if index < len(trunk_ln):
            hidden = trunk_ln[index](hidden)
    outputs = {
        name: outs[name](torch.relu(towers[name](hidden))).tolist()
        for name in outs
    }
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(outputs, stream)
"""


@unittest.skipUnless(
    _CUDA_VENV_PYTHON.is_file() and _TRAINED_MANIFEST.is_file(),
    "CUDA venv interpreter or trained v8 artifact is absent",
)
class TorchParityTests(unittest.TestCase):
    def test_forward_v3_matches_the_torch_network(self) -> None:
        manifest = json.loads(_TRAINED_MANIFEST.read_text(encoding="utf-8"))
        weights_path = _TRAINED_MANIFEST.parent / manifest["weights_file"]
        document = json.loads(weights_path.read_text(encoding="utf-8"))
        normalization = document["feature_normalization"]
        means = [float(value) for value in normalization["means"]]
        stds = [max(1e-6, float(value)) for value in normalization["stds"]]

        rng = random.Random(20260816)
        vectors = []
        for _ in range(20):
            raw = [
                1.0 if rng.random() < 0.05 else 0.0
                for _ in range(schema3.CARD_BLOCK_SIZE)
            ]
            raw.extend(
                rng.uniform(-3.0, 3.0)
                for _ in range(schema3.CONTEXT_BLOCK_SIZE)
            )
            vectors.append(
                [
                    (value - mean) / std
                    for value, mean, std in zip(raw, means, stds)
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "torch_parity.py"
            script.write_text(_TORCH_PARITY_SCRIPT, encoding="utf-8")
            inputs = root / "inputs.json"
            inputs.write_text(
                json.dumps(
                    {
                        "card_indices": list(schema3.CARD_INDICES),
                        "context_indices": list(schema3.CONTEXT_INDICES),
                        "vectors": vectors,
                    }
                ),
                encoding="utf-8",
            )
            outputs_path = root / "outputs.json"
            completed = subprocess.run(
                [
                    str(_CUDA_VENV_PYTHON),
                    str(script),
                    str(weights_path),
                    str(inputs),
                    str(outputs_path),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"torch subprocess failed:\n{completed.stderr}",
            )
            torch_outputs = json.loads(outputs_path.read_text(encoding="utf-8"))

        architecture = manifest["architecture"]
        worst = 0.0
        for row, vector in enumerate(vectors):
            ours = _forward_v3(architecture, document["weights"], vector)
            for name in V8_HEAD_SIZES:
                for ours_value, torch_value in zip(
                    ours[name], torch_outputs[name][row]
                ):
                    worst = max(worst, abs(ours_value - torch_value))
        self.assertLess(worst, 1e-5, f"max per-output disagreement {worst}")

    def test_the_parity_probe_is_not_vacuous(self) -> None:
        # Guard against an all-zero comparison: the trained heads must
        # actually vary across the probe vectors (the residual head is
        # zero by construction in a Phase-A artifact and is exempt).
        manifest = json.loads(_TRAINED_MANIFEST.read_text(encoding="utf-8"))
        weights_path = _TRAINED_MANIFEST.parent / manifest["weights_file"]
        document = json.loads(weights_path.read_text(encoding="utf-8"))
        rng = random.Random(20260816)
        outputs = []
        for _ in range(3):
            vector = [rng.uniform(-1.0, 1.0) for _ in range(schema3.INPUT_SIZE_V8)]
            outputs.append(
                _forward_v3(manifest["architecture"], document["weights"], vector)
            )
        for name in ("fold_through", "range", "equity_called"):
            self.assertNotEqual(outputs[0][name], outputs[1][name])
        for row in outputs:
            self.assertEqual(row["residual"], [0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
