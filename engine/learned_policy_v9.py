"""Serve a format-4 (v9 composed-value) artifact. Layer 2 of the restructure.

Additive sibling of ``learned_policy_v8`` — the v8 path is untouched.
Plan: ``.handoff/notes/V9_RESTRUCTURE_PLAN.md`` L2; contract:
``engine/branch_contract_v9.py``; sizing: ``engine/aggression_sizing`` +
``engine/rules`` (design record ``engine/rules/README.md``).

What is new versus the v8 serve path, exactly:

- **The emitted set is the contract's**, never re-derived here:
  ``legal_branch_labels(available, to_call)`` decides which of
  ``fatal / passive / active / aggressive`` exist, and the composition
  values exactly that set. ``to_call == 0`` emits {passive, active-bet};
  ``to_call > 0`` emits {fatal, active-call, aggressive}.
- **Sizes come from the composed sizing** (``compose_active_wager`` /
  ``compose_aggressive_target``): g's continuous read-driven target with
  the C2/C3A/C5 dials applied when the artifact enables them. This class
  declares ``serves_composed_sizing`` — the base engine's constructor
  guard otherwise refuses those dials, because only this path's sizing
  honours them.
- **The read is g's own**: boldness comes from ``table_boldness`` on the
  depth-invariant temperature fed by the UNCONDITIONED multiway equity
  (``self._equity(table)``, top-fraction 1.0) — the same convention
  ``feature_extract_v9`` fixes, so the features and the sizes describe
  the same read. The engine's range-conditioned equity stays what the
  GATES consume; the two are different quantities on purpose.
- **Value arithmetic per branch** (all purse-normalized, residual capped
  at ±cap·pot and consulted on the WAGER-making executions only — the
  v8 discipline: the correction belongs to sized aggression, and the
  other residual slots stay dark until deliberately enabled)::

      fatal        = 0
      passive      = eq_passive · pot/purse
      active(call) = eq_active · (pot + to_call)/purse − to_call/purse
                     (no fold-through: a call closes the action and buys
                     no folds — contract rule)
      active(bet)  = p_ft · pot/purse
                     + (1 − p_ft)(eq_active · pot_if_called/purse − w/purse)
      aggressive   = same fold-through form over its own slots
      pot_if_called = pot + 2·w − to_call

  ``w`` is the composed wager clamped into the STATED legal range — the
  v8 feature-time approximation kept verbatim (no big-blind floor, no
  integer rounding; ``_sized_action`` alone legalizes the real wager).
  The pot fraction handed to the engine derives from the UNCLAMPED
  composed target, so the engine realizes it with every clamp intact
  (the E6 discipline).
- **Projection, not vocabulary**: the argmax picks a v9 branch; the
  engine receives ``branch_engine_family(branch, to_call)`` — the frozen
  ``fold``/``check_call``/``aggress`` trio. Argmax ties break toward the
  earlier slot in ``BRANCH_LABELS_V9`` (``fatal`` first: conservative).

The loader refuses, fail-loud: any format but 4, any schema but 4 (412
inputs, the exact ``schema4`` name list), any labels but
``BRANCH_LABELS_V9``, a missing or foreign ``sizing`` block (the
composed record carries g's identity and every dial state — an artifact
that cannot state its sizing cannot be served), a v8 architecture, a
tampered weights checksum — and any ``serve.margin_quantiles`` block:
v9 defines no hybrid mode, and 4-branch quantiles describing a 4-way
margin distribution must never be inherited by this contract.

Failure posture and determinism follow ``learned_policy_v8`` exactly:
load fails loud; the serve path fails CLOSED to the heuristic
``DecisionEngine._equity_family``; identical snapshots give identical
decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bluff import BluffSettings

from engine import schema4
from engine.aggression_sizing import SizingParameters, table_boldness
from engine.belief_provider import BeliefProvider
from engine.branch_contract_v9 import (
    BRANCH_LABELS_V9,
    EQUITY_SLOTS_V9,
    FOLD_THROUGH_BRANCHES_V9,
    MODEL_FORMAT_VERSION_V9,
    branch_engine_family,
    legal_branch_labels,
)
from engine.decision_engine import (
    DecisionEngine,
    SafetyGates,
    SharedEquityCache,
    TemperatureShaping,
)
from engine.feature_extract_v9 import (
    _covered_allin_to_amounts,
    extract_features_v9,
)
from engine.game_state import (
    _hero_and_seats,
    _integer,
    effective_stack_chips,
)
from engine.hand_strength import prewarm
from engine.learned_policy import (
    DEFAULT_SERVE_EQUITY_TRIALS,
    LearnedPolicyError,
    _load_engine_parameters,
)
from engine.learned_policy_v8 import (
    RESIDUAL_CAP_POT_FRACTION,
    _clip01,
    _forward_v3,
    _validate_v8_weights,
)
from engine.learning_contract import MODEL_FORMAT
from engine.offline_trainer import _sigmoid
from engine.opponent_model import AggressionTracker
from engine.rules.composition import (
    ComposedWager,
    RuleLayerParams,
    compose_active_wager,
    compose_aggressive_target,
    parameters_and_rules_from_record,
)
from engine.rules.ruin_damper import table_exposure
from engine.v8_trainer import validate_v9_architecture


class LearnedPolicyV9Error(LearnedPolicyError):
    """Raised when a format-4 artifact cannot be loaded safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LearnedPolicyV9Error(message)


def _clamped_wager(
    to_amount: float, contribution: int, lane_range: tuple[int, int] | None
) -> float:
    """The wager a stated legal range admits — the v8 value approximation."""

    if lane_range is None:
        return max(0.0, to_amount - contribution)
    low, high = lane_range
    return max(0.0, min(high, max(low, to_amount)) - contribution)


def _lane_range(allowed: Mapping[str, Any], key: str) -> tuple[int, int] | None:
    raw = allowed.get(key)
    if raw is None or not isinstance(raw, Mapping):
        return None
    try:
        low = _integer(raw.get("min"), f"{key}.min")
        high = _integer(raw.get("max"), f"{key}.max")
    except Exception:  # noqa: BLE001 — a malformed range is an absent range
        return None
    if high < low:
        return None
    return low, high


def compose_branch_values_v9(
    head_outputs: Mapping[str, Sequence[float]],
    *,
    pot: int,
    to_call: int,
    contribution: int,
    effective_stack: int,
    purse: int,
    boldness: float,
    street: str,
    bankroll: int,
    exposure: int,
    covered_allin_to_amounts: Sequence[int],
    legal_labels: frozenset[str],
    bet_range: tuple[int, int] | None,
    raise_range: tuple[int, int] | None,
    sizing: SizingParameters,
    rules: RuleLayerParams,
    use_residual: bool = True,
    residual_cap_pot_fraction: float = RESIDUAL_CAP_POT_FRACTION,
) -> tuple[dict[str, float], dict[str, ComposedWager]]:
    """Composed (uncentered) values for exactly the contract's legal set.

    Every argument is a raw quantity a corpus row can record — the
    Phase-B parity replay reconstructs this call from the stored context
    (``boldness`` decodes from the row's ``10·T`` int through the
    artifact's own sizing parameters) and must reproduce these floats.

    Returns ``(values, wagers)``: values keyed by branch label for the
    legal set only, and the raw :class:`ComposedWager` per wager-making
    execution (``active`` at a free spot, ``aggressive``) so the caller
    can derive the engine's pot fraction from the UNCLAMPED target.
    """

    purse_chips = max(1, int(purse))
    pot_unit = pot / purse_chips
    fold_through = head_outputs["fold_through"]
    equity_called = head_outputs["equity_called"]
    residual = head_outputs["residual"]
    cap = abs(float(residual_cap_pot_fraction)) * pot_unit

    def corrected(branch: str, value: float) -> float:
        if not use_residual:
            return value
        correction = float(residual[BRANCH_LABELS_V9.index(branch)])
        return value + min(cap, max(-cap, correction))

    values: dict[str, float] = {}
    wagers: dict[str, ComposedWager] = {}

    if "fatal" in legal_labels:
        values["fatal"] = 0.0
    if "passive" in legal_labels:
        eq = _clip01(float(equity_called[EQUITY_SLOTS_V9.index("passive")]))
        values["passive"] = eq * pot_unit
    if "active" in legal_labels:
        eq = _clip01(float(equity_called[EQUITY_SLOTS_V9.index("active")]))
        if to_call > 0:
            # A call buys no folds: no fold-through, no residual — the
            # correction belongs to sized aggression (v8 discipline).
            values["active"] = (
                eq * (pot + to_call) / purse_chips - to_call / purse_chips
            )
        else:
            composed = compose_active_wager(
                boldness=boldness,
                pot=pot,
                effective_stack=effective_stack,
                contribution=contribution,
                street=street,
                bankroll=bankroll,
                exposure=exposure,
                covered_allin_to_amounts=covered_allin_to_amounts,
                sizing=sizing,
                geometric=rules.geometric,
                snap=rules.snap,
                damper=rules.damper,
            )
            wagers["active"] = composed
            wager = _clamped_wager(composed.to_amount, contribution, bet_range)
            p_ft = _sigmoid(
                float(fold_through[FOLD_THROUGH_BRANCHES_V9.index("active")])
            )
            pot_if_called = pot + 2.0 * wager
            values["active"] = corrected(
                "active",
                p_ft * pot_unit
                + (1.0 - p_ft)
                * (eq * pot_if_called / purse_chips - wager / purse_chips),
            )
    if "aggressive" in legal_labels:
        composed = compose_aggressive_target(
            boldness=boldness,
            pot=pot,
            to_call=to_call,
            effective_stack=effective_stack,
            contribution=contribution,
            street=street,
            bankroll=bankroll,
            exposure=exposure,
            covered_allin_to_amounts=covered_allin_to_amounts,
            sizing=sizing,
            geometric=rules.geometric,
            snap=rules.snap,
            damper=rules.damper,
        )
        wagers["aggressive"] = composed
        wager = _clamped_wager(composed.to_amount, contribution, raise_range)
        eq = _clip01(float(equity_called[EQUITY_SLOTS_V9.index("aggressive")]))
        p_ft = _sigmoid(
            float(fold_through[FOLD_THROUGH_BRANCHES_V9.index("aggressive")])
        )
        pot_if_called = pot + 2.0 * wager - to_call
        values["aggressive"] = corrected(
            "aggressive",
            p_ft * pot_unit
            + (1.0 - p_ft)
            * (eq * pot_if_called / purse_chips - wager / purse_chips),
        )
    return values, wagers


class LearnedPokerPolicyV9(DecisionEngine):
    """Serve a format-4 checkpoint through the composed-value contract.

    The heads are components; the fixed arithmetic above composes them
    per CONTRACT-legal branch, the argmax runs over exactly that set, and
    the chosen branch reaches the engine as its projected family with the
    composed pot fraction pinned for wager-making executions. Every hard
    safety gate, the temperature shaping, the tracker and the bluff path
    stay exactly as in the heuristic engine, which remains the fallback.
    """

    # This class's sizing routes through engine.rules.composition, so the
    # C2/C3A dials are honoured here — the base engine's guard defers.
    serves_composed_sizing = True

    def __init__(
        self,
        *,
        model_version: str,
        architecture: Mapping[str, Any],
        weights: Mapping[str, Any],
        normalization: Mapping[str, Sequence[float]],
        sizing_record: Mapping[str, Any],
        serve: Mapping[str, Any] | None = None,
        equity_trials: int = 200,
        seed: int = 7,
        temperature_shaping: TemperatureShaping | None = None,
        safety_gates: SafetyGates | None = None,
        opponent_tracker: AggressionTracker | None = None,
        bluff_settings: BluffSettings | None = None,
        equity_cache: SharedEquityCache | None = None,
        hyper_aggression_chance: float | None = None,
        belief_provider: BeliefProvider | None = None,
        potential_trials: int = 400,
        feature_seed: int = 7,
        use_residual: bool = True,
        residual_cap_pot_fraction: float = RESIDUAL_CAP_POT_FRACTION,
    ) -> None:
        try:
            sizing, rules = parameters_and_rules_from_record(sizing_record)
        except ValueError as error:
            raise LearnedPolicyV9Error(f"invalid sizing record: {error}") from error
        if hyper_aggression_chance is None:
            # OWNER DECISION 2026-08-30: the v9 line ships with the
            # hyper-aggression roll OFF — the bluff mixer already provides
            # deliberate salted unpredictability, and the roll's decisions
            # were training-excluded noise. Scoped to v9 ON PURPOSE: the
            # module constant (2%) survives for the v7/v8 paths because
            # every frozen instrument's per-seed numbers were produced
            # with it and record no chance of their own — zeroing it
            # globally would break the reproduction gates repo-wide.
            # Explicit override still works (measuring the noise's price).
            hyper_aggression_chance = 0.0
        super().__init__(
            equity_trials=equity_trials,
            seed=seed,
            temperature_shaping=temperature_shaping,
            safety_gates=safety_gates,
            opponent_tracker=opponent_tracker,
            bluff_settings=bluff_settings,
            equity_cache=equity_cache,
            hyper_aggression_chance=hyper_aggression_chance,
            rule_layer=rules,
        )
        self.policy_version = str(model_version)
        try:
            validate_v9_architecture(architecture)
        except ValueError as error:
            raise LearnedPolicyV9Error(f"invalid v9 architecture: {error}") from error
        self._architecture = dict(architecture)
        self._weights = _validate_v8_weights(weights, architecture)
        self._means = [float(value) for value in normalization["means"]]
        self._stds = [max(1e-6, float(value)) for value in normalization["stds"]]
        _require(
            len(self._means) == schema4.INPUT_SIZE_V9
            and len(self._stds) == schema4.INPUT_SIZE_V9,
            "feature normalization must cover the full schema-4 vector",
        )
        _require(
            not (serve or {}).get("margin_quantiles"),
            "v9 defines no hybrid mode: margin_quantiles must not be present",
        )
        _require(
            math.isfinite(float(residual_cap_pot_fraction))
            and float(residual_cap_pot_fraction) >= 0.0,
            "residual_cap_pot_fraction must be finite and non-negative",
        )
        _require(potential_trials >= 1, "potential_trials must be positive")
        self._sizing_params: SizingParameters = sizing
        if belief_provider is None:
            # OWNER DECISION 2026-08-30 (block 8 wired): the v9 line
            # serves the FITTED P3 belief provider by default — the eight
            # bucket inputs were constant 0.125 on every vector ever
            # produced before this. Loading fails LOUD if the fit
            # artifact is missing: serving with buckets the corpus was
            # not trained on is the exact defect the wire-in closes.
            # Explicit NeutralBeliefProvider() remains the test override.
            from engine.p3_belief_provider import P3BeliefProvider

            try:
                belief_provider = P3BeliefProvider.from_artifact()
            except Exception as error:  # noqa: BLE001 — refuse, never degrade
                raise LearnedPolicyV9Error(
                    f"cannot load the P3 belief fit: {error}"
                ) from error
        self._belief_provider = belief_provider
        self._potential_trials = int(potential_trials)
        self._feature_seed = int(feature_seed)
        self._use_residual = bool(use_residual)
        self._residual_cap_pot_fraction = float(residual_cap_pot_fraction)
        self._branch_pot_fraction: float | None = None

    def _family(self, features: tuple[float, ...]) -> str:
        del features
        return "check_call"

    def _composed_decision(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
    ) -> tuple[dict[str, float], dict[str, ComposedWager], int]:
        """Extract, forward, compose over the contract's legal set."""

        vector = extract_features_v9(
            table,
            belief_provider=self._belief_provider,
            potential_trials=self._potential_trials,
            seed=self._feature_seed,
            sizing=self._sizing_params,
            rules=self.rule_layer,
        )
        normalized = tuple(
            (value - mean) / std
            for value, mean, std in zip(vector, self._means, self._stds)
        )
        outputs = _forward_v3(self._architecture, self._weights, normalized)
        hero, seats = _hero_and_seats(table)
        pot = _integer(table.get("potChips"), "potChips")
        to_call = _integer(allowed.get("callChips", 0), "callChips")
        contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
        stack = _integer(hero.get("stackChips"), "hero stackChips", minimum=1)
        total_committed = _integer(
            hero.get("totalCommittedChips", contribution), "hero totalCommittedChips"
        )
        # g's own read: UNCONDITIONED multiway equity through the
        # depth-invariant table temperature — the extractor's convention,
        # so the cost features and the served sizes describe one read.
        read_equity = self._equity(table)
        boldness = (
            table_boldness(table, allowed, read_equity, self._sizing_params)
            if read_equity is not None
            else 0.0
        )
        values, wagers = compose_branch_values_v9(
            outputs,
            pot=pot,
            to_call=to_call,
            contribution=contribution,
            effective_stack=effective_stack_chips(table),
            purse=stack + total_committed,
            boldness=boldness,
            street=str(table.get("street") or "").casefold(),
            bankroll=stack,
            exposure=table_exposure(table),
            covered_allin_to_amounts=_covered_allin_to_amounts(hero, seats),
            legal_labels=legal_branch_labels(available, to_call),
            bet_range=_lane_range(allowed, "betRange")
            or _lane_range(allowed, "raiseRange"),
            raise_range=_lane_range(allowed, "raiseRange"),
            sizing=self._sizing_params,
            rules=self.rule_layer,
            use_residual=self._use_residual,
            residual_cap_pot_fraction=self._residual_cap_pot_fraction,
        )
        if not all(math.isfinite(value) for value in values.values()):
            raise LearnedPolicyV9Error("composed branch values are not finite")
        return values, wagers, to_call

    def _equity_family(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float,
        features: tuple[float, ...] | None = None,
    ) -> str:
        self._branch_pot_fraction = None
        if equity is None:
            return super()._equity_family(table, allowed, available, equity)
        try:
            values, wagers, to_call = self._composed_decision(
                table, allowed, available
            )
        except Exception:
            # Fail closed: any malformed snapshot, feature, or artifact
            # behaviour degrades to the heuristic engine, never a guess.
            return super()._equity_family(table, allowed, available, equity)
        if not values:
            return super()._equity_family(table, allowed, available, equity)
        # Centered within the decision; centering never moves an argmax.
        offset = sum(values.values()) / len(values)
        centered = {branch: value - offset for branch, value in values.items()}
        # Slot order breaks ties toward the earlier branch (fatal first).
        best = max(
            (branch for branch in BRANCH_LABELS_V9 if branch in centered),
            key=lambda branch: centered[branch],
        )
        family = branch_engine_family(best, to_call)
        if family == "aggress" and best in wagers:
            composed = wagers[best]
            # E6 discipline: the fraction derives from the UNCLAMPED
            # composed target, so the engine's own sizing path realizes
            # it with every clamp intact.
            if best == "aggressive":
                self._branch_pot_fraction = (composed.target - to_call) / max(
                    1, _integer(table.get("potChips"), "potChips") + to_call
                )
            else:  # active executing as a bet
                self._branch_pot_fraction = composed.target / max(
                    1, _integer(table.get("potChips"), "potChips")
                )
        return family

    def _sized_action(
        self,
        action: str,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float | None,
        pot_fraction: float | None = None,
    ):
        if pot_fraction is None and self._branch_pot_fraction is not None:
            pot_fraction = self._branch_pot_fraction
        return super()._sized_action(
            action, table, allowed, equity, pot_fraction=pot_fraction
        )


_REQUIRED_MANIFEST_KEYS = (
    "format",
    "format_version",
    "model_version",
    "feature_schema_version",
    "input_size",
    "feature_names",
    "action_labels",
    "architecture",
    "sizing",
    "weights_file",
    "weights_sha256",
)


def load_policy_v9(
    manifest_path: str | Path,
    *,
    equity_trials: int | None = None,
    equity_cache: SharedEquityCache | None = None,
    hyper_aggression_chance: float | None = None,
    belief_provider: BeliefProvider | None = None,
    potential_trials: int = 400,
    feature_seed: int = 7,
    use_residual: bool = True,
    residual_cap_pot_fraction: float = RESIDUAL_CAP_POT_FRACTION,
) -> LearnedPokerPolicyV9:
    """Load, verify, and assemble a v9 playing policy from one manifest.

    Fail-loud before anything is served; mirrors the v8 loader's checks
    at the v9 constants, plus the sizing record and the hybrid refusal.
    ``equity_trials`` follows ``load_policy``'s precedence: an explicit
    argument wins, then the manifest's ``serve.equity_trials`` pin (the
    v9 Phase-B trainer records the corpus header's value there — harvest
    == serve, one number), then the module default. Loading neither
    promotes nor deploys anything, and ``artifacts/approved.json`` is
    never read or written here.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LearnedPolicyV9Error(f"cannot read manifest: {error}") from error
    _require(isinstance(manifest, Mapping), "manifest must be an object")
    for key in _REQUIRED_MANIFEST_KEYS:
        _require(key in manifest, f"manifest is missing {key!r}")
    _require(
        manifest["format"] == MODEL_FORMAT,
        f"manifest format must be {MODEL_FORMAT!r}",
    )
    _require(
        manifest["format_version"] == MODEL_FORMAT_VERSION_V9,
        f"format_version must be {MODEL_FORMAT_VERSION_V9}",
    )
    _require(
        manifest["feature_schema_version"] == schema4.SCHEMA_VERSION_V9,
        f"feature_schema_version must be {schema4.SCHEMA_VERSION_V9}",
    )
    _require(
        manifest["input_size"] == schema4.INPUT_SIZE_V9,
        f"input_size must be {schema4.INPUT_SIZE_V9}",
    )
    _require(
        list(manifest["feature_names"]) == list(schema4.FEATURE_NAMES_V9),
        "feature_names must match schema4.FEATURE_NAMES_V9",
    )
    _require(
        list(manifest["action_labels"]) == list(BRANCH_LABELS_V9),
        f"action_labels must be {list(BRANCH_LABELS_V9)}",
    )
    try:
        validate_v9_architecture(manifest["architecture"])
    except ValueError as error:
        raise LearnedPolicyV9Error(f"invalid v9 architecture: {error}") from error

    weights_file = manifest_file.parent / str(manifest["weights_file"])
    try:
        raw = weights_file.read_bytes()
    except OSError as error:
        raise LearnedPolicyV9Error(f"cannot read weights: {error}") from error
    digest = hashlib.sha256(raw.rstrip(b"\n")).hexdigest()
    _require(
        digest == manifest["weights_sha256"],
        "weights checksum does not match the manifest",
    )
    try:
        document = json.loads(raw.decode("utf-8"))
    except ValueError as error:
        raise LearnedPolicyV9Error(f"cannot parse weights: {error}") from error
    _require(
        document.get("format_version") == MODEL_FORMAT_VERSION_V9,
        "weights file format_version must match the manifest",
    )
    _require(
        document.get("model_version") == manifest["model_version"],
        "weights and manifest disagree on model_version",
    )
    normalization = document.get("feature_normalization")
    _require(
        isinstance(normalization, Mapping)
        and "means" in normalization
        and "stds" in normalization,
        "weights file lacks feature normalization",
    )
    serve = manifest.get("serve") or {}
    _require(isinstance(serve, Mapping), "manifest serve block must be an object")
    # Explicit argument wins; then the artifact's own pin; then the
    # module default — load_policy's rule, kept so the harvest's
    # recorded precision is the served precision by default.
    if equity_trials is None:
        pinned = serve.get("equity_trials")
        if pinned is None:
            equity_trials = DEFAULT_SERVE_EQUITY_TRIALS
        else:
            _require(
                isinstance(pinned, int)
                and not isinstance(pinned, bool)
                and pinned > 0,
                "serve.equity_trials must be a positive integer",
            )
            equity_trials = int(pinned)
    prewarm()
    return LearnedPokerPolicyV9(
        model_version=str(manifest["model_version"]),
        architecture=manifest["architecture"],
        weights=document["weights"],
        normalization=normalization,
        sizing_record=manifest["sizing"],
        serve=serve,
        equity_trials=equity_trials,
        equity_cache=equity_cache,
        hyper_aggression_chance=hyper_aggression_chance,
        belief_provider=belief_provider,
        potential_trials=potential_trials,
        feature_seed=feature_seed,
        use_residual=use_residual,
        residual_cap_pot_fraction=residual_cap_pot_fraction,
        **_load_engine_parameters(manifest),
    )


__all__ = [
    "LearnedPokerPolicyV9",
    "LearnedPolicyV9Error",
    "compose_branch_values_v9",
    "load_policy_v9",
]
