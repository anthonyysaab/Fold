"""The two deterministic-equity heuristic policies.

The live agent uses this module as its policy entry point whenever no learned
artifact is approved, and the gauntlet uses it as the reference opponent. Both
policies delegate legal actions and sizing to the decision engine; equity
thresholds pick the action family. Live play needs no PyTorch.

This module holds no model and reads nothing from disk. It used to: until P2
(2026-09-04) ``PokerPolicy.__init__`` loaded ``artifacts/tiny-policy-pure.json``
into a 125-input ``FixedPolicyNetwork``. That network never chose an action --
it was gated on a ``table_sizes`` key the shipped export did not declare -- but
the load was a start-up liability (a missing or corrupt artifact killed
``run_agent.py --standard``/``--aggressive`` before the first hand) and the gate
was data, not code, so any export declaring ``table_sizes`` would have armed it
on the live path. The whole path is gone; the checkpoint it derived from is
recorded at `archive/nn-poker-training-2026-09-04/`.

Deleting it changed no decision: `tests/test_heuristic_policy_parity.py` pins
every family, size and feature vector across three seat counts and both
anti-modeling arms against goldens captured before the removal.
"""

from __future__ import annotations

from dataclasses import replace

from engine.decision_engine import (
    DecisionEngine,
    DEFAULT_SAFETY_GATES,
    SharedEquityCache,
)
from engine.hand_strength import prewarm


class PokerPolicy(DecisionEngine):
    """Torch-free standard policy: deterministic equity thresholds throughout.

    Constructed straight from :class:`DecisionEngine` -- there is no override.
    Before P2 this class carried an ``__init__`` that loaded the fixed network
    and otherwise passed every argument through unchanged, and a ``_family``
    that the base class only ever reached at ``equity_trials == 0``. That
    entrance is now closed at the source: :class:`DecisionEngine` refuses to
    construct below 1 trial.
    """

    policy_version = "heuristic-standard-v5"


AGGRESSIVE_SETTINGS = {
    "aggression_base": 0.42,
    "aggression_per_opponent": 0.03,
    "aggression_cap": 0.66,
    "call_margin": {"preflop": -0.02, "flop": 0.0, "turn": 0.02, "river": 0.04},
}

# The aggressive policy keeps a single, later-triggering stack-off gate.
# Softened 30% on 2026-08-12 (trigger x1.30, floor keeps 70% of its excess
# over 0.50); was (0.5, 0.62). Like every SafetyGates field, a future
# learned artifact may replace it.
AGGRESSIVE_SAFETY_GATES = replace(
    DEFAULT_SAFETY_GATES, call_stack_gates=((0.65, 0.584),)
)


class AggressivePokerPolicy(PokerPolicy):
    """Current higher-volume policy option for two-to-six-player tables."""

    # v6 = the escalation-decayed opponent range floor (the 2026-08-13 bust
    # fix). Shipping the change under a new version string, never by
    # mutating v5 in place, is what keeps heuristic-aggressive-v5 the
    # frozen, comparable baseline in the archived gauntlet reports
    # (DECISIONS.md).
    policy_version = "heuristic-aggressive-v6"
    default_safety_gates = AGGRESSIVE_SAFETY_GATES

    def _aggression_floor(self, table, opponent_count):
        settings = AGGRESSIVE_SETTINGS
        return min(
            settings["aggression_cap"],
            settings["aggression_base"]
            + settings["aggression_per_opponent"] * max(0, opponent_count - 1),
        )

    def _call_margin(self, table):
        return AGGRESSIVE_SETTINGS["call_margin"].get(self._street(table), 0.04)


def build_policy(
    *,
    aggressive: bool = False,
    equity_trials: int = 200,
    equity_cache: SharedEquityCache | None = None,
    hyper_aggression_chance: float | None = None,
) -> PokerPolicy:
    """Prepare hand evaluation and build the selected live policy.

    ``equity_cache`` opts into memoized equity estimates (see
    :class:`DecisionEngine`); the harvest passes a fresh dict per policy,
    live construction leaves it ``None``.

    ``hyper_aggression_chance`` overrides the anti-modeling floor; ``None``
    keeps ``HYPER_AGGRESSION_CHANCE``. The harvest passes ``0.0`` because its
    opponents cannot model anyone, live construction leaves it ``None``.

    The ``weights_path`` parameter was removed with P2. It had no caller in
    `engine/`, `tools/`, `tests/` or `deploy/` on the day it was deleted.
    """

    prewarm()
    policy_type = AggressivePokerPolicy if aggressive else PokerPolicy
    return policy_type(
        equity_trials=equity_trials,
        equity_cache=equity_cache,
        hyper_aggression_chance=hyper_aggression_chance,
    )


__all__ = [
    "AGGRESSIVE_SAFETY_GATES",
    "AGGRESSIVE_SETTINGS",
    "AggressivePokerPolicy",
    "build_policy",
    "PokerPolicy",
]
