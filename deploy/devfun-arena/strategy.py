"""dev.fun Arena static-agent adapter for the heuristic v5 multiway policy.

The Arena sandbox runs this file and calls ``choose_action(table)`` (or
``act(table)``) once per decision, handing it the live Playground table
snapshot -- exactly the contract the decision engine already consumes. This
adapter forwards the table through ``AggressivePokerPolicy`` and maps the
resulting payload onto the sandbox's ``{action, amount, reasoning_text}``
return shape.

Robust by construction: the policy is built inside a guard so a module-level
failure can never fault the agent, the required proposal-network weights are
an inert contract-matching zero network built programmatically (the sandbox
blocks filesystem reads, and the deterministic v5 equity rules drive every
decision anyway because no ``table_sizes`` are declared), and every return
path emits a legal action read from the live table.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Fallback reasoning strings double as markers the build's smoke test greps
# for, so a silently degraded bundle can never pass the local gate.
_FALLBACK_NOTES = (
    "playing it safe",
    "keeping it simple here",
    "adjusting to the legal set",
    "sizing fallback",
)

_policy = None
_safest_passive = None
_init_error = ""
try:
    from engine.decision_engine import safest_passive_action
    from engine.hand_strength import prewarm
    from engine.poker_policy import AggressivePokerPolicy
    from engine.policy_features import FEATURE_NAMES, LABELS

    def _inert_weights() -> dict:
        """A valid zero network; never consulted without ``table_sizes``."""
        inputs = len(FEATURE_NAMES)
        return {
            "input_size": inputs,
            "hidden_size": 1,
            "feature_names": list(FEATURE_NAMES),
            "labels": list(LABELS),
            "w1": [[0.0] * inputs],
            "b1": [0.0],
            "w2": [[0.0] for _ in LABELS],
            "b2": [0.0 for _ in LABELS],
        }

    prewarm()
    _safest_passive = safest_passive_action
    _policy = AggressivePokerPolicy(weights=_inert_weights(), equity_trials=200)
except Exception as exc:  # keep the agent importable no matter what
    _init_error = repr(exc)


def _legal_actions(table):
    allowed = (table.get("allowedActions") if hasattr(table, "get") else None) or {}
    actions = [str(a) for a in (allowed.get("availableActions") or [])]
    if not actions:
        for name, flag in (
            ("check", "canCheck"),
            ("fold", "canFold"),
            ("call", "canCall"),
            ("bet", "canBet"),
            ("raise", "canRaise"),
            ("all-in", "canAllIn"),
        ):
            if allowed.get(flag):
                actions.append(name)
    return actions


def _fallback(table, note):
    actions = _legal_actions(table)
    if _safest_passive is not None:
        preferred = _safest_passive(actions)
        if preferred is not None:
            return {"action": preferred, "reasoning_text": note}
    for preferred in ("check", "call", "fold"):
        if preferred in actions:
            return {"action": preferred, "reasoning_text": note}
    if actions:
        return {"action": actions[0], "reasoning_text": note}
    return {"action": "fold", "reasoning_text": note}


def choose_action(table):
    if _policy is None:
        return _fallback(table, "playing it safe")
    try:
        payload = _policy.decide_with_diagnostics(table, deadline_s=8.0).to_payload()
    except Exception:
        return _fallback(table, "keeping it simple here")

    action = payload.get("action")
    if action not in _legal_actions(table):
        return _fallback(table, "adjusting to the legal set")

    result = {
        "action": action,
        "reasoning_text": payload.get("message") or "range-aware line",
    }
    amount = payload.get("amount")
    if amount is not None:
        try:
            result["amount"] = int(amount)
        except (TypeError, ValueError):
            return _fallback(table, "sizing fallback")
    return result


# The sandbox contract accepts either name.
act = choose_action
