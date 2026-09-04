# cython: language_level=3str
"""Chipzen bot: the 0Fold heuristic-aggressive-v6 multiway policy.

``decide()`` maps the Chipzen ``GameState`` onto the Arena-shaped table
snapshot the engine consumes, runs the dependency-free aggressive policy,
and maps the engine payload back onto the SDK Action API (fold / check /
call / raise_to(total)). Raise amounts are TOTAL bets on both sides, so the
amount crosses unchanged; shoves become ``raise_to(max_raise)`` because the
wire protocol has no all_in action.

The policy is built lazily on the first decision so container startup stays
inside the attach budget, and every return path emits a legal action read
from ``state.valid_actions``.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from chipzen import Action, Bot, GameState  # noqa: E402  (needs the sys.path insert above)

KNOWN_ACTIONS = frozenset({"fold", "check", "call", "raise"})

_policy = None
_init_error = ""


def _ensure_policy() -> None:
    # The `_inert_weights()` zero network that used to be built here is gone
    # with P2 (2026-09-04): the policy carries no network at all now, so there
    # is nothing to satisfy. It existed only because the constructor demanded
    # weights while this image never shipped `artifacts/` for it to read.
    global _policy, _init_error
    if _policy is not None or _init_error:
        return
    try:
        from engine.hand_strength import prewarm
        from engine.poker_policy import AggressivePokerPolicy

        prewarm()
        _policy = AggressivePokerPolicy(equity_trials=200)
    except Exception as exc:
        _init_error = repr(exc)


def _seat_stacks(state: GameState) -> dict:
    stacks: dict = {}
    opponent_index = 0
    num = len(state.opponent_stacks) + 1
    for seat in range(num):
        if seat == state.your_seat:
            stacks[seat] = int(state.your_stack)
        else:
            stacks[seat] = int(state.opponent_stacks[opponent_index])
            opponent_index += 1
    return stacks


def _street_contributions(state: GameState) -> tuple:
    contributions: dict = {}
    folded: set = set()
    blinds: dict = {}
    for entry in state.action_history:
        seat = entry.get("seat")
        if seat is None:
            continue
        action = str(entry.get("action") or "")
        amount = int(entry.get("amount") or 0)
        if action == "fold":
            folded.add(int(seat))
        elif action in ("post_small_blind", "post_big_blind"):
            blinds[action] = max(int(blinds.get(action, 0)), amount)
            if entry.get("phase") == state.phase:
                contributions[seat] = max(int(contributions.get(seat, 0)), amount)
        elif action in ("call", "raise") and entry.get("phase") == state.phase:
            contributions[seat] = max(int(contributions.get(seat, 0)), amount)
    return contributions, folded, blinds


def _to_table(state: GameState) -> dict:
    stacks = _seat_stacks(state)
    contributions, folded, blinds = _street_contributions(state)
    hero_street = int(contributions.get(state.your_seat, 0))
    to_call = int(state.to_call)
    num = len(state.opponent_stacks) + 1
    seats = []
    for seat in range(num):
        entry = {
            "seatNumber": seat + 1,
            "status": "Folded" if seat in folded else "Active",
            "stackChips": int(stacks.get(seat, 0)),
            "currentBetChips": int(contributions.get(seat, 0)),
        }
        if seat == state.your_seat:
            entry["holeCards"] = [str(card) for card in state.hole_cards]
        else:
            entry["holeCards"] = None
        seats.append(entry)

    valid = list(state.valid_actions)
    can_raise = "raise" in valid
    min_raise = int(state.min_raise)
    max_raise = int(state.max_raise)
    available = list(valid)
    if can_raise:
        if to_call == 0:
            available.append("bet")
        available.append("all-in")
    allowed = {
        "canFold": "fold" in valid,
        "canCheck": "check" in valid,
        "canCall": "call" in valid,
        "canBet": can_raise and to_call == 0,
        "canRaise": can_raise and to_call > 0,
        "callChips": to_call,
        "callAmount": to_call,
        "callToAmount": to_call,
        "minBet": min_raise if to_call == 0 else None,
        "minRaiseTo": min_raise,
        "betRange": {"min": max(min_raise, 1), "max": max_raise}
        if can_raise
        else None,
        "raiseRange": {"min": max(min_raise, 1), "max": max_raise}
        if can_raise
        else None,
        "allInToAmount": max_raise if can_raise else None,
        "amountSemantics": "toAmount",
        "reasoningRequired": False,
        "availableActions": available,
    }

    events = []
    for entry in state.action_history:
        seat = entry.get("seat")
        if seat is None:
            continue
        action = str(entry.get("action") or "")
        amount = int(entry.get("amount") or 0)
        phase = entry.get("phase") or state.phase
        if action in ("post_small_blind", "post_big_blind"):
            events.append(
                {
                    "type": "BlindPosted",
                    "summary": {"seatNumber": int(seat) + 1, "amount": amount},
                }
            )
        elif action in ("fold", "check", "call", "raise"):
            events.append(
                {
                    "type": "ActionTaken",
                    "street": phase,
                    "summary": {
                        "action": action,
                        "seatNumber": int(seat) + 1,
                        "amount": amount,
                    },
                }
            )

    small_blind = int(blinds.get("post_small_blind", 1))
    big_blind = int(blinds.get("post_big_blind", max(1, small_blind)))
    return {
        "id": state.round_id or "chipzen",
        "tableId": state.round_id or "chipzen",
        "street": state.phase,
        "potChips": int(state.pot),
        "currentBet": hero_street + to_call,
        "boardCards": [str(card) for card in state.board],
        "smallBlindChips": small_blind,
        "bigBlindChips": big_blind,
        "selfSeatNumber": int(state.your_seat) + 1,
        "seats": seats,
        "recentEvents": events,
        "allowedActions": allowed,
    }


def _fallback_action(state: GameState) -> Action:
    valid = list(state.valid_actions)
    for preferred in ("check", "call", "fold"):
        if preferred in valid:
            return getattr(Action, preferred)()
    return Action.fold()


def _to_action(payload: dict, state: GameState) -> Action:
    valid = list(state.valid_actions)
    action = payload.get("action")
    if action == "fold" and "fold" in valid:
        return Action.fold()
    if action == "check" and "check" in valid:
        return Action.check()
    if action == "call" and "call" in valid:
        return Action.call()
    if action in ("bet", "raise") and "raise" in valid:
        try:
            size = int(payload.get("amount") or 0)
        except (TypeError, ValueError):
            size = 0
        if size <= 0:
            size = int(state.min_raise)
        size = max(min(size, int(state.max_raise)), max(int(state.min_raise), 1))
        return Action.raise_to(size)
    if action == "all-in":
        if "raise" in valid:
            return Action.raise_to(int(state.max_raise))
        if "call" in valid:
            return Action.call()
    return _fallback_action(state)


class MyBot(Bot):
    """The heuristic-aggressive-v6 engine behind the Chipzen adapter."""

    def __init__(self) -> None:
        self._fallback_note = ""
        self._reported_unknown = frozenset()

    def decide(self, state: GameState) -> Action:
        self._fallback_note = ""
        unknown = frozenset(set(state.valid_actions) - KNOWN_ACTIONS)
        if unknown:
            if unknown != self._reported_unknown:
                self._reported_unknown = unknown
                print(
                    f"note: actions not implemented: {sorted(unknown)} "
                    f"(phase={state.phase!r}); safe fallback",
                    file=sys.stderr,
                )
            self._fallback_note = "playing it safe"
            return _fallback_action(state)

        _ensure_policy()
        if _policy is None:
            self._fallback_note = "playing it safe"
            print(
                f"policy unavailable ({_init_error!r:.120}); safe fallback",
                file=sys.stderr,
            )
            return _fallback_action(state)

        try:
            table = _to_table(state)
            decision = _policy.decide_with_diagnostics(table, deadline_s=4.0)
            payload = decision.to_payload()
        except Exception as exc:
            self._fallback_note = "keeping it simple here"
            print(
                f"decide error ({exc!r:.120}); safe fallback", file=sys.stderr
            )
            return _fallback_action(state)

        result = _to_action(payload, state)
        if result.action not in state.valid_actions:
            self._fallback_note = "adjusting to the legal set"
            print(
                f"mapped {payload.get('action')!r} is not legal here; "
                "safe fallback",
                file=sys.stderr,
            )
            return _fallback_action(state)
        return result


def main() -> None:
    """Entry point -- invoked by the Dockerfile ENTRYPOINT.

    The Chipzen platform injects ``CHIPZEN_WS_URL`` and ``CHIPZEN_TOKEN``
    (or ``CHIPZEN_TICKET``) at container launch time. For local testing
    against your own stack, set them yourself or pass the URL as the
    first positional argument.
    """
    import asyncio

    from chipzen.client import run_bot

    url = os.environ.get("CHIPZEN_WS_URL") or (
        sys.argv[1] if len(sys.argv) > 1 else None
    )
    if not url:
        print(
            "error: CHIPZEN_WS_URL not set and no URL passed on the command line",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(
        run_bot(
            url,
            MyBot(),
            token=os.environ.get("CHIPZEN_TOKEN"),
            ticket=os.environ.get("CHIPZEN_TICKET"),
        )
    )


if __name__ == "__main__":
    main()
