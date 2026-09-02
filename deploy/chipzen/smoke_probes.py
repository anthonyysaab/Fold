"""Local probes for the Chipzen adapter; not shipped in the image.

Run from the repo root::

    python deploy/chipzen/smoke_probes.py

Every probe must return a legal action with no degraded-mode fallback note,
and no decide() call may exceed the ranked-play budget. Exit 0 only when all
probes pass.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bot import MyBot  # noqa: E402
from chipzen.models import Card, GameState  # noqa: E402

DECIDE_BUDGET_S = 2.0
FALLBACK_NOTES = (
    "playing it safe",
    "keeping it simple here",
    "adjusting to the legal set",
)


def cards(text: str) -> list:
    return [Card.from_str(text[i : i + 2]) for i in range(0, len(text), 2)]


def entry(seat: int, action: str, amount: int, phase: str) -> dict:
    return {
        "seat": seat,
        "action": action,
        "amount": amount,
        "phase": phase,
        "is_timeout": False,
    }


def _state(
    *,
    phase: str,
    hole: str,
    board: str,
    pot: int,
    stack: int,
    opp_stacks: list,
    seat: int,
    dealer: int,
    to_call: int,
    min_raise: int,
    max_raise: int,
    valid: list,
    history: list,
) -> GameState:
    return GameState(
        hand_number=1,
        phase=phase,
        hole_cards=cards(hole),
        board=cards(board) if board else [],
        pot=pot,
        your_stack=stack,
        opponent_stacks=opp_stacks,
        your_seat=seat,
        dealer_seat=dealer,
        to_call=to_call,
        min_raise=min_raise,
        max_raise=max_raise,
        valid_actions=valid,
        action_history=history,
        round_id="probe",
        request_id="probe",
    )


def main() -> int:
    bot = MyBot()
    cases = [
        (
            "K6 boat 777K turn facing raise",
            _state(
                phase="turn",
                hole="Kd6d",
                board="7s7cKs7h",
                pot=4700,
                stack=7990,
                opp_stacks=[5210],
                seat=1,
                dealer=0,
                to_call=1540,
                min_raise=3080,
                max_raise=7990,
                valid=["fold", "call", "raise"],
                history=[
                    entry(0, "post_small_blind", 50, "preflop"),
                    entry(1, "post_big_blind", 100, "preflop"),
                    entry(0, "raise", 300, "preflop"),
                    entry(1, "call", 300, "preflop"),
                    entry(1, "check", 0, "flop"),
                    entry(0, "raise", 400, "flop"),
                    entry(1, "call", 400, "flop"),
                    entry(1, "check", 0, "turn"),
                    entry(0, "raise", 1540, "turn"),
                ],
            ),
            "not-fold",
        ),
        (
            "AKs heads-up preflop facing open",
            _state(
                phase="preflop",
                hole="AsKs",
                board="",
                pot=350,
                stack=9700,
                opp_stacks=[9700],
                seat=1,
                dealer=0,
                to_call=200,
                min_raise=500,
                max_raise=9700,
                valid=["fold", "call", "raise"],
                history=[
                    entry(0, "post_small_blind", 50, "preflop"),
                    entry(1, "post_big_blind", 100, "preflop"),
                    entry(0, "raise", 300, "preflop"),
                ],
            ),
            "not-fold",
        ),
        (
            "72o 4-max preflop facing open",
            _state(
                phase="preflop",
                hole="7c2d",
                board="",
                pot=700,
                stack=9700,
                opp_stacks=[9700, 9700, 9700],
                seat=2,
                dealer=1,
                to_call=250,
                min_raise=500,
                max_raise=9700,
                valid=["fold", "call", "raise"],
                history=[
                    entry(2, "post_small_blind", 50, "preflop"),
                    entry(3, "post_big_blind", 100, "preflop"),
                    entry(0, "raise", 300, "preflop"),
                    entry(1, "call", 300, "preflop"),
                ],
            ),
            "observe",
        ),
        (
            "A3c 777K turn facing raise",
            _state(
                phase="turn",
                hole="Ac3c",
                board="7s7cKs7h",
                pot=4700,
                stack=7990,
                opp_stacks=[5210],
                seat=1,
                dealer=0,
                to_call=1540,
                min_raise=3080,
                max_raise=7990,
                valid=["fold", "call", "raise"],
                history=[
                    entry(0, "post_small_blind", 50, "preflop"),
                    entry(1, "post_big_blind", 100, "preflop"),
                    entry(0, "raise", 300, "preflop"),
                    entry(1, "call", 300, "preflop"),
                    entry(1, "check", 0, "flop"),
                    entry(0, "raise", 400, "flop"),
                    entry(1, "call", 400, "flop"),
                    entry(1, "check", 0, "turn"),
                    entry(0, "raise", 1540, "turn"),
                ],
            ),
            "observe",
        ),
        (
            "K4s 7766K river facing lead",
            _state(
                phase="river",
                hole="4sKs",
                board="7h6s7s6dKh",
                pot=1600,
                stack=9600,
                opp_stacks=[4800],
                seat=1,
                dealer=0,
                to_call=800,
                min_raise=1600,
                max_raise=9600,
                valid=["fold", "call", "raise"],
                history=[
                    entry(0, "post_small_blind", 50, "preflop"),
                    entry(1, "post_big_blind", 100, "preflop"),
                    entry(0, "call", 100, "preflop"),
                    entry(1, "check", 0, "preflop"),
                    entry(1, "check", 0, "flop"),
                    entry(0, "check", 0, "flop"),
                    entry(1, "check", 0, "turn"),
                    entry(0, "check", 0, "turn"),
                    entry(1, "check", 0, "river"),
                    entry(0, "raise", 800, "river"),
                ],
            ),
            "observe",
        ),
        (
            "3-handed flop button check-or-bet",
            _state(
                phase="flop",
                hole="AsQd",
                board="Ts7h2d",
                pot=300,
                stack=9800,
                opp_stacks=[9750, 9700],
                seat=2,
                dealer=2,
                to_call=0,
                min_raise=100,
                max_raise=9800,
                valid=["check", "raise"],
                history=[
                    entry(0, "post_small_blind", 50, "preflop"),
                    entry(1, "post_big_blind", 100, "preflop"),
                    entry(2, "call", 100, "preflop"),
                    entry(0, "call", 50, "preflop"),
                    entry(1, "check", 0, "preflop"),
                    entry(1, "check", 0, "flop"),
                    entry(0, "check", 0, "flop"),
                ],
            ),
            "observe",
        ),
        (
            "AA preflop re-raise spot",
            _state(
                phase="preflop",
                hole="AdAc",
                board="",
                pot=650,
                stack=1950,
                opp_stacks=[9400],
                seat=1,
                dealer=0,
                to_call=500,
                min_raise=1100,
                max_raise=1950,
                valid=["fold", "call", "raise"],
                history=[
                    entry(0, "post_small_blind", 50, "preflop"),
                    entry(1, "post_big_blind", 100, "preflop"),
                    entry(0, "raise", 600, "preflop"),
                ],
            ),
            "observe",
        ),
    ]
    ok = True
    for name, state, want in cases:
        worst = 0.0
        result = None
        for _ in range(3):
            started = time.perf_counter()
            result = bot.decide(state)
            worst = max(worst, time.perf_counter() - started)
        action = result.action
        note = bot._fallback_note
        legal = action in state.valid_actions
        passed = legal and not note and isinstance(result.action, str)
        if want == "not-fold":
            passed = passed and action != "fold"
        if action == "raise":
            passed = passed and state.min_raise <= result.amount <= state.max_raise
        passed = passed and worst < DECIDE_BUDGET_S
        ok = ok and passed
        shown = f"{action}" + (
            f" to {result.amount}" if action == "raise" else ""
        )
        print(
            f"  {'ok ' if passed else 'XX '}{name}: {shown} "
            f"({worst * 1000:.0f} ms"
            + (f", note={note!r}" if note else "")
            + ")"
        )

    degraded = _state(
        phase="preflop",
        hole="AsKs",
        board="",
        pot=150,
        stack=9950,
        opp_stacks=[9950],
        seat=1,
        dealer=0,
        to_call=50,
        min_raise=0,
        max_raise=0,
        valid=["check"],
        history=[
            entry(0, "post_small_blind", 50, "preflop"),
            entry(1, "post_big_blind", 100, "preflop"),
            entry(0, "check", 0, "preflop"),
        ],
    )
    result = bot.decide(degraded)
    ok = ok and result.action == "check"
    print(f"  {'ok ' if result.action == 'check' else 'XX '}check-only table: {result.action}")

    unknown = _state(
        phase="preflop",
        hole="AsKs",
        board="",
        pot=150,
        stack=9950,
        opp_stacks=[9950],
        seat=1,
        dealer=0,
        to_call=50,
        min_raise=0,
        max_raise=0,
        valid=["check", "draw"],
        history=[
            entry(0, "post_small_blind", 50, "preflop"),
            entry(1, "post_big_blind", 100, "preflop"),
            entry(0, "check", 0, "preflop"),
        ],
    )
    result = bot.decide(unknown)
    passed = result.action == "check" and bot._fallback_note == "playing it safe"
    ok = ok and passed
    print(f"  {'ok ' if passed else 'XX '}unknown-action table: {result.action}")

    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
