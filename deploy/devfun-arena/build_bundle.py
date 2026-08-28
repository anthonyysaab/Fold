"""Assemble the dev.fun Arena sandbox bundle (``bundle.zip``) and smoke-test it.

The dev.fun sandbox runs an uploaded ``bundle.zip`` in a restricted container.
Two constraints shape this build:

* the sandbox blocks filesystem reads, so the strategy builds an inert
  contract-matching zero proposal network programmatically instead of loading
  ``artifacts/tiny-policy-pure.json`` (the network is never consulted anyway:
  no ``table_sizes`` are declared, so the deterministic v5 equity rules drive
  every decision);
* ``harness/`` must stay under 256 KB uncompressed, so training-only modules
  are dropped and only the live decision closure ships.

Layout inside the zip::

    harness/strategy.py
    harness/bluff.py                        (root advisor modules)
    harness/lead_position.py
    harness/risk_temperature.py
    harness/engine/...     (trimmed live-decision package)

Run from the repo root or anywhere::

    python deploy/devfun-arena/build_bundle.py

then upload ``bundle.zip`` via ``tools/submit.py`` (defaults to the Eval S1
competition).
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PKG_SRC = REPO / "engine"
ROOT_MODULES = ("bluff.py", "lead_position.py", "risk_temperature.py")
TRAINING_ONLY = (
    "foreign_data.py",
    "learned_policy.py",
    "offline_trainer.py",
    "table_simulator.py",
    "torch_network.py",
    "torch_policy.py",
    "training_telemetry.py",
    "README.md",
)

BUILD = REPO / "build"
HARNESS = BUILD / "harness"
BUNDLE = REPO / "bundle.zip"
SIZE_CAP = 262144  # 256 KB uncompressed harness/ cap enforced by the sandbox

# Fallback reasoning markers from strategy.py; seeing one in the smoke test
# means the engine rejected a probe and the bundle degraded to safe mode.
FALLBACK_NOTES = (
    "playing it safe",
    "keeping it simple here",
    "adjusting to the legal set",
    "sizing fallback",
)


def _ignore(_dir, names):
    return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]


def assemble() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    HARNESS.mkdir(parents=True)

    pkg = HARNESS / "engine"
    shutil.copytree(PKG_SRC, pkg, ignore=_ignore)
    for name in TRAINING_ONLY:
        target = pkg / name
        if target.exists():
            target.unlink()
    treys_readme = pkg / "_vendor" / "treys" / "README.md"
    if treys_readme.exists():
        treys_readme.unlink()
    vendor_readme = pkg / "_vendor" / "README.md"
    if vendor_readme.exists():
        vendor_readme.unlink()

    for name in ROOT_MODULES:
        shutil.copy2(REPO / name, HARNESS / name)
    shutil.copy2(HERE / "strategy.py", HARNESS / "strategy.py")

    harness_bytes = sum(p.stat().st_size for p in HARNESS.rglob("*") if p.is_file())
    print(f"harness/ uncompressed: {harness_bytes} bytes (cap {SIZE_CAP})")
    if harness_bytes >= SIZE_CAP:
        raise SystemExit(f"harness/ too big: {harness_bytes} >= {SIZE_CAP}")

    if BUNDLE.exists():
        BUNDLE.unlink()
    files = 0
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(HARNESS.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(BUILD).as_posix())
                files += 1
    print(f"bundle.zip: {files} files, {BUNDLE.stat().st_size // 1024} KB -> {BUNDLE}")


def _board(text):
    return [text[i : i + 2] for i in range(0, len(text), 2)]


def _table(street, hole, board, pot, stack, to_call, raises, available, seats=2):
    hero_bet = 0
    seat_list = []
    for number in range(1, seats + 1):
        is_hero = number == 2
        seat_list.append(
            {
                "seatNumber": number,
                "status": "Active",
                "stackChips": stack,
                "currentBetChips": to_call if number == 1 else hero_bet,
                "holeCards": _board(hole) if is_hero else None,
            }
        )
    return {
        "id": "probe", "tableId": "probe", "street": street, "potChips": pot,
        "currentBet": to_call, "boardCards": _board(board),
        "smallBlindChips": 50, "bigBlindChips": 100, "selfSeatNumber": 2,
        "seats": seat_list,
        "recentEvents": [
            {"type": "ActionTaken", "street": street,
             "summary": {"action": "raise", "seatNumber": 1, "amount": to_call}}
            for _ in range(raises)
        ],
        "allowedActions": {
            "canFold": "fold" in available, "canCheck": "check" in available,
            "canCall": "call" in available, "canBet": "bet" in available,
            "canRaise": "raise" in available, "canAllIn": "all-in" in available,
            "callChips": to_call, "callAmount": to_call, "callToAmount": to_call,
            "minBet": None, "minRaiseTo": (to_call * 2 if to_call else 100),
            "betRange": None, "raiseRange": {"min": to_call * 2 if to_call else 100, "max": stack},
            "allInToAmount": None, "availableActions": available,
            "amountSemantics": "toAmount", "reasoningRequired": False,
        },
    }


def smoke_test() -> int:
    sys.path.insert(0, str(HARNESS))
    import strategy  # noqa: E402  (imported from the assembled bundle)

    if strategy._init_error:
        print(f"XX bundle init failed: {strategy._init_error}")
        return 1

    facing = ["fold", "call", "raise"]
    cases = [
        ("K6 boat 777K turn raise", _table("turn", "Kd6d", "7s7cKs7h", 4700, 7990, 1540, 2, facing), "not-fold"),
        ("AKs heads-up preflop open", _table("preflop", "AsKs", "", 450, 9700, 300, 1, facing), "not-fold"),
        ("72o 4-max preflop open", _table("preflop", "7c2d", "", 550, 9700, 300, 1, facing, seats=4), "observe"),
        ("A3c 777K turn raise", _table("turn", "Ac3c", "7s7cKs7h", 4700, 7990, 1540, 2, facing), "observe"),
        ("K4s 7766K river lead", _table("river", "4sKs", "7h6s7s6dKh", 1600, 9600, 800, 1, facing), "observe"),
    ]
    ok = True
    for name, table, want in cases:
        result = strategy.choose_action(table)
        action = result.get("action")
        note = result.get("reasoning_text") or ""
        legal = action in table["allowedActions"]["availableActions"]
        degraded = note in FALLBACK_NOTES
        passed = legal and not degraded and isinstance(result, dict)
        if want == "not-fold":
            passed = passed and action != "fold"
        ok = ok and passed
        amount = result.get("amount")
        shown = f"{action}" + (f" to {amount}" if amount is not None else "")
        print(f"  {'ok ' if passed else 'XX '}{name}: {shown}")
    return 0 if ok else 1


if __name__ == "__main__":
    assemble()
    print("=== local smoke test ===")
    rc = smoke_test()
    print("SMOKE", "PASS" if rc == 0 else "FAIL")
    raise SystemExit(rc)
