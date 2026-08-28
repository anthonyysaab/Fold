"""Generate the fixed preflop percentile table for the canonical strength metric.

V8_DESIGN.md §2: preflop strength is the percentile of the hero's holding
among the 169 canonical classes under a fixed heads-up all-in-equity
ordering, combo-weighted (pairs 6, suited 4, offsuit 12; 1,326 total) with
the mid-rank convention. The table is generated ONCE by this seeded script
and committed as ``engine/preflop_percentiles.py`` so live
inference does a dict lookup — no per-decision simulation, no drift.

Run as ``python -m tools.build_preflop_percentiles`` from the repo root.
Regenerating with the same seed and trials reproduces the file byte for
byte; regenerating with different parameters is a metric change and
therefore a schema event, not a patch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from random import Random

from engine._vendor.treys import Card, Evaluator

RANKS = "23456789TJQKA"
SEED = 20260816
TRIALS = 20_000

_SUITS = "shdc"


def canonical_classes() -> list[str]:
    """The 169 classes, high card first, in a fixed deterministic order."""

    classes: list[str] = []
    for i in range(len(RANKS) - 1, -1, -1):
        for j in range(i, -1, -1):
            if i == j:
                classes.append(RANKS[i] * 2)
            else:
                classes.append(f"{RANKS[i]}{RANKS[j]}s")
                classes.append(f"{RANKS[i]}{RANKS[j]}o")
    return classes


def representative(cls: str) -> tuple[str, str]:
    """A concrete two-card instance of the class; suits are arbitrary but fixed."""

    if len(cls) == 2:  # pair
        return f"{cls[0]}s", f"{cls[0]}h"
    if cls.endswith("s"):
        return f"{cls[0]}s", f"{cls[1]}s"
    return f"{cls[0]}s", f"{cls[1]}h"


def combo_weight(cls: str) -> int:
    if len(cls) == 2:
        return 6
    return 4 if cls.endswith("s") else 12


def headsup_equity(cls: str, evaluator: Evaluator, rng: Random, trials: int) -> float:
    """Monte-Carlo heads-up all-in equity vs a uniform random holding."""

    hole = [Card.new(c) for c in representative(cls)]
    full = [Card.new(r + s) for r in RANKS for s in _SUITS]
    remaining = [c for c in full if c not in hole]
    wins = ties = 0
    for _ in range(trials):
        drawn = rng.sample(remaining, 7)
        board, opp = drawn[:5], drawn[5:]
        mine = evaluator.evaluate(board, hole)
        theirs = evaluator.evaluate(board, opp)
        if mine < theirs:  # treys: lower rank is stronger
            wins += 1
        elif mine == theirs:
            ties += 1
    return (wins + 0.5 * ties) / trials


def build(trials: int, seed: int) -> tuple[dict[str, float], dict[str, float]]:
    evaluator = Evaluator()
    rng = Random(seed)
    equities = {
        cls: headsup_equity(cls, evaluator, rng, trials)
        for cls in canonical_classes()
    }
    total = sum(combo_weight(cls) for cls in equities)
    assert total == 1326, total
    percentiles: dict[str, float] = {}
    for cls, eq in equities.items():
        below = sum(
            combo_weight(other)
            for other, other_eq in equities.items()
            if other_eq < eq
        )
        percentiles[cls] = (below + 0.5 * combo_weight(cls)) / total
    return equities, percentiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    equities, percentiles = build(args.trials, args.seed)
    ordered = sorted(percentiles, key=percentiles.get, reverse=True)

    out = Path(__file__).resolve().parent.parent / (
        "engine/preflop_percentiles.py"
    )
    lines = [
        '"""Preflop percentile table for the canonical strength metric.',
        "",
        "GENERATED FILE -- do not edit by hand.",
        f"Generator: tools/build_preflop_percentiles.py "
        f"(seed {args.seed}, {args.trials} trials/class, "
        f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}).",
        "Percentile: combo-weighted mid-rank of the class in the heads-up",
        "all-in-equity ordering (pairs 6, suited 4, offsuit 12; 1,326 combos).",
        "Regeneration with different parameters is a metric change and a",
        "schema event. See V8_DESIGN.md section 2.",
        '"""',
        "",
        "PREFLOP_PERCENTILES: dict[str, float] = {",
    ]
    for cls in ordered:
        lines.append(f'    "{cls}": {percentiles[cls]:.6f},')
    lines.append("}")
    lines.append("")
    lines.append("HEADSUP_EQUITY: dict[str, float] = {")
    for cls in ordered:
        lines.append(f'    "{cls}": {equities[cls]:.6f},')
    lines.append("}")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    top = ", ".join(f"{c}={percentiles[c]:.4f}" for c in ordered[:5])
    bottom = ", ".join(f"{c}={percentiles[c]:.4f}" for c in ordered[-3:])
    print(f"wrote {out}")
    print(f"top: {top}")
    print(f"bottom: {bottom}")


if __name__ == "__main__":
    main()
