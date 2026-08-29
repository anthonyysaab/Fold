"""Estimate kappa_e for C4 — how much stronger the k-th aggressor is.

Recipe (spec: ``engine/rules/README.md``, C4): in the complete-information
archive, every bet/raise/all-in event by a field agent carries the raiser's
TRUE holding. For the k-th aggressive action of a street, measure the
raiser's equity against one uniform random holding on the board at that
moment (``hand_strength.estimate_equity`` — the canonical estimator, 200
seeded trials) and the canonical strength percentile. ``kappa_e`` is the
per-extra-raise SHIFT in that equity: the amount a typical continuing hand
loses per additional raise, in equity units — exactly what the C4 margin
adds per raise beyond the first.

Estimand notes, fixed before looking at results:

- The unit of analysis is the AGGRESSIVE EVENT, not the responder's
  decision, so the estimate is free of responder-selection confounding
  (who chose to stick around does not move it).
- k is the action's ordinal among ALL aggressive actions of its street
  (bet = 1st, raise over it = 2nd, ...): the quantity a responder facing
  "k raises so far" actually faces.
- Our own agent's aggressions are EXCLUDED (same reason and mechanism as
  ``build_p3_dataset``): the margin prices the field's escalation, and our
  refuted policy's raise range must not leak into the instrument.

Replay parsing is not re-derived: ``_read_json`` / ``_unwrap_rpc`` /
``_reconstruct_state`` come from ``tools.collect_foreign_play_data``, hole
cards from ``tools.build_phase_a_dataset._hole_cards_by_seat``, and the
board at the event from ``tools.build_p3_dataset.decision_board`` — the
proven parser stack of this archive.

The instrument is validated before the result (house rule). Three gates,
each impossible-by-construction; any failure aborts without reporting:

1. **Estimator sanity** — AA heads-up preflop equity vs one random holding
   must land in [0.83, 0.87] at 2,000 seeded trials.
2. **Planted slope** — synthetic (k, equity) data built from a known
   linear rule + noise must recover the planted slope within 3 standard
   errors.
3. **Shuffle null** — permuting the real k labels must produce a |slope|
   below 3 standard errors of zero: the pipeline must not manufacture a
   trend from nothing.

Read-only over the archive; writes only
``artifacts/evaluations/escalation-shift-estimate-<date>.json``. No
promotion, no network, no engine imports beyond the canonical estimator
and strength metric.

Usage::

    python -m tools.estimate_escalation_shift               # full run
    python -m tools.estimate_escalation_shift --limit 40    # smoke
    python -m tools.estimate_escalation_shift --selftest-only
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from engine.hand_strength import estimate_equity
from engine.strength_metric import strength_percentile
from tools.build_p3_dataset import DEFAULT_ROOTS, decision_board
from tools.build_phase_a_dataset import _hole_cards_by_seat
from tools.collect_foreign_play_data import _read_json, _unwrap_rpc

EXCLUDED_AGENTS = frozenset({"Fold-ver-4"})
AGGRESSIVE_ACTIONS = frozenset({"bet", "raise", "all-in"})
EQUITY_TRIALS = 200
#: k buckets in the DISPLAY table only. It must never decide a shipped
#: parameter: it did twice (as a fitted slope's regressor, then as a step
#: table's pooled terminal cell), and both times the value moved with it.
#: ``per_k_steps`` is the consumable estimate and ignores this.
K_CAP = 3
OUTPUT_DIR = Path("artifacts") / "evaluations"


def _seeded(table_id: str, sequence: int) -> int:
    # hashlib, not hash(): the builtin is salted per process and would
    # break the determinism convention every live-inference module keeps.
    import hashlib

    digest = hashlib.sha256(f"{table_id}:{sequence}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def event_rows(replay: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    """One row per eligible aggressive event: (k, equity, percentile)."""

    stats: Counter[str] = Counter()
    events = replay.get("events")
    if not isinstance(events, list) or not isinstance(replay.get("table"), Mapping):
        stats["replay_unusable"] += 1
        return [], stats
    holes = _hole_cards_by_seat(events)
    if not holes:
        stats["replay_without_hole_cards"] += 1
        return [], stats
    table_id = str((replay.get("table") or {}).get("id") or "")

    rows: list[dict[str, Any]] = []
    per_street_ordinal: dict[str, int] = defaultdict(int)
    for event in events:
        if not isinstance(event, Mapping) or event.get("type") != "ActionTaken":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        action = str(payload.get("action") or "").casefold()
        street = str(event.get("street") or "").casefold()
        if action not in AGGRESSIVE_ACTIONS:
            continue
        per_street_ordinal[street] += 1
        k = per_street_ordinal[street]
        agent = payload.get("agentName")
        if isinstance(agent, str) and agent in EXCLUDED_AGENTS:
            stats["excluded_agent_aggressions"] += 1
            continue
        seat = payload.get("seatNumber")
        if not isinstance(seat, int) or isinstance(seat, bool):
            stats["skip:no_seat"] += 1
            continue
        hole = holes.get(seat)
        if hole is None or len(hole) != 2:
            stats["skip:hole_not_revealed"] += 1
            continue
        sequence = event.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            stats["skip:no_sequence"] += 1
            continue
        try:
            board = tuple(decision_board(events, sequence, street))
            equity = estimate_equity(
                (hole[0], hole[1]),
                board,
                1,
                trials=EQUITY_TRIALS,
                seed=_seeded(table_id, sequence),
            )
            percentile = strength_percentile(hole, board)
        except Exception as error:  # noqa: BLE001 — count and skip, never guess
            stats[f"skip:{type(error).__name__}"] += 1
            continue
        rows.append(
            {
                "k": min(k, K_CAP),
                "k_raw": k,
                "street": street,
                "equity_vs_random": equity,
                "strength_percentile": percentile,
                "action": action,
            }
        )
        stats["rows"] += 1
    return rows, stats


def per_k_steps(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """UNCAPPED per-k means and their step from k = 1.

    This is the estimate C4 consumes. It is deliberately independent of
    ``K_CAP``: that constant aggregates the *display* table below, and an
    earlier version let it decide a shipped parameter — first as the
    regressor of a fitted slope, then as the terminal cell of a step
    table pooled over ``k >= K_CAP``. A pooled tail read at its own lower
    edge over-prices that edge (the k>=3 pool is +0.1190 while k=3 alone
    is +0.0978), and both values move with the cap. Per-k means do not.
    """

    by_k: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_k[row["k_raw"]].append(row["equity_vs_random"])
    if 1 not in by_k:
        raise ValueError("no k = 1 rows: the base case is undefined")
    base = statistics.fmean(by_k[1])
    out: dict[str, Any] = {}
    for k in sorted(by_k):
        values = by_k[k]
        out[str(k)] = {
            "n": len(values),
            "mean_equity_vs_random": statistics.fmean(values),
            "step_from_k1": statistics.fmean(values) - base,
            "se": (
                statistics.stdev(values) / len(values) ** 0.5
                if len(values) > 1
                else None
            ),
        }
    return {"base_k1_mean": base, "by_k": out}


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_k: dict[int, list[float]] = defaultdict(list)
    by_k_pct: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_k[row["k"]].append(row["equity_vs_random"])
        by_k_pct[row["k"]].append(row["strength_percentile"])
    table = {}
    for k in sorted(by_k):
        eq = by_k[k]
        table[k] = {
            "n": len(eq),
            "mean_equity_vs_random": statistics.fmean(eq),
            "se_equity": statistics.stdev(eq) / len(eq) ** 0.5 if len(eq) > 1 else None,
            "mean_strength_percentile": statistics.fmean(by_k_pct[k]),
        }
    return table


def slope_and_se(rows: Sequence[Mapping[str, Any]], key: str = "equity_vs_random"
                 ) -> tuple[float, float]:
    """OLS slope of ``key`` on k (capped), with its standard error."""

    xs = [row["k"] for row in rows]
    ys = [row[key] for row in rows]
    n = len(xs)
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        return 0.0, float("inf")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    residual = sum(
        (y - (mean_y + slope * (x - mean_x))) ** 2 for x, y in zip(xs, ys)
    )
    se = (residual / max(1, n - 2) / sxx) ** 0.5
    return slope, se


def selftest(verbose: bool = True) -> None:
    """Three impossible-by-construction gates; raises on any failure."""

    aa = estimate_equity(("As", "Ah"), (), 1, trials=2000, seed=11)
    if not 0.83 <= aa <= 0.87:
        raise AssertionError(f"gate 1: AA vs random preflop read {aa:.4f}")

    rng = random.Random(1234)
    planted = -0.05
    synth = [
        {"k": k, "equity_vs_random": 0.5 + planted * k + rng.gauss(0, 0.02)}
        for _ in range(4000)
        for k in (1, 2, 3)
    ]
    slope, se = slope_and_se(synth)
    if abs(slope - planted) > 3 * se:
        raise AssertionError(f"gate 2: planted {planted}, recovered {slope:.5f} (se {se:.5f})")

    if verbose:
        print(f"selftest PASS: AA={aa:.4f}, planted slope recovered {slope:.5f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--selftest-only", action="store_true")
    args = parser.parse_args(argv)

    selftest()
    if args.selftest_only:
        return 0

    files: list[Path] = []
    for root in DEFAULT_ROOTS:
        files.extend(sorted((Path(root) / "raw" / "tables").glob("*.json")))
    if args.limit:
        files = files[: args.limit]

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for index, path in enumerate(files):
        try:
            replay = _unwrap_rpc(_read_json(path))
        except (OSError, ValueError):
            stats["unreadable_files"] += 1
            continue
        if not isinstance(replay, dict):
            stats["unreadable_files"] += 1
            continue
        file_rows, file_stats = event_rows(replay)
        rows.extend(file_rows)
        stats.update(file_stats)
        if (index + 1) % 200 == 0:
            print(f"  {index + 1}/{len(files)} files, {len(rows)} rows...")

    if not rows:
        print("NO ROWS — refusing to report")
        return 1

    # Gate 3 — shuffle null on the real rows.
    rng = random.Random(99)
    ks = [row["k"] for row in rows]
    rng.shuffle(ks)
    shuffled = [
        {"k": k, "equity_vs_random": row["equity_vs_random"]}
        for k, row in zip(ks, rows)
    ]
    null_slope, null_se = slope_and_se(shuffled)
    if abs(null_slope) > 3 * null_se:
        print(f"GATE 3 FAIL: shuffled slope {null_slope:.5f} (se {null_se:.5f}) — refusing to report")
        return 1

    slope, se = slope_and_se(rows)
    pct_slope, pct_se = slope_and_se(rows, key="strength_percentile")
    table = summarize(rows)
    document = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tool": "tools.estimate_escalation_shift",
        "recipe": "engine/rules/README.md C4",
        "roots": [str(root) for root in DEFAULT_ROOTS],
        "files_walked": len(files),
        "equity_trials": EQUITY_TRIALS,
        "k_cap": K_CAP,
        "excluded_agents": sorted(EXCLUDED_AGENTS),
        "rows": len(rows),
        "by_k_capped_for_display": {str(k): v for k, v in table.items()},
        # The estimate C4 consumes: uncapped, per-k, cap-independent.
        "per_k_steps": per_k_steps(rows),
        # Raiser equity vs a random holding RISES with k; a typical
        # continuing hand loses exactly that much, so kappa_e = +slope.
        "kappa_e_equity_per_raise": slope,
        "kappa_e_se": se,
        "strength_percentile_shift_per_raise": pct_slope,
        "strength_percentile_shift_se": pct_se,
        "shuffle_null": {"slope": null_slope, "se": null_se},
        "stats": dict(stats),
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"escalation-shift-estimate-{datetime.now(UTC).date()}.json"
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")

    print(f"\nrows: {len(rows)}   files: {len(files)}   elapsed {document['elapsed_s']}s")
    for k, entry in table.items():
        se_txt = f"{entry['se_equity']:.4f}" if entry["se_equity"] else "n/a"
        print(
            f"  k={k}: n={entry['n']:5}  raiser equity vs random "
            f"{entry['mean_equity_vs_random']:.4f} (se {se_txt})  "
            f"strength pct {entry['mean_strength_percentile']:.4f}"
        )
    print(
        f"\nkappa_e (equity a typical hand LOSES per extra raise): "
        f"{slope:+.5f} (se {se:.5f})"
    )
    print(f"secondary — raiser percentile shift per raise: {pct_slope:+.5f} (se {pct_se:.5f})")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
