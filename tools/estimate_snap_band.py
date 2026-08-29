"""Estimate kappa for C3 — the snap-to-cover band.

Recipe (spec: ``engine/rules/README.md``, C3): in the archive, take every
decision where a field agent faced chips to call with fold legal, and
measure the fold response as a function of

    c = to_call / responder's stack behind      (c = 1: calling is all-in)

kappa is the widest band below c = 1 whose fold response is statistically
indistinguishable from the response to exact all-in prices: inside that
band, a bet already buys the same fold decision an all-in buys, so the
snap rule loses nothing by jumping to the covered opponent's all-in.

Estimand notes, fixed before looking at results:

- The unit is the RESPONDER'S decision; their hole cards are not needed
  (the response curve is behavior, not holdings), so eligibility is wider
  than the P3 dataset's.
- Our own agent's responses are EXCLUDED (the band prices the FIELD's
  response; our refuted policy's 14.3% fold rate must not leak in).
- Reference bucket is c in [0.97, 1.0]; the walk proceeds down the c bins
  and stops at the first two-proportion z-test rejection (alpha 0.05).
  Pot-odds terciles are reported alongside as a confounding check, never
  silently corrected.

Replay parsing is not re-derived: ``_read_json`` / ``_unwrap_rpc`` /
``_reconstruct_state`` come from ``tools.collect_foreign_play_data`` —
the proven parser stack.

The instrument is validated before the result (house rule). Gates, each
impossible-by-construction; any failure aborts without reporting:

1. **Planted step** — synthetic decisions whose fold rule steps at
   c* = 0.85 must recover a band edge in [0.80, 0.90].
2. **Flat null** — synthetic decisions with a c-independent fold rule
   must resolve NOTHING: with no behavioural edge anywhere there is no
   band to report, and emitting one would be the instrument inventing a
   result.
3. **Support** — on real data every c must lie in (0, 1] and the
   reference bucket must be non-empty, or there is nothing to
   distinguish from and the band is undefined.

Read-only over the archive; writes only
``artifacts/evaluations/snap-band-estimate-<date>.json``.

Usage::

    python -m tools.estimate_snap_band
    python -m tools.estimate_snap_band --limit 40
    python -m tools.estimate_snap_band --selftest-only
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.build_p3_dataset import DEFAULT_ROOTS
from tools.collect_foreign_play_data import _read_json, _reconstruct_state, _unwrap_rpc

EXCLUDED_AGENTS = frozenset({"Fold-ver-4"})
#: Bin edges for c; the last bin is the all-in reference.
C_EDGES = (0.0, 0.20, 0.40, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 1.0)
ALPHA_Z = 1.959964  # two-sided 0.05
OUTPUT_DIR = Path("artifacts") / "evaluations"


def _bin_of(c: float) -> int:
    for index in range(len(C_EDGES) - 1):
        if C_EDGES[index] < c <= C_EDGES[index + 1]:
            return index
    return 0 if c <= C_EDGES[0] else len(C_EDGES) - 2


def decision_rows(replay: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    """One row per eligible facing-a-wager decision: (c, folded, pot_odds)."""

    stats: Counter[str] = Counter()
    events = replay.get("events")
    if not isinstance(events, list) or not isinstance(replay.get("table"), Mapping):
        stats["replay_unusable"] += 1
        return [], stats

    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping) or event.get("type") != "ActionTaken":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        agent = payload.get("agentName")
        if isinstance(agent, str) and agent in EXCLUDED_AGENTS:
            stats["excluded_agent_decisions"] += 1
            continue
        allowed = payload.get("allowedActions")
        if not isinstance(allowed, Mapping):
            continue
        call_chips = allowed.get("callChips")
        if isinstance(call_chips, bool) or not isinstance(call_chips, (int, float)):
            continue
        to_call = int(call_chips)
        if to_call <= 0:
            stats["not_facing_a_wager"] += 1
            continue
        if "fold" not in (allowed.get("availableActions") or []):
            stats["skip:fold_not_legal"] += 1
            continue
        seat = payload.get("seatNumber")
        if not isinstance(seat, int) or isinstance(seat, bool):
            continue
        action = str(payload.get("action") or "").casefold()
        if not action:
            continue
        try:
            state = dict(_reconstruct_state(dict(replay), dict(event)))
        except Exception as error:  # noqa: BLE001 — count and skip
            stats[f"skip:{type(error).__name__}"] += 1
            continue
        stack = None
        for seat_entry in state.get("seats") or []:
            if isinstance(seat_entry, Mapping) and seat_entry.get("seatNumber") == seat:
                raw = seat_entry.get("stackChips")
                if not isinstance(raw, bool) and isinstance(raw, (int, float)):
                    stack = int(raw)
                break
        if stack is None or stack <= 0:
            stats["skip:no_responder_stack"] += 1
            continue
        pot = state.get("potChips")
        pot = int(pot) if isinstance(pot, (int, float)) and not isinstance(pot, bool) else 0
        c = min(1.0, to_call / stack)
        rows.append(
            {
                "c": c,
                "folded": 1 if action == "fold" else 0,
                "pot_odds": to_call / max(pot + to_call, 1),
            }
        )
        stats["rows"] += 1
    return rows, stats


def band_edge(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Walk bins downward from the all-in reference; stop at first rejection."""

    bins: dict[int, list[int]] = {index: [] for index in range(len(C_EDGES) - 1)}
    for row in rows:
        bins[_bin_of(row["c"])].append(row["folded"])
    reference = bins[len(C_EDGES) - 2]
    if not reference:
        return {"error": "empty reference bucket"}
    p_ref = sum(reference) / len(reference)

    per_bin = {}
    lowest_indistinguishable = len(C_EDGES) - 2
    walking = True
    stopped_on = None
    for index in range(len(C_EDGES) - 3, -1, -1):
        labels = bins[index]
        if not labels:
            per_bin[index] = {"n": 0}
            if walking and stopped_on is None:
                stopped_on = "empty_bin"
            walking = False
            continue
        p = sum(labels) / len(labels)
        pooled = (sum(labels) + sum(reference)) / (len(labels) + len(reference))
        se = math.sqrt(
            max(1e-12, pooled * (1 - pooled))
            * (1 / len(labels) + 1 / len(reference))
        )
        z = (p - p_ref) / se
        rejected = abs(z) > ALPHA_Z
        per_bin[index] = {
            "range": [C_EDGES[index], C_EDGES[index + 1]],
            "n": len(labels),
            "fold_rate": p,
            "z_vs_allin": z,
            "rejected": rejected,
        }
        if walking and not rejected:
            lowest_indistinguishable = index
        else:
            if walking and stopped_on is None:
                stopped_on = "rejection"
            walking = False
    edge_c = C_EDGES[lowest_indistinguishable]
    # A walk that ended on an EMPTY bin measured nothing below its last
    # step: the band is bounded by data absence, not by a behavioral
    # change, and reporting a kappa there would dress a corpus hole up as
    # a measurement (the exact P3 blind spot: the field almost never
    # faces near-stack bets).
    # Reaching bin 0 without a rejection is NOT a resolved band: it means
    # the walk never found a behavioural edge anywhere, and reporting
    # kappa = 1 - C_EDGES[0] = 1.0 would emit a value SnapToCoverParams
    # refuses (band must be <= 0.5). That is the flat-null shape, and the
    # honest report is "no edge found".
    resolved = stopped_on == "rejection"
    return {
        "reference": {
            "range": [C_EDGES[-2], C_EDGES[-1]],
            "n": len(reference),
            "fold_rate": p_ref,
        },
        "per_bin": per_bin,
        "edge_c": edge_c,
        "kappa": round(1.0 - edge_c, 4) if resolved else None,
        "status": (
            "RESOLVED"
            if resolved
            else (
                "UNRESOLVED: support ends at an empty bin"
                if stopped_on == "empty_bin"
                else "UNRESOLVED: no behavioural edge found in any bin"
            )
        ),
        "stopped_on": stopped_on,
    }


def _synthetic(rule, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        c = rng.random()
        rows.append({"c": c, "folded": 1 if rng.random() < rule(c) else 0,
                     "pot_odds": 0.3})
    return rows


def selftest(verbose: bool = True) -> None:
    step = band_edge(_synthetic(lambda c: 0.7 if c >= 0.85 else 0.4, 20000, 7))
    if not 0.80 <= step["edge_c"] <= 0.90:
        raise AssertionError(f"gate 1: planted step at 0.85, recovered edge {step['edge_c']}")
    flat = band_edge(_synthetic(lambda c: 0.55, 20000, 8))
    # A c-independent rule has NO behavioural edge, so the walk must
    # report no resolved band rather than "the band is everything".
    if flat["kappa"] is not None or flat["status"].startswith("RESOLVED"):
        raise AssertionError(
            f"gate 2: a flat rule must resolve nothing, got {flat['status']}"
        )
    if verbose:
        print(
            f"selftest PASS: step edge {step['edge_c']} (kappa {step['kappa']}),"
            f" flat -> {flat['status']}"
        )


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
        file_rows, file_stats = decision_rows(replay)
        rows.extend(file_rows)
        stats.update(file_stats)
        if (index + 1) % 200 == 0:
            print(f"  {index + 1}/{len(files)} files, {len(rows)} rows...")

    if not rows:
        print("NO ROWS — refusing to report")
        return 1
    if any(not 0.0 < row["c"] <= 1.0 for row in rows):
        print("GATE 3 FAIL: c outside (0, 1] — refusing to report")
        return 1

    result = band_edge(rows)
    if "error" in result:
        print(f"GATE 3 FAIL: {result['error']} — refusing to report")
        return 1

    # Confounding check: the same walk inside each pot-odds tercile.
    ordered = sorted(row["pot_odds"] for row in rows)
    t1, t2 = ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]
    terciles = {}
    for name, low, high in (("low", -1.0, t1), ("mid", t1, t2), ("high", t2, 2.0)):
        subset = [row for row in rows if low < row["pot_odds"] <= high]
        sub = band_edge(subset)
        terciles[name] = {
            "n": len(subset),
            "kappa": sub.get("kappa"),
            "edge_c": sub.get("edge_c"),
        }

    document = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tool": "tools.estimate_snap_band",
        "recipe": "engine/rules/README.md C3",
        "roots": [str(root) for root in DEFAULT_ROOTS],
        "files_walked": len(files),
        "excluded_agents": sorted(EXCLUDED_AGENTS),
        "rows": len(rows),
        "c_edges": list(C_EDGES),
        "result": result,
        "pot_odds_terciles": terciles,
        "stats": dict(stats),
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"snap-band-estimate-{datetime.now(UTC).date()}.json"
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")

    print(f"\nrows: {len(rows)}   files: {len(files)}   elapsed {document['elapsed_s']}s")
    ref = result["reference"]
    print(f"  all-in reference (c in {ref['range']}): n={ref['n']}, fold rate {ref['fold_rate']:.3f}")
    for index in sorted(result["per_bin"], reverse=True):
        entry = result["per_bin"][index]
        if entry.get("n"):
            print(
                f"  c in {entry['range']}: n={entry['n']:5}  fold {entry['fold_rate']:.3f}"
                f"  z={entry['z_vs_allin']:+.2f}  {'REJECT' if entry['rejected'] else 'same'}"
            )
    print(f"\nstatus: {result['status']}")
    print(f"kappa (snap band below all-in): {result['kappa']}   edge c = {result['edge_c']}")
    if result["kappa"] is None:
        print("-> per the C3 spec fallback: kappa falls to OWNER-SET (proposed 0.15), flagged")
    print(f"pot-odds terciles: {terciles}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
