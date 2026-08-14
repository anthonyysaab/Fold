"""Audit the foreign teacher CSV before any model fitting.

Reads the collector's combined decision CSV and reports the distributions
that decide the learning setup: action-family mix by street and season,
facing-a-bet fold rates per street (the bluff advisor's calibration
targets), per-agent aggression frequencies (the opponent tracker's prior),
sizing fractions, stack-depth and player-count mixes, and suggested
inverse-frequency class weights. Read-only; makes no network requests.

Example:
    python tools/audit_foreign_play_data.py \
        "foreign play data/last 5 seasons top 15/combined_top15_decisions_s9-s13.csv"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

STREETS = ("preflop", "flop", "turn", "river")
FAMILIES = ("fold", "check_call", "aggress")
_STACK_BUCKETS = ((0, 10), (10, 25), (25, 50), (50, 10_000))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize_rows(rows: list[dict]) -> dict:
    """Aggregate eligible-decision statistics for the report."""

    eligible = [row for row in rows if str(row.get("teacher_eligible")) == "True"]
    by_season = Counter(row["season_number"] for row in rows)
    family_mix = Counter(row["action_family"] for row in eligible)
    street_family: dict[str, Counter] = defaultdict(Counter)
    facing_bet: dict[str, Counter] = defaultdict(Counter)
    unopened: dict[str, Counter] = defaultdict(Counter)
    sizing_pot_fraction: dict[str, list[float]] = defaultdict(list)
    risk_fractions: list[float] = []
    agent_totals: dict[str, Counter] = defaultdict(Counter)
    stack_family: dict[str, Counter] = defaultdict(Counter)
    player_counts = Counter()
    reward_by_family: dict[str, list[float]] = defaultdict(list)

    for row in eligible:
        street = row["street"]
        family = row["action_family"]
        street_family[street][family] += 1
        to_call = int(float(row.get("call_chips") or 0))
        if to_call > 0:
            facing_bet[street][family] += 1
        elif street != "preflop":
            unopened[street][family] += 1
        if family == "aggress":
            pot_before = float(row.get("pot_before_chips") or 0)
            new_chips = float(row.get("action_new_chips") or 0)
            if pot_before > 0 and new_chips > 0:
                sizing_pot_fraction[street].append(
                    min(4.0, new_chips / (pot_before + to_call))
                )
            risk = float(row.get("action_effective_stack_fraction") or 0)
            risk_fractions.append(risk)
        agent_key = f"s{row['season_number']}:{row['agent_id']}"
        agent_totals[agent_key]["decisions"] += 1
        if family == "aggress":
            agent_totals[agent_key]["aggressions"] += 1
        big_blind = max(1.0, float(row.get("big_blind_chips") or 1))
        depth_bb = float(row.get("effective_stack_before_chips") or 0) / big_blind
        for low, high in _STACK_BUCKETS:
            if low <= depth_bb < high:
                stack_family[f"{low}-{high}bb"][family] += 1
                break
        player_counts[row.get("active_player_count") or "?"] += 1
        reward = float(row.get("hand_reward_bb") or 0.0)
        if math.isfinite(reward):
            reward_by_family[family].append(reward)

    aggression_frequencies = [
        counts["aggressions"] / counts["decisions"]
        for counts in agent_totals.values()
        if counts["decisions"] >= 10
    ]

    def _fold_rates(table: dict[str, Counter]) -> dict[str, dict[str, float | int]]:
        rates = {}
        for street in STREETS:
            counts = table.get(street)
            if not counts:
                continue
            total = sum(counts.values())
            rates[street] = {
                "decisions": total,
                "fold_rate": round(counts.get("fold", 0) / total, 4),
                "continue_rate": round(counts.get("check_call", 0) / total, 4),
                "raise_rate": round(counts.get("aggress", 0) / total, 4),
            }
        return rates

    total_eligible = max(1, len(eligible))
    class_weights = {
        family: round(total_eligible / (len(FAMILIES) * max(1, family_mix[family])), 3)
        for family in FAMILIES
    }
    return {
        "rows": len(rows),
        "eligible_rows": len(eligible),
        "rows_by_season": dict(sorted(by_season.items())),
        "family_mix": {
            family: {
                "count": family_mix.get(family, 0),
                "share": round(family_mix.get(family, 0) / total_eligible, 4),
            }
            for family in FAMILIES
        },
        "family_mix_by_street": {
            street: dict(street_family[street])
            for street in STREETS
            if street in street_family
        },
        "facing_bet_response_by_street": _fold_rates(facing_bet),
        "unopened_postflop_by_street": {
            street: {
                "decisions": sum(unopened[street].values()),
                "bet_rate": round(
                    unopened[street].get("aggress", 0)
                    / max(1, sum(unopened[street].values())),
                    4,
                ),
            }
            for street in STREETS
            if street in unopened
        },
        "aggressive_sizing_pot_fraction": {
            street: {
                "count": len(values),
                "median": round(_median(values), 4),
                "mean": round(_mean(values), 4),
            }
            for street, values in sorted(sizing_pot_fraction.items())
            if values
        },
        "aggressive_risk_fraction": {
            "median": _median(risk_fractions),
            "p90": _percentile(risk_fractions, 0.9),
        },
        "agent_aggression_frequency": {
            "agents_with_10_plus_decisions": len(aggression_frequencies),
            "min": _percentile(aggression_frequencies, 0.0),
            "median": _median(aggression_frequencies),
            "mean": _mean(aggression_frequencies),
            "p90": _percentile(aggression_frequencies, 0.9),
            "max": _percentile(aggression_frequencies, 1.0),
        },
        "family_mix_by_stack_depth": {
            bucket: dict(counts) for bucket, counts in sorted(stack_family.items())
        },
        "active_player_count_mix": dict(sorted(player_counts.items())),
        "mean_reward_bb_by_family": {
            family: round(_mean(values), 4)
            for family, values in sorted(reward_by_family.items())
            if values
        },
        "suggested_inverse_frequency_class_weights": class_weights,
    }


def _render(summary: dict) -> str:
    lines = [
        f"rows: {summary['rows']} (eligible {summary['eligible_rows']})",
        f"by season: {summary['rows_by_season']}",
        "family mix: "
        + ", ".join(
            f"{family} {data['share']:.1%}"
            for family, data in summary["family_mix"].items()
        ),
        "facing a bet (fold / continue / raise):",
    ]
    for street, data in summary["facing_bet_response_by_street"].items():
        lines.append(
            f"  {street:8} n={data['decisions']:<5} fold {data['fold_rate']:.1%}"
            f"  continue {data['continue_rate']:.1%}  raise {data['raise_rate']:.1%}"
        )
    lines.append("aggressive sizing (fraction of attacked pot):")
    for street, data in summary["aggressive_sizing_pot_fraction"].items():
        lines.append(
            f"  {street:8} n={data['count']:<4} median {data['median']:.2f}"
            f"  mean {data['mean']:.2f}"
        )
    frequency = summary["agent_aggression_frequency"]
    lines.extend(
        (
            "per-agent aggression frequency "
            f"(n={frequency['agents_with_10_plus_decisions']}): "
            f"median {frequency['median']:.3f}, mean {frequency['mean']:.3f}, "
            f"p90 {frequency['p90']:.3f}, max {frequency['max']:.3f}",
            f"player counts: {summary['active_player_count_mix']}",
            f"stack-depth family mix: {summary['family_mix_by_stack_depth']}",
            f"mean reward (bb) by family: {summary['mean_reward_bb_by_family']}",
            "suggested class weights: "
            f"{summary['suggested_inverse_frequency_class_weights']}",
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="collector decision CSV to audit")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)
    with open(Path(args.csv_path), encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize_rows(rows)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
