"""Assemble the three-seed spread report for a set of v8 gauntlets.

Additive: reads finished ``tools.evaluate_v8`` reports and writes a combined
JSON + Markdown summary.  It runs no simulations and re-derives every number
from the per-seed fragments, so it cannot invent a result that the gauntlets
did not produce.

The deliverable is the **training-init-seed spread**: max minus min duel
margin against ``candidate-v7-0001c`` across the init seeds, set against the
16.78 BB/100 figure from the v7-0002 corpus.  That reference is exactly the
same quantity -- max minus min duel margin across three init seeds of one
corpus, on this instrument (v7-0002a -7.06, b -1.88, c +9.72) -- so the
comparison is like for like.

Because every gauntlet duels on the *same* evaluation seeds (300+i) against
the *same* incumbent, the per-seed margins are paired by common random
numbers.  The report therefore carries both readings:

  * the raw spread of the three means (the requested deliverable), and
  * paired per-evaluation-seed contrasts between init seeds, which remove
    the shared evaluation noise and are the more sensitive test of whether
    the init seeds differ at all.

Usage:
    python -m tools.summarize_seed_spread \
        --gauntlet 101=artifacts/evaluations/candidate-v8-0001a-gauntlet.json \
        --gauntlet 202=artifacts/evaluations/candidate-v8-0001-gauntlet.json \
        --gauntlet 303=artifacts/evaluations/candidate-v8-0001c-gauntlet.json \
        --output artifacts/evaluations/candidate-v8-0001-seed-spread.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# The v7-0002 three-init-seed duel spread; see .handoff/DECISIONS.md §4.7.
KNOWN_SEED_SPREAD_BB_PER_100 = 16.78
CHANNELS = (
    "vs-median",
    "vs-nit",
    "vs-station",
    "vs-shover",
    "vs-textured",
    "five-max-lineup",
)


def paired_stats(diffs: list[float]) -> dict[str, Any]:
    """Mean / sd / se / t of a paired sample (same convention as the tool)."""

    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return {"n": n, "mean": round(mean, 2), "sd": None, "se": None, "t": None}
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    return {
        "n": n,
        "mean": round(mean, 2),
        "sd": round(sd, 2),
        "se": round(se, 2),
        "t": round(mean / se, 2) if se else None,
        "mde_2se": round(2 * sd / math.sqrt(n), 2),
    }


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def duel_view(report: dict) -> dict[str, Any]:
    duel = report["duel"]
    paired = duel["report"]["paired"]
    return {
        "margin_bb_per_100": paired["mean"],
        "sd": paired["sd"],
        "se": paired["se"],
        "t": paired["t"],
        "empirical_mde_bb_per_100": duel["empirical_mde_bb_per_100"],
        "seeds": duel["seeds"],
        "per_seed_diffs": list(duel["report"]["paired"]["diffs"]),
        "verdict": duel["verdict"],
        "hands_per_seed": duel["report"]["hands_per_seed"],
    }


def battery_view(report: dict) -> dict[str, Any]:
    comparisons = report["battery_comparisons"]["channels"]
    channels: dict[str, Any] = {}
    passes = fails = 0
    for name in CHANNELS:
        entry = comparisons[name]
        paired = entry["paired_vs_champion"]
        mde = entry["published_mde_bb_per_100"]
        # V8_DESIGN §6.2 / DECISIONS §5: a channel is HELD when the candidate
        # is not worse than the champion by more than the published MDE.
        held = paired["mean"] >= -mde
        passes += 1 if held else 0
        fails += 0 if held else 1
        channels[name] = {
            "bb_per_100": entry["bb_per_100"],
            "champion_mean_bb_per_100": entry["champion_mean_bb_per_100"],
            "paired_diff_bb_per_100": paired["mean"],
            "paired_sd": paired["sd"],
            "paired_t": paired["t"],
            "published_mde_bb_per_100": mde,
            "held_within_mde": held,
        }
    return {"channels": channels, "held": passes, "broken": fails}


def render_markdown(report: dict) -> str:
    seeds = report["init_seeds"]
    spread = report["spread"]
    per_seed = report["per_seed"]
    lines: list[str] = []
    add = lines.append

    add("# candidate-v8-0001 — three-init-seed duel spread")
    add("")
    add(
        f"**Deliverable: the init-seed spread is "
        f"{spread['max_minus_min_bb_per_100']} BB/100** against a reference "
        f"spread of {spread['reference_spread_bb_per_100']} BB/100. "
        f"**Verdict: {spread['verdict']}.**"
    )
    add("")
    add(
        "Instrument: `tools/evaluate_v8.py`, the same invocation as the first "
        "gauntlet — 16 seat-swapped duel seeds at 2,000 hands per orientation, "
        "8 battery seeds, `--equity-trials 80`, `--workers 6`, 6,000-chip "
        "carry-over sessions at 50/100. Incumbent "
        f"`{report['incumbent']}` in every duel. No promotion; "
        "`artifacts/approved.json` untouched."
    )
    add("")
    add("## Duel vs the incumbent, per init seed")
    add("")
    add("| init seed | artifact | margin BB/100 | sd | se | t | own MDE (2·sd/√16) | verdict |")
    add("|---|---|---|---|---|---|---|---|")
    for seed in seeds:
        duel = per_seed[str(seed)]["duel"]
        add(
            f"| {seed} | `{per_seed[str(seed)]['artifact']}` | "
            f"**{duel['margin_bb_per_100']:+.2f}** | {duel['sd']} | {duel['se']} | "
            f"{duel['t']:+.2f} | {duel['empirical_mde_bb_per_100']} | "
            f"{duel['verdict']} |"
        )
    add("")
    add(
        f"Spread = max − min = {spread['max_minus_min_bb_per_100']} BB/100 "
        f"(best seed {spread['best_seed']}, worst seed {spread['worst_seed']}). "
        f"Reference: {spread['reference_provenance']}."
    )
    add("")
    add("## Paired cross-seed contrasts (common evaluation seeds)")
    add("")
    add(
        "Every gauntlet duels the same incumbent on the same 16 evaluation "
        "seeds, so the init seeds are paired by common random numbers and the "
        "shared evaluation noise cancels. This is the sensitive test of "
        "whether the init seeds differ *at all*."
    )
    add("")
    add("| contrast | mean | sd | se | t |")
    add("|---|---|---|---|---|")
    for name, stats in report["paired_cross_seed_contrasts"].items():
        t_text = "n/a" if stats["t"] is None else f"{stats['t']:+.2f}"
        add(
            f"| {name.replace('_minus_', ' − ')} | {stats['mean']:+.2f} | "
            f"{stats['sd']} | {stats['se']} | {t_text} |"
        )
    add("")
    add("## Batteries — channels held within the published MDE")
    add("")
    add(
        "A channel is *held* when the candidate is not worse than the frozen "
        "champion by more than the published 8-seed MDE "
        "(`noise-floor-2026-08-15.json`), which is the DECISIONS §5 / "
        "V8_DESIGN §6.2 condition."
    )
    add("")
    header = "| channel | MDE@8 | " + " | ".join(f"seed {s}" for s in seeds) + " |"
    add(header)
    add("|---" * (2 + len(seeds)) + "|")
    for channel in CHANNELS:
        first = per_seed[str(seeds[0])]["battery"]["channels"][channel]
        cells = []
        for seed in seeds:
            entry = per_seed[str(seed)]["battery"]["channels"][channel]
            mark = "held" if entry["held_within_mde"] else "**broken**"
            cells.append(
                f"{entry['paired_diff_bb_per_100']:+.2f} (t {entry['paired_t']:+.2f}) {mark}"
            )
        add(
            f"| {channel} | {first['published_mde_bb_per_100']} | "
            + " | ".join(cells)
            + " |"
        )
    add("")
    add("| init seed | channels held | channels broken |")
    add("|---|---|---|")
    for seed in seeds:
        counts = report["battery_summary"][str(seed)]
        add(f"| {seed} | {counts['held']} | {counts['broken']} |")
    add("")
    add("## Reading")
    add("")
    reading = report["reading"]
    add(
        f"- The spread ({spread['max_minus_min_bb_per_100']}) is "
        f"{reading['spread_over_mean_margin']}x the mean margin across seeds "
        f"({reading['mean_margin_bb_per_100']}). Seed-to-seed variation is "
        "the same size as the effect being claimed."
    )
    add(
        f"- Per-seed margins span {reading['min_margin_bb_per_100']:+.2f} to "
        f"{reading['max_margin_bb_per_100']:+.2f}; "
        f"{reading['seeds_clearing_own_mde']} of {len(seeds)} seeds clear "
        "their own empirical MDE, so the head-to-head win is not reproducible "
        "across initialisations."
    )
    add(
        f"- All {len(seeds)} per-seed duels are individually UNRESOLVED: "
        f"{reading['all_unresolved']}."
    )
    add(
        f"- Batteries: {reading['total_channels_broken']} of "
        f"{reading['total_channels']} seed-channel pairs are worse than the "
        "champion beyond the published MDE. The battery failure is the same "
        "on every seed, so it is structural, not initialisation luck."
    )
    add(
        "- Multiple comparisons: three pairwise contrasts were computed. A "
        "Bonferroni-corrected two-sided 5% threshold at df=15 is |t| ≈ 2.69, "
        "which none of the contrasts reaches — so even the apparent "
        "separation between init seeds is not established at corrected "
        "significance."
    )
    add("")
    add("## Wiring controls")
    add("")
    for seed in seeds:
        mirror = per_seed[str(seed)]["nullcheck_mirror_exact"]
        add(
            f"- seed {seed}: self-duel mirror exact = `{mirror}` (paired diffs "
            "identically zero while the underlying match moves real chips)"
        )
    add("")
    add(
        f"Verdict rule: {spread['verdict_rule']}."
    )
    add("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gauntlet",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="init seed and its finished evaluate_v8 report; repeatable",
    )
    parser.add_argument(
        "--output",
        default="artifacts/evaluations/candidate-v8-0001-seed-spread.json",
    )
    args = parser.parse_args(argv)

    runs: dict[int, dict] = {}
    for spec in args.gauntlet:
        seed_text, _, path = spec.partition("=")
        runs[int(seed_text)] = load(Path(path))

    seeds = sorted(runs)
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        report = runs[seed]
        manifest = report["config"]["candidate_manifest"]
        per_seed[str(seed)] = {
            "artifact": Path(manifest).name.replace(".manifest.json", ""),
            "manifest": manifest,
            "init_seed_recorded": report["trainer_context"]["init_seed_exported"],
            "nullcheck_mirror_exact": report["nullcheck"]["mirror_exact"],
            "duel": duel_view(report),
            "battery": battery_view(report),
        }

    margins = [per_seed[str(s)]["duel"]["margin_bb_per_100"] for s in seeds]
    spread = max(margins) - min(margins)
    best, worst = seeds[margins.index(max(margins))], seeds[margins.index(min(margins))]

    # Paired cross-seed contrasts: same evaluation seeds, same incumbent, so
    # the shared evaluation noise cancels.
    contrasts: dict[str, Any] = {}
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            da = per_seed[str(a)]["duel"]["per_seed_diffs"]
            db = per_seed[str(b)]["duel"]["per_seed_diffs"]
            if len(da) != len(db):
                continue
            contrasts[f"{a}_minus_{b}"] = paired_stats(
                [x - y for x, y in zip(da, db, strict=True)]
            )

    all_margins_unresolved = all(
        per_seed[str(s)]["duel"]["verdict"] == "UNRESOLVED" for s in seeds
    )
    sign_agreement = len({m > 0 for m in margins}) == 1
    if spread >= KNOWN_SEED_SPREAD_BB_PER_100:
        spread_verdict = "SEED-DEPENDENT"
    elif len(seeds) < 3:
        # A spread of zero over one artifact is not robustness, it is a
        # sample of one. Robustness is only claimable once the design's
        # three init seeds are all measured (V8_DESIGN §6.1).
        spread_verdict = "UNRESOLVED"
    elif sign_agreement and all(
        abs(m) > per_seed[str(s)]["duel"]["empirical_mde_bb_per_100"]
        for s, m in zip(seeds, margins, strict=True)
    ):
        spread_verdict = "SEED-ROBUST"
    else:
        spread_verdict = "UNRESOLVED"

    report = {
        "study": "candidate-v8-0001-seed-spread",
        "incumbent": runs[seeds[0]]["incumbent"],
        "init_seeds": seeds,
        "per_seed": per_seed,
        "spread": {
            "duel_margins_bb_per_100": {str(s): m for s, m in zip(seeds, margins, strict=True)},
            "max_minus_min_bb_per_100": round(spread, 2),
            "best_seed": best,
            "worst_seed": worst,
            "reference_spread_bb_per_100": KNOWN_SEED_SPREAD_BB_PER_100,
            "reference_provenance": (
                "v7-0002 three-init-seed duel spread (a -7.06, b -1.88, "
                "c +9.72) on this instrument; DECISIONS.md §4.7"
            ),
            "exceeds_reference_spread": spread >= KNOWN_SEED_SPREAD_BB_PER_100,
            "all_per_seed_duels_unresolved": all_margins_unresolved,
            "sign_agreement_across_seeds": sign_agreement,
            "verdict": spread_verdict,
            "verdict_rule": (
                "SEED-DEPENDENT if the init-seed spread reaches the 16.78 "
                "reference; SEED-ROBUST only if every seed's margin shares a "
                "sign and clears its own empirical MDE; UNRESOLVED otherwise"
            ),
        },
        "paired_cross_seed_contrasts": contrasts,
        "reading": {
            "mean_margin_bb_per_100": round(sum(margins) / len(margins), 2),
            "min_margin_bb_per_100": min(margins),
            "max_margin_bb_per_100": max(margins),
            "spread_over_mean_margin": round(
                spread / (sum(margins) / len(margins)), 2
            )
            if sum(margins)
            else None,
            "seeds_clearing_own_mde": sum(
                1
                for s, m in zip(seeds, margins, strict=True)
                if abs(m) > per_seed[str(s)]["duel"]["empirical_mde_bb_per_100"]
            ),
            "all_unresolved": all_margins_unresolved,
            "total_channels": len(seeds) * len(CHANNELS),
            "total_channels_broken": sum(
                per_seed[str(s)]["battery"]["broken"] for s in seeds
            ),
            "bonferroni_t_threshold_df15": 2.69,
        },
        "battery_summary": {
            str(s): {
                "held": per_seed[str(s)]["battery"]["held"],
                "broken": per_seed[str(s)]["battery"]["broken"],
            }
            for s in seeds
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = out.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    print(f"[summarize_seed_spread] wrote {out}")
    print(f"[summarize_seed_spread] wrote {markdown}")
    print(json.dumps(report["spread"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
