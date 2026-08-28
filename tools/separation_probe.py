"""Does a candidate's aggression carry information about its cards?

`head_degeneracy_audit` measures whether the NETWORK returns a constant.
That is a property of the model. The property that matters for play is
different and this project already names it: **strength separation**, the
mean canonical hand strength when the policy aggresses minus the mean when
it folds. A player whose aggression says nothing about its hand separates
at zero, however lively its logits are.

The reference points, all on the canonical metric
(`strength_metric.strength_percentile`):

* the real S14 field   **+0.386**
* `candidate-v7-0001c`  +0.170
* `candidate-v8-0001`   +0.110

The incumbent was stopped *because* of this number, not because of its
BB/100 -- it bets the flop at a median hand strength of 0.488, a random
hand. So a candidate that is less head-degenerate has only been shown to
be a less degenerate PLAYER if this moves.

Runs on `vs-p3`, the only battery seat that folds by a model fitted to the
real field's revealed hole cards, using the same seed convention and the
same `separation_report` as `p3-gate-2026-08-16` so the numbers are
comparable to the published ones.

Offline and seeded. Nothing here trains, promotes, or deploys.

Example::

    python -m tools.separation_probe \\
        --manifests artifacts/candidates/candidate-v7-0001c.approved.manifest.json \\
                    artifacts/deadhead/deadhead-treated-s92.manifest.json \\
        --output artifacts/evaluations/separation-probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.evaluate_policies import _gauntlet_workers, _run_tasks
from tools.evaluate_v8 import P3_CHANNEL, separation_report
from tools.p3_gate import DEFAULT_FIT_PATH, LogTask, family_rates


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass


@dataclass(frozen=True)
class ManifestSpec:
    """A learned artifact addressed by value, rebuildable in a worker."""

    label: str
    manifest: str
    equity_trials: int

    def build(self) -> object:
        from devfun_poker_playground.learned_policy import load_policy

        return load_policy(self.manifest, equity_trials=self.equity_trials)


def probe(args: argparse.Namespace) -> dict[str, Any]:
    specs = [
        ManifestSpec(Path(path).stem.replace(".manifest", ""), path, args.equity_trials)
        for path in args.manifests
    ]
    tasks = [
        LogTask(
            hero=spec,
            channel=P3_CHANNEL,
            opponent_seed=13 + index,
            hands=max(50, round(1_000 * args.scale)),
            seed=100 + index,
            stack=args.starting_stack,
            fit_path=args.p3_fit,
        )
        for spec in specs
        for index in range(args.seeds)
    ]
    print(
        f"[separation_probe] {len(specs)} artifacts x {args.seeds} seeds "
        f"= {len(tasks)} recording matches",
        flush=True,
    )
    outcomes = _run_tasks(tasks, _gauntlet_workers(args.workers, len(tasks)))

    pooled: dict[str, list] = {}
    violations = 0
    for task, outcome in zip(tasks, outcomes, strict=True):
        if not outcome["chips_conserved"]:
            violations += 1
        pooled.setdefault(task.hero.label, []).extend(outcome["hero"])

    report: dict[str, Any] = {
        "channel": P3_CHANNEL,
        "seeds": args.seeds,
        "hands_per_seed": max(50, round(1_000 * args.scale)),
        "starting_stack": args.starting_stack,
        "equity_trials": args.equity_trials,
        "metric": "strength_metric.strength_percentile (canonical)",
        "chip_conservation_violations": violations,
        "reference": {
            "field_s14": 0.386,
            "candidate_v7_0001c_published": 0.170,
            "candidate_v8_0001_published": 0.110,
        },
        "policies": {},
    }
    for label, records in pooled.items():
        report["policies"][label] = {
            **family_rates(records),
            "strength_separation": separation_report(
                [(row[0], row[1], row[2]) for row in records]
            ),
        }
    return report


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Strength separation probe")
    add("")
    add(
        f"`{report['channel']}` · {report['seeds']} seeds x "
        f"{report['hands_per_seed']} hands · {report['starting_stack']} chips · "
        f"{report['equity_trials']} equity trials"
    )
    add("")
    add(
        "> Separation is mean canonical hand strength when the policy "
        "**aggresses** minus when it **folds**. It is the metric the previous "
        "architecture was retired over. Head constancy is a property of the "
        "network; this is a property of the play."
    )
    add("")
    ref = report["reference"]
    add(
        f"Reference: field **+{ref['field_s14']}**, "
        f"`candidate-v7-0001c` +{ref['candidate_v7_0001c_published']}, "
        f"`candidate-v8-0001` +{ref['candidate_v8_0001_published']}."
    )
    add("")
    add("| policy | decisions | aggression | fold | separation | 95% CI |")
    add("|---|---|---|---|---|---|")
    for label, entry in sorted(report["policies"].items()):
        pooled = entry["strength_separation"]["all_streets"]
        value = pooled.get("separation_aggress_minus_fold")
        ci = pooled.get("ci95") or [None, None]
        add(
            f"| `{label}` | {pooled.get('decisions', 0)} | "
            f"{entry.get('aggression_rate', '-')} | {entry.get('fold_rate', '-')} | "
            f"**{value if value is not None else 'n/a'}** | "
            f"[{ci[0]}, {ci[1]}] |"
        )
    add("")
    add("Per street (separation; `n/a` means too few of one family to compare):")
    add("")
    add("| policy | preflop | flop | turn | river |")
    add("|---|---|---|---|---|")
    for label, entry in sorted(report["policies"].items()):
        sep = entry["strength_separation"]
        cells = " | ".join(
            str(
                (sep.get(street) or {}).get("separation_aggress_minus_fold")
                if (sep.get(street) or {}).get("separation_aggress_minus_fold")
                is not None
                else "n/a"
            )
            for street in ("preflop", "flop", "turn", "river")
        )
        add(f"| `{label}` | {cells} |")
    add("")
    add(
        f"Chip conservation violations: "
        f"{report['chip_conservation_violations']}."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--starting-stack", type=int, default=6_000)
    parser.add_argument("--equity-trials", type=int, default=80)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--p3-fit", default=str(DEFAULT_FIT_PATH))
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    report = probe(args)
    text = render(report)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        out.with_suffix(".md").write_text(text, encoding="utf-8")
        print(f"[separation_probe] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
