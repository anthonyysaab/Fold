"""The three unmeasured gate changes, ablated on a card-aware opponent.

``.handoff/NEXT.md`` item 2: the effective-stack risk cap, the
effective-stack call gates, and ``reveal_expense_equity_slope = 0.12``
ship on the next supervisor restart whether or not anyone intends it, and
the standing instruction is **measure them or revert them**.

The only evidence to date is **-13.97 BB/100 (t = -5.18)** against the
risk cap, collected on batteries whose every opponent is card-blind. That
caveat is not a footnote: an opponent that cannot see its own cards
cannot punish overcommitment, so the instrument prices what a gate costs
and is structurally blind to what it prevents. It can only ever argue for
reverting.

``vs-p3`` is the first battery seat that folds by a model fitted to the
real field's revealed hole cards (``p3-gate-2026-08-16``: 80.3pp fold
probability span across its own cards, +0.2244 continue-minus-fold
strength separation at t = +26.58). It narrows exactly that blind spot,
so it is the channel this ablation is built around. ``vs-median`` --
P3's card-blind structural twin, same aggression and same nominal
``fold_vs_bet`` -- and ``vs-station`` are carried as controls, so a
"card-awareness changed the verdict" claim is made within one run
instead of across two reports.

**It does not close the blind spot.** P3's aggression and shove rate are
still the inherited card-blind knobs; only its folding is card-aware. A
policy still cannot be punished for the hands it shows down, only for the
prices it lays.

Four arms, one edit apart
-------------------------

Every arm is the *same* policy artifact; the only difference is the gate
dataclass, so a contrast is attributable to exactly one edit.

* ``live``          -- both denominators on the effective stack, slope 0.12.
  This is what restarts.
* ``revert-cap``    -- the sizing risk cap back on hero's purse.
* ``revert-calls``  -- the call gates back on hero's purse, slope 0.
  The slope travels with the call gates because it has nowhere else to
  act: it raises a call gate's equity floor and never creates a gate
  (``gate-binding-audit-2026-08-26``: it modifies a verdict on the same
  12 calls the call-gate re-denomination reaches, and nothing else).
* ``revert-all``    -- the pre-2026-08-15 gates.
* ``live-mirror``   -- ``live`` under a second label. Its paired
  differences against ``live`` must be identically zero or the arms are
  not what they say they are.

Read beside ``gate-binding-audit-2026-08-26``, which prices how often
each edit changes a verdict on the stored live journal. This tool prices
what the changed verdicts are worth in BB/100; that one prices which real
hands they land on. Neither is sufficient alone.

Nothing here trains, harvests, promotes, deploys, or touches
``artifacts/approved.json``. Stdlib-only, seeded throughout, offline.

Example::

    python -m tools.gate_ablation --seeds 16 --workers 14 \\
        --output artifacts/evaluations/gate-ablation-2026-08-26.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.evaluate_policies import _gauntlet_workers, _run_tasks, paired_stats
from tools.evaluate_v8 import P3_CHANNEL, channel_battery_plan

DEFAULT_SUBJECT = "artifacts/candidates/candidate-v7-0001c.approved.manifest.json"

#: ``vs-p3`` is the subject; the other two are the card-blind controls.
CHANNELS: tuple[str, ...] = (P3_CHANNEL, "vs-median", "vs-station")

BASELINE = "live"
MIRROR = "live-mirror"


@dataclass(frozen=True)
class GateArm:
    """One gate configuration of a single policy, rebuildable in a worker.

    Mirrors the ``label``/``build()`` duck interface that
    ``evaluate_policies``' task machinery expects. The gates are applied
    to the *constructed* policy rather than threaded through the loader
    on purpose: a learned artifact rebuilds its gates from its own
    manifest, so overriding after construction is the only place where
    the edit is guaranteed to be the single difference between arms --
    and it leaves the live serve path untouched.
    """

    label: str
    manifest: str
    equity_trials: int
    risk_cap_on_effective_stack: bool
    call_gates_on_effective_stack: bool
    reveal_expense_equity_slope: float
    gate_stack_counts_committed_chips: bool = False
    pot_odds_exclude_uncallable: bool = False
    condition_range_without_price: bool = False

    def build(self) -> object:
        from devfun_poker_playground.learned_policy import load_policy

        policy = load_policy(self.manifest, equity_trials=self.equity_trials)
        policy.safety_gates = dataclasses.replace(
            policy.safety_gates,
            risk_cap_on_effective_stack=self.risk_cap_on_effective_stack,
            call_gates_on_effective_stack=self.call_gates_on_effective_stack,
            reveal_expense_equity_slope=self.reveal_expense_equity_slope,
            gate_stack_counts_committed_chips=(
                self.gate_stack_counts_committed_chips
            ),
            pot_odds_exclude_uncallable=self.pot_odds_exclude_uncallable,
            condition_range_without_price=self.condition_range_without_price,
        )
        return policy


def build_arms(args: argparse.Namespace) -> list[GateArm]:
    def arm(
        label: str,
        cap: bool,
        calls: bool,
        slope: float,
        *,
        committed: bool = False,
        winnable_price: bool = False,
        condition_unpriced: bool = False,
    ) -> GateArm:
        return GateArm(
            label=label,
            manifest=args.subject,
            equity_trials=args.equity_trials,
            risk_cap_on_effective_stack=cap,
            call_gates_on_effective_stack=calls,
            reveal_expense_equity_slope=slope,
            gate_stack_counts_committed_chips=committed,
            pot_odds_exclude_uncallable=winnable_price,
            condition_range_without_price=condition_unpriced,
        )

    return [
        arm(BASELINE, True, True, 0.12),
        arm("revert-cap", False, True, 0.12),
        arm("revert-calls", True, False, 0.0),
        arm("revert-all", False, False, 0.0),
        # Each repair is `live` exactly one field apart, so a contrast is
        # attributable to one edit. They are NOT combined into a single
        # arm: the range fix changes the equity every threshold sees and
        # the denominator fix changes what those thresholds divide by, so
        # a combined arm cannot be attributed.
        arm("fix-a", True, True, 0.12, committed=True),
        arm("winnable-price", True, True, 0.12, winnable_price=True),
        arm("condition-unpriced", True, True, 0.12, condition_unpriced=True),
        arm(MIRROR, True, True, 0.12),
    ]


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0] if values else 0.0), 0.0
    return statistics.mean(values), statistics.stdev(values)


def independent_seed_mde(values: Sequence[float]) -> dict:
    """``2 sigma / sqrt(n)`` -- the published noise-floor construction.

    The sigma is rounded to 2dp before dividing, which is not cosmetic:
    the published artifact was generated that way and
    ``tests/test_p3_gate.py`` pins the reproduction.
    """

    _, sd = _mean_sd(values)
    sigma = round(sd, 2)
    count = len(values)
    return {
        "sigma_bb_per_100": sigma,
        "mde_bb_per_100": round(2.0 * sigma / math.sqrt(count), 2) if count else None,
    }


def run_arms(args: argparse.Namespace) -> dict:
    arms = build_arms(args)
    seeds = tuple(range(args.seeds))
    plans = {
        arm.label: channel_battery_plan(
            arm,
            CHANNELS,
            seeds=seeds,
            scale=args.scale,
            stack=args.starting_stack,
            fit_path=args.p3_fit,
        )
        for arm in arms
    }
    tasks = [task for plan in plans.values() for _, _, task in plan]
    print(f"[gate_ablation] {len(arms)} arms x {len(CHANNELS)} channels "
          f"x {len(seeds)} seeds = {len(tasks)} matches", flush=True)

    started = time.time()
    outcomes = _run_tasks(tasks, _gauntlet_workers(args.workers, len(tasks)))
    elapsed = time.time() - started

    violations = sum(1 for outcome in outcomes if not outcome.chips_conserved)
    if violations:
        raise AssertionError(f"chip conservation violated in {violations} matches")

    per_arm: dict[str, dict] = {}
    cursor = 0
    for label, plan in plans.items():
        channels: dict[str, Any] = {}
        for channel in CHANNELS:
            entries = [item for item in plan if item[0] == channel]
            values = [
                outcomes[cursor + i].result.bb_per_100("hero")
                for i in range(len(entries))
            ]
            busts = [
                outcomes[cursor + i].result.busts.get("hero", 0)
                for i in range(len(entries))
            ]
            hero_hands = [
                (getattr(outcomes[cursor + i].result, "hands_by_agent", None) or {}).get(
                    "hero", outcomes[cursor + i].result.hands
                )
                for i in range(len(entries))
            ]
            cursor += len(entries)
            # Per-seed ruin, not just the pooled rate. A gate that buys
            # ruin reduction at an EV cost is the whole question here, and
            # a pooled rate cannot be tested against the same seeds the
            # BB/100 contrast uses -- it can only be eyeballed.
            bust_rates = [
                100.0 * bust / max(1, hands)
                for bust, hands in zip(busts, hero_hands, strict=True)
            ]
            channels[channel] = {
                "hands_per_seed": entries[0][1],
                "bb_per_100": round(statistics.mean(values), 2),
                "seeds": [round(value, 2) for value in values],
                "busts_per_100_hands": round(
                    100.0 * sum(busts) / max(1, sum(hero_hands)), 4
                ),
                "busts_per_100_by_seed": [round(rate, 4) for rate in bust_rates],
                "busts_total": sum(busts),
                **independent_seed_mde(values),
            }
        per_arm[label] = channels

    return {
        "elapsed_seconds": round(elapsed, 1),
        "matches": len(tasks),
        "chip_conservation_violations": violations,
        "seed_count": len(seeds),
        "arms": per_arm,
    }


def contrasts(per_arm: Mapping[str, Mapping[str, Any]]) -> dict:
    """Every arm against ``live``, paired on shared seeds."""

    out: dict[str, Any] = {}
    for label, channels in per_arm.items():
        if label == BASELINE:
            continue
        rows: dict[str, Any] = {}
        for channel in CHANNELS:
            base = per_arm[BASELINE][channel]["seeds"]
            other = channels[channel]["seeds"]
            diffs = [b - a for a, b in zip(base, other, strict=True)]
            stats = paired_stats(diffs)
            mde = round(2.0 * stats["sd"] / math.sqrt(len(diffs)), 2)
            if abs(stats["mean"]) < mde:
                verdict = "UNRESOLVED"
            elif stats["mean"] > 0:
                verdict = f"{label} ahead"
            else:
                verdict = f"{BASELINE} ahead"

            # The same paired construction on ruin. A positive mean means
            # the reverted arm BUSTS MORE, i.e. the change it reverts was
            # buying ruin reduction.
            ruin_diffs = [
                b - a
                for a, b in zip(
                    per_arm[BASELINE][channel]["busts_per_100_by_seed"],
                    channels[channel]["busts_per_100_by_seed"],
                    strict=True,
                )
            ]
            ruin = paired_stats(ruin_diffs)
            ruin_mde = round(2.0 * ruin["sd"] / math.sqrt(len(ruin_diffs)), 4)
            if abs(ruin["mean"]) < ruin_mde:
                ruin_verdict = "UNRESOLVED"
            elif ruin["mean"] > 0:
                ruin_verdict = f"{label} busts more"
            else:
                ruin_verdict = f"{label} busts less"

            rows[channel] = {
                **stats,
                "paired_mde": mde,
                "verdict": verdict,
                "ruin": {**ruin, "paired_mde": ruin_mde, "verdict": ruin_verdict},
            }
        out[label] = rows
    return out


#: The frozen artifact this tool must reproduce to be the same instrument.
FROZEN_REPORT = "artifacts/evaluations/p3-gate-2026-08-16.json"
FROZEN_ARM = "candidate-v7-0001c"


def reproduction_gate(report: Mapping[str, Any]) -> dict:
    """`live` must reproduce a frozen artifact's per-seed values exactly.

    ``p3_gate`` carries a check of this shape and this tool originally
    did not -- it kept only the null mirror and chip conservation, both of
    which are invariant to stack depth *by construction*. That omission is
    exactly why a 10bb run was read as a 60bb one: nothing in the report
    could fail on it. This check can.

    The ``live`` arm is the incumbent under its own manifest gates, so on
    the shared seeds it must equal ``p3-gate-2026-08-16``'s incumbent arm
    to the cent. Any difference in depth, equity trials, fit, or seed
    convention breaks it, which is the entire point.
    """

    frozen_path = Path(FROZEN_REPORT)
    if not frozen_path.exists():
        return {"verdict": "SKIP", "reason": f"{FROZEN_REPORT} not found"}
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    config = frozen.get("config", {})
    published = frozen.get("arms", {}).get("arms", {}).get(FROZEN_ARM)
    if not published:
        return {"verdict": "SKIP", "reason": f"no {FROZEN_ARM} arm in the artifact"}

    mine = report["arms"][BASELINE]
    channels: dict[str, Any] = {}
    for channel in CHANNELS:
        want = published.get(channel, {}).get("seeds") or []
        got = mine[channel]["seeds"]
        shared = min(len(want), len(got))
        identical = want[:shared] == got[:shared]
        channels[channel] = {
            "seeds_compared": shared,
            "identical": identical,
            "published_head": want[:3],
            "mine_head": got[:3],
        }

    matched = all(row["identical"] for row in channels.values())
    return {
        "against": FROZEN_REPORT,
        "arm": FROZEN_ARM,
        "published_config": {
            key: config.get(key)
            for key in ("starting_stack", "equity_trials", "scale", "seeds")
        },
        "channels": channels,
        "verdict": "PASS" if matched else "FAIL",
        "meaning": (
            "a FAIL means this run is not the instrument the frozen reports "
            "were produced on, and none of its numbers may be quoted beside "
            "them"
        ),
    }


def stage_instrument(report: Mapping[str, Any]) -> dict:
    """The wiring null, reported before any contrast is read."""

    mirror = report["contrasts"][MIRROR]
    nonzero = {
        channel: row["diffs"]
        for channel, row in mirror.items()
        if any(diff != 0 for diff in row["diffs"])
    }
    return {
        "null_mirror": {
            "channels_with_a_nonzero_difference": sorted(nonzero),
            "detail": nonzero,
            "verdict": "PASS" if not nonzero else "FAIL",
        },
        "chip_conservation": {
            "violations": report["chip_conservation_violations"],
            "verdict": "PASS" if not report["chip_conservation_violations"] else "FAIL",
        },
        "reproduction_gate": reproduction_gate(report),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Gate ablation on a card-aware opponent")
    add("")
    add(
        f"subject `{report['subject']}` · {report['seed_count']} seeds/arm · "
        f"{report['matches']} matches · {report['elapsed_seconds']}s"
    )
    add("")
    add(
        "> Every arm is the same artifact one gate edit apart. Nothing here "
        "trains, promotes or deploys."
    )
    add("")

    add("## 0. The instrument, before any result")
    add("")
    inst = report["instrument"]
    add("| check | result | verdict |")
    add("|---|---|---|")
    add(
        f"| null mirror (`{BASELINE}` under a second label) — every paired "
        "difference must be exactly 0 | "
        f"{len(inst['null_mirror']['channels_with_a_nonzero_difference'])} "
        f"channels differ | {inst['null_mirror']['verdict']} |"
    )
    add(
        f"| chip conservation | {inst['chip_conservation']['violations']} "
        f"violations | {inst['chip_conservation']['verdict']} |"
    )
    repro = inst["reproduction_gate"]
    if repro.get("verdict") == "SKIP":
        detail = repro.get("reason", "")
    else:
        detail = ", ".join(
            f"{channel} {'identical' if row['identical'] else 'DIFFERS'} "
            f"({row['seeds_compared']} seeds)"
            for channel, row in repro["channels"].items()
        )
    add(
        f"| reproduction gate — `{BASELINE}` vs the frozen "
        f"`p3-gate-2026-08-16` incumbent arm | {detail} | "
        f"{repro['verdict']} |"
    )
    add("")
    cfg = report["config"]
    add(
        f"Depth **{cfg['depth_bb']}bb** ({cfg['starting_stack']} chips at a "
        f"{cfg['big_blind']} big blind), {cfg['equity_trials']} equity trials, "
        f"scale {cfg['scale']}, fit `{cfg['p3_fit']}`. "
        f"Published instrument for comparison: "
        f"{repro.get('published_config', {})}."
    )
    add("")

    add("## 1. Arms")
    add("")
    header = "| arm | " + " | ".join(f"`{c}`" for c in CHANNELS) + " |"
    add(header)
    add("|---" * (len(CHANNELS) + 1) + "|")
    for label, channels in report["arms"].items():
        cells = " | ".join(
            f"{channels[c]['bb_per_100']:+.2f} (sd {channels[c]['sigma_bb_per_100']})"
            for c in CHANNELS
        )
        add(f"| {label} | {cells} |")
    add("")

    add(f"## 2. Every arm against `{BASELINE}`, paired on shared seeds")
    add("")
    for label, rows in report["contrasts"].items():
        if label == MIRROR:
            continue
        add(f"### `{label}`")
        add("")
        add("| channel | BB/100 difference | t | paired MDE | verdict |")
        add("|---|---|---|---|---|")
        for channel in CHANNELS:
            row = rows[channel]
            add(
                f"| `{channel}` | {row['mean']:+.2f} | {row['t']} | "
                f"{row['paired_mde']} | {row['verdict']} |"
            )
        add("")
        add("| channel | busts/100 difference | t | paired MDE | verdict |")
        add("|---|---|---|---|---|")
        for channel in CHANNELS:
            ruin = rows[channel]["ruin"]
            add(
                f"| `{channel}` | {ruin['mean']:+.4f} | {ruin['t']} | "
                f"{ruin['paired_mde']} | {ruin['verdict']} |"
            )
        add("")

    add("## Caveats")
    add("")
    add(
        "- P3's aggression and shove rate are still card-blind knobs. Only "
        "its folding is card-aware, so a policy can be punished for the "
        "prices it lays and still not for the hands it shows down."
    )
    add(
        "- `vs-p3` is heads-up. Multiway strength-aware play is untested."
    )
    add(
        "- A positive difference means the reverted arm scored higher, i.e. "
        "evidence *against* the change that arm reverts."
    )
    add(
        "- Batteries remain a fit diagnostic. A trivial floor still beats "
        "every real policy on these channels, so no arm's absolute BB/100 is "
        "evidence of generalisation."
    )
    return "\n".join(lines) + "\n"


def _utf8_stdout() -> None:
    """Windows consoles default to cp1252, which cannot encode the report.

    The file is always written UTF-8; this only keeps the echoed copy
    from killing an otherwise finished run.
    """

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - stream without it
        pass


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    # 6_000 = 60bb, the depth `p3_gate` and `evaluate_v8` both default to.
    # Matching them is not cosmetic: it is what makes a number here
    # comparable to the frozen reports. An earlier run of this tool used
    # 1_000 (10bb) and had to be discarded -- at 10bb a hero has almost no
    # room to cover the table, which is the ONLY condition under which
    # either re-denomination can change a verdict, so it measured the
    # gates in the one regime where they cannot act.
    parser.add_argument("--starting-stack", type=int, default=6_000)
    # 80 is the frozen instrument's value. Not inert: vs-p3 seed 0 at
    # stack 6000 moves 62.65 (80 trials) -> 81.73 (200), a shift larger
    # than the effect under measurement, and it breaks comparability with
    # every published artifact.
    parser.add_argument("--equity-trials", type=int, default=80)
    parser.add_argument("--p3-fit", default="artifacts/p3/p3-fit.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {
        "subject": args.subject,
        "channels": list(CHANNELS),
        "seed_convention": (
            "opponent seed 13+i, match seed 100+i, carry-over sessions, "
            "starting stack per --starting-stack — the published "
            "evaluate_policies.battery_tasks convention verbatim"
        ),
        # Recorded because its absence is what let a 10bb run be read as a
        # 60bb one. An artifact that cannot state its own depth cannot be
        # audited, and the depth IS the treatment variable for these edits.
        "config": {
            "starting_stack": args.starting_stack,
            "big_blind": 100,
            "depth_bb": round(args.starting_stack / 100, 1),
            "equity_trials": args.equity_trials,
            "scale": args.scale,
            "seeds": args.seeds,
            "p3_fit": args.p3_fit,
            "stacks": "carry-over-sessions",
        },
        **run_arms(args),
    }
    report["contrasts"] = contrasts(report["arms"])
    report["instrument"] = stage_instrument(report)

    text = render_markdown(report)
    print(text)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        out.with_suffix(".md").write_text(text, encoding="utf-8")
        print(f"[gate_ablation] wrote {out} and {out.with_suffix('.md')}")

    passed = all(
        item["verdict"] == "PASS" for item in report["instrument"].values()
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
