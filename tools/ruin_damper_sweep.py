"""The C5 ruin-damper frontier: kappa_r swept on the frozen instrument.

Spec: ``engine/rules/README.md`` C5 — the damper's shape is Kelly-derived
and its SCALE is estimated, not authored: sweep ``kappa_r`` in the
battery and read the ruin-probability / BB-per-100 frontier. This tool is
that sweep, built as a sibling of ``tools.gate_ablation`` on the same
frozen instrument (60bb, equity trials 80, the published seed convention)
so its numbers sit beside the frozen reports rather than beside nothing.

Arms: ``damper-off`` (the incumbent, byte-identical to gate_ablation's
``live``), one arm per kappa_r grid point, and ``damper-off-mirror``
whose paired differences against the baseline must be identically zero.
Every arm is the same artifact; the ONLY difference is
``policy.rule_layer`` set after construction — the same
single-difference guarantee GateArm gets by overriding
``policy.safety_gates``.

Channels: the gate tool's trio (``vs-p3`` card-aware, ``vs-median`` and
``vs-station`` controls) plus ``vs-shover`` — the all-in-pressure channel
a ruin damper exists for. The reproduction gate compares the baseline's
trio against the subject's own frozen arm in ``p3-gate-2026-08-16``
per-seed to the cent; a FAIL means
this run is not the frozen instrument and nothing here may be quoted.

The damper at this instrument is ACTIVE from hand one: at a symmetric
60bb table the exposure equals hero's roll, so d starts at 1/kappa_r and
rises only as hero covers the table. That is the honest regime for the
sweep — the battery lives where the damper lives, unlike live median
depth (2,875bb) where d = 1 and the dial is inert.

Instrument checks, reported before any result is read: the selftest gate
(damping arithmetic at known states, arm-construction override
verified), the null mirror, chip conservation, and the reproduction
gate.

Nothing here trains, harvests, promotes, deploys, or touches
``artifacts/approved.json``. Stdlib-only, seeded, offline.

Example::

    python -m tools.ruin_damper_sweep --seeds 2 --workers 4   # smoke
    python -m tools.ruin_damper_sweep --seeds 32 --workers 14
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from engine.rules.composition import RuleLayerParams
from engine.rules.ruin_damper import RuinDamperParams, damping
from tools.evaluate_policies import _gauntlet_workers, _run_tasks, paired_stats
from tools.evaluate_v8 import P3_CHANNEL, channel_battery_plan

DEFAULT_SUBJECT = "artifacts/candidates/candidate-v7-0001c.approved.manifest.json"
BASELINE = "damper-off"
MIRROR = "damper-off-mirror"
KAPPA_GRID = (2.0, 3.0, 5.0, 8.0, 10.0)

#: The trio the reproduction gate covers, plus the all-in-pressure channel.
FROZEN_CHANNELS: tuple[str, ...] = (P3_CHANNEL, "vs-median", "vs-station")
CHANNELS: tuple[str, ...] = (*FROZEN_CHANNELS, "vs-shover")

FROZEN_REPORT = "artifacts/evaluations/p3-gate-2026-08-16.json"
#: The frozen arm the baseline must reproduce, per subject.
FROZEN_ARMS = {
    "heuristic": "heuristic-aggressive-v6",
    DEFAULT_SUBJECT: "candidate-v7-0001c",
}
OUTPUT_DIR = Path("artifacts") / "evaluations"


@dataclass(frozen=True)
class DamperArm:
    """One kappa_r configuration of a single policy, worker-rebuildable.

    ``manifest`` may be the sentinel ``"heuristic"``: the sweep's default
    subject is the noise-floor champion ``heuristic-aggressive-v6``, NOT
    the learned incumbent — measured 2026-08-29 (probe in the session
    record): the v7 head pins ``_branch_pot_fraction`` on every aggressive
    decision, so the temperature-sizer arm where C5 lives was consulted
    **zero times in 1,000 hands** and a v7-subject sweep is structurally
    a null. The heuristic champion sizes through the arm the damper
    cools. A kappa_r chosen here prices the MECHANISM on this instrument;
    re-sweep on the v9 composed lanes before enabling the dial on any v9
    artifact.
    """

    label: str
    manifest: str
    equity_trials: int
    kappa_r: float | None  # None = dial off (the incumbent)

    def build(self) -> object:
        if self.manifest == "heuristic":
            from engine.poker_policy import AggressivePokerPolicy

            policy = AggressivePokerPolicy(equity_trials=self.equity_trials)
        else:
            from engine.learned_policy import load_policy

            policy = load_policy(self.manifest, equity_trials=self.equity_trials)
        if self.kappa_r is not None:
            policy.rule_layer = RuleLayerParams(
                damper=RuinDamperParams(enabled=True, kappa_r=self.kappa_r)
            )
            # The single-difference guarantee, asserted rather than trusted.
            assert policy.rule_layer.damper.enabled
            assert policy.rule_layer.damper.kappa_r == self.kappa_r
            assert not policy.rule_layer.geometric.enabled
            assert not policy.rule_layer.snap.enabled
            assert not policy.rule_layer.commitment.enabled
            assert not policy.rule_layer.escalation.enabled
        return policy


def build_arms(args: argparse.Namespace) -> list[DamperArm]:
    arms = [DamperArm(BASELINE, args.subject, args.equity_trials, None)]
    arms.extend(
        DamperArm(f"damper-kr-{value:g}", args.subject, args.equity_trials, value)
        for value in KAPPA_GRID
    )
    arms.append(DamperArm(MIRROR, args.subject, args.equity_trials, None))
    return arms


def selftest(verbose: bool = True) -> None:
    """Damping arithmetic at known states; refuses to run on failure."""

    on = RuinDamperParams(enabled=True, kappa_r=5.0)
    d_equal = damping(on, bankroll=6_000, exposure=6_000).d
    if abs(d_equal - 0.2) > 1e-12:
        raise AssertionError(f"equal-stack d must be 1/kappa_r, got {d_equal}")
    if damping(on, bankroll=60_000, exposure=6_000).d != 1.0:
        raise AssertionError("10x coverage must not damp at kappa_r=5")
    if damping(on, bankroll=6_000, exposure=0).d != 1.0:
        raise AssertionError("zero exposure must not damp")
    off = RuinDamperParams()
    if damping(off, bankroll=1, exposure=1_000_000).d != 1.0:
        raise AssertionError("the off dial must never damp")
    if verbose:
        print(f"selftest PASS: d(equal stacks, kr=5) = {d_equal}")


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
    print(
        f"[ruin_damper_sweep] {len(arms)} arms x {len(CHANNELS)} channels "
        f"x {len(seeds)} seeds = {len(tasks)} matches",
        flush=True,
    )
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
            bust_rates = [
                100.0 * bust / max(1, hands)
                for bust, hands in zip(busts, hero_hands, strict=True)
            ]
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            sigma = round(sd, 2)
            channels[channel] = {
                "hands_per_seed": entries[0][1],
                "bb_per_100": round(statistics.mean(values), 2),
                "seeds": [round(value, 2) for value in values],
                "busts_per_100_hands": round(
                    100.0 * sum(busts) / max(1, sum(hero_hands)), 4
                ),
                "busts_per_100_by_seed": [round(rate, 4) for rate in bust_rates],
                "busts_total": sum(busts),
                "sigma_bb_per_100": sigma,
                "mde_bb_per_100": (
                    round(2.0 * sigma / math.sqrt(len(values)), 2) if values else None
                ),
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
    """Every arm against ``damper-off``, paired on shared seeds."""

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
            # <= , not <: a zero difference at a zero MDE is the NULL
            # MIRROR's signature, and `<` classified it as a resolved
            # directional win — the control the instrument exists to check.
            if abs(stats["mean"]) <= mde:
                verdict = "UNRESOLVED"
            elif stats["mean"] > 0:
                verdict = f"{label} ahead"
            else:
                verdict = f"{BASELINE} ahead"
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
            if abs(ruin["mean"]) <= ruin_mde:
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


def reproduction_gate(report: Mapping[str, Any]) -> dict:
    """``damper-off`` must equal the frozen subject arm per-seed, to the cent."""

    frozen_arm = FROZEN_ARMS.get(report.get("subject"))
    if frozen_arm is None:
        return {"verdict": "SKIP", "reason": "no frozen arm for this subject"}
    frozen_path = Path(FROZEN_REPORT)
    if not frozen_path.exists():
        return {"verdict": "SKIP", "reason": f"{FROZEN_REPORT} not found"}
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    published = frozen.get("arms", {}).get("arms", {}).get(frozen_arm)
    if not published:
        return {"verdict": "SKIP", "reason": f"no {frozen_arm} arm in the artifact"}
    mine = report["arms"][BASELINE]
    channels: dict[str, Any] = {}
    for channel in FROZEN_CHANNELS:
        want = published.get(channel, {}).get("seeds") or []
        got = mine[channel]["seeds"]
        shared = min(len(want), len(got))
        channels[channel] = {
            "seeds_compared": shared,
            # A channel with nothing to compare is NOT a pass: an empty
            # frozen arm would otherwise satisfy a gate whose stated
            # meaning is "this run IS the frozen instrument".
            "identical": shared > 0 and want[:shared] == got[:shared],
        }
    matched = all(row["identical"] for row in channels.values())
    return {
        "against": FROZEN_REPORT,
        "arm": frozen_arm,
        "channels": channels,
        "verdict": "PASS" if matched else "FAIL",
        "meaning": (
            "a FAIL means this run is not the frozen instrument and none of "
            "its numbers may be quoted beside the frozen reports"
        ),
    }


def stage_instrument(report: Mapping[str, Any]) -> dict:
    mirror = report["contrasts"][MIRROR]
    nonzero = {
        channel: row["diffs"]
        for channel, row in mirror.items()
        if any(diff != 0 for diff in row["diffs"])
    }
    return {
        "null_mirror": {
            "channels_with_a_nonzero_difference": sorted(nonzero),
            "verdict": "PASS" if not nonzero else "FAIL",
        },
        "chip_conservation": {
            "violations": report["chip_conservation_violations"],
            "verdict": "PASS" if not report["chip_conservation_violations"] else "FAIL",
        },
        "reproduction_gate": reproduction_gate(report),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Default subject is the HEURISTIC champion, not the learned
    # incumbent — see DamperArm's docstring for the measured reason.
    parser.add_argument("--subject", default="heuristic")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    # The frozen instrument's values — see gate_ablation for why both
    # numbers are load-bearing (a 10bb run was discarded; 80 vs 200
    # trials moves a seed by more than the effects under measurement).
    parser.add_argument("--starting-stack", type=int, default=6_000)
    parser.add_argument("--equity-trials", type=int, default=80)
    parser.add_argument("--p3-fit", default="artifacts/p3/p3-fit.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--selftest-only", action="store_true")
    args = parser.parse_args(argv)

    selftest()
    if args.selftest_only:
        return 0

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tool": "tools.ruin_damper_sweep",
        "subject": args.subject,
        "kappa_grid": list(KAPPA_GRID),
        "channels": list(CHANNELS),
        "config": {
            "starting_stack": args.starting_stack,
            "big_blind": 100,
            "depth_bb": round(args.starting_stack / 100, 1),
            "equity_trials": args.equity_trials,
            "scale": args.scale,
            "seeds": args.seeds,
            "p3_fit": args.p3_fit,
        },
    }
    report.update(run_arms(args))
    report["contrasts"] = contrasts(report["arms"])
    report["instrument"] = stage_instrument(report)

    output = Path(
        args.output
        or OUTPUT_DIR / f"ruin-damper-sweep-{datetime.now(UTC).date()}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    instrument = report["instrument"]
    print(f"\ninstrument: mirror {instrument['null_mirror']['verdict']}, "
          f"chips {instrument['chip_conservation']['verdict']}, "
          f"reproduction {instrument['reproduction_gate']['verdict']}")
    for label in (f"damper-kr-{value:g}" for value in KAPPA_GRID):
        row = report["contrasts"][label][P3_CHANNEL]
        shover = report["contrasts"][label]["vs-shover"]
        print(
            f"  {label:14} vs-p3 dBB/100 {row['mean']:+7.2f} (mde {row['paired_mde']}) "
            f"{row['verdict']:22} | vs-shover ruin d {shover['ruin']['mean']:+8.4f} "
            f"(mde {shover['ruin']['paired_mde']}) {shover['ruin']['verdict']}"
        )
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
