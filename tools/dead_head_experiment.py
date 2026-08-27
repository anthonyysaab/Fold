"""Run the degenerate-group retrain experiment and apply the frozen rule.

The decision rule, its thresholds, and every VOID gate were fixed before any
run existed, in
``artifacts/evaluations/dead-head-retrain-prereg-2026-08-27.md``. Nothing here
recomputes a threshold from the data it judges.

Three arms, one field apart:

* ``control``     ``--degenerate-group-filter off``          the incumbent recipe
* ``treated``     ``--degenerate-group-filter zero_weight``  the intervention
* ``attribution`` ``--degenerate-group-filter random``       size-matched control

Runs are **sequential**: each needs ~8 GiB of host RAM, and two at once would
page and invalidate every timing. GPU is not the constraint (897 MiB of 6,144).

This trains and measures. It does **not** promote or deploy, and it never
contacts the Arena. A clean result produces candidates, nothing more.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CUDA_PYTHON = r"C:\Users\user\poker-nn-training\.venv\Scripts\python.exe"
STDLIB_PYTHON = r"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"
CORPUS = "artifacts/corpora/candidate-v7-0001.corpus"
OUTPUT_DIR = "artifacts/deadhead"

#: Frozen in the pre-registration. Never recomputed from the runs.
PAIRED_DIFFERENCE_SD = 9.223
BOUNDARY = {5: 8.25, 8: 6.52}

#: Group counts `reward_batch_loss` actually iterates, per split.
EXPECTED_GROUPS = {
    "control": {"train": 39_996, "validation": 10_045},
    "treated": {"train": 28_143, "validation": 7_056},
}

ARMS = {
    "control": ["--degenerate-group-filter", "off"],
    "treated": ["--degenerate-group-filter", "zero_weight"],
    "attribution": ["--degenerate-group-filter", "random"],
}

RECIPE = [
    "--examples-in", CORPUS,
    "--architecture", "v7",
    "--device", "cuda",
    "--epochs", "40",
    "--batch-size", "256",
    "--learning-rate", "0.001",
    "--return-scale-pct", "50",
    "--reinforcement-multiplier", "1.0",
    "--counterfactual-rollouts", "1",
    "--split-seed", "17",
]


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass


def train(arm: str, seed: int, *, dry_run: bool = False) -> dict[str, Any]:
    version = f"deadhead-{arm}-s{seed}"
    command = [
        CUDA_PYTHON, "-m", "tools.self_play_cycle",
        *RECIPE,
        "--init-seed", str(seed),
        "--model-version", version,
        "--output-dir", OUTPUT_DIR,
        *ARMS[arm],
    ]
    if arm == "attribution":
        command += ["--degenerate-group-filter-seed", str(seed)]
    if dry_run:
        command.append("--dry-run")
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    return {
        "arm": arm,
        "seed": seed,
        "model_version": version,
        "returncode": result.returncode,
        "seconds": round(time.time() - started, 1),
        "stderr_tail": result.stderr[-1500:] if result.returncode else "",
    }


def audit(version: str) -> dict[str, Any]:
    manifest = f"{OUTPUT_DIR}/candidates/{version}.manifest.json"
    if not Path(manifest).exists():
        alt = list(Path(OUTPUT_DIR).rglob(f"{version}.manifest.json"))
        if not alt:
            return {"error": f"no manifest for {version}"}
        manifest = str(alt[0])
    out = f"{OUTPUT_DIR}/audits/{version}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            STDLIB_PYTHON, "-m", "tools.head_degeneracy_audit",
            "--manifest", manifest, "--output", out,
        ],
        capture_output=True, text=True, errors="replace",
    )
    if not Path(out).exists():
        return {"error": "audit produced no report", "stderr": result.stderr[-800:]}
    report = json.loads(Path(out).read_text(encoding="utf-8"))
    heads = report.get("heads") or {}
    action = heads.get("action_value") or {}
    return {
        "manifest": manifest,
        "instrument_passed": report["instrument"]["all_passed"],
        "constant_pct": action.get("constant_pct"),
        "min_spread": action.get("min_spread"),
        "state_value_constant_pct": (heads.get("state_value") or {}).get("constant_pct"),
    }


def paired(control: list[float], other: list[float]) -> dict[str, Any]:
    diffs = [c - o for c, o in zip(control, other, strict=True)]
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    half = 2.0 * sd / (len(diffs) ** 0.5) if len(diffs) > 1 else float("inf")
    return {
        "diffs": [round(value, 2) for value in diffs],
        "mean": round(mean, 3),
        "sd": round(sd, 3),
        "ci_half_width": round(half, 3),
    }


def decide(rows: list[dict], seeds: int) -> dict[str, Any]:
    """The frozen rule. Thresholds come from the table, never the data."""

    boundary = BOUNDARY.get(seeds)
    by_arm: dict[str, list[float]] = {}
    for row in rows:
        if row.get("constant_pct") is None:
            continue
        by_arm.setdefault(row["arm"], []).append(row["constant_pct"])

    voids: list[str] = []
    for arm, values in by_arm.items():
        if len(values) != seeds:
            voids.append(f"{arm}: {len(values)} usable runs, expected {seeds}")
    if not any(r.get("instrument_passed") for r in rows):
        voids.append("no arm passed the audit's own controls")
    for row in rows:
        if row.get("instrument_passed") is False:
            voids.append(f"{row['model_version']}: audit controls FAILED")
        if row.get("constant_pct") == 0.0 and row.get("min_spread") == 0.0:
            voids.append(f"{row['model_version']}: constant at a non-bias value")
    control = by_arm.get("control") or []
    if len(set(control)) <= 1 and len(control) > 1:
        voids.append("control arm shows zero seed variation")

    if voids or boundary is None:
        return {
            "verdict": "VOID",
            "reasons": voids or [f"no frozen boundary for n={seeds}"],
        }

    treated = paired(control, by_arm["treated"])
    attribution = paired(by_arm["attribution"], by_arm["treated"])
    D, A = treated["mean"], attribution["mean"]

    if D <= 0:
        verdict = "REFUTED"
    elif D > boundary and A > boundary:
        verdict = "CONFIRMED"
    elif D > boundary and attribution["ci_half_width"] < boundary:
        verdict = "REFUTED_BY_SIZE"
    elif D > boundary:
        verdict = "UNRESOLVED_PENDING_ATTRIBUTION"
    else:
        verdict = "UNRESOLVED"
    return {
        "verdict": verdict,
        "boundary": boundary,
        "seeds": seeds,
        "control_minus_treated": treated,
        "random_minus_treated": attribution,
        "levels": {arm: values for arm, values in by_arm.items()},
    }


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[89, 90, 91, 92, 93])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--output", default="artifacts/evaluations/dead-head-retrain-2026-08-27.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--decide-only", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.output)
    rows: list[dict] = []
    if args.decide_only and out.exists():
        rows = json.loads(out.read_text(encoding="utf-8"))["runs"]
    else:
        total = len(args.arms) * len(args.seeds)
        index = 0
        for seed in args.seeds:
            for arm in args.arms:
                index += 1
                print(f"[{index}/{total}] {arm} seed {seed} ...", flush=True)
                row = train(arm, seed, dry_run=args.dry_run)
                if row["returncode"] == 0 and not args.dry_run:
                    row.update(audit(row["model_version"]))
                print(
                    f"    rc={row['returncode']} {row['seconds']}s "
                    f"constant={row.get('constant_pct')}",
                    flush=True,
                )
                rows.append(row)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    json.dumps({"runs": rows}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

    verdict = decide(rows, len(args.seeds))
    payload = {"runs": rows, "decision": verdict}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print()
    print(json.dumps(verdict, indent=2))
    print(f"\n[dead_head_experiment] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
