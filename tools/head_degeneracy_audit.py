"""How often a network head returns a constant, measured so it can fail.

The ``action_value`` head of ``candidate-v7-0001c`` returns its output
bias, bit for bit, on about a third of real decisions: the 64-unit tower
ReLUs to all zeros, so ``out = out_b`` regardless of the input. Those
rows produce **no folds at all**. Both of hero's bets in the hand that
ended the 2026-08-26 deployment were that constant -- its argmax is
``aggress_half_pot``, and half pot is exactly what was bet, twice.

Why this tool exists rather than a one-off script
-------------------------------------------------

A first attempt at repairing this was rejected because its detector was
circular. It tested "are all tower pre-activations <= 0", which stops
being measurable the moment a LayerNorm sits in front of the ReLU -- the
proposed fix -- and it read 0.00% on a control that must read 100%. It
would have passed a fix that did nothing.

So the predicate here is **head-output constancy**: does the head return
``out_b`` bit-identically? That is what actually matters (a constant
output carries no information), it is measurable under any architecture,
and it can be validated against controls with known answers. A softer
companion measure, per-coordinate output spread across rows, survives
even a head that is constant-but-not-at-its-bias.

Measure the instrument before the result
----------------------------------------

Three controls run first and their results are reported first. Each has
an answer that is forced by construction, and **the audit refuses to
report a result if any of them misses**:

* **Degenerate control** -- zero ``out_w``. The head can then only ever
  emit ``out_b``, so constancy MUST be 100%. A detector that reads
  anything else is not detecting constancy.
* **Live control** -- push ``tower_b`` far positive. Every tower unit is
  then active on every row, so constancy MUST be ~0%. A detector that
  still reads high is measuring something other than the tower.
* **Self-null** -- the same weights scored twice must agree exactly.

Stdlib-only, entirely offline, read-only. Nothing here trains, promotes,
deploys, or touches ``artifacts/approved.json``.

Example::

    python -m tools.head_degeneracy_audit \\
        --output artifacts/evaluations/head-degeneracy-2026-08-27.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from devfun_poker_playground.offline_trainer import _forward_v2

DEFAULT_MANIFEST = "artifacts/candidates/candidate-v7-0001c.approved.manifest.json"
DEFAULT_JOURNAL = ".arena-training.jsonl"


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_artifact(manifest_path: Path) -> tuple[dict, dict, list[float], list[float]]:
    """Architecture, weights, and the normalisation the serve path uses."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_file = manifest_path.parent / str(manifest["weights_file"])
    document = json.loads(weights_file.read_text(encoding="utf-8"))
    normalization = document["feature_normalization"]
    return (
        manifest["architecture"],
        document["weights"],
        list(normalization["means"]),
        list(normalization["stds"]),
    )


def load_rows(journal: Path, expected: int) -> list[dict]:
    """Stored decisions carrying a feature vector of the expected width.

    The journal records RAW features; the serve path normalises with the
    artifact's own means and stds before the forward pass, so this must
    too or every number below would describe a different network.
    """

    rows: list[dict] = []
    with journal.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except ValueError:
                continue
            if record.get("event") != "decision":
                continue
            features = record.get("features")
            if not isinstance(features, list) or len(features) != expected:
                continue
            rows.append(
                {
                    "features": features,
                    "street": str(record.get("street") or "").casefold(),
                    "to_call": ((record.get("legal") or {}).get("call_chips") or 0),
                    "policy_version": record.get("policy_version"),
                    "action": record.get("action"),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def head_outputs(
    architecture: Mapping[str, Any],
    weights: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    means: Sequence[float],
    stds: Sequence[float],
    head: str,
) -> list[list[float]]:
    out: list[list[float]] = []
    for row in rows:
        normalized = tuple(
            (value - mean) / std
            for value, mean, std in zip(row["features"], means, stds, strict=True)
        )
        out.append(_forward_v2(architecture, weights, normalized, heads=(head,))[head])
    return out


def constancy(outputs: Sequence[Sequence[float]], out_b: Sequence[float]) -> dict:
    """Share of rows whose head output IS the bias, bit for bit.

    Bit-identical rather than near: a head that merely lands close to its
    bias is still discriminating, and rounding the comparison would turn
    a live head into a dead one at some arbitrary tolerance.
    """

    bias = list(out_b)
    at_bias = sum(1 for row in outputs if list(row) == bias)
    spreads = []
    if outputs:
        for index in range(len(bias)):
            column = [row[index] for row in outputs]
            spreads.append(max(column) - min(column))
    return {
        "rows": len(outputs),
        "at_bias": at_bias,
        "constant_pct": round(100.0 * at_bias / len(outputs), 2) if outputs else 0.0,
        "output_spread": [round(value, 8) for value in spreads],
        "min_spread": round(min(spreads), 8) if spreads else None,
    }


def _zeroed_out_w(weights: Mapping[str, Any], head: str) -> dict:
    """Degenerate control: the head can only emit its bias."""

    copied = copy.deepcopy(dict(weights))
    block = copied["heads"][head]
    block["out_w"] = [[0.0] * len(row) for row in block["out_w"]]
    return copied


def _forced_live_tower(weights: Mapping[str, Any], head: str) -> dict:
    """Live control: every tower unit active on every row."""

    copied = copy.deepcopy(dict(weights))
    block = copied["heads"][head]
    block["tower_b"] = [1_000.0 for _ in block["tower_b"]]
    return copied


def stage_instrument(
    architecture: Mapping[str, Any],
    weights: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    means: Sequence[float],
    stds: Sequence[float],
    head: str,
) -> dict:
    out_b = weights["heads"][head]["out_b"]
    sample = rows[: min(len(rows), 400)]

    degenerate = constancy(
        head_outputs(architecture, _zeroed_out_w(weights, head), sample, means, stds, head),
        out_b,
    )
    live = constancy(
        head_outputs(
            architecture, _forced_live_tower(weights, head), sample, means, stds, head
        ),
        out_b,
    )
    first = head_outputs(architecture, weights, sample, means, stds, head)
    second = head_outputs(architecture, weights, sample, means, stds, head)

    checks = {
        "degenerate_control_zero_out_w": {
            "constant_pct": degenerate["constant_pct"],
            "must_be": 100.0,
            "verdict": "PASS" if degenerate["constant_pct"] == 100.0 else "FAIL",
        },
        "live_control_forced_tower": {
            "constant_pct": live["constant_pct"],
            "must_be": 0.0,
            "verdict": "PASS" if live["constant_pct"] == 0.0 else "FAIL",
        },
        "self_null": {
            "identical": first == second,
            "verdict": "PASS" if first == second else "FAIL",
        },
        "rows_in_controls": len(sample),
    }
    checks["all_passed"] = all(
        item["verdict"] == "PASS"
        for item in checks.values()
        if isinstance(item, dict) and "verdict" in item
    )
    return checks


def stage_result(
    architecture: Mapping[str, Any],
    weights: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    means: Sequence[float],
    stds: Sequence[float],
) -> dict:
    report: dict[str, Any] = {}
    for head in sorted(weights["heads"]):
        out_b = weights["heads"][head]["out_b"]
        outputs = head_outputs(architecture, weights, rows, means, stds, head)
        summary = constancy(outputs, out_b)
        bias = list(out_b)
        dead_flags = [list(row) == bias for row in outputs]

        by_street: dict[str, Any] = {}
        for street in ("preflop", "flop", "turn", "river"):
            picked = [
                flag
                for flag, row in zip(dead_flags, rows, strict=True)
                if row["street"] == street
            ]
            if picked:
                by_street[street] = {
                    "rows": len(picked),
                    "constant_pct": round(100.0 * sum(picked) / len(picked), 2),
                }

        unpriced = [
            flag
            for flag, row in zip(dead_flags, rows, strict=True)
            if (row["to_call"] or 0) <= 0
        ]
        priced = [
            flag
            for flag, row in zip(dead_flags, rows, strict=True)
            if (row["to_call"] or 0) > 0
        ]
        folds_on_constant = sum(
            1
            for flag, row in zip(dead_flags, rows, strict=True)
            if flag and row["action"] == "fold"
        )

        report[head] = {
            **summary,
            "out_b": [round(value, 9) for value in out_b],
            "by_street": by_street,
            "acts_first_constant_pct": (
                round(100.0 * sum(unpriced) / len(unpriced), 2) if unpriced else None
            ),
            "facing_a_bet_constant_pct": (
                round(100.0 * sum(priced) / len(priced), 2) if priced else None
            ),
            "folds_among_constant_rows": folds_on_constant,
        }
    return report


def render(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Head degeneracy audit")
    add("")
    add(
        f"artifact `{report['manifest']}` · journal `{report['journal']}` · "
        f"{report['rows']} decisions"
    )
    add("")
    add(
        "> Predicate: does the head return `out_b` **bit-identically**? A "
        "constant output carries no information, and unlike a "
        "pre-activation test this stays measurable if the architecture "
        "changes."
    )
    add("")
    add("## 0. The instrument, before any result")
    add("")
    inst = report["instrument"]
    add("| control | reads | must be | verdict |")
    add("|---|---|---|---|")
    for key, label in (
        ("degenerate_control_zero_out_w", "zeroed `out_w` — can only emit the bias"),
        ("live_control_forced_tower", "forced-live tower — can never die"),
    ):
        row = inst[key]
        add(
            f"| {label} | {row['constant_pct']}% | {row['must_be']}% | "
            f"{row['verdict']} |"
        )
    add(
        f"| self-null — same weights twice | "
        f"{'identical' if inst['self_null']['identical'] else 'DIFFERS'} | identical | "
        f"{inst['self_null']['verdict']} |"
    )
    add("")
    if not inst["all_passed"]:
        add("**Instrument failed. No result computed.**")
        return "\n".join(lines) + "\n"

    add("## 1. Per head")
    add("")
    add("| head | constant | acts first | facing a bet | folds among constant rows |")
    add("|---|---|---|---|---|")
    for head, row in report["heads"].items():
        add(
            f"| `{head}` | **{row['constant_pct']}%** "
            f"({row['at_bias']}/{row['rows']}) | "
            f"{row['acts_first_constant_pct']}% | "
            f"{row['facing_a_bet_constant_pct']}% | "
            f"{row['folds_among_constant_rows']} |"
        )
    add("")
    add("## 2. By street")
    add("")
    add("| head | " + " | ".join(("preflop", "flop", "turn", "river")) + " |")
    add("|---|---|---|---|---|")
    for head, row in report["heads"].items():
        cells = " | ".join(
            f"{row['by_street'].get(s, {}).get('constant_pct', '—')}%"
            if s in row["by_street"]
            else "—"
            for s in ("preflop", "flop", "turn", "river")
        )
        add(f"| `{head}` | {cells} |")
    add("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--journal", default=DEFAULT_JOURNAL)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    architecture, weights, means, stds = load_artifact(manifest_path)
    rows = load_rows(Path(args.journal), len(means))
    if not rows:
        raise SystemExit("no stored decisions carry a feature vector of that width")

    instrument = stage_instrument(
        architecture, weights, rows, means, stds, "action_value"
    )
    report: dict[str, Any] = {
        "manifest": str(manifest_path),
        "journal": args.journal,
        "rows": len(rows),
        "instrument": instrument,
    }
    if instrument["all_passed"]:
        report["heads"] = stage_result(architecture, weights, rows, means, stds)

    text = render(report)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        out.with_suffix(".md").write_text(text, encoding="utf-8")
        print(f"[head_degeneracy_audit] wrote {out} and {out.with_suffix('.md')}")
    return 0 if instrument["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
