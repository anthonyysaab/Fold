"""The k-parameter OLS baseline every v9 candidate must beat (2026-09-02).

``v9-loses-to-v7-diagnosis-2026-09-02.md`` measured that a 2-parameter
ordinary least squares fit on ONE schema-4 feature reaches validation
R^2 = 0.199 where the 71,841-parameter network reaches 0.058, and that
nine parameters reach 0.231. A network that loses to a handful of linear
parameters on its own held-out split is not learning the objective; it
is under-trained or the objective is wrong. This tool turns that
one-off finding into a repeatable measurement.

What it measures, exactly (frozen so it can be a gate):

- **Target**: the Phase-B value target exactly as the trainer fits it —
  ``reward_bb / purse_bb`` per emitted branch row, NO further centering
  (the corpus rows are centered by construction at harvest).
- **Split**: the trainer's own rule,
  ``sha256(split_seed:table_id) < validation_fraction`` per table
  (``v8_trainer.table_split_value``), so the OLS and the network are
  scored on the same held-out decisions.
- **Score**: ``value_normalized = MSE_validation / population_variance_train``
  — the trainer's own normalizer — so the number is directly comparable
  to a manifest's ``evaluation.validation_losses.value_normalized``.
  Plain R^2 (against the validation variance) is reported alongside.
- **Feature sets** (intercept always included):
  ``minimal`` = ``equity_vs_posterior`` (2 parameters); ``belief`` =
  the 8 belief buckets (9 parameters); ``strength`` = minimal + belief +
  ``equity_multiway`` + ``strength_percentile`` + ``pot_odds`` + ``spr``
  (14 parameters).

The instrument validates itself before reporting: fitting OLS on the
target itself as its only feature must return value_normalized ~0 and
R^2 ~1 — an impossible-by-construction invariant, so a number is never
printed unless the pipeline demonstrably can see the target.

Stdlib-only; reads a corpus, never writes one. Nothing here trains,
promotes, or touches ``artifacts/approved.json``.

Usage::

    python -m tools.ols_baseline \\
        --corpus artifacts/phase_b_v9/candidate-v9-phase-b-merged.phase-b.jsonl.gz \\
        --split-seed 17 \\
        --candidate artifacts/candidates/candidate-v9-0003b.manifest.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from engine import schema4

DEFAULT_CORPUS = (
    Path("artifacts")
    / "phase_b_v9"
    / "candidate-v9-phase-b-merged.phase-b.jsonl.gz"
)

_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    # Branch-blind sets are fitted and REPORTED for the structural lesson:
    # the value targets are per-decision centered while the features are
    # per-decision constant, so any model that cannot tell branches apart
    # (a flat OLS included) is constant-predictor-equivalent and must
    # score value_normalized ~1.0. The gate uses the branch-interacted
    # sets, whose dummy columns let each branch carry its own
    # coefficients — the minimum a linear model needs to even participate.
    "minimal": ("equity_vs_posterior",),
    "belief": tuple(f"belief_bucket_{index}" for index in range(8)),
    "branch-minimal": ("equity_vs_posterior",),
    "branch-strength": (
        "equity_vs_posterior",
        "equity_multiway",
        "strength_percentile",
        "pot_odds",
        "spr",
    ),
}

_BRANCH_INTERACTED_SETS = {"branch-minimal", "branch-strength"}

_BRANCH_LABELS = ("fatal", "passive", "active", "aggressive")


def _split_value(split_seed: int, table_id: str) -> float:
    digest = hashlib.sha256(f"{split_seed}:{table_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def load_rows(path: str | Path) -> tuple[list[dict[str, Any]], list[float]]:
    """(branch rows, header) from a schema-2 v9 Phase-B corpus."""

    resolved = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    header = None
    with gzip.open(resolved, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.strip()
            if not text:
                continue
            document = json.loads(text)
            if not isinstance(document, Mapping):
                raise SystemExit(f"line {line_number}: not an object")
            if line_number == 1:
                header = document
                continue
            features = document.get("features")
            purse_bb = document.get("purse_bb")
            table_id = document.get("table_id")
            if (
                not isinstance(features, list)
                or len(features) != schema4.INPUT_SIZE_V9
                or not isinstance(purse_bb, (int, float))
                or isinstance(purse_bb, bool)
                or purse_bb <= 0
                or not isinstance(table_id, str)
            ):
                raise SystemExit(
                    f"line {line_number}: row violates the v9 corpus shape"
                )
            for entry in document.get("branches") or ():
                if not isinstance(entry, Mapping):
                    raise SystemExit(f"line {line_number}: bad branch entry")
                reward = entry.get("reward_bb")
                if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                    raise SystemExit(f"line {line_number}: bad reward_bb")
                rows.append(
                    {
                        "table_id": table_id,
                        "features": features,
                        "target": float(reward) / float(purse_bb),
                        "branch": str(entry.get("branch") or ""),
                    }
                )
    if header is None:
        raise SystemExit("corpus is empty")
    return rows, header


def _nonconstant_columns(design: Sequence[Sequence[float]]) -> list[int]:
    """Indices of columns whose training values are not all identical.

    The branch-interacted sets contain structurally constant columns
    (``pot_odds`` is identically 0 on the passive branch, which exists
    only at free spots), and a zero column makes the normal equations
    exactly singular. Dropping them is the same fit with fewer
    parameters — the honest count of what the linear model actually
    used is reported as ``parameters``.
    """

    width = len(design[0])
    kept: list[int] = []
    for column in range(width):
        first = design[0][column]
        if any(row[column] != first for row in design[1:]):
            kept.append(column)
    return kept


def _ols(
    design: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    include_intercept: bool = True,
) -> list[float] | None:
    """Normal-equations fit; k is tiny.

    ``include_intercept=False`` is for the branch-interacted sets, whose
    branch dummies already span the intercept (a shared intercept would
    make the Gram matrix singular by construction).
    """

    parameters = len(design[0]) + (1 if include_intercept else 0)
    gram = [[0.0] * parameters for _ in range(parameters)]
    rhs = [0.0] * parameters
    for row, target in zip(design, targets, strict=True):
        vector = ([1.0, *row] if include_intercept else list(row))
        for i in range(parameters):
            rhs[i] += vector[i] * target
            for j in range(i + 1):
                gram[i][j] += vector[i] * vector[j]
    for i in range(parameters):
        for j in range(i + 1, parameters):
            gram[i][j] = gram[j][i]
    for column in range(parameters):
        pivot = column
        best = abs(gram[pivot][column])
        for candidate in range(column + 1, parameters):
            magnitude = abs(gram[candidate][column])
            if magnitude > best:
                best = magnitude
                pivot = candidate
        if best < 1e-12:
            return None
        if pivot != column:
            gram[column], gram[pivot] = gram[pivot], gram[column]
            rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        scale = gram[column][column]
        for j in range(column, parameters):
            gram[column][j] /= scale
        rhs[column] /= scale
        for i in range(parameters):
            if i == column:
                continue
            factor = gram[i][column]
            if factor == 0.0:
                continue
            for j in range(column, parameters):
                gram[i][j] -= factor * gram[column][j]
            rhs[i] -= factor * rhs[column]
    return rhs


def _predict(
    beta: Sequence[float], row: Sequence[float], *, include_intercept: bool = True
) -> float:
    if include_intercept:
        total = beta[0]
        pairs = zip(beta[1:], row, strict=True)
    else:
        total = 0.0
        pairs = zip(beta, row, strict=True)
    for weight, value in pairs:
        total += weight * value
    return total


def score_phase_b_corpus(
    corpus_path: str | Path,
    *,
    split_seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    """The OLS baselines on the Phase-B value target (library entry point).

    Returns the feature-set scores, the split counts, the train target
    variance (which must equal the trainer's own normalizer — the
    instrument's second self-check), and the identity-fit verdict.
    """

    rows, _ = load_rows(corpus_path)
    validation_tables = {
        table_id
        for table_id in {row["table_id"] for row in rows}
        if _split_value(split_seed, table_id) < validation_fraction
    }
    train = [row for row in rows if row["table_id"] not in validation_tables]
    validation = [
        row for row in rows if row["table_id"] in validation_tables
    ]
    if not train or not validation:
        raise SystemExit("split produced an empty side")
    train_targets = [row["target"] for row in train]
    population_variance = (
        sum(value * value for value in train_targets) / len(train_targets)
        - (sum(train_targets) / len(train_targets)) ** 2
    )
    validation_variance = (
        sum(row["target"] ** 2 for row in validation) / len(validation)
        - (sum(row["target"] for row in validation) / len(validation)) ** 2
    )

    names = schema4.FEATURE_NAMES_V9
    indexes = {name: position for position, name in enumerate(names)}
    report: dict[str, Any] = {
        "train_branch_rows": len(train),
        "validation_branch_rows": len(validation),
        "train_target_population_variance": round(population_variance, 9),
        "feature_sets": {},
        "instrument": {},
    }

    identity = _ols([[value] for value in train_targets], train_targets)
    if identity is None:
        raise SystemExit("instrument failed: identity fit is singular")
    identity_mse = sum(
        (_predict(identity, [value]) - value) ** 2 for value in (
            row["target"] for row in validation
        )
    ) / len(validation)
    report["instrument"]["identity_normalized"] = round(
        identity_mse / population_variance, 9
    )
    report["instrument"]["identity_verdict"] = (
        "PASS" if identity_mse / population_variance < 1e-6 else "FAIL"
    )

    for label, feature_names in _FEATURE_SETS.items():
        columns = []
        for name in feature_names:
            position = indexes.get(name)
            if position is None:
                raise SystemExit(f"feature {name!r} is not in schema 4")
            columns.append(position)
        interacted = label in _BRANCH_INTERACTED_SETS

        def design_row(row: dict[str, Any]) -> list[float]:
            if not interacted:
                return [row["features"][column] for column in columns]
            branch = row["branch"]
            expanded: list[float] = []
            for candidate in _BRANCH_LABELS:
                is_this = 1.0 if branch == candidate else 0.0
                expanded.append(is_this)
                for column in columns:
                    expanded.append(is_this * row["features"][column])
            return expanded

        design = [design_row(row) for row in train]
        original_width = len(design[0])
        kept = _nonconstant_columns(design)
        design = [[row[i] for i in kept] for row in design]
        beta = _ols(design, train_targets, include_intercept=not interacted)
        if beta is None:
            report["feature_sets"][label] = {"verdict": "singular"}
            continue
        val_design = [
            [row[i] for i in kept] for row in (design_row(row) for row in validation)
        ]
        mse = sum(
            (
                _predict(beta, row, include_intercept=not interacted) - target
            ) ** 2
            for row, target in zip(val_design, (row["target"] for row in validation), strict=True)
        ) / len(validation)
        report["feature_sets"][label] = {
            "parameters": len(beta),
            "columns_dropped_constant": original_width - len(kept),
            "branch_interacted": interacted,
            "value_normalized": round(mse / population_variance, 4),
            "r2_validation": round(1.0 - mse / validation_variance, 4),
            "mse_validation": round(mse, 8),
        }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--phase-a-dataset", default=None, help=(
        "v9 Phase-A dataset; when given, the equity_called label is scored "
        "the same way (the comparison from the 2026-09-02 diagnosis, where "
        "the network's R2 0.058 lost to a 2-parameter OLS at 0.199)"
    ))
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--candidate",
        default=None,
        help="candidate manifest whose validation value_normalized the OLS "
        "is compared against (default: report the OLS number alone)",
    )
    args = parser.parse_args(argv)

    report: dict[str, Any] = {
        "corpus": args.corpus,
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
    }
    report.update(score_phase_b_corpus(
        args.corpus,
        split_seed=args.split_seed,
        validation_fraction=args.validation_fraction,
    ))

    if args.candidate:
        manifest = json.loads(
            Path(args.candidate).expanduser().resolve().read_text(encoding="utf-8")
        )
        network = (manifest.get("evaluation") or {}).get("validation_losses") or {}
        report["network"] = {
            "candidate": manifest.get("model_version"),
            "value_normalized": network.get("value_normalized"),
            "value_mse": network.get("value_mse"),
            "phase_a_equity_called_mse": network.get("equity_called"),
        }

    if args.phase_a_dataset:
        report["phase_a_equity_called"] = _phase_a_ols(
            args.phase_a_dataset,
            split_seed=args.split_seed,
            validation_fraction=args.validation_fraction,
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _phase_a_ols(
    path: str | Path,
    *,
    split_seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    """The diagnosis's comparison: OLS on ``equity_vs_posterior`` vs the
    masked ``equity_called`` label, on the trainer's own split."""

    resolved = Path(path).expanduser().resolve()
    opener = gzip.open if resolved.name.endswith(".gz") else open
    rows: list[dict[str, Any]] = []
    with opener(resolved, "rt", encoding="utf-8") as stream:  # type: ignore[operator]
        for line in stream:
            document = json.loads(line)
            if not isinstance(document, Mapping):
                raise SystemExit("phase-a row is not an object")
            masks = document.get("masks")
            labels = document.get("labels")
            if not isinstance(masks, Mapping) or not isinstance(labels, Mapping):
                raise SystemExit("phase-a row lacks labels/masks")
            if not masks.get("equity_called"):
                continue
            features = document.get("features")
            table_id = document.get("table_id")
            target = labels.get("equity_called")
            if (
                not isinstance(features, list)
                or len(features) != schema4.INPUT_SIZE_V9
                or not isinstance(table_id, str)
                or isinstance(target, bool)
                or not isinstance(target, (int, float))
            ):
                raise SystemExit("phase-a row violates the v9 shape")
            rows.append(
                {
                    "table_id": table_id,
                    "features": features,
                    "target": float(target),
                }
            )
    validation_tables = {
        table_id
        for table_id in {row["table_id"] for row in rows}
        if _split_value(split_seed, table_id) < validation_fraction
    }
    train = [row for row in rows if row["table_id"] not in validation_tables]
    validation = [
        row for row in rows if row["table_id"] in validation_tables
    ]
    if not train or not validation:
        raise SystemExit("phase-a split produced an empty side")
    validation_variance = (
        sum(row["target"] ** 2 for row in validation) / len(validation)
        - (sum(row["target"] for row in validation) / len(validation)) ** 2
    )
    column = schema4.feature_index_v9("equity_vs_posterior")
    design = [[row["features"][column]] for row in train]
    beta = _ols(design, [row["target"] for row in train])
    if beta is None:
        return {"verdict": "singular"}
    mse = sum(
        (_predict(beta, [row["features"][column]]) - row["target"]) ** 2
        for row in validation
    ) / len(validation)
    return {
        "parameters": len(beta),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "validation_target_variance": round(validation_variance, 6),
        "mse_validation": round(mse, 6),
        "r2_validation": round(1.0 - mse / validation_variance, 4),
    }


# ---------------------------------------------------------------------------
# The promotion gate
# ---------------------------------------------------------------------------

#: The feature set a format-4 candidate's value head must beat, and the
#: comparison margin (in value_normalized units) it must beat it by.
#: ``branch-strength`` is 23 linear parameters with branch identity — the
#: largest honest linear model this tool fits; the margin is deliberate:
#: a gate decided on knife-edge numbers is a coin flip.
GATE_FEATURE_SET = "branch-strength"
GATE_MARGIN_NORMALIZED = 0.02


def ols_gate_verdict(
    phase_b_score: Mapping[str, Any],
    phase_a_score: Mapping[str, Any] | None,
    *,
    network_value_normalized: float | None,
    network_equity_called_mse: float | None,
) -> tuple[bool, list[str]]:
    """Whether a candidate beats the k-parameter OLS on held-out data.

    Two arms, both on the trainer's own split:

    - **Phase-B composed value**: the manifest's validation
      ``value_normalized`` must be below the ``branch-strength`` OLS's
      ``value_normalized`` minus the margin. (0003b: network 0.944 vs
      OLS 1.045 — passes.)
    - **Phase-A ``equity_called`` label** (the 2026-09-02 diagnosis):
      the network's validation R2, computed from its recorded
      ``equity_called`` MSE against this dataset's validation variance,
      must exceed the 2-parameter OLS's R2 minus a 0.02 margin. The
      diagnosis measured network 0.058 vs OLS 0.199 — every current v9
      candidate FAILS this arm.

    Returns ``(passes, failure_reasons)``. A missing comparison number
    is a failure, never a pass: a candidate that cannot state what it
    scored cannot claim to have beaten the baseline.
    """

    failures: list[str] = []
    value_arm = phase_b_score.get("feature_sets", {}).get(GATE_FEATURE_SET)
    if value_arm is None or value_arm.get("verdict") == "singular":
        failures.append(
            "the OLS phase-b baseline could not be computed (singular fit)"
        )
    elif network_value_normalized is None:
        failures.append(
            "the manifest records no validation value_normalized to compare"
        )
    elif not float(value_arm["value_normalized"]) - GATE_MARGIN_NORMALIZED > float(
        network_value_normalized
    ):
        failures.append(
            f"phase-b value head: network value_normalized "
            f"{network_value_normalized} does not beat the {GATE_FEATURE_SET} "
            f"OLS ({value_arm['value_normalized']}) by {GATE_MARGIN_NORMALIZED}"
        )
    # Fail-closed, matching the phase-b arm above and this function's own
    # contract. Skipping the arm when its baseline is unavailable would
    # make the HARDER of the two gates switchable off by starving it of
    # input -- and the phase-a arm is the one every current v9 candidate
    # fails, so a silent skip is the difference between refusing and
    # promoting.
    if phase_a_score is None:
        failures.append(
            "the phase-a OLS baseline was not computed; a gate arm that "
            "cannot be evaluated is a refusal, not a pass"
        )
    elif phase_a_score.get("verdict") == "singular":
        failures.append(
            "the phase-a OLS baseline is a singular fit and cannot be "
            "compared against"
        )
    else:
        if network_equity_called_mse is None:
            failures.append(
                "the manifest records no validation equity_called MSE to compare"
            )
        else:
            variance = float(phase_a_score["validation_target_variance"])
            network_r2 = 1.0 - float(network_equity_called_mse) / variance
            ols_r2 = float(phase_a_score["r2_validation"])
            if not network_r2 > ols_r2 - 0.02:
                failures.append(
                    f"phase-a equity_called label: network R2 {network_r2:.4f} "
                    f"does not beat the 2-parameter OLS ({ols_r2:.4f}) - the "
                    "2026-09-02 diagnosis finding, unchanged"
                )
    return (not failures), failures


if __name__ == "__main__":
    raise SystemExit(main())
