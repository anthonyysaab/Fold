"""Promote a candidate artifact to the approved live version, reversibly.

Promotion re-validates the candidate manifest, re-verifies the weights
checksum, writes an ``approved`` manifest beside the candidate (state
transition plus promotion metadata, as the learning contract requires),
and atomically replaces ``artifacts/approved.json`` -- the single pointer
the runner follows. The previous pointer is preserved inside the new one,
and ``--rollback`` restores it. No Arena requests are made; deploying to
live play is still the owner starting the runner.

Format-4 (v9) candidates must also pass the OLS baseline gate
(``tools/ols_baseline.py``, the 2026-09-02 diagnosis made permanent): a
network that loses to a k-parameter linear model on its own held-out
split is not promotable, on either the Phase-B composed-value target or
the Phase-A ``equity_called`` label. The gate is enforced by default;
``--ols-gate skip`` is an explicit owner override for a documented
reason, and ``warn`` prints the verdict without refusing.

Examples (module mode puts the repository root on the import path):
    python -m tools.promote_candidate artifacts/candidates/<v>.manifest.json \
        --reason "beat heuristic v5 by +9 bb/100 over 6k simulated hands"
    python -m tools.promote_candidate --rollback
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.branch_contract_v9 import MODEL_FORMAT_VERSION_V9
from engine.learning_contract import (
    LearningContractError,
    validate_artifact_manifest,
)

APPROVED_POINTER = "approved.json"


def _ols_gate(manifest: Mapping[str, Any], mode: str) -> None:
    """Run the OLS baseline gate for a format-4 candidate.

    The comparison numbers are computed FRESH from the artifact's own
    recorded training inputs (the corpus and dataset paths plus the
    split seed in the manifest), so a candidate can neither inherit nor
    hand-write a passing score. A missing input fails the gate, never
    passes it.
    """

    from tools.ols_baseline import (
        _phase_a_ols,
        ols_gate_verdict,
        score_phase_b_corpus,
    )

    training = manifest.get("training")
    window = manifest.get("training_window") or {}
    if not isinstance(training, Mapping) or not isinstance(window, Mapping):
        raise SystemExit(
            "OLS gate: format-4 manifest lacks training/training_window "
            "metadata; refusing to promote"
        )
    split = training.get("split") or {}
    split_seed = split.get("split_seed")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise SystemExit("OLS gate: manifest records no split seed")
    corpus = window.get("phase_b_corpus")
    dataset = window.get("phase_a_dataset")
    if not isinstance(corpus, str) or not isinstance(dataset, str):
        raise SystemExit(
            "OLS gate: manifest records no phase_b_corpus/phase_a_dataset; "
            "refusing to promote"
        )
    losses = (manifest.get("evaluation") or {}).get("validation_losses") or {}
    phase_b_score = score_phase_b_corpus(
        Path(corpus), split_seed=int(split_seed), validation_fraction=0.1
    )
    phase_a_score = _phase_a_ols(
        Path(dataset), split_seed=int(split_seed), validation_fraction=0.1
    )
    value_normalized = losses.get("value_normalized")
    equity_called_mse = losses.get("equity_called")
    passes, failures = ols_gate_verdict(
        phase_b_score,
        phase_a_score,
        network_value_normalized=value_normalized,
        network_equity_called_mse=equity_called_mse,
    )
    for failure in failures:
        print(f"OLS gate: {failure}")
    if passes:
        print(
            "OLS gate: PASS — the network beats the k-parameter OLS on "
            "both held-out targets"
        )
        return
    print(f"OLS gate: FAIL ({mode})")
    if mode == "enforce":
        raise SystemExit(
            "refusing to promote: the candidate does not beat a k-parameter "
            "OLS on held-out data. Override only with an explicit, "
            "documented owner decision: --ols-gate skip"
        )


def _write_atomic(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_suffix(".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def promote(
    manifest_path: Path,
    artifacts_dir: Path,
    reason: str,
    evaluation_note: str | None,
    ols_gate: str = "enforce",
) -> Path:
    manifest = _load_json(manifest_path)
    validate_artifact_manifest(manifest)
    if manifest.get("state") != "candidate":
        raise SystemExit("only candidate manifests can be promoted")
    weights_file = manifest_path.parent / str(manifest["weights_file"])
    digest = hashlib.sha256(weights_file.read_bytes().rstrip(b"\n")).hexdigest()
    if digest != manifest["weights_sha256"]:
        raise SystemExit("weights checksum mismatch; refusing to promote")
    if manifest.get("format_version") == MODEL_FORMAT_VERSION_V9:
        if ols_gate not in {"enforce", "warn", "skip"}:
            raise SystemExit("--ols-gate must be enforce, warn, or skip")
        if ols_gate != "skip":
            _ols_gate(manifest, ols_gate)

    approved = dict(manifest)
    approved["state"] = "approved"
    approved["promotion"] = {
        "approved_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reason": reason,
    }
    if evaluation_note:
        approved["promotion"]["evaluation_note"] = evaluation_note
    try:
        validate_artifact_manifest(approved)
    except LearningContractError as error:
        raise SystemExit(f"approved manifest failed validation: {error}") from error

    approved_name = manifest_path.name.replace(
        ".manifest.json", ".approved.manifest.json"
    )
    approved_path = manifest_path.parent / approved_name
    _write_atomic(approved_path, approved)

    pointer_path = artifacts_dir / APPROVED_POINTER
    previous = None
    if pointer_path.exists():
        previous = _load_json(pointer_path)
        previous.pop("previous", None)  # keep one generation of history
    _write_atomic(
        pointer_path,
        {
            "model_version": approved["model_version"],
            # POSIX separators always: the pointer is promoted on Windows but
            # read by the Linux host that serves live play, where a backslash
            # is an ordinary filename character rather than a separator.
            "manifest_file": approved_path.relative_to(artifacts_dir).as_posix(),
            "weights_sha256": approved["weights_sha256"],
            "approved_at": approved["promotion"]["approved_at"],
            "previous": previous,
        },
    )
    return pointer_path


def rollback(artifacts_dir: Path) -> Path:
    pointer_path = artifacts_dir / APPROVED_POINTER
    if not pointer_path.exists():
        raise SystemExit("no approved pointer to roll back")
    pointer = _load_json(pointer_path)
    previous = pointer.get("previous")
    if not previous:
        raise SystemExit("approved pointer has no previous version recorded")
    previous["previous"] = None
    # The stored generation was copied verbatim and may predate POSIX
    # separators, so a rollback could otherwise resurrect a pointer the Linux
    # host cannot resolve.
    stored = previous.get("manifest_file")
    if isinstance(stored, str):
        previous["manifest_file"] = stored.replace("\\", "/")
    _write_atomic(pointer_path, previous)
    return pointer_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", help="candidate manifest to promote")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--reason", help="required promotion reason")
    parser.add_argument(
        "--evaluation-note", help="optional evaluation summary"
    )
    parser.add_argument(
        "--ols-gate",
        choices=("enforce", "warn", "skip"),
        default="enforce",
        help=(
            "the k-parameter OLS baseline gate for format-4 candidates "
            "(default: enforce; the 2026-09-02 diagnosis finding, made "
            "permanent)"
        ),
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="restore the previously approved version",
    )
    args = parser.parse_args(argv)
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()

    if args.rollback:
        if args.manifest:
            parser.error("--rollback takes no manifest")
        pointer = rollback(artifacts_dir)
        print(f"rolled back: {pointer}")
        print(json.dumps(_load_json(pointer), indent=2, sort_keys=True))
        return 0

    if not args.manifest or not args.reason:
        parser.error("promotion needs a manifest and --reason")
    pointer = promote(
        Path(args.manifest).expanduser().resolve(),
        artifacts_dir,
        args.reason,
        args.evaluation_note,
        ols_gate=args.ols_gate,
    )
    print(f"approved pointer updated: {pointer}")
    print(json.dumps(_load_json(pointer), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
