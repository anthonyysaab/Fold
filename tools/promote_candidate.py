"""Promote a candidate artifact to the approved live version, reversibly.

Promotion re-validates the candidate manifest, re-verifies the weights
checksum, writes an ``approved`` manifest beside the candidate (state
transition plus promotion metadata, as the learning contract requires),
and atomically replaces ``artifacts/approved.json`` -- the single pointer
the runner follows. The previous pointer is preserved inside the new one,
and ``--rollback`` restores it. No Arena requests are made; deploying to
live play is still the owner starting the runner.

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
from datetime import UTC, datetime
from pathlib import Path

from engine.learning_contract import (
    LearningContractError,
    validate_artifact_manifest,
)

APPROVED_POINTER = "approved.json"


def _write_atomic(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_suffix(".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def promote(
    manifest_path: Path, artifacts_dir: Path, reason: str, evaluation_note: str | None
) -> Path:
    manifest = _load_json(manifest_path)
    validate_artifact_manifest(manifest)
    if manifest.get("state") != "candidate":
        raise SystemExit("only candidate manifests can be promoted")
    weights_file = manifest_path.parent / str(manifest["weights_file"])
    digest = hashlib.sha256(weights_file.read_bytes().rstrip(b"\n")).hexdigest()
    if digest != manifest["weights_sha256"]:
        raise SystemExit("weights checksum mismatch; refusing to promote")

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
    parser.add_argument("--evaluation-note", help="optional evaluation summary")
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
    )
    print(f"approved pointer updated: {pointer}")
    print(json.dumps(_load_json(pointer), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
