"""Refuse to train on a retired Phase-A corpus.

This module owns ONE mechanism: deciding whether a Phase-A dataset on
disk is one the project still trains on, and refusing it loudly when it
is not.

Why it exists. On 2026-09-03 the project retired the Arena replay
archive and switched to PHH/Pluribus (``.handoff/DECISIONS.md`` section
6). The Arena-built dataset ``phase-a-dataset-v9.jsonl.gz`` stayed on
disk as the builder's hash oracle and a possible ablation arm -- and it
was the *default* for ``engine.v9_trainer --dataset`` and
``engine.v9_trainer_phase_b --phase-a-dataset``. Running either trainer
with no dataset flag therefore trained on the retired corpus, silently
and successfully: the file exists, it loads, and nothing complains. The
quarantine was a directory move and a documentation note; neither is a
gate. This is the gate.

How it decides. A dataset is live iff its sidecar declares a
``generator.source`` in :data:`LIVE_SOURCES`. The check is a positive
list and it fails CLOSED: a missing sidecar, an unreadable one, or one
with no ``source`` key is refused, because that is exactly the shape of
the Arena sidecar (built before the field existed, with ``roots``
pointing into the quarantined archive). Every retired corpus is
therefore refused by default rather than by enumeration -- nothing has
to remember to add it to a deny list.

Deliberate use of a retired corpus is still possible, but never by
accident: pass ``allow_retired=True`` (the trainers expose it as
``--allow-retired-dataset``). It is loud, and the caller is expected to
record the arm.

The sidecar is the record, not the archive itself: ``.jsonl.gz`` files
are gitignored bulk while their ``.summary.json`` siblings are tracked,
so provenance survives even where the corpus does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Sources this project still trains on. A positive list: everything
#: else is retired until someone deliberately adds it here.
LIVE_SOURCES = frozenset({"phh"})

#: The suffix a Phase-A dataset carries, and what its sidecar uses
#: instead. Mirrors ``PhaseARowSink`` in ``tools.build_phase_a_dataset_v9``.
_DATASET_SUFFIX = ".jsonl.gz"
_SIDECAR_SUFFIX = ".summary.json"


class RetiredDatasetError(RuntimeError):
    """A dataset was refused because it is not a live training corpus."""


def sidecar_path(dataset: str | Path) -> Path:
    """The ``.summary.json`` that records how ``dataset`` was built."""

    path = Path(dataset)
    return path.with_name(path.name.removesuffix(_DATASET_SUFFIX) + _SIDECAR_SUFFIX)


def read_provenance(dataset: str | Path) -> dict[str, Any] | None:
    """The sidecar's ``generator`` block, or None when unreadable.

    None is not "fine": :func:`require_live_dataset` treats it as a
    refusal. It is returned rather than raised so a caller that only
    wants to describe a dataset does not have to catch.
    """

    try:
        raw = sidecar_path(dataset).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        document = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(document, dict):
        return None
    generator = document.get("generator")
    return generator if isinstance(generator, dict) else None


def describe(dataset: str | Path) -> str:
    """One line naming the corpus and where it came from, for the log.

    A training run should say what it trained on in its own output, so
    the answer does not depend on someone opening the sidecar later.
    """

    generator = read_provenance(dataset)
    if generator is None:
        return f"{dataset} (no readable sidecar -- provenance unknown)"
    source = generator.get("source") or "unrecorded"
    roots = generator.get("roots") or []
    roots_text = ", ".join(str(root) for root in roots) or "unrecorded roots"
    commit = generator.get("dataset_commit")
    suffix = f", dataset commit {commit}" if commit else ""
    return f"{dataset} (source {source}; roots {roots_text}{suffix})"


def require_live_dataset(
    dataset: str | Path, *, allow_retired: bool = False
) -> str:
    """Return the dataset's source, or raise :class:`RetiredDatasetError`.

    ``allow_retired`` downgrades the refusal to a returned marker so a
    deliberate ablation can run; it never silences the caller's own
    logging, which is why the trainers print :func:`describe` either
    way.
    """

    generator = read_provenance(dataset)
    source = (generator or {}).get("source")
    if isinstance(source, str) and source in LIVE_SOURCES:
        return source

    if generator is None:
        why = f"it has no readable sidecar at {sidecar_path(dataset)}"
    elif source is None:
        roots = ", ".join(str(root) for root in generator.get("roots") or [])
        why = (
            "its sidecar records no generator.source, which is the shape of "
            "a corpus built before the PHH switch"
            + (f" (roots: {roots})" if roots else "")
        )
    else:
        why = f"its generator.source is {source!r}"

    if allow_retired:
        return f"retired:{source or 'unknown'}"

    raise RetiredDatasetError(
        f"refusing to train on {dataset}: {why}. Live sources are "
        f"{sorted(LIVE_SOURCES)}. Build a current Phase-A dataset with "
        "'python -m tools.build_phase_a_dataset_phh' (PROCEDURES section 16) "
        "and pass it explicitly, or pass --allow-retired-dataset to use this "
        "corpus deliberately as a named ablation arm."
    )


__all__ = [
    "LIVE_SOURCES",
    "RetiredDatasetError",
    "describe",
    "read_provenance",
    "require_live_dataset",
    "sidecar_path",
]
