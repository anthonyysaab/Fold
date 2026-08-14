"""Foreign teacher decisions as validated training examples.

The public-data collector writes one CSV row per top-agent decision with the
exact 138 learning features, the chosen action family, the effective-stack
risk fraction, and the settled hand reward. This module is the trainer-facing
boundary: it converts eligible rows into the same :class:`TrainingExample`
contract the private telemetry loader yields, without pretending the actions
were locally accepted submissions -- every example's ``policy_version`` is
``foreign-teacher-s<season>-<agent_id>``, so provenance survives into
candidate manifests via ``training_window.source_policy_versions``.

Teacher eligibility (non-timeout, legal action, at least two legal families)
comes from the collector; this loader re-checks the fields it consumes and
refuses malformed rows instead of guessing.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from devfun_poker_playground.learning_contract import LEARNING_FEATURE_NAMES
from devfun_poker_playground.policy_features import LABELS
from devfun_poker_playground.training_telemetry import TrainingExample

FOREIGN_POLICY_PREFIX = "foreign-teacher"
# Collector CSVs carry the schema-1 feature set; the four schema-2 opponent
# and standing inputs cannot exist in single-hand public replays, so the
# loader appends them: the recorded conditioning fraction stands in for the
# range width, and the session-evidence inputs get their honest no-evidence
# neutrals (wildness/stickiness 0.0, lead 0.5).
_SCHEMA2_APPENDED = (
    "opponent_range_width",
    "opponent_max_wildness",
    "opponent_max_stickiness",
    "lead_position_unit",
)
FEATURE_COLUMNS = tuple(
    f"feature_{name}"
    for name in LEARNING_FEATURE_NAMES
    if name not in _SCHEMA2_APPENDED
)
_REQUIRED_COLUMNS = (
    "season_number",
    "agent_id",
    "table_id",
    "action_family",
    "action_effective_stack_fraction",
    "hand_reward_bb",
    "hero_stack_before_chips",
    "hero_total_committed_before_chips",
    "big_blind_chips",
    "teacher_eligible",
    "selected_action_is_legal",
    "equity_top_fraction",
    *FEATURE_COLUMNS,
)


class ForeignDataError(ValueError):
    """Raised when the foreign CSV violates the training contract."""


def _flag(row: dict, name: str, line: int) -> bool:
    value = str(row.get(name) or "").strip().casefold()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise ForeignDataError(f"row {line}: {name} must be true or false")


def _finite(row: dict, name: str, line: int) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ForeignDataError(f"row {line}: {name} must be a number") from exc
    if not math.isfinite(value):
        raise ForeignDataError(f"row {line}: {name} must be finite")
    return value


def _identifier(row: dict, name: str, line: int) -> str:
    value = str(row.get(name) or "").strip()
    if not value:
        raise ForeignDataError(f"row {line}: {name} must be a non-empty string")
    return value


def load_foreign_training_examples(
    path: str | Path,
    *,
    eligible_only: bool = True,
) -> tuple[TrainingExample, ...]:
    """Read collector CSV rows into the shared training-example contract."""

    csv_path = Path(path).expanduser()
    examples: list[TrainingExample] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = [name for name in _REQUIRED_COLUMNS if name not in fields]
        if missing:
            raise ForeignDataError(
                f"{csv_path.name} is missing columns: {', '.join(missing[:5])}"
                + ("..." if len(missing) > 5 else "")
            )
        for line, row in enumerate(reader, start=2):
            if eligible_only and not _flag(row, "teacher_eligible", line):
                continue
            if not _flag(row, "selected_action_is_legal", line):
                raise ForeignDataError(
                    f"row {line}: eligible rows must record a legal action"
                )
            family = str(row.get("action_family") or "").strip()
            if family not in LABELS:
                raise ForeignDataError(
                    f"row {line}: action_family must be one of {', '.join(LABELS)}"
                )
            season = int(_finite(row, "season_number", line))
            risk_fraction = _finite(row, "action_effective_stack_fraction", line)
            if risk_fraction < 0.0:
                raise ForeignDataError(
                    f"row {line}: action_effective_stack_fraction cannot be negative"
                )
            # Commitments beyond the effective stack (all-ins nobody can
            # fully call) clamp to 1.0, mirroring the telemetry contract's
            # min(1.0, new_chips / effective_stack).
            risk_fraction = min(1.0, risk_fraction)
            range_width = _finite(row, "equity_top_fraction", line)
            if not 0.0 <= range_width <= 1.0:
                raise ForeignDataError(f"row {line}: equity_top_fraction must be 0..1")
            features = (
                *(_finite(row, column, line) for column in FEATURE_COLUMNS),
                range_width,  # opponent_range_width
                0.0,  # opponent_max_wildness: no session evidence in replays
                0.0,  # opponent_max_stickiness: no session evidence
                0.5,  # lead_position_unit: neutral standing
            )
            purse_chips = _finite(row, "hero_stack_before_chips", line) + _finite(
                row, "hero_total_committed_before_chips", line
            )
            big_blind = _finite(row, "big_blind_chips", line)
            if purse_chips <= 0.0 or big_blind <= 0.0:
                raise ForeignDataError(
                    f"row {line}: purse and big_blind_chips must be positive"
                )
            examples.append(
                TrainingExample(
                    table_id=_identifier(row, "table_id", line),
                    policy_version=(
                        f"{FOREIGN_POLICY_PREFIX}-s{season}-"
                        f"{_identifier(row, 'agent_id', line)}"
                    ),
                    features=features,
                    action_family_index=LABELS.index(family),
                    behavior_probabilities=tuple(
                        1.0 if label == family else 0.0 for label in LABELS
                    ),
                    submitted_risk_fraction=risk_fraction,
                    purse_bb=purse_chips / big_blind,
                    reward_bb=_finite(row, "hand_reward_bb", line),
                    counterfactual=False,
                    opponent_confidence=0.0,
                )
            )
    return tuple(examples)


__all__ = [
    "FEATURE_COLUMNS",
    "FOREIGN_POLICY_PREFIX",
    "ForeignDataError",
    "load_foreign_training_examples",
]
