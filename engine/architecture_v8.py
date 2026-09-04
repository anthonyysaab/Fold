"""The composed-value architecture contract, and its two fail-closed validators.

Sibling to :mod:`engine.branch_contract_v9`: that module is the normative
record of WHICH branches exist, this one of WHAT SHAPE the network that scores
them has. Both are contracts the serve path reads and the trainers must satisfy
-- neither is training code.

These names lived in `v8_trainer.py` until 2026-09-04. That was the single
reason `engine/` could not be separated from the training package:
`learned_policy_v8.py` needs six of them and `learned_policy_v9.py` needs
`validate_v9_architecture`, so every serving artifact formally depended on a
trainer module to describe its own shape. Worse, `learning_contract` had to
import the validator LAZILY, inside a function, to dodge a genuine import cycle
(`v8_trainer` imports `MODEL_FORMAT` back from `learning_contract`). That cycle
does not exist any more, and the lazy workaround is gone with it.

The lazy import also concealed a live defect: `deploy/chipzen/engine/` ships
`learning_contract.py` but NOT `v8_trainer.py`, so the deferred import would
have raised ``ModuleNotFoundError`` inside the Chipzen image the first time
that path ran. Moving the validator here puts it in a module the bundle
actually carries.

Two validators, never one widened to accept both shapes: a permissive
validator is how a v8 artifact serves under v9 slot meanings without anyone
noticing.
"""

from __future__ import annotations

from collections.abc import Mapping

from engine import schema3

MODEL_FORMAT_VERSION_V8 = 3
MODEL_FAMILY_V8 = "v8-composed-value"

CARD_ENCODER_WIDTH = 64
CONTEXT_ENCODER_WIDTH = 48
TRUNK_WIDTHS = (128, 128)
HEAD_TOWER_WIDTH = 32
V8_HEAD_SIZES: dict[str, int] = {
    "fold_through": 2,
    "range": schema3.BELIEF_BUCKETS,
    "equity_called": 3,
    "residual": 4,
}
BRANCH_LABELS_V8 = ("fold", "check_call", "aggress_small", "aggress_large")
FOLD_THROUGH_BRANCHES = ("aggress_small", "aggress_large")
EQUITY_SLOTS = ("aggress_small", "aggress_large", "check_call")


def validate_v9_architecture(architecture: Mapping[str, object]) -> None:
    """Fail-closed validation of a format-4 architecture block.

    A separate validator, never a widening of the v8 one: a validator
    that accepts both shapes is how a v8 artifact serves under v9 slot
    meanings without anyone noticing.
    """

    from engine import schema4
    from engine.branch_contract_v9 import (
        EQUITY_SLOTS_V9,
        FOLD_THROUGH_BRANCHES_V9,
        MODEL_FAMILY_V9,
        V9_HEAD_SIZES,
    )

    if architecture.get("family") != MODEL_FAMILY_V9:
        raise ValueError(f"architecture family must be {MODEL_FAMILY_V9!r}")
    if list(architecture.get("card_indices") or []) != list(
        schema4.CARD_INDICES_V9
    ):
        raise ValueError("card_indices must match schema4.CARD_INDICES_V9")
    if list(architecture.get("context_indices") or []) != list(
        schema4.CONTEXT_INDICES_V9
    ):
        raise ValueError("context_indices must match schema4.CONTEXT_INDICES_V9")
    if architecture.get("card_encoder_width") != CARD_ENCODER_WIDTH:
        raise ValueError(f"card_encoder_width must be {CARD_ENCODER_WIDTH}")
    if architecture.get("context_encoder_width") != CONTEXT_ENCODER_WIDTH:
        raise ValueError(f"context_encoder_width must be {CONTEXT_ENCODER_WIDTH}")
    if list(architecture.get("trunk_widths") or []) != list(TRUNK_WIDTHS):
        raise ValueError(f"trunk_widths must be {list(TRUNK_WIDTHS)}")
    if dict(architecture.get("heads") or {}) != V9_HEAD_SIZES:
        raise ValueError(f"heads must be {V9_HEAD_SIZES}")
    towers = dict(architecture.get("head_towers") or {})
    if towers != {name: HEAD_TOWER_WIDTH for name in V9_HEAD_SIZES}:
        raise ValueError(f"head_towers must all be {HEAD_TOWER_WIDTH}")
    if list(architecture.get("fold_through_branches") or []) != list(
        FOLD_THROUGH_BRANCHES_V9
    ):
        raise ValueError(
            f"fold_through_branches must be {list(FOLD_THROUGH_BRANCHES_V9)}"
        )
    if list(architecture.get("equity_slots") or []) != list(EQUITY_SLOTS_V9):
        raise ValueError(f"equity_slots must be {list(EQUITY_SLOTS_V9)}")
    dropout = architecture.get("dropout")
    if dropout is not None and not (
        isinstance(dropout, float) and 0.0 <= dropout < 1.0
    ):
        raise ValueError("dropout must be None or a float in [0, 1)")


def validate_v8_architecture(architecture: Mapping[str, object]) -> None:
    if architecture.get("family") != MODEL_FAMILY_V8:
        raise ValueError(f"architecture family must be {MODEL_FAMILY_V8!r}")
    if list(architecture.get("card_indices") or []) != list(schema3.CARD_INDICES):
        raise ValueError("card_indices must match schema3.CARD_INDICES")
    if list(architecture.get("context_indices") or []) != list(
        schema3.CONTEXT_INDICES
    ):
        raise ValueError("context_indices must match schema3.CONTEXT_INDICES")
    if architecture.get("card_encoder_width") != CARD_ENCODER_WIDTH:
        raise ValueError(f"card_encoder_width must be {CARD_ENCODER_WIDTH}")
    if architecture.get("context_encoder_width") != CONTEXT_ENCODER_WIDTH:
        raise ValueError(f"context_encoder_width must be {CONTEXT_ENCODER_WIDTH}")
    if list(architecture.get("trunk_widths") or []) != list(TRUNK_WIDTHS):
        raise ValueError(f"trunk_widths must be {list(TRUNK_WIDTHS)}")
    if dict(architecture.get("heads") or {}) != V8_HEAD_SIZES:
        raise ValueError(f"heads must be {V8_HEAD_SIZES}")
    towers = dict(architecture.get("head_towers") or {})
    if towers != {name: HEAD_TOWER_WIDTH for name in V8_HEAD_SIZES}:
        raise ValueError(f"head_towers must all be {HEAD_TOWER_WIDTH}")
    dropout = architecture.get("dropout")
    if dropout is not None and not (
        isinstance(dropout, float) and 0.0 <= dropout < 1.0
    ):
        raise ValueError("dropout must be None or a float in [0, 1)")


__all__ = [
    "BRANCH_LABELS_V8",
    "CARD_ENCODER_WIDTH",
    "CONTEXT_ENCODER_WIDTH",
    "EQUITY_SLOTS",
    "FOLD_THROUGH_BRANCHES",
    "HEAD_TOWER_WIDTH",
    "MODEL_FAMILY_V8",
    "MODEL_FORMAT_VERSION_V8",
    "TRUNK_WIDTHS",
    "V8_HEAD_SIZES",
    "validate_v8_architecture",
    "validate_v9_architecture",
]
