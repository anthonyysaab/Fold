"""Independent policy adapter for dev.fun Poker Playground."""

from devfun_poker_playground.pure_model import PurePolicy
from devfun_poker_playground.rules import ArenaAction, DecisionRules
from devfun_poker_playground.snapshots import ArenaSnapshotError, features_from_table

try:
    from devfun_poker_playground.playground import PlaygroundPolicy, decide
except ModuleNotFoundError as _exc:  # pragma: no cover - deployment images only
    # Torch stays optional so torch-free deployment builds (e.g. the Chipzen
    # container) can import the package; any other missing module is a bug.
    if _exc.name is None or _exc.name.split(".")[0] != "torch":
        raise
    PlaygroundPolicy = None  # type: ignore[assignment]
    decide = None  # type: ignore[assignment]

__all__ = [
    "ArenaAction",
    "ArenaSnapshotError",
    "DecisionRules",
    "PlaygroundPolicy",
    "PurePolicy",
    "decide",
    "features_from_table",
]
