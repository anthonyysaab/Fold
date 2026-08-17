"""Tier-2 belief plumbing: the opponent-model bucket interface, nothing more.

Buckets are the fitted opponent model's price-conditioned continuing-range
distribution over 8 strength-percentile octiles (``schema3.BELIEF_BUCKETS``),
uniform until a fitted provider exists. The fitted P3 model does not exist
yet; this module ships the interface plus a neutral default so P3 later
plugs in with zero schema change — feature producers depend on the
``BeliefProvider`` protocol, never on a concrete model.

Constraints:

- The bucket count and feature names live in ``schema3`` and are imported,
  never restated (schema-3 rule: one coordination point).
- Every provider output must survive ``require_buckets``; the neutral
  provider validates its own output on every call, so "the default is a
  distribution" is impossible-by-construction rather than assumed.
- No randomness anywhere in this module: the neutral prior is a constant,
  so identical inputs give identical outputs by construction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from devfun_poker_playground.schema3 import BELIEF_BUCKETS

# Sum-of-probabilities tolerance. Exact float accumulation error over
# BELIEF_BUCKETS additions is < 1e-14; 1e-6 is nine orders of magnitude
# looser (forgiving any reasonable provider arithmetic) while still
# catching every real distribution bug. Ablatable: tightening toward 1e-12
# only rejects providers doing sloppy renormalisation, which may be wanted.
_SUM_TOLERANCE = 1e-6


class BeliefProviderError(ValueError):
    """Raised when a claimed belief-bucket distribution violates the contract."""


@runtime_checkable
class BeliefProvider(Protocol):
    """Anything that can report the opponents' continuing-range distribution.

    ``runtime_checkable`` so wiring code can ``isinstance``-guard a plugged
    provider; note the runtime check only verifies the method exists, not
    its signature — ``require_buckets`` on the output is the real gate.
    """

    def continuing_range_buckets(
        self,
        table: Mapping[str, object],
        recent_actions: Sequence[Mapping[str, object]],
    ) -> tuple[float, ...]:
        """Probability mass per strength-percentile octile, lowest first."""
        ...


def require_buckets(buckets: Sequence[float]) -> tuple[float, ...]:
    """Validate a claimed bucket distribution; return it unchanged as a tuple.

    Checks length == ``schema3.BELIEF_BUCKETS``, every entry finite and
    >= 0, and the total within ``_SUM_TOLERANCE`` of 1.0. Raises
    ``BeliefProviderError`` otherwise. Entries are not coerced or
    renormalised — a provider that needs renormalising is broken.
    """

    result = tuple(buckets)
    if len(result) != BELIEF_BUCKETS:
        raise BeliefProviderError(
            f"belief distribution must have {BELIEF_BUCKETS} buckets, "
            f"found {len(result)}"
        )
    for index, value in enumerate(result):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise BeliefProviderError(
                f"bucket {index} is not numeric: {value!r}"
            ) from None
        if not math.isfinite(numeric):
            raise BeliefProviderError(f"bucket {index} is not finite: {value!r}")
        if numeric < 0.0:
            raise BeliefProviderError(f"bucket {index} is negative: {value!r}")
    total = math.fsum(float(value) for value in result)
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise BeliefProviderError(
            f"belief distribution sums to {total!r}, "
            f"not within {_SUM_TOLERANCE} of 1.0"
        )
    return result


# Derived, not authored: the maximum-entropy prior over BELIEF_BUCKETS
# octiles. 1/8 is a power of two, so the entries and their sum are exact
# in binary floating point.
_UNIFORM_PRIOR: tuple[float, ...] = tuple(
    1.0 / BELIEF_BUCKETS for _ in range(BELIEF_BUCKETS)
)


class NeutralBeliefProvider:
    """The no-model default: a uniform prior, unconditional on the inputs.

    Ignores ``table`` and ``recent_actions`` by definition — neutrality
    means conditioning on nothing. Validates its own output through
    ``require_buckets`` on every call so the default can never drift out
    of contract without a test failing.
    """

    def continuing_range_buckets(
        self,
        table: Mapping[str, object],
        recent_actions: Sequence[Mapping[str, object]],
    ) -> tuple[float, ...]:
        return require_buckets(_UNIFORM_PRIOR)


__all__ = [
    "BeliefProvider",
    "BeliefProviderError",
    "NeutralBeliefProvider",
    "require_buckets",
]
