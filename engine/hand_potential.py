"""The hand-potential decomposition — Ppot and Npot, heads-up.

``strength_metric.strength_percentile`` scores the hero's *current* made hand
and deliberately ignores future cards. This module supplies the complementary
pair in the spirit of the Loki/Poki effective-hand-strength literature
(Billings et al.):

- ``ppot`` — hero's expected showdown score (win 1, tie 0.5) over the
  sampled matchups in which hero is behind or tied on the current board
- ``npot`` — hero's expected showdown loss share (loss 1, tie 0.5) over
  the sampled matchups in which hero is ahead or tied on the current board

against **one opponent holding sampled uniformly from the unseen cards**.
Showdown ties credit 0.5 to the favorable outcome, and a tie on the current
board conditions *both* components; this is the accounting the code
implements — close to, but not a claim of, the papers' exact transition-
matrix bookkeeping. The single-uniform-opponent model keeps the two numbers
street-comparable and player-count invariant; conditioning on a modelled
range is a separate, later concern.

Constraints, matching every live-inference module:

- Stdlib-only plus the vendored ``treys`` (``Evaluator.evaluate(board, hole)``;
  **lower rank is stronger**).
- Deterministic: all sampling comes from ``random.Random(seed)`` constructed
  inside the call. No global RNG, no time dependence.
- Preflop (empty board) and river (five board cards) return ``(0.0, 0.0)`` by
  definition — there are no future cards, so potential is identically zero.
- An empty conditioning event returns ``0.0`` for that component (e.g. a hero
  who is ahead of every sampled holding has no "currently behind-or-tied"
  sample, so ``ppot`` is 0.0 — nothing to improve from).
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Sequence

from engine._vendor.treys import Card, Evaluator
from engine.schema3 import CARD_CODES

# The card universe is the schema's, never restated (F8): the deck and all
# membership validation derive from schema3.CARD_CODES.
_CARD_CODE_SET = frozenset(CARD_CODES)

# Game rules, not tunables: a holding is two cards, a full board is five.
_HOLE_SIZE = 2
_FULL_BOARD = 5

# Default sample size. Ablatable: at the worst case p = 0.5 the binomial
# standard error is sqrt(0.25 / 1000) ~= 0.016, below any threshold the
# consumers of these numbers distinguish; raise it if a consumer ever does.
_DEFAULT_TRIALS = 1000

# Default seed. Arbitrary but fixed so bare calls are reproducible; ablatable
# (any int is equally valid — callers doing statistics must vary it themselves).
_DEFAULT_SEED = 7


class HandPotentialError(ValueError):
    """Raised when cards passed to the potential estimator are malformed."""


_EVALUATOR: Evaluator | None = None


def _evaluator() -> Evaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = Evaluator()
    return _EVALUATOR


@lru_cache(maxsize=1)
def _full_deck() -> tuple[int, ...]:
    return tuple(Card.new(code) for code in CARD_CODES)


def _parse(cards: Sequence[str], name: str) -> list[int]:
    parsed = []
    for card in cards:
        if not isinstance(card, str) or card not in _CARD_CODE_SET:
            raise HandPotentialError(f"{name} card {card!r} is not a card code")
        parsed.append(Card.new(card))
    if len(set(parsed)) != len(parsed):
        raise HandPotentialError(f"{name} contains duplicate cards")
    return parsed


def hand_potential(
    hole: Sequence[str],
    board: Sequence[str],
    *,
    trials: int = _DEFAULT_TRIALS,
    seed: int = _DEFAULT_SEED,
) -> tuple[float, float]:
    """Estimate ``(ppot, npot)`` for ``hole`` on ``board``.

    Monte-Carlo over ``trials`` samples from ``random.Random(seed)``: each
    trial draws an opponent holding uniformly from the unseen cards plus the
    remaining board cards, classifies the *current* board matchup, and scores
    the showdown on the completed board.

    Status on the current board counts a tie as ahead-or-tied, so a tied
    sample conditions **both** components. At showdown a tie counts half
    toward the favorable outcome (half a win for ``ppot``, half a loss for
    ``npot``), matching the Loki/Poki accounting.

    Returns ``(0.0, 0.0)`` preflop and on the river (no future cards), and
    ``0.0`` for a component whose conditioning event never occurs in the
    sample.
    """

    hole_cards = _parse(hole, "hole")
    if len(hole_cards) != _HOLE_SIZE:
        raise HandPotentialError("a holding is exactly two cards")
    board_cards = _parse(board, "board")
    if len(board_cards) not in (0, 3, 4, _FULL_BOARD):
        raise HandPotentialError("a board has 0, 3, 4 or 5 cards")
    if set(hole_cards) & set(board_cards):
        raise HandPotentialError("hole and board share a card")
    if trials < 1:
        raise HandPotentialError("trials must be a positive integer")

    if len(board_cards) in (0, _FULL_BOARD):
        return (0.0, 0.0)

    rng = random.Random(seed)
    evaluator = _evaluator()
    hero_now = evaluator.evaluate(board_cards, hole_cards)
    dead = set(hole_cards) | set(board_cards)
    unseen = [card for card in _full_deck() if card not in dead]
    missing = _FULL_BOARD - len(board_cards)

    ppot_favorable = 0.0
    ppot_conditioned = 0
    npot_favorable = 0.0
    npot_conditioned = 0
    for _ in range(trials):
        drawn = rng.sample(unseen, _HOLE_SIZE + missing)
        opponent = drawn[:_HOLE_SIZE]
        runout = board_cards + drawn[_HOLE_SIZE:]
        opponent_now = evaluator.evaluate(board_cards, opponent)
        hero_end = evaluator.evaluate(runout, hole_cards)
        opponent_end = evaluator.evaluate(runout, opponent)
        # treys: lower rank is stronger. hero_share is the hero's showdown
        # score — win 1, tie 0.5, loss 0.
        if hero_end < opponent_end:
            hero_share = 1.0
        elif hero_end == opponent_end:
            hero_share = 0.5
        else:
            hero_share = 0.0
        if hero_now >= opponent_now:  # behind-or-tied now
            ppot_conditioned += 1
            ppot_favorable += hero_share
        if hero_now <= opponent_now:  # ahead-or-tied now
            npot_conditioned += 1
            npot_favorable += 1.0 - hero_share
    ppot = ppot_favorable / ppot_conditioned if ppot_conditioned else 0.0
    npot = npot_favorable / npot_conditioned if npot_conditioned else 0.0
    return (ppot, npot)


__all__ = [
    "HandPotentialError",
    "hand_potential",
]
