"""The v9 branch contract: four live branches, no fixed sizes.

Owner decision 2026-08-28, and this module is the normative record of it.
The v7/v8 branch tuples ("fold", "check_call", "aggress_half_pot" /
"aggress_small", "aggress_pot" / "aggress_large") are retired going
forward. Their measured defects:

- Two of four slots were aggression, so a constant (dead) ``action_value``
  head argmaxed to an aggression branch by construction; both hero bets in
  the hand that ended the 2026-08-26 deployment were exactly that constant
  (``tools/head_degeneracy_audit.py``).
- Check and call, whose values have entirely different structure (a check
  risks nothing; a call pays ``to_call`` for a showdown share), shared one
  ``check_call`` slot.
- The fixed pot fractions (0.5 / 1.0) overrode the engine's continuous
  temperature-sized wager whenever the learned head chose the branch --
  the discretization *deleted* sizing information at serve.

The v9 contract::

    fatal      -> fold    legal only when there is a price to escape
    passive    -> check   legal only when checking is free
    active     -> call    when facing a wager
               -> bet     when unprovoked, sized continuously by the
                          strength/temperature read -- never a fixed
                          pot fraction
    aggressive -> raise   escalation beyond the current price, up to
                          shove; legal only when facing a wager

By construction, per state:

- ``to_call == 0`` emits ``(passive, active)``; ``to_call > 0`` emits
  ``(fatal, active, aggressive)`` (aggressive drops out when no raise is
  legal). No two emitted branches ever execute the same action, so there
  is nothing for a dedup order to collapse and every emitted slot
  supervises.
- Folding a free check is dominated and MASKED here, not mopped up
  downstream (the v7 path executed it as a check and relabeled it in the
  simulator's dedup). Symmetrically -- owner: "when nobody has bet, no
  need to be aggressive; the active measure sizes the pot by hand
  strength" -- unprovoked wagers belong to ``active`` and ``aggressive``
  is escalation-only.
- Exactly one wager-matching/making branch and one escalation branch: a
  dead head can no longer land on aggression by a 2-in-4 coin flip.

Sizing is a state function, never a slot constant: this contract exposes
NO pot fractions. The engine's continuous sizer (``_sized_action``'s
temperature path) is the only sizing authority; wiring it as such is the
engine-layer edit. The Arena's ``"all-in"`` action is a *realization* of
``active``-bet or ``aggressive`` at stack-reaching size and is owned by
the engine's rendering, not by this contract.

This module is additive. v7 (format 2) and v8 (format 3) artifacts keep
loading through their own contracts; nothing imports this module until
its consumer layer is rebuilt. Restructure proceeds layer by layer with
direct edits, by owner decision -- no measurement gate on this path.
"""

from __future__ import annotations

from collections.abc import Collection

from engine.schema3 import BELIEF_BUCKETS

MODEL_FORMAT_VERSION_V9 = 4
MODEL_FAMILY_V9 = "v9-composed-value"

BRANCH_LABELS_V9 = ("fatal", "passive", "active", "aggressive")

# Wager-making branches carry fold equity. ``active`` carries it only in
# its bet state (``to_call == 0``): a call closes hero's action and buys
# no folds, so consumers must not consult the active fold-through slot
# when the branch executes as a call.
FOLD_THROUGH_BRANCHES_V9 = ("active", "aggressive")

# Showdown-share slots: hero's equity share if the branch's wager is
# matched (or the street checks through, for ``passive``). ``fatal`` is
# worth exactly 0 by definition and has no head slot.
EQUITY_SLOTS_V9 = ("passive", "active", "aggressive")

# Head widths follow the composed-value roles; the composition arithmetic
# itself is the value-layer edit, not this contract's.
V9_HEAD_SIZES: dict[str, int] = {
    "fold_through": len(FOLD_THROUGH_BRANCHES_V9),
    "range": BELIEF_BUCKETS,
    "equity_called": len(EQUITY_SLOTS_V9),
    "residual": len(BRANCH_LABELS_V9),
}


class BranchContractError(ValueError):
    """Raised when a state and a branch cannot be reconciled."""


def branch_index(branch: str) -> int:
    try:
        return BRANCH_LABELS_V9.index(branch)
    except ValueError:
        raise BranchContractError(f"unknown v9 branch: {branch!r}") from None


def branch_action(branch: str, to_call: int) -> str:
    """The canonical action a branch executes at this price.

    Returns one of ``fold`` / ``check`` / ``call`` / ``bet`` / ``raise``.
    Raises :class:`BranchContractError` for a branch this state masks --
    callers are expected to have consulted :func:`legal_branches` first,
    and executing a masked branch is a caller bug, never a fallback.
    """

    if to_call < 0:
        raise BranchContractError("to_call cannot be negative")
    if branch == "fatal":
        if to_call == 0:
            raise BranchContractError("fatal is dominated when checking is free")
        return "fold"
    if branch == "passive":
        if to_call > 0:
            raise BranchContractError("passive cannot check a positive price")
        return "check"
    if branch == "active":
        return "call" if to_call > 0 else "bet"
    if branch == "aggressive":
        if to_call == 0:
            raise BranchContractError(
                "aggressive is escalation-only: nobody has bet"
            )
        return "raise"
    raise BranchContractError(f"unknown v9 branch: {branch!r}")


def legal_branches(available: Collection[str], to_call: int) -> tuple[int, ...]:
    """Indices of the branches this state can execute, in slot order.

    ``available`` is the Arena's ``availableActions`` set, lowered
    (``fold`` / ``check`` / ``call`` / ``bet`` / ``raise`` / ``all-in``).
    ``all-in`` counts as an escalation when facing a wager and as the top
    of the ``active`` bet range when unprovoked (a short hero whose only
    legal wager is the whole stack).

    Fails closed: a state that admits no branch raises rather than
    guessing, matching ``game_state``'s "no supported legal action"
    discipline.
    """

    if to_call < 0:
        raise BranchContractError("to_call cannot be negative")
    actions = {str(action) for action in available}
    if not actions:
        raise BranchContractError("no actions are available")
    emitted: list[int] = []
    if "fold" in actions and to_call > 0:
        emitted.append(branch_index("fatal"))
    if "check" in actions and to_call == 0:
        emitted.append(branch_index("passive"))
    if ("call" in actions and to_call > 0) or (
        to_call == 0 and ("bet" in actions or "all-in" in actions)
    ):
        emitted.append(branch_index("active"))
    if to_call > 0 and ("raise" in actions or "all-in" in actions):
        emitted.append(branch_index("aggressive"))
    if not emitted:
        raise BranchContractError(
            "no v9 branch is legal for this state; "
            f"available={sorted(actions)!r}, to_call={to_call}"
        )
    return tuple(emitted)


__all__ = [
    "BRANCH_LABELS_V9",
    "BranchContractError",
    "EQUITY_SLOTS_V9",
    "FOLD_THROUGH_BRANCHES_V9",
    "MODEL_FAMILY_V9",
    "MODEL_FORMAT_VERSION_V9",
    "V9_HEAD_SIZES",
    "branch_action",
    "branch_index",
    "legal_branches",
]
