"""PHH hands to the Arena-shaped replay dicts the v9 Phase-A builder reads.

One mechanism, one module: this is the ONLY place a PHH hand history
(``phh-dataset`` v3, parsed by pokerkit 0.7.4) becomes the
``{'table': ..., 'events': ...}`` shape that
``tools.build_phase_a_dataset_v9.replay_rows_v9`` consumes unchanged —
the same shape ``tools.collect_foreign_play_data`` reconstructed from
the quarantined Arena archive. The PHH sink
(``tools.build_phase_a_dataset_phh``) feeds every ``*.phh`` / ``*.phhs``
file through here and through ``replay_rows_v9`` with no Arena-shaped
shim in between, so the dataset contract, the labels, and the trainer
loader see exactly what they saw on Arena rows.

Ground truth is pokerkit's state machine, replayed one action at a
time. Before every player action the pre-action state is read off
(stacks, street bets, pot, the price to call, and the legal completion
range), the action is mapped onto the Arena vocabulary, and an
``ActionTaken`` event is emitted with the pre-action facts in
``payload`` (``stackBefore``, ``pot``, ``currentBetBefore``,
``allowedActions``, ...) and a post-action Arena-style ``snapshot``
(applying the action's own chip movement to the pre-action state, so
the last action of a hand never shows settlement artifacts). Winners
come from pokerkit's settled payoffs; the street board from the ``d db``
deal lines, so no snapshot ever leaks a future street's cards (the
defect the v9 builder repairs on Arena rows cannot occur here).

Action mapping (Arena semantics): ``f`` -> ``fold``; ``cc`` -> ``check``
when nothing is owed else ``call``; ``cbr X`` -> ``raise`` when a bet
stands on the street (blinds count preflop) else ``bet``; ANY action
that commits the actor's whole stack is ``all-in`` with ``toAmount``
equal to the raise-to total (a call-for-less included — the builder's
``wager_lane`` already prices it). PHH ``cbr`` amounts are raise-to
totals, exactly like Arena ``toAmount``.

Refused and counted (never emitted): non-``NT`` variants, any ante,
straddles, null starting stacks, blinds below one, no blinds at all,
and files pokerkit cannot parse. Unknown hole cards (``??``) simply
stay out of ``HoleCardsDealt``.

Two Pluribus quirks, measured on all 10,000 hands (post-showdown
runouts 2026-09-03; half-chip splits re-measured 2026-09-04, correcting
this note):

- **Post-showdown runouts** (92 files): when every live player is
  all-in, the file lists the ``sm`` showdown lines BEFORE the ``d db``
  runout. The runout lines are moved back before the showdown
  (``_repair_actions``) so pokerkit settles on the real board; player
  action indices are untouched.
- **Half-chip splits** (8 files): the file's own ``finishing_stacks``
  are FRACTIONAL — ``x.5`` on each of the two winners of a split pot
  whose pot is odd — and they sum to the starting sum EXACTLY. (No
  Pluribus file drops a chip: measured 2026-09-04 over all 10,000,
  zero files fail their own conservation. The "sum is one short" claim
  this note used to carry was an artefact of casting the file's
  ``Decimal('10112.5')`` to ``int``, not a fact about the data.) An
  Arena replay is integer-chip, so pokerkit gives the whole odd chip
  to one of the two split winners: the replay differs from the file by
  +0.5 on one winner's seat and -0.5 on the other's, and conserves.
  ``tools.validate_phh_replay`` records that class — after checking
  that every differing seat is one the FILE recorded fractionally and
  a winner of the hand — rather than forcing the replay to lose a
  chip. Nothing here rounds: the adapter reports pokerkit's settled
  integer stacks unchanged.

Event ``sequence``: ``ActionTaken`` / ``StreetDealt`` carry the exact
index of their line in the PHH ``actions`` list (the per-decision seed
and row order key on it); the four synthetic pre-events
(``TableStarted``, ``HoleCardsDealt``, the two ``BlindPosted``) carry
negative sequences so they stay strictly before the first action in
every hand, short-handed included — the reconstructed state's
``recentEvents`` reads the blinds from exactly those events.

Offline and read-only: no Arena requests, no credentials, no
promotion. Deterministic: pure functions over the file, stable event
order, stable table ids.

Version: :data:`PHH_REPLAY_VERSION` must be bumped on any semantic
change; the PHH sink stamps it into its sidecar.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

try:
    import pokerkit
except ImportError as error:  # pragma: no cover - environment-dependent
    raise ImportError(
        "tools.phh_replay needs pokerkit==0.7.4; install it with "
        "`PY -m pip install -r requirements-tools.txt`"
    ) from error

PHH_REPLAY_VERSION = "1"

#: The refusal reasons; the PHH sink's sidecar reports one counter per
#: reason, so these strings are the reporting vocabulary.
REASON_VARIANT = "variant"
REASON_ANTES = "antes"
REASON_STRADDLES = "straddles"
REASON_NULL_STARTING_STACK = "null_starting_stack"
REASON_BLINDS_BELOW_ONE = "blinds_below_one"
REASON_MISSING_BLINDS = "missing_blinds"
REASON_PARSE_ERROR = "parse_error"

#: Arena-capitalised street names in deal order.
_STREETS = ("Preflop", "Flop", "Turn", "River")

#: Sequences for the synthetic pre-events; strictly below every PHH
#: action index (which is >= the player count, the deal lines first).
_TABLE_STARTED_SEQ = -4
_HOLE_CARDS_SEQ = -3
_SMALL_BLIND_SEQ = -2
_BIG_BLIND_SEQ = -1


class PhhRefusal(NamedTuple):
    """One refused hand: why, from which file, at which hand index."""

    reason: str
    path: str
    hand_index: int


class RefusalCounter:
    """Per-reason refusal counts; pass one instance into every walk."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def record(self, refusal: PhhRefusal) -> None:
        self._counts[refusal.reason] += 1

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    @property
    def counts(self) -> Mapping[str, int]:
        return dict(self._counts)


def _refusal_reason(hh: Any) -> str | None:
    """The pinned refusal rules, checked in this order."""
    if hh.variant != "NT":
        return REASON_VARIANT
    if any((ante or 0) != 0 for ante in hh.antes):
        return REASON_ANTES
    blinds = [(amount or 0) for amount in hh.blinds_or_straddles]
    posted = [amount for amount in blinds if amount > 0]
    if len(posted) > 2:
        return REASON_STRADDLES
    if any(stack is None for stack in hh.starting_stacks):
        return REASON_NULL_STARTING_STACK
    if any(amount < 1 for amount in posted):
        return REASON_BLINDS_BELOW_ONE
    if not posted:
        return REASON_MISSING_BLINDS
    return None


def _dealer_seat(
    small_pos: int | None, big_pos: int | None, player_count: int
) -> int:
    """The button, 1-based: heads-up it is the small blind's seat."""
    if player_count == 2:
        return small_pos if small_pos is not None else (big_pos - 2) % 2 + 1
    first = small_pos if small_pos is not None else big_pos
    return (first - 2) % player_count + 1


def _blinds(hh: Any) -> tuple[int | None, int | None, int | None, int | None]:
    """(small seat, small amount, big seat, big amount), 1-based seats.

    The PHH heads-up convention (which pokerkit implements): the roles
    swap for two players — p1 posts the big blind and p2 the small (the
    button), so p1 acts first preflop. A small seat of ``None`` means
    the small blind was not posted (dead small blind).
    """
    values = [(amount or 0) for amount in hh.blinds_or_straddles]
    posted = [
        (index, amount) for index, amount in enumerate(values) if amount > 0
    ]
    if len(hh.starting_stacks) == 2:
        small_seat = (1, values[0]) if values[0] > 0 else (None, None)
        big_seat = (0, values[1]) if values[1] > 0 else (1, values[0])
    else:
        small_seat = posted[0] if posted else (None, None)
        big_seat = posted[1] if len(posted) > 1 else (None, None)
        if big_seat[0] is None:
            big_seat = small_seat
            small_seat = (None, None)
    return (
        small_seat[0] + 1 if small_seat[0] is not None else None,
        small_seat[1],
        big_seat[0] + 1 if big_seat[0] is not None else None,
        big_seat[1],
    )


def _repair_actions(actions: Sequence[str]) -> list[str]:
    """Move runout deal lines listed after the showdown to before it.

    A Pluribus quirk: when every live player is all-in, the file records
    the ``sm`` (showdown) lines FIRST and the ``d db`` runout AFTER them.
    pokerkit then settles the hand on the incomplete board (and, for a
    split pot, can hand the odd chip to the wrong seat — measured on
    2026-09-03: 1 of 92 such hands). Reordering the runout before the
    showdown reconstructs the true deal order; player actions keep their
    indices (they all precede the first ``sm``), so every ``ActionTaken``
    sequence stays the file's action index.
    """
    first_sm = next(
        (
            index
            for index, action in enumerate(actions)
            if action.split()[1:2] == ["sm"]
        ),
        None,
    )
    if first_sm is None:
        return list(actions)
    tail = list(actions[first_sm:])
    if not any(action.startswith("d db") for action in tail):
        return list(actions)
    deals = [action for action in tail if action.startswith("d db")]
    others = [action for action in tail if not action.startswith("d db")]
    return list(actions[:first_sm]) + deals + others


def _state_snapshot(state: Any) -> dict[str, Any]:
    """The pre-action facts, copied into plain values while the state is
    still the pre-action state (pokerkit mutates the yielded state in
    place, so everything must be read here and now)."""
    if state.actor_index is None:
        return {
            "stacks": list(state.stacks),
            "bets": list(state.bets),
            "statuses": list(state.statuses),
            "pot": int(state.total_pot_amount),
            "street": state.street_index,
            "actor": None,
            "to_call": 0,
            "can_complete": False,
            "min_to": None,
            "max_to": None,
            "payoffs": list(state.payoffs),
        }
    can_complete = bool(state.can_complete_bet_or_raise_to())
    return {
        "stacks": list(state.stacks),
        "bets": list(state.bets),
        "statuses": list(state.statuses),
        "pot": int(state.total_pot_amount),
        "street": state.street_index,
        "actor": state.actor_index,
        "to_call": int(state.checking_or_calling_amount),
        "can_complete": can_complete,
        "min_to": (
            int(state.min_completion_betting_or_raising_to_amount)
            if can_complete
            else None
        ),
        "max_to": (
            int(state.max_completion_betting_or_raising_to_amount)
            if can_complete
            else None
        ),
        "payoffs": list(state.payoffs),
    }


def _action_event(
    *,
    table_id: str,
    pre: Mapping[str, Any],
    text: str,
    sequence: int,
    names: Sequence[str],
    holes: Mapping[int, list[str]],
    board: Sequence[str],
    starting: Sequence[int],
    dealer: int,
    small_amount: int | None,
    big_amount: int | None,
) -> dict[str, Any]:
    """One ``ActionTaken`` event: pre-action payload, post-action snapshot."""
    parts = text.split()
    seat = int(parts[0][1:])
    actor = seat - 1
    verb = parts[1]

    bet_before = int(pre["bets"][actor])
    stack_before = int(pre["stacks"][actor])
    to_call = int(pre["to_call"])
    pot = int(pre["pot"])
    current_bet = max(pre["bets"]) if pre["bets"] else 0
    can_complete = bool(pre["can_complete"])
    min_to = pre["min_to"] if can_complete else None
    max_to = pre["max_to"] if can_complete else None
    street_name = _STREETS[pre["street"]]

    if verb == "f":
        action = "fold"
        added = 0
    elif verb == "cc":
        if to_call == 0:
            action = "check"
            added = 0
        else:
            added = min(to_call, stack_before)
            action = "call" if added < stack_before else "all-in"
    elif verb == "cbr":
        to_amount = int(parts[2])
        added = to_amount - bet_before
        if to_amount == bet_before + stack_before:
            action = "all-in"
        elif current_bet > 0:
            action = "raise"
        else:
            action = "bet"
    else:  # pragma: no cover - the caller filters to f/cc/cbr
        raise ValueError(f"unexpected PHH action {text!r}")

    available: list[str] = []
    if to_call > 0:
        available.extend(("fold", "call"))
    else:
        available.append("check")
    if can_complete:
        available.append("raise" if current_bet > 0 else "bet")
    if stack_before > 0:
        available.append("all-in")

    allowed: dict[str, Any] = {
        "availableActions": available,
        "canFold": to_call > 0,
        "canCheck": to_call == 0,
        "canCall": to_call > 0,
        "canBet": can_complete and current_bet == 0,
        "canRaise": can_complete and current_bet > 0,
        "canAllIn": stack_before > 0,
        "callChips": to_call,
        "callToAmount": current_bet if to_call > 0 else None,
        "minRaiseTo": min_to,
        "minBet": (
            min_to if can_complete and to_call == 0 and current_bet == 0 else None
        ),
        "raiseRange": (
            {"min": min_to, "max": max_to}
            if can_complete and current_bet > 0
            else None
        ),
        "betRange": (
            {"min": min_to, "max": max_to}
            if can_complete and to_call == 0 and current_bet == 0
            else None
        ),
        "allInToAmount": bet_before + stack_before,
        "maxCommit": bet_before + stack_before,
        "amountSemantics": "toAmount",
    }

    stacks_after = list(pre["stacks"])
    bets_after = list(pre["bets"])
    statuses_after = list(pre["statuses"])
    stacks_after[actor] -= added
    bets_after[actor] += added
    if verb == "f":
        statuses_after[actor] = False

    seats = []
    for index in range(len(starting)):
        stack = int(stacks_after[index])
        if not statuses_after[index]:
            status = "Folded"
        elif stack == 0:
            status = "AllIn"
        else:
            status = "Active"
        seats.append(
            {
                "seatNumber": index + 1,
                "agentId": f"p{index + 1}",
                "agentName": names[index],
                "status": status,
                "stackChips": stack,
                "currentBetChips": int(bets_after[index]),
                "totalCommittedChips": int(starting[index]) - stack,
                "holeCards": list(holes.get(index + 1, [])),
            }
        )

    return {
        "type": "ActionTaken",
        "sequence": sequence,
        "street": street_name,
        "agentId": f"p{seat}",
        "payload": {
            "seatNumber": seat,
            "action": action,
            "amount": added if action == "call" else None,
            "toAmount": (
                bet_before + added
                if action in ("bet", "raise", "all-in")
                else None
            ),
            "callAmount": to_call,
            "stackBefore": stack_before,
            "actorCurrentBetBefore": bet_before,
            "pot": pot,
            "currentBetBefore": current_bet,
            "agentName": names[actor],
            "dealerSeatNumber": dealer,
            "minRaiseToBefore": min_to,
            "allowedActions": allowed,
        },
        "snapshot": {
            "id": table_id,
            "tableId": table_id,
            "seats": seats,
            "boardCards": list(board),
            "potChips": pot + added,
            "currentBet": max(bets_after),
            "street": street_name,
            "smallBlindChips": small_amount,
            "bigBlindChips": big_amount,
        },
    }


def _hand_replay(hh: Any, table_id: str) -> dict[str, Any]:
    """One parsed ``HandHistory`` to the Arena-shaped replay dict."""
    actions = _repair_actions(hh.actions)
    if actions != list(hh.actions):
        hh = dataclasses.replace(hh, actions=actions)
    starting = [int(stack) for stack in hh.starting_stacks]
    player_count = len(starting)
    names = list(hh.players) if hh.players else []
    while len(names) < player_count:
        names.append(f"p{len(names) + 1}")

    small_pos, small_amount, big_pos, big_amount = _blinds(hh)
    dealer = _dealer_seat(small_pos, big_pos, player_count)

    # The deal lines are the authoritative hole cards (``??`` stays out).
    holes: dict[int, list[str]] = {}
    for action_line in hh.actions:
        parts = action_line.split()
        if parts[0] != "d" or parts[1] != "dh" or "?" in parts[3]:
            continue
        seat = int(parts[2][1:])
        holes[seat] = [parts[3][:2], parts[3][2:]]

    events: list[dict[str, Any]] = [
        {
            "type": "TableStarted",
            "sequence": _TABLE_STARTED_SEQ,
            "payload": {"dealerSeatNumber": dealer},
        },
        {
            "type": "HoleCardsDealt",
            "sequence": _HOLE_CARDS_SEQ,
            "payload": {
                "seats": [
                    {
                        "seatNumber": seat,
                        "agentId": f"p{seat}",
                        "agentName": names[seat - 1],
                        "holeCards": list(cards),
                    }
                    for seat, cards in sorted(holes.items())
                ]
            },
        },
    ]
    if small_pos is not None:
        events.append(
            {
                "type": "BlindPosted",
                "sequence": _SMALL_BLIND_SEQ,
                "street": "Preflop",
                "payload": {
                    "blind": "small",
                    "amount": small_amount,
                    "seatNumber": small_pos,
                },
            }
        )
    events.append(
        {
            "type": "BlindPosted",
            "sequence": _BIG_BLIND_SEQ,
            "street": "Preflop",
            "payload": {
                "blind": "big",
                "amount": big_amount,
                "seatNumber": big_pos,
            },
        }
    )

    board: list[str] = []
    showdown_seen = False
    showdown_sequence = len(hh.actions)
    committed = [0] * player_count
    if small_pos is not None:
        committed[small_pos - 1] += small_amount
    committed[big_pos - 1] += big_amount

    pre: dict[str, Any] | None = None
    action_index = 0
    for state, action_text in hh.state_actions:
        if action_text is not None:
            parts = action_text.split()
            if parts[0] == "d":
                if parts[1] == "db":
                    dealt = [
                        parts[2][offset : offset + 2]
                        for offset in range(0, len(parts[2]), 2)
                    ]
                    board.extend(dealt)
                    if state.street_index is None:  # pragma: no cover
                        street_name = _STREETS[min(max(len(board) - 2, 0), 3)]
                    else:
                        street_name = _STREETS[state.street_index]
                    events.append(
                        {
                            "type": "StreetDealt",
                            "sequence": action_index,
                            "street": street_name,
                            "payload": {
                                "street": street_name,
                                "cards": dealt,
                                "boardCards": list(board),
                            },
                        }
                    )
            elif parts[1:2] == ["sm"]:
                showdown_seen = True
                showdown_sequence = action_index
            elif parts[0].startswith("p") and parts[1] in ("f", "cc", "cbr"):
                assert pre is not None and pre["actor"] == int(parts[0][1:]) - 1
                committed[int(parts[0][1:]) - 1] += _added_chips(pre, action_text)
                events.append(
                    _action_event(
                        table_id=table_id,
                        pre=pre,
                        text=action_text,
                        sequence=action_index,
                        names=names,
                        holes=holes,
                        board=board,
                        starting=starting,
                        dealer=dealer,
                        small_amount=small_amount,
                        big_amount=big_amount,
                    )
                )
            action_index += 1
        pre = _state_snapshot(state)

    if showdown_seen:
        events.append(
            {
                "type": "Showdown",
                "sequence": showdown_sequence,
                "street": _STREETS[min(len(board), 3)],
                "payload": {
                    "seats": [
                        {"seatNumber": seat, "agentId": f"p{seat}"}
                        for seat in sorted(holes)
                    ]
                },
            }
        )

    final = pre or {}
    payoffs = [int(value) for value in final.get("payoffs") or ()]
    finishing = [int(value) for value in final.get("stacks") or ()]
    if not finishing:
        raise ValueError("hand history never settled")

    events.append(
        {
            "type": "Payout",
            "sequence": len(hh.actions),
            "payload": {
                "payouts": [
                    {"agentId": f"p{index + 1}", "amount": payoff}
                    for index, payoff in enumerate(payoffs)
                    if payoff > 0
                ]
            },
        }
    )
    events.append(
        {
            "type": "TableEnded",
            "sequence": len(hh.actions) + 1,
            "payload": {},
        }
    )

    table = {
        "id": table_id,
        "tableId": table_id,
        "smallBlindChips": small_amount,
        "bigBlindChips": big_amount,
        "seats": [
            {
                "seatNumber": index + 1,
                "agentId": f"p{index + 1}",
                "agentName": names[index],
                "status": "Settled"
                if bool(final["statuses"][index])
                else "Folded",
                "stackChips": finishing[index],
                "currentBetChips": 0,
                "totalCommittedChips": committed[index],
                "holeCards": list(holes.get(index + 1, [])),
            }
            for index in range(player_count)
        ],
        "winners": [
            {"agentId": f"p{index + 1}"}
            for index, payoff in enumerate(payoffs)
            if payoff > 0
        ],
    }
    return {"table": table, "events": events}


def _added_chips(pre: Mapping[str, Any], text: str) -> int:
    """Chips the actor put in with this action (for gross-commit bookkeeping)."""
    parts = text.split()
    actor = int(parts[0][1:]) - 1
    verb = parts[1]
    if verb == "f":
        return 0
    if verb == "cc":
        return min(int(pre["to_call"]), int(pre["stacks"][actor]))
    return int(parts[2]) - int(pre["bets"][actor])


def _replays_from_file(
    path: Path,
    table_base: str,
    refusals: RefusalCounter | None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """All hands of one ``.phh`` / ``.phhs`` file as (table_id, replay)."""
    if path.suffix == ".phhs":
        with path.open("rb") as stream:
            histories = list(pokerkit.HandHistory.load_all(stream))
    else:
        with path.open("rb") as stream:
            histories = [pokerkit.HandHistory.load(stream)]
    for hand_index, history in enumerate(histories):
        reason = _refusal_reason(history)
        if reason is not None:
            if refusals is not None:
                refusals.record(PhhRefusal(reason, str(path), hand_index))
            continue
        if path.suffix == ".phhs":
            table_id = f"{table_base}#{hand_index}"
        else:
            table_id = table_base
        try:
            yield table_id, _hand_replay(history, table_id)
        except ValueError as error:
            if refusals is not None:
                refusals.record(
                    PhhRefusal(REASON_PARSE_ERROR, str(path), hand_index)
                )
            else:  # pragma: no cover - loud without a counter to read
                raise ValueError(f"{path}: {error}") from error


def replays_from_path(
    path: str | Path,
    *,
    refusals: RefusalCounter | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """One ``.phh`` / ``.phhs`` file to (table_id, replay) pairs.

    ``table_id`` is ``phh/<file stem>``; every hand of a ``.phhs`` file
    adds ``#<hand index>``. Refused hands are counted, never emitted.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    yield from _replays_from_file(
        resolved, f"phh/{resolved.stem}", refusals
    )


def replays_from_root(
    root: str | Path,
    *,
    refusals: RefusalCounter | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Walk a directory recursively for ``*.phh`` / ``*.phhs`` files.

    ``table_id`` is ``phh/<root name>/<relative path without suffix>``
    unless the root is the PHH layout's ``data`` directory itself, whose
    name is dropped — so both ``phh-dataset/data`` and
    ``phh-dataset/data/pluribus`` yield ``phh/pluribus/<session>/<n>``.
    Files are visited in sorted order; refused hands are counted.
    """
    for path in root_files(root):
        yield from _replays_from_file(
            path, root_table_base(root, path), refusals
        )


def root_files(root: str | Path) -> list[Path]:
    """The ``*.phh`` / ``*.phhs`` files under ``root``, in walk order.

    The order :func:`replays_from_root` visits them, factored out so a
    caller can shard the same walk across processes and still produce
    the ids that function would.
    """
    resolved = Path(root)
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return sorted(
        set(resolved.rglob("*.phh")) | set(resolved.rglob("*.phhs"))
    )


def root_table_base(root: str | Path, path: str | Path) -> str:
    """The root-scoped table id stem :func:`replays_from_root` gives ``path``.

    THE id rule, in one place: ``phh/<root name>/<relative path without
    suffix>``, except that the PHH layout's own ``data`` directory drops
    its name, so ``phh-dataset/data`` and ``phh-dataset/data/pluribus``
    both yield ``phh/pluribus/<session>/<n>``. The per-decision training
    seed is ``sha256(seed:table_id:sequence)``, so this string decides
    dataset bytes: shard the walk however you like, but derive every id
    here.
    """
    resolved = Path(root)
    prefix = "phh" if resolved.name == "data" else f"phh/{resolved.name}"
    relative = Path(path).relative_to(resolved)
    return f"{prefix}/{relative.with_suffix('').as_posix()}"


def replays_from_file_in_root(
    root: str | Path,
    path: str | Path,
    *,
    refusals: RefusalCounter | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """One file of ``root``, with the ids :func:`replays_from_root` gives it.

    The shard of :func:`replays_from_root`: iterating
    :func:`root_files` and calling this for each yields exactly what
    that function yields, in the same order, with the same ids — which
    is what lets a builder parallelize per file without moving a single
    dataset byte. Use :func:`replays_from_path` instead only when there
    is no root: its ids are file-stem-scoped and collide between
    sessions.
    """
    yield from _replays_from_file(
        Path(path), root_table_base(root, path), refusals
    )


__all__ = [
    "PHH_REPLAY_VERSION",
    "PhhRefusal",
    "RefusalCounter",
    "replays_from_file_in_root",
    "replays_from_path",
    "replays_from_root",
    "root_files",
    "root_table_base",
]
