"""Validate the PHH -> Arena-shaped replay adapter against its sources.

One mechanism, one module: this is the ONLY place that measures
``tools.phh_replay`` against the PHH files it converts, and it is
read-only — it never writes inside a root, never touches the adapter,
and its only output is the pair of record files it is told to write.
Before any dataset built through the PHH sink is believed, this tool
must pass on every hand under the roots (``PROCEDURES.md`` §16 names it
as the gate; the dated record is
``artifacts/evaluations/phh-adapter-validation-<date>.{json,md}``).

Ground truth is the PHH file itself: for every hand the adapter emits,
the file's own ``starting_stacks`` and ``finishing_stacks`` are read
back (the adapter's table id ``phh/<...>/<file>[#<index>]`` encodes the
path and hand index, so the pairing is exact, refusal handling
included) and the replay is checked invariant by invariant:

1. finishing stacks against the file's ``finishing_stacks``, compared as
   exact ``Decimal`` — pokerkit parses PHH stacks as ``Decimal`` and the
   file may record a HALF chip, so nothing is truncated here (an ``int``
   cast on ``Decimal('10112.5')`` manufactures a one-chip shortfall that
   is not in the file). Exact equality is the reported headline. The one
   accepted class of inequality is the half-chip split: an Arena replay
   is integer-chip, so when a split pot leaves a half chip the file
   records ``x.5`` on both split winners and pokerkit gives the whole
   odd chip to one of them. Such a hand is accepted ONLY when every seat
   whose stack differs (a) differs by less than one chip, (b) is a seat
   the FILE recorded fractionally, and (c) is one of the hand's winners
   (``table['winners']``) — pokerkit can hand the odd chip to the wrong
   seat (``tools.phh_replay._repair_actions`` records the failure mode),
   and that is exactly what clause (c) refuses. Anything else is
   ``mismatch`` and fails. The report always states how many hands are
   NOT exactly equal, and the record's verdict is scoped to match;
2. every ``ActionTaken`` is legal under its own ``allowedActions``:
   the action is in ``availableActions``, ``bet``/``raise``
   ``toAmount`` inside ``betRange``/``raiseRange``, ``call`` ``amount``
   equal to ``callChips``, ``all-in`` ``toAmount`` equal to
   ``allInToAmount``;
3. ``payload.pot`` equals the sum of every seat's committed chips
   before the action (a committed tally seeded from the ``BlindPosted``
   events), the snapshot's ``potChips`` equals pre-action pot plus the
   action's own chips, and the final tally equals the table's
   ``totalCommittedChips`` per seat;
4. the board length seen by every action matches its street —
   0/3/4/5 for preflop/flop/turn/river;
5. chip conservation, both sides: the replay's finishing stacks sum to
   the file's starting sum, AND the file's own finishing stacks sum to
   it too (the second half is a fact about the source data, measured
   rather than assumed);
6. zero refusals and zero decisions the v9 builder would skip for
   7+-way action (seat count against ``risk_temperature.MAX_PLAYERS``,
   imported, not re-authored; seat count bounds the active-player count
   the builder actually tests, so this is a sufficient condition);
7. on a seeded sample of hands, the rows the v9 builder actually builds
   under the tests' ``_FAST`` trial counts. Every number invariant 7
   prints is also asserted: one row per sampled decision, the row's
   street matching the decision's street count for count, the row's
   ``to_call_zero`` matching the adapter's own ``callChips == 0`` count,
   both fold-through lanes actually produced (a corpus that produced
   none would fail), never both lanes on one row, and zero
   ``skipped_decisions`` / ``timeout_actions`` / ``board_corrected`` /
   ``rows_missing``.

The invariant-7 sample is a real seeded sample, not a head slice: each
accepted hand is selected independently when
``blake2b(f"{seed}:{table_id}")`` falls under ``--sample`` divided by the
file count, so the draw is spread over the whole walk and is reproduced
exactly by re-running with the same ``--seed``. ``hands_sampled`` is the
realised size (a Bernoulli draw, so it is near ``--sample``, not equal to
it; on multi-hand ``.phhs`` roots the file count under-counts hands and
the realised sample runs larger). ``--seed`` also seeds the per-decision
row build.

The instrument is measured before the result (DECISIONS §3.5): at
start-up the tool replays four inline cases — a clean integer hand
(every invariant must pass), a copy of it with a finishing stack
corrupted (invariant 1 must fail, class ``mismatch``), a clean
half-chip-split hand (invariant 1 must classify ``half_chip_split``),
and a copy of that hand whose winners list no longer contains the seat
holding the odd chip (invariant 1 must fail, class ``mismatch``). All
four must behave or the tool refuses to report — a checker that cannot
fail proves nothing.

What this instrument CANNOT catch is listed in the report's
``limitations`` block and rendered into the record; read it before
quoting a zero from invariant 7.

A failed invariant means the adapter is not an instrument: report it,
do not tune the adapter here (that is the adapter lane's file).

Usage:
    python -m tools.validate_phh_replay --output <path>.json

``--output`` is required and the sibling ``<path>.md`` is written from
the same report, so the two can never disagree; both refuse to overwrite
an existing file unless ``--overwrite`` is passed (the dated records are
frozen).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

try:
    import pokerkit
except ImportError as error:  # pragma: no cover - environment-dependent
    raise ImportError(
        "tools.validate_phh_replay needs pokerkit==0.7.4; install it with "
        "`PY -m pip install -r requirements-tools.txt`"
    ) from error

from risk_temperature import MAX_PLAYERS
from tools.build_phase_a_dataset import (
    _continuing_opponents,
    _hole_cards_by_seat,
    _in_hand_opponents,
)
from tools.build_phase_a_dataset_v9 import replay_rows_v9
from tools.phh_replay import (
    PHH_REPLAY_VERSION,
    RefusalCounter,
    replays_from_path,
    replays_from_root,
)

DEFAULT_ROOTS = (Path("phh-dataset") / "data" / "pluribus",)
DEFAULT_SAMPLE = 500
DEFAULT_SEED = 7

#: The trial counts every test file uses for cheap builds; invariant 7
#: mirrors them so the sample rows are exactly what tests exercise.
_FAST_TRIALS = {"potential_trials": 32, "equity_trials": 64}

_EXPECTED_BOARD = {"Preflop": 0, "Flop": 3, "Turn": 4, "River": 5}

#: Per-hand example lists in the report are capped; the counts beside
#: them are always the full totals, and the record says when a list is
#: truncated (a 21M-hand root must not inline every example).
_EXAMPLE_CAP = 50

#: What the instrument structurally cannot catch. Each entry names the
#: code that makes it so; measured claims live in ``caveats`` instead.
_LIMITATIONS = (
    "invariant 2 compares two expressions the adapter builds from the "
    "same pre-action state (tools/phh_replay.py `_action_event`): it "
    "catches a disagreement between `availableActions` and the chosen "
    "action, not a wrong shared premise. pokerkit already refused any "
    "action it considered illegal when it applied it.",
    "invariant 6's seat-count test is trivially satisfied on a 6-max-only "
    "corpus; the builder's real skip is an active-player count at the "
    "decision. The direct measurement of that skip is invariant 7's "
    "`skipped_decisions`, which covers only the sample.",
    "invariant 7's `hero_cards_unknown` cannot be non-zero: a decision "
    "whose hero holding is unknown raises in the builder "
    "(tools/build_phase_a_dataset_v9.py) and is counted as "
    "`skipped_decisions`, so no row exists and the decision is counted "
    "in `rows_missing` before the hero test is reached. `rows_missing` "
    "is the counter that actually fires.",
    "invariant 7's `both_lanes_masked_rows` cannot be non-zero: the "
    "builder raises PhaseAInvariantError on that condition before the "
    "row is returned, and that error aborts the run rather than being "
    "counted. It is reported as a restatement of the loader contract, "
    "not as an independent measurement.",
    "invariant 7's `available_label_unmasked_rows` recomputes the "
    "masking condition from the same private helpers the builder "
    "imports (`_hole_cards_by_seat`, `_in_hand_opponents`, "
    "`_continuing_opponents`), so it mirrors the implementation it "
    "checks. `unknown_continuing_opponent_rows` and `rows_missing` are "
    "the two counters in that block that fire independently.",
    "nothing here measures play quality, label correctness against a "
    "solver, or Arena-vs-PHH row equivalence. This is a conversion gate "
    "only. An earlier draft of this record carried a hand-built "
    "Arena-vs-PHH stretch diff; it was removed because its translator "
    "lived outside the repo (unreproducible the moment the record "
    "froze) and because it read the quarantined Arena archive for a "
    "purpose .handoff/DATA.md section 1.1 does not sanction. The "
    "removed text is in the lane handoff for the owner to rule on.",
)

#: ``phh-dataset/data/pluribus/100/0.phh`` verbatim: the clean
#: impossible-by-construction self-check hand (all-integer stacks).
_SELF_CHECK_HAND = """variant = 'NT'
ante_trimming_status = true
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ['d dh p1 TcQc', 'd dh p2 8s4c', 'd dh p3 9c3d', 'd dh p4 Ah4h', 'd dh p5 Th5s', 'd dh p6 6c7s', 'p3 f', 'p4 cbr 210', 'p5 f', 'p6 f', 'p1 cc', 'p2 f', 'd db 7d5h9d', 'p1 cc', 'p4 cc', 'd db 7c', 'p1 cc', 'p4 cc', 'd db Qh', 'p1 cbr 230', 'p4 f']
hand = 0
players = ['MrBlue', 'MrBlonde', 'MrWhite', 'MrPink', 'MrBrown', 'Pluribus']
finishing_stacks = [10310, 9900, 10000, 9790, 10000, 10000]
"""

#: ``phh-dataset/data/pluribus/102/0.phh`` verbatim: a half-chip split.
#: The file records 10112.5 on BOTH winners (p1 and p5) and its stacks
#: sum to exactly 60000 — the integer-chip replay gives the whole odd
#: chip to p1. Without this case the self-check would be blind to the
#: only class of inequality the tool accepts.
_HALF_CHIP_HAND = """variant = 'NT'
ante_trimming_status = true
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ['d dh p1 2cAc', 'd dh p2 9sTc', 'd dh p3 4c6s', 'd dh p4 Ks2d', 'd dh p5 6dAd', 'd dh p6 Jh5d', 'p3 f', 'p4 f', 'p5 cbr 225', 'p6 f', 'p1 cc', 'p2 cc', 'd db Qs9c4s', 'p1 cc', 'p2 cc', 'p5 cc', 'd db As', 'p1 cc', 'p2 cc', 'p5 cc', 'd db 8d', 'p1 cc', 'p2 cc', 'p5 cbr 337', 'p1 cc', 'p2 f', 'p5 sm 6dAd', 'p1 sm 2cAc']
hand = 0
players = ['MrBlue', 'MrBlonde', 'MrWhite', 'MrPink', 'MrBrown', 'Pluribus']
finishing_stacks = [10112.5, 9775.0, 10000.0, 10000.0, 10112.5, 10000.0]
"""


def _dec(value: Any) -> Decimal:
    """Any PHH / replay stack as an exact ``Decimal``.

    pokerkit already yields ``Decimal`` for a fractional PHH stack; the
    ``str()`` round-trip keeps an ``int`` or a stray ``float`` exact too.
    Nothing in this module casts a stack to ``int``.
    """

    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _number(value: Decimal) -> int | float:
    """A ``Decimal`` as the JSON number the record shows."""

    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _signed(value: Decimal) -> str:
    """A delta as an exact signed string (``'+0.5'``, ``'-1'``)."""

    text = format(value.normalize(), "f")
    return text if text.startswith("-") else f"+{text}"


def _dataset_commit() -> str:
    """The ``phh-dataset`` commit the hands were validated against."""

    try:
        completed = subprocess.run(
            ["git", "-C", "phh-dataset", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return "unknown"


def _winner_seats(table: Mapping[str, Any]) -> set[int]:
    """1-based seat numbers of the hand's winners, from the replay."""

    seats: set[int] = set()
    for winner in table.get("winners") or []:
        if not isinstance(winner, Mapping):
            continue
        agent = winner.get("agentId")
        if isinstance(agent, str) and agent.startswith("p"):
            try:
                seats.add(int(agent[1:]))
            except ValueError:  # pragma: no cover - adapter writes p<N>
                continue
    return seats


def _classify_finishing(
    finishing: Sequence[Decimal],
    recorded: Sequence[Decimal],
    winners: set[int],
) -> tuple[str, list[Decimal], list[str]]:
    """Invariant 1's class for one hand: (class, per-seat deltas, reasons).

    ``equal`` is exact ``Decimal`` equality. ``half_chip_split`` is the
    ONE accepted inequality and every clause of it is checked: the totals
    agree exactly, and every seat that differs differs by less than one
    chip, was recorded fractionally by the FILE, and is a winner of the
    hand. Everything else is ``mismatch``, which fails.
    """

    if len(finishing) != len(recorded):
        return "mismatch", [], ["seat_count_differs"]
    deltas = [left - right for left, right in zip(finishing, recorded)]
    if all(delta == 0 for delta in deltas):
        return "equal", deltas, []
    reasons: list[str] = []
    if sum(finishing) != sum(recorded):
        reasons.append("totals_differ")
    for index, delta in enumerate(deltas):
        if delta == 0:
            continue
        seat = index + 1
        if abs(delta) >= 1:
            reasons.append(f"seat{seat}_delta_is_a_whole_chip_or_more")
        if recorded[index] == recorded[index].to_integral_value():
            reasons.append(f"seat{seat}_file_stack_is_whole_chips")
        if seat not in winners:
            reasons.append(f"seat{seat}_is_not_a_winner")
    if reasons:
        return "mismatch", deltas, reasons
    return "half_chip_split", deltas, []


def _action_findings(
    payload: Mapping[str, Any],
    allowed: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    street: Any,
    committed: list[int],
) -> dict[str, Any]:
    """Invariants 2-4 for one ``ActionTaken`` event; mutates ``committed``."""

    action = payload.get("action")
    available = allowed.get("availableActions") or []
    violations: Counter[str] = Counter()

    if action not in available:
        violations["action_not_available"] += 1

    if action in ("bet", "raise"):
        range_key = "betRange" if action == "bet" else "raiseRange"
        rng = allowed.get(range_key)
        to_amount = payload.get("toAmount")
        low = rng.get("min") if isinstance(rng, Mapping) else None
        high = rng.get("max") if isinstance(rng, Mapping) else None
        if isinstance(low, int) and isinstance(high, int) and isinstance(
            to_amount, int
        ):
            if not low <= to_amount <= high:
                violations["range_violation"] += 1
        else:
            violations["range_missing_or_unsized"] += 1
    elif action == "call":
        if payload.get("amount") != allowed.get("callChips"):
            violations["call_amount_mismatch"] += 1
    elif action == "all-in":
        if payload.get("toAmount") != allowed.get("allInToAmount"):
            violations["allin_amount_mismatch"] += 1
    elif action in ("check", "fold"):
        if payload.get("amount") is not None:
            violations["check_or_fold_with_amount"] += 1
    else:
        violations["unexpected_action"] += 1

    seat = payload.get("seatNumber")
    actor = seat - 1 if isinstance(seat, int) else None
    pot = payload.get("pot")
    pot_pre_violation = not (
        isinstance(pot, int) and sum(committed) == pot
    )

    stack_before = payload.get("stackBefore")
    actor_stack_after: Any = None
    for snap_seat in snapshot.get("seats") or []:
        if not isinstance(snap_seat, Mapping):
            continue
        if snap_seat.get("seatNumber") == seat:
            actor_stack_after = snap_seat.get("stackChips")
    added: int | None = None
    if isinstance(stack_before, int) and isinstance(actor_stack_after, int):
        added = stack_before - actor_stack_after
    unreadable_delta = not (
        isinstance(added, int) and 0 <= added <= (stack_before or 0)
    )
    if isinstance(actor, int) and isinstance(added, int) and added >= 0:
        committed[actor] += added

    snapshot_pot = snapshot.get("potChips")
    snapshot_pot_violation = not (
        isinstance(pot, int)
        and isinstance(added, int)
        and isinstance(snapshot_pot, int)
        and snapshot_pot == pot + added
    )

    post_committed = 0
    readable = True
    for snap_seat in snapshot.get("seats") or []:
        if not isinstance(snap_seat, Mapping):
            continue
        chips = snap_seat.get("totalCommittedChips")
        if isinstance(chips, int):
            post_committed += chips
        else:
            readable = False
    post_commit_violation = (
        not readable or not isinstance(snapshot_pot, int)
        or post_committed != snapshot_pot
    )

    board = snapshot.get("boardCards") or []
    expected = _EXPECTED_BOARD.get(street)
    board_violation = expected is None or len(board) != expected

    return {
        "violations": violations,
        "pot_pre_violation": pot_pre_violation,
        "snapshot_pot_violation": snapshot_pot_violation,
        "post_commit_violation": post_commit_violation,
        "stack_delta_unreadable": unreadable_delta,
        "board_violation": board_violation,
        "street": str(street or "?"),
    }


def check_hand(
    table_id: str,
    replay: Mapping[str, Any],
    recorded_starting: Sequence[Any],
    recorded_finishing: Sequence[Any],
) -> dict[str, Any]:
    """Every per-hand invariant for one replay; pure, no IO.

    ``recorded_starting`` / ``recorded_finishing`` are the PHH file's own
    stacks for this hand, in whatever numeric type pokerkit produced —
    they are converted with :func:`_dec` and never truncated. The
    returned dict is the per-hand finding; :func:`validate_roots`
    aggregates them into the report.
    """

    table = replay["table"]
    events = replay["events"]
    seats = table["seats"]
    finishing = [_dec(seat["stackChips"]) for seat in seats]
    committed_table = [
        int(seat["totalCommittedChips"]) for seat in seats
    ]

    recorded = [_dec(value) for value in recorded_finishing]
    starting = [_dec(value) for value in recorded_starting]
    winners = _winner_seats(table)
    inv1_class, deltas, reasons = _classify_finishing(
        finishing, recorded, winners
    )

    committed = [0] * len(seats)
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("type") != "BlindPosted":
            continue
        payload = event.get("payload") or {}
        seat = payload.get("seatNumber")
        amount = payload.get("amount")
        if isinstance(seat, int) and isinstance(amount, int):
            committed[seat - 1] += amount

    violations2: Counter[str] = Counter()
    pot_pre_violations = 0
    snapshot_pot_violations = 0
    post_commit_violations = 0
    stack_delta_unreadable = 0
    board_violations: Counter[str] = Counter()
    actions = 0
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("type") != "ActionTaken":
            continue
        actions += 1
        payload = event.get("payload") or {}
        allowed = payload.get("allowedActions") or {}
        snapshot = event.get("snapshot") or {}
        findings = _action_findings(
            payload,
            allowed,
            snapshot,
            event.get("street"),
            committed,
        )
        violations2.update(findings["violations"])
        if findings["pot_pre_violation"]:
            pot_pre_violations += 1
        if findings["snapshot_pot_violation"]:
            snapshot_pot_violations += 1
        if findings["post_commit_violation"]:
            post_commit_violations += 1
        if findings["stack_delta_unreadable"]:
            stack_delta_unreadable += 1
        if findings["board_violation"]:
            board_violations[findings["street"]] += 1

    committed_seat_violations = sum(
        1
        for left, right in zip(committed, committed_table)
        if left != right
    )
    inv3_ok = (
        pot_pre_violations == 0
        and snapshot_pot_violations == 0
        and post_commit_violations == 0
        and stack_delta_unreadable == 0
        and committed_seat_violations == 0
    )
    inv2_ok = not violations2
    inv4_ok = not board_violations

    conservation_delta = sum(finishing) - sum(starting)
    file_conservation_delta = sum(recorded) - sum(starting)
    inv5_ok = (
        conservation_delta == 0
        and file_conservation_delta == 0
        and len(finishing) == len(starting)
    )

    invariant1: dict[str, Any] = {
        "verdict": "fail" if inv1_class == "mismatch" else "pass",
        "class": inv1_class,
        "exactly_equal": inv1_class == "equal",
        "finishing": [_number(value) for value in finishing],
        "recorded": [_number(value) for value in recorded],
    }
    if inv1_class != "equal":
        invariant1["deltas"] = {
            str(index + 1): _signed(delta)
            for index, delta in enumerate(deltas)
            if delta != 0
        }
        invariant1["winner_seats"] = sorted(winners)
    if reasons:
        invariant1["reasons"] = sorted(set(reasons))

    return {
        "table_id": table_id,
        "players": len(seats),
        "actions": actions,
        "invariant1": invariant1,
        "invariant2": {
            "verdict": "pass" if inv2_ok else "fail",
            "violations": dict(violations2),
        },
        "invariant3": {
            "verdict": "pass" if inv3_ok else "fail",
            "pot_pre_violations": pot_pre_violations,
            "snapshot_pot_violations": snapshot_pot_violations,
            "post_commit_violations": post_commit_violations,
            "stack_delta_unreadable": stack_delta_unreadable,
            "final_committed_seat_violations": committed_seat_violations,
        },
        "invariant4": {
            "verdict": "pass" if inv4_ok else "fail",
            "violations": dict(board_violations),
        },
        "invariant5": {
            "verdict": "pass" if inv5_ok else "fail",
            "conservation_delta": _number(conservation_delta),
            "file_conservation_delta": _number(file_conservation_delta),
        },
    }


def _replay_one(text: str) -> tuple[dict[str, Any], list[Any], list[Any]]:
    """One inline hand text to (replay, starting stacks, file stacks)."""

    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "selfcheck.phh"
        path.write_text(text, encoding="utf-8")
        with path.open("rb") as stream:
            history = pokerkit.HandHistory.load(stream)
        starting = list(history.starting_stacks)
        recorded = list(history.finishing_stacks)
        pairs = list(replays_from_path(path))
    return pairs[0][1], starting, recorded


def _self_check() -> dict[str, bool]:
    """The impossible-by-construction gate the tool must clear first.

    Four cases, and every one must behave or the tool must not report:

    * a clean all-integer hand — every invariant passes;
    * a deep copy of it with one finishing stack corrupted (+7) —
      invariant 1 fails with class ``mismatch``;
    * a clean HALF-CHIP-SPLIT hand — invariant 1 classifies
      ``half_chip_split`` (so the gate is not blind to the fractional
      class it then accepts, which an all-integer self-check would be);
    * a deep copy of that hand whose ``winners`` no longer list the seat
      holding the odd chip — invariant 1 fails with class ``mismatch``.
      This is the clause that refuses an odd chip paid to the wrong
      seat, the failure mode ``tools.phh_replay`` records for pokerkit.
    """

    replay, starting, recorded = _replay_one(_SELF_CHECK_HAND)
    clean = check_hand("selfcheck", replay, starting, recorded)
    clean_passed = all(
        clean[name]["verdict"] == "pass"
        for name in (
            "invariant1",
            "invariant2",
            "invariant3",
            "invariant4",
            "invariant5",
        )
    ) and clean["invariant1"]["class"] == "equal"

    corrupted = copy.deepcopy(replay)
    corrupted["table"]["seats"][0]["stackChips"] = (
        int(_dec(recorded[0])) + 7
    )
    corrupt = check_hand(
        "selfcheck-corrupted", corrupted, starting, recorded
    )
    corrupt_failed = (
        corrupt["invariant1"]["verdict"] == "fail"
        and corrupt["invariant1"]["class"] == "mismatch"
    )

    split_replay, split_starting, split_recorded = _replay_one(
        _HALF_CHIP_HAND
    )
    split = check_hand(
        "selfcheck-half-chip", split_replay, split_starting, split_recorded
    )
    split_classified = (
        split["invariant1"]["class"] == "half_chip_split"
        and split["invariant1"]["verdict"] == "pass"
        and split["invariant1"]["exactly_equal"] is False
        and split["invariant5"]["verdict"] == "pass"
        and len(split["invariant1"].get("winner_seats", ())) >= 2
    )

    wrong_seat = copy.deepcopy(split_replay)
    wrong_seat["table"]["winners"] = wrong_seat["table"]["winners"][:1]
    wrong = check_hand(
        "selfcheck-wrong-seat", wrong_seat, split_starting, split_recorded
    )
    wrong_seat_failed = (
        wrong["invariant1"]["verdict"] == "fail"
        and wrong["invariant1"]["class"] == "mismatch"
        and any(
            reason.endswith("_is_not_a_winner")
            for reason in wrong["invariant1"].get("reasons", ())
        )
    )

    return {
        "clean_passed": clean_passed,
        "corrupted_failed_invariant_1": corrupt_failed,
        "half_chip_split_classified": split_classified,
        "wrong_seat_odd_chip_failed": wrong_seat_failed,
    }


def _file_and_index(
    table_id: str, root: Path, cache: dict[Path, list[Any]]
) -> tuple[list[Any], int]:
    """The file's parsed hands and this hand's index, from the table id."""

    prefix = "phh" if root.name == "data" else f"phh/{root.name}"
    if not table_id.startswith(prefix + "/"):
        raise ValueError(
            f"table id {table_id!r} does not sit under root {root}"
        )
    relative, separator, index_text = table_id[len(prefix) + 1 :].partition(
        "#"
    )
    hand_index = int(index_text) if separator else 0
    path = root / (relative + ".phh")
    if not path.is_file():
        path = root / (relative + ".phhs")
    if not path.is_file():
        raise FileNotFoundError(path)
    histories = cache.get(path)
    if histories is None:
        with path.open("rb") as stream:
            if path.suffix == ".phhs":
                histories = list(pokerkit.HandHistory.load_all(stream))
            else:
                histories = [pokerkit.HandHistory.load(stream)]
        cache[path] = histories
    return histories, hand_index


def _recorded_stacks(
    table_id: str, root: Path, cache: dict[Path, list[Any]]
) -> tuple[list[Decimal], list[Decimal]]:
    """The PHH file's own starting and finishing stacks, exact."""

    histories, hand_index = _file_and_index(table_id, root, cache)
    history = histories[hand_index]
    return (
        [_dec(value) for value in history.starting_stacks],
        [_dec(value) for value in history.finishing_stacks],
    )


def _selected(table_id: str, seed: int, probability: float) -> bool:
    """Is this hand in the seeded invariant-7 sample?

    A Bernoulli draw keyed on the hand's own table id, so the selection
    is independent of walk order, spread over the whole corpus, and
    reproduced exactly by re-running with the same ``--seed``.
    """

    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    digest = hashlib.blake2b(
        f"{seed}:{table_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2.0**64 < probability


def _label_availability(
    replay: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Invariant 7 facts for one hand's built rows: known cards, lanes.

    ``per_street_decisions`` and ``free_spot_decisions`` are counted off
    the adapter's own events, so the row-side counters they sit beside
    (``per_street`` rows, ``free_spot_rows``) have something independent
    to be asserted against rather than merely printed.
    """

    events = replay["events"]
    holes_by_seat = _hole_cards_by_seat(events)
    rows_by_sequence = {
        row.get("sequence"): row for row in rows
    }
    hero_unknown = 0
    opponent_unknown = 0
    available_unmasked = 0
    rows_missing = 0
    lane_active = 0
    lane_aggressive = 0
    both_lanes = 0
    per_street: Counter[str] = Counter()
    per_street_decisions: Counter[str] = Counter()
    free_spots = 0
    free_spot_decisions = 0
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        if event.get("type") != "ActionTaken":
            continue
        payload = event.get("payload") or {}
        seat = payload.get("seatNumber")
        per_street_decisions[str(event.get("street") or "?").casefold()] += 1
        allowed_pre = payload.get("allowedActions") or {}
        if (allowed_pre.get("callChips") or 0) == 0:
            free_spot_decisions += 1
        row = rows_by_sequence.get(event.get("sequence"))
        if row is None:
            rows_missing += 1
            continue
        street = str(row.get("street") or "?")
        per_street[street] += 1
        if row.get("to_call_zero"):
            free_spots += 1
        masks = row.get("masks") or {}
        active_set = masks.get("fold_through_active") == 1
        aggressive_set = masks.get("fold_through_aggressive") == 1
        if active_set:
            lane_active += 1
        if aggressive_set:
            lane_aggressive += 1
        if active_set and aggressive_set:
            both_lanes += 1
        if isinstance(seat, int) and seat not in holes_by_seat:
            hero_unknown += 1
            continue
        snapshot = event.get("snapshot") or {}
        in_hand = _in_hand_opponents(
            {"seats": snapshot.get("seats") or []}, seat
        )
        continuing = _continuing_opponents(events, index, in_hand)
        if not continuing:
            continue
        known = all(
            number in holes_by_seat for number in continuing
        )
        if not known:
            opponent_unknown += 1
        elif masks.get("range_bucket") != 1:
            available_unmasked += 1
    return {
        "rows": len(rows),
        "rows_missing": rows_missing,
        "hero_cards_unknown": hero_unknown,
        "unknown_continuing_opponent_rows": opponent_unknown,
        "available_label_unmasked_rows": available_unmasked,
        "fold_through_active_rows": lane_active,
        "fold_through_aggressive_rows": lane_aggressive,
        "both_lanes_masked_rows": both_lanes,
        "free_spot_rows": free_spots,
        "free_spot_decisions": free_spot_decisions,
        "per_street": dict(per_street),
        "per_street_decisions": dict(per_street_decisions),
    }


def validate_roots(
    roots: Sequence[str | Path],
    *,
    sample: int = DEFAULT_SAMPLE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run every invariant over every accepted hand under the roots."""

    self_check = _self_check()
    if not all(self_check.values()):
        raise SystemExit(
            "self-check did not behave: "
            f"{json.dumps(self_check)} — refusing to report"
        )

    root_paths = [Path(root) for root in roots]
    for root in root_paths:
        if not root.is_dir():
            raise FileNotFoundError(root)
    file_count = 0
    for root in root_paths:
        files = set(root.rglob("*.phh")) | set(root.rglob("*.phhs"))
        file_count += len(files)
    probability = 1.0 if file_count <= 0 else min(1.0, sample / file_count)

    refusals = RefusalCounter()
    cache: dict[Path, list[Any]] = {}
    findings: list[dict[str, Any]] = []
    hands = 0
    actions = 0
    hands_over_max = 0
    decisions_over_max = 0
    sample_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    sample_hands = 0
    sample_decisions = 0
    per_street: Counter[str] = Counter()
    per_street_decisions: Counter[str] = Counter()

    for root in root_paths:
        for table_id, replay in replays_from_root(root, refusals=refusals):
            starting, recorded = _recorded_stacks(table_id, root, cache)
            finding = check_hand(table_id, replay, starting, recorded)
            findings.append(finding)
            hands += 1
            actions += finding["actions"]
            if finding["players"] > MAX_PLAYERS:
                hands_over_max += 1
                decisions_over_max += finding["actions"]
            if _selected(table_id, seed, probability):
                sample_hands += 1
                rows, stats = replay_rows_v9(
                    replay, seed=seed, **_FAST_TRIALS
                )
                sample_counter.update(stats)
                label = _label_availability(replay, rows)
                for key in (
                    "rows",
                    "rows_missing",
                    "hero_cards_unknown",
                    "unknown_continuing_opponent_rows",
                    "available_label_unmasked_rows",
                    "fold_through_active_rows",
                    "fold_through_aggressive_rows",
                    "both_lanes_masked_rows",
                    "free_spot_rows",
                    "free_spot_decisions",
                ):
                    label_counter[key] += label[key]
                sample_decisions += sum(label["per_street_decisions"].values())
                for street, count in label["per_street"].items():
                    per_street[street] += count
                for street, count in label["per_street_decisions"].items():
                    per_street_decisions[street] += count

    inv1_equal = sum(
        1 for f in findings if f["invariant1"]["class"] == "equal"
    )
    inv1_split = sum(
        1
        for f in findings
        if f["invariant1"]["class"] == "half_chip_split"
    )
    inv1_mismatch = sum(
        1
        for f in findings
        if f["invariant1"]["class"] == "mismatch"
    )
    inv2_violations: Counter[str] = Counter()
    inv3 = Counter()
    inv4_violations: Counter[str] = Counter()
    inv5_violations = 0
    inv5_file_violations = 0
    for finding in findings:
        inv2_violations.update(finding["invariant2"]["violations"])
        inv3["pot_pre_violations"] += finding["invariant3"][
            "pot_pre_violations"
        ]
        inv3["snapshot_pot_violations"] += finding["invariant3"][
            "snapshot_pot_violations"
        ]
        inv3["post_commit_violations"] += finding["invariant3"][
            "post_commit_violations"
        ]
        inv3["stack_delta_unreadable"] += finding["invariant3"][
            "stack_delta_unreadable"
        ]
        inv3["final_committed_seat_violations"] += finding["invariant3"][
            "final_committed_seat_violations"
        ]
        for street, count in finding["invariant4"]["violations"].items():
            inv4_violations[street] += count
        if finding["invariant5"]["conservation_delta"] != 0:
            inv5_violations += 1
        if finding["invariant5"]["file_conservation_delta"] != 0:
            inv5_file_violations += 1

    skipped = sample_counter.get("skipped_decisions", 0)
    timeouts = sample_counter.get("timeout_actions", 0)
    board_corrected = sample_counter.get("board_corrected", 0)
    inv7_failures: list[str] = []
    if sample_hands <= 0:
        inv7_failures.append("no hands were sampled")
    if skipped:
        inv7_failures.append(f"skipped_decisions {skipped}")
    if timeouts:
        inv7_failures.append(f"timeout_actions {timeouts}")
    if board_corrected:
        inv7_failures.append(f"board_corrected {board_corrected}")
    for key in (
        "rows_missing",
        "hero_cards_unknown",
        "unknown_continuing_opponent_rows",
        "available_label_unmasked_rows",
        "both_lanes_masked_rows",
    ):
        if label_counter[key]:
            inv7_failures.append(f"{key} {label_counter[key]}")
    if label_counter["rows"] != sample_decisions:
        inv7_failures.append(
            f"rows {label_counter['rows']} != sampled decisions "
            f"{sample_decisions}"
        )
    if dict(per_street) != dict(per_street_decisions):
        inv7_failures.append(
            f"rows by street {dict(sorted(per_street.items()))} != decisions "
            f"by street {dict(sorted(per_street_decisions.items()))}"
        )
    if label_counter["free_spot_rows"] != label_counter["free_spot_decisions"]:
        inv7_failures.append(
            f"free-spot rows {label_counter['free_spot_rows']} != free-spot "
            f"decisions {label_counter['free_spot_decisions']}"
        )
    if sample_hands > 0 and label_counter["fold_through_active_rows"] <= 0:
        inv7_failures.append("no fold_through_active row was produced")
    if sample_hands > 0 and label_counter["fold_through_aggressive_rows"] <= 0:
        inv7_failures.append("no fold_through_aggressive row was produced")

    invariants = {
        "1_finishing_stacks": {
            "passed": inv1_mismatch == 0,
            "statement": (
                "every replay's finishing stacks equal the PHH file's, "
                "compared as exact Decimal; the one accepted inequality "
                "is a half-chip split, where every differing seat "
                "differs by less than one chip, was recorded "
                "fractionally by the file, and is a winner of the hand, "
                "and the totals agree exactly"
            ),
            "hands_exactly_equal": inv1_equal,
            "hands_not_exactly_equal": inv1_split + inv1_mismatch,
            "exact_equality_holds_on_every_hand": inv1_mismatch == 0
            and inv1_split == 0,
            "hands_half_chip_split": inv1_split,
            "hands_mismatch": inv1_mismatch,
            "half_chip_split_hands": [
                {
                    "table_id": finding["table_id"],
                    "deltas": finding["invariant1"]["deltas"],
                    "winner_seats": finding["invariant1"]["winner_seats"],
                }
                for finding in findings
                if finding["invariant1"]["class"] == "half_chip_split"
            ][:_EXAMPLE_CAP],
            "mismatch_examples": [
                {
                    "table_id": finding["table_id"],
                    "reasons": finding["invariant1"].get("reasons", []),
                    "deltas": finding["invariant1"].get("deltas", {}),
                }
                for finding in findings
                if finding["invariant1"]["class"] == "mismatch"
            ][:_EXAMPLE_CAP],
        },
        "2_action_legality": {
            "passed": sum(inv2_violations.values()) == 0 and actions > 0,
            "statement": (
                "every ActionTaken is in its own availableActions with a "
                "bet/raise inside its range, a call equal to callChips, "
                "an all-in equal to allInToAmount, and no amount on a "
                "check or fold"
            ),
            "actions": actions,
            "violations": dict(inv2_violations),
        },
        "3_pot_equals_committed": {
            "passed": sum(inv3.values()) == 0 and actions > 0,
            "statement": (
                "payload.pot equals the committed tally before the "
                "action, the snapshot's potChips equals that pot plus "
                "the action's own chips and the snapshot's per-seat "
                "totalCommittedChips, and the tally ends at the table's"
            ),
            "violations": dict(inv3),
        },
        "4_board_length_by_street": {
            "passed": sum(inv4_violations.values()) == 0 and actions > 0,
            "statement": (
                "the board every action sees is 0/3/4/5 cards for "
                "preflop/flop/turn/river"
            ),
            "violations": dict(inv4_violations),
        },
        "5_chip_conservation": {
            "passed": inv5_violations == 0
            and inv5_file_violations == 0
            and hands > 0,
            "statement": (
                "the replay's finishing stacks sum to the file's "
                "starting sum, and so do the file's own finishing stacks"
            ),
            "hands_conservation_violations": inv5_violations,
            "hands_file_conservation_violations": inv5_file_violations,
        },
        "6_refusals_and_seven_plus": {
            "passed": refusals.total == 0 and decisions_over_max == 0,
            "statement": (
                "the adapter refused no hand, and no hand seats more "
                f"than risk_temperature.MAX_PLAYERS ({MAX_PLAYERS}) "
                "players — a sufficient condition for the builder's "
                "active-player skip, which invariant 7 measures directly"
            ),
            "max_players": MAX_PLAYERS,
            "refusal_counts": dict(refusals.counts),
            "refusal_total": refusals.total,
            "hands_over_max_players": hands_over_max,
            "decisions_over_max_players": decisions_over_max,
        },
        "7_fast_sample_labels": {
            "passed": not inv7_failures,
            "statement": (
                "on the seeded sample: one row per decision, rows by "
                "street equal to decisions by street, free-spot rows "
                "equal to the adapter's callChips==0 decisions, both "
                "fold-through lanes produced and never both on one row, "
                "and zero skipped_decisions / timeout_actions / "
                "board_corrected / rows_missing"
            ),
            "failures": inv7_failures,
            "hands_sampled": sample_hands,
            "hands_total": hands,
            "decisions_sampled": sample_decisions,
            "rows": label_counter["rows"],
            "skipped_decisions": skipped,
            "timeout_actions": timeouts,
            "board_corrected": board_corrected,
            "belief_degrades": sample_counter.get("belief_degrades", 0),
            "allin_call_for_less": sample_counter.get(
                "allin_call_for_less", 0
            ),
            "unsized_wagers": sample_counter.get("unsized_wagers", 0),
            "rows_missing": label_counter["rows_missing"],
            "hero_cards_unknown": label_counter["hero_cards_unknown"],
            "unknown_continuing_opponent_rows": label_counter[
                "unknown_continuing_opponent_rows"
            ],
            "available_label_unmasked_rows": label_counter[
                "available_label_unmasked_rows"
            ],
            "fold_through_active_rows": label_counter[
                "fold_through_active_rows"
            ],
            "fold_through_aggressive_rows": label_counter[
                "fold_through_aggressive_rows"
            ],
            "both_lanes_masked_rows": label_counter[
                "both_lanes_masked_rows"
            ],
            "free_spot_rows": label_counter["free_spot_rows"],
            "free_spot_decisions": label_counter["free_spot_decisions"],
            "per_street_rows": dict(per_street),
            "per_street_decisions": dict(per_street_decisions),
        },
    }

    overall = (
        "pass"
        if all(invariant["passed"] for invariant in invariants.values())
        else "fail"
    )

    caveats: list[str] = []
    if inv1_split:
        caveats.append(
            f"invariant 1: exact equality does NOT hold on {inv1_split} of "
            f"{hands} hands. Those files record a half chip on both winners "
            "of a split pot (their stacks still sum to the starting sum "
            "exactly); an integer-chip replay cannot, so pokerkit gives the "
            "whole odd chip to one of the two. Each such hand was checked "
            "seat by seat: |delta| < 1 chip, on a seat the file recorded "
            "fractionally, and that seat a winner of the hand."
        )
    if inv1_mismatch:
        caveats.append(
            f"invariant 1 FAILED on {inv1_mismatch} hands; see "
            "mismatch_examples."
        )
    share = (100.0 * sample_hands / hands) if hands else 0.0
    caveats.append(
        f"invariant 7 ran on {sample_hands} of {hands} hands "
        f"({share:.1f}%), a seeded Bernoulli sample keyed on each hand's "
        f"table id (seed {seed}, selection probability "
        f"{probability:.6f}); invariants 1-6 ran on every hand."
    )
    if sample_hands and skipped == 0:
        caveats.append(
            "skipped_decisions == 0 is measured on the sample only; the "
            f"other {hands - sample_hands} hands were converted and checked "
            "by invariants 1-6 but their rows were not built."
        )

    return {
        "tool": "tools.validate_phh_replay",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "adapter_version": PHH_REPLAY_VERSION,
        "dataset_commit": _dataset_commit(),
        "roots": [str(root) for root in root_paths],
        "seed": seed,
        "sample": sample,
        "sampling": {
            "rule": (
                "each accepted hand is selected independently when "
                "blake2b('<seed>:<table_id>') / 2**64 < probability; the "
                "draw does not depend on walk order and is reproduced by "
                "re-running with the same --seed"
            ),
            "target": sample,
            "population_files": file_count,
            "probability": probability,
            "hands_selected": sample_hands,
            "note": (
                "probability is --sample divided by the FILE count, so on "
                "multi-hand .phhs roots the realised sample runs larger "
                "than --sample; hands_selected is the realised size"
            ),
        },
        "fast_trials": _FAST_TRIALS,
        "counts": {
            "files": file_count,
            "hands": hands,
            "actions": actions,
        },
        "self_check": self_check,
        "invariants": invariants,
        "caveats": caveats,
        "limitations": list(_LIMITATIONS),
        "overall": overall,
    }


def _render_markdown(report: Mapping[str, Any], output: Path) -> str:
    """The sibling record, rendered from the report it cites.

    Every number in the narrative comes from ``report``; nothing is
    authored beside it, so the ``.md`` and the ``.json`` cannot
    desynchronise. ``output`` supplies only the record's own file names
    and, when the stem carries one, the date the record is filed under —
    which is not necessarily the day the measure ran, so both are shown.
    """

    invariants = report["invariants"]
    counts = report["counts"]
    inv1 = invariants["1_finishing_stacks"]
    inv7 = invariants["7_fast_sample_labels"]
    run_date = str(report["run_at"])[:10]
    stamped = re.search(r"(\d{4}-\d{2}-\d{2})$", output.stem)
    date = stamped.group(1) if stamped else run_date
    lines: list[str] = []
    lines.append(f"# PHH adapter validation — `tools.phh_replay` — {date}")
    lines.append("")
    lines.append(
        f"Record: `{output.name}` + `{output.with_suffix('.md').name}`, "
        f"filed under {date}; measured {report['run_at']}."
    )
    lines.append("")
    lines.append(
        f"**Overall: {report['overall'].upper()}** — {len(invariants)} "
        f"invariants over {counts['hands']} hands / {counts['files']} files "
        f"/ {counts['actions']} actions. Invariants 1-6 ran on every hand; "
        f"invariant 7 ran on {inv7['hands_sampled']} of "
        f"{inv7['hands_total']}. This file is GENERATED from "
        "`tools.validate_phh_replay` alongside its `.json` sibling — do "
        "not hand-edit it; rerun the tool."
    )
    lines.append("")
    lines.append(
        f"Adapter version `PHH_REPLAY_VERSION = "
        f"\"{report['adapter_version']}\"`; dataset commit "
        f"`{report['dataset_commit']}`; roots "
        + ", ".join(f"`{root}`" for root in report["roots"])
        + f"; seed {report['seed']}; fast trials "
        f"`{json.dumps(report['fast_trials'], sort_keys=True)}`; run at "
        f"{report['run_at']}."
    )
    lines.append("")
    lines.append("## Read this first — what the verdict does not say")
    lines.append("")
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    lines.append("## 0. Self-check (run first, reported first)")
    lines.append("")
    lines.append(
        "The tool refuses to report unless all four cases behave; a "
        "checker that cannot fail proves nothing."
    )
    lines.append("")
    lines.append("| case | measured |")
    lines.append("|---|---|")
    for name, value in report["self_check"].items():
        lines.append(f"| `{name}` | **{str(bool(value)).lower()}** |")
    lines.append("")
    lines.append("## Sampling rule (invariant 7)")
    lines.append("")
    sampling = report["sampling"]
    lines.append(f"- {sampling['rule']}.")
    lines.append(
        f"- target `--sample` **{sampling['target']}**, population "
        f"**{sampling['population_files']}** files, probability "
        f"**{sampling['probability']:.6f}**, realised "
        f"**{sampling['hands_selected']}** hands."
    )
    lines.append(f"- {sampling['note']}.")
    lines.append("")

    def _verdict(invariant: Mapping[str, Any]) -> str:
        return "PASS" if invariant["passed"] else "FAIL"

    lines.append("## Invariants")
    lines.append("")
    for name, invariant in invariants.items():
        lines.append(f"### {name} — {_verdict(invariant)}")
        lines.append("")
        lines.append(f"Asserted: {invariant['statement']}.")
        lines.append("")
        lines.append("| key | value |")
        lines.append("|---|---|")
        for key, value in invariant.items():
            if key in ("passed", "statement"):
                continue
            if key in ("half_chip_split_hands", "mismatch_examples"):
                # Only point at the per-hand table when one is rendered
                # below; an empty list must not carry a dangling pointer.
                shown = (
                    f"{len(value)} listed (hand by hand below and in the "
                    "JSON)"
                    if value
                    else "none"
                )
                lines.append(f"| `{key}` | {shown} |")
                continue
            rendered = json.dumps(value, sort_keys=True)
            if len(rendered) > 400:
                rendered = rendered[:380] + "... (truncated; full in JSON)"
            lines.append(f"| `{key}` | `{rendered}` |")
        lines.append("")
    if inv1["hands_half_chip_split"]:
        shown = len(inv1["half_chip_split_hands"])
        lines.append("### The half-chip-split hands, seat by seat")
        lines.append("")
        lines.append(
            f"Showing {shown} of {inv1['hands_half_chip_split']} "
            "(the list in the JSON is capped; the count is not)."
        )
        lines.append("")
        lines.append("| table id | seat deltas (replay - file) | winners |")
        lines.append("|---|---|---|")
        for hand in inv1["half_chip_split_hands"]:
            deltas = ", ".join(
                f"seat {seat} {delta}"
                for seat, delta in sorted(hand["deltas"].items())
            )
            winners = ", ".join(str(seat) for seat in hand["winner_seats"])
            lines.append(f"| `{hand['table_id']}` | {deltas} | {winners} |")
        lines.append("")
    if inv1["hands_mismatch"]:
        lines.append("### The mismatching hands (invariant 1 FAILED)")
        lines.append("")
        lines.append(
            f"Showing {len(inv1['mismatch_examples'])} of "
            f"{inv1['hands_mismatch']}."
        )
        lines.append("")
        lines.append("| table id | seat deltas (replay - file) | why |")
        lines.append("|---|---|---|")
        for hand in inv1["mismatch_examples"]:
            deltas = ", ".join(
                f"seat {seat} {delta}"
                for seat, delta in sorted(hand["deltas"].items())
            )
            lines.append(
                f"| `{hand['table_id']}` | {deltas or 'n/a'} | "
                + ", ".join(hand["reasons"])
                + " |"
            )
        lines.append("")
    lines.append("## What this instrument cannot catch")
    lines.append("")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("## Method note")
    lines.append("")
    lines.append(
        "Ground truth is the PHH file: for every hand the adapter emits, "
        "the file's own `starting_stacks` and `finishing_stacks` are read "
        "back through the table id and compared as exact `Decimal` "
        "(pokerkit parses a PHH stack as `Decimal`; casting to `int` "
        "truncates a half chip and manufactures a shortfall the file does "
        "not have). A failing invariant would mean the adapter is not an "
        "instrument — report it, never tune the adapter to the record."
    )
    lines.append("")
    lines.append(
        "Reproduce: `python -m tools.validate_phh_replay --roots "
        + " ".join(report["roots"])
        + f" --sample {report['sample']} --seed {report['seed']} --output "
        "<path>.json` (`--output` is required; the sibling `.md` is written "
        "beside it and both refuse to overwrite an existing file without "
        "`--overwrite`)."
    )
    lines.append("")
    return "\n".join(lines)


def _print_summary(report: Mapping[str, Any]) -> None:
    print(
        f"validated {report['counts']['hands']} hands "
        f"({report['counts']['files']} files, "
        f"{report['counts']['actions']} actions) through "
        f"tools.phh_replay v{report['adapter_version']} "
        f"(dataset commit {report['dataset_commit']})"
    )
    for name, invariant in report["invariants"].items():
        verdict = "PASS" if invariant["passed"] else "FAIL"
        detail = {
            key: value
            for key, value in invariant.items()
            if key not in ("passed", "statement")
        }
        print(f"  {name}: {verdict} {json.dumps(detail, sort_keys=True)}")
    for caveat in report["caveats"]:
        print(f"  CAVEAT: {caveat}")
    print(f"overall: {report['overall']}")


def _refuse_existing(output: Path, overwrite: bool) -> None:
    """Refuse to rewrite a record pair that is already on disk.

    Called twice on purpose: once before the walk, so a mistyped rerun
    costs a second rather than an hour, and once at the write itself, so
    the guard is not merely advisory.
    """

    if overwrite:
        return
    existing = [
        path
        for path in (output, output.with_suffix(".md"))
        if path.exists()
    ]
    if existing:
        raise SystemExit(
            "refusing to overwrite "
            + ", ".join(str(path) for path in existing)
            + " -- the dated records are frozen (CLAUDE.md section 6.8); "
            "choose another --output or pass --overwrite when the owner "
            "asks for a regeneration"
        )


def _write_report(
    report: Mapping[str, Any], output: Path, overwrite: bool
) -> None:
    """Write the ``.json`` and its ``.md`` sibling, or refuse."""

    markdown = output.with_suffix(".md")
    _refuse_existing(output, overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(_render_markdown(report, output), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {markdown}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate tools.phh_replay against the PHH files it converts."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[str(root) for root in DEFAULT_ROOTS],
        help="directories walked recursively for *.phh and *.phhs",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "path of the JSON report; the sibling <stem>.md is written "
            "from the same run. There is no default: the dated record is "
            "a frozen artifact and must be named on purpose"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "rewrite an existing report/record pair (they are frozen; "
            "pass this only for a deliberate regeneration)"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=(
            "target size of the seeded random sample of hands whose v9 "
            "rows are built with the fast trial counts for invariant 7"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="seeds BOTH the invariant-7 hand selection and the row build",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.sample < 1:
        raise SystemExit("--sample must be >= 1")
    output = Path(args.output)
    if output.suffix != ".json":
        raise SystemExit("--output must end in .json")
    _refuse_existing(output, args.overwrite)
    report = validate_roots(
        args.roots, sample=args.sample, seed=args.seed
    )
    _write_report(report, output, args.overwrite)
    _print_summary(report)
    return 0 if report["overall"] == "pass" else 1


__all__ = [
    "DEFAULT_ROOTS",
    "check_hand",
    "validate_roots",
]


if __name__ == "__main__":
    sys.exit(main())
