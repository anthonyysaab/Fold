"""Tests for the PHH -> Arena-shaped replay adapter (``tools.phh_replay``).

The instrument is measured before its results are believed (house
rule): the Pluribus hand ``phh-dataset/data/pluribus/100/0.phh`` is
pasted inline with every expected value pinned by hand; a side-pot
all-in hand, a heads-up hand (the PHH heads-up blind convention), a
hand with unknown hole cards, an ante hand, a straddle hand, and a
two-hand ``.phhs`` file exercise the refusal rules and the mapping;
the smoke test replays the first 200 hands of the real clone and holds
the finishing-stack, chip-conservation, and action-legality invariants.
The v9 builder and the trainer loader prove the output feeds
``replay_rows_v9`` unchanged, exactly as the PHH sink will use it.

Every test that needs pokerkit is skipped when it is not installed
(the stdlib test interpreter carries it via ``requirements-tools.txt``).
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

_POKERKIT_AVAILABLE = importlib.util.find_spec("pokerkit") is not None

#: ``phh-dataset/data/pluribus/100/0.phh`` verbatim.
PLURIBUS_HAND = """variant = 'NT'
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

#: Three players, p3 short-stacked all-in, a p1/p2 side pot, showdown.
SIDE_POT_HAND = """variant = 'NT'
ante_trimming_status = true
antes = [0, 0, 0]
blinds_or_straddles = [50, 100, 0]
min_bet = 100
starting_stacks = [10000, 10000, 500]
actions = ['d dh p1 AsAd', 'd dh p2 9c9d', 'd dh p3 Qh2h', 'p3 cbr 500', 'p1 cc', 'p2 cbr 1500', 'p1 cc', 'd db 2c7dJh', 'p1 cbr 1000', 'p2 cc', 'd db 3s', 'p1 cbr 2000', 'p2 cc', 'd db 5c', 'p1 cc', 'p2 cc', 'p1 sm AsAd', 'p2 sm 9c9d', 'p3 sm Qh2h']
hand = 0
players = ['A', 'B', 'C']
finishing_stacks = [15000, 5500, 0]
"""

#: Heads-up: p2 is the small blind (the button) and acts first preflop.
HEADS_UP_HAND = """variant = 'NT'
ante_trimming_status = true
antes = [0, 0]
blinds_or_straddles = [50, 100]
min_bet = 100
starting_stacks = [10000, 10000]
actions = ['d dh p1 AhKh', 'd dh p2 QsQd', 'p2 cc', 'p1 cbr 300', 'p2 cc', 'd db 2c7d9c', 'p1 cc', 'p2 cbr 400', 'p1 f']
hand = 0
players = ['D', 'E']
finishing_stacks = [9700, 10300]
"""

ANTE_HAND = """variant = 'NT'
antes = [100, 100, 100]
blinds_or_straddles = [50, 100, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000]
actions = ['d dh p1 AsKd', 'd dh p2 9c9d', 'd dh p3 Qh2h', 'p3 f', 'p1 f']
hand = 0
players = ['A', 'B', 'C']
finishing_stacks = [9950, 10100, 9950]
"""

STRADDLE_HAND = """variant = 'NT'
antes = [0, 0, 0, 0]
blinds_or_straddles = [50, 100, 200, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000]
actions = ['d dh p1 AsKd', 'd dh p2 9c9d', 'd dh p3 Qh2h', 'd dh p4 7c7d', 'p4 f', 'p1 f', 'p2 f']
hand = 0
players = ['A', 'B', 'C', 'D']
finishing_stacks = [9900, 10150, 10000, 9950]
"""

UNKNOWN_CARDS_HAND = """variant = 'NT'
ante_trimming_status = true
antes = [0, 0, 0]
blinds_or_straddles = [50, 100, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000]
actions = ['d dh p1 ????', 'd dh p2 9c9d', 'd dh p3 Qh2h', 'p3 f', 'p1 cc', 'p2 cc', 'd db 2c7dJh', 'p1 cc', 'p2 cbr 300', 'p1 f']
hand = 0
players = ['X', 'Y', 'Z']
finishing_stacks = [9900, 10100, 10000]
"""

#: The multi-hand PHH format: one numbered TOML table per hand.
TWO_HANDS_PHHS = "[1]\n" + SIDE_POT_HAND + "\n\n[2]\n" + HEADS_UP_HAND


def _convert(text: str, suffix: str = ".phh"):
    """Write one hand-history text to a temp file and convert it."""
    from tools.phh_replay import RefusalCounter, replays_from_path

    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / f"hand{suffix}"
        path.write_text(text, encoding="utf-8")
        counter = RefusalCounter()
        pairs = list(replays_from_path(path, refusals=counter))
    return pairs, counter


def _of_type(events, event_type):
    return [event for event in events if event["type"] == event_type]


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class PluribusHandTests(unittest.TestCase):
    """The pasted Pluribus hand, every value pinned by hand."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs, cls.counter = _convert(PLURIBUS_HAND)
        cls.table_id, cls.replay = cls.pairs[0]

    def test_one_replay_and_no_refusals(self) -> None:
        self.assertEqual(len(self.pairs), 1)
        self.assertEqual(self.counter.total, 0)
        self.assertEqual(self.table_id, "phh/hand")

    def test_event_order_and_strictly_increasing_sequences(self) -> None:
        events = self.replay["events"]
        self.assertEqual(
            [event["type"] for event in events],
            [
                "TableStarted",
                "HoleCardsDealt",
                "BlindPosted",
                "BlindPosted",
                *["ActionTaken"] * 6,
                "StreetDealt",
                *["ActionTaken"] * 2,
                "StreetDealt",
                *["ActionTaken"] * 2,
                "StreetDealt",
                *["ActionTaken"] * 2,
                "Payout",
                "TableEnded",
            ],
        )
        sequences = [event["sequence"] for event in events]
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(len(sequences), len(set(sequences)))

    def test_dealer_and_blinds(self) -> None:
        started = _of_type(self.replay["events"], "TableStarted")[0]
        self.assertEqual(started["payload"]["dealerSeatNumber"], 6)
        blinds = _of_type(self.replay["events"], "BlindPosted")
        self.assertEqual(
            [blind["payload"] for blind in blinds],
            [
                {"blind": "small", "amount": 50, "seatNumber": 1},
                {"blind": "big", "amount": 100, "seatNumber": 2},
            ],
        )

    def test_hole_cards_are_dealt_to_all_six_seats(self) -> None:
        dealt = _of_type(self.replay["events"], "HoleCardsDealt")[0]
        seats = {seat["seatNumber"]: seat for seat in dealt["payload"]["seats"]}
        self.assertEqual(
            [seat["holeCards"] for seat in dealt["payload"]["seats"]],
            [
                ["Tc", "Qc"],
                ["8s", "4c"],
                ["9c", "3d"],
                ["Ah", "4h"],
                ["Th", "5s"],
                ["6c", "7s"],
            ],
        )
        self.assertEqual(set(seats), {1, 2, 3, 4, 5, 6})

    def test_first_action_is_p3_folding_into_150(self) -> None:
        event = _of_type(self.replay["events"], "ActionTaken")[0]
        payload = event["payload"]
        self.assertEqual(event["sequence"], 6)
        self.assertEqual(event["street"], "Preflop")
        self.assertEqual(payload["seatNumber"], 3)
        self.assertEqual(payload["action"], "fold")
        self.assertEqual(payload["stackBefore"], 10000)
        self.assertEqual(payload["pot"], 150)
        self.assertEqual(payload["currentBetBefore"], 100)
        allowed = payload["allowedActions"]
        self.assertEqual(allowed["callChips"], 100)
        self.assertEqual(allowed["callToAmount"], 100)
        self.assertEqual(allowed["minRaiseTo"], 200)
        self.assertEqual(
            allowed["availableActions"], ["fold", "call", "raise", "all-in"]
        )
        self.assertTrue(allowed["canFold"])
        self.assertFalse(allowed["canCheck"])
        self.assertTrue(allowed["canCall"])
        self.assertFalse(allowed["canBet"])
        self.assertTrue(allowed["canRaise"])
        self.assertTrue(allowed["canAllIn"])
        self.assertEqual(allowed["allInToAmount"], 10000)
        self.assertEqual(allowed["maxCommit"], 10000)
        self.assertEqual(allowed["amountSemantics"], "toAmount")

    def test_action_mapping(self) -> None:
        by_sequence = {
            event["sequence"]: event
            for event in _of_type(self.replay["events"], "ActionTaken")
        }
        raise_event = by_sequence[7]
        self.assertEqual(raise_event["payload"]["action"], "raise")
        self.assertEqual(raise_event["payload"]["toAmount"], 210)
        self.assertIsNone(raise_event["payload"]["amount"])
        self.assertEqual(raise_event["payload"]["pot"], 150)
        call_event = by_sequence[10]
        self.assertEqual(call_event["payload"]["action"], "call")
        self.assertEqual(call_event["payload"]["amount"], 160)
        self.assertIsNone(call_event["payload"]["toAmount"])
        check_event = by_sequence[13]
        self.assertEqual(check_event["payload"]["action"], "check")
        self.assertEqual(check_event["payload"]["callAmount"], 0)
        bet_event = by_sequence[19]
        self.assertEqual(bet_event["street"], "River")
        self.assertEqual(bet_event["payload"]["action"], "bet")
        self.assertEqual(bet_event["payload"]["toAmount"], 230)
        self.assertEqual(bet_event["payload"]["pot"], 520)
        self.assertEqual(bet_event["payload"]["currentBetBefore"], 0)

    def test_street_dealt_cumulative_boards(self) -> None:
        dealt = _of_type(self.replay["events"], "StreetDealt")
        self.assertEqual(
            [(event["street"], event["payload"]["cards"],
              event["payload"]["boardCards"]) for event in dealt],
            [
                ("Flop", ["7d", "5h", "9d"], ["7d", "5h", "9d"]),
                ("Turn", ["7c"], ["7d", "5h", "9d", "7c"]),
                ("River", ["Qh"], ["7d", "5h", "9d", "7c", "Qh"]),
            ],
        )

    def test_winner_and_finishing_stacks(self) -> None:
        self.assertEqual(self.replay["table"]["winners"], [{"agentId": "p1"}])
        finishing = [
            seat["stackChips"] for seat in self.replay["table"]["seats"]
        ]
        self.assertEqual(finishing, [10310, 9900, 10000, 9790, 10000, 10000])
        statuses = [
            seat["status"] for seat in self.replay["table"]["seats"]
        ]
        self.assertEqual(
            statuses,
            ["Settled", "Folded", "Folded", "Folded", "Folded", "Folded"],
        )
        committed = [
            seat["totalCommittedChips"] for seat in self.replay["table"]["seats"]
        ]
        self.assertEqual(committed, [440, 100, 0, 210, 0, 0])

    def test_no_showdown_when_everyone_folded(self) -> None:
        self.assertEqual(_of_type(self.replay["events"], "Showdown"), [])
        self.assertEqual(len(_of_type(self.replay["events"], "Payout")), 1)
        self.assertEqual(len(_of_type(self.replay["events"], "TableEnded")), 1)

    def test_every_action_is_legal_under_its_own_allowed_actions(self) -> None:
        for event in _of_type(self.replay["events"], "ActionTaken"):
            payload = event["payload"]
            self.assertIn(
                payload["action"], payload["allowedActions"]["availableActions"]
            )

    def test_builder_rows_and_trainer_loader(self) -> None:
        from tools.build_phase_a_dataset_v9 import replay_rows_v9
        from test_build_phase_a_dataset_v9 import _FAST

        rows, stats = replay_rows_v9(self.replay, seed=7, **_FAST)
        actions = _of_type(self.replay["events"], "ActionTaken")
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(rows), len(actions))
        self.assertEqual(stats["skipped_decisions"], 0)
        self.assertEqual(stats["board_corrected"], 0)
        self.assertEqual(stats["timeout_actions"], 0)
        self.assertEqual(
            {row["sequence"] for row in rows},
            {event["sequence"] for event in actions},
        )
        with tempfile.TemporaryDirectory() as raw:
            dataset = Path(raw) / "phase-a.jsonl"
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            from engine.v9_trainer import load_phase_a_dataset_v9

            loaded = load_phase_a_dataset_v9(dataset)
        self.assertEqual(len(loaded), 12)


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class SidePotAllInHandTests(unittest.TestCase):
    """A short-stack all-in with a side pot: labelling and chips."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs, cls.counter = _convert(SIDE_POT_HAND)
        cls.replay = cls.pairs[0][1]

    def test_all_in_and_raise_mapping(self) -> None:
        by_sequence = {
            event["sequence"]: event
            for event in _of_type(self.replay["events"], "ActionTaken")
        }
        shove = by_sequence[3]
        self.assertEqual(shove["payload"]["action"], "all-in")
        self.assertEqual(shove["payload"]["toAmount"], 500)
        self.assertEqual(shove["payload"]["seatNumber"], 3)
        raise_event = by_sequence[5]
        self.assertEqual(raise_event["payload"]["action"], "raise")
        self.assertEqual(raise_event["payload"]["toAmount"], 1500)

    def test_short_stack_snapshot_is_all_in_with_zero_stack(self) -> None:
        shove = _of_type(self.replay["events"], "ActionTaken")[0]
        short = next(
            seat
            for seat in shove["snapshot"]["seats"]
            if seat["seatNumber"] == 3
        )
        self.assertEqual(short["status"], "AllIn")
        self.assertEqual(short["stackChips"], 0)
        self.assertEqual(short["currentBetChips"], 500)

    def test_pot_tracks_the_side_pot(self) -> None:
        by_sequence = {
            event["sequence"]: event["payload"]
            for event in _of_type(self.replay["events"], "ActionTaken")
        }
        self.assertEqual(by_sequence[6]["pot"], 2500)
        self.assertEqual(by_sequence[11]["pot"], 5500)
        self.assertEqual(by_sequence[12]["pot"], 7500)

    def test_winners_finishing_stacks_and_showdown(self) -> None:
        self.assertEqual(self.replay["table"]["winners"], [{"agentId": "p1"}])
        finishing = [
            seat["stackChips"] for seat in self.replay["table"]["seats"]
        ]
        self.assertEqual(finishing, [15000, 5500, 0])
        showdown = _of_type(self.replay["events"], "Showdown")
        self.assertEqual(len(showdown), 1)
        self.assertEqual(showdown[0]["street"], "River")

    def test_chip_conservation_and_legality(self) -> None:
        starting = sum(seat["stackChips"] for seat in self.replay["table"]["seats"])
        finishing = 0
        for event in self.replay["events"]:
            if event["type"] != "ActionTaken":
                continue
            payload = event["payload"]
            self.assertIn(
                payload["action"], payload["allowedActions"]["availableActions"]
            )
            seats = event["snapshot"]["seats"]
            self.assertEqual(
                event["snapshot"]["potChips"],
                sum(seat["totalCommittedChips"] for seat in seats),
            )
        finishing = sum(
            seat["stackChips"] for seat in self.replay["table"]["seats"]
        )
        self.assertEqual(finishing, starting)


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class HeadsUpHandTests(unittest.TestCase):
    """Two seats, the PHH heads-up blind convention, position."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs, cls.counter = _convert(HEADS_UP_HAND)
        cls.replay = cls.pairs[0][1]

    def test_two_seats_and_the_heads_up_blind_swap(self) -> None:
        seats = self.replay["table"]["seats"]
        self.assertEqual([seat["seatNumber"] for seat in seats], [1, 2])
        started = _of_type(self.replay["events"], "TableStarted")[0]
        self.assertEqual(started["payload"]["dealerSeatNumber"], 2)
        blinds = _of_type(self.replay["events"], "BlindPosted")
        self.assertEqual(
            [blind["payload"] for blind in blinds],
            [
                {"blind": "small", "amount": 50, "seatNumber": 2},
                {"blind": "big", "amount": 100, "seatNumber": 1},
            ],
        )
        self.assertEqual(self.replay["table"]["smallBlindChips"], 50)
        self.assertEqual(self.replay["table"]["bigBlindChips"], 100)

    def test_action_mapping(self) -> None:
        by_sequence = {
            event["sequence"]: event
            for event in _of_type(self.replay["events"], "ActionTaken")
        }
        limp = by_sequence[2]
        self.assertEqual(limp["payload"]["action"], "call")
        self.assertEqual(limp["payload"]["amount"], 50)
        raise_event = by_sequence[3]
        self.assertEqual(raise_event["payload"]["action"], "raise")
        self.assertEqual(raise_event["payload"]["toAmount"], 300)
        check_event = by_sequence[6]
        self.assertEqual(check_event["payload"]["action"], "check")
        bet_event = by_sequence[7]
        self.assertEqual(bet_event["payload"]["action"], "bet")
        self.assertEqual(bet_event["payload"]["toAmount"], 400)
        fold_event = by_sequence[8]
        self.assertEqual(fold_event["payload"]["action"], "fold")

    def test_winner_and_finishing_stacks(self) -> None:
        self.assertEqual(self.replay["table"]["winners"], [{"agentId": "p2"}])
        finishing = [
            seat["stackChips"] for seat in self.replay["table"]["seats"]
        ]
        self.assertEqual(finishing, [9700, 10300])


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class RefusalTests(unittest.TestCase):
    """The pinned refusal rules count without emitting."""

    def test_ante_hand_is_refused_with_reason_antes(self) -> None:
        pairs, counter = _convert(ANTE_HAND)
        self.assertEqual(pairs, [])
        self.assertEqual(counter.total, 1)
        self.assertEqual(counter.counts, {"antes": 1})

    def test_straddle_hand_is_refused(self) -> None:
        pairs, counter = _convert(STRADDLE_HAND)
        self.assertEqual(pairs, [])
        self.assertEqual(counter.total, 1)
        self.assertEqual(counter.counts, {"straddles": 1})


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class UnknownCardsHandTests(unittest.TestCase):
    """``??`` holdings stay out of HoleCardsDealt; their actions stay."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs, cls.counter = _convert(UNKNOWN_CARDS_HAND)
        cls.replay = cls.pairs[0][1]

    def test_unknown_seat_absent_from_hole_cards_dealt(self) -> None:
        dealt = _of_type(self.replay["events"], "HoleCardsDealt")[0]
        self.assertEqual(
            [seat["seatNumber"] for seat in dealt["payload"]["seats"]], [2, 3]
        )

    def test_unknown_seat_still_acts(self) -> None:
        events = _of_type(self.replay["events"], "ActionTaken")
        p1_actions = [
            event for event in events if event["payload"]["seatNumber"] == 1
        ]
        self.assertEqual(
            [event["payload"]["action"] for event in p1_actions],
            ["call", "check", "fold"],
        )
        for event in p1_actions:
            hero = next(
                seat
                for seat in event["snapshot"]["seats"]
                if seat["seatNumber"] == 1
            )
            self.assertEqual(hero["holeCards"], [])

    def test_winner(self) -> None:
        self.assertEqual(self.replay["table"]["winners"], [{"agentId": "p2"}])


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class MultiHandPhhsTests(unittest.TestCase):
    """A ``.phhs`` file yields one table id per hand, indexed."""

    def test_two_hands_share_the_stem_with_hash_indexes(self) -> None:
        pairs, counter = _convert(TWO_HANDS_PHHS, suffix=".phhs")
        self.assertEqual(counter.total, 0)
        self.assertEqual(
            [table_id for table_id, _ in pairs], ["phh/hand#0", "phh/hand#1"]
        )
        self.assertEqual(pairs[0][1]["table"]["winners"], [{"agentId": "p1"}])
        self.assertEqual(pairs[1][1]["table"]["winners"], [{"agentId": "p2"}])


@unittest.skipUnless(_POKERKIT_AVAILABLE, "pokerkit not installed")
class PluribusSmokeTests(unittest.TestCase):
    """The first 300 hands of the real clone: the invariants hold.

    300, not 200: the first half-chip-split hand of the sorted walk is
    ``102/0`` at index 279, and a smoke test that stops before it never
    touches the fractional class at all.
    """

    ROOT = Path("phh-dataset/data/pluribus")
    HANDS = 300

    #: ``102/0.phh``: the file records 10112.5 on BOTH winners and its
    #: stacks sum to exactly 60000 — nothing is dropped. The integer-chip
    #: replay gives the whole odd chip to p1.
    HALF_CHIP_HAND = "102/0"
    HALF_CHIP_FILE_STACKS = [
        "10112.5", "9775.0", "10000.0", "10000.0", "10112.5", "10000.0"
    ]
    HALF_CHIP_REPLAY_STACKS = [10113, 9775, 10000, 10000, 10112, 10000]

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.ROOT.is_dir():
            raise unittest.SkipTest("PHH Pluribus subset not on disk")

    def _recorded_stacks(self, table_id):
        """The file's own stacks, EXACT: pokerkit parses a PHH stack as
        ``Decimal`` and a split pot can leave ``x.5`` on two seats, so
        nothing here casts to ``int`` — that truncation manufactures a
        one-chip shortfall the files do not have."""
        import pokerkit

        relative = table_id[len("phh/pluribus/"):]
        with (self.ROOT / f"{relative}.phh").open("rb") as stream:
            history = pokerkit.HandHistory.load(stream)
        return (
            [Decimal(str(value)) for value in history.finishing_stacks],
            [Decimal(str(value)) for value in history.starting_stacks],
        )

    @staticmethod
    def _matches_record(finishing, recorded, winners):
        """Equal, or the PHH half-chip split.

        An independent restatement of the rule (the authoritative
        classifier is ``tools.validate_phh_replay._classify_finishing``;
        the adapter's own test must not lean on the validator it is
        measured by): the totals agree exactly and every seat that
        differs differs by less than one chip, on a seat the FILE
        recorded fractionally, and is a winner of the hand.
        """
        if finishing == recorded:
            return True
        if sum(finishing) != sum(recorded):
            return False
        for index, (left, right) in enumerate(zip(finishing, recorded)):
            delta = left - right
            if delta == 0:
                continue
            if abs(delta) >= 1:
                return False
            if right == right.to_integral_value():
                return False
            if index + 1 not in winners:
                return False
        return True

    def test_the_half_chip_split_hand_is_fractional_not_short(self) -> None:
        """``102/0``, pinned by hand: the premise the adapter documents.

        Fails on the pre-2026-09-04 comparison, which read the file's
        stacks through ``int()``, saw a sum one short of 60,000 and
        called the hand a dropped-chip defect.
        """
        from tools.phh_replay import replays_from_path

        path = self.ROOT / f"{self.HALF_CHIP_HAND}.phh"
        if not path.is_file():
            self.skipTest(f"{path} not on disk")
        recorded, starting = self._recorded_stacks(
            f"phh/pluribus/{self.HALF_CHIP_HAND}"
        )
        self.assertEqual(
            [str(value) for value in recorded], self.HALF_CHIP_FILE_STACKS
        )
        self.assertEqual(sum(recorded), Decimal(60000))
        self.assertEqual(sum(starting), Decimal(60000))

        _, replay = list(replays_from_path(path))[0]
        table = replay["table"]
        self.assertEqual(
            [seat["stackChips"] for seat in table["seats"]],
            self.HALF_CHIP_REPLAY_STACKS,
        )
        self.assertEqual(
            table["winners"], [{"agentId": "p1"}, {"agentId": "p5"}]
        )
        finishing = [
            Decimal(str(seat["stackChips"])) for seat in table["seats"]
        ]
        self.assertEqual(
            [str(left - right) for left, right in zip(finishing, recorded)],
            ["0.5", "0.0", "0.0", "0.0", "-0.5", "0.0"],
        )
        self.assertTrue(self._matches_record(finishing, recorded, {1, 5}))
        # The odd chip must land on a winner: pretend it did not.
        self.assertFalse(self._matches_record(finishing, recorded, {5}))
        self.assertFalse(self._matches_record(finishing, recorded, set()))

    def test_first_hands_convert_with_the_invariants(self) -> None:
        from tools.phh_replay import RefusalCounter, replays_from_root

        counter = RefusalCounter()
        checked = 0
        half_chip_hands = 0
        for table_id, replay in replays_from_root(self.ROOT, refusals=counter):
            if checked >= self.HANDS:
                break
            table = replay["table"]
            finishing = [
                Decimal(str(seat["stackChips"])) for seat in table["seats"]
            ]
            winners = {
                int(winner["agentId"][1:])
                for winner in table["winners"]
                if str(winner.get("agentId", "")).startswith("p")
            }
            recorded, starting = self._recorded_stacks(table_id)
            starting_sum = sum(starting)
            self.assertTrue(
                self._matches_record(finishing, recorded, winners),
                f"{table_id}: {finishing} vs file {recorded} "
                f"(winners {sorted(winners)})",
            )
            self.assertEqual(sum(finishing), starting_sum)
            # The file conserves too: no Pluribus hand drops a chip.
            self.assertEqual(
                sum(recorded), starting_sum, f"{table_id}: {recorded}"
            )
            if finishing != recorded:
                half_chip_hands += 1
            for event in replay["events"]:
                if event["type"] != "ActionTaken":
                    continue
                payload = event["payload"]
                self.assertIn(
                    payload["action"],
                    payload["allowedActions"]["availableActions"],
                    f"{table_id} seq {event['sequence']}",
                )
                seats = event["snapshot"]["seats"]
                self.assertEqual(
                    event["snapshot"]["potChips"],
                    sum(seat["totalCommittedChips"] for seat in seats),
                    f"{table_id} seq {event['sequence']}",
                )
            checked += 1
        self.assertEqual(checked, self.HANDS)
        self.assertEqual(counter.total, 0)
        # Coverage, pinned: the walk must reach the fractional class, or
        # this smoke test proves nothing about it.
        self.assertGreaterEqual(half_chip_hands, 1)


if __name__ == "__main__":
    unittest.main()
