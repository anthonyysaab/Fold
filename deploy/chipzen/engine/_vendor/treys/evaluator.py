import itertools
from typing import Sequence

from .card import Card
from .lookup import LookupTable

# Suit bit value (1, 2, 4, 8) -> index 0..3. Tuple lookup beats a dict here.
_SUIT_INDEX = (0, 0, 1, 0, 2, 0, 0, 0, 3)

#: Wheel ranks, from the wheel's "high" card (the five) down to the ace.
_WHEEL_RANKS = (3, 2, 1, 0, 12)


def _straight_high_rank(mask: int) -> int:
    """Highest rank index of a 5-run in a 13-bit rank mask, or -1.

    Rank 0 is the deuce, rank 12 the ace. The wheel (A-2-3-4-5) straddles
    the top bit, so it is checked only AFTER the ordinary runs: it is the
    LOWEST straight, and a hand holding A-2-3-4-5-6 plays the 2-6 run.
    With at most seven cards two 5-runs cannot be disjoint, so the first
    run found high-to-low is THE best run.
    """

    for shift in range(8, -1, -1):
        if (mask >> shift) & 0x1F == 0x1F:
            return shift + 4
    if mask & 0x100F == 0x100F:  # ace (bit 12) plus deuce-through-five (bits 0-3)
        return 3
    return -1


def _run_ranks(high: int) -> tuple[int, ...]:
    """The five rank indices of a straight whose high card is ``high``."""

    if high == 3:
        return _WHEEL_RANKS
    return tuple(range(high, high - 5, -1))


#: Straight lookups for every 13-bit rank mask. 8,192 entries, built once
#: at import: a dict read replaces a nine-shift scan on every evaluation
#: (the 2026-09-02 profile showed the scan at ~1.4us a call).
_STRAIGHT_TABLE: dict[int, int] = {
    mask: _straight_high_rank(mask) for mask in range(1 << 13)
}


class Evaluator:
    """
    Evaluates hand strengths using a variant of Cactus Kev's algorithm:
    http://suffe.cool/poker/evaluator.html

    I make considerable optimizations in terms of speed and memory usage, 
    in fact the lookup table generation can be done in under a second and 
    consequent evaluations are very fast. Won't beat C, but very fast as 
    all calculations are done with bit arithmetic and table lookups. 
    """

    HAND_LENGTH = 2
    BOARD_LENGTH = 5

    def __init__(self) -> None:

        self.table = LookupTable()
        
        self.hand_size_map = {
            5: self._five,
            6: self._six,
            7: self._seven
        }

    def evaluate(self, hand: list[int], board: list[int]) -> int:
        """
        This is the function that the user calls to get a hand rank. 

        No input validation because that's cycles!
        """
        all_cards = hand + board
        return self.hand_size_map[len(all_cards)](all_cards)

    def _five(self, cards: Sequence[int]) -> int:
        """
        Performs an evalution given cards in integer form, mapping them to
        a rank in the range [1, 7462], with lower ranks being more powerful.

        Variant of Cactus Kev's 5 card evaluator, though I saved a lot of memory
        space using a hash table and condensing some of the calculations. 
        """
        # if flush
        if cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000:
            handOR = (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16
            prime = Card.prime_product_from_rankbits(handOR)
            return self.table.flush_lookup[prime]

        # otherwise
        else:
            prime = Card.prime_product_from_hand(cards)
            return self.table.unsuited_lookup[prime]

    def _six(self, cards: Sequence[int]) -> int:
        """
        Selects the best five of six by structure, then ranks it with the
        same five-card lookup as every other path.
        """

        return self._five(self._best_five(cards))

    def _six_reference(self, cards: Sequence[int]) -> int:
        """The original brute-force six-card scan, kept as the oracle.

        Never used on the play path: it exists so the structural selector
        can be verified against an independent implementation (see
        ``tests/test_treys_evaluator.py``).
        """

        minimum = LookupTable.MAX_HIGH_CARD

        for combo in itertools.combinations(cards, 5):

            score = self._five(combo)
            if score < minimum:
                minimum = score

        return minimum

    def _seven(self, cards: Sequence[int]) -> int:
        """
        Selects the best five of seven by structure, then ranks it once.

        The old implementation ran ``_five`` over all 21 subsets, which
        the harvest profiler (2026-09-02) measured as ~52% of runtime.
        Selection is exact, not heuristic: quads, flush/straight-flush,
        full house, straight, trips, two pair, pair, high card — each
        case picks the same five cards the 21-subset scan would minimise
        over, and the rank integer still comes from the same lookup
        table, so agreement with ``_seven_reference`` is by construction
        whenever the selection is right (and is checked by tests).
        """

        return self._five(self._best_five(cards))

    def _seven_reference(self, cards: Sequence[int]) -> int:
        """The original brute-force seven-card scan, kept as the oracle.

        Never used on the play path: it exists so the structural selector
        can be verified against an independent implementation (see
        ``tests/test_treys_evaluator.py``).
        """

        minimum = LookupTable.MAX_HIGH_CARD

        for combo in itertools.combinations(cards, 5):

            score = self._five(combo)
            if score < minimum:
                minimum = score

        return minimum

    def _best_five(self, cards: Sequence[int]) -> list[int]:
        """The five cards whose ``_five`` is minimal over all subsets.

        One histogram pass over the cards classifies the hand into bit
        masks (quads / trips / pairs / seen ranks / per-suit counts) and
        keeps up to four cards per rank, then a case mirroring the poker
        hand ordering selects the five cards. Every selection is the one
        the exhaustive 21-subset scan would pick:

        - quads beat a flush, so they are checked first (both can coexist
          in seven cards);
        - inside a 5+ suited block, a straight flush is looked for before
          falling back to the five highest suited cards (both are always
          better than any straight, so the flush branch returns before
          the straight check);
        - a full house takes the best trips and the best pair, where the
          pair may be a second trips;
        - pairs and kickers are always taken from the top of the deck,
          which is exactly the ordering the lookup tables were built in.

        Only flat arrays are allocated per call (the 2026-09-02 profile
        showed the first version's seventeen per-rank/per-suit lists
        costing more than the selection itself), and no branch rescans
        the cards — the histogram keeps every card the selection can
        need. The flush branch is the exception: it is rare, and the
        suited cards are collected there with one comprehension.
        """

        counts = [0] * 13
        first = [0] * 13
        second = [0] * 13
        third = [0] * 13
        fourth = [0] * 13
        suit_counts = [0] * 4
        rank_mask = 0
        pair_mask = 0
        trips_mask = 0
        quads_mask = 0
        for card in cards:
            rank = (card >> 8) & 0xF
            count = counts[rank]
            counts[rank] = count + 1
            rank_mask |= 1 << rank
            if count == 0:
                first[rank] = card
            elif count == 1:
                second[rank] = card
                pair_mask |= 1 << rank
            elif count == 2:
                third[rank] = card
                pair_mask &= ~(1 << rank)
                trips_mask |= 1 << rank
            elif count == 3:
                fourth[rank] = card
                trips_mask &= ~(1 << rank)
                quads_mask |= 1 << rank
            suit_counts[_SUIT_INDEX[(card >> 12) & 0xF]] += 1

        # Four of a kind: the quads plus the best kicker.
        if quads_mask:
            rank = quads_mask.bit_length() - 1
            quads = [first[rank], second[rank], third[rank], fourth[rank]]
            for kicker in range(12, -1, -1):
                if kicker != rank and first[kicker]:
                    quads.append(first[kicker])
                    return quads

        # Flush, or straight flush when the suited ranks hold a run.
        for suit in range(4):
            if suit_counts[suit] >= 5:
                suited = [
                    card
                    for card in cards
                    if _SUIT_INDEX[(card >> 12) & 0xF] == suit
                ]
                mask = 0
                for card in suited:
                    mask |= 1 << ((card >> 8) & 0xF)
                run = _STRAIGHT_TABLE[mask]
                if run >= 0:
                    by_rank = {}
                    for card in suited:
                        by_rank.setdefault((card >> 8) & 0xF, card)
                    return [by_rank[rank] for rank in _run_ranks(run)]
                suited.sort(key=lambda card: (card >> 8) & 0xF, reverse=True)
                return suited[:5]

        # Full house: best trips, best pair (possibly a second trips).
        if trips_mask and (pair_mask or trips_mask & (trips_mask - 1)):
            trips = trips_mask.bit_length() - 1
            if pair_mask:
                pair = pair_mask.bit_length() - 1
            else:
                pair = (trips_mask ^ (1 << trips)).bit_length() - 1
            return [
                first[trips],
                second[trips],
                third[trips],
                first[pair],
                second[pair],
            ]

        # Straight: one card per rank of the best run.
        run = _STRAIGHT_TABLE[rank_mask]
        if run >= 0:
            return [first[rank] for rank in _run_ranks(run)]

        # Trips plus the two best kickers.
        if trips_mask:
            trips = trips_mask.bit_length() - 1
            kickers = []
            rest = rank_mask
            while rest and len(kickers) < 2:
                rank = rest.bit_length() - 1
                rest ^= 1 << rank
                if rank != trips:
                    kickers.append(first[rank])
            return [first[trips], second[trips], third[trips], *kickers]

        # Two pair: the two best pairs plus the best kicker.
        if pair_mask & (pair_mask - 1):
            high_pair = pair_mask.bit_length() - 1
            low_pair = (pair_mask ^ (1 << high_pair)).bit_length() - 1
            rest = rank_mask
            while rest:
                rank = rest.bit_length() - 1
                rest ^= 1 << rank
                if rank != high_pair and rank != low_pair:
                    return [
                        first[high_pair],
                        second[high_pair],
                        first[low_pair],
                        second[low_pair],
                        first[rank],
                    ]

        # One pair plus the three best kickers.
        if pair_mask:
            pair_rank = pair_mask.bit_length() - 1
            kickers = []
            rest = rank_mask
            while rest and len(kickers) < 3:
                rank = rest.bit_length() - 1
                rest ^= 1 << rank
                if rank != pair_rank:
                    kickers.append(first[rank])
            return [first[pair_rank], second[pair_rank], *kickers]

        # High card: the five best cards.
        five = []
        rest = rank_mask
        while rest and len(five) < 5:
            rank = rest.bit_length() - 1
            rest ^= 1 << rank
            five.append(first[rank])
        return five

    def get_rank_class(self, hr: int) -> int:
        """
        Returns the class of hand given the hand hand_rank
        returned from evaluate. 
        """
        if hr >= 0 and hr <= LookupTable.MAX_ROYAL_FLUSH:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_ROYAL_FLUSH]
        elif hr <= LookupTable.MAX_STRAIGHT_FLUSH:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT_FLUSH]
        elif hr <= LookupTable.MAX_FOUR_OF_A_KIND:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FOUR_OF_A_KIND]
        elif hr <= LookupTable.MAX_FULL_HOUSE:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FULL_HOUSE]
        elif hr <= LookupTable.MAX_FLUSH:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FLUSH]
        elif hr <= LookupTable.MAX_STRAIGHT:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT]
        elif hr <= LookupTable.MAX_THREE_OF_A_KIND:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_THREE_OF_A_KIND]
        elif hr <= LookupTable.MAX_TWO_PAIR:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_TWO_PAIR]
        elif hr <= LookupTable.MAX_PAIR:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_PAIR]
        elif hr <= LookupTable.MAX_HIGH_CARD:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_HIGH_CARD]
        else:
            raise Exception("Inavlid hand rank, cannot return rank class")

    def class_to_string(self, class_int: int) -> str:
        """
        Converts the integer class hand score into a human-readable string.
        """
        return LookupTable.RANK_CLASS_TO_STRING[class_int]

    def get_five_card_rank_percentage(self, hand_rank: int) -> float:
        """
        Scales the hand rank score to the [0.0, 1.0] range.
        """
        return float(hand_rank) / float(LookupTable.MAX_HIGH_CARD)

    def hand_summary(self, board: list[int], hands: list[list[int]]) -> None:
        """
        Gives a sumamry of the hand with ranks as time proceeds. 

        Requires that the board is in chronological order for the 
        analysis to make sense.
        """

        assert len(board) == self.BOARD_LENGTH, "Invalid board length"
        for hand in hands:
            assert len(hand) == self.HAND_LENGTH, "Invalid hand length"

        line_length = 10
        stages = ["FLOP", "TURN", "RIVER"]

        for i in range(len(stages)):
            line = "=" * line_length
            print("{} {} {}".format(line,stages[i],line))
            
            best_rank = 7463  # rank one worse than worst hand
            winners = []
            for player, hand in enumerate(hands):

                # evaluate current board position
                rank = self.evaluate(hand, board[:(i + 3)])
                rank_class = self.get_rank_class(rank)
                class_string = self.class_to_string(rank_class)
                percentage = 1.0 - self.get_five_card_rank_percentage(rank)  # higher better here
                print("Player {} hand = {}, percentage rank among all hands = {}".format(player + 1, class_string, percentage))

                # detect winner
                if rank == best_rank:
                    winners.append(player)
                    best_rank = rank
                elif rank < best_rank:
                    winners = [player]
                    best_rank = rank

            # if we're not on the river
            if i != stages.index("RIVER"):
                if len(winners) == 1:
                    print("Player {} hand is currently winning.\n".format(winners[0] + 1))
                else:
                    print("Players {} are tied for the lead.\n".format([x + 1 for x in winners]))

            # otherwise on all other streets
            else:
                hand_result = self.class_to_string(self.get_rank_class(self.evaluate(hands[winners[0]], board)))
                print()
                print("{} HAND OVER {}".format(line, line))
                if len(winners) == 1:
                    print("Player {} is the winner with a {}\n".format(winners[0] + 1, hand_result))
                else:
                    print("Players {} tied for the win with a {}\n".format([x + 1 for x in winners],hand_result))


class PLOEvaluator(Evaluator):

    HAND_LENGTH = 4

    def evaluate(self, hand: list[int], board: list[int]) -> int:
        minimum = LookupTable.MAX_HIGH_CARD

        for hand_combo in itertools.combinations(hand, 2):
            for board_combo in itertools.combinations(board, 3):
                score = Evaluator._five(self, list(board_combo) + list(hand_combo))
                if score < minimum:
                    minimum = score

        return minimum
