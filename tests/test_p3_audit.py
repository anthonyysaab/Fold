"""Invariants the P3 stage relies on but does not check anywhere.

This file was written by an adversarial audit of ``strength_aware_opponent``
and ``tools/build_p3_dataset``. Every test here is one of two kinds, and each
one says which it is:

**Kind A — a missing guard.** The P3 fit is only meaningful if some property
holds, the property does hold today, and nothing in the tree would notice if
it stopped holding. Example: the replay field the price is computed from must
be the pot *before* the actor acted. If the arena ever switched it to the
post-action pot, callers would carry their own call inside the denominator and
``pot_odds`` would encode the label — the fit would still pass its sign
invariant and every number downstream would be confidently wrong.

**Kind B — a characterisation of a measured defect.** These assert what the
shipped artifact *does*, not what it should do, so that fixing it is a
deliberate act that turns a test red rather than a silent change. Each one
names the defect in its docstring and says which direction the assertion must
be flipped when it is fixed. They are not endorsements.

Nothing here modifies P3 code; the audit was read-only.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import unittest
from collections import defaultdict
from pathlib import Path

from engine.strength_aware_opponent import (
    DEFAULT_FIT_PATH,
    FIT_FEATURE_NAMES,
    PRICE_INDEX,
    STRENGTH_INDEX,
    P3Decision,
    StrengthAwareAgent,
    load_fit,
    strength_aware_lineup,
)
from engine.table_simulator import (
    MatchResult,
    ScriptedAgent,
    SimSeat,
    TableSimulator,
)
from tools.build_p3_dataset import (
    DEFAULT_L2,
    DEFAULT_ROOTS,
    _design,
    auc,
    fit_logistic,
)
from tools.collect_foreign_play_data import _action_size, _read_json, _unwrap_rpc

REPO_ROOT = Path(__file__).resolve().parents[1]
FIT_PATH = REPO_ROOT / DEFAULT_FIT_PATH
DATASET_PATH = REPO_ROOT / "artifacts" / "p3" / "p3-dataset.jsonl"


def _require(path: Path):
    if not path.exists():  # pragma: no cover - artifacts present in repo
        raise unittest.SkipTest(f"missing {path}")
    return path


def _rows():
    _require(DATASET_PATH)
    with DATASET_PATH.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _flop_table(*, pot_before: int, bet: int, hole: tuple[str, ...],
                board: tuple[str, ...] = ("2c", "7d", "9h")) -> dict:
    """A format-exact snapshot: villain (seat 2) faces ``bet`` into ``pot_before``."""

    pot = pot_before + bet
    return {
        "id": "audit", "tableId": "audit", "handId": "audit-h0",
        "street": "flop" if len(board) == 3 else
                  {0: "preflop", 4: "turn", 5: "river"}[len(board)],
        "potChips": pot,
        "currentBet": bet,
        "boardCards": list(board),
        "smallBlindChips": 50, "bigBlindChips": 100,
        "selfSeatNumber": 2,
        "seats": [
            {"seatNumber": 1, "agentId": "hero", "status": "Active",
             "stackChips": 1_000_000, "currentBetChips": bet,
             "totalCommittedChips": pot_before // 2 + bet, "holeCards": None},
            {"seatNumber": 2, "agentId": "villain", "status": "Active",
             "stackChips": 1_000_000, "currentBetChips": 0,
             "totalCommittedChips": pot_before // 2, "holeCards": list(hole)},
        ],
        "allowedActions": {
            "canFold": True, "canCheck": False, "canCall": True, "canBet": False,
            "canRaise": True, "canAllIn": True, "callAmount": bet,
            "callChips": bet, "callToAmount": bet, "minBet": None,
            "minRaiseTo": 2 * bet, "betRange": None,
            "raiseRange": {"min": 2 * bet, "max": 1_000_000},
            "allInToAmount": 1_000_000,
            "availableActions": ["fold", "call", "raise", "all-in"],
            "amountSemantics": "toAmount", "reasoningRequired": False,
        },
        "recentEvents": [
            {"type": "BlindPosted", "street": "preflop",
             "summary": {"seatNumber": 1, "action": "smallBlind", "amount": 50}},
            {"type": "BlindPosted", "street": "preflop",
             "summary": {"seatNumber": 2, "action": "bigBlind", "amount": 100}},
        ],
        "simulationRollout": None,
    }


def _mean_fold(agent, *, pot_before: int, bet: int, board=("2c", "7d", "9h"),
               limit: int = 120) -> float:
    """Average fold probability over the villain's whole (unseen) range.

    The bettor cannot see the villain's cards, so the fold frequency it faces
    is this average, not the fold probability of any one holding.
    """

    from engine.schema3 import CARD_CODES

    deck = [code for code in CARD_CODES if code not in board]
    combos = list(itertools.combinations(deck, 2))
    picks = combos if len(combos) <= limit else random.Random(3).sample(combos, limit)
    total = 0.0
    for hole in picks:
        table = _flop_table(pot_before=pot_before, bet=bet, hole=hole, board=board)
        total += agent._fold_probability(table, table["allowedActions"])
    return total / len(picks)


def _standard_errors(model, design) -> list[float]:
    """Asymptotic SEs on the standardised coefficients (ridge included).

    Index 0 is the intercept; regressor ``i`` is at ``i + 1``.
    """

    width = len(model.coefficients)
    centred = [
        [(row[i] - model.mean[i]) / model.std[i] for i in range(width)]
        for row in design
    ]
    size = width + 1
    info = [[0.0] * size for _ in range(size)]
    for row in centred:
        z = model.intercept + sum(c * x for c, x in zip(model.coefficients, row))
        probability = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else (
            math.exp(z) / (1.0 + math.exp(z))
        )
        variance = probability * (1.0 - probability)
        full = [1.0] + list(row)
        for a in range(size):
            for b in range(size):
                info[a][b] += variance * full[a] * full[b]
    for i in range(1, size):
        info[i][i] += DEFAULT_L2
    augmented = [
        info[i][:] + [1.0 if i == j else 0.0 for j in range(size)]
        for i in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        value = augmented[column][column]
        augmented[column] = [x / value for x in augmented[column]]
        for row in range(size):
            if row != column and augmented[row][column] != 0.0:
                factor = augmented[row][column]
                augmented[row] = [
                    x - factor * y for x, y in zip(augmented[row], augmented[column])
                ]
    return [math.sqrt(max(augmented[i][size + i], 0.0)) for i in range(size)]


class StandardErrorInstrumentTests(unittest.TestCase):
    """Validate the SE helper before any test below quotes a z-score.

    House rule: a measurement script is not believable until it has been
    checked against something that cannot come out right if it is wrong.
    """

    def test_standard_errors_shrink_with_the_square_root_of_n(self) -> None:
        def synthetic(count, seed):
            source = random.Random(seed)
            truth = [-3.0, 2.5, 0.4, 0.3, -0.5]
            design, labels = [], []
            for _ in range(count):
                row = [source.random(), 0.1 + 0.6 * source.random(), source.random(),
                       float(source.randint(2, 6)), source.random()]
                z = 0.2 + sum(a * b for a, b in zip(truth, row))
                design.append(row)
                labels.append(1 if source.random() < 1 / (1 + math.exp(-z)) else 0)
            return design, labels

        small_design, small_labels = synthetic(2_000, 1)
        large_design, large_labels = synthetic(8_000, 1)
        small = fit_logistic(small_design, small_labels, l2=1e-6)
        large = fit_logistic(large_design, large_labels, l2=1e-6)
        ratio = (
            _standard_errors(small, small_design)[1 + STRENGTH_INDEX]
            / _standard_errors(large, large_design)[1 + STRENGTH_INDEX]
        )
        self.assertGreater(ratio, 1.6)
        self.assertLess(ratio, 2.5)


class ReplayPotSemanticsTests(unittest.TestCase):
    """Kind A. The price must be built from the PRE-action pot.

    ``_reconstruct_state`` sets ``potChips`` from the ``ActionTaken`` payload's
    ``pot`` field, and ``decision_features`` divides the call by it. Nothing in
    the tree asserts which side of the action that field is on. If it were the
    post-action pot, a caller's own chips would sit in the denominator and a
    folder's would not — ``pot_odds`` would partly encode the label, the sign
    invariant would still pass, and the fitted price response would be an
    artefact. Checked here against the archive itself.
    """

    def test_payload_pot_is_the_pot_before_the_actor_acted(self) -> None:
        files: list[Path] = []
        for root in DEFAULT_ROOTS:
            tables = REPO_ROOT / root / "raw" / "tables"
            if tables.is_dir():
                files.extend(sorted(tables.glob("*.json"))[:80])
        if not files:  # pragma: no cover - archive present in repo
            raise unittest.SkipTest("foreign play data archive is not present")

        before = after = neither = 0
        for path in files:
            try:
                replay = _unwrap_rpc(_read_json(path))
            except (OSError, ValueError):  # pragma: no cover - defensive
                continue
            if not isinstance(replay, dict):  # pragma: no cover - defensive
                continue
            events = replay.get("events")
            if not isinstance(events, list):  # pragma: no cover - defensive
                continue
            actions = [
                event for event in events
                if isinstance(event, dict) and event.get("type") == "ActionTaken"
            ]
            for first, second in zip(actions, actions[1:]):
                if first.get("street") != second.get("street"):
                    continue
                first_payload = first.get("payload") or {}
                second_payload = second.get("payload") or {}
                if first_payload.get("pot") is None or second_payload.get("pot") is None:
                    continue
                _, first_chips = _action_size(first_payload)
                _, second_chips = _action_size(second_payload)
                if first_chips == second_chips:
                    # Cannot tell the two hypotheses apart on this pair.
                    continue
                delta = int(second_payload["pot"]) - int(first_payload["pot"])
                if delta == first_chips:
                    before += 1
                elif delta == second_chips:
                    after += 1
                else:
                    neither += 1

        self.assertGreater(before, 200, "not enough discriminating pairs to decide")
        self.assertEqual(after, 0, "payload['pot'] looks like the POST-action pot")
        self.assertEqual(neither, 0, "payload['pot'] is neither pot; the price is unsafe")


class SplitAndLeakageTests(unittest.TestCase):
    """Kind A. The held-out split and the absence of an outcome-encoding input."""

    def test_no_table_contributes_to_both_splits(self) -> None:
        splits = defaultdict(set)
        for row in _rows():
            splits[row["table_id"]].add(row["split"])
        offenders = [table for table, seen in splits.items() if len(seen) > 1]
        self.assertEqual(offenders, [])

    def test_no_regressor_other_than_strength_separates_the_label(self) -> None:
        """A leaked input shows up as a univariate AUC far from 0.5.

        ``strength_percentile`` is meant to separate — it is the signal. Every
        other recorded quantity, including the ones derived from the pot, must
        not, or the price term is reading the outcome instead of the price.
        """

        rows = _rows()
        labels = [int(row["label_fold"]) for row in rows]
        strength_auc = auc([float(row["strength_percentile"]) for row in rows], labels)
        self.assertLess(strength_auc, 0.25, "the intended signal vanished")
        for name in ("pot_odds", "bet_to_pot", "texture", "position_unit",
                     "active_players", "to_call", "pot"):
            with self.subTest(feature=name):
                value = auc([float(row[name]) for row in rows], labels)
                self.assertLess(
                    abs(value - 0.5), 0.15,
                    f"{name} separates the label like a leak (AUC {value:.3f})",
                )


class RandomnessDisciplineTests(unittest.TestCase):
    """Kind A. ``_fold_probability`` must not move any RNG the caller owns.

    ``ScriptedAgent.decide`` spends exactly one ``roll.random()`` on the fold
    test and then reuses the same stream for aggression and sizing. A card-aware
    override that drew from any shared stream would shift every later draw and
    desynchronise counterfactual replay. Documented in the module; asserted
    nowhere until here.
    """

    def setUp(self) -> None:
        _require(FIT_PATH)
        self.agent = StrengthAwareAgent(
            "audit", 0.226, 0.5, 0.0, 5, fit=load_fit(str(FIT_PATH))
        )
        self.table = _flop_table(pot_before=1_000, bet=500, hole=("Ah", "Kd"))

    def test_it_does_not_advance_a_callers_stream(self) -> None:
        allowed = self.table["allowedActions"]
        reference = random.Random(4242)
        expected = [reference.random() for _ in range(3)]

        stream = random.Random(4242)
        observed = [stream.random()]
        for _ in range(10):
            self.agent._fold_probability(self.table, allowed)
        observed.append(stream.random())
        for _ in range(10):
            self.agent._fold_probability(self.table, allowed)
        observed.append(stream.random())
        self.assertEqual(observed, expected)

    def test_it_does_not_disturb_the_module_level_random(self) -> None:
        allowed = self.table["allowedActions"]
        random.seed(1234)
        expected = random.random()
        random.seed(1234)
        self.agent._fold_probability(self.table, allowed)
        self.assertEqual(random.random(), expected)

    def test_it_is_bit_identical_across_repeats_and_instances(self) -> None:
        allowed = self.table["allowedActions"]
        values = {
            self.agent._fold_probability(self.table, allowed) for _ in range(25)
        }
        self.assertEqual(len(values), 1)
        fresh = StrengthAwareAgent(
            "audit", 0.226, 0.5, 0.0, 5, fit=load_fit(str(FIT_PATH))
        )
        self.assertEqual(fresh._fold_probability(self.table, allowed), values.pop())
        self.assertEqual(self.agent.fallback_count, 0)
        self.assertEqual(fresh.fallback_count, 0)


class ChanceSaltExclusionTests(unittest.TestCase):
    """Kind A. Trace the salt rather than trusting ``reads_cards``.

    ``table_simulator`` resamples a seat only when ``reads_cards`` is exactly
    ``False``. The consequence for a card-aware seat — its holding is frozen
    across every rollout — is asserted here by running the salt, because a
    later refactor could satisfy the flag and still resample the seat.
    """

    def _holes_across_rollouts(self, inner, rollouts: int = 6):
        seen: list[tuple[str, ...]] = []

        class Recorder:
            reads_cards = getattr(inner, "reads_cards", False)
            policy_version = "recorder"
            name = "rec"

            def decide(self, table):
                hero = next(
                    seat for seat in table["seats"]
                    if seat["seatNumber"] == table["selfSeatNumber"]
                )
                if hero.get("holeCards"):
                    seen.append(tuple(hero["holeCards"]))
                return inner.decide(table)

        for rollout in range(rollouts):
            seats = [
                SimSeat(agent_id="owner",
                        agent=ScriptedAgent("owner", 0.5, 0.2, 0.0, 1),
                        seat_number=1, stack=10_000),
                SimSeat(agent_id="rec", agent=Recorder(), seat_number=2, stack=10_000),
                SimSeat(agent_id="other",
                        agent=ScriptedAgent("other", 0.2, 0.4, 0.0, 3),
                        seat_number=3, stack=10_000),
            ]
            simulator = TableSimulator(seed=99)
            result = MatchResult(
                hands=0, big_blind=100,
                chip_deltas={seat.agent_id: 0 for seat in seats},
                decisions={seat.agent_id: 0 for seat in seats},
            )
            simulator._play_hand(seats, button_index=0, hand_index=3,
                                 result=result, chance_salt=(rollout, 0, "owner"))
        return seen

    def test_a_card_blind_seat_is_resampled(self) -> None:
        seen = self._holes_across_rollouts(ScriptedAgent("rec", 0.2, 0.4, 0.0, 3))
        self.assertGreater(len(seen), 1)
        self.assertGreater(len(set(seen)), 1)

    def test_the_strength_aware_seat_is_frozen_across_rollouts(self) -> None:
        _require(FIT_PATH)
        agent = StrengthAwareAgent(
            "rec", 0.2, 0.4, 0.0, 3, fit=load_fit(str(FIT_PATH))
        )
        seen = self._holes_across_rollouts(agent)
        self.assertGreater(len(seen), 1)
        self.assertEqual(
            len(set(seen)), 1,
            "a card-aware seat whose holding moves between rollouts is the "
            "replay desync that closed candidate 0016",
        )


class ArtifactSelfConsistencyTests(unittest.TestCase):
    """Kind A. The artifact's recorded envelope must match the shipped models."""

    def test_recorded_predicted_range_matches_the_shipped_coefficients(self) -> None:
        _require(FIT_PATH)
        document = json.loads(FIT_PATH.read_text(encoding="utf-8"))
        fit = load_fit(str(FIT_PATH))
        train = [row for row in _rows() if row["split"] == "train"]
        predictions = [
            fit.model_for(row["street"]).predict(
                [float(row[name]) for name in FIT_FEATURE_NAMES]
            )
            for row in train
        ]
        recorded = document["band"]["predicted_range"]
        self.assertAlmostEqual(min(predictions), recorded[0], places=9)
        self.assertAlmostEqual(max(predictions), recorded[1], places=9)


class ClampBandWorkloadTests(unittest.TestCase):
    """Kind B — characterisation. The clamp band cannot bind on its own data.

    ``_band`` takes the union of the model's predicted range with the observed
    decile range, so the rail is placed *outside* every prediction the model
    makes on the rows it was fitted on. Its documented purpose is that "a model
    extrapolated off its support must never be able to fold always"; the
    derivation guarantees it never binds on-support and it sits at 0.9954, so
    the opponent can fold 99.5% of the time. When the band is re-derived from
    the observed envelope alone, this assertion must be flipped to expect a
    non-zero clamped count.
    """

    def test_no_training_row_is_clamped(self) -> None:
        _require(FIT_PATH)
        fit = load_fit(str(FIT_PATH))
        low, high = fit.band
        clamped = 0
        for row in _rows():
            value = fit.model_for(row["street"]).predict(
                [float(row[name]) for name in FIT_FEATURE_NAMES]
            )
            if value < low or value > high:
                clamped += 1
        self.assertEqual(clamped, 0)
        self.assertGreater(high, 0.99)


class PriceSupportTests(unittest.TestCase):
    """Kind B — characterisation. The price slope is used far off its support.

    ``pot_odds`` at serve is ``B / (P + 2B)``: 0.250 at a half-pot bet, 0.333 at
    pot, 0.400 at 2x pot, 0.444 at 4x, and ~0.5 in the limit. The postflop rows
    the price coefficient was fitted on never reach 0.42, so every overbet the
    opponent is asked to price is an extrapolation. Flip these bounds only
    together with a fit that has support out there.
    """

    def test_postflop_price_support_stops_short_of_overbet_prices(self) -> None:
        rows = _rows()
        for street in ("flop", "turn"):
            with self.subTest(street=street):
                prices = [row["pot_odds"] for row in rows if row["street"] == street]
                self.assertGreater(len(prices), 50)
                self.assertLess(max(prices), 0.42)

    def test_a_two_times_pot_bet_prices_outside_the_flop_support(self) -> None:
        rows = _rows()
        flop = [row["pot_odds"] for row in rows if row["street"] == "flop"]
        two_times_pot = 2.0 / (1.0 + 2 * 2.0)
        self.assertAlmostEqual(two_times_pot, 0.4)
        self.assertGreater(two_times_pot, max(flop))


class OverbetPunishmentTests(unittest.TestCase):
    """Kind B — was a characterisation of the overbet defect; now its repair.

    As originally written (pre-2026-08-16-repair) this class recorded the
    defect: the postflop price coefficient, extrapolated far beyond its
    fitted support (field bets stop at ~1.1x pot), drove the fold rate up
    faster than the price it laid, so a zero-equity bluff of 4x pot was
    *more* profitable than one of half pot, and its docstring required the
    assertions to be inverted "when the price response is bounded to its
    support". That is what happened: the shipped fit now carries a
    per-model ``price_cap`` (the p99 of the model's own training prices,
    ``price_support`` in the artifact), the price input is clamped from
    above, and beyond the cap the fold probability is flat in bet size.
    The assertions below are the inverted ones the original docstring
    demanded: a bigger overbet buys no extra fold equity, so EV is strictly
    decreasing in size beyond the cap.
    """

    def setUp(self) -> None:
        _require(FIT_PATH)
        self.p3 = StrengthAwareAgent(
            "audit", 0.226, 0.5, 0.0, 5, fit=load_fit(str(FIT_PATH))
        )
        self.blind = ScriptedAgent("blind", 0.226, 0.5, 0.0, 5)

    @staticmethod
    def _bluff_ev(fold_probability: float, pot: float, bet: float) -> float:
        return fold_probability * pot - (1.0 - fold_probability) * bet

    def test_a_card_blind_folder_already_penalises_bigger_bluffs(self) -> None:
        """The control: EV must fall with size against a constant folder."""

        pot = 1_000
        evs = [
            self._bluff_ev(_mean_fold(self.blind, pot_before=pot, bet=int(pot * m)),
                           pot, pot * m)
            for m in (0.5, 1.0, 2.0, 4.0)
        ]
        self.assertEqual(evs, sorted(evs, reverse=True))

    def test_the_clamped_opponent_no_longer_rewards_the_bigger_bluff(self) -> None:
        """The inversion the pre-repair docstring required, verbatim.

        On the flop the fitted support ends at ``price_cap`` 0.3626, below
        the 2x-pot price (0.400), so 2x and 4x pot lay the *same* clamped
        price: identical fold probability, and therefore strictly less EV
        for the bigger bluff. Pre-repair this test asserted ``quad > full``.
        """

        pot = 1_000
        fold_two = _mean_fold(self.p3, pot_before=pot, bet=2 * pot)
        fold_quad = _mean_fold(self.p3, pot_before=pot, bet=4 * pot)
        self.assertAlmostEqual(
            fold_two, fold_quad, places=12,
            msg="above the support cap the fold probability must be flat in size",
        )
        full = self._bluff_ev(
            _mean_fold(self.p3, pot_before=pot, bet=pot), pot, pot
        )
        two = self._bluff_ev(fold_two, pot, 2 * pot)
        quad = self._bluff_ev(fold_quad, pot, 4 * pot)
        # With the fold rate flat above the cap, EV(2x) - EV(4x) is exactly
        # 2 * pot * (1 - fold), which the output band keeps strictly positive
        # — decreasing-in-size beyond the cap is now impossible to lose.
        self.assertGreater(two, quad)
        self.assertGreater(
            full, quad, "pre-repair this was quad > full, the defect itself"
        )
        self.assertEqual(self.p3.fallback_count, 0)


class ThinStreetSignTests(unittest.TestCase):
    """Kind B — characterisation. The turn's own price sign is not resolvable.

    The artifact records ``price_coefficient_positive: true`` for every fitted
    model, checked as a point estimate with no uncertainty. Turn rows are
    pooled into the ``postflop`` model, where the flop supplies two thirds of
    the data. Fitted on turn rows alone the price coefficient is positive but
    inside its own noise, so "worse prices fold more" is asserted on the turn
    rather than established there. Flip this when the turn has enough rows.
    """

    def test_the_turn_price_coefficient_is_inside_its_own_noise(self) -> None:
        rows = [
            row for row in _rows()
            if row["street"] == "turn" and row["split"] == "train"
        ]
        self.assertGreater(len(rows), 50)
        design, labels = _design(rows)
        model = fit_logistic(design, labels, l2=DEFAULT_L2)
        errors = _standard_errors(model, design)
        price_z = model.coefficients[PRICE_INDEX] / errors[1 + PRICE_INDEX]
        strength_z = model.coefficients[STRENGTH_INDEX] / errors[1 + STRENGTH_INDEX]
        self.assertLess(
            abs(price_z), 1.96,
            "the turn price sign became resolvable — update the artifact's claim",
        )
        self.assertLess(strength_z, -1.96)


class LineupDiversityTests(unittest.TestCase):
    """Kind B — characterisation. The P3 lineup is one archetype three times.

    ``calibrated_lineup`` spans fold rates from 0.05 (station) to 0.75 (tight).
    ``strength_aware_lineup`` gives all three seats the fitted model, and
    ``fold_vs_bet`` survives only as the degrade constant, so the three seats
    fold identically in identical spots and differ only in aggression, shove
    rate and mixer seed. A battery built from it has one folding opponent, not
    three.
    """

    def test_every_lineup_seat_folds_identically(self) -> None:
        _require(FIT_PATH)
        fit = load_fit(str(FIT_PATH))
        table = _flop_table(pot_before=1_000, bet=700, hole=("Qs", "8d"))
        allowed = table["allowedActions"]
        values = {
            round(agent._fold_probability(table, allowed), 12)
            for _, agent in strength_aware_lineup(fit)
        }
        self.assertEqual(len(values), 1)
        seats = strength_aware_lineup(fit)
        self.assertEqual(len({agent.fold_vs_bet for _, agent in seats}), 1)


class DecisionFeatureContractTests(unittest.TestCase):
    """Kind A. The vector order the sign invariant is stated against."""

    def test_vector_order_matches_the_named_indices(self) -> None:
        decision = P3Decision(
            street="flop", strength_percentile=0.11, pot_odds=0.22, bet_to_pot=0.33,
            texture=0.44, active_players=3, position_unit=0.55, to_call=10, pot=20,
        )
        self.assertEqual(decision.vector[STRENGTH_INDEX], 0.11)
        self.assertEqual(decision.vector[PRICE_INDEX], 0.22)
        self.assertEqual(len(decision.vector), len(FIT_FEATURE_NAMES))
        self.assertNotIn("bet_to_pot", FIT_FEATURE_NAMES)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
