"""Checks for the standalone bluff advisor."""

from __future__ import annotations

import dataclasses
import unittest

from bluff import (
    BluffSettings,
    DEFAULT_BLUFF_SETTINGS,
    evaluate_bluff,
    self_test,
)

ALWAYS = dataclasses.replace(
    DEFAULT_BLUFF_SETTINGS,
    steal_frequency=1.0,
    continuation_frequency=1.0,
    semi_bluff_frequency=1.0,
    barrel_frequency=1.0,
    probe_frequency=1.0,
    raise_bluff_frequency=1.0,
    river_frequency=1.0,
)


def _combo_draw(**overrides):
    spot = {
        "hole_cards": ("Ah", "Kh"),
        "board_cards": ("7h", "2h", "9c"),
        "street": "flop",
        "pot": 100,
        "stack": 1_000,
        "opponents": 1,
        "hero_aggressions": 1,
        "in_position": True,
        "settings": ALWAYS,
    }
    spot.update(overrides)
    return evaluate_bluff(**spot)


class BluffAdvisorTests(unittest.TestCase):
    def test_module_self_test_passes(self) -> None:
        self_test()

    def test_semi_bluff_prices_itself_in(self) -> None:
        advice = _combo_draw()
        self.assertTrue(advice.bluff)
        self.assertTrue(advice.semi_bluff)
        self.assertEqual(advice.kind, "continuation")
        self.assertEqual(advice.action, "bet")
        # Twelve effective outs at 3.5% clear the called price entirely.
        self.assertEqual(advice.outs, 12.0)
        self.assertEqual(advice.required_fold_probability, 0.0)
        self.assertEqual(advice.pot_fraction, 0.60)

    def test_multiway_spots_are_gated_off(self) -> None:
        advice = _combo_draw(opponents=3)
        self.assertFalse(advice.bluff)
        self.assertEqual(advice.bluff_score, 0.0)
        self.assertIsNone(advice.action)

    def test_estimated_folds_shrink_with_each_opponent(self) -> None:
        single = _combo_draw(opponents=1)
        double = _combo_draw(opponents=2)
        self.assertLess(
            double.estimated_fold_probability, single.estimated_fold_probability
        )

    def test_dry_boards_fold_more_than_wet_ones(self) -> None:
        dry = evaluate_bluff(
            hole_cards=("Qc", "4d"),
            board_cards=("Ks", "7d", "2h"),
            street="flop",
            pot=100,
            stack=1_000,
            opponents=1,
            hero_aggressions=1,
            settings=ALWAYS,
        )
        wet = evaluate_bluff(
            hole_cards=("Qc", "4d"),
            board_cards=("9h", "8h", "7h"),
            street="flop",
            pot=100,
            stack=1_000,
            opponents=1,
            hero_aggressions=1,
            settings=ALWAYS,
        )
        self.assertGreater(
            dry.factors["fold_probability_single"],
            wet.factors["fold_probability_single"],
        )

    def test_river_needs_blockers_or_a_story_and_blockers_add_margin(self) -> None:
        def river(hole, aggressions):
            return evaluate_bluff(
                hole_cards=hole,
                board_cards=("7h", "2h", "9h", "Jc", "3s"),
                street="river",
                pot=100,
                stack=1_000,
                opponents=1,
                hero_aggressions=aggressions,
                in_position=True,
                settings=ALWAYS,
            )

        with_ace = river(("Ah", "Qd"), 1)
        bare_story = river(("Qd", "4c"), 1)
        no_story = river(("Qd", "4c"), 0)
        self.assertTrue(with_ace.bluff)
        self.assertEqual(with_ace.factors["blockers"], 0.8)
        self.assertTrue(bare_story.bluff)  # arena rivers over-fold (audited)
        self.assertGreater(with_ace.margin, bare_story.margin)
        self.assertFalse(no_story.bluff)

    def test_opponent_wildness_shrinks_estimated_folds(self) -> None:
        calm = _combo_draw(opponent_wildness=0.0)
        wild = _combo_draw(opponent_wildness=0.8)
        self.assertAlmostEqual(
            calm.factors["fold_probability_single"]
            - wild.factors["fold_probability_single"],
            0.35 * 0.8,
        )
        with self.assertRaises(ValueError):
            _combo_draw(opponent_wildness=1.5)

    def test_showdown_value_is_never_bluffed(self) -> None:
        pocket_pair = _combo_draw(hole_cards=("9d", "9s"), board_cards=("7h", "2h", "Kc"))
        board_pair = _combo_draw(hole_cards=("Kd", "Qc"), board_cards=("Kh", "7h", "2c"))
        self.assertFalse(pocket_pair.bluff)
        self.assertFalse(board_pair.bluff)

    def test_preflop_steals_come_from_the_middle_of_the_range(self) -> None:
        def preflop(hole):
            return evaluate_bluff(
                hole_cards=hole,
                street="preflop",
                pot=150,
                to_call=100,
                stack=2_000,
                opponents=2,
                settings=ALWAYS,
            )

        steal = preflop(("Kh", "Th"))
        self.assertTrue(steal.bluff)
        self.assertEqual(steal.kind, "steal")
        self.assertEqual(steal.action, "raise")
        self.assertFalse(preflop(("As", "Ad")).bluff)  # premium: value, not bluff
        self.assertFalse(preflop(("7c", "2d")).bluff)  # trash: no steal

    def test_discipline_gates_stop_the_spew(self) -> None:
        barrel_capped = evaluate_bluff(
            hole_cards=("Qd", "4c"),
            board_cards=("7h", "2h", "9h", "Jc", "3s"),
            street="river",
            pot=100,
            stack=1_000,
            opponents=1,
            hero_aggressions=2,
            settings=ALWAYS,
        )
        committed = _combo_draw(pot=1_000, stack=500)
        oversized = _combo_draw(pot=400, stack=500)
        raise_bluff_vs_big_bet = evaluate_bluff(
            hole_cards=("Ah", "Kh"),
            board_cards=("7h", "2h", "9c"),
            street="flop",
            pot=100,
            to_call=80,
            stack=1_000,
            opponents=1,
            settings=ALWAYS,
        )
        for advice, fragment in (
            (barrel_capped, "barrel cap"),
            (committed, "shallow"),
            (oversized, "risk too much"),
            (raise_bluff_vs_big_bet, "too large to bluff-raise"),
        ):
            self.assertFalse(advice.bluff)
            self.assertTrue(
                any(fragment in reason for reason in advice.reasons),
                msg=f"missing {fragment!r} in {advice.reasons}",
            )

    def test_density_and_lead_gradient_scale_the_frequency_cap(self) -> None:
        tilted = dataclasses.replace(DEFAULT_BLUFF_SETTINGS, lead_density_gain=0.5)
        leading = _combo_draw(settings=tilted, lead_position=100.0)
        neutral = _combo_draw(settings=tilted, lead_position=None)
        trailing = _combo_draw(settings=tilted, lead_position=-100.0)
        # Semi-bluff cap 0.80 x density 1.5 / 1.0 / 0.5.
        self.assertEqual(leading.frequency_cap, 1.0)
        self.assertEqual(neutral.frequency_cap, 0.8)
        self.assertEqual(trailing.frequency_cap, 0.4)
        self.assertEqual(leading.factors["lead"], 1.0)

        negative_gain = dataclasses.replace(
            DEFAULT_BLUFF_SETTINGS, lead_density_gain=-0.5
        )
        hunter = _combo_draw(settings=negative_gain, lead_position=-100.0)
        self.assertEqual(hunter.frequency_cap, 1.0)  # trailer bluffs more

        muted = _combo_draw(
            settings=dataclasses.replace(DEFAULT_BLUFF_SETTINGS, bluff_density=0.0)
        )
        self.assertFalse(muted.bluff)
        self.assertEqual(muted.frequency_cap, 0.0)

    def test_lead_input_and_settings_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            _combo_draw(lead_position=150.0)
        for kwargs in ({"bluff_density": 3.0}, {"lead_density_gain": 1.5}):
            with self.assertRaises(ValueError, msg=f"no error for {kwargs}"):
                BluffSettings(**kwargs)

    def test_default_mix_key_varies_with_the_hand_context(self) -> None:
        base = _combo_draw()
        bigger_pot = _combo_draw(pot=250)
        self.assertNotEqual(base.frequency_roll, bigger_pot.frequency_roll)
        keyed = _combo_draw(mix_key="table-123")
        rekeyed = _combo_draw(mix_key="table-456")
        self.assertNotEqual(keyed.frequency_roll, rekeyed.frequency_roll)

    def test_mixing_is_deterministic_and_honors_a_zero_cap(self) -> None:
        self.assertEqual(_combo_draw().to_dict(), _combo_draw().to_dict())
        withheld = _combo_draw(
            settings=dataclasses.replace(
                DEFAULT_BLUFF_SETTINGS,
                continuation_frequency=0.0,
                semi_bluff_frequency=0.0,
            )
        )
        self.assertFalse(withheld.bluff)
        self.assertGreater(withheld.bluff_score, 0.0)  # good spot, mixed out

    def test_malformed_situations_raise(self) -> None:
        for kwargs in (
            {"street": "flop", "board_cards": ()},  # wrong board size
            {"board_cards": ("Ah", "2h", "9c")},  # duplicate card
            {"opponents": 0},
            {"opponents": 9},
            {"pot": 0},
            {"pot": True},
            {"hole_cards": ("Ah", "Xx"), "board_cards": ("7h", "2h", "9c")},
        ):
            spot = {
                "hole_cards": ("Ah", "Kh"),
                "board_cards": ("7h", "2h", "9c"),
                "street": "flop",
                "pot": 100,
                "stack": 1_000,
                "opponents": 1,
            }
            spot.update(kwargs)
            with self.assertRaises(ValueError, msg=f"no error for {kwargs}"):
                evaluate_bluff(**spot)

    def test_settings_validation_rejects_bad_ranges(self) -> None:
        for kwargs in (
            {"max_opponents": 0},
            {"max_opponents": 6},
            {"flop_fold": 1.5},
            {"river_pot_fraction": 0.0},
            {"value_chen": 4.0},  # below min_steal_chen
            {"salt": ""},
        ):
            with self.assertRaises(ValueError, msg=f"no error for {kwargs}"):
                BluffSettings(**kwargs)


if __name__ == "__main__":
    unittest.main()
