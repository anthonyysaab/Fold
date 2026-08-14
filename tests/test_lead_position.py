"""Checks for the standalone player lead-position gauge."""

from __future__ import annotations

import unittest

from lead_position import measure_lead_position, self_test


class LeadPositionTests(unittest.TestCase):
    def test_module_self_test_passes(self) -> None:
        self_test()

    def test_bounds_and_bands(self) -> None:
        dominant = measure_lead_position(
            hero_stack=3_000,
            opponent_stacks=(500, 500, 500, 500, 500),
            position=1.0,
        )
        self.assertEqual(dominant.lead, 100.0)
        self.assertEqual(dominant.band, "leading")

        flat = measure_lead_position(
            hero_stack=1_000,
            opponent_stacks=(1_000, 1_000, 1_000, 1_000, 1_000),
        )
        self.assertEqual(flat.lead, 0.0)
        self.assertEqual(flat.band, "contending")

    def test_lead_is_monotone_in_chip_count(self) -> None:
        stacks = (1_000, 800, 600)
        leads = [
            measure_lead_position(hero_stack=hero, opponent_stacks=stacks).lead
            for hero in (200, 700, 900, 2_000)
        ]
        self.assertEqual(leads, sorted(leads))

    def test_position_accentuates_but_never_flips_the_sign(self) -> None:
        trailer = {"hero_stack": 100, "opponent_stacks": (1_000, 1_000, 1_000)}
        leader = {"hero_stack": 2_000, "opponent_stacks": (900, 700)}

        trailer_early = measure_lead_position(**trailer, position=0.0)
        trailer_late = measure_lead_position(**trailer, position=1.0)
        self.assertLess(trailer_late.lead, trailer_early.lead)
        self.assertLess(trailer_late.lead, 0.0)

        leader_early = measure_lead_position(**leader, position=0.0)
        leader_late = measure_lead_position(**leader, position=1.0)
        self.assertGreater(leader_late.lead, leader_early.lead)
        self.assertGreater(leader_early.lead, 0.0)

    def test_rank_reads_top_three_versus_bottom_three_continuously(self) -> None:
        table = (1_200, 1_100, 1_000, 900, 800)
        second = measure_lead_position(hero_stack=1_150, opponent_stacks=table)
        fourth = measure_lead_position(hero_stack=950, opponent_stacks=table)
        self.assertGreater(second.lead, 0.0)
        self.assertLess(fourth.lead, second.lead)

    def test_validation_rejects_bad_inputs(self) -> None:
        for kwargs in (
            {"hero_stack": 0, "opponent_stacks": (100,)},
            {"hero_stack": 100, "opponent_stacks": ()},
            {"hero_stack": 100, "opponent_stacks": (100,) * 6},
            {"hero_stack": 100, "opponent_stacks": (True,)},
            {"hero_stack": 100, "opponent_stacks": (100,), "position": -0.1},
        ):
            with self.assertRaises(ValueError, msg=f"no error for {kwargs}"):
                measure_lead_position(**kwargs)


if __name__ == "__main__":
    unittest.main()
