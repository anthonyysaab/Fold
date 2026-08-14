"""Behavior checks for the standalone risk-temperature calculation."""

from __future__ import annotations

import unittest

from risk_temperature import measure_risk_temperature


def _measure(**changes):
    inputs = {
        "hand_strength": 60,
        "purse": 1_000,
        "bet": 100,
        "street": "flop",
        "players": 4,
    }
    inputs.update(changes)
    return measure_risk_temperature(**inputs)


class RiskTemperatureTests(unittest.TestCase):
    def test_extreme_inputs_cover_the_full_scale(self) -> None:
        safest = _measure(
            hand_strength=100, purse=1_000, bet=0, street="river", players=2
        )
        riskiest = _measure(
            hand_strength=0, purse=100, bet=100, street="preflop", players=6
        )
        self.assertEqual(safest.temperature, 0.0)
        self.assertEqual(riskiest.temperature, 100.0)

    def test_weaker_hands_and_more_players_are_hotter(self) -> None:
        self.assertGreater(
            _measure(hand_strength=20).temperature,
            _measure(hand_strength=80).temperature,
        )
        self.assertGreater(
            _measure(players=6).temperature,
            _measure(players=2).temperature,
        )

    def test_earlier_streets_are_hotter(self) -> None:
        readings = [
            _measure(street=street).temperature
            for street in ("preflop", "flop", "turn", "river")
        ]
        self.assertEqual(readings, sorted(readings, reverse=True))

    def test_bet_pressure_is_relative_to_the_purse(self) -> None:
        self.assertGreater(
            _measure(purse=100, bet=100).temperature,
            _measure(purse=1_000, bet=100).temperature,
        )

    def test_invalid_bet_cannot_exceed_the_purse(self) -> None:
        with self.assertRaises(ValueError):
            _measure(purse=100, bet=101)


if __name__ == "__main__":
    unittest.main()
