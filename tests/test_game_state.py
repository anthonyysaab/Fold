"""Checks for validation at the untrusted Arena snapshot boundary."""

from __future__ import annotations

import unittest

from engine.game_state import ArenaSnapshotError, _integer


class GameStateValidationTests(unittest.TestCase):
    def test_non_finite_chip_values_raise_snapshot_errors(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ArenaSnapshotError):
                _integer(value, "chips")


if __name__ == "__main__":
    unittest.main()
