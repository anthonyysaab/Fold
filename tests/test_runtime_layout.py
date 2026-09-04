"""Small checks for the distilled runtime layout."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.hand_strength import prewarm
from engine.poker_policy import (
    AggressivePokerPolicy,
    PokerPolicy,
    build_policy,
)


ROOT = Path(__file__).resolve().parents[1]


class RuntimeLayoutTests(unittest.TestCase):
    def test_neither_live_policy_reads_a_file_to_construct(self) -> None:
        """FAILS ON THE UNFIXED CODE (`.handoff/DECISIONS.md` section 3.5).

        Before P2 both policies opened `artifacts/tiny-policy-pure.json` in
        their constructor, so a missing or corrupt artifact killed
        ``run_agent.py --standard``/``--aggressive`` at start-up, before a
        single hand -- a live-money path failing on a file that chose no
        actions. On the pre-P2 code this raises from ``PokerPolicy.__init__``.

        ``prewarm()`` is hoisted out of the patched window deliberately:
        `build_policy` calls it, and it only builds in-memory evaluator and
        deck tables, but patching ``open`` around it would be testing the
        warm-up rather than the policy.
        """
        prewarm()
        with patch("builtins.open", side_effect=AssertionError("policy read a file")):
            self.assertIsInstance(build_policy(), PokerPolicy)
            self.assertIsInstance(
                build_policy(aggressive=True), AggressivePokerPolicy
            )

    def test_runner_help_needs_no_credentials(self) -> None:
        environment = os.environ.copy()
        environment.pop("ARENA_CREDENTIALS", None)
        result = subprocess.run(
            [sys.executable, "run_agent.py", "--help"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plain_package_import_does_not_load_torch(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, engine; print('torch' in sys.modules)",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")

    def test_manual_tool_imports_do_not_read_credentials_or_contact_arena(self) -> None:
        environment = os.environ.copy()
        environment.pop("ARENA_CREDENTIALS", None)
        result = subprocess.run(
            [sys.executable, "-c", "import tools.leave, tools.peek; print('ok')"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
