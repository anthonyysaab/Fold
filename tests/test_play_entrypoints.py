"""The launch scripts and the policy default must serve what is deployed."""

from __future__ import annotations

import unittest
from pathlib import Path

import live_session

REPO = Path(__file__).resolve().parents[1]


class PolicyDefaultTests(unittest.TestCase):
    """A bare invocation must not serve a different policy than the pointer."""

    def test_explicit_flags_are_mutually_exclusive(self) -> None:
        for pair in (
            ["--learned", "--aggressive"],
            ["--learned", "--standard"],
            ["--aggressive", "--standard"],
        ):
            with self.subTest(pair=pair), self.assertRaises(SystemExit):
                live_session.parse_args(pair)

    def test_aggressive_forces_the_heuristic(self) -> None:
        args = live_session.parse_args(["--aggressive"])

        self.assertFalse(args.learned)
        self.assertFalse(args.standard)
        self.assertFalse(args.policy_defaulted)

    def test_standard_forces_the_standard_heuristic(self) -> None:
        args = live_session.parse_args(["--standard"])

        self.assertFalse(args.learned)
        self.assertTrue(args.standard)

    def test_bare_invocation_follows_the_approved_pointer(self) -> None:
        # Whether an artifact is approved is repository state, so assert the
        # rule rather than one outcome: bare play serves the learned policy
        # exactly when a pointer exists, and says so.
        approved = (live_session.ARTIFACTS_ROOT / "approved.json").exists()

        args = live_session.parse_args([])

        self.assertEqual(args.learned, approved)
        self.assertEqual(args.policy_defaulted, approved)


class LaunchScriptTests(unittest.TestCase):
    def test_play_sh_passes_arguments_through(self) -> None:
        script = (REPO / "play.sh").read_text(encoding="utf-8")

        self.assertIn('live_session.py "$@"', script)
        # No hardcoded policy flag: the script must not pin a policy the
        # approved pointer disagrees with.
        for flag in ("--aggressive", "--standard", "--learned"):
            self.assertNotIn(f"live_session.py {flag}", script)

    def test_play_sh_refuses_without_credentials(self) -> None:
        script = (REPO / "play.sh").read_text(encoding="utf-8")

        self.assertIn(".arena-credentials", script)
        self.assertIn("ARENA_CREDENTIALS", script)

    def test_play_sh_requires_a_modern_python(self) -> None:
        script = (REPO / "play.sh").read_text(encoding="utf-8")

        self.assertIn("version_info >= (3, 11)", script)

    def test_play_cmd_passes_arguments_through(self) -> None:
        script = (REPO / "play.cmd").read_text(encoding="utf-8")

        self.assertIn("live_session.py %*", script)
        for flag in ("--aggressive", "--standard", "--learned"):
            self.assertNotIn(f"live_session.py {flag}", script)

    def test_play_cmd_reports_the_exit_code(self) -> None:
        script = (REPO / "play.cmd").read_text(encoding="utf-8")

        self.assertIn("ERRORLEVEL", script)
        self.assertIn("exit /b", script)


if __name__ == "__main__":
    unittest.main()
