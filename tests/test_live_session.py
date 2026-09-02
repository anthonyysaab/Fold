"""Guards for the continuous live-session supervisor.

The money guards matter most: two active Arena competitions charge real funds,
so discovery must only ever accept the free Playground.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import live_session

# Verbatim shapes returned by /api/arena/competition/list-active on
# 2026-08-13, so the guards are tested against real Arena copy.
PLAYGROUND = {
    "id": "cmsg35zvs001hbagh1wdjc1me",
    "name": "[Poker] Playground S13",
    "description": (
        "No-Limit Texas Hold'em poker playground. Agents compete at tables of "
        "2-6 players, managing a bankroll of chips across multiple hands."
    ),
    "seasonNumber": 13,
    "gameType": "TexasHoldem",
}
EVAL_OPEN_PAID = {
    "id": "cmsqbrzz7057tvijtlb0thm79",
    "name": "[Poker] Eval Open S3",
    "description": (
        "Paid no-limit Texas Hold'em benchmark, built with OKX. Agents pay per "
        "run in USD₮0 on X Layer and play a fixed reference bot heads-up, "
        "producing a raw bb/100 leaderboard that pays out from the entry pool."
    ),
    "seasonNumber": 3,
    "gameType": "TexasHoldem",
}
TOURNAMENT_BUYIN = {
    "id": "cmslrboge8c2zmpfmv5adq4pd",
    "name": "[Poker] Tournament S12",
    "description": (
        "No-Limit Texas Hold'em poker tournament. Agents buy in once for a "
        "fixed starting stack and compete across tables of 2-6 players until a "
        "champion remains."
    ),
    "seasonNumber": 12,
    "gameType": "TexasHoldem",
}


def run_supervisor(
    *,
    runner_result,
    list_answers=((200, [PLAYGROUND]),),
    participants=({"chipState": "available", "totalChips": 966},),
    argv=(),
):
    """Drive ``live_session.main`` against scripted Arena answers; no network.

    ``list_answers`` and ``participants`` are consumed in order with the last
    entry repeating, so a test can script a transient failure followed by a
    steady state. Returns ``(exit code, stdout, runner mock, calls)``.
    """

    calls: list[tuple[str, str]] = []
    lists = list(list_answers)
    seats = list(participants)

    def fake_request(_key, method, path, body=None):
        calls.append((method, path))
        if "list-active" in path:
            return lists.pop(0) if len(lists) > 1 else lists[0]
        if "pending-actions" in path:
            seat = seats.pop(0) if len(seats) > 1 else seats[0]
            return 200, {"participant": seat}
        return 200, {}

    out = io.StringIO()
    with (
        tempfile.TemporaryDirectory() as raw,
        # Never let a test write into the repository's real runs/ archive.
        patch.object(live_session, "ARCHIVE_ROOT", Path(raw)),
        patch.object(
            live_session.run_agent, "load_credentials", return_value=("k", "a")
        ),
        patch.object(live_session.run_agent, "request_arena", side_effect=fake_request),
        patch.object(live_session.time, "sleep"),
        patch.object(live_session.run_agent, "main") as runner,
        contextlib.redirect_stdout(out),
    ):
        if isinstance(runner_result, list):
            runner.side_effect = runner_result
        else:
            runner.return_value = runner_result
        code = live_session.main(list(argv))
    return code, out.getvalue(), runner, calls


class CompetitionGuardTests(unittest.TestCase):
    def test_free_playground_is_accepted(self) -> None:
        self.assertTrue(live_session.is_free_playground(PLAYGROUND))

    def test_paid_competitions_are_refused(self) -> None:
        self.assertFalse(live_session.is_free_playground(EVAL_OPEN_PAID))
        self.assertFalse(live_session.is_free_playground(TOURNAMENT_BUYIN))

    def test_other_games_and_names_are_refused(self) -> None:
        self.assertFalse(
            live_session.is_free_playground({**PLAYGROUND, "gameType": "Showdown"})
        )
        self.assertFalse(
            live_session.is_free_playground({**PLAYGROUND, "name": "[Poker] Eval S1"})
        )

    def test_newest_free_season_wins_discovery(self) -> None:
        older = {**PLAYGROUND, "id": "old", "seasonNumber": 12}
        newer = {**PLAYGROUND, "id": "new", "seasonNumber": 14}
        free = sorted(
            [
                item
                for item in (older, EVAL_OPEN_PAID, newer)
                if live_session.is_free_playground(item)
            ],
            key=lambda item: item.get("seasonNumber") or 0,
            reverse=True,
        )
        self.assertEqual([item["id"] for item in free], ["new", "old"])


class BankrollGuardTests(unittest.TestCase):
    def test_busted_bankroll_stops(self) -> None:
        reason = live_session.bankroll_stop_reason(
            {"chipState": "busted", "totalChips": 0}
        )
        self.assertIsNotNone(reason)
        self.assertIn("busted", reason)

    def test_empty_bankroll_stops(self) -> None:
        self.assertIsNotNone(
            live_session.bankroll_stop_reason(
                {"chipState": "available", "totalChips": 0}
            )
        )

    def test_healthy_bankroll_continues(self) -> None:
        self.assertIsNone(
            live_session.bankroll_stop_reason(
                {"chipState": "available", "totalChips": 966}
            )
        )

    def test_unknown_participant_does_not_stop(self) -> None:
        self.assertIsNone(live_session.bankroll_stop_reason(None))


class ExitCodeGuardTests(unittest.TestCase):
    def test_money_and_access_codes_stop(self) -> None:
        self.assertIsNotNone(live_session.exit_code_stop_reason(2))
        self.assertIsNotNone(live_session.exit_code_stop_reason(3))

    def test_success_and_transient_failure_do_not_stop(self) -> None:
        self.assertIsNone(live_session.exit_code_stop_reason(0))
        self.assertIsNone(live_session.exit_code_stop_reason(1))
        self.assertIsNone(live_session.exit_code_stop_reason(4))


class SupervisorLoopTests(unittest.TestCase):
    """The loop is unattended for hours, so its stop paths are pinned here."""

    def _run(self, runner_result, participant=None):
        calls: list[tuple[str, str]] = []
        seat = participant or {"chipState": "available", "totalChips": 966}

        def fake_request(_key, method, path, body=None):
            calls.append((method, path))
            if "list-active" in path:
                return 200, [PLAYGROUND, EVAL_OPEN_PAID]
            if "pending-actions" in path:
                return 200, {"participant": seat}
            return 200, {}

        with (
            tempfile.TemporaryDirectory() as raw,
            # Never let a test write into the repository's real runs/ archive.
            patch.object(live_session, "ARCHIVE_ROOT", Path(raw)),
            patch.object(
                live_session.run_agent, "load_credentials", return_value=("k", "a")
            ),
            patch.object(
                live_session.run_agent, "request_arena", side_effect=fake_request
            ),
            patch.object(live_session.time, "sleep"),
            patch.object(live_session.run_agent, "main") as runner,
        ):
            if isinstance(runner_result, list):
                runner.side_effect = runner_result
            else:
                runner.return_value = runner_result
            live_session.main([])
        return runner, calls

    def test_entry_fee_stops_the_loop_and_leaves_the_table(self) -> None:
        runner, calls = self._run(2)

        runner.assert_called_once()
        self.assertIn(("POST", "/api/arena/texas/leave"), calls)

    def test_busted_bankroll_never_starts_a_session(self) -> None:
        runner, calls = self._run(
            0, participant={"chipState": "busted", "totalChips": 0}
        )

        runner.assert_not_called()
        self.assertNotIn(("POST", "/api/arena/texas/join"), calls)

    def test_finished_sessions_restart_until_interrupted(self) -> None:
        runner, calls = self._run([0, 0, KeyboardInterrupt()])

        self.assertEqual(runner.call_count, 3)
        self.assertIn(("POST", "/api/arena/texas/leave"), calls)

    def test_each_session_targets_the_free_playground_only(self) -> None:
        runner, _ = self._run([0, KeyboardInterrupt()])

        for call in runner.call_args_list:
            self.assertEqual(call.args[0][0], PLAYGROUND["id"])


class DiscoveryRetryTests(unittest.TestCase):
    """Discovery is one HTTP call, so one failure says nothing.

    It used to end the supervisor outright, and end it with exit 0 -- a
    silent success on failure, indistinguishable from a clean shutdown.
    """

    def test_a_transient_list_failure_is_retried(self) -> None:
        answers = [(500, {"error": "bad gateway"}), (200, [PLAYGROUND])]

        def fake_request(_key, _method, _path, body=None):
            return answers.pop(0) if len(answers) > 1 else answers[0]

        with (
            patch.object(
                live_session.run_agent, "request_arena", side_effect=fake_request
            ),
            patch.object(live_session.time, "sleep") as sleep,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            competition, label = live_session.discover_competition_with_retry("k")

        self.assertEqual(competition, PLAYGROUND["id"])
        self.assertEqual(label, PLAYGROUND["name"])
        sleep.assert_called_once_with(live_session.DISCOVERY_BACKOFF_S[0])

    def test_a_momentarily_absent_playground_is_retried(self) -> None:
        answers = [(200, [EVAL_OPEN_PAID]), (200, [EVAL_OPEN_PAID, PLAYGROUND])]

        def fake_request(_key, _method, _path, body=None):
            return answers.pop(0) if len(answers) > 1 else answers[0]

        with (
            patch.object(
                live_session.run_agent, "request_arena", side_effect=fake_request
            ),
            patch.object(live_session.time, "sleep"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            competition, _ = live_session.discover_competition_with_retry("k")

        # A season rollover can make "no free Playground" briefly true; the
        # paid competitions must still never be chosen.
        self.assertEqual(competition, PLAYGROUND["id"])

    def test_discovery_gives_up_only_after_the_whole_backoff(self) -> None:
        with (
            patch.object(
                live_session.run_agent, "request_arena", return_value=(500, {})
            ),
            patch.object(live_session.time, "sleep") as sleep,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            competition, reason = live_session.discover_competition_with_retry("k")

        self.assertIsNone(competition)
        self.assertIn("gave up after", reason)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            list(live_session.DISCOVERY_BACKOFF_S),
        )

    def test_one_failed_discovery_no_longer_ends_the_supervisor(self) -> None:
        code, out, runner, _ = run_supervisor(
            runner_result=[KeyboardInterrupt()],
            list_answers=[(500, {"error": "bad gateway"}), (200, [PLAYGROUND])],
        )

        runner.assert_called_once()
        self.assertEqual(code, live_session.EXIT_OK)
        self.assertIn("competition discovery failed", out)

    def test_a_permanent_discovery_failure_exits_nonzero_and_loudly(self) -> None:
        code, out, runner, _ = run_supervisor(
            runner_result=0,
            list_answers=[(500, {"error": "bad gateway"})],
        )

        runner.assert_not_called()
        self.assertEqual(code, live_session.EXIT_DISCOVERY_FAILED)
        self.assertNotEqual(code, 0)
        self.assertIn("SUPERVISOR FAILED", out)
        self.assertIn("gave up after", out)


class BankrollConfirmationTests(unittest.TestCase):
    """A bankroll stop ends the whole supervisor, so one sample cannot cause it.

    Same false signal the runner hit on 2026-08-14: a ``chipState: busted``
    reading contradicted by the very next poll.
    """

    BUSTED = {"chipState": "busted", "totalChips": 0}
    HEALTHY = {"chipState": "available", "totalChips": 2870}

    def test_an_unconfirmed_busted_reading_starts_the_session_anyway(self) -> None:
        code, out, runner, _ = run_supervisor(
            runner_result=[KeyboardInterrupt()],
            participants=[self.BUSTED, self.HEALTHY],
        )

        runner.assert_called_once()
        self.assertIn("BANKROLL STOP REJECTED", out)
        self.assertEqual(code, live_session.EXIT_OK)

    def test_a_confirmed_bust_still_ends_the_supervisor(self) -> None:
        code, out, runner, _ = run_supervisor(
            runner_result=0,
            participants=[self.BUSTED],
        )

        runner.assert_not_called()
        self.assertIn("bankroll busted", out)
        self.assertIn("rebuy is owner-gated", out)
        # A deliberate money stop is not a failure: exit 0 keeps a service
        # manager from restarting past it.
        self.assertEqual(code, live_session.EXIT_OK)

    def test_the_min_chips_floor_still_stops_when_confirmed(self) -> None:
        code, _, runner, _ = run_supervisor(
            runner_result=0,
            participants=[{"chipState": "available", "totalChips": 40}],
            argv=("--min-chips", "50"),
        )

        runner.assert_not_called()
        self.assertEqual(code, live_session.EXIT_OK)

    def test_confirmation_never_invents_a_stop_from_an_unreadable_poll(self) -> None:
        with (
            patch.object(live_session, "participant_state", return_value=None),
            patch.object(live_session.time, "sleep"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertIsNone(
                live_session.confirm_bankroll_stop("k", "comp", "bankroll busted")
            )


class RetryMapTests(unittest.TestCase):
    """PENDING_EDITS item 14: exits 5 and 6 are retried, on purpose."""

    def test_the_new_exits_are_retried_and_the_reason_is_recorded(self) -> None:
        self.assertIsNone(live_session.exit_code_stop_reason(5))
        self.assertIsNone(live_session.exit_code_stop_reason(6))
        self.assertIn("malformed", live_session.RETRY_NOTES[5])
        self.assertIn("seated", live_session.RETRY_NOTES[6])

    def test_an_unconfirmed_leave_is_retried_after_releasing_the_seat(self) -> None:
        code, out, runner, calls = run_supervisor(
            runner_result=[6, KeyboardInterrupt()]
        )

        self.assertEqual(runner.call_count, 2)
        self.assertIn("could not confirm its leave", out)
        # One recovery leave before the restart, plus the shutdown leave.
        leaves = [call for call in calls if call == ("POST", "/api/arena/texas/leave")]
        self.assertEqual(len(leaves), 2)
        self.assertEqual(code, live_session.EXIT_OK)

    def test_a_malformed_pending_exhaustion_is_retried_with_its_reason(self) -> None:
        _, out, runner, _ = run_supervisor(runner_result=[5, KeyboardInterrupt()])

        self.assertEqual(runner.call_count, 2)
        self.assertIn("session exited 5; retrying", out)
        self.assertIn("bounded retry on malformed HTTP 200s", out)

    def test_consecutive_failed_sessions_exit_nonzero_and_loudly(self) -> None:
        code, out, runner, _ = run_supervisor(runner_result=[1] * 6)

        self.assertEqual(runner.call_count, len(live_session.RESTART_BACKOFF_S) + 1)
        self.assertEqual(code, live_session.EXIT_SESSION_FAILURES)
        self.assertNotEqual(code, 0)
        self.assertIn("SUPERVISOR FAILED", out)

    def test_a_money_stop_still_exits_zero(self) -> None:
        for money_exit in (2, 3):
            with self.subTest(exit=money_exit):
                code, out, _, _ = run_supervisor(runner_result=money_exit)
                self.assertEqual(code, live_session.EXIT_OK)
                self.assertNotIn("SUPERVISOR FAILED", out)


class StopSignalTests(unittest.TestCase):
    """systemd stops with SIGTERM; unhandled, it would skip the table release."""

    def test_sigterm_is_mapped_to_the_shutdown_path(self) -> None:
        import signal as signal_module

        previous = signal_module.getsignal(signal_module.SIGTERM)
        try:
            live_session.install_stop_signals()
            handler = signal_module.getsignal(signal_module.SIGTERM)
            self.assertTrue(callable(handler))
            with self.assertRaises(KeyboardInterrupt):
                handler(signal_module.SIGTERM, None)
        finally:
            signal_module.signal(signal_module.SIGTERM, previous)


class ConsoleShutdownHandlerTests(unittest.TestCase):
    """Window close, logoff, and shutdown are not Python signals on Windows."""

    def test_handler_installs_on_windows_and_keeps_its_callback_alive(self) -> None:
        before = len(live_session._CONSOLE_HANDLER_REFS)
        installed = live_session.install_console_shutdown_handler(lambda: None)

        if not sys.platform.startswith("win"):
            self.assertFalse(installed)
            return
        self.assertTrue(installed)
        # A collected callback would make Windows call freed memory.
        self.assertEqual(len(live_session._CONSOLE_HANDLER_REFS), before + 1)

    def test_a_failing_handler_never_blocks_startup(self) -> None:
        with (
            patch.object(live_session.sys, "platform", "win32"),
            patch.dict("sys.modules", {"ctypes": None}),
        ):
            self.assertFalse(
                live_session.install_console_shutdown_handler(lambda: None)
            )


class DeploymentPurgeTests(unittest.TestCase):
    """Owner policy: a new deployment purges the old agent's archived plays."""

    def test_same_deployment_accumulates_runs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "old-run").mkdir()
            live_session.sync_archive_to_deployment(root, "heuristic-aggressive-v5")

            purged, previous = live_session.sync_archive_to_deployment(
                root, "heuristic-aggressive-v5"
            )

            self.assertEqual((purged, previous), (0, "heuristic-aggressive-v5"))
            self.assertTrue((root / "old-run").exists())

    def test_new_deployment_purges_previous_runs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            live_session.sync_archive_to_deployment(root, "heuristic-aggressive-v5")
            (root / "run-a").mkdir()
            (root / "run-b").mkdir()

            purged, previous = live_session.sync_archive_to_deployment(
                root, "heuristic-aggressive-v6"
            )

            self.assertEqual((purged, previous), (2, "heuristic-aggressive-v5"))
            self.assertFalse((root / "run-a").exists())
            self.assertFalse((root / "run-b").exists())
            marker = root / live_session.DEPLOYMENT_MARKER
            self.assertEqual(
                marker.read_text(encoding="utf-8").strip(), "heuristic-aggressive-v6"
            )

    def test_unstamped_runs_are_adopted_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pre-versioning-run").mkdir()

            purged, previous = live_session.sync_archive_to_deployment(
                root, "heuristic-aggressive-v5"
            )

            self.assertEqual((purged, previous), (0, None))
            self.assertTrue((root / "pre-versioning-run").exists())

    def test_keep_flag_adopts_across_a_version_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            live_session.sync_archive_to_deployment(root, "heuristic-aggressive-v5")
            (root / "run-a").mkdir()

            purged, _ = live_session.sync_archive_to_deployment(
                root, "heuristic-aggressive-v6", keep=True
            )

            self.assertEqual(purged, 0)
            self.assertTrue((root / "run-a").exists())

    def test_identity_names_the_live_policy_version(self) -> None:
        self.assertEqual(
            live_session.policy_identity(False),
            live_session.poker_policy.AggressivePokerPolicy.policy_version,
        )
        self.assertEqual(
            live_session.policy_identity(True),
            live_session.poker_policy.PokerPolicy.policy_version,
        )


class RunArchiveTests(unittest.TestCase):
    """Runs are archived for later study; a lost log is lost evidence."""

    def _run_with_archive(self, tmp: Path, runner_result) -> Path:
        def fake_request(_key, method, path, body=None):
            if "list-active" in path:
                return 200, [PLAYGROUND]
            if "pending-actions" in path:
                return 200, {
                    "participant": {"chipState": "available", "totalChips": 966}
                }
            return 200, {}

        def fake_runner(argv):
            result = runner_result.pop(0)
            if isinstance(result, BaseException):
                raise result
            print("act[1] Flop bet 9 -> ok temp=34.0/warm")
            return result

        with (
            patch.object(live_session, "ARCHIVE_ROOT", tmp),
            patch.object(
                live_session.run_agent, "load_credentials", return_value=("k", "a")
            ),
            patch.object(
                live_session.run_agent, "request_arena", side_effect=fake_request
            ),
            patch.object(live_session.run_agent, "main", side_effect=fake_runner),
        ):
            live_session.main([])
        directories = [child for child in tmp.iterdir() if child.is_dir()]
        self.assertEqual(len(directories), 1)
        return directories[0]

    def test_archive_captures_log_manifest_and_stop_reason(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = self._run_with_archive(Path(raw), [0, KeyboardInterrupt()])

            log_text = (run_dir / "session-001.log").read_text(encoding="utf-8")
            self.assertIn("act[1] Flop bet 9", log_text)

            manifest = json.loads(
                (run_dir / "session-001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["exit_code"], 0)
            self.assertEqual(manifest["competition_id"], PLAYGROUND["id"])
            self.assertEqual(manifest["chips_before"], 966)
            self.assertEqual(manifest["chips_after"], 966)

            # The interrupt landed during session 2, and the archive records
            # that honestly: a manifest with no exit code.
            interrupted = json.loads(
                (run_dir / "session-002.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(interrupted["exit_code"])

            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["sessions"], 2)
            self.assertEqual(run["stop_reason"], "stopped by you")
            self.assertEqual(run["session_seconds"], 21600)

    def test_console_still_receives_teed_output(self) -> None:
        console = io.StringIO()
        with tempfile.TemporaryDirectory() as raw, contextlib.redirect_stdout(console):
            self._run_with_archive(Path(raw), [0, KeyboardInterrupt()])

        self.assertIn("act[1] Flop bet 9", console.getvalue())


class TableReleaseTests(unittest.TestCase):
    def test_table_is_released_exactly_once(self) -> None:
        leaves: list[str] = []

        def fake_request(_key, method, path, body=None):
            if method == "POST" and path.endswith("/texas/leave"):
                leaves.append(path)
            if "list-active" in path:
                return 200, [PLAYGROUND]
            if "pending-actions" in path:
                return 200, {
                    "participant": {"chipState": "available", "totalChips": 966}
                }
            return 200, {}

        with (
            tempfile.TemporaryDirectory() as raw,
            # Never let a test write into the repository's real runs/ archive.
            patch.object(live_session, "ARCHIVE_ROOT", Path(raw)),
            patch.object(
                live_session.run_agent, "load_credentials", return_value=("k", "a")
            ),
            patch.object(
                live_session.run_agent, "request_arena", side_effect=fake_request
            ),
            patch.object(
                live_session.run_agent, "main", side_effect=KeyboardInterrupt()
            ),
        ):
            live_session.main([])

        # The console handler and the finally must not both send it.
        self.assertEqual(len(leaves), 1)


class ExplicitCompetitionGuardTests(unittest.TestCase):
    """``--competition`` must clear the same money guard as discovery.

    Until 2026-09-02 an explicit id skipped discovery outright, and
    ``is_free_playground`` with it, so a mistyped id could seat the agent
    in a paid competition. Every refusal below is a real shape from the
    active list.
    """

    @staticmethod
    def _answer(items):
        def fake_request(_key, _method, _path, body=None):
            return 200, items

        return fake_request

    def test_the_free_playground_is_accepted(self) -> None:
        with patch.object(
            live_session.run_agent,
            "request_arena",
            side_effect=self._answer([EVAL_OPEN_PAID, PLAYGROUND]),
        ):
            ok, label = live_session.verify_explicit_competition(
                "k", PLAYGROUND["id"]
            )
        self.assertTrue(ok)
        self.assertEqual(label, PLAYGROUND["name"])

    def test_a_paid_competition_is_refused(self) -> None:
        for paid in (EVAL_OPEN_PAID, TOURNAMENT_BUYIN):
            with self.subTest(paid=paid["name"]):
                with patch.object(
                    live_session.run_agent,
                    "request_arena",
                    side_effect=self._answer([PLAYGROUND, paid]),
                ):
                    ok, reason = live_session.verify_explicit_competition(
                        "k", paid["id"]
                    )
                self.assertFalse(ok)
                self.assertIn("not the free Playground", reason)

    def test_an_unknown_id_is_refused_rather_than_assumed(self) -> None:
        with patch.object(
            live_session.run_agent,
            "request_arena",
            side_effect=self._answer([PLAYGROUND]),
        ):
            ok, reason = live_session.verify_explicit_competition("k", "no-such-id")
        self.assertFalse(ok)
        self.assertIn("not in the active list", reason)

    def test_an_unreadable_list_refuses_rather_than_passing(self) -> None:
        # Fail-closed: if the guard cannot be evaluated it must not seat.
        def fake_request(_key, _method, _path, body=None):
            return 500, {"error": "bad gateway"}

        with patch.object(
            live_session.run_agent, "request_arena", side_effect=fake_request
        ):
            ok, reason = live_session.verify_explicit_competition(
                "k", PLAYGROUND["id"]
            )
        self.assertFalse(ok)
        self.assertIn("cannot verify", reason)


if __name__ == "__main__":
    unittest.main()
