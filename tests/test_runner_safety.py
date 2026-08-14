"""Regression checks for runner and deadline failure behavior."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
import urllib.error
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest import mock

import run_agent
from devfun_poker_playground.decision_engine import (
    DecisionEngine,
    safest_passive_action,
)
from run_agent import (
    confirm_leave,
    is_reconnectable_join_conflict,
    MalformedResponse,
    parse_args,
    request_arena,
    validate_pending_snapshot,
)


class ScriptedArena:
    """request_arena stand-in scripted per endpoint; no network.

    ``pending`` responses are consumed in order with the last one repeating;
    an Exception entry is raised to simulate an unexpected loop crash.
    """

    def __init__(self, *, join, pending, leave, replay=(200, [])):
        self.join = join
        self.pending = list(pending)
        self.leave = leave
        self.replay = replay
        self.calls: list[tuple[str, str]] = []

    def __call__(self, api_key, method, path, body=None):
        self.calls.append((method, path))
        if path.startswith("/api/arena/texas/join"):
            return self.join
        if path.startswith("/api/arena/texas/leave"):
            return self.leave
        if "/replays" in path:
            return self.replay
        if path.startswith("/api/arena/texas/pending-actions"):
            response = self.pending.pop(0) if len(self.pending) > 1 else self.pending[0]
            if isinstance(response, Exception):
                raise response
            return response
        raise AssertionError(f"unexpected Arena path {path}")

    def count(self, suffix: str) -> int:
        return sum(1 for _, path in self.calls if suffix in path)


class RunnerSafetyTests(unittest.TestCase):
    def test_fallback_prefers_check_then_fold_then_call(self) -> None:
        self.assertEqual(safest_passive_action({"check", "fold", "call"}), "check")
        self.assertEqual(safest_passive_action({"fold", "call"}), "fold")
        self.assertEqual(safest_passive_action({"call"}), "call")
        self.assertIsNone(safest_passive_action({"raise"}))

    def test_deadline_path_folds_instead_of_paying_a_call(self) -> None:
        action = DecisionEngine()._deadline_action({}, {}, {"fold", "call"})
        self.assertEqual(action, ("fold", None))

    def test_both_arena_join_conflicts_are_reconnectable(self) -> None:
        self.assertTrue(
            is_reconnectable_join_conflict(409, {"code": "max_concurrent_tables"})
        )
        self.assertTrue(is_reconnectable_join_conflict(409, {"code": "already_queued"}))
        self.assertFalse(
            is_reconnectable_join_conflict(403, {"code": "already_queued"})
        )

    def test_session_length_must_be_positive(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(["competition", "--seconds", "0"])

    def test_telemetry_flag_has_a_safe_local_default(self) -> None:
        self.assertEqual(
            parse_args(["competition", "--telemetry"]).telemetry,
            ".arena-training.jsonl",
        )
        self.assertEqual(
            parse_args(["competition", "--telemetry", "run.jsonl"]).telemetry,
            "run.jsonl",
        )


class ArenaEnvelopeTests(unittest.TestCase):
    def test_undecodable_success_body_is_marked_malformed(self) -> None:
        connection = mock.MagicMock()
        connection.__enter__.return_value = connection
        connection.status = 200
        connection.read.return_value = b"<html>maintenance</html>"
        with mock.patch("urllib.request.urlopen", return_value=connection):
            status, payload = request_arena("key", "GET", "/x")
        self.assertEqual(status, 200)
        self.assertIsInstance(payload, MalformedResponse)
        self.assertIsNone(validate_pending_snapshot(payload))

    def test_http_error_with_non_mapping_body_stays_a_mapping(self) -> None:
        error = urllib.error.HTTPError(
            url="u", code=402, msg="payment", hdrs=None, fp=io.BytesIO(b"[1, 2]")
        )
        with mock.patch("urllib.request.urlopen", side_effect=error):
            status, payload = request_arena("key", "POST", "/x", {})
        self.assertEqual(status, 402)
        self.assertIsInstance(payload, Mapping)

    def test_non_mapping_pending_payloads_are_rejected(self) -> None:
        for payload in ([], "ok", 7, None, MalformedResponse("<html>")):
            self.assertIsNone(validate_pending_snapshot(payload))

    def test_malformed_inner_shapes_are_rejected(self) -> None:
        self.assertIsNone(validate_pending_snapshot({"tables": {"t1": {}}}))
        self.assertIsNone(validate_pending_snapshot({"tables": [{"a": 1}, 5]}))
        self.assertIsNone(validate_pending_snapshot({"participant": []}))
        self.assertIsNone(validate_pending_snapshot({"runner": [1]}))

    def test_valid_envelopes_pass_through(self) -> None:
        snapshot = {
            "participant": {"totalChips": 5},
            "tables": [{"tableId": "t"}],
            "runner": {"canStopSafely": True},
        }
        self.assertIs(validate_pending_snapshot(snapshot), snapshot)
        self.assertIsNotNone(validate_pending_snapshot({}))


class RunnerStopSafetyTests(unittest.TestCase):
    BUSTED = {
        "participant": {"chipState": "busted", "totalChips": 0},
        "runner": {"canStopSafely": True},
    }

    def _run_main(self, arena, argv, extra_patches=()):
        patches = [
            mock.patch("run_agent.request_arena", arena),
            mock.patch("run_agent.load_credentials", return_value=("key", "agent-1")),
            mock.patch("run_agent.time.sleep"),
            *extra_patches,
        ]
        out = StringIO()
        with redirect_stdout(out):
            for patch in patches:
                patch.start()
                self.addCleanup(patch.stop)
            code = run_agent.main(argv)
        return code, out.getvalue()

    def test_repeated_unusable_pending_success_stops_nonzero(self) -> None:
        arena = ScriptedArena(
            join=(200, {"participant": {"totalChips": 400}}),
            pending=[
                (200, MalformedResponse("<html>")),
                (200, ["not-a-mapping"]),
                (200, {"tables": [7]}),
                (200, self.BUSTED),
            ],
            leave=(200, {}),
        )
        code, out = self._run_main(arena, ["comp", "--seconds", "60"])
        self.assertEqual(code, 5)
        self.assertIn("STOP", out)
        leave_index = next(
            index for index, (_, path) in enumerate(arena.calls) if "/leave" in path
        )
        pending_before_leave = sum(
            1 for _, path in arena.calls[:leave_index] if "pending-actions" in path
        )
        self.assertEqual(pending_before_leave, 3)

    def test_one_unusable_pending_response_is_retried_not_fatal(self) -> None:
        arena = ScriptedArena(
            join=(200, {"participant": {"totalChips": 400}}),
            pending=[(200, MalformedResponse("<html>")), (200, self.BUSTED)],
            leave=(200, {}),
        )
        code, out = self._run_main(arena, ["comp", "--seconds", "60"])
        self.assertEqual(code, 0)
        self.assertIn("unusable pending-actions body", out)
        self.assertIn("STOP: busted", out)

    def test_failed_leave_returns_nonzero_with_recovery_command(self) -> None:
        arena = ScriptedArena(
            join=(200, {"participant": {"totalChips": 400}}),
            pending=[(200, self.BUSTED)],
            leave=(503, {"error": "unavailable"}),
        )
        code, out = self._run_main(arena, ["comp", "--seconds", "60"])
        self.assertEqual(code, 6)
        self.assertIn("python -m tools.leave comp", out)
        self.assertEqual(arena.count("/leave"), 2)

    def test_confirmed_leave_keeps_exit_zero_without_retry(self) -> None:
        arena = ScriptedArena(
            join=(200, {"participant": {"totalChips": 400}}),
            pending=[(200, self.BUSTED)],
            leave=(200, {"left": True}),
        )
        code, out = self._run_main(arena, ["comp", "--seconds", "60"])
        self.assertEqual(code, 0)
        self.assertNotIn("WARNING", out)
        self.assertEqual(arena.count("/leave"), 1)

    def test_malformed_leave_body_is_not_a_confirmed_departure(self) -> None:
        responder = mock.Mock(return_value=(200, MalformedResponse("<html>")))
        with (
            mock.patch("run_agent.request_arena", responder),
            redirect_stdout(StringIO()),
        ):
            self.assertFalse(confirm_leave("key", "comp"))
        self.assertEqual(responder.call_count, 2)

    def test_settlement_reconcile_runs_when_an_error_escapes_the_loop(self) -> None:
        arena = ScriptedArena(
            join=(200, {"participant": {"totalChips": 400}}),
            pending=[RuntimeError("loop crash"), (200, self.BUSTED)],
            leave=(200, {}),
            replay=(200, []),
        )
        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = os.path.join(tmp, "telemetry.jsonl")
            with (
                mock.patch("run_agent.request_arena", arena),
                mock.patch(
                    "run_agent.load_credentials", return_value=("key", "agent-1")
                ),
                mock.patch("run_agent.time.sleep"),
                redirect_stdout(StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "loop crash"):
                    run_agent.main(
                        ["comp", "--seconds", "60", "--telemetry", telemetry_path]
                    )
        self.assertEqual(arena.count("/replays"), 1)
        self.assertEqual(arena.count("/leave"), 1)

    def test_settlement_reconcile_failure_cannot_mask_the_exit_reason(self) -> None:
        arena = ScriptedArena(
            join=(200, {"participant": {"totalChips": 400}}),
            pending=[(200, self.BUSTED)],
            leave=(200, {}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = os.path.join(tmp, "telemetry.jsonl")
            code, out = self._run_main(
                arena,
                ["comp", "--seconds", "60", "--telemetry", telemetry_path],
                extra_patches=(
                    mock.patch(
                        "run_agent.reconcile_telemetry",
                        side_effect=RuntimeError("settlement down"),
                    ),
                ),
            )
        self.assertEqual(code, 0)
        self.assertIn("final settlement reconcile failed", out)

    def test_no_settlement_reconcile_without_telemetry(self) -> None:
        arena = ScriptedArena(
            join=(200, {"participant": {"totalChips": 400}}),
            pending=[(200, self.BUSTED)],
            leave=(200, {}),
        )
        self._run_main(arena, ["comp", "--seconds", "60"])
        self.assertEqual(arena.count("/replays"), 0)


if __name__ == "__main__":
    unittest.main()
