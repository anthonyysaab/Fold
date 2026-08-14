from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from devfun_poker_playground.decision_engine import DecisionEngine
from tools.collect_foreign_play_data import _decision_row
from tools.rebuild_foreign_corpus import (
    Collection,
    archived_agents,
    cached_receipts,
    derive_delta,
)
from tools.reconcile_foreign_raw_data import (
    classify_replay,
    collection_inventory,
    reconcile_collection,
)

SEATS = [
    {
        "seatNumber": 1,
        "agentId": "hero",
        "agentName": "Teacher",
        "status": "Active",
        "stackChips": 94,
        "currentBetChips": 6,
        "totalCommittedChips": 6,
        "holeCards": ["As", "Kd"],
    },
    {
        "seatNumber": 2,
        "agentId": "rival",
        "agentName": "Rival",
        "status": "Active",
        "stackChips": 98,
        "currentBetChips": 2,
        "totalCommittedChips": 2,
        "holeCards": ["Qc", "Jh"],
    },
]


def _action_event(agent_id: str, seat_number: int, sequence: int = 5) -> dict:
    return {
        "sequence": sequence,
        "type": "ActionTaken",
        "street": "Preflop",
        "occurredAt": 1_700_000_001_000,
        "agentId": agent_id,
        "payload": {
            "action": "raise",
            "toAmount": 6,
            "pot": 3,
            "callAmount": 1,
            "seatNumber": seat_number,
            "stackBefore": 99,
            "currentBetBefore": 2,
            "dealerSeatNumber": 1,
            "minRaiseToBefore": 4,
            "actorCurrentBetBefore": 1,
            "allowedActions": {
                "availableActions": ["fold", "call", "raise", "all-in"],
                "canFold": True,
                "canCheck": False,
                "canCall": True,
                "canBet": False,
                "canRaise": True,
                "canAllIn": True,
                "callChips": 1,
                "callToAmount": 2,
                "minRaiseTo": 4,
                "maxCommit": 100,
                "raiseRange": {"min": 4, "max": 100},
                "allInToAmount": 100,
            },
        },
        "snapshot": {
            "street": "Preflop",
            "potChips": 8,
            "currentBet": 6,
            "minRaiseTo": 10,
            "boardCards": [],
            "seats": SEATS,
        },
    }


def _replay(table_id: str, events: list[dict]) -> dict:
    return {
        "table": {
            "id": table_id,
            "tableId": table_id,
            "tableNumber": 1,
            "competitionId": "competition-1",
            "startedAt": 1_700_000_000_000,
            "smallBlindChips": 1,
            "bigBlindChips": 2,
            "winners": [{"agentId": "hero", "handName": "Uncontested"}],
        },
        "events": events,
    }


def _table_summary(table_id: str) -> dict:
    return {
        "id": table_id,
        "status": "Completed",
        "endedAt": "2026-08-12T08:00:00.000Z",
        "winners": [{"agentId": "hero"}],
        "seats": [
            {"agentId": "hero", "agentHandle": "hero-handle", "chipDelta": 5},
            {"agentId": "rival", "agentHandle": "rival-handle", "chipDelta": -5},
        ],
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _rpc(value: object) -> dict:
    return {"result": {"data": {"json": value}}}


class ForeignCorpusFixture:
    """A miniature collection with every gap the real archive contains.

    ``derived`` is in the frozen CSV. ``shared`` is downloaded and derived
    for hero only, so the archived agent's pair on it is still pending.
    ``quiet`` is downloaded but holds no in-scope decision. ``absent`` is
    inventoried with no replay body on disk.
    """

    def __init__(self, root: Path) -> None:
        self.directory = root / "20260812T000000Z_poker-playground_s11_top15"
        self.raw = self.directory / "raw"
        hero_tables = ["derived", "shared", "quiet", "absent"]
        _write_json(
            self.raw / "agents" / "01-hero" / "tables-000000.json",
            _rpc(
                {
                    "tables": [_table_summary(name) for name in hero_tables],
                    "nextCursor": None,
                }
            ),
        )
        _write_json(
            self.raw / "agents" / "02-rival" / "tables-000000.json",
            _rpc(
                {
                    "tables": [_table_summary("shared"), _table_summary("rival-only")],
                    "nextCursor": None,
                }
            ),
        )
        _write_json(
            self.raw / "agents" / "03-sampled" / "receipts.json",
            [{"tableId": "shared", "handId": "shared", "chipDelta": 1}],
        )
        _write_json(
            self.raw / "leaderboard.json",
            _rpc(
                {
                    "agents": [
                        {
                            "id": "hero",
                            "name": "Teacher",
                            "claimed": True,
                            "totalScore": 100,
                            "totalSubmissions": 10,
                        },
                        {
                            "id": "rival",
                            "name": "Rival",
                            "claimed": True,
                            "totalScore": 90,
                            "totalSubmissions": 9,
                        },
                        {
                            "id": "sampled",
                            "name": "Sampled",
                            "claimed": True,
                            "totalScore": 80,
                            "totalSubmissions": 8,
                        },
                    ]
                }
            ),
        )
        _write_json(
            self.raw / "tables" / "derived.json",
            _rpc(_replay("derived", [_action_event("hero", 1)])),
        )
        _write_json(
            self.raw / "tables" / "shared.json",
            _rpc(
                _replay(
                    "shared",
                    [_action_event("hero", 1), _action_event("rival", 2, sequence=6)],
                )
            ),
        )
        _write_json(
            self.raw / "tables" / "quiet.json",
            _rpc(_replay("quiet", [{"sequence": 1, "type": "BlindPosted"}])),
        )
        # Only the archived agent sat here: invisible to the manifest scope,
        # and exactly the season-11 shape.
        _write_json(
            self.raw / "tables" / "rival-only.json",
            _rpc(_replay("rival-only", [_action_event("rival", 2)])),
        )

        engine = DecisionEngine(equity_trials=10, seed=7)
        agent = {
            "rank": 1,
            "id": "hero",
            "name": "Teacher",
            "totalScore": 100,
            "totalSubmissions": 10,
        }
        rows = [
            _decision_row(
                _replay(table_id, [_action_event("hero", 1)]),
                _action_event("hero", 1),
                agent,
                {
                    "handId": table_id,
                    "tableId": table_id,
                    "settledAt": 1_700_000_002_000,
                    "chipDelta": 5,
                    "winnerHandle": "hero-handle",
                },
                engine,
                season_number=11,
            )
            for table_id in ("derived", "shared")
        ]
        # The frozen CSV predates the schema-2 inputs, exactly like the real
        # collection files.
        self.header = [
            name
            for name in rows[0]
            if name
            not in {
                "feature_opponent_range_width",
                "feature_opponent_max_wildness",
                "feature_opponent_max_stickiness",
                "feature_lead_position_unit",
            }
        ]
        csv_path = self.directory / "top15_decisions.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=self.header, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        _write_json(
            self.directory / "manifest.json",
            {
                "schema_version": 1,
                "scope": {
                    "arena_slug": "poker-playground",
                    "season_number": 11,
                    "season_status": "Ended",
                    "history_mode": "all_paginated_tables",
                },
                "counts": {"decision_rows": len(rows)},
                "agents": [agent],
                "files": {
                    "decision_csv": "top15_decisions.csv",
                    "decision_csv_sha256": "unchecked",
                },
            },
        )


class ReconcileTests(unittest.TestCase):
    def test_inventory_separates_paginated_and_sampled_agents(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            fixture = ForeignCorpusFixture(Path(name))
            inventory = collection_inventory(fixture.raw)

            self.assertEqual(
                inventory["table_ids"],
                {"derived", "shared", "quiet", "absent", "rival-only"},
            )
            self.assertEqual(inventory["incomplete_agents"], [])
            by_dir = {entry["agent_dir"]: entry for entry in inventory["agents"]}
            self.assertEqual(by_dir["01-hero"]["pages"], 1)
            self.assertEqual(by_dir["03-sampled"]["pages"], 0)

    def test_replay_without_an_in_scope_decision_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            fixture = ForeignCorpusFixture(Path(name))
            tables = fixture.raw / "tables"

            self.assertEqual(
                classify_replay(tables / "quiet.json", {"hero"}),
                "no_selected_agent_decision",
            )
            self.assertEqual(
                classify_replay(tables / "shared.json", {"hero"}),
                "has_selected_agent_decision",
            )
            self.assertEqual(
                classify_replay(tables / "shared.json", {"stranger"}),
                "no_selected_agent_decision",
            )

    def test_report_separates_manifest_scope_from_archived_agents(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            fixture = ForeignCorpusFixture(Path(name))

            report = reconcile_collection(fixture.directory, deep=True)
            scope = report["agent_scope"]

            self.assertEqual(scope["manifest_agents"], 1)
            self.assertEqual(scope["full_history_outside_manifest"], ["rival"])
            self.assertEqual(scope["sampled_only_outside_manifest"], ["sampled"])
            # A replay only an archived agent played is a narrower scope, not
            # a missing row. Conflating the two is what first read season 11
            # as a broken collection.
            self.assertEqual(
                report["underived_classification"],
                {"no_selected_agent_decision": 1, "decision_by_archive_only_agent": 1},
            )
            self.assertEqual(report["archive_only_derivable_tables"], 1)

    def test_report_counts_every_gap_class(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            fixture = ForeignCorpusFixture(Path(name))
            report = reconcile_collection(fixture.directory, deep=True)

            self.assertEqual(report["gaps"]["inventoried_missing_replay"], 1)
            self.assertEqual(report["gaps"]["downloaded_not_inventoried"], 0)
            self.assertEqual(report["gaps"]["downloaded_not_in_csv"], 2)
            self.assertEqual(report["gaps"]["csv_table_without_replay"], 0)
            self.assertFalse(report["csv"]["sha256_matches"])


class RebuildTests(unittest.TestCase):
    def test_archive_scope_adds_paginated_agents_only(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            fixture = ForeignCorpusFixture(Path(name))
            _, paginated = cached_receipts(fixture.raw)
            added = archived_agents(fixture.raw, {"hero": {}}, paginated)

            self.assertEqual(set(added), {"rival"})
            self.assertEqual(added["rival"]["rank"], 2)

    def test_manifest_scope_ignores_agents_the_manifest_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            fixture = ForeignCorpusFixture(Path(name))
            collection = Collection(fixture.directory)

            self.assertEqual(set(collection.agents), {"hero"})
            self.assertEqual(collection.pending_pairs(), {("hero", "quiet")})

    def test_archive_scope_finds_pairs_on_already_derived_tables(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            fixture = ForeignCorpusFixture(Path(name))
            collection = Collection(fixture.directory, agent_scope="archive")

            self.assertEqual(
                collection.pending_pairs(),
                {("hero", "quiet"), ("rival", "shared"), ("rival", "rival-only")},
            )
            self.assertEqual(collection.missing_replays(), {"absent"})

    def test_delta_shares_the_frozen_header_and_leaves_it_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            fixture = ForeignCorpusFixture(Path(name))
            frozen = fixture.directory / "top15_decisions.csv"
            before = frozen.read_bytes()
            collection = Collection(fixture.directory, agent_scope="archive")

            summary = derive_delta(collection, dry_run=False)

            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["skipped"], 0)
            self.assertEqual(frozen.read_bytes(), before)
            with collection.delta_csv.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(list(reader.fieldnames or ()), fixture.header)
                delta_rows = list(reader)
            self.assertEqual({row["agent_id"] for row in delta_rows}, {"rival"})
            self.assertEqual(
                {row["table_id"] for row in delta_rows}, {"shared", "rival-only"}
            )
            manifest = json.loads(collection.delta_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "delta")
            self.assertEqual(
                [
                    entry["id"]
                    for entry in manifest["agent_scope"]["added_from_archive"]
                ],
                ["rival"],
            )

    def test_delta_rows_never_duplicate_a_frozen_decision(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            fixture = ForeignCorpusFixture(Path(name))
            collection = Collection(fixture.directory, agent_scope="archive")
            derive_delta(collection, dry_run=False)

            def keys(path: Path) -> set[tuple[str, str, str]]:
                with path.open(encoding="utf-8", newline="") as handle:
                    return {
                        (row["agent_id"], row["table_id"], row["event_sequence"])
                        for row in csv.DictReader(handle)
                    }

            frozen_keys = keys(fixture.directory / "top15_decisions.csv")
            delta_keys = keys(collection.delta_csv)
            self.assertTrue(frozen_keys)
            self.assertTrue(delta_keys)
            self.assertEqual(frozen_keys & delta_keys, set())


if __name__ == "__main__":
    unittest.main()
