from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("pokerkit")

from poker_nn_training.data import (
    FEATURE_NAMES,
    LABELS,
    InvalidHandHistory,
    extract_decisions,
    split_for_row,
)


def _heads_up_row() -> dict[str, object]:
    return {
        "variant": "NT",
        "ante_trimming_status": False,
        "antes": [0, 0],
        "blinds_or_straddles": [1, 2],
        "min_bet": 2,
        "starting_stacks": [100, 100],
        "source_file": "cash/site/day/table.phhs",
        "actions": [
            "d dh p1 AsKd",
            "d dh p2 QhQc",
            "p2 cbr 6",
            "p1 cc",
            "d db 2c3d4h",
            "p1 cc",
            "p2 cbr 10",
            "p1 f",
        ],
    }


def _feature(example_features: tuple[float, ...], name: str) -> float:
    return example_features[FEATURE_NAMES.index(name)]


def test_replays_valid_heads_up_phh_and_captures_pre_action_features() -> None:
    examples = extract_decisions(_heads_up_row())

    assert [LABELS[example.label] for example in examples] == [
        "aggress",
        "check_call",
        "check_call",
        "aggress",
        "fold",
    ]
    assert [example.actor_index for example in examples] == [1, 0, 0, 1, 0]
    assert all(len(example.features) == len(FEATURE_NAMES) for example in examples)

    first = examples[0].features
    assert _feature(first, "hole_Qc") == 1
    assert _feature(first, "hole_Qh") == 1
    assert sum(first[52:104]) == 0
    assert _feature(first, "street_preflop") == 1
    assert _feature(first, "legal_aggress") == 1

    first_flop = examples[2].features
    assert _feature(first_flop, "board_2c") == 1
    assert _feature(first_flop, "board_3d") == 1
    assert _feature(first_flop, "board_4h") == 1
    assert _feature(first_flop, "street_flop") == 1


def test_unknown_actor_holes_are_skipped_by_default_or_explicitly_encoded() -> None:
    row = _heads_up_row()
    actions = list(row["actions"])
    actions[0] = "d dh p1 ????"
    row["actions"] = actions

    known_only = extract_decisions(row)
    all_examples = extract_decisions(row, require_known_hole_cards=False)

    assert [example.actor_index for example in known_only] == [1, 1]
    assert len(all_examples) == 5
    unknown = next(example for example in all_examples if example.actor_index == 0)
    assert sum(unknown.features[:52]) == 0
    assert _feature(unknown.features, "hole_known_fraction") == 0


def test_opponent_hole_cards_never_change_actor_features() -> None:
    first_row = _heads_up_row()
    second_row = deepcopy(first_row)
    actions = list(second_row["actions"])
    actions[0] = "d dh p1 9s8s"
    second_row["actions"] = actions

    first_actor = [
        example.features for example in extract_decisions(first_row) if example.actor_index == 1
    ]
    second_actor = [
        example.features for example in extract_decisions(second_row) if example.actor_index == 1
    ]

    assert first_actor == second_actor


def test_any_late_malformed_action_rejects_the_whole_hand() -> None:
    row = _heads_up_row()
    actions = list(row["actions"])
    actions[-1] = "p2 f"
    row["actions"] = actions

    with pytest.raises(InvalidHandHistory, match="does not match PokerKit actor"):
        extract_decisions(row)


def test_source_group_split_is_stable_and_keeps_pluribus_matches_together() -> None:
    same_file = {"source_file": "handhq/site/a.phhs"}
    assert split_for_row(same_file) == split_for_row(same_file)
    assert split_for_row(same_file) in {"train", "val", "test"}

    pluribus_a = {"source_file": "pluribus/98/hand-001.phh"}
    pluribus_b = {"source_file": "pluribus/98/hand-999.phh"}
    for seed in range(20):
        assert split_for_row(pluribus_a, seed) == split_for_row(pluribus_b, seed)

    other_file = {"source_file": "handhq/site/b.phhs"}
    assert any(
        split_for_row(same_file, seed) != split_for_row(other_file, seed)
        for seed in range(100)
    )
