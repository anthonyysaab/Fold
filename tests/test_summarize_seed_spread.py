"""Controls for the three-seed spread summarizer.

The summarizer turns three finished gauntlets into one spread number that a
promotion argument may lean on, so it is validated the same way every other
probe in this repo is: against outcomes that are impossible by construction
if the code is right, and impossible to fake if it is wrong.

  IDENTITY  the same gauntlet relabelled as three init seeds must produce a
            spread of exactly 0.00 and pairwise contrasts of exactly 0.0.
            A summarizer that manufactures spread out of identical inputs
            fails here and nowhere else.
  SENSITIVE two genuinely different gauntlets must produce the arithmetic
            max-minus-min of their margins -- so IDENTITY passing is not
            just the code returning zero for everything.
  VERDICT   the verdict ladder is exercised at both sides of the 16.78
            reference, including the boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.summarize_seed_spread import (
    KNOWN_SEED_SPREAD_BB_PER_100,
    CHANNELS,
    main,
    paired_stats,
)


def _gauntlet(margin: float, diffs: list[float], *, broken: int = 6) -> dict:
    """A minimal report with the fields the summarizer reads."""

    channels = {}
    for index, name in enumerate(CHANNELS):
        # First ``broken`` channels are worse than the champion beyond MDE.
        diff = -100.0 if index < broken else 1.0
        channels[name] = {
            "bb_per_100": 10.0,
            "champion_mean_bb_per_100": 110.0,
            "paired_vs_champion": {"mean": diff, "sd": 5.0, "t": -3.0},
            "published_mde_bb_per_100": 20.0,
        }
    return {
        "incumbent": "candidate-v7-0001c",
        "config": {"candidate_manifest": "artifacts/candidates/x.manifest.json"},
        "trainer_context": {"init_seed_exported": 1},
        "nullcheck": {"mirror_exact": True},
        "duel": {
            "empirical_mde_bb_per_100": 8.0,
            "seeds": len(diffs),
            "verdict": "UNRESOLVED",
            "report": {
                "hands_per_seed": 2000,
                "paired": {
                    "mean": margin,
                    "sd": 17.0,
                    "se": 4.3,
                    "t": 2.2,
                    "diffs": diffs,
                },
            },
        },
        "battery_comparisons": {"channels": channels},
    }


def _run(tmp_path: Path, gauntlets: dict[int, dict]) -> dict:
    specs = []
    for seed, payload in gauntlets.items():
        path = tmp_path / f"g{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        specs.extend(["--gauntlet", f"{seed}={path}"])
    output = tmp_path / "spread.json"
    assert main([*specs, "--output", str(output)]) == 0
    return json.loads(output.read_text(encoding="utf-8"))


DIFFS_A = [1.0, -2.0, 3.0, 10.0, -4.0, 6.0, 0.0, 5.0]


def test_identity_inputs_give_exactly_zero_spread(tmp_path: Path) -> None:
    """Impossible by construction: identical runs cannot differ."""

    same = _gauntlet(9.77, DIFFS_A)
    report = _run(tmp_path, {101: same, 202: dict(same), 303: dict(same)})

    assert report["spread"]["max_minus_min_bb_per_100"] == 0.0
    assert report["spread"]["exceeds_reference_spread"] is False
    for contrast in report["paired_cross_seed_contrasts"].values():
        assert contrast["mean"] == 0.0
        assert contrast["sd"] == 0.0
    # And the battery counts must agree seed for seed.
    counts = {tuple(v.values()) for v in report["battery_summary"].values()}
    assert len(counts) == 1


def test_spread_is_the_arithmetic_range_of_the_margins(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        {
            101: _gauntlet(-7.06, DIFFS_A),
            202: _gauntlet(-1.88, DIFFS_A),
            303: _gauntlet(9.72, DIFFS_A),
        },
    )
    spread = report["spread"]
    # The v7-0002 numbers reproduce their published 16.78 spread exactly.
    assert spread["max_minus_min_bb_per_100"] == pytest.approx(16.78)
    assert spread["best_seed"] == 303
    assert spread["worst_seed"] == 101
    assert spread["exceeds_reference_spread"] is True
    assert spread["verdict"] == "SEED-DEPENDENT"
    assert spread["sign_agreement_across_seeds"] is False


def test_small_spread_without_sign_agreement_is_unresolved(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        {
            101: _gauntlet(1.0, DIFFS_A),
            202: _gauntlet(-1.0, DIFFS_A),
            303: _gauntlet(0.5, DIFFS_A),
        },
    )
    assert report["spread"]["max_minus_min_bb_per_100"] == pytest.approx(2.0)
    assert report["spread"]["verdict"] == "UNRESOLVED"


def test_seed_robust_requires_sign_agreement_and_clearing_each_mde(
    tmp_path: Path,
) -> None:
    report = _run(
        tmp_path,
        {
            101: _gauntlet(20.0, DIFFS_A),
            202: _gauntlet(24.0, DIFFS_A),
            303: _gauntlet(22.0, DIFFS_A),
        },
    )
    # Spread 4.0 < 16.78, all positive, all above the 8.0 empirical MDE.
    assert report["spread"]["verdict"] == "SEED-ROBUST"


def test_a_single_seed_can_never_be_called_seed_robust(tmp_path: Path) -> None:
    """A spread of zero over one artifact is a sample of one, not robustness."""

    report = _run(tmp_path, {202: _gauntlet(9.77, DIFFS_A)})
    assert report["spread"]["max_minus_min_bb_per_100"] == 0.0
    assert report["spread"]["verdict"] == "UNRESOLVED"

    # Two seeds are still short of the design's three.
    two = _run(tmp_path, {101: _gauntlet(20.0, DIFFS_A), 202: _gauntlet(22.0, DIFFS_A)})
    assert two["spread"]["verdict"] == "UNRESOLVED"


def test_spread_at_the_reference_boundary_is_seed_dependent(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        {
            101: _gauntlet(0.0, DIFFS_A),
            202: _gauntlet(KNOWN_SEED_SPREAD_BB_PER_100, DIFFS_A),
        },
    )
    assert report["spread"]["verdict"] == "SEED-DEPENDENT"


def test_battery_counts_split_held_from_broken(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        {
            101: _gauntlet(1.0, DIFFS_A, broken=6),
            202: _gauntlet(1.0, DIFFS_A, broken=0),
        },
    )
    assert report["battery_summary"]["101"] == {"held": 0, "broken": 6}
    assert report["battery_summary"]["202"] == {"held": 6, "broken": 0}


def test_paired_stats_matches_hand_computation() -> None:
    stats = paired_stats([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["sd"] == pytest.approx(2.14, abs=0.01)  # n-1 denominator
    assert stats["n"] == 8


def test_paired_contrast_of_a_series_against_itself_is_zero() -> None:
    stats = paired_stats([0.0] * 16)
    assert stats["mean"] == 0.0
    assert stats["sd"] == 0.0
    assert stats["t"] is None
