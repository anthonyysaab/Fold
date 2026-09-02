"""Benchmarks for the v9 harvest hot path (2026-09-02 diagnosis, phase 0).

The harvest profiler (diagnosis section 6) attributes the runtime to two
hotspots: the treys evaluator (~52%: ``_seven`` brute-forces all 21
five-card subsets) and ``copy.deepcopy`` of seats/policies in the
counterfactual replay (~24%: five sites, ~671 top-level copies per tiny
run, each descending through ~75,000 atomic values). This tool is the
instrument that will judge any optimization of either, built BEFORE the
optimizations per the project's measure-the-instrument-first rule.

Two tiers:

``micro``
    Seconds. ``Evaluator.evaluate`` over a fixed seeded set of 7-card
    hands, ``estimate_equity`` at harvest precision (1,000 trials) per
    street, and the deepcopy sites over a constructor-shaped captured
    state (hero policy + the harvest opponent panel). Fast enough to
    iterate against.

``macro``
    The real harvester at benchmark settings (``--hands-scale 0.01
    --harvest-workers 1``, fixed seeds, no profiler) into a fresh
    directory; reports wall time and the corpus sha256. The corpus
    writer already pins gzip mtime to zero and sorts every JSON object,
    so a deterministic harvest is byte-identical across runs -- that
    identity is the oracle: any change that alters one byte is not an
    optimization, it is a different experiment.

``oracle``
    Runs the macro N times and refuses to bless the instrument unless
    every run's corpus hashes identically; reports the wall-time spread,
    which is the resolution the timing measurements are readable to.

Offline, read-only, stdlib-only. No Arena requests, no credentials, no
promotion. The candidate manifest is only LOADED, never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_DEFAULT_CANDIDATE = (
    Path("artifacts") / "candidates" / "candidate-v9-0001a.manifest.json"
)
_MICRO_HANDS = 2_000
_MICRO_EQUITY_STATES_PER_STREET = 12
_MACRO_HANDS_SCALE = 0.01
_MACRO_SEED = 71
_CORPUS_NAME = "bench"


def _bench_root() -> Path:
    return Path(tempfile.gettempdir()) / "fold-harvest-bench"


# ---------------------------------------------------------------------------
# micro
# ---------------------------------------------------------------------------


def _fixed_hands(count: int) -> list[tuple[list[int], list[int]]]:
    """A fixed seeded set of (hole, board) pairs in treys integer form."""

    from engine._vendor.treys import Deck

    rng = random.Random(20260902)
    hands: list[tuple[list[int], list[int]]] = []
    while len(hands) < count:
        deck = Deck.GetFullDeck()
        rng.shuffle(deck)
        hole = deck[:2]
        board = deck[2:7]
        if len({*hole, *board}) == 7:
            hands.append((hole, board))
    assert all(isinstance(card, int) for hand, board in hands for card in hand + board)
    return hands


def _time_block(seconds_target: float, fn) -> tuple[float, float]:
    """(median, min) per call in seconds; adapts the iteration count."""

    first = None
    samples: list[float] = []
    started = time.perf_counter()
    iterations = 1
    while time.perf_counter() - started < seconds_target:
        before = time.perf_counter()
        for _ in range(iterations):
            fn()
        elapsed = time.perf_counter() - before
        samples.append(elapsed / iterations)
        if first is None:
            first = elapsed / iterations
            iterations = max(1, int(seconds_target / first) // 4)
    return statistics.median(samples), min(samples)


def _captured_state(candidate: str) -> dict[str, Any]:
    """The hero policy and opponent panel exactly as the harvester builds them."""

    from engine.decision_engine import SharedEquityCache
    from engine.learned_policy_v9 import load_policy_v9
    from engine.p3_belief_provider import P3BeliefProvider
    from engine.table_simulator import SimSeat
    from tools.build_phase_b_corpus import LegSpec, _build_opponents
    from tools.build_phase_b_corpus_v9 import ContractForcingRecorder

    provider = P3BeliefProvider.from_artifact()
    spec = LegSpec(
        name="bench",
        opponents=("p3-median", "p3-passive", "p3-aggressive", "median-bot"),
        hands=1,
        seed=71,
        session_hands=250,
        candidate=candidate,
        equity_trials=1_000,
        potential_trials=400,
        feature_seed=7,
        counterfactual_rollouts=2,
        accept_threshold=0.35,
        resample_tries=40,
    )
    hero = ContractForcingRecorder(
        load_policy_v9(
            candidate,
            equity_trials=1_000,
            equity_cache=SharedEquityCache(),
            belief_provider=provider,
            potential_trials=400,
            feature_seed=7,
        )
    )
    opponents = _build_opponents(spec)
    seats = [SimSeat(1, "hero", hero, 6_000)] + [
        SimSeat(index + 2, label, agent, 6_000)
        for index, (label, agent) in enumerate(opponents)
    ]
    return {"hero": hero, "seats": seats, "provider": provider}


def micro(candidate: str, seconds: float) -> None:
    """Evaluator + equity + deepcopy-site timings; prints a JSON report."""

    from engine.hand_strength import estimate_equity, _shared_evaluator

    evaluator = _shared_evaluator()
    hands = _fixed_hands(_MICRO_HANDS)

    def eval_all() -> None:
        for hole, board in hands:
            evaluator.evaluate(hole, board)

    median, minimum = _time_block(seconds, eval_all)
    evals_per_s = _MICRO_HANDS / median

    state = _captured_state(candidate)
    seats = state["seats"]
    policy = state["hero"].policy

    import copy

    seats_median, seats_min = _time_block(seconds, lambda: copy.deepcopy(seats))
    policy_median, policy_min = _time_block(seconds, lambda: copy.deepcopy(policy))

    streets = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
    equity_rows: dict[str, dict[str, float]] = {}
    from engine._vendor.treys import Card, Deck

    for street, board_count in streets.items():
        states = []
        for index in range(_MICRO_EQUITY_STATES_PER_STREET):
            deck = Deck.GetFullDeck()
            random.Random(5000 + index).shuffle(deck)
            hero_cards = (Card.int_to_str(deck[0]), Card.int_to_str(deck[1]))
            board_cards = tuple(
                Card.int_to_str(card) for card in deck[2 : 2 + board_count]
            )
            opponents = 2 if street != "preflop" else 3
            states.append((hero_cards, board_cards, opponents))
        timings = []
        for hero_cards, board_cards, opponents in states:
            before = time.perf_counter()
            estimate_equity(
                hero_cards,
                board_cards,
                opponents,
                trials=1_000,
                seed=77,
                top_fraction=1.0,
            )
            timings.append(time.perf_counter() - before)
        equity_rows[street] = {
            "per_estimate_s": round(statistics.median(timings), 4),
            "spread_s": round(max(timings) - min(timings), 4),
        }

    report = {
        "mode": "micro",
        "candidate": candidate,
        "evaluator": {
            "hands": _MICRO_HANDS,
            "median_us_per_eval": round(median * 1e6 / _MICRO_HANDS, 3),
            "min_us_per_eval": round(minimum * 1e6 / _MICRO_HANDS, 3),
            "evals_per_s": round(evals_per_s, 1),
        },
        "estimate_equity_1000_trials": equity_rows,
        "deepcopy": {
            "seats_list_ms": round(seats_median * 1e3, 3),
            "seats_list_min_ms": round(seats_min * 1e3, 3),
            "hero_policy_ms": round(policy_median * 1e3, 3),
            "hero_policy_min_ms": round(policy_min * 1e3, 3),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# macro / oracle
# ---------------------------------------------------------------------------


def _macro_command(candidate: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.build_phase_b_corpus_v9",
        "--candidate",
        candidate,
        "--hands-scale",
        str(_MACRO_HANDS_SCALE),
        "--harvest-workers",
        "1",
        "--seed",
        str(_MACRO_SEED),
        "--corpus-name",
        _CORPUS_NAME,
        "--output-dir",
        str(output_dir),
    ]


def _run_macro(candidate: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        _macro_command(candidate, output_dir),
        capture_output=True,
        text=True,
        timeout=1_800,
        cwd=Path(__file__).resolve().parent.parent,
    )
    wall = time.perf_counter() - started
    corpus = output_dir / f"{_CORPUS_NAME}.phase-b.jsonl.gz"
    if completed.returncode != 0 or not corpus.exists():
        tail = (completed.stdout + completed.stderr)[-2000:]
        raise SystemExit(
            f"harvest failed (exit {completed.returncode}); tail:\n{tail}"
        )
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    summary = json.loads((output_dir / f"{_CORPUS_NAME}.phase-b.summary.json").read_text())
    return {
        "hash": digest,
        "bytes": corpus.stat().st_size,
        "wall_s": round(wall, 1),
        "summary_wall_s": summary.get("wall_seconds"),
        "decisions": summary.get("validation", {}).get("decisions"),
        "branch_rows": summary.get("validation", {}).get("branch_rows"),
    }


def oracle(candidate: str, runs: int, root: Path) -> None:
    """Macro N times; byte-identity is the gate, the spread the resolution."""

    root.mkdir(parents=True, exist_ok=True)
    results = []
    for index in range(runs):
        started = time.perf_counter()
        result = _run_macro(candidate, root / f"run-{index}")
        result["run_wall_including_launch_s"] = round(time.perf_counter() - started, 1)
        results.append(result)
        print(
            f"run {index}: {result['wall_s']}s "
            f"({result['decisions']} decisions, {result['branch_rows']} rows) "
            f"sha256 {result['hash'][:16]}"
        )
    digests = {result["hash"] for result in results}
    walls = [result["wall_s"] for result in results]
    verdict = "PASS" if len(digests) == 1 else "FAIL"
    spread = max(walls) - min(walls)
    print(
        json.dumps(
            {
                "mode": "oracle",
                "runs": runs,
                "byte_identical": len(digests) == 1,
                "verdict": verdict,
                "wall_s": walls,
                "wall_spread_s": round(spread, 1),
                "wall_spread_pct": round(100.0 * spread / statistics.median(walls), 1),
                "hashes": [result["hash"][:16] for result in results],
                "bytes": [result["bytes"] for result in results],
                "root": str(root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if verdict == "FAIL":
        raise SystemExit(1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--candidate", default=str(_DEFAULT_CANDIDATE))

    micro_parser = sub.add_parser("micro", parents=[common], help="hotspot micro timings")
    micro_parser.add_argument("--seconds", type=float, default=1.0)

    oracle_parser = sub.add_parser("oracle", parents=[common], help="macro N times")
    oracle_parser.add_argument("--runs", type=int, default=3)
    oracle_parser.add_argument("--root", default=None)

    args = parser.parse_args(argv)
    if args.mode == "micro":
        micro(args.candidate, args.seconds)
        return 0
    root = Path(args.root).expanduser() if args.root else _bench_root() / "oracle"
    return oracle(args.candidate, args.runs, root)


if __name__ == "__main__":
    raise SystemExit(main())
