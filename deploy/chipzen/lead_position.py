"""Standalone player lead-position gauge.

This script does not import or modify the live agent. It converts the table's
chip counts and the hero's seat position into one explainable gradient from
-100 (clearly trailing) to +100 (clearly leading), the same way
``risk_temperature.py`` reduces a decision to one 0-100 reading.

Two chip factors set the base gradient:

* stack rank -- how many opponents the hero covers minus how many cover the
  hero, so "top three or bottom three" is continuous instead of a cutoff.
* stack share -- the hero's chips against an equal share of the table, so a
  bare rank lead on a flat table reads smaller than a dominant one.

Seat position then accentuates the gradient: acting later multiplies the
magnitude up (a leader on the button reads more leading, a trailer on the
button more trailing), acting earlier damps it. The sign always comes from
chips; position only sharpens or blurs the story.

The bluff advisor consumes this reading through its learned
``bluff_density`` and ``lead_density_gain`` parameters.

Example:
    python lead_position.py --hero-stack 800 --opponents 1200 950 600 \
        --position 0.8
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

FACTOR_WEIGHTS = {
    "stack_rank": 0.60,
    "stack_share": 0.40,
}

# How strongly acting last (position 1.0) or first (0.0) accentuates the
# chip gradient: the magnitude multiplier spans [1 - gain, 1 + gain].
POSITION_GAIN = 0.30

MIN_PLAYERS = 2
MAX_PLAYERS = 6

LEADING_BAND = 100.0 / 3.0


@dataclass(frozen=True, slots=True)
class LeadPosition:
    """One lead reading plus the values needed to explain it."""

    lead: float
    band: str
    factor_lead: dict[str, float]
    position_accent: float
    hero_stack: int
    opponent_stacks: tuple[int, ...]
    position: float | None

    def to_dict(self) -> dict[str, object]:
        factors = {
            name: {
                "lead": round(100.0 * value, 1),
                "weight": round(100.0 * FACTOR_WEIGHTS[name], 1),
                "points": round(100.0 * value * FACTOR_WEIGHTS[name], 1),
            }
            for name, value in self.factor_lead.items()
        }
        return {
            "lead": self.lead,
            "band": self.band,
            "inputs": {
                "hero_stack_chips": self.hero_stack,
                "opponent_stack_chips": list(self.opponent_stacks),
                "position": self.position,
            },
            "factors": factors,
            "context": {
                "position_accent": round(self.position_accent, 4),
            },
        }


def _band(lead: float) -> str:
    if lead >= LEADING_BAND:
        return "leading"
    if lead <= -LEADING_BAND:
        return "trailing"
    return "contending"


def _stack(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer chip count")
    if value < 1:
        raise ValueError(f"{name} must be at least 1 chip")
    return value


def measure_lead_position(
    *,
    hero_stack: int,
    opponent_stacks: tuple[int, ...] | list[int],
    position: float | None = None,
) -> LeadPosition:
    """Measure table standing without choosing or changing a poker action.

    ``opponent_stacks`` holds every active opponent's chips. ``position`` is
    the hero's relative acting order from 0.0 (first to act postflop) to 1.0
    (button); ``None`` applies no positional accent.
    """

    hero_stack = _stack("hero_stack", hero_stack)
    stacks = tuple(
        _stack(f"opponent_stacks[{index}]", value)
        for index, value in enumerate(opponent_stacks)
    )
    players = 1 + len(stacks)
    if not MIN_PLAYERS <= players <= MAX_PLAYERS:
        raise ValueError(
            f"active players must be between {MIN_PLAYERS} and {MAX_PLAYERS}"
        )
    if position is not None:
        position = float(position)
        if not math.isfinite(position) or not 0.0 <= position <= 1.0:
            raise ValueError("position must be between 0 and 1")

    covered = sum(stack < hero_stack for stack in stacks)
    covering = sum(stack > hero_stack for stack in stacks)
    stack_rank = (covered - covering) / len(stacks)

    total = hero_stack + sum(stacks)
    fair_share_ratio = hero_stack * players / total
    stack_share = min(1.0, max(-1.0, fair_share_ratio - 1.0))

    factor_lead = {"stack_rank": stack_rank, "stack_share": stack_share}
    base = sum(FACTOR_WEIGHTS[name] * value for name, value in factor_lead.items())

    if position is None:
        accent = 1.0
    else:
        accent = 1.0 + POSITION_GAIN * (2.0 * position - 1.0)
    lead = round(100.0 * min(1.0, max(-1.0, base * accent)), 1)

    return LeadPosition(
        lead=lead,
        band=_band(lead),
        factor_lead=factor_lead,
        position_accent=accent,
        hero_stack=hero_stack,
        opponent_stacks=stacks,
        position=position,
    )


def render(reading: LeadPosition) -> str:
    width = 20
    filled = round((reading.lead + 100.0) / 200.0 * width)
    gauge = "-" * filled + "|" + "-" * (width - filled)
    lines = [
        f"Lead position: {reading.lead:+.1f}/100 ({reading.band.upper()})",
        f"Gauge: trail [{gauge}] lead",
        "Breakdown:",
    ]
    labels = {
        "stack_rank": "stack rank",
        "stack_share": "share of chips",
    }
    for name, value in reading.factor_lead.items():
        weight = FACTOR_WEIGHTS[name]
        lines.append(
            f"  {labels[name]:14} {100 * value:+6.1f} lead x "
            f"{100 * weight:4.0f}% = {100 * value * weight:+5.1f} points"
        )
    lines.extend(
        (
            "Context:",
            f"  position accent   x{reading.position_accent:.2f}",
            f"  hero stack        {reading.hero_stack} chips vs "
            f"{list(reading.opponent_stacks)}",
        )
    )
    return "\n".join(lines)


def self_test() -> None:
    dominant = measure_lead_position(
        hero_stack=3_000,
        opponent_stacks=(500, 500, 500, 500, 500),
        position=1.0,
    )
    assert dominant.lead == 100.0 and dominant.band == "leading"

    flat = measure_lead_position(
        hero_stack=1_000, opponent_stacks=(1_000, 1_000, 1_000, 1_000, 1_000)
    )
    assert flat.lead == 0.0 and flat.band == "contending"

    last_early = measure_lead_position(
        hero_stack=100,
        opponent_stacks=(1_000, 1_000, 1_000, 1_000, 1_000),
        position=0.0,
    )
    last_late = measure_lead_position(
        hero_stack=100,
        opponent_stacks=(1_000, 1_000, 1_000, 1_000, 1_000),
        position=1.0,
    )
    assert last_early.band == "trailing"
    # Position accentuates magnitude and never flips the chip story's sign.
    assert last_late.lead < last_early.lead < 0.0

    bigger = measure_lead_position(hero_stack=1_400, opponent_stacks=(1_000, 800))
    smaller = measure_lead_position(hero_stack=900, opponent_stacks=(1_000, 800))
    assert bigger.lead > smaller.lead

    leader_early = measure_lead_position(
        hero_stack=2_000, opponent_stacks=(900, 700), position=0.0
    )
    leader_late = measure_lead_position(
        hero_stack=2_000, opponent_stacks=(900, 700), position=1.0
    )
    assert leader_late.lead > leader_early.lead > 0.0

    for bad in (
        {"hero_stack": 0, "opponent_stacks": (100,)},
        {"hero_stack": 100, "opponent_stacks": ()},
        {"hero_stack": 100, "opponent_stacks": (100,) * 6},
        {"hero_stack": 100, "opponent_stacks": (100,), "position": 1.5},
    ):
        try:
            measure_lead_position(**bad)
        except ValueError:
            pass
        else:  # pragma: no cover - self-test guard
            raise AssertionError(f"expected ValueError for {bad}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone -100..+100 player lead-position gauge.",
        epilog=(
            "Example: python lead_position.py --hero-stack 800 "
            "--opponents 1200 950 600 --position 0.8"
        ),
    )
    parser.add_argument("--hero-stack", type=int, help="hero chips")
    parser.add_argument(
        "--opponents",
        type=int,
        nargs="+",
        help="each active opponent's chips",
    )
    parser.add_argument(
        "--position",
        type=float,
        help="relative acting order, 0 first to act to 1 on the button",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--self-test", action="store_true", help="run built-in checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        print("self-test: ok")
        return 0

    missing = [
        flag
        for flag, value in (("--hero-stack", args.hero_stack), ("--opponents", args.opponents))
        if value is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))

    try:
        reading = measure_lead_position(
            hero_stack=args.hero_stack,
            opponent_stacks=tuple(args.opponents),
            position=args.position,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(reading.to_dict(), indent=2, sort_keys=True))
    else:
        print(render(reading))
    return 0


__all__ = [
    "FACTOR_WEIGHTS",
    "LeadPosition",
    "measure_lead_position",
    "POSITION_GAIN",
    "render",
]


if __name__ == "__main__":
    raise SystemExit(main())
