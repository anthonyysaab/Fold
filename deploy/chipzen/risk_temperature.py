"""Standalone poker risk-temperature gauge.

This script does not import or modify the live agent. It converts five situation
inputs into an explainable 0-100 risk temperature:

* hand strength
* purse size
* bet size relative to that purse
* distance from the river
* active player count

Example:
    python risk_temperature.py --hand-strength 40 --purse 973 --bet 20 \
        --street preflop --players 6
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


FACTOR_WEIGHTS = {
    "hand_strength": 0.40,
    "bet_pressure": 0.30,
    "distance_from_river": 0.15,
    "players": 0.15,
}

STREET_DISTANCE = {
    "preflop": 1.0,
    "flop": 2.0 / 3.0,
    "turn": 1.0 / 3.0,
    "river": 0.0,
}

MIN_PLAYERS = 2
MAX_PLAYERS = 6


@dataclass(frozen=True, slots=True)
class RiskTemperature:
    """One risk reading plus the values needed to explain it."""

    temperature: float
    band: str
    factor_risk: dict[str, float]
    hand_strength: float
    purse: float
    bet: float
    street: str
    players: int
    bet_fraction: float

    def to_dict(self) -> dict[str, object]:
        factors = {
            name: {
                "risk": round(100.0 * risk, 1),
                "weight": round(100.0 * FACTOR_WEIGHTS[name], 1),
                "points": round(100.0 * risk * FACTOR_WEIGHTS[name], 1),
            }
            for name, risk in self.factor_risk.items()
        }
        return {
            "temperature": self.temperature,
            "band": self.band,
            "inputs": {
                "hand_strength": self.hand_strength,
                "purse_chips": self.purse,
                "bet_chips": self.bet,
                "street": self.street,
                "active_players": self.players,
            },
            "factors": factors,
            "context": {
                "bet_fraction_of_purse": round(self.bet_fraction, 4),
            },
        }


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _band(temperature: float) -> str:
    if temperature < 25.0:
        return "cold"
    if temperature < 50.0:
        return "warm"
    if temperature < 75.0:
        return "hot"
    return "critical"


def measure_risk_temperature(
    *,
    hand_strength: float,
    purse: float,
    bet: float,
    street: str,
    players: int,
) -> RiskTemperature:
    """Measure situational risk without choosing or changing a poker action.

    ``hand_strength`` is a percentage from 0 to 100. ``bet`` is the number of
    chips at risk in the current decision, not the total pot. ``players``
    includes the hero and every active opponent still in the hand.
    """

    hand_strength = _finite("hand_strength", hand_strength)
    purse = _finite("purse", purse)
    bet = _finite("bet", bet)
    street = str(street).casefold()

    if not 0.0 <= hand_strength <= 100.0:
        raise ValueError("hand_strength must be between 0 and 100")
    if purse <= 0.0:
        raise ValueError("purse must be greater than zero")
    if not 0.0 <= bet <= purse:
        raise ValueError("bet must be between zero and the purse size")
    if street not in STREET_DISTANCE:
        choices = ", ".join(STREET_DISTANCE)
        raise ValueError(f"street must be one of: {choices}")
    if isinstance(players, bool) or not isinstance(players, int):
        raise ValueError("players must be an integer")
    if not MIN_PLAYERS <= players <= MAX_PLAYERS:
        raise ValueError(f"players must be between {MIN_PLAYERS} and {MAX_PLAYERS}")

    bet_fraction = bet / purse

    factor_risk = {
        "hand_strength": 1.0 - hand_strength / 100.0,
        "bet_pressure": bet_fraction,
        "distance_from_river": STREET_DISTANCE[street],
        "players": (players - MIN_PLAYERS) / (MAX_PLAYERS - MIN_PLAYERS),
    }
    temperature = round(
        100.0 * sum(FACTOR_WEIGHTS[name] * risk for name, risk in factor_risk.items()),
        1,
    )
    return RiskTemperature(
        temperature=temperature,
        band=_band(temperature),
        factor_risk=factor_risk,
        hand_strength=hand_strength,
        purse=purse,
        bet=bet,
        street=street,
        players=players,
        bet_fraction=bet_fraction,
    )


def render(reading: RiskTemperature) -> str:
    width = 20
    filled = round(reading.temperature / 100.0 * width)
    gauge = "#" * filled + "." * (width - filled)
    lines = [
        f"Risk temperature: {reading.temperature:.1f}/100 ({reading.band.upper()})",
        f"Gauge: [{gauge}]",
        "Breakdown:",
    ]
    labels = {
        "hand_strength": "weak hand",
        "bet_pressure": "bet / purse",
        "distance_from_river": "distance to river",
        "players": "active players",
    }
    for name, risk in reading.factor_risk.items():
        weight = FACTOR_WEIGHTS[name]
        lines.append(
            f"  {labels[name]:18} {100 * risk:5.1f} risk x "
            f"{100 * weight:4.0f}% = {100 * risk * weight:4.1f} points"
        )
    lines.extend(
        (
            "Context:",
            f"  purse             {reading.purse:g} chips",
            f"  current bet       {reading.bet:g} chips "
            f"({100 * reading.bet_fraction:.1f}% of purse)",
        )
    )
    return "\n".join(lines)


def self_test() -> None:
    safest = measure_risk_temperature(
        hand_strength=100,
        purse=1_000,
        bet=0,
        street="river",
        players=2,
    )
    riskiest = measure_risk_temperature(
        hand_strength=0,
        purse=10,
        bet=10,
        street="preflop",
        players=6,
    )
    threshold = measure_risk_temperature(
        hand_strength=40, purse=1_000, bet=0, street="preflop", players=6
    )
    baseline = measure_risk_temperature(
        hand_strength=60, purse=1_000, bet=100, street="flop", players=4
    )
    short_purse = measure_risk_temperature(
        hand_strength=60, purse=100, bet=100, street="flop", players=4
    )

    assert safest.temperature == 0.0 and safest.band == "cold"
    assert riskiest.temperature == 100.0 and riskiest.band == "critical"
    assert threshold.temperature == 54.0 and threshold.band == "hot"
    assert baseline.temperature == 36.5 and baseline.band == "warm"
    assert short_purse.temperature == 63.5 and short_purse.band == "hot"
    assert baseline.temperature < short_purse.temperature


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone 0-100 poker risk-temperature gauge.",
        epilog=(
            "Example: python risk_temperature.py --hand-strength 40 "
            "--purse 973 --bet 20 --street preflop --players 6"
        ),
    )
    parser.add_argument(
        "--hand-strength", type=float, help="hand strength percentage, 0-100"
    )
    parser.add_argument("--purse", type=float, help="current stack in chips")
    parser.add_argument(
        "--bet",
        type=float,
        help="new chips put at risk by the current decision",
    )
    parser.add_argument(
        "--street", choices=tuple(STREET_DISTANCE), help="current street"
    )
    parser.add_argument(
        "--players", type=int, help="active players including the hero, 2-6"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    parser.add_argument("--self-test", action="store_true", help="run built-in checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        print("self-test: ok")
        return 0

    required = ("hand_strength", "purse", "bet", "street", "players")
    missing = [
        f"--{name.replace('_', '-')}"
        for name in required
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))

    try:
        reading = measure_risk_temperature(
            hand_strength=args.hand_strength,
            purse=args.purse,
            bet=args.bet,
            street=args.street,
            players=args.players,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(reading.to_dict(), indent=2, sort_keys=True))
    else:
        print(render(reading))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
