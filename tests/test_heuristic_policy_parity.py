"""Decision goldens for the two heuristic policies.

WHAT THIS IS: a REGRESSION PIN, not a fix's evidence (`.handoff/DECISIONS.md`
section 3.5). It PASSES on the code before P2 and after it. That is the whole
point -- P2 deleted the 125-input fixed network from `engine/poker_policy.py`,
and the claim being defended is that the deletion changed no decision. A dial
with an ablation arm (CLAUDE.md section 1.7) can only measure a difference; a
golden hash asserts there is none, bit for bit, which is the stronger statement
when the change is a removal.

Its independent value outlives P2. The heuristic is the noise-floor champion and
the gauntlet's reference opponent (`tools/evaluate_v8.py`), so a silent drift
here invalidates every duel comparison in the record -- including comparisons
already frozen in `artifacts/evaluations/`.

WHY THREE SEAT COUNTS: the branch P2 removed
(``PokerPolicy._equity_family``) was gated on
``len(table["seats"]) in self.table_sizes``. A corpus that never varies the seat
count cannot see it. 2, 4 and 6 seats cover heads-up, short-handed and the
6-max the Arena actually deals.

WHY TWO ARMS: ``no_dice`` pins ``hyper_aggression_chance=0.0``, the idiom
`tests/test_hyper_aggression.py` uses when a decision must be certain.
``live_default`` runs at the shipped ``HYPER_AGGRESSION_CHANCE`` and is the arm
that matters, since it is what plays. Both were captured twice on the unmodified
tree and reproduced bit-identically before either was frozen (CLAUDE.md section
6 rule 2 -- validate the instrument before believing the result).

THE HASHES ARE VALID ONLY FOR THE CONSTANTS BELOW. ``EQUITY_TRIALS`` in
particular is part of the measurement, not a speed knob: change it and every
digest changes, legitimately. Recapture with
``python tests/test_heuristic_policy_parity.py --recapture`` and say in the
commit message why the old numbers stopped applying.
"""

from __future__ import annotations

import hashlib
import json
import unittest

from engine.decision_engine import HYPER_AGGRESSION_CHANCE
from engine.poker_policy import AggressivePokerPolicy, PokerPolicy
from engine.table_simulator import RecordingPolicy, ScriptedAgent, run_sessions

#: Part of the measurement. See the module docstring before touching these.
SEATS = (2, 4, 6)
EQUITY_TRIALS = 12
TARGET_HANDS = 40
STARTING_STACK = 6000
SESSION_SEED = 31

#: Fixed archetypes, ordered so the mix keeps folds, calls and raises all
#: reachable at every seat count -- a digest over an all-fold corpus would be
#: insensitive to exactly the family flips this pin exists to catch.
OPPONENTS = (
    ("median", 0.226, 0.50, 0.0, 6),
    ("station", 0.150, 0.05, 0.0, 4),
    ("shover", 0.000, 0.00, 1.0, 13),
    ("nit", 0.100, 0.80, 0.0, 27),
    ("maniac", 0.700, 0.10, 0.1, 44),
)

#: Captured 2026-09-04 on the pre-P2 tree at `ad7b2b9`, twice, bit-identical.
GOLDENS = {
    "heuristic-standard-v5|no_dice|2seat":
        "06f64e78ef9ff2fb75cd88476a7f9993c0011cbaffcbd981488904448d06b3bd",
    "heuristic-standard-v5|no_dice|4seat":
        "b0f139d5725b8cb68d7ea28ad2624b34a2ac60f32c4ddef8a9b786028c287ffb",
    "heuristic-standard-v5|no_dice|6seat":
        "808d04b85dfb8fc9a66e3cc6b8b16bd3f87e27ddcbc5a9afbdc935e7f852551a",
    "heuristic-standard-v5|live_default|2seat":
        "06f64e78ef9ff2fb75cd88476a7f9993c0011cbaffcbd981488904448d06b3bd",
    "heuristic-standard-v5|live_default|4seat":
        "a2e25fdfcb78f277f3f33324c391fe35f570723cae71db5119223296ba6a8731",
    "heuristic-standard-v5|live_default|6seat":
        "808d04b85dfb8fc9a66e3cc6b8b16bd3f87e27ddcbc5a9afbdc935e7f852551a",
    "heuristic-aggressive-v6|no_dice|2seat":
        "2638e2ecd9a6235a433c3ec8287c83e93213228c4c2cf6011ea31090b20573b5",
    "heuristic-aggressive-v6|no_dice|4seat":
        "e746760b9b7e1fd376a24b9321940730f17df2a445c728bb83a8f2646bc0bcde",
    "heuristic-aggressive-v6|no_dice|6seat":
        "03bfd6d5eba5ef5e65bf609b16c737b53c2059ad7c20272c5a5c649ee1903d9c",
    "heuristic-aggressive-v6|live_default|2seat":
        "110a170cabde84bcd9877afb4c311668c2ae3b9d4983312bce82b292af21cb6b",
    "heuristic-aggressive-v6|live_default|4seat":
        "e746760b9b7e1fd376a24b9321940730f17df2a445c728bb83a8f2646bc0bcde",
    "heuristic-aggressive-v6|live_default|6seat":
        "77e1b54f0fe388ac1ba93e5f993dec2005e2cec4f7bdc57df8770903e0433a0d",
}

#: Expected example counts, carried alongside the digests. A digest mismatch on
#: its own says "something changed"; a digest mismatch with an unchanged count
#: says "the same decisions came out different", which is the interesting case.
EXAMPLE_COUNTS = {
    "heuristic-standard-v5|no_dice|2seat": 79,
    "heuristic-standard-v5|no_dice|4seat": 66,
    "heuristic-standard-v5|no_dice|6seat": 68,
    "heuristic-standard-v5|live_default|2seat": 79,
    "heuristic-standard-v5|live_default|4seat": 64,
    "heuristic-standard-v5|live_default|6seat": 68,
    "heuristic-aggressive-v6|no_dice|2seat": 66,
    "heuristic-aggressive-v6|no_dice|4seat": 66,
    "heuristic-aggressive-v6|no_dice|6seat": 56,
    "heuristic-aggressive-v6|live_default|2seat": 63,
    "heuristic-aggressive-v6|live_default|4seat": 66,
    "heuristic-aggressive-v6|live_default|6seat": 53,
}

_FACTORIES = {
    "heuristic-standard-v5": PokerPolicy,
    "heuristic-aggressive-v6": AggressivePokerPolicy,
}
_ARMS = {"no_dice": 0.0, "live_default": HYPER_AGGRESSION_CHANCE}


def digest_examples(examples) -> str:
    """The `tests/test_harvest_parallelism.py` digest shape, reused verbatim.

    Rounding to 12 places is deliberate: it is below any decision-relevant
    resolution and above the last-bit float noise that would make the hash
    depend on the machine rather than on the policy.
    """
    digest = hashlib.sha256()
    for example in examples:
        digest.update(
            json.dumps(
                {
                    "table_id": example.table_id,
                    "policy_version": example.policy_version,
                    "features": [round(v, 12) for v in example.features],
                    "action_family_index": example.action_family_index,
                    "risk": round(example.submitted_risk_fraction, 12),
                    "purse_bb": round(example.purse_bb, 12),
                    "reward_bb": round(example.reward_bb, 12),
                    "counterfactual": example.counterfactual,
                    "decision_id": getattr(example, "decision_id", None),
                },
                sort_keys=True,
            ).encode()
        )
    return digest.hexdigest()


def run_leg(policy_version: str, arm: str, seats: int) -> tuple[str, int]:
    """One (policy, dice arm, seat count) corpus, digested."""
    hero = _FACTORIES[policy_version](
        equity_trials=EQUITY_TRIALS, hyper_aggression_chance=_ARMS[arm]
    )
    table = [("hero", lambda h=hero: RecordingPolicy(h))]
    for name, aggression, fold_vs_bet, shove, seed in OPPONENTS[: seats - 1]:
        table.append(
            (
                name,
                lambda n=name, a=aggression, f=fold_vs_bet, s=shove, d=seed:
                    ScriptedAgent(n, a, f, s, seed=d),
            )
        )
    result = run_sessions(
        table,
        target_hands=TARGET_HANDS,
        seed=SESSION_SEED,
        starting_stack=STARTING_STACK,
        collect_examples=True,
    )
    return digest_examples(result.examples), len(result.examples)


class HeuristicPolicyParityTests(unittest.TestCase):
    """Every heuristic decision is byte-identical to the pre-P2 capture."""

    def test_decisions_match_the_frozen_goldens(self) -> None:
        for key, expected in sorted(GOLDENS.items()):
            policy_version, arm, seat_label = key.split("|")
            seats = int(seat_label.removesuffix("seat"))
            with self.subTest(policy=policy_version, arm=arm, seats=seats):
                actual, count = run_leg(policy_version, arm, seats)
                self.assertEqual(
                    count,
                    EXAMPLE_COUNTS[key],
                    f"{key}: the corpus itself changed shape, so the digest "
                    f"below cannot be compared meaningfully",
                )
                self.assertEqual(
                    actual,
                    expected,
                    f"{key}: a heuristic decision changed. This policy is the "
                    f"gauntlet's reference opponent and the noise-floor "
                    f"champion; a drift here invalidates frozen duel numbers. "
                    f"Do not update the golden without saying what changed.",
                )

    def test_the_two_dice_arms_are_not_secretly_the_same_measurement(self) -> None:
        """Guard the instrument, not the engine (CLAUDE.md section 6 rule 2).

        At the shipped 2% chance the dice may legitimately never fire in a
        short corpus, and several legs do collide. If EVERY leg collided the
        `live_default` arm would be measuring nothing, and the collision would
        be invisible -- the suite would stay green while half the pin quietly
        stopped existing.
        """
        collisions = [
            seats
            for policy in _FACTORIES
            for seats in SEATS
            if GOLDENS[f"{policy}|no_dice|{seats}seat"]
            == GOLDENS[f"{policy}|live_default|{seats}seat"]
        ]
        self.assertLess(
            len(collisions),
            len(_FACTORIES) * len(SEATS),
            "every live_default leg equals its no_dice twin, so the "
            "live_default arm is not exercising the anti-modeling roll at all",
        )


def _recapture() -> int:
    """Reprint the tables above. Never called by the suite."""
    goldens, counts = {}, {}
    for policy_version in _FACTORIES:
        for arm in _ARMS:
            for seats in SEATS:
                key = f"{policy_version}|{arm}|{seats}seat"
                goldens[key], counts[key] = run_leg(policy_version, arm, seats)
                print(f"{key:52s} {goldens[key][:16]}  n={counts[key]}")
    print(json.dumps({"GOLDENS": goldens, "EXAMPLE_COUNTS": counts}, indent=4))
    return 0


if __name__ == "__main__":
    import sys

    if "--recapture" in sys.argv:
        raise SystemExit(_recapture())
    unittest.main()
