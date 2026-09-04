from __future__ import annotations

import argparse
import copy
import hashlib
import json
import unittest

from engine.poker_policy import build_policy
from tools.self_play_cycle import (
    _harvest_workers,
    _LegSpec,
    harvest,
    harvest_specs,
)


def _args(**overrides) -> argparse.Namespace:
    values = {
        "lineup_hands": 6,
        "shover_hands": 6,
        "station_hands": 0,
        "nit_hands": 0,
        "textured_hands": 0,
        "sparring": None,
        "sparring_hands": 0,
        "seed": 71,
        "equity_trials": 4,
        "starting_stack": 600,
        "on_policy": None,
        "counterfactual_rollouts": 2,
        "harvest_workers": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _digest(examples: list) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(
            json.dumps(
                {
                    "table_id": example.table_id,
                    "policy_version": example.policy_version,
                    "features": [round(value, 12) for value in example.features],
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


class HarvestWorkerCountTests(unittest.TestCase):
    def test_one_worker_stays_sequential(self) -> None:
        self.assertEqual(_harvest_workers(1, 5), 1)

    def test_single_leg_never_spawns_a_pool(self) -> None:
        self.assertEqual(_harvest_workers(0, 1), 1)

    def test_explicit_count_never_exceeds_the_leg_count(self) -> None:
        self.assertEqual(_harvest_workers(9, 3), 3)

    def test_auto_stays_within_the_leg_count(self) -> None:
        self.assertLessEqual(_harvest_workers(0, 4), 4)
        self.assertGreaterEqual(_harvest_workers(0, 4), 1)


class HarvestSpecTests(unittest.TestCase):
    def test_specs_keep_their_fixed_order_and_seeds(self) -> None:
        specs = harvest_specs(
            _args(
                station_hands=4,
                nit_hands=4,
                textured_hands=4,
                sparring="x.json",
                sparring_hands=4,
            )
        )

        self.assertEqual(
            [spec.name for spec in specs],
            [
                "five-max lineup",
                "heads-up vs shover",
                "heads-up vs station",
                "heads-up vs nit",
                "heads-up vs textured",
                "sparring",
            ],
        )
        self.assertEqual([spec.seed for spec in specs], [71, 91, 92, 93, 94, 101])
        # Distinct leg seeds are what make parallel and sequential harvests
        # produce identical examples; a collision would silently correlate legs.
        self.assertEqual(len({spec.seed for spec in specs}), len(specs))

    def test_specs_are_picklable_by_value(self) -> None:
        # Worker processes rebuild opponents from the spec; shipping live
        # policy objects instead is what makes leg parallelism expensive.
        spec = harvest_specs(_args())[0]
        self.assertEqual(copy.deepcopy(spec), spec)
        self.assertIsInstance(spec, _LegSpec)

    def test_archetype_specs_rebuild_their_opponent(self) -> None:
        shover = harvest_specs(_args())[1]
        opponents = shover.opponents()

        self.assertEqual(len(opponents), 1)
        label, agent = opponents[0]
        self.assertEqual(label, "shover")
        self.assertEqual(agent.shove_rate, 1.0)
        self.assertEqual(agent.seed, 81)

    def test_lineup_spec_rebuilds_the_calibrated_table(self) -> None:
        lineup = harvest_specs(_args())[0]
        self.assertEqual(len(lineup.opponents()), 4)


class HarvestEquivalenceTests(unittest.TestCase):
    def test_parallel_harvest_matches_sequential_exactly(self) -> None:
        sequential = harvest(_args(harvest_workers=1))
        parallel = harvest(_args(harvest_workers=2))

        self.assertTrue(sequential)
        self.assertEqual(len(parallel), len(sequential))
        self.assertEqual(_digest(parallel), _digest(sequential))


class PolicyDeepCopyTests(unittest.TestCase):
    """What counterfactual replay must and must not share across a deep copy.

    This class used to also pin ``FixedPolicyNetwork.__deepcopy__``, a
    weight-sharing opt-out that existed because copying the retired 125-input
    network's ~8,300 immutable floats per rollout dominated harvest time. P2
    deleted the network on 2026-09-04, so there is nothing left to share and
    the optimization is not "missing" -- do not reintroduce a ``__deepcopy__``
    here hunting for it.

    The tracker assertion below is unrelated to that and still load-bearing:
    it pins ``DecisionEngine.__deepcopy__``, which must give each replay its
    OWN opponent tracker or rollouts would contaminate each other's beliefs.
    """

    def test_each_deep_copy_gets_its_own_opponent_tracker(self) -> None:
        policy = build_policy(aggressive=True, equity_trials=4)
        clone = copy.deepcopy(policy)

        self.assertIsNot(clone.opponent_tracker, policy.opponent_tracker)


if __name__ == "__main__":
    unittest.main()
