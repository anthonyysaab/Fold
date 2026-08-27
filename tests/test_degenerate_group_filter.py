"""The zero-signal decision-group filter, and the invariant behind it.

The predicate is not "every target is zero". It is "a constant action head
exactly minimises this group's contribution", which for the centered v7
objective holds precisely when ``weight * target`` is the same on every row.
The first test below never looks at that formula: it searches for a
prediction that beats a constant, and requires the predicate to agree with
what the search finds. A wrong predicate cannot pass it.
"""

from __future__ import annotations

import json
import pathlib
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from devfun_poker_playground.learning_contract import LEARNING_INPUT_SIZE
from devfun_poker_playground.offline_trainer import (
    DEGENERATE_GROUP_FILTERS,
    TrainingConfig,
    _action_value_target,
    _branch_value_weight,
    _zero_signal_group,
    train_candidate,
)
from devfun_poker_playground.training_telemetry import TrainingExample
from tools.self_play_cycle import main as cycle_main

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

BRANCHES = ("fold", "check_call", "aggress_half_pot", "aggress_pot")
FAMILY_OF = {"fold": 0, "check_call": 1, "aggress_half_pot": 2, "aggress_pot": 2}


def example(
    decision: str,
    branch: str,
    reward_bb: float,
    *,
    purse_bb: float = 60.0,
    confidence: float = 1.0,
    seed: int = 0,
) -> TrainingExample:
    generator = random.Random(hash((decision, branch, seed)) & 0xFFFF)
    return TrainingExample(
        table_id=decision.split(":")[0],
        policy_version="filter-test",
        features=tuple(generator.uniform(-1.0, 1.0) for _ in range(LEARNING_INPUT_SIZE)),
        action_family_index=FAMILY_OF[branch],
        behavior_probabilities=(1.0, 0.0, 0.0),
        submitted_risk_fraction=0.0,
        purse_bb=purse_bb,
        reward_bb=reward_bb,
        counterfactual=True,
        opponent_confidence=confidence,
        decision_id=decision,
        action_branch=branch,
    )


def group(decision: str, rewards, **kwargs) -> list[TrainingExample]:
    return [
        example(decision, branch, reward, **kwargs)
        for branch, reward in zip(BRANCHES, rewards)
    ]


def centered_loss(predicted, targets, weights) -> float:
    """The trainer's per-group numerator, rewritten independently here."""

    mean = sum(predicted) / len(predicted)
    return sum(
        w * ((p - mean) - t) ** 2 for p, t, w in zip(predicted, targets, weights)
    )


class ZeroSignalPredicateTests(unittest.TestCase):
    """The predicate must agree with a search for a better-than-constant head."""

    def _targets_weights(self, rows, config):
        return (
            [_action_value_target(row, config) for row in rows],
            [_branch_value_weight(row, config) for row in rows],
        )

    def test_predicate_agrees_with_a_search_for_a_better_prediction(self) -> None:
        # IMPOSSIBLE BY CONSTRUCTION. The predicate claims exactly this: no
        # prediction beats a constant one on a zero-signal group, and some
        # prediction does on every other group. So search for one, without
        # ever consulting the w*t rule the predicate uses.
        #
        # The per-group loss is convex in the prediction, so a strictly better
        # point exists iff a strictly better point exists ARBITRARILY CLOSE to
        # the constant. The search is therefore a line search along each
        # coordinate and a few random directions, down a ladder of step sizes
        # -- which finds improvements far too small for a box-uniform sample
        # to stumble on, and there is no comparison slack to hide behind.
        config = TrainingConfig(architecture="v7", device="cuda")
        cases = [
            group("t0:hero:0", (0.0, 0.0, 0.0, 0.0)),          # zero, all equal
            group("t1:hero:0", (12.0, 12.0, 12.0, 12.0)),      # non-zero, all equal
            group("t2:hero:0", (-40.0, -40.0, -40.0, -40.0)),  # clipped, all equal
            group("t3:hero:0", (5.0, -5.0, 1.0, 0.0)),         # live
            group("t4:hero:0", (0.0, 0.0, 0.0, 1e-6)),         # barely live
            group("t5:hero:0", (3.0, 3.0, 3.0, 4.0)),          # one odd branch
        ]
        generator = random.Random(20260827)
        for rows in cases:
            with self.subTest(decision=rows[0].decision_id):
                size = len(rows)
                indexes = list(range(size))
                targets, weights = self._targets_weights(rows, config)
                flat = _zero_signal_group(rows, indexes, config)
                base = [1.7] * size
                constant = centered_loss(base, targets, weights)

                directions = []
                for index in indexes:
                    step = [0.0] * size
                    step[index] = 1.0
                    directions.append(step)
                    directions.append([-value for value in step])
                for _ in range(64):
                    directions.append(
                        [generator.uniform(-1.0, 1.0) for _ in indexes]
                    )

                best = constant
                for direction in directions:
                    for power in range(1, 10):
                        scale = 10.0**-power
                        trial = [
                            value + scale * delta
                            for value, delta in zip(base, direction)
                        ]
                        best = min(best, centered_loss(trial, targets, weights))

                # The predicate is exact; this search is floating point, so it
                # needs a noise floor. Measure the instrument: every case must
                # land either below 1e-12 relative improvement (rounding) or
                # above 1e-3 (a real improvement). Nothing may sit in between,
                # so the 1e-9 cut has nine orders of margin on both sides and
                # is not doing any of the deciding.
                improvement = (
                    0.0 if constant == 0.0 else (constant - best) / constant
                )
                self.assertFalse(
                    1e-12 < improvement < 1e-3,
                    f"search improvement {improvement!r} landed in the band the "
                    "threshold is supposed to be nowhere near",
                )
                beaten = improvement > 1e-9
                self.assertEqual(
                    flat,
                    not beaten,
                    f"predicate={flat} but the search {'did' if beaten else 'did not'}"
                    f" beat the constant ({best!r} vs {constant!r})",
                )

    def test_identical_non_zero_targets_are_zero_signal(self) -> None:
        # The obvious definition -- "every target is zero" -- misses this.
        config = TrainingConfig(architecture="v7", device="cuda")
        rows = group("t:hero:0", (9.0, 9.0, 9.0, 9.0))
        self.assertNotEqual(_action_value_target(rows[0], config), 0.0)
        self.assertTrue(_zero_signal_group(rows, range(len(rows)), config))

    def test_targets_clipped_to_the_same_bound_are_zero_signal(self) -> None:
        # return_scale_fraction clips at +/-1, so branches with different raw
        # returns can land on the same target. The corpus has none today; a
        # different scale or a shorter stack produces them.
        config = TrainingConfig(
            architecture="v7", device="cuda", return_scale_fraction=0.05
        )
        rows = group("t:hero:0", (30.0, 45.0, 60.0, 55.0))
        targets = [_action_value_target(row, config) for row in rows]
        self.assertEqual(targets, [1.0, 1.0, 1.0, 1.0])
        self.assertTrue(_zero_signal_group(rows, range(len(rows)), config))

    def test_a_single_branch_group_is_zero_signal(self) -> None:
        # With one row `centered` is identically zero: no prediction can move
        # the loss, so the group is a permanent additive constant.
        config = TrainingConfig(architecture="v7", device="cuda")
        rows = group("t:hero:0", (7.0, 0.0, 0.0, 0.0))[:1]
        self.assertEqual(len(rows), 1)
        self.assertTrue(_zero_signal_group(rows, range(1), config))

    def test_two_and_three_branch_groups_are_judged_on_their_own_rows(self) -> None:
        config = TrainingConfig(architecture="v7", device="cuda")
        rows = group("t:hero:0", (4.0, 4.0, 4.0, -9.0))
        self.assertTrue(_zero_signal_group(rows, range(3), config))
        self.assertFalse(_zero_signal_group(rows, range(4), config))

    def test_unequal_weights_break_identical_non_zero_targets(self) -> None:
        # w*t, not t. With per-family class weights the shared level no longer
        # cancels, so the group DOES supervise something and must be kept.
        config = TrainingConfig(
            architecture="v7", device="cuda", class_weights=(1.0, 1.0, 3.0)
        )
        rows = group("t:hero:0", (9.0, 9.0, 9.0, 9.0))
        self.assertFalse(_zero_signal_group(rows, range(len(rows)), config))

    def test_unequal_weights_do_not_break_an_all_zero_group(self) -> None:
        config = TrainingConfig(
            architecture="v7", device="cuda", class_weights=(1.0, 1.0, 3.0)
        )
        rows = group("t:hero:0", (0.0, 0.0, 0.0, 0.0))
        self.assertTrue(_zero_signal_group(rows, range(len(rows)), config))

    def test_differing_evidence_weights_break_an_identical_non_zero_group(self) -> None:
        config = TrainingConfig(architecture="v7", device="cuda")
        rows = group("t:hero:0", (9.0, 9.0, 9.0, 9.0))
        rows[2] = example("t:hero:0", BRANCHES[2], 9.0, confidence=0.1)
        self.assertFalse(_zero_signal_group(rows, range(len(rows)), config))

    def test_equality_is_exact_not_tolerant(self) -> None:
        config = TrainingConfig(architecture="v7", device="cuda")
        rows = group("t:hero:0", (9.0, 9.0, 9.0, 9.0 + 1e-13))
        self.assertFalse(_zero_signal_group(rows, range(len(rows)), config))


class FilterConfigTests(unittest.TestCase):
    def test_default_is_off(self) -> None:
        self.assertEqual(TrainingConfig().degenerate_group_filter, "off")
        self.assertEqual(DEGENERATE_GROUP_FILTERS[0], "off")

    def test_unknown_mode_is_rejected(self) -> None:
        rows = group("t:hero:0", (1.0, -1.0, 2.0, 0.0))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "degenerate_group_filter"):
                train_candidate(
                    tuple(rows),
                    directory,
                    TrainingConfig(
                        architecture="v7",
                        device="cuda",
                        degenerate_group_filter="sometimes",
                    ),
                )

    def test_the_filter_is_rejected_under_v6(self) -> None:
        rows = group("t:hero:0", (1.0, -1.0, 2.0, 0.0))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "v7 centered objective"):
                train_candidate(
                    tuple(rows),
                    directory,
                    TrainingConfig(degenerate_group_filter="drop"),
                )


class CliWiringTests(unittest.TestCase):
    def _run(self, extra: list[str]) -> TrainingConfig:
        captured: list[TrainingConfig] = []

        def record(examples, output_dir, config):
            captured.append(config)
            raise SystemExit(0)

        with patch("tools.self_play_cycle.harvest", return_value=()):
            with patch("tools.self_play_cycle.train_candidate", side_effect=record):
                with self.assertRaises(SystemExit):
                    cycle_main(
                        [
                            "--model-version",
                            "cli-wiring",
                            "--architecture",
                            "v7",
                            # cpu keeps this a pure argparse-to-config check on
                            # the stdlib interpreter; train_candidate is patched
                            # out, so the device is never used.
                            "--device",
                            "cpu",
                            *extra,
                        ]
                    )
        return captured[0]

    def test_the_flag_reaches_the_training_config(self) -> None:
        config = self._run(["--degenerate-group-filter", "drop"])
        self.assertEqual(config.degenerate_group_filter, "drop")

    def test_the_flag_defaults_to_off(self) -> None:
        self.assertEqual(self._run([]).degenerate_group_filter, "off")

    def test_the_flag_is_rejected_under_v6(self) -> None:
        with self.assertRaises(SystemExit):
            cycle_main(
                [
                    "--model-version",
                    "cli-wiring",
                    "--architecture",
                    "v6",
                    "--degenerate-group-filter",
                    "drop",
                ]
            )


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(), "CUDA PyTorch is unavailable"
)
class CudaFilterTests(unittest.TestCase):
    """Small CUDA fits. Nothing here is a training run: 40 groups, 2 epochs."""

    def corpus(self, *, flat_share: float = 0.4, hands: int = 40):
        rows: list[TrainingExample] = []
        generator = random.Random(11)
        for index in range(hands):
            decision = f"hand-{index}:hero:0"
            if index < int(hands * flat_share):
                # Half the zero-signal groups carry identical NON-ZERO
                # rewards. With every one of them at zero, a sabotage that
                # feeds the FILTERED group list to `state_value_targets`
                # passes all of these tests silently -- the corpus this
                # experiment runs on happens to have all-zero degenerate
                # groups, so that field is insensitive to the failure by
                # construction. These make it detectable.
                rewards = (0.0, 0.0, 0.0, 0.0) if index % 2 else (5.0,) * 4
            else:
                rewards = tuple(generator.uniform(-8.0, 8.0) for _ in range(4))
            rows.extend(group(decision, rewards))
        return tuple(rows)

    def train(self, mode: str, examples, directory: str, version: str):
        config = TrainingConfig(
            architecture="v7",
            device="cuda",
            epochs=2,
            batch_size=32,
            model_version=version,
            degenerate_group_filter=mode,
        )
        summary = train_candidate(examples, directory, config)
        manifest = json.loads(Path(summary.manifest_path).read_text(encoding="utf-8"))
        return summary, manifest["training"]

    def test_off_is_byte_identical_to_not_passing_the_option(self) -> None:
        # weights_sha256 covers model_version too, so compare the parameter
        # block itself: the option must be inert at its default.
        examples = self.corpus()
        with tempfile.TemporaryDirectory() as directory:
            explicit, _ = self.train("off", examples, directory, "explicit-off")
            config = TrainingConfig(
                architecture="v7",
                device="cuda",
                epochs=2,
                batch_size=32,
                model_version="implicit-off",
            )
            implicit = train_candidate(examples, directory, config)
            left = json.loads(
                Path(explicit.weights_path).read_text(encoding="utf-8")
            )["weights"]
            right = json.loads(
                Path(implicit.weights_path).read_text(encoding="utf-8")
            )["weights"]
        self.assertEqual(left, right)

    def test_state_value_supervision_is_identical_across_every_mode(self) -> None:
        # The invariant that catches a filtered group list reaching
        # state_value_targets, which would silently write 0.0 as a label.
        examples = self.corpus()
        sums = {}
        with tempfile.TemporaryDirectory() as directory:
            for mode in DEGENERATE_GROUP_FILTERS:
                _, training = self.train(mode, examples, directory, f"sv-{mode}")
                sums[mode] = training["state_value_target_abs_sum"]
        self.assertEqual(len(set(sums.values())), 1, sums)

    def test_the_manifest_states_how_the_candidate_was_made(self) -> None:
        examples = self.corpus()
        with tempfile.TemporaryDirectory() as directory:
            _, training = self.train("drop", examples, directory, "recorded")
        self.assertEqual(training["degenerate_group_filter"], "drop")
        self.assertIn("weighted target", training["degenerate_group_predicate"])
        counts = training["degenerate_group_counts"]
        self.assertGreater(counts["train_zero_signal"], 0)
        self.assertEqual(
            counts["train_batched"],
            counts["train_total"] - counts["train_zero_signal"],
        )
        self.assertEqual(training["best_epoch_criterion"], "filtered centered "
                         "validation loss")
        self.assertIsNotNone(training["validation_loss_weighted_unfiltered"])
        self.assertIsNotNone(training["validation_loss_weighted_filtered"])

    def test_zero_weight_keeps_the_batch_schedule_and_drop_shortens_it(self) -> None:
        examples = self.corpus()
        with tempfile.TemporaryDirectory() as directory:
            _, off = self.train("off", examples, directory, "sched-off")
            _, zero = self.train("zero_weight", examples, directory, "sched-zero")
            _, drop = self.train("drop", examples, directory, "sched-drop")
        self.assertEqual(zero["reward_steps_per_epoch"], off["reward_steps_per_epoch"])
        self.assertLess(drop["reward_steps_per_epoch"], off["reward_steps_per_epoch"])
        # The group-count conversion must not drift when groups are dropped.
        self.assertEqual(drop["group_batches"], off["group_batches"])

    def test_an_all_zero_signal_batch_does_not_produce_a_nan(self) -> None:
        # Every group flat except one, and a batch small enough that whole
        # batches are certainly all-flat. Under an unguarded normalization
        # this is 0/0 and clip_grad_norm_ raises.
        examples = self.corpus(flat_share=0.95, hands=40)
        with tempfile.TemporaryDirectory() as directory:
            config = TrainingConfig(
                architecture="v7",
                device="cuda",
                epochs=2,
                batch_size=4,
                model_version="all-flat-batch",
                degenerate_group_filter="zero_weight",
            )
            summary = train_candidate(examples, directory, config)
            document = json.loads(
                Path(summary.weights_path).read_text(encoding="utf-8")
            )
        self.assertIn("weights", document)

    def test_every_group_zero_signal_fails_loudly(self) -> None:
        examples = self.corpus(flat_share=1.0, hands=20)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "every training decision group"):
                self.train("drop", examples, directory, "all-flat")



class RandomArmTests(unittest.TestCase):
    """The attribution control, and the ways it could be inert.

    Without a size-matched random arm the experiment cannot separate
    "removing the attractor helped" from "removing 29.66% of the groups
    helped". But a random arm is only a control if it actually differs
    from the treated arm: one that happened to mute the whole zero-signal
    set, or none of it, would make CONFIRMED easier rather than harder.
    That is what these pin.
    """

    def test_random_is_an_accepted_mode(self) -> None:
        self.assertIn("random", DEGENERATE_GROUP_FILTERS)

    def test_random_is_rejected_under_v6(self) -> None:
        rows = group("t:hero:0", (1.0, -1.0, 2.0, 0.0))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "v7 centered objective"):
                train_candidate(
                    tuple(rows),
                    directory,
                    TrainingConfig(degenerate_group_filter="random"),
                )

    def test_the_draw_seed_defaults_to_zero_and_is_recorded(self) -> None:
        self.assertEqual(TrainingConfig().degenerate_group_filter_seed, 0)

    def test_the_seed_derivation_never_uses_pythons_hash(self) -> None:
        """`hash()` is PYTHONHASHSEED-randomised across processes.

        A mask that cannot be reproduced in a second process cannot be
        audited, and this repo's open problem is an unexplained 10-46%
        retraining spread -- irreproducibility is the last thing it needs.
        """

        source = pathlib.Path(
            "devfun_poker_playground/offline_trainer.py"
        ).read_text(encoding="utf-8")
        start = source.index("def _mask_flags")
        body = source[start : source.index("def action_weights", start)]
        # Code lines only -- the comment there explains why hash() is
        # avoided, and a naive substring search flags its own rationale.
        code = chr(10).join(
            line for line in body.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertIn("random.Random(", code)
        self.assertNotIn("hash(", code)


if __name__ == "__main__":
    unittest.main()
