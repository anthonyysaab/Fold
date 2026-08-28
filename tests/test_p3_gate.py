"""Tests for the held-out P3 gate and the ``vs-p3`` battery channel.

The house rule this file exists to enforce: *validate the probe against an
impossible-by-construction invariant before believing its numbers*. So the
statistics helpers are checked against the **published** noise-floor artifact
(a known answer computed by a different tool on a different day), the channel
is checked against the frozen instrument's own seed conventions, and the
opponent is checked for the one failure that would silently turn ``vs-p3``
back into ``vs-median``.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from engine.strength_aware_opponent import (
    DEFAULT_FIT_PATH,
    P3Decision,
    StrengthAwareAgent,
    load_fit,
)
from tools import evaluate_v8, p3_gate
from tools.evaluate_policies import _BATTERIES, battery_tasks, _PolicySpec

REPO_ROOT = Path(__file__).resolve().parents[1]
NOISE_FLOOR = REPO_ROOT / "artifacts" / "evaluations" / "noise-floor-2026-08-15.json"
FIT_PATH = REPO_ROOT / DEFAULT_FIT_PATH


def _table(*, pot: int, to_call: int, hero_bet: int = 0) -> dict:
    return {
        "tableId": "t",
        "street": "flop",
        "potChips": pot,
        "selfSeatNumber": 1,
        "boardCards": ["Ah", "7d", "2c"],
        "seats": [
            {
                "seatNumber": 1,
                "status": "Active",
                "stackChips": 5_000,
                "currentBetChips": hero_bet,
                "totalCommittedChips": hero_bet,
                "holeCards": ["Kh", "Kd"],
            },
            {
                "seatNumber": 2,
                "status": "Active",
                "stackChips": 5_000,
                "currentBetChips": to_call + hero_bet,
                "totalCommittedChips": to_call + hero_bet,
                "holeCards": None,
            },
        ],
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callChips": to_call,
            "raiseRange": {"min": 1, "max": 5_000},
            "betRange": None,
            "allInToAmount": 5_000,
        },
    }


class WagerFractionTests(unittest.TestCase):
    """``(new chips - call) / (pot + call)`` — the engine's own pot fraction."""

    def test_pot_sized_bet_reads_one(self) -> None:
        table = _table(pot=300, to_call=0)
        fraction = p3_gate.wager_pot_fraction(table, {"action": "bet", "amount": 300})
        self.assertAlmostEqual(fraction, 1.0)

    def test_half_pot_bet_reads_one_half(self) -> None:
        table = _table(pot=300, to_call=0)
        fraction = p3_gate.wager_pot_fraction(table, {"action": "bet", "amount": 150})
        self.assertAlmostEqual(fraction, 0.5)

    def test_raise_does_not_double_count_the_call(self) -> None:
        # PENDING_EDITS 18d: the earlier formula counted the call as wager and
        # reported 1.333x for a pot-sized raise. Pot 300, call 100: a pot-sized
        # raise puts in 100 + 400 = 500 new chips over a 400 denominator.
        table = _table(pot=300, to_call=100)
        fraction = p3_gate.wager_pot_fraction(table, {"action": "raise", "amount": 500})
        self.assertAlmostEqual(fraction, 1.0)

    def test_passive_and_folding_actions_have_no_wager(self) -> None:
        table = _table(pot=300, to_call=100)
        for action in ("call", "check", "fold"):
            self.assertIsNone(p3_gate.wager_pot_fraction(table, {"action": action}))


class SummaryTests(unittest.TestCase):
    def test_family_rates_counts_and_median(self) -> None:
        records = [
            ("flop", "aggress", 0.9, 1.0),
            ("flop", "aggress", 0.8, 0.5),
            ("flop", "aggress", 0.7, 0.25),
            ("turn", "fold", 0.1, None),
            ("turn", "check_call", 0.4, None),
        ]
        rates = p3_gate.family_rates(records)
        self.assertEqual(rates["decisions"], 5)
        self.assertEqual(rates["counts"], {"aggress": 3, "check_call": 1, "fold": 1})
        self.assertAlmostEqual(rates["aggression_rate"], 0.6)
        self.assertAlmostEqual(rates["fold_rate"], 0.2)
        self.assertAlmostEqual(rates["median_wager_pot_fraction"], 0.5)

    def test_continue_minus_fold_is_signed_the_obvious_way(self) -> None:
        records = [
            ("flop", "aggress", 0.9, 1.0),
            ("flop", "check_call", 0.8, None),
            ("flop", "fold", 0.2, None),
            ("flop", "fold", 0.1, None),
        ]
        entry = p3_gate.continue_minus_fold(records)
        self.assertAlmostEqual(entry["value"], 0.85 - 0.15)
        self.assertEqual(entry["n_continue"], 2)
        self.assertEqual(entry["n_fold"], 2)
        self.assertAlmostEqual(entry["realized_fold_rate"], 0.5)

    def test_continue_minus_fold_declines_to_answer_on_thin_samples(self) -> None:
        entry = p3_gate.continue_minus_fold([("flop", "fold", 0.2, None)])
        self.assertIsNone(entry["value"])


class DerivedMdeTests(unittest.TestCase):
    """The derived MDE must reproduce the published construction exactly.

    ``noise-floor-2026-08-15.json`` was written by a different tool on a
    different day. If this file's ``2σ/√n`` disagrees with it on the published
    per-seed values, the gate's own MDE is not the published construction and
    nothing derived from it may be quoted alongside published channel MDEs.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not NOISE_FLOOR.exists():  # pragma: no cover - artifact present in repo
            raise unittest.SkipTest(f"missing {NOISE_FLOOR}")
        cls.noise = json.loads(NOISE_FLOOR.read_text(encoding="utf-8"))

    def test_reproduces_every_published_channel_sigma_and_mde(self) -> None:
        for channel, data in self.noise["battery_channels"].items():
            with self.subTest(channel=channel):
                values = data["per_seed_bb_per_100"]
                derived = p3_gate.independent_seed_mde(values)
                self.assertAlmostEqual(
                    derived["sigma_bb_per_100"], data["sigma_bb_per_100"], places=2
                )
                for key, published in data["mde_bb_per_100"].items():
                    if key in derived["mde_bb_per_100"]:
                        self.assertAlmostEqual(
                            derived["mde_bb_per_100"][key], published, places=2
                        )

    def test_zero_variance_gives_a_zero_floor(self) -> None:
        derived = p3_gate.independent_seed_mde([12.0] * 8)
        self.assertEqual(derived["sigma_bb_per_100"], 0.0)
        self.assertEqual(derived["mde_bb_per_100"]["8_seeds"], 0.0)


class PairedContrastTests(unittest.TestCase):
    def test_identical_arms_are_unresolved_with_a_zero_mde(self) -> None:
        values = [10.0, -3.0, 22.5, 7.25, 0.0, -11.0, 4.0, 9.5]
        contrast = p3_gate.paired_contrast(values, values, label_a="a", label_b="b")
        self.assertEqual(contrast["mean"], 0.0)
        self.assertEqual(contrast["paired_mde_bb_per_100"], 0.0)
        self.assertEqual(contrast["verdict"], "UNRESOLVED")
        self.assertTrue(all(diff == 0.0 for diff in contrast["diffs"]))

    def test_a_constant_offset_resolves_and_names_the_winner(self) -> None:
        base = [10.0, -3.0, 22.5, 7.25, 0.0, -11.0, 4.0, 9.5]
        contrast = p3_gate.paired_contrast(
            [value + 40.0 for value in base], base, label_a="a", label_b="b"
        )
        self.assertAlmostEqual(contrast["mean"], 40.0)
        self.assertEqual(contrast["paired_mde_bb_per_100"], 0.0)
        self.assertEqual(contrast["verdict"], "a ahead")

    def test_a_difference_inside_the_noise_stays_unresolved(self) -> None:
        base = [0.0, 100.0, -100.0, 50.0, -50.0, 20.0, -20.0, 0.0]
        noisy = [value + shift for value, shift in zip(base, base[::-1])]
        contrast = p3_gate.paired_contrast(noisy, base, label_a="a", label_b="b")
        self.assertEqual(contrast["verdict"], "UNRESOLVED")
        self.assertLess(abs(contrast["mean"]), contrast["paired_mde_bb_per_100"])


def _arms_fixture(seeds: int = 8) -> dict:
    """A synthetic arms fragment covering every label ``analyse`` reaches for."""

    base = [10.0, -3.0, 22.5, 7.25, 0.0, -11.0, 4.0, 9.5][:seeds]

    def arm(offset: float) -> dict:
        values = [round(value + offset, 2) for value in base]
        return {
            channel: {
                "hands_per_seed": 1_000,
                "bb_per_100": round(sum(values) / len(values), 2),
                "seeds": values,
            }
            for channel in p3_gate.CHANNELS
        }

    labels = {
        "cand": 5.0,
        "inc": -5.0,
        "champ": 0.0,
        p3_gate.NULL_MIRROR_LABEL: 0.0,
        **{f"trivial-{mode}": 1.0 for mode in evaluate_v8.TRIVIAL_MODES},
    }
    return {label: arm(offset) for label, offset in labels.items()}


class AnalysisTests(unittest.TestCase):
    def test_self_null_mirror_is_detected_as_exactly_zero(self) -> None:
        analysis = p3_gate.analyse(
            _arms_fixture(), candidate="cand", incumbent="inc", champion="champ"
        )
        self.assertTrue(analysis["self_null"]["passed"])
        for entry in analysis["self_null"]["per_channel"].values():
            self.assertEqual(entry["mean"], 0.0)

    def test_a_drifted_mirror_fails_the_self_null(self) -> None:
        arms = _arms_fixture()
        arms[p3_gate.NULL_MIRROR_LABEL]["vs-p3"]["seeds"][0] += 0.01
        analysis = p3_gate.analyse(
            arms, candidate="cand", incumbent="inc", champion="champ"
        )
        self.assertFalse(analysis["self_null"]["passed"])

    def test_every_channel_reports_a_contrast_and_a_dominance_verdict(self) -> None:
        analysis = p3_gate.analyse(
            _arms_fixture(), candidate="cand", incumbent="inc", champion="champ"
        )
        for channel in p3_gate.CHANNELS:
            entry = analysis["channels"][channel]
            self.assertIn("candidate_vs_champion", entry["contrasts"])
            self.assertIn(
                "always_aggress_large_bb_per_100", entry["trivial_baseline_gate"]
            )
            # The mirror never appears as a rankable arm.
            ranked = [label for label, _ in entry["trivial_baseline_gate"]["arms_ranked"]]
            self.assertNotIn(p3_gate.NULL_MIRROR_LABEL, ranked)
        self.assertIn("mechanism_test", analysis)


class ReproductionGateTests(unittest.TestCase):
    """Same seeds, deterministic champion, so the values must be identical."""

    @classmethod
    def setUpClass(cls) -> None:
        if not NOISE_FLOOR.exists():  # pragma: no cover - artifact present in repo
            raise unittest.SkipTest(f"missing {NOISE_FLOOR}")
        cls.noise = json.loads(NOISE_FLOOR.read_text(encoding="utf-8"))

    def _champion_arm(self, channel: str, values: list, hands: int) -> dict:
        return {
            "champ": {
                channel: {
                    "hands_per_seed": hands,
                    "bb_per_100": round(sum(values) / len(values), 2),
                    "seeds": values,
                }
            }
        }

    def test_published_values_reproduce_themselves(self) -> None:
        published = self.noise["battery_channels"]["vs-median"]
        values = [round(value, 2) for value in published["per_seed_bb_per_100"][:16]]
        gate = p3_gate.reproduction_gate(
            self._champion_arm("vs-median", values, published["hands_per_seed"]),
            "champ",
            self.noise,
        )
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["channels"]["vs-median"]["per_seed_identical"])

    def test_one_perturbed_seed_fails_the_gate(self) -> None:
        published = self.noise["battery_channels"]["vs-median"]
        values = [round(value, 2) for value in published["per_seed_bb_per_100"][:16]]
        values[3] += 0.01
        gate = p3_gate.reproduction_gate(
            self._champion_arm("vs-median", values, published["hands_per_seed"]),
            "champ",
            self.noise,
        )
        self.assertFalse(gate["passed"])
        self.assertAlmostEqual(gate["channels"]["vs-median"]["max_abs_delta"], 0.01)

    def test_a_different_hand_count_is_not_comparable(self) -> None:
        published = self.noise["battery_channels"]["vs-median"]
        values = [round(value, 2) for value in published["per_seed_bb_per_100"][:16]]
        gate = p3_gate.reproduction_gate(
            self._champion_arm("vs-median", values, 50), "champ", self.noise
        )
        self.assertFalse(gate["comparable"])
        self.assertFalse(gate["passed"])
        self.assertIn("note", gate)

    def test_without_a_noise_floor_the_gate_says_so(self) -> None:
        self.assertEqual(
            p3_gate.reproduction_gate({}, "champ", None), {"available": False}
        )


class HarnessGateTests(unittest.TestCase):
    """The half that separates "the harness moved" from "the policy moved"."""

    GAUNTLET = {
        "batteries": {
            "channels": {
                "vs-median": {"seeds": [95.6, 137.38, 132.01, 95.83]},
                "vs-station": {"seeds": [53.7, 12.0, -8.25, 66.0]},
            }
        }
    }

    def _arms(self, median: list, station: list) -> dict:
        return {
            "cand": {
                "vs-median": {"seeds": median, "hands_per_seed": 1_000},
                "vs-station": {"seeds": station, "hands_per_seed": 1_000},
            }
        }

    def test_identical_per_seed_values_pass(self) -> None:
        gate = p3_gate.harness_gate(
            self._arms(
                [95.6, 137.38, 132.01, 95.83, 1.0], [53.7, 12.0, -8.25, 66.0, 2.0]
            ),
            "cand",
            self.GAUNTLET,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["channels"]["vs-median"]["seeds_compared"], 4)

    def test_one_changed_value_fails(self) -> None:
        gate = p3_gate.harness_gate(
            self._arms([95.6, 137.38, 132.01, 95.84], [53.7, 12.0, -8.25, 66.0]),
            "cand",
            self.GAUNTLET,
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["channels"]["vs-median"]["per_seed_identical"])
        self.assertTrue(gate["channels"]["vs-station"]["per_seed_identical"])

    def test_without_a_gauntlet_the_gate_says_so(self) -> None:
        self.assertEqual(
            p3_gate.harness_gate({}, "cand", None), {"available": False}
        )


class ShippedReportTests(unittest.TestCase):
    """The report this gate shipped must keep asserting what it claims."""

    PATH = REPO_ROOT / "artifacts" / "evaluations" / "p3-gate-2026-08-16.json"

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.PATH.exists():  # pragma: no cover - artifact present in repo
            raise unittest.SkipTest(f"missing {cls.PATH}")
        cls.report = json.loads(cls.PATH.read_text(encoding="utf-8"))

    def test_the_instrument_checks_all_passed(self) -> None:
        self.assertTrue(self.report["instrument"]["known_answer_test"]["passed"])
        self.assertTrue(self.report["instrument"]["monotonicity_sweep"]["passed"])
        self.assertTrue(self.report["analysis"]["self_null"]["passed"])
        self.assertTrue(self.report["analysis"]["harness_gate"]["passed"])
        self.assertEqual(self.report["arms"]["chip_conservation_violations"], 0)

    def test_the_gate_ran_at_at_least_eight_seeds_on_every_arm(self) -> None:
        self.assertGreaterEqual(self.report["arms"]["seed_count"], 8)
        for label, channels in self.report["arms"]["arms"].items():
            for channel, entry in channels.items():
                with self.subTest(arm=label, channel=channel):
                    self.assertEqual(
                        len(entry["seeds"]), self.report["arms"]["seed_count"]
                    )

    def test_vs_p3_is_present_and_carries_a_derived_mde(self) -> None:
        channel = self.report["analysis"]["channels"][p3_gate.P3_CHANNEL]
        champion = self.report["arms_of_interest"]["champion"]
        self.assertIn(champion, channel["independent_seed_mde"])
        self.assertGreater(channel["channel_sigma_bb_per_100"][champion], 0.0)


class ChannelWiringTests(unittest.TestCase):
    def test_vs_p3_mirrors_vs_median_term_for_term(self) -> None:
        """The control is only a control if one thing differs."""

        median = next(build for name, build, _ in _BATTERIES if name == "vs-median")(13)
        p3_seat = evaluate_v8.p3_opponents(13)[0][1]
        self.assertEqual(p3_seat.aggression, median.aggression)
        self.assertEqual(p3_seat.fold_vs_bet, median.fold_vs_bet)
        self.assertEqual(p3_seat.shove_rate, median.shove_rate)
        self.assertEqual(p3_seat.seed, median.seed)
        self.assertEqual(
            evaluate_v8.channel_hands("vs-p3"), evaluate_v8.channel_hands("vs-median")
        )
        # ...and the one thing that does: the P3 seat reads its own cards.
        self.assertFalse(median.reads_cards)
        self.assertTrue(p3_seat.reads_cards)

    def test_channel_plan_uses_the_frozen_instrument_seed_convention(self) -> None:
        spec = _PolicySpec(label="x", kind="heuristic", equity_trials=8)
        published = battery_tasks(
            spec, seeds=(0, 1, 2), scale=1.0, stack=6_000, reset_stacks=False
        )
        mine = evaluate_v8.channel_battery_plan(
            spec, ("vs-median",), seeds=(0, 1, 2), scale=1.0, stack=6_000
        )
        reference = [item for item in published if item[0] == "vs-median"]
        self.assertEqual(len(mine), len(reference))
        for (_, hands, task), (_, ref_hands, ref_task) in zip(mine, reference):
            self.assertEqual(hands, ref_hands)
            self.assertEqual(task.seed, ref_task.seed)
            self.assertEqual(task.opponent_seed, ref_task.opponent_seed)
            self.assertEqual(task.stack, ref_task.stack)

    def test_unknown_channel_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_v8.channel_opponents("vs-nothing", 13)

    def test_trivial_spec_builds_the_published_baseline_agent(self) -> None:
        for mode in evaluate_v8.TRIVIAL_MODES:
            spec = evaluate_v8.TrivialSpec(mode)
            self.assertEqual(spec.label, f"trivial-{mode}")
            self.assertIsInstance(spec.build(), evaluate_v8.TrivialAgent)

    def test_hero_wrapping_matches_the_published_convention(self) -> None:
        class Diagnostic:
            def decide_with_diagnostics(self, table):  # pragma: no cover - stub
                raise AssertionError("not called")

        class Bare:
            def decide(self, table):  # pragma: no cover - stub
                raise AssertionError("not called")

        self.assertNotIsInstance(evaluate_v8._wrap_hero(Bare()), object.__class__)
        self.assertTrue(
            hasattr(evaluate_v8._wrap_hero(Diagnostic()), "policy_version")
        )
        bare = Bare()
        self.assertIs(evaluate_v8._wrap_hero(bare), bare)


class DegradationGuardTests(unittest.TestCase):
    """A silently degraded P3 seat is ``vs-median`` wearing the wrong label."""

    def test_the_plain_agent_degrades_and_counts_it(self) -> None:
        agent = StrengthAwareAgent("p3", 0.226, 0.5, 0.0, 13, fit_path=str(FIT_PATH))
        table = _table(pot=300, to_call=100)
        table["seats"][0]["holeCards"] = None
        probability = agent._fold_probability(table, table["allowedActions"])
        self.assertEqual(probability, agent.fold_vs_bet)
        self.assertEqual(agent.fallback_count, 1)

    def test_the_strict_agent_refuses_instead(self) -> None:
        agent = evaluate_v8.StrictStrengthAwareAgent(
            "p3", 0.226, 0.5, 0.0, 13, fit_path=str(FIT_PATH)
        )
        table = _table(pot=300, to_call=100)
        table["seats"][0]["holeCards"] = None
        with self.assertRaises(AssertionError):
            agent._fold_probability(table, table["allowedActions"])

    def test_the_strict_agent_is_otherwise_the_same_opponent(self) -> None:
        table = _table(pot=300, to_call=100)
        plain = StrengthAwareAgent("p3", 0.226, 0.5, 0.0, 13, fit_path=str(FIT_PATH))
        strict = evaluate_v8.StrictStrengthAwareAgent(
            "p3", 0.226, 0.5, 0.0, 13, fit_path=str(FIT_PATH)
        )
        self.assertEqual(
            plain._fold_probability(table, table["allowedActions"]),
            strict._fold_probability(table, table["allowedActions"]),
        )
        self.assertEqual(plain.decide(table), strict.decide(table))
        self.assertEqual(plain.fallback_count, 0)


class MonotonicityTests(unittest.TestCase):
    """The two properties the opponent exists for, on the shipped artifact."""

    def test_shipped_fit_sweeps_the_right_way_on_every_street(self) -> None:
        if not FIT_PATH.exists():  # pragma: no cover - artifact present in repo
            raise unittest.SkipTest(f"missing {FIT_PATH}")
        report = p3_gate.monotonicity_sweep(FIT_PATH)
        self.assertTrue(report["passed"], report)
        for street, entry in report["streets"].items():
            with self.subTest(street=street):
                self.assertTrue(entry["decreasing_in_strength"])
                self.assertTrue(entry["increasing_in_price"])

    def test_the_sweep_is_deterministic(self) -> None:
        if not FIT_PATH.exists():  # pragma: no cover - artifact present in repo
            raise unittest.SkipTest(f"missing {FIT_PATH}")
        self.assertEqual(
            p3_gate.monotonicity_sweep(FIT_PATH), p3_gate.monotonicity_sweep(FIT_PATH)
        )

    def test_a_hopeless_hand_at_a_terrible_price_folds_more_than_the_nuts_free(
        self,
    ) -> None:
        if not FIT_PATH.exists():  # pragma: no cover - artifact present in repo
            raise unittest.SkipTest(f"missing {FIT_PATH}")
        agent = evaluate_v8.StrictStrengthAwareAgent(
            "p3", 0.226, 0.5, 0.0, 13, fit=load_fit(str(FIT_PATH))
        )

        def probe(strength: float, price: float) -> float:
            return agent.fold_probability_for(
                P3Decision(
                    street="flop",
                    strength_percentile=strength,
                    pot_odds=price,
                    bet_to_pot=price / max(1e-9, 1.0 - price),
                    texture=0.3,
                    active_players=2,
                    position_unit=0.5,
                    to_call=100,
                    pot=200,
                )
            )

        self.assertGreater(probe(0.02, 0.66), probe(0.98, 0.09))


class ChannelMatchTests(unittest.TestCase):
    """One real (tiny) match on the new channel, end to end."""

    def test_a_vs_p3_match_conserves_chips_and_never_degrades(self) -> None:
        if not FIT_PATH.exists():  # pragma: no cover - artifact present in repo
            raise unittest.SkipTest(f"missing {FIT_PATH}")
        task = evaluate_v8.ChannelBatteryTask(
            hero=evaluate_v8.TrivialSpec("always_aggress_large"),
            channel=evaluate_v8.P3_CHANNEL,
            opponent_seed=13,
            hands=60,
            seed=100,
            stack=6_000,
            fit_path=str(FIT_PATH),
        )
        # The strict seat raises on any degradation, so completing is the proof.
        outcome = task.run()
        self.assertTrue(outcome.chips_conserved)
        self.assertGreater(outcome.result.hands, 0)
        self.assertTrue(math.isfinite(outcome.result.bb_per_100("hero")))

    def test_the_channel_is_deterministic_at_a_fixed_seed(self) -> None:
        if not FIT_PATH.exists():  # pragma: no cover - artifact present in repo
            raise unittest.SkipTest(f"missing {FIT_PATH}")
        task = evaluate_v8.ChannelBatteryTask(
            hero=evaluate_v8.TrivialSpec("always_check_call"),
            channel=evaluate_v8.P3_CHANNEL,
            opponent_seed=13,
            hands=60,
            seed=100,
            stack=6_000,
            fit_path=str(FIT_PATH),
        )
        self.assertEqual(
            task.run().result.bb_per_100("hero"), task.run().result.bb_per_100("hero")
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
