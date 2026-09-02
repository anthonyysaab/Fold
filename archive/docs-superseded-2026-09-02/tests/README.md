> **ARCHIVED 2026-09-02 — SUPERSEDED. NOT STATE, NOT PLAN, NOT RULES.**
> Do not use this file to learn what the system is, what its state is, or what to do.
> The live manual is `CLAUDE.md` → `.handoff/CONTEXT.md`. Open this file only for a
> specific fact that a live document cites here by path. Never edit it, never restore it.

# Tests

`python -m pytest tests/ -q` with the stdlib interpreter
(`C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe`) is the
source of truth. Baseline as of **2026-09-02: 1007 passed, 21 skipped, 355
subtests passed** (~3 min). Every skip is a torch-availability skip and passes
on the CUDA interpreter (`C:\Users\user\poker-nn-training\.venv\Scripts\python.exe`).
Tests that read `foreign play data/` skip when the archive is absent, so a
machine without it reports FEWER passes, not failures. `ruff check .` is not
clean and is not expected to be: 7 pre-existing errors, listed in `CLAUDE.md`.

Nothing here contacts the Arena. **A green suite is not evidence of safety on
engine code** — two live-money holes shipped past one on 2026-08-30 — so
engine-touching changes are swept adversarially as well as tested.

All 61 `test_*.py` files, one line each (map rewritten 2026-09-02).

## Live path and money safety

- `test_live_session.py` — guards for the continuous live-session supervisor: the free-Playground guard, the hard stops, the release paths.
- `test_runner_safety.py` — runner and deadline failure behaviour: emergency action order, reconnect handling, runner arguments.
- `test_play_entrypoints.py` — `play.sh` / `play.cmd` and the policy default must serve what is deployed.
- `test_runtime_layout.py` — policy loading, credential-free help, dependency-free package import.
- `test_safety_gates.py` — the injectable `SafetyGates` parameters, their ranges and JSON round-trips, that injected gates change decisions, and discriminating cases for the three dark 2026-08-27 repairs.
- `test_temperature_shaping.py` — the bounded temperature response: cold loosens, hot tightens, hard gates never shift.
- `test_hyper_aggression.py` — the anti-modeling dice roll: hard gates and the risk cap hold, rolls are deterministic, fires are excluded from training.
- `test_policy_fixes.py` — six-player routing and the aggressive policy's call and raise thresholds.
- `test_bluff.py` / `test_bluff_wiring.py` — the standalone bluff advisor, and its wiring into the engine's passive spots.
- `test_opponent_model.py` — aggression tracking, escalation decay, and the perma-shove remedy.
- `test_game_state.py` — validation at the untrusted Arena snapshot boundary.
- `test_lead_position.py` / `test_risk_temperature.py` — the two standalone gauges the engine imports from the repo root.

## Engine internals

- `test_hand_strength.py` — deterministic boundary tests for the Monte Carlo equity estimator.
- `test_hand_potential.py` — the Loki/Poki Ppot/Npot decomposition contract.
- `test_strength_metric.py` — the canonical hand-strength metric.
- `test_treys_evaluator.py` — structural best-five selection agrees with a brute-force oracle (added 2026-09-02, untracked).
- `test_table_simulator.py` — chip conservation, side pots, snapshot validity, determinism, counterfactual replay, the perma-shover tripwire.
- `test_training_telemetry.py` — append-only decisions and settled-hand rewards.
- `test_action_history.py` — the schema-3 history encoder.
- `test_belief_provider.py` / `test_p3_belief_provider.py` — the belief-bucket plumbing, and the fitted P3 provider (block 8).
- `test_strength_aware_opponent.py` / `test_p3_audit.py` — the P3 opponent on the shipped fit, and the invariants the P3 stage relies on but does not check anywhere.
- `test_rules_composition.py` — `engine/rules` invariants, including the zero-diff fuzz proving the dark dials change nothing.

## The v9 line (served)

- `test_learned_policy_v9.py` — composition, projection, the format-4 loader and its stamped normalization block.
- `test_feature_extract_v9.py` — schema-4 assembler invariants.
- `test_v9_engine_coupling.py` — L5: the hardened catch-alls and the gated shove lane.
- `test_v9_trainer.py` / `test_v9_trainer_phase_b.py` — the Phase-A and Phase-B v9 trainers (the torch parts skip on the stdlib interpreter).
- `test_supervised_loss_normalization_v9.py` — the Phase-1 supervised-loss normalization knob.
- `test_build_phase_a_dataset_v9.py` / `test_build_phase_b_corpus_v9.py` — the v9 dataset builder (including the board look-ahead regression) and the v9 harvester (owner-decision pins, the purity check, `--merge`).
- `test_ols_baseline.py` — the OLS promotion gate (added 2026-09-02, untracked).
- `test_evaluate_v8.py` — the gauntlet wrapper: trivial baselines (the v8 and v9 floors), plan structure, strength-separation math, the format-3/4 policy spec.
- `test_head_degeneracy_audit.py` — the head-degeneracy detector and its controls.

## The v8 line (frozen)

- `test_learned_policy_v8.py`, `test_feature_extract_v8.py`, `test_schema3.py`, `test_v8_parity.py`, `test_v8_trainer.py`, `test_v8_trainer_phase_b.py`, `test_build_phase_b_corpus.py`, `test_phase_a_dataset.py`, `test_summarize_seed_spread.py` — the format-3 runtime, the 413-input schema and its parity pins, both v8 trainers, the v8 dataset builder and harvester, and the three-seed spread summarizer.

## The v7 line and the frozen instruments

- `test_learned_policy.py` — the learned runtime, promotion and rollback path (reads `candidate-v7-0001c`).
- `test_learning_contract.py` — the 142-input contract and immutable manifest metadata.
- `test_offline_trainer.py`, `test_v7_core.py`, `test_cuda_trainer.py`, `test_degenerate_group_filter.py` — the v7 trainer, its pure-Python forward pass, the CUDA smoke, and the zero-signal group filter.
- `test_evaluate_policies.py` — the gauntlet's statistics and report structure.
- `test_p3_gate.py` — the held-out P3 gate and the `vs-p3` battery channel.
- `test_gate_binding_audit.py` — the live-journal gate binding audit.
- `test_self_play_cycle.py` / `test_harvest_parallelism.py` — the v7 harvest cycle and the parallel-leg equivalence proof.
- `test_measure_field_separation.py` — the field strength-separation benchmark: the instrument before the result.

## The replay archive pipeline

- `test_collect_foreign_play_data.py` — settlement receipts and pre-action state reconstruction from paginated replays.
- `test_foreign_corpus_rebuild.py` — reconcile and rebuild against a fixture holding every gap class the real archive contains.
- `test_foreign_data.py` — the foreign-CSV training boundary and the audit summary.
