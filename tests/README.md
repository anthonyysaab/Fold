# Tests

> **This map is incomplete and partly stale** (it omits ~10 test files added
> since it was written). `python -m pytest tests/ -q` is the source of truth;
> the current baseline is 344 passed, 1 expected CUDA skip. folder

This folder contains checks that run locally without joining Arena.

- `README.md` is this folder map.
- `test_bluff.py` checks the standalone bluff advisor's pricing math, gates, blockers, determinism, and validation.
- `test_bluff_wiring.py` checks the engine's bluff path: priced semi-bluffs in passive spots, the paired-board and raising-war gates, wildness vetoes, and telemetry marking.
- `test_foreign_data.py` checks the foreign CSV training boundary (eligibility, clamping, refusals) and the audit tool's calibration summaries.
- `test_game_state.py` checks numeric validation at the untrusted Arena snapshot boundary.
- `test_hyper_aggression.py` checks the anti-modeling dice roll: full-pot pressure on trigger, hard gates and the risk cap holding, deterministic rolls, and training exclusion.
- `test_lead_position.py` checks the lead gauge's bounds, chip monotonicity, positional accentuation, and validation.
- `test_learned_policy.py` checks artifact loading (checksums, engine parameters), legality of learned decisions, atomic promotion, and rollback.
- `test_learning_contract.py` protects the exact 142-input model shape and immutable artifact metadata.
- `test_offline_trainer.py` checks candidate artifact writing and the empty-data guard.
- `test_opponent_model.py` checks aggression tracking (dedup, hand resets, identity keys, decayed evidence) and proves repeated shoves flip the engine from folding to calling.
- `test_policy_fixes.py` protects six-player routing and the aggressive policy's authoritative call and raise thresholds.
- `test_risk_temperature.py` checks the gauge's bounds and monotonic response to strength, street, purse pressure, and player count.
- `test_runner_safety.py` protects emergency action order, reconnect handling, and runner arguments.
- `test_safety_gates.py` checks the injectable SafetyGates parameters: softened defaults, the unsoftened originals, validation ranges, JSON round-trips, and that injected gates change decisions.
- `test_runtime_layout.py` checks policy loading, credential-free runner help, and the dependency-free package import.
- `test_table_simulator.py` checks chip conservation, side pots, snapshot contract validity, determinism, self-play capture, and the perma-shover tripwire.
- `test_temperature_shaping.py` checks the bounded temperature response: cold loosens, hot tightens, hard gates never shift, and neutral shaping reproduces legacy play.
- `test_training_telemetry.py` protects identity-gated decisions, replay validation, settlement joins, and big-blind rewards.
