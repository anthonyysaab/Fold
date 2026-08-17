# Poker policy package

> **Partly pre-reset.** References below to a "128-64 shared trunk" and "three
> model heads" describe the v6 shape; every v7 artifact on disk uses trunk
> `[256, 256, 128]` with four heads, and `learned_policy.py` carries a format-2
> runtime (`LearnedPokerPolicyV7`) beside the v6 one. `.handoff/CONTEXT.md` is
> authoritative for current state.

This folder contains poker decisions and their inputs. It performs no Arena network requests.

- `README.md` is this folder map.
- `__init__.py` marks this directory as a Python package without loading optional dependencies.
- `_vendor/` contains a copied poker-card evaluator required at runtime.
- `decision_engine.py` turns policy proposals into safe, legal actions. It shifts normal thresholds a bounded amount with the situation temperature (`TemperatureShaping`), reads every hard gate from the injectable `SafetyGates` parameter set, consults the bluff advisor in passive spots (fed by the lead gauge and opponent model), and attaches private equity, temperature, lead, and bluff diagnostics.
- `game_state.py` validates Arena table data and turns it into policy features.
- `hand_strength.py` estimates showdown strength and evaluates how much the hole cards improve the board.
- `foreign_data.py` converts the public collector's teacher-eligible CSV rows into validated, behavior-only training examples with explicit foreign provenance. It upgrades stored schema-1 rows to schema 2 with honest neutral values for live-only opponent context.
- `learned_policy.py` loads a checksummed candidate or the approved artifact into a playing policy. V6 interprets the three action outputs as counterfactual values, uses heuristic sizing by default, and can operate as a confidence/OOD-gated correction layer; every hard gate, temperature, tracker, and bluff path stays in charge.
- `learning_contract.py` defines the versioned 142 inputs (schema 2 adds opponent evidence and standing), 128-64 shared trunk, three model heads, and immutable artifact manifest.
- `opponent_model.py` tracks each opponent's observed aggression frequency during a session and floors the engine's range conditioning with it, so a permanent shover stops being credited with strength.
- `offline_trainer.py` trains same-state legal-family value estimates from counterfactual simulator groups, pretrains the shared trunk with a disposable behavior head, reports best-action accuracy, regret, and held-out hybrid calibration, and writes local candidate artifacts. It does not deploy them.
- `poker_policy.py` is the live policy entry point. It exposes standard and aggressive rule settings and can read legacy fixed weights.
- `table_simulator.py` deals seeded 2-6-player no-limit hands that emit Arena-shaped snapshots, with side pots, BB/100 scoring, stable decision-group IDs, and averaged same-state legal-family replay for counterfactual value targets.
- `policy_features.py` defines the exact 125 inputs and three action-family labels used by policy weights.
- `training_telemetry.py` validates and appends private local decision and settlement records for later offline learning.
- `torch_network.py` defines the legacy optional PyTorch network used with old checkpoints.
- `torch_policy.py` is the legacy PyTorch-backed adapter. Live play does not import it, and its default equity path bypasses its logits.
