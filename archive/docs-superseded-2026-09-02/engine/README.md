> **ARCHIVED 2026-09-02 — SUPERSEDED. NOT STATE, NOT PLAN, NOT RULES.**
> Do not use this file to learn what the system is, what its state is, or what to do.
> The live manual is `CLAUDE.md` → `.handoff/CONTEXT.md`. Open this file only for a
> specific fact that a live document cites here by path. Never edit it, never restore it.

# engine — the decision brain and its training modules

Package map rewritten 2026-09-02 from the module docstrings and the import
graph. The live runner (`run_agent.py`, driven by `live_session.py`) imports
this package **stdlib-only**: every `torch` import in here is function-local,
and importing the serve path leaves `torch` out of `sys.modules` (checked
2026-09-02). Nothing in this package performs an Arena request.

Authority for the current architecture: `branch_contract_v9.py` (the
four-branch contract: `fatal` / `passive` / `active` / `aggressive`) and
`.handoff/notes/V9_RESTRUCTURE_PLAN.md`. The rule layer's design record and
defect ledger is `rules/README.md`.

## The serve path — what one live decision touches

- `poker_policy.py` — the live policy entry point (`--standard` / `--aggressive` heuristics, `heuristic-aggressive-v6` is the evaluation champion) and the legacy fixed-weight proposal network it loads from `artifacts/tiny-policy-pure.json` (validated at start-up; it chooses no live actions).
- `decision_engine.py` — turns any proposal into a safe, legal action: the hard `SafetyGates`, temperature shaping, sizing and the wager floors, the bluff hookup, the opponent tracker, the deadline path, and the dark C1–C5 dials. **This is the live decision path; every gate change here is a live-money change** and is measured before it ships (2026-08-26 is why).
- `game_state.py` — validates the untrusted Arena snapshot and derives `effective_stack_chips`, `card_reveal_expense`, positions and the policy features.
- `hand_strength.py` — Monte Carlo equity, range conditioning, `equity_multiway`, `equity_vs_posterior`. `hand_potential.py` — Ppot/Npot. `strength_metric.py` + `preflop_percentiles.py` — the ONE canonical hand-strength scale that every feature, label and report uses.
- `opponent_model.py` — session-scoped aggression tracking that floors the engine's range conditioning (the arena reseats every hand, so it holds at most one hand of evidence per opponent).
- `learned_policy.py` — `load_approved`, the checksummed loader, and the format-1/2 (v6/v7) runtime; format 4 is dispatched to `learned_policy_v9.load_policy_v9`. Format 3 (v8) has no `load_approved` arm — v8 was never promotable — and is built directly by `tools/evaluate_v8.py`.
- `learned_policy_v9.py` — the served runtime (`candidate-v9-0003b` is format 4): composition, projection onto the four branches, the format-4 loader that refuses an unstamped `feature_normalization` block, the P3 belief provider by default, hyper-aggression roll OFF. `aggression_sizing.py` is its sizing function g — one definition, every consumer imports it.
- `feature_extract_v9.py` / `schema4.py` — the **414-input** v9 vector (FROZEN for the harvest), assembled on top of `feature_extract_v8.py` / `schema3.py` (413 inputs, v8), `action_history.py` (the history block), `belief_provider.py` + `p3_belief_provider.py` (Bayes over strength octiles with the fitted P3 fold model as the likelihood of each priced continue), and `strength_aware_opponent.py` (the fitted P3 opponent, `artifacts/p3/p3-fit.json`).
- `branch_contract_v9.py` — the four live branches with no fixed sizes; artifact format 4, family `v9-composed-value`.
- `rules/` — the composed rule-layer candidates C1–C5, wired DARK: every dial ships OFF and each needs its own measurement pass before it is ever enabled. Spec: `rules/README.md`.
- `training_telemetry.py` — the append-only decision / settled-hand journal (`.arena-training.jsonl`, journal schema 3 with the additive `proposed_branch` / belief-degrade fields) and `TrainingExample`.
- `learning_contract.py` — the versioned v7 feature and architecture contract plus the immutable manifest validator every artifact format still passes through.
- `policy_features.py` — the 125 legacy feature names and the frozen action labels the journal writes.
- `_vendor/treys/` — the vendored hand evaluator (`_vendor/README.md`); `tests/test_treys_evaluator.py` checks it against a brute-force oracle.

Two trainers are also on the serve path's import closure: `offline_trainer.py`
(its `_forward_v2` is the pure-Python v7 serve pass that keeps
`candidate-v7-0001c` servable as the rollback target) and `v8_trainer.py`
(the network factory the v8/v9 runtimes build from). Both keep their torch
imports function-local.

The engine also imports three modules from the repository root on every
decision: `bluff.py` (the bluff advisor), `lead_position.py` (the lead
gauge) and `risk_temperature.py` (the temperature gauge). They look
pre-reset; they are live.

## Training only — never imported by the runner

- `v9_trainer.py` — the Phase-A supervised trainer for the v9 network; `v9_trainer_phase_b.py` — the Phase-B composed-value trainer (its `--supervised-normalization` knob defaults to `raw`; the `constant-predictor` arm is a tested ablation); `supervised_loss_normalization_v9.py` — the normalizers behind that knob. CUDA interpreter: `C:\Users\user\poker-nn-training\.venv\Scripts\python.exe`.
- `v8_trainer_phase_b.py` — the byte-frozen v8 Phase-B trainer the v9 fork was cut from.
- `offline_trainer.py` — the v7 counterfactual-value trainer (frozen line, still importable for the serve pass above).
- `foreign_data.py` — foreign teacher CSV rows as validated behaviour-only examples (v7 line).
- `table_simulator.py` — the seeded Arena-shaped simulator, the scripted card-blind archetypes, and `decide_forced`, the counterfactual branch replay through the acting policy's own serve path. `arena_shaped_call_amounts` (default False = the frozen v8 bytes) is the v9 size encoding opt-in.

## The lines and where they stand (2026-09-02)

| line | family / format | modules | status |
|---|---|---|---|
| v9 | `v9-composed-value` / 4 | `*_v9.py`, `schema4`, `aggression_sizing`, `branch_contract_v9`, `rules/` | **served**: `candidate-v9-0003b`, promoted 2026-09-01, busted Playground S17 on 2026-09-02 (see `.handoff/STATUS.md`) |
| v8 | `v8-composed-value` / 3 | `*_v8.py`, `schema3`, `v8_trainer*` | frozen; measured non-promotable (2026-08-17); imported by the v9 modules |
| v7 | `v7-two-branch` / 2 | `learning_contract`, `learned_policy`, `offline_trainer` | frozen; `candidate-v7-0001c` is the rollback target and the frozen instruments' subject |
| v6 | format 1 | the same modules | loadable; its artifacts are in `archive/pre-reset-2026-08-16/` |

Retiring the v7/v8 code is the owner-run post-promotion pass recorded in
`.handoff/notes/V9_RESTRUCTURE_PLAN.md` ("Post-promotion cleanup"). It is not
a file move: the v9 modules import the v8 ones, `load_approved` still needs
the format-2 arm for the rollback target, and the frozen tripwires
(`tools/gate_ablation.py`, `tools/p3_gate.py`) need their v7 subject.
