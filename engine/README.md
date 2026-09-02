# engine

Revision 2026-09-02. Stdlib-only on the serve path: every `torch` import is
function-local (verified: importing the serve modules leaves `torch` out of
`sys.modules`). No Arena requests from this package. Architecture:
`.handoff/notes/V9_ARCHITECTURE.md`. Previous map:
`archive/docs-superseded-2026-09-02/engine/README.md`.

## Serve path — what one live decision touches

| module | role |
|---|---|
| `decision_engine.py` | turns any proposal into a safe legal action: `SafetyGates`, `TemperatureShaping`, sizing floors, bluff hookup, opponent tracker, deadline path, the dark C1–C5 dials. **Every edit here is a live-money edit.** |
| `game_state.py` | validates the Arena snapshot; `effective_stack_chips`, `contested_stack_chips`, `card_reveal_expense`, positions |
| `hand_strength.py` | Monte Carlo equity, range conditioning, `equity_multiway`, `equity_vs_posterior` |
| `hand_potential.py` | Ppot / Npot |
| `strength_metric.py`, `preflop_percentiles.py` | the one canonical hand-strength scale (features, labels, reports) |
| `opponent_model.py` | session-scoped aggression tracking → range floor |
| `poker_policy.py` | heuristic policies (`--standard` / `--aggressive`), the fallback when nothing is approved; loads the legacy `tiny-policy-pure.json` at start-up (chooses no actions) |
| `learned_policy.py` | `load_approved`; format-1/2 runtime; dispatches format 4 to `learned_policy_v9` |
| `learned_policy_v9.py` | **the served runtime**: composition, projection, format-4 loader, P3 belief provider, hyper roll 0.0 |
| `aggression_sizing.py` | g — the one sizing function |
| `branch_contract_v9.py` | the four-branch contract (normative) |
| `feature_extract_v9.py`, `schema4.py` | the 414-input vector (frozen); built on `feature_extract_v8.py`, `schema3.py`, `action_history.py` |
| `belief_provider.py`, `p3_belief_provider.py` | belief-bucket interface; the fitted P3 posterior |
| `strength_aware_opponent.py` | the P3 fold model and battery opponent |
| `rules/` | C1–C5, wired dark (`rules/README.md`) |
| `training_telemetry.py` | the journal (schema 3) and `TrainingExample` |
| `learning_contract.py`, `policy_features.py` | the v7 contract and manifest validator every format passes; the 125 legacy names and the frozen action labels |
| `_vendor/treys/` | vendored hand evaluator |

Also imported on the serve closure: `offline_trainer.py` (`_forward_v2`, the
pure-Python v7 serve pass for the rollback target) and `v8_trainer.py` (the
network factory). Root modules imported on every decision: `bluff.py`,
`lead_position.py`, `risk_temperature.py`.

## Training only

| module | role |
|---|---|
| `v9_trainer.py` | Phase A (supervised heads from the replay archive) |
| `v9_trainer_phase_b.py` | Phase B (composed value from the harvest; `--supervised-normalization`) |
| `supervised_loss_normalization_v9.py` | the constant-predictor normalizers behind that knob |
| `table_simulator.py` | seeded Arena-shaped simulator, scripted archetypes, `decide_forced`; `arena_shaped_call_amounts` is the v9 size-encoding opt-in |
| `v8_trainer_phase_b.py`, `offline_trainer.py`, `foreign_data.py` | frozen v8 / v7 trainers and the foreign-CSV boundary |

## Lines

| line | format | modules | status |
|---|---|---|---|
| v9 | 4 | `*_v9.py`, `schema4`, `aggression_sizing`, `branch_contract_v9`, `rules/` | served (`candidate-v9-0003b`; busted S17 2026-09-02) |
| v8 | 3 | `*_v8.py`, `schema3`, `v8_trainer*` | frozen; imported by v9 |
| v7 | 2 | `learning_contract`, `learned_policy`, `offline_trainer` | frozen; rollback target and tripwire subject |
| v6 | 1 | same modules | artifacts archived |

Retiring v7/v8 is an explicit owner pass (`.handoff/DECISIONS.md` §5.6), not a
file move.
