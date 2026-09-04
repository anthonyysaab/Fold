# engine

Revision 2026-09-04. Stdlib-only on the serve path: every `torch` import is
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
| `poker_policy.py` | heuristic policies (`--standard` / `--aggressive`), the fallback when nothing is approved. Holds no model and reads nothing from disk: P2 retired the 125-input `tiny-policy-pure.json` load path on 2026-09-04 |
| `learned_policy.py` | `load_approved`; format-1/2 runtime; dispatches format 4 to `learned_policy_v9` |
| `learned_policy_v9.py` | **the served runtime**: composition, projection, format-4 loader, P3 belief provider, hyper roll 0.0 |
| `aggression_sizing.py` | g — the one sizing function |
| `branch_contract_v9.py` | the four-branch contract (normative) |
| `feature_extract_v9.py`, `schema4.py` | the 414-input vector (frozen); built on `feature_extract_v8.py`, `schema3.py`, `action_history.py` |
| `belief_provider.py`, `p3_belief_provider.py` | belief-bucket interface; the fitted P3 posterior |
| `strength_aware_opponent.py` | the P3 fold model and battery opponent |
| `rules/` | C1–C5, wired dark (`rules/README.md`) |
| `training_telemetry.py` | the journal (schema 3) and `TrainingExample` |
| `learning_contract.py`, `policy_features.py` | the v7 contract and manifest validator every format passes; the 125-name schema-2 base block every schema-3/4 vector is composed from, and the frozen action labels |
| `forward_kernel.py` | the pure-Python forward pass: `_dot`, `_forward`, `_forward_v2`, `_layer_norm`, `_sigmoid`, `_softmax`. Imports nothing from `engine`, so it can never join a cycle |
| `architecture_v8.py` | the composed-value architecture contract (widths, head sizes, branch tuples) and both fail-closed validators. Sibling to `branch_contract_v9.py`: that says WHICH branches exist, this says what SHAPE scores them |
| `_vendor/treys/` | vendored hand evaluator |

Root modules imported on every decision: `bluff.py`, `lead_position.py`,
`risk_temperature.py`.

## Training is not in this package

Moved to `training/` on 2026-09-04 (`v9_trainer.py`, `v9_trainer_phase_b.py`,
`v8_trainer*.py`, `offline_trainer.py`, `supervised_loss_normalization_v9.py`,
`dataset_provenance.py`). Run them as `python -m training.<name>` on the CUDA
interpreter; `tools/interpreters.py` holds its path.

Nothing in `engine/` imports `training/`, and that is the invariant to keep:
the dependency runs one way only. It did not before the move — the serve path
reached into `offline_trainer` for arithmetic and into `v8_trainer` for its own
architecture description, and `learning_contract` had to defer an import to
break a genuine cycle. `forward_kernel.py` and `architecture_v8.py` above are
where those two things live now.

`table_simulator.py` and `foreign_data.py` stay here: the simulator is the
harvest AND the gauntlet's table, and both are stdlib-only.
`training_telemetry.py` stays too, despite the name — `action_history.py` and
`feature_extract_v8.py` import it on the serve path.

## Lines

| line | format | modules | status |
|---|---|---|---|
| v9 | 4 | `*_v9.py`, `schema4`, `aggression_sizing`, `branch_contract_v9`, `rules/` | served (`candidate-v9-0003b`; busted S17 2026-09-02) |
| v8 | 3 | `*_v8.py`, `schema3`, `architecture_v8`, `training/v8_trainer*` | frozen; imported by v9 |
| v7 | 2 | `learning_contract`, `learned_policy`, `forward_kernel`, `training/offline_trainer` | frozen; rollback target and tripwire subject |
| v6 | 1 | same modules | artifacts archived |

Retiring v7/v8 is an explicit owner pass (`.handoff/DECISIONS.md` §5.6), not a
file move.
