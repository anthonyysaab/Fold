# tests

Revision 2026-09-03. `python -m pytest tests/ -q` with the stdlib interpreter
is the source of truth: **1145 passed / 29 skipped / 370 subtests** (~2.5 min),
measured 2026-09-03 after the PHH build run. The 29 skips are 21
torch-availability and 8 that read the quarantined Arena archive.

**Both skip classes were checked, not assumed.** The torch half was RUN on
the CUDA interpreter on 2026-09-03: 104 passed / 2 skipped over the seven
torch files, and those last 2 skipped because their guard hard-coded
`C:/Users/user/poker-nn-training/.venv/...`, a path that stopped existing
when the training repo moved into this tree — they would have skipped
forever, on every interpreter, silently. The guard is repo-relative now.
There are **no pokerkit skips on this machine**: `pokerkit` 0.7.4 is
installed (`requirements-tools.txt`), so the PHH tests really run; on a
machine without it the whole new adapter surface would go green while
testing nothing (16 `skipUnless` guards across the three PHH files). A skip
guard is a claim about the environment, and it decays like any other.

Nothing here contacts the Arena. **A green suite is not evidence of safety on engine code** —
sweep adversarially and run the tripwire (`.handoff/PROCEDURES.md` §7.3).
`tests/test_summarize_seed_spread.py` is pytest-only; every other file is
`unittest`-compatible. Previous map: `archive/docs-superseded-2026-09-02/tests/README.md`.

| area | files | covers |
|---|---|---|
| live path, money | `test_live_session`, `test_runner_safety`, `test_play_entrypoints`, `test_runtime_layout` | free-Playground guard, hard stops, release paths; deadline and reconnect handling; launchers serve what is deployed; stdlib-only import |
| gates and shaping | `test_safety_gates`, `test_temperature_shaping`, `test_hyper_aggression`, `test_policy_fixes`, `test_v9_engine_coupling` | `SafetyGates` ranges and round-trips, discriminating cases for the three dark repairs; bounded temperature; the dice roll; thresholds; L5 hardened catch-alls and the gated shove lane |
| advisors and reads | `test_bluff`, `test_bluff_wiring`, `test_opponent_model`, `test_game_state`, `test_lead_position`, `test_risk_temperature` | bluff advisor and its passive-spot wiring; aggression tracking and the perma-shove remedy; snapshot validation; the two root gauges |
| equity and metric | `test_hand_strength`, `test_hand_potential`, `test_strength_metric`, `test_treys_evaluator` | MC estimator boundaries; Ppot/Npot; the canonical metric; evaluator vs a brute-force oracle |
| simulator and journal | `test_table_simulator`, `test_training_telemetry`, `test_action_history` | chip conservation, side pots, determinism, counterfactual replay; append-only journal; history encoder |
| P3 and beliefs | `test_strength_aware_opponent`, `test_p3_audit`, `test_belief_provider`, `test_p3_belief_provider`, `test_p3_gate` | the fit's invariants on the shipped artifact; the posterior; the `vs-p3` gate |
| rule layer | `test_rules_composition` | zero-diff fuzz, disjoint firing, damper supremacy, verdict telemetry |
| v9 line | `test_learned_policy_v9`, `test_feature_extract_v9`, `test_v9_trainer`, `test_v9_trainer_phase_b`, `test_supervised_loss_normalization_v9`, `test_build_phase_a_dataset_v9`, `test_build_phase_b_corpus_v9`, `test_ols_baseline`, `test_evaluate_v8`, `test_head_degeneracy_audit` | composition, projection, format-4 loader and stamps; schema-4 invariants; both trainers (torch parts skip on stdlib); the Phase-1 knob; the builder (board look-ahead regression) and harvester (owner pins, purity, merge); the OLS gate; the gauntlet wrapper and floors; the degeneracy detector |
| PHH path | `test_phh_replay`, `test_build_phase_a_dataset_phh`, `test_validate_phh_replay` | the adapter against inline PHH hands and the real Pluribus clone (event order, blinds, action mapping, `.phhs` multi-hand ids, refusals, finishing-stack equality); the PHH entry point to the shared row sink (byte-identical rerun, sidecar keys); the validation gate — its self-check must FAIL on a deliberately corrupted hand before any report is written, and the half-chip-split class must refuse a whole-chip delta, a non-fractional file stack and a non-winner seat |
| dataset provenance | `test_dataset_provenance` | the retired-corpus gate: a PHH sidecar passes, the Arena shape (no `generator.source`, roots in the quarantined archive) is refused with its own recovery in the message, an unknown source is refused, and a missing or unparsable sidecar fails CLOSED. Plus the two argparse pins that fail on the unfixed code — neither trainer may carry a dataset default |
| session forensics | `test_session_postmortem` | the journal-to-ledger reader on a synthetic journal with known sums; `OverrideDetectionTests` is the regression for the 2026-09-03 bluff-blindness defect — a row shaped like the S17 bust preflop (`proposed_branch` active, price 3, executed raise, `bluff_kind` steal) must report `override`, never `literal` |
| v8 line (frozen) | `test_learned_policy_v8`, `test_feature_extract_v8`, `test_schema3`, `test_v8_parity`, `test_v8_trainer`, `test_v8_trainer_phase_b`, `test_build_phase_b_corpus`, `test_phase_a_dataset`, `test_summarize_seed_spread` | freeze guards |
| v7 line and instruments | `test_learned_policy`, `test_learning_contract`, `test_offline_trainer`, `test_v7_core`, `test_cuda_trainer`, `test_degenerate_group_filter`, `test_evaluate_policies`, `test_gate_binding_audit`, `test_self_play_cycle`, `test_harvest_parallelism`, `test_measure_field_separation` | promotion, rollback and the defect-26 write property (reads `candidate-v7-0001c`; a real OLS-gate FAIL on a synthetic corpus, a corrupt-pointer half-stamp, the `--dry-run` verb against the same inputs promoted for real, and one injected clock for the shared stamp); the 142-input contract; the v7 trainer and forward pass; the gauntlet statistics; the journal audit; the v7 harvest and its parallel equivalence; the field benchmark instrument |
| replay archive | `test_collect_foreign_play_data`, `test_foreign_corpus_rebuild`, `test_foreign_data` | receipts and state reconstruction; reconcile/rebuild on a fixture with every gap class; the CSV training boundary |

65 files. `ruff check .` reports 7 pre-existing errors (three of them in
this folder); not a regression.
