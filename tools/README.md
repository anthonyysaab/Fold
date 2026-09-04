# tools

Revision 2026-09-03. Run as `python -m tools.<name>` from the repo root.
None is imported by the live runner. Every flag below was verified against
`--help` or the source on 2026-09-02. Long-form rationale for each tool:
`archive/docs-superseded-2026-09-02/tools/README.md` (reference only).
Procedures that chain these: `.handoff/PROCEDURES.md`.

## The v9 pipeline

| tool | does | key flags | reads → writes |
|---|---|---|---|
| `build_phase_a_dataset_v9.py` | replay archive → schema-4 supervised rows through the fitted P3 provider; renamed label keys, `to_call_zero`, `read_temperature_x10`; board rewound per decision; self-loads through the trainer | `--roots`, `--output`, `--seed`, `--equity-trials` (500), `--potential-trials`, `--workers`, `--limit`, `--dedupe`, `--chunk-rows`. **`--output` is required** — it used to default to the frozen artifact's own path, so a bare rerun overwrote the `ecb4739df9d1b9ec` oracle in place | archive → `artifacts/phase_a_v9/` |
| `phh_replay.py` | the ONE adapter turning a PHH hand into the Arena-shaped replay dict `replay_rows_v9` already consumes, so the v9 row builder runs on it unchanged; refuses antes, straddles, non-NT variants and null stacks, counting each | `replays_from_path`, `replays_from_root`, `RefusalCounter`, `PHH_REPLAY_VERSION` (library, no CLI) | `phh-dataset/` → in-memory replays |
| `build_phase_a_dataset_phh.py` | the PHH entry point to the same Phase-A row sink as the Arena builder (`PhaseARowSink`: chunk flush, per-table dedupe, k-way merge, `mtime=0` gzip, atomic replace, sidecar, trainer self-load). Sidecar records the dataset commit, the adapter version and refusals by reason. `--workers` shards the walk PER FILE (it was a no-op until 2026-09-03 — one work item per root); ids come from `phh_replay.root_table_base`, so the dataset is byte-identical at any worker count | `--roots` (default `phh-dataset/data/pluribus`), `--output`, `--seed` (7), `--equity-trials` (500), `--potential-trials` (400), `--workers`, `--limit`, `--chunk-rows` | `phh-dataset/` → `artifacts/phase_a_v9/` |
| `validate_phh_replay.py` | **the gate before believing a PHH dataset** (`PROCEDURES.md` §16): runs the adapter over every hand and checks 7 invariants — finishing stacks (exact `Decimal`), action legality against the row's own `availableActions`, pot identity, board length by street, chip conservation, refusal counters, and label availability on a seeded sample. Self-checks against a deliberately corrupted hand first and refuses to report if the corruption does not fail. `--output` is required and refuses to overwrite an existing record without `--overwrite` | `--roots`, `--output` (required), `--sample`, `--seed`, `--overwrite` | `phh-dataset/` → `artifacts/evaluations/` |
| `engine.v9_trainer` (CUDA) | Phase-A trainer. Refuses a retired corpus before it loads a row (`engine/dataset_provenance.py`) and prints the dataset's source and roots either way | **`--dataset` (required — no default; it used to default to the retired Arena corpus)**, `--allow-retired-dataset`, `--model-version`, `--init-seeds`, `--sizing-record`, recipe flags | dataset → `artifacts/candidates/` |
| `build_phase_b_corpus_v9.py` | counterfactual harvest vs P3 lineups with a v9 hero; contract-literal forcing; purity check on action and amount; schema-2 corpus; self-loads through the trainer | `--candidate`, `--corpus-name`, `--output-dir`, `--seed`, `--equity-trials` (**1000**), `--hands-scale`, `--legs`, `--harvest-workers`, `--counterfactual-rollouts`, `--p3-accept-threshold`, `--p3-resample-tries`, `--postflop`, `--street-targets` (15000,10000,6000), `--merge BASE EXTRA`, `--validate`, `--dry-run` | candidate + P3 fit → `artifacts/phase_b_v9/` |
| `engine.v9_trainer_phase_b` (CUDA) | Phase-B trainer; parity replay at export. Same retired-corpus gate as Phase A | `--phase-b-corpus`, **`--phase-a-dataset` (required)**, `--allow-retired-dataset`, `--model-version`, `--init-seeds`, `--supervised-normalization raw|constant-predictor`, loss weights, recipe flags | corpus → `artifacts/candidates/` |
| `evaluate_v8.py` | the gauntlet: seat-swapped duel vs `candidate-v7-0001c`, batteries vs MDE, `vs-p3`, trivial floors, per-street separation, self-duel null; fragments per stage | `--candidate` (format 3) or `--candidate-v9` (format 4; switches floors, stamps, study name), `--floors v8|v9|both`, `--stages`, `--workers`, `--output`, `--workdir`, `--p3-fit`, `--*-seeds`, `--scale`, `--equity-trials` (80), `--starting-stack` (6000) | → `artifacts/evaluations/` |
| `ols_baseline.py` | the k-parameter OLS baseline a v9 candidate must beat (value target and `equity_called`); self-validates on the target itself | `--corpus`, `--phase-a-dataset`, `--split-seed`, `--validation-fraction`, `--candidate` | reads only |
| `promote_candidate.py` (OWNER) | writes `<v>.approved.manifest.json` and `artifacts/approved.json` atomically; enforces the OLS gate for format 4; `--dry-run` runs every check and writes nothing (defect 26: a gate review must not re-promote); the pointer payload is built before either write, so a failure reading the old pointer cannot half-stamp a promotion; never contacts the Arena | `manifest`, `--reason` (required for a promotion), `--evaluation-note`, `--ols-gate enforce|warn|skip`, `--rollback`, `--dry-run` | → `artifacts/` |
| `head_degeneracy_audit.py` | how often a head returns its bias, bit for bit; refuses to report unless three construction-forced controls pass | `--manifest`, `--journal`, `--head`, `--output` | journal → report |
| `bench_harvest.py` | micro-benchmarks of the harvest hot path (`harvest-benchmark-2026-09-02.md`); `oracle` reruns the macro harvest N times and gates on a byte-identical corpus — compare its sha256 to `79e61dbd4edf410a` yourself, the tool checks only self-consistency | `micro` or `oracle --runs 3 --root DIR`; `--candidate` (default `candidate-v9-0001a`) | candidate + P3 fit → `%TEMP%\fold-harvest-bench` |

## Instruments and estimators (frozen; do not loosen)

| tool | does | note |
|---|---|---|
| `gate_ablation.py` | the three 2026-08-15 gate changes ablated on `vs-p3` / `vs-median` / `vs-station`, null mirror, paired stats on BB/100 and ruin | **the tripwire**: `--seeds 16 --scale 1.0 --starting-stack 6000 --equity-trials 80` must reproduce `p3-gate-2026-08-16` bit-identically after any engine edit |
| `gate_binding_audit.py` | the same three changes priced on the stored live journal | association, not a counterfactual |
| `p3_gate.py` | the held-out P3 gate: `vs-p3` vs the structural-twin `vs-median` control, own paired MDE, instrument stage first | `StrictStrengthAwareAgent` raises rather than degrading |
| `evaluate_policies.py` | the v7 gauntlet the wrappers import their statistics from; fields formats 1/2 only | `--starting-stack` 6000, `--duel-seeds`, `--ablate-sizing`, `--hybrid-min-advantage` |
| `measure_field_separation.py` | the field strength-separation benchmark on the canonical metric (+0.386) | reproduction gate and a null-holding gate before any number |
| `separation_probe.py` | canonical separation of a candidate on `vs-p3` | reference: field +0.386, v7-0001c +0.170 |
| `summarize_seed_spread.py` | three-seed spread from finished gauntlets; runs nothing | `--gauntlet SEED=PATH` |
| `estimate_escalation_shift.py`, `estimate_snap_band.py`, `ruin_damper_sweep.py` | the C4 / C3 / C5 parameter estimators | artifacts `escalation-shift-`, `snap-band-`, `ruin-damper-sweep-2026-08-29.json`; each self-tests first |
| `dead_head_experiment.py` | the closed 2026-08-27 degenerate-group retrain experiment (frozen decision rule) | reads `archive/data/corpora/`, writes `archive/data/deadhead/` |
| `build_preflop_percentiles.py` | generates `engine/preflop_percentiles.py` once (`--seed 20260816 --trials 20000`) | regenerating with other parameters is a metric change = a schema event |

## Frozen v8 / v7 builders

`build_phase_a_dataset.py` (schema 3), `build_phase_b_corpus.py` (corpus
schema 1, E6 branches), `build_p3_dataset.py` (the P3 fit: dataset + ridge
IRLS logistic, sign invariant, `--fit-only`, `--selftest-only`),
`self_play_cycle.py` (the v7 harvest + `offline_trainer`; `--examples-out/in`,
`--textured-hands`, `--on-policy`). Byte-frozen; their outputs are the frozen
instruments' inputs.

## Session forensics

| tool | does | key flags | reads → writes |
|---|---|---|---|
| `session_postmortem.py` | one live journal → a per-table chip ledger for a single `policy_version`, plus the gate rows (call gates, risk caps, all-ins, denominator collapses) and a proposed-vs-executed verdict per decision. The verdict projects the composed branch through `branch_contract_v9.branch_action` and reads all 8 override fields, so a bluff override or a rails demotion reads as `override`, never as `literal` — a `proposed_branch` → `action` frequency table is what got the first S17 draft wrong. Skips malformed lines with a count; never rewrites the journal | `--journal`, `--policy-version`, `--windows NAME:FROM:TO`, `--output` | journal → `artifacts/evaluations/` |

## Replay archive

`collect_foreign_play_data.py` (public endpoints, no credentials, ~8 req/s,
`--arena`, `--season`, `--top`, `--hands`), `reconcile_foreign_raw_data.py
--deep --report` (audit), `rebuild_foreign_corpus.py --fetch/--derive
--agent-scope archive --dry-run` (adds, never rewrites),
`audit_foreign_play_data.py <csv>` (offline summary).

## Arena utilities (credentials from `ARENA_CREDENTIALS`, else `.arena-credentials`; the key is never printed)

| tool | does |
|---|---|
| `api.py <path>...` | authenticated GET, read-only; `/api/arena/agent/me`, `/api/arena/competition/list-active` |
| `peek.py <competitionId>` | participant / runner / table state |
| `leave.py <competitionId>` | leave one competition — **the seat-release recovery** |
| `update_agent_profile.py` | `PATCH /agent/me`; dry-run unless `--apply`; quote/description default `Hello`; the handle cannot be changed |
| `submit.py`, `poll.py` | Eval-sandbox bundle submission and polling (dormant path) |

Each of the five carries its own copy of the credential resolution (only
`update_agent_profile` imports `api.py`).
