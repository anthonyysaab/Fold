# artifacts

Revision 2026-09-02. Nothing here deploys itself: the live runner follows
`approved.json` only, and promotion is an owner act. Schemas and formats:
`.handoff/DATA.md`. Previous map and the two folder notes it absorbed
(`candidates/README-v8-seeds.md`, `phase_b/README.md`):
`archive/docs-superseded-2026-09-02/artifacts/`.

| path | contents | tracked |
|---|---|---|
| `approved.json` | the pointer: `candidate-v9-0003b` (2026-09-01T23:48Z); `previous` `candidate-v7-0001c`; written only by `tools/promote_candidate.py` (structure + checksum + OLS gate for format 4; no gauntlet gate); POSIX `manifest_file` | yes (modified, uncommitted) |
| `candidates/` | `<v>.manifest.json` + `<v>.weights.json` per candidate; `*.approved.manifest.json` from promotions. Formats: 2 `candidate-v7-*`, 3 `candidate-v8-*`, 4 `candidate-v9-*`. `candidate-v7-0001c-gates-reverted` is the S16 bust artifact. `candidate-v9-0003b.manifest.json.pre-amendment` is the pre-promotion copy. v8 Phase-A seeds: `0001a` = seed 101, `0001` = seed 202 (the `b` member, never renamed because its sha is cited), `0001c` = seed 303 | v7/v8 yes; **v9 untracked** |
| `evaluations/` | frozen measurement records: gauntlets (`<v>-gauntlet.json` + `.stages/`), noise floor, P3 gate, field benchmark, gate decision and repairs, C-rule estimates, the 2026-09-02 v9 records. **Never edited.** `gate-ablation-2026-08-26` (no suffix) is retracted; cite `-60bb` | v8 and older yes; v9 untracked |
| `p3/` | `p3-dataset.jsonl`, `p3-fit.json` (served), `p3-fit.pre-clamp-2026-08-16.json` (kept for the frozen gate) | yes |
| `phase_a_v9/`, `phase_b_v9/` | the v9 dataset (9,084 rows) and corpora (base 40,012 decisions; postflop supplement; merged 56,043 / 139,921 rows) | **untracked** |
| `phase_a/`, `phase_b/` | the v8 dataset and corpus (schema 3 / corpus schema 1); frozen | yes |
| `training-runs/` | v7 recipes and logs, foreign-archive backfill logs, the CUDA install log; v2-era files archived | yes |
| `tiny-policy-pure.json` | legacy 125-input network `engine/poker_policy.py` loads at start-up; chooses no actions | yes |
| `corpora/` | absent: the v7 corpora are at `archive/data/corpora/`; the path stays ignored | — |
