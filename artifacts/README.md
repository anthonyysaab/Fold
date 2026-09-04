# artifacts

Revision 2026-09-04. Nothing here deploys itself: the live runner follows
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
| `phase_a_v9/`, `phase_b_v9/` | **PHH/Pluribus since 2026-09-04** (`DATA.md` §1.1). Dataset `phase-a-dataset-v9-pluribus-2026-09-04.jsonl.gz`, **91,356 rows** from 10,000 hands; its `.summary.json` sidecar is the provenance record `engine/dataset_provenance.py` reads, and is tracked. Corpora `phh-0004a-scaled*`: base 39,797 decisions, postflop supplement 17,228, **merged 57,025 decisions / 141,373 branch rows**. `phh-0004a*` without `-scaled` is a default-scale run kept as evidence for `PENDING_EDITS.md` row 30 — do NOT train on it. The retired Arena build `phase-a-dataset-v9.jsonl.gz` (9,084 rows) is still present and is REFUSED by the provenance gate; the older v9 corpora (base 40,012 / merged 56,043 / 139,921 rows) were built from it | `.jsonl.gz` **untracked**; the `.summary.json` sidecars tracked |
| `phase_a/`, `phase_b/` | the v8 dataset and corpus (schema 3 / corpus schema 1); frozen | yes |
| `training-runs/` | v7 recipes and logs, foreign-archive backfill logs, the CUDA install log; v2-era files archived | yes |
| `corpora/` | absent: the v7 corpora are at `archive/data/corpora/`; the path stays ignored | — |
