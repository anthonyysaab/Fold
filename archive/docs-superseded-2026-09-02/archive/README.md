> **ARCHIVED 2026-09-02 — SUPERSEDED. NOT STATE, NOT PLAN, NOT RULES.**
> Do not use this file to learn what the system is, what its state is, or what to do.
> The live manual is `CLAUDE.md` → `.handoff/CONTEXT.md`. Open this file only for a
> specific fact that a live document cites here by path. Never edit it, never restore it.

# Archive folder

Created by the **2026-09-02 janitor pass**. Everything here was moved out of
the working layout because nothing on the live path, in the test suite, or in
the active v9 workstream reads it. **Nothing in this folder is importable and
nothing here is a promotion, a deployment, or a served artifact.**

Frozen reports and `.handoff/` records still cite the ORIGINAL paths. That is
deliberate — frozen records are never edited. Map a cited path to its new home
with the tables below; the relative layout under each half is preserved, so
`artifacts/candidates/candidate-v2-0016.manifest.json` is now
`archive/pre-reset-2026-08-16/artifacts/candidates/candidate-v2-0016.manifest.json`.

## Two halves

| half | tracked? | how it was moved | how to restore |
|---|---|---|---|
| `pre-reset-2026-08-16/` | yes — `git mv`, history intact (`git log --follow`) | `git mv` | `git mv` it back |
| `data/` | **no** — gitignored (`.gitignore`: `archive/data/`), ~777 MB | plain `mv` | `mv` it back |

## `pre-reset-2026-08-16/` — code and artifacts of the retired architecture

The project was reset on 2026-08-16 (`.handoff/notes/PROTOTYPE_POSTMORTEM.md`).
These belong to the architecture that reset discarded and had no remaining
reader.

| moved | from | why |
|---|---|---|
| `engine/torch_network.py`, `engine/torch_policy.py` | `engine/` | Legacy PyTorch checkpoint adapter. Import graph on 2026-09-02: imported by nothing except each other, no tests, never on the serve path (`deploy/devfun-arena/build_bundle.py` already listed them as training-only drops). `.handoff/notes/SCRIPT_REVIEW.md` verdict: "replace or remove". |
| 22 files: `candidate-foreign-warmstart-0002`, `candidate-mixed-0003..0005`, `candidate-v2-0006..0008`, `candidate-v2-0013..0016` (`.manifest.json` + `.weights.json`) | `artifacts/candidates/` | Format-1 ("v6") candidates from before the reset. No test loads them; the only code mention was a usage example in `tools/self_play_cycle.py`, now updated. |
| 18 files: `candidate-v2-*-gauntlet.json` and their `.err.txt` companions (all zero bytes) | `artifacts/evaluations/` | The gauntlet reports of those candidates; kept beside them so the pair stays citable. |
| 34 files: `candidate-v2-0007..0016.*` launch recipes (`.ps1`), logs, `.pid`, `.status.txt` | `artifacts/training-runs/` | The v2-era training and pipeline logs. The v7 logs, the foreign-backfill logs and the torch install log stay in place: v7 is the frozen instruments' subject and the backfills are the replay archive's provenance. |

Not moved, on purpose: `candidate-v7-*` (frozen-instrument subject and the
rollback target in `artifacts/approved.json`'s `previous`), `candidate-v8-*`
(the v9 modules still import the v8 modules), and every `candidate-v9-*`.

## `data/` — gitignored bulk

| moved | from | why | who still points at it |
|---|---|---|---|
| `corpora/` (729 MB: `candidate-v7-000{1,2,3}.corpus` + `.meta.json`) | `artifacts/corpora/` | The three v7 harvest corpora. The v7 line is frozen; the v9 corpora are `artifacts/phase_b_v9/`. Reproducible from the recipes in `artifacts/training-runs/`. | `tools/dead_head_experiment.py` (constant updated); `.handoff/notes/evidence/2026-08-16-v8-design/` probes and `artifacts/evaluations/v9-loses-to-v7-diagnosis-2026-09-02.md` cite the old path (frozen). |
| `deadhead/` (35 MB) | `artifacts/deadhead/` | The 15 retrain candidates of the closed 2026-08-27 dead-head experiment; reproducible from its pre-registration. The result that matters, `artifacts/evaluations/dead-head-retrain-2026-08-27.json`, is tracked and untouched. | `tools/dead_head_experiment.py` and the `tools/separation_probe.py` usage example (both updated). |
| `eval-bundle-2026-08-14/` (`build/` + `bundle.zip`, 634 KB) | repo root | A generated Eval-sandbox bundle from 2026-08-14 still carrying the pre-rename package name `devfun_poker_playground`, so it could not be resubmitted as-is. `python deploy/devfun-arena/build_bundle.py` regenerates a current one at the root. | nothing |
| `antislop-audit-2026-08-12/` (181 KB) | `.antislop/` | An external code audit of the pre-reset tree at commit `0aab9e9`. Its `.gitignore` line was removed. | nothing |
| `arena-training.jsonl.backup-2026-08-26` (12 MB) | repo root | The journal backup taken before the 2026-08-26 bust-day repair. The live journal `.arena-training.jsonl` is untouched. | `.handoff/PENDING_EDITS.md` §18 (frozen record). |

## Deliberately NOT archived

- **The v7/v8 serve arms, trainers, builders and their freeze-guard tests**
  (`engine/offline_trainer.py`, `learned_policy.py`'s format-1/2 arms,
  `learned_policy_v8.py`, `feature_extract_v8.py`, `schema3.py`,
  `v8_trainer*.py`, `tools/build_phase_a_dataset.py`,
  `tools/build_phase_b_corpus.py`, `tools/self_play_cycle.py`,
  `tools/evaluate_policies.py`, …). `.handoff/notes/V9_RESTRUCTURE_PLAN.md`
  "Post-promotion cleanup" makes their retirement an explicit owner-run pass,
  and it is not a file move: `learned_policy_v9` imports `learned_policy_v8`
  and `v8_trainer`, `feature_extract_v9` imports `feature_extract_v8`,
  `schema4` imports `schema3`, `learned_policy` needs `offline_trainer` to
  serve the format-2 rollback target, and the frozen tripwires
  (`gate_ablation`, `p3_gate`) need their v7 subject.
- **`deploy/`, `tools/submit.py`, `tools/poll.py`** — the Eval-sandbox
  submission path. Dormant since the reset, but documented, tested for
  layout, and already ported to the `engine` package name.
- **`bluff.py`, `lead_position.py`, `risk_temperature.py`** at the root — they
  look pre-reset but `engine/decision_engine.py` imports all three on the
  live path.
- **`artifacts/tiny-policy-pure.json`** — `engine/poker_policy.py` still loads
  and validates it at start-up.
- **`artifacts/p3/p3-fit.pre-clamp-2026-08-16.json`** — kept so the frozen
  16-seed P3 gate report stays reproducible (`.handoff/STATUS.md`).
- **`artifacts/candidates/candidate-v9-0003b.manifest.json.pre-amendment`** —
  written by the 2026-09-02 promotion amendment; part of in-flight work.
- The zero-byte `.err.txt` companions of the v7/v8 frozen gauntlet reports.
- Anything under `.handoff/` — gitignored and irreversible; not this pass's
  business.
