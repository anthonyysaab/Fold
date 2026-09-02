# v9 Phase-B training-schedule sweep — PRE-REGISTERED 2026-09-02

**Status: pre-registration. Written BEFORE the runs, per the standing
rule (dead-head-retrain-prereg-2026-08-27.md). Nothing below may be
re-interpreted after the numbers land; the decision criteria are frozen
now.**

## The finding this experiment is about

`v9-loses-to-v7-diagnosis-2026-09-02.md` §2 ("Cause A"): the best
validation checkpoint of every v9 candidate lands at or within a few
steps of the warmup boundary (warmup_steps = 200). `candidate-v9-0003b`'s
manifest states it outright: **best_epoch 1, epochs_run 11** — early stop
fired after 10 stale epochs, and the cosine schedule had decayed the
learning rate by ~1% in that window. The network's validation loss rises
monotonically from the moment the LR reaches peak, at every corpus size
and both phases. A 6x larger corpus moved best_epoch DOWN, which is not
data-starvation behaviour. The 71,841-parameter network loses 3.4x to a
2-parameter OLS on held-out data.

## Hypothesis under test

The peak learning rate (1e-3, AdamW betas 0.9/0.95, eps 1e-8) is too hot
for this objective, so the model only improves while the warmup ramp
holds the LR down and every subsequent epoch damages validation past
recovery. The cosine decay never engages because early stopping cuts the
run at ~7% of the schedule.

## Instrument (frozen)

- Trainer: `python -m engine.v9_trainer_phase_b` on the CUDA venv
  (`C:\Users\user\poker-nn-training\.venv\Scripts\python.exe`).
- Corpus: `artifacts/phase_b_v9/candidate-v9-phase-b-merged.phase-b.jsonl.gz`
  (56,043 decisions) + `artifacts/phase_a_v9/phase-a-dataset-v9.jsonl.gz`
  (9,084 rows) — the exact inputs of the 0003 family.
- `--split-seed 17` (the 0003 split), `--validation-fraction 0.1`,
  `--init-seeds 401 402 403` for every arm.
- Outputs to `artifacts/evaluations/schedule-sweep-2026-09-02/`.
- Reported per arm: best_epoch, epochs_run, optimizer_steps,
  validation `value_normalized` and `value_mse`, residual share.

## Arms

| arm | learning rate | batch size | everything else |
|---|---|---|---|
| `control` | 1e-3 | 256 | shipped defaults (epochs 150, warmup 200, patience 10, wd 0.01, residual wd 0.1) |
| `lr-3e-4` | 3e-4 | 256 | same |
| `lr-1e-4` | 1e-4 | 256 | same |
| `lr-3e-5` | 3e-5 | 256 | same |
| `batch-2048` | 1e-3 | 2048 | same |

## Decision criteria — frozen before the runs

With `steps_per_epoch` = 198 at batch 256 and 25 at batch 2048, and
warmup = 200 steps:

1. **"The schedule was the problem"** if any arm shows, in at least 2 of
   its 3 seeds, BOTH of:
   - the best checkpoint lands **more than one full epoch past the warmup
     boundary** (best_epoch > ceil(200 / steps_per_epoch) + 1), i.e. the
     model keeps improving while training at its decayed rate; and
   - validation `value_normalized` beats the control arm's per-seed
     median by **> 0.02** (control expectation ≈ 0.944, the 0003b value)
     or equivalently `value_mse` falls by the same proportion.
2. If the best checkpoint still lands at the warmup boundary at every
   learning rate (3 of 3 arms below control), **the schedule alone is
   not the fix** — the objective or the data is, and the next gate is
   the OLS baseline (a k-parameter linear model must not beat the
   network on the same split).
3. The control arm is a sanity check on the instrument: it must
   reproduce the shipped pattern (best_epoch ≈ 1, early stop ≈ epoch 11)
   within one epoch, or the sweep's environment differs from the one
   that produced the 0003 family and every comparison is void.

## What happens with the answer

- Criterion 1 met → adopt the winning learning rate as the v9 trainer
  default (a training-side default, not a serve change), retrain the
  0003-family seeds, and re-run the OLS comparison and the gauntlet.
- Criterion 2 → leave the schedule alone; proceed to the objective and
  the OLS gate. A schedule that is not the problem must not be edited
  into looking like the fix.

---

## RESULTS (2026-09-02, appended after the runs)

All five arms ran on the CUDA venv with the frozen instrument; the
control arm reproduced the shipped pattern exactly (best_epoch 1,
epochs_run 11, validation value_normalized 0.9437 vs 0003b's 0.94368) —
criterion 3 holds, so the sweep environment is the one that produced
the 0003 family and every comparison below is valid.

| arm | best_epoch (seeds) | epochs_run | val value_normalized |
|---|---|---|---|
| `control` (1e-3, bs 256) | 1 / 1 / 1 | 11 | **0.9437** |
| `lr-3e-4` | 2 / 3 / 2 | 12-13 | 0.9373 |
| `lr-1e-4` | 7 / 6 / 6 | 16-17 | 0.9364 |
| `lr-3e-5` | 23 / 20 / 20 | 30-33 | 0.9363 |
| `batch-2048` (1e-3) | 9 / 7 / 7 | 17-19 | 0.9360 |

**Criterion 1(a)** — best checkpoint lands more than one full epoch past
the warmup boundary — is met by `lr-1e-4` and `lr-3e-5` (all three
seeds). The lower the peak rate, the longer the model improves past
warmup, exactly the schedule mechanism the hypothesis named.

**Criterion 1(b)** — validation `value_normalized` beating the control
by > 0.02 — is met by NO arm. Every arm lands at 0.936-0.937, marginally
WORSE than the control's 0.9437.

**VERDICT: criterion 2 fires — the schedule alone is not the fix.**
Slowing the optimizer moves the best checkpoint later without buying
generalization: the value head's ceiling sits at ~0.94 normalized
(~6% of target variance) at every learning rate, batch size, and
optimizer-step count tried. The frozen pre-registration says the
schedule "must not be edited into looking like the fix", so the
shipped defaults stay. The next gate is the OLS baseline, now built as
a permanent tool and a promotion gate:

- Phase-B composed value (the network's actual objective): the network
  (0003b: 0.9437) DOES beat the 23-parameter branch-interacted OLS
  (1.045) — margin 0.10.
- Phase-A `equity_called` label (the diagnosis's comparison, reproduced
  bit-for-bit by `tools/ols_baseline.py`: OLS R2 0.198 on the same 726
  validation rows): the network (R2 0.051) loses 3.9x.
- `tools/promote_candidate.py` now refuses format-4 candidates that
  fail either arm (`--ols-gate enforce`, the default; `warn` / `skip`
  are explicit owner overrides). Every current v9 candidate fails the
  Phase-A arm, exactly as the diagnosis predicted.

