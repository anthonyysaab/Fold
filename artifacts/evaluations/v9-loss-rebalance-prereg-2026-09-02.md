# v9 Phase-B supervised-loss rebalance — PRE-REGISTERED 2026-09-02

**Status: pre-registration. Written BEFORE the runs, per the standing
rule (`v9-schedule-sweep-prereg-2026-09-02.md`, `dead-head-retrain-prereg-2026-08-27.md`).
The decision criteria below are frozen now and may not be re-read after
the numbers land.**

This is **Phase 1** of `v9-next-layer-plan-2026-09-02.md` ("fix the
optimization, no architecture change"). Phase 0 (the out-of-distribution
instrument) is not a prerequisite for this experiment: its readout is the
OLS promotion gate, which already exists and is enforced.

## The finding this experiment is about

`v9-next-layer-plan-2026-09-02.md` §0: the Phase-B loss budget is
inverted. The composed-value term is divided by the training-target
variance (so 1.0 means the constant predictor), but the three Phase-A
supervised terms are summed raw (`engine/v9_trainer_phase_b.py:1033`).
On the schedule-sweep control arm (seed 401, the shipped defaults) the
validation loss decomposes as:

| term | validation loss | share of total |
|---|---|---|
| `range` (8-way cross-entropy; output read NOWHERE at decision time) | 1.577 | **48.4%** |
| `fold_through` (binary cross-entropy, 2 lanes) | 0.691 | 21.2% |
| composed value (variance-normalized) | 0.944 | 29.0% |
| `equity_called` (MSE on a [0,1] label) | 0.0448 | **1.4%** |

Nobody chose 48/21/29/1.4; it fell out of the natural scale of each
loss (ln 8 = 2.08 for an 8-way NLL, ln 2 = 0.69 for a BCE, ~0.048 for
an MSE on a label with that variance). The head that carries 1.4% of
the loss is the one that loses 3.9x to a 2-parameter OLS on
`equity_vs_posterior` — and it is the head the promotion gate refuses
every current v9 candidate on.

A second observation, recorded now so it is not read into the result
later: `fold_through`'s validation BCE of 0.691 is ln 2 to two decimals,
i.e. that head too sits at roughly the constant predictor.

## Hypothesis under test

The `equity_called` head is not learning because its gradient is
swamped, not because the label or the feature is wrong. Normalizing
every supervised term by its own constant-predictor loss on the
training split (the same treatment the value term already gets) puts
the four terms on equal footing and lets the head learn at least what
two linear parameters learn.

## The change (code, landed before the runs)

- New module `engine/supervised_loss_normalization_v9.py`: the
  constant-predictor baselines (a bias-only head's masked loss on the
  Phase-A **training** split — per-lane fold-through marginal, range
  bucket marginal, per-slot equity mean), the `SupervisedLossConfigV9`
  knob (normalization mode + one weight per head), and its checker.
- `engine/v9_trainer_phase_b.py`: the supervised sum becomes
  `Σ_h weight_h · loss_h / baseline_h`; in `raw` mode every baseline is
  1.0 and every weight 1.0, which is the shipped objective unchanged.
  The manifest stamps the mode, the measured baselines, the weights and
  the effective scales under `training.loss.supervised_normalization`;
  `evaluation.*_losses` keeps the RAW per-head numbers (the OLS gate
  reads `equity_called` as a raw MSE) and adds the normalized ones.
- CLI: `--supervised-normalization {raw,constant-predictor}` (default
  `raw` — the default does not move until this experiment says so),
  `--fold-through-loss-weight`, `--range-loss-weight`,
  `--equity-called-loss-weight` (default 1.0 each).
- The Phase-A trainer (`engine/v9_trainer.py`) is NOT touched: its
  candidates are not deployable and are not the subject here.

## Instrument (frozen)

- Trainer: `python -m engine.v9_trainer_phase_b` on the CUDA venv
  (`C:\Users\user\poker-nn-training\.venv\Scripts\python.exe`).
- Corpus: `artifacts/phase_b_v9/candidate-v9-phase-b-merged.phase-b.jsonl.gz`
  (56,043 decisions) + `artifacts/phase_a_v9/phase-a-dataset-v9.jsonl.gz`
  (9,084 rows) — the exact inputs of the 0003 family and the sweep.
- `--split-seed 17`, `--validation-fraction 0.1`, `--init-seeds 401`
  for every arm (one seed, paired against the sweep control's seed 401),
  everything else the shipped defaults (lr 1e-3, batch 256, epochs 150,
  warmup 200, patience 10, wd 0.01, residual wd 0.1).
- Outputs to `artifacts/evaluations/loss-rebalance-2026-09-02/`.
- Gate arithmetic, measured today by `python -m tools.ols_baseline
  --phase-a-dataset artifacts/phase_a_v9/phase-a-dataset-v9.jsonl.gz
  --split-seed 17` (identity self-check PASS):
  - Phase-A arm: OLS R² **0.1979** on 726 validation rows whose target
    variance is **0.048467**. The gate passes when the network's R²
    exceeds 0.1979 − 0.02 = **0.1779**, i.e. when its validation
    `equity_called` MSE is below 0.048467 × (1 − 0.1779) =
    **0.039845**. Control seed 401 scores 0.044799 (R² 0.0757).
  - Phase-B arm: `branch-strength` OLS `value_normalized` **1.045**; the
    gate passes below 1.045 − 0.02 = **1.025**. Control: 0.943691.

## Arms

| arm | `--supervised-normalization` | fold_through w | range w | equity_called w |
|---|---|---|---|---|
| `rebalance-control` | `raw` | 1 | 1 | 1 |
| `rebalance-equal` | `constant-predictor` | 1 | 1 | 1 |
| `rebalance-range-quarter` | `constant-predictor` | 1 | 0.25 | 1 |
| `rebalance-no-range` | `constant-predictor` | 1 | 0 | 1 |

`equal` is the hypothesis. `range-quarter` and `no-range` answer the
plan's open question ("should `range` carry half the gradient at
all?") — the auxiliary task is legitimate trunk shaping but auxiliaries
are normally down-weighted, not dominant.

## Decision criteria — frozen before the runs

1. **Instrument check.** `rebalance-control` runs the NEW code in `raw`
   mode and must reproduce the sweep control's seed-401 numbers —
   validation `value_normalized` 0.943691, `equity_called` 0.044799,
   `best_epoch` 1 — within **1e-4** on both losses. If it does not, the
   refactor changed the raw path and every comparison below is void;
   fix the code before reading any other arm.
2. **"The weighting was the problem for `equity_called`"** if an arm's
   validation `equity_called` MSE is below **0.039845** (network R² >
   0.1779), confirmed afterwards by `tools.ols_baseline --candidate`
   reporting the Phase-A arm as beaten. This is exactly the promotion
   gate's own arithmetic; nothing here is a new threshold.
3. **Value guard.** The same arm must keep validation `value_normalized`
   below **1.025** (the Phase-B gate arm) AND no more than **0.02**
   worse than control (i.e. below 0.963691). An arm that buys the
   equity head with the value head is a trade, reported as such, not a
   pass.
4. **Adoption.** If at least one arm meets 2 and 3, the arm with the
   best validation equity R² among those becomes the v9 Phase-B
   default (mode and weights). Arms within 0.02 R² of each other tie,
   and a tie goes to the arm with the fewest hand-chosen constants:
   `equal` > `range-quarter` > `no-range`. The adopted arm is then run
   on seeds 402 and 403; adoption stands only if **2 of 3 seeds** meet
   criterion 2 (the sweep's own rule). Seed spread is reported either
   way.
5. **Falsification.** If no arm meets criterion 2, the weighting is not
   the problem and the plan's §2 decision point fires: the next step is
   the label/feature investigation, and NO further weight tuning is
   done on this objective. Pre-registered diagnostic to localize it,
   read from the same manifests: if the TRAIN-split `equity_called`
   MSE falls below 0.0398 while validation does not, the head fits the
   label but does not generalize (a data/coverage problem, Phase 2);
   if train does not fall either with ~20x the gradient, the head
   cannot fit the label from these inputs (a label/feature mismatch —
   check what the Phase-A `equity_called` label actually measures
   against what `equity_vs_posterior` measures before anything else).

## What happens with the answer

- Criterion 4 met → flip the trainer default, retrain the 0003-family
  seeds under it, re-run the OLS gate, then the gauntlet — with the
  standing caveat that the battery is in-distribution and the v7 duel
  is the only out-of-distribution number until Phase 0 lands.
- Criterion 5 → leave the default at `raw`; the knob stays as an
  ablation, documented as a null result. A weighting that is not the
  problem must not be edited into looking like the fix.

---

## RESULTS (2026-09-02, appended after the runs)

All four arms ran on the CUDA venv with the frozen instrument (seed 401,
split 17, shipped schedule); train log and manifests in
`artifacts/evaluations/loss-rebalance-2026-09-02/`, the CLI gate run on
the best arm in `ols-rebalance-equal.json` there.

**Criterion 1 HOLDS, exactly.** `rebalance-control` (new code, `raw`
mode) reproduces the sweep control's seed-401 numbers with zero delta:
validation `value_normalized` 0.943691, `equity_called` 0.044799,
best_epoch 1, epochs_run 11. The raw path is unchanged and every
comparison below is valid.

The measured training-split baselines (identical in every arm, as they
must be — they are a property of the split, not the run):
`fold_through` 0.677245 over 1,807 lane-rows, `range` 1.657728 over
7,458 rows, `equity_called` 0.053790 over 7,458 rows. Under `equal`
the effective scales are fold_through ×1.48, range ×0.60,
equity_called ×18.6.

| arm | mode / range w | best_epoch | epochs_run | val `equity_called` MSE | val R² | **train** `equity_called` MSE | val `value_normalized` | val `fold_through` | val `range` |
|---|---|---|---|---|---|---|---|---|---|
| `rebalance-control` | raw | 1 | 11 | 0.044799 | 0.0757 | 0.048847 | **0.943691** | 0.6909 | 1.5770 |
| `rebalance-equal` | c-p / 1.0 | 1 | 11 | **0.040758** | **0.1591** | 0.033254 | 0.948859 | 0.6874 | 1.5867 |
| `rebalance-range-quarter` | c-p / 0.25 | 1 | 11 | 0.041465 | 0.1445 | 0.033054 | 0.952577 | 0.6886 | 1.5926 |
| `rebalance-no-range` | c-p / 0 | 1 | 11 | 0.041600 | 0.1417 | 0.033262 | 0.951216 | 0.6889 | 2.0794 |

Bar for criterion 2: validation MSE < 0.039845 (R² > 0.1779).

**Criterion 2 — NOT met by any arm.** The best arm, `equal`, reaches
R² 0.1591: more than double the control's 0.0757, and short of the bar
by 0.019. `tools.ols_baseline --candidate` on it reports the Phase-A
arm as still lost (network 0.1591 vs OLS 0.1979) and the Phase-B arm
as passed (0.9489 vs 1.045).

**Criterion 3 — met by all three normalized arms.** `value_normalized`
0.949-0.953, each 0.005-0.009 worse than control, inside the 0.02
margin; all far below the 1.025 gate arm.

**Criterion 4 — not reached.** The default stays `raw`.

**VERDICT: criterion 5 fires — the weighting is not the problem, or
not the whole of it.** Per the frozen text, no further weight tuning on
this objective. The pre-registered diagnostic localizes it: in every
normalized arm the TRAIN-split `equity_called` MSE falls to 0.033
(train R² 0.38 against the train baseline 0.0538) while validation
stays at 0.041 — **the head fits the label and does not generalize.**
That is the data/coverage side, so the plan's §2 decision point resolves
toward Phase 2; the label/feature check is not what this readout
indicates.

Observations recorded beside the verdict (not criteria, not
re-interpretations):

- **The checkpoint pattern is untouched by the loss budget.** Every arm
  still selects epoch 1 and stops at epoch 11. Under `equal` no term
  dominates the total (value 0.95, fold_through 1.02, range 0.96,
  equity 0.76 at the constant-predictor scale) and the best checkpoint
  still lands at the warmup boundary.
- **A mechanism for the overfit, visible in the trainer:** Phase-A
  batches (256 rows) interleave on EVERY optimizer step, so one Phase-B
  epoch (204 steps at batch 256 on 50,459 decisions) is ~7 passes over
  the 7,458 labelled Phase-A training rows. The supervised heads see
  their whole dataset seven times before the first validation read.
  A 125× Phase-A widening (Phase 2) changes exactly this ratio.
- **The `range` auxiliary neither helps nor hurts the gated head.**
  Removing it entirely (`no-range`: the head stays at uniform, val NLL
  2.0794 = ln 8) costs the value head nothing visible (0.951 vs 0.949 /
  0.953) and gives the equity head nothing (0.1417 vs 0.1591). Weight
  0.25 sits between. The question the plan asked is answered: it is
  inert on these numbers.
- **`fold_through` is at, or fractionally worse than, its constant
  predictor** in every arm: validation BCE 0.687-0.691 against a
  training-split baseline of 0.677 (normalized 1.015-1.020). Only
  1,807 lane-rows carry the label. The Phase-3a response head replaces
  this head, so this is a note for that design, not a repair to make.

**What happens next, per the pre-registration:** the trainer default
stays `raw`; `constant-predictor` remains as a documented, tested
ablation. **Recommendation for the owner (a recommendation, not an
action taken):** carry `--supervised-normalization constant-predictor`
with unit weights as the experiment arm into Phase 2 — it is the
principled objective (all four terms at 1.0 = constant predictor) and
it moved the gated head in the right direction on the same data; it
should be re-read against the same bar once the widened Phase-A dataset
exists, not adopted on this result.
