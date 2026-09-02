# v9 — plan from here to a deployable candidate (2026-09-02)

Extends `v9-loses-to-v7-diagnosis-2026-09-02.md` (the evidence) and
`v9-schedule-sweep-prereg-2026-09-02.md` (the first experiment).
Authority for the restructure itself remains
`.handoff/notes/V9_RESTRUCTURE_PLAN.md`.

**Where we are.** All 8 gauntleted v9 candidates lose head-to-head to
`candidate-v7-0001c` by 26-45 BB/100. `candidate-v9-0003b` was promoted by
owner decision, played Playground S17, and busted (mostly variance — it got
in as a 57/43 flop favourite and lost an 11-out, 25% river). The bankroll is
0; playing again needs an owner rebuy. Nothing is running.

---

## 0. What this session established

| finding | status |
|---|---|
| The training **schedule** is not the fix | 5-arm pre-registered sweep; criterion 2 fired. Lower LR moves the best checkpoint later (best_epoch 1→2→7→23) but **every** arm's validation is worse than control. |
| `equity_called` loses **3.9×** to a 2-parameter OLS on `equity_vs_posterior` (index 236) | Measured, reproducible, now an enforced promotion gate. |
| The network **beats** OLS on its own composed-value objective | 0.9437 vs 1.045. The failure is localized to one head, not the whole net. |
| The loss budget is **inverted** | `range` 48% of total loss (output read NOWHERE — auxiliary only), `fold_through` 21%, composed value 29% (variance-normalized), **`equity_called` 1.4%**. The value term was scale-corrected; the three supervised terms were summed raw. |
| Archive coverage is **0.758%** | 1,196 of 158,299 table files, from a hardcoded `DEFAULT_ROOTS` inherited from v8. ~125× more available for ~29 CPU-hours. |
| The gauntlet **battery is in-distribution** | It shares scripted opponents with the harvest panel. The v7 duel is the only out-of-distribution test in the whole gauntlet — and it is the one every candidate fails. |
| Harvest is **2.2× faster** | Evaluator structural best-five + hero-eval caching + `__deepcopy__` clones. Corpus byte-identical (`79e61dbd4edf410a`). |

**Structural gaps in the value composition** (4 heads, 17 outputs, shared
`[128,128]` trunk over a 64-wide card encoder and 48-wide context encoder):

1. **No P(raise).** `fold_through` is binary, so the complement of a fold is
   priced entirely as a call. Being blown off a hand costs nothing in the
   model, which systematically overvalues wagering with medium strength.
2. **Equity is a point estimate.** One scalar per branch cannot express
   "usually crushed, occasionally good" — the exact shape of a river
   decision on a completing board.
3. **One-step horizon.** `active` facing a bet is priced as
   `eq × (pot + to_call) − to_call`: showdown now, no future betting. That
   is reverse-implied-odds blindness on every flop and turn call.

---

## 1. Phase 0 — the instrument (blocks everything)

Nothing below is measurable today. The battery is in-distribution by
construction and the duel is a single aggregate number, so a change that
helps or hurts river behaviour is invisible. This is where "five
demonstrably-more-correct changes, zero measurable improvement, two made it
worse" came from.

**Build an out-of-distribution evaluation** with two parts:

- **Held-out opponents.** Opponents absent from the harvest panel. Note the
  hard constraint from §5: they cannot be learned policies *in the harvest*,
  but evaluation has no hole-resampling requirement, so v7, the v6 champion
  and fresh archetypes are all admissible **here**.
- **A named decision slice.** Stack-committing calls facing greater-than-pot
  bets on completing boards. Reported as its own number, not folded into an
  aggregate.

**Acceptance:** the instrument must separate v7 from v9 at least as clearly
as the duel does, and must report its own seed spread / MDE. An instrument
whose noise exceeds the effect is not an instrument — the seed spread on one
corpus was once 16.78 BB/100, larger than most effects worth chasing.

---

## 2. Phase 1 — fix the optimization (no architecture change)

**Rebalance the per-head loss.** The three supervised losses are summed raw
(`v9_trainer_phase_b.py:1033`) while the value term is divided by
`target_variance`. An 8-way cross-entropy naturally sits near ln(8) ≈ 2.08;
an MSE on a [0,1] equity sits near 0.046. Nobody chose 48/21/1.4 — it fell
out of the sum.

Also decide whether `range` should carry ~half the gradient at all. It is a
deliberate auxiliary task shaping the shared trunk
(`V9_RESTRUCTURE_PLAN.md:745`), which is legitimate — but auxiliary tasks are
normally down-weighted, not dominant.

**Target, already enforced:** `equity_called` must beat the 2-parameter OLS
(R² 0.198) on the same held-out split. `tools/ols_baseline.py` measures it and
`tools/promote_candidate.py` refuses format-4 candidates that fail.

**Method:** normalize, retrain ONE seed, read the gate. Cheap and falsifiable.

**Status (2026-09-02): RUN — criterion 5 fired; the default stays `raw`.** Criteria are frozen in
`v9-loss-rebalance-prereg-2026-09-02.md` (results appended there). The knob
landed as `engine/supervised_loss_normalization_v9.py` + Phase-B trainer flags
`--supervised-normalization {raw,constant-predictor}` and one
`--<head>-loss-weight` per head; the default stays `raw` until the
pre-registered criteria say otherwise. Arms in
`artifacts/evaluations/loss-rebalance-2026-09-02/`.

**Result:** normalizing the supervised terms doubled the `equity_called`
held-out R² (0.076 → 0.159, seed 401) but did not clear the OLS bar
(0.178); value_normalized moved 0.944 → 0.949. The decision point below is
answered by the pre-registered diagnostic: the head fits the training
label (train R² 0.38) and does not generalize — Phase-A batches interleave
every step, so the 7,458 labelled rows are cycled ~7× per Phase-B epoch.
That is the data side: **Phase 2 is next**; the label/feature check is not
indicated by this readout. Down-weighting or removing the `range`
auxiliary was inert. Recommendation carried forward: run Phase 2 with
`constant-predictor` normalization as the experiment arm, re-read against
the same bar.

**Decision point.** If `equity_called` still loses to two parameters after
rebalancing, the problem is the **label or the feature**, not the weighting.
Establish which before spending 29 CPU-hours on data or touching the
architecture. Pre-register this the way the schedule sweep was.

---

## 3. Phase 2 — data

Only after Phase 1. More data does not fix a gradient that never arrives, and
the 6× corpus already moved `best_epoch` *down*.

**The three blockers this section originally listed are already FIXED**
(2026-09-02, `tools/build_phase_a_dataset_v9.py`). Do not redo them:

- **RAM.** Rows now stream through fixed-size sorted chunks (`chunk_rows`,
  default 50,000) written to a temp directory and `heapq` k-way merged into
  the archive, in the same `(table_id, sequence)` order the in-memory sort
  produced. The 17.8 GB resident estimate no longer applies.
- **Container / empty directories.** Now collected into `skipped_roots` and
  reported with a warning instead of raising `FileNotFoundError`. Passing a
  container directory is no longer fatal.
- **Duplicate tables.** `dedupe_tables` (default on) drops rows whose table
  id was already emitted, covering the 536 byte-identical duplicates.

**Verified 2026-09-02: the chunked merge is byte-identical.** A default-roots
rebuild reproduces the pre-streaming artifact exactly — 1,648,641 bytes,
sha256 `ecb4739df9d1b9ec`, gzip container and decompressed bytes both
identical, 9,084 rows. Use that hash as the Phase-A analogue of the
corpus-hash oracle: any future change to this builder must reproduce it.

**Still outstanding — dedupe and skipped-roots are untested.** The test diff
only adjusted an existing fixture (distinct table ids per root, since dedupe
would otherwise collapse them). The byte-identity check above does NOT cover
either behaviour: the default roots contain no duplicates and no unreadable
directories, so neither code path executes. Both fire for the first time on
the widened run.

The dedupe deserves a test most — it **silently drops rows** by table id, and
nothing asserts it drops exactly the 536 duplicates and nothing else. On a
pipeline whose whole problem is too little data, a wrong id extraction would
discard real training rows with no error. Add tests for both before widening.

Then:

1. **Widen `--roots`** to the 11 unread leaf season directories. No code
   change — `--roots` is `nargs="+"`. ~29 CPU-hours at a measured
   **0.09 s/decision** (Phase-A; do NOT conflate with the 1.577 s/decision
   Phase-B harvest figure).
2. Retrain, re-gate.

---

## 4. Phase 3 — architecture

**3a. Response head — fold / call / raise.**
Replace the 2-wide binary `fold_through` with a 2×3 softmax per wager lane
(`active`-bet, `aggressive`). Width 2 → 6.

Composition, stage 1, with **no new value machinery**:

```
value = p_fold  × pot
      + p_call  × (eq × pot_if_called − wager)
      + p_raise × (−wager)
```

Pricing a raise as "we fold to it" is a strict lower bound, needs no second
value estimate, and corrects in the direction the evidence points — this
agent over-aggresses (66.9% vs the incumbent's 64.6%).

- **Labels already exist.** The Phase-A builder derives `fold_through` by
  walking replays gated on raw wager actions, lane by state; the same walk
  sees fold vs call vs raise.
- **Cost is contained.** Heads are outputs, so **schema 4's 414 inputs are
  untouched — no feature re-harvest.** Needs a Phase-A re-extraction for the
  new labels. The Phase-B value target is outcome-based and unchanged.
- **Stage 2, only if measured to help:** a learned `value_if_raised` head
  replacing the −wager bound.

**3b. Distributional equity — conditional on the Phase 0 slice.**
`equity_called` 3 → 9 (quantiles per slot); mean for value, downside for the
call branch. This changes the **pinned** value arithmetic and the train/serve
parity contract the Phase-B corpus replays against. Expensive. Do it only if
the slice shows the call side actually leaking.

**3c. One-step horizon — named and deferred.**
Needs a continuation value. Largest and most invasive. Recorded so it is not
mistaken for an oversight.

---

## 5. Phase 4 — evaluation and the promotion bar

Re-gauntlet with the Phase 0 slice included. **Promotion requires all three:**

1. The OLS gate passes (automatic, fail-closed, both arms).
2. The v7 duel is not a loss.
3. The out-of-distribution slice is non-negative.

The battery alone is not evidence — that was the mistake in the 0003b
promotion case, where a +36.89 paired battery win was an in-distribution
number presented as strength.

---

## 6. Phase 5 — deployment

1. **Owner rebuy.** Bankroll is 0. This is the owner's action alone.
2. **Restart the live session**, which also picks up two fixes that only take
   effect on a fresh process:
   - the `--competition` money-guard (an explicit id used to bypass
     `is_free_playground()` entirely, with a real-money competition active);
   - **schema-4 telemetry** — live hands become usable v9 training and audit
     rows instead of being journalled at 142-wide.
3. **Set `--min-chips`.** It defaults to 0, i.e. play to the bottom.
4. **Feed it back.** Live play is the only genuinely out-of-distribution data
   source available. Once it records schema 4, it becomes a Phase 2 input.

---

## 7. Standing rules for every step

- **The corpus-hash oracle is the arbiter.** Any engine-touching change must
  reproduce `79e61dbd4edf410a` via `python -m tools.bench_harvest oracle`, or
  it is not an optimization — it is a different experiment.
- **Pre-register experiments** with criteria frozen before the numbers land.
- **A gate arm that cannot be evaluated is a refusal, not a pass.**
- **Sweep every engine-touching layer adversarially.** A green suite is not
  evidence of safety here; two live-money holes have shipped past one.

---

## 8. Rejected, with reasons — do not re-attempt without new evidence

- **GPU/CUDA harvest rewrite.** Training already runs on CUDA. The harvest's
  cost was an O(21) hand evaluator and a gratuitous `deepcopy`, both fixed in
  pure Python for 2.2×. A GPU port would mean a second implementation of the
  decision engine, validated only statistically.
- **A hand-coded river rule.** It would paper over the head whose entire job
  is the number in question, was argued from n=1, and this project has found
  six hand-authored constants wrong or inert in a single day.
- **Learned opponents in the harvest panel.** Attempted and reverted
  2026-09-02. Phase B resamples every card-reading opponent's holes
  conditional on that opponent's own prefix decisions; only `P3SeatWrapper`
  records that prefix. A learned seat needs a conditional hole sampler — i.e.
  inverting the network. Belongs in **evaluation**, not the harvest.

---

## 9. Claims from this session that did NOT survive

Recorded so nothing is rebuilt on them.

- **"The extractor discards ~98% of the archive for want of labels."**
  Refuted on every part. 9,084 v9 *rows* and 584,704 v7 *examples* are
  different units — only 259,539 of v7's came from the archive at all. Per
  hand v9 extracts **more** (7.60 vs 1.75 rows/hand). Within the roots it
  reads, extraction yield is **100%**, and label masking drops zero rows. The
  real number is coverage: 0.758%.
- **"A 2-parameter OLS beats the network, so nothing downstream matters."**
  True only on the Phase-A `equity_called` label. On its own composed-value
  objective the network wins, 0.9437 vs 1.045.
- **"The network never improves across the remaining ~146 epochs."**
  It runs 11 epochs — early stopping fires after 10 stale ones. The point is
  sharper, not weaker: the run dies at ~7% of the schedule.
- **"The bust hand was diagnostic, the same shape as 2026-08-26."**
  Mostly variance. 57/43 on the flop, 75% going to the river, lost to an
  11-out 25% card. 2026-08-26 was a broken mechanism firing; this was a sound
  line that lost. The river **call** is probably still −EV, but it is one
  marginal call, not the reason the money went in — and n=1 supports neither
  conclusion.
