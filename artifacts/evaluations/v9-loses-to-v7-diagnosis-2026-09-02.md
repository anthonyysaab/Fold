# Why no v9 candidate beats v7 — diagnosis, 2026-09-02

**Status:** all 8 gauntleted v9 candidates lose head-to-head to
`candidate-v7-0001c`. `candidate-v9-0003b` was nevertheless promoted by owner
decision on 2026-09-02 and is live on Playground S17.

**Method note:** the extraction half of this report came from a 51-agent
adversarial audit (0 errors, 0 empty results). Every non-minor claim was handed
to a refuter instructed to default to "refuted" when it could not verify
independently. **The headline claim this investigation started from was killed
by that process.** The two load-bearing survivors were then re-verified by hand.

---

## 1. The claim that did not survive

An earlier reading of this session held that *"the v9 Phase-A builder extracts
~2% of what the v7 line got from the same archive — 9,084 rows against 584,704
examples — so the extractor is discarding nearly everything for want of
labels."*

**That is wrong in all three of its parts.** Recorded here because it was
briefed to the owner before it was checked.

**The units are not comparable.** 9,084 is Phase-A v9 *decision rows*, one per
`ActionTaken`, every seat, over 1,196 hands. 584,704 is v7 *examples* = 384,540
behavior + 200,164 counterfactual. The counterfactuals are **simulator
rollouts, not archive extractions** — measured as 50,041 decision points × 4
branch labels (`engine/table_simulator.py:842-860`). A further 125,001 of the
"behavior" examples are also simulator output. Only **259,539** of v7's
examples came from the archive at all
(`artifacts/corpora/candidate-v7-0001.corpus.meta.json`).

**Per hand, v9 extracts MORE than v7.** 9,084/1,196 = **7.60 rows/hand**
against v7's 259,539/148,054 = **1.75 archive rows/hand**. v7's path was
hero-only across 56 tracked agents; v9 takes every seat. The "2%" was a
*hand-count* ratio wearing an extraction-rate costume.

**The extractor discards nothing.** Within the roots it reads, yield is
**100%** — an identity-level bijection on `(table_id, sequence)`: 0 missing, 0
extra, 0 duplicates, matching street-by-street (6591/1431/633/429). Every skip
counter in the sidecar fired **zero** times.

**Label masking loses 0 rows.** Rows are emitted unconditionally
(`build_phase_a_dataset_v9.py:395`) with labels defaulting to 0.0/mask 0
(`:289-291`); the trainer masks per-head with clamped denominators
(`engine/v9_trainer.py:573-582`). 7,115 rows carry no fold-through label
because the action was not a wager — undefined by construction, not dropped.

---

## 2. Cause A — the training never actually happens

**This is the finding with the best cost/benefit in the report, and it was not
what anyone was looking for.**

`warmup_steps` is 200. The best validation checkpoint of **every** v9 candidate
lands at or within a few steps of that boundary:

| candidate | rows | steps/epoch | best_epoch | best step | vs. warmup 200 |
|---|---|---|---|---|---|
| 0001a/b/c | 9,084 | 29 | 6 | **174** | before warmup ends |
| 0002a | 49,096 | 154 | 2 | 308 | just past |
| 0002b/c | 49,096 | 154 | 1 | **154** | before warmup ends |
| 0003a/b/c | 65,127 | 204 | 1 | **204** | 4 steps after |

Each network is checkpoint-selected at the moment the learning rate first
reaches peak, and never improves again. **9 of 10 best checkpoints sit at or
before step 200.**

*Corrected 2026-09-02:* an earlier version of this section said the network
"never improves across the remaining ~146 epochs." It does not run 146 more
epochs — early stopping fires after 10 stale ones, so `candidate-v9-0003b`
records `best_epoch 1, epochs_run 11` and the cosine schedule decays the
learning rate by ~1% before the run ends. The substance is unchanged and is
in fact sharper: the run terminates at ~7% of the schedule, so the decay
never engages at all.

Two corroborating facts:

- **A 6× larger corpus moved `best_epoch` DOWN** (6 → 1). More data made the
  stopping point earlier, which is not how data-starvation behaves.
- **A 2-parameter OLS beats the network.** On the 0001a split, ordinary least
  squares on the single feature `equity_vs_posterior` (schema-4 index 236)
  reaches validation **R² = 0.199**; the 71,841-parameter network reaches
  **0.058**. Nine parameters reach 0.231.

A network losing 3.4× to a two-parameter linear model on one input is not
under-fed. It is under-trained, or the objective/schedule is wrong.

**This also explains the monotone that looked like "better value head → worse
play."** `value_norm` measures fit to the training corpus. Higher value_norm
tracked *more overfitting to a tiny in-distribution corpus*, which is why the
0003 family (0.939-0.946) duels worst at −43 BB/100 and the 0001 family, whose
head is near-constant, duels least badly at −28.

---

## 3. Cause B — coverage is 0.758%

`DEFAULT_ROOTS` (`tools/build_phase_a_dataset.py:113-116`) hardcodes exactly
two collections. The v9 builder imports it verbatim
(`build_phase_a_dataset_v9.py:89`) and uses it as the `--roots` default
(`:690`). The run took the default with `limit: null`.

**Measured: 1,196 of 158,299 table files. 128 MB of 18.1 GB.** Unread: 155,642
under `last 5 seasons top 15`, 1,184 tournament, 277 final tables.

Worse, both roots are *deliberately truncated snapshots* — their manifests
carry `scope.recent_hand_limit_per_agent: 50`. The s13 root's competition has a
**full-history twin on disk at 41,511 tables**; the builder read 587 of them.

**Not a v9 regression.** `artifacts/phase_a/phase-a-dataset.summary.json` shows
the v8 builder producing the identical 1196/1195/9084 from the same roots. This
is an inherited v8 constant with no recorded rationale.

Projected yield if widened: **~1.13–1.14M rows, ~125×**. *This is an
extrapolation from per-collection densities (exhaustive counts on small
collections, 400–1,500-file samples on the large ones), not a build.*

---

## 4. Cause C — no learned opponent is in the harvest panel

The entire opponent roster is eight entries, each three hand-tuned floats
(`tools/build_phase_b_corpus.py:1197-1206`): three P3-wrapped strength-aware
agents and five scripted/textured bots. **No v6, v7, v8, or v9 policy appears
anywhere.**

The gauntlet **battery** channels (`vs-median`, `vs-station`, `vs-textured`)
share opponents with the harvest. The battery is therefore substantially
**in-distribution**, and the duel against v7 is the only out-of-distribution
test in the gauntlet — which is exactly the test all 8 candidates fail.

**This invalidates part of the promotion case for 0003b.** Its battery result
(150.05 vs the incumbent's 113.16, paired +36.89) is an in-distribution number
and is much weaker evidence of live strength than it was presented as.

---

## 5. Recommended sequence

1. **Fix the training schedule.** Near-zero cost, and the OLS result says it
   may be worth more than any amount of data. Establish why the best checkpoint
   always lands inside warmup; check `warmup_steps` against
   `steps_per_epoch`, the early-stopping criterion, and the LR peak.
2. **Re-check the objective against the OLS baseline.** A 71,841-parameter
   network must beat 2 parameters on one feature. Until it does, nothing
   downstream matters.
3. **Widen `--roots`.** No code change — `--roots` is `nargs="+"`. Cost ~29
   CPU-hours at a measured **0.09 s/decision** (Phase-A; do **not** conflate
   with the 1.577 s/decision Phase-B harvest figure).
   - **Blocking prerequisite:** `:549`/`:562`/`:594` accumulate and sort all
     rows in RAM. At 15,578 bytes/row, 1.14M rows ≈ **17.8 GB resident** — the
     widened run would likely OOM. Cheap fix, must land first.
   - **Trap:** the three container directories have no `raw/tables` and will
     raise `FileNotFoundError` at `:539-540`. Pass leaf dirs. Two leaf dirs are
     empty and will also raise.
   - **Trap:** 536 files are byte-identical duplicates between roots
     (md5-verified 536/536). Dedupe or accept train/val contamination.
4. **Add v7 and the v6 champion to the harvest panel.** `_OPPONENT_KINDS` is a
   `(kind, float, float, float)` table that cannot carry a manifest path;
   needs widening plus a `kind == "learned"` branch in `_build_opponents`
   (shared by both builders via `build_phase_b_corpus_v9.py:152`). **Prices
   in:** a learned opponent runs its own equity Monte Carlo, plausibly doubling
   harvest cost at `equity_trials: 1000`.
5. **Recover the 756 `TimeoutAction` rows** — optional, defensible exclusion,
   documented at `build_phase_a_dataset.py:51-53`. Does not touch the coverage
   gap.

## 6. Harvest throughput — do not build a GPU port

Profiled (`cProfile`, `--harvest-workers 1`, `--hands-scale 0.01`, faithful
config: hero `candidate-v9-0001a`, `equity_trials 1000`,
`counterfactual_rollouts 2`). Self time:

| module | share | detail |
|---|---|---|
| `evaluator.py` | **33.7%** | `_five` 22.0% (98.6M calls), `_seven` 10.3% |
| `copy.py` | **23.9%** | `deepcopy` 15.8% (50.2M calls) |
| `card.py` | **18.1%** | `prime_product_from_hand` 17.5% (98.4M calls) |
| builtins | 10.9% | `dict.get` 3.5% (101.6M calls) |

`_seven` brute-forces all 21 five-card combinations — **1.67M five-card
evaluations per decision** (call counts are exact under cProfile, so this
figure is profiler-independent). `deepcopy`'s 50.2M calls come from only ~671
top-level copies at five engine sites, each descending through ~75,000 atomic
values; 49.7M of them are `_deepcopy_atomic`.

Both are pure-Python algorithmic problems, both **exactly verifiable** (a hand
ranking is right or wrong; a restored state equals the copied one or does not),
and together they are ~76% of runtime. A GPU port would instead require a
second implementation of the decision engine — the code that has already
shipped two live-money holes past a fully green suite — validated only
statistically. **Fix the algorithm before buying hardware parallelism.**

Also measured: `_counterfactual_examples` is 396.4s of the 477s run. **83% of
harvest cost is counterfactual branch replay**, not hero decisions.

## 7. Unknown

- Whether the schedule fix alone closes the gap. Cheapest thing to find out.
- The ~1.14M row projection is extrapolated, not built.
- Whether widening roots changes the *distribution* enough to matter, or just
  the volume.
- A latent asymmetry at `build_phase_a_dataset_v9.py:282-284`: a missing actor
  holding drops the whole row, though it is only used for `equity_called` —
  a label other rows routinely mask. Reachable (demonstrated), zero
  occurrences archive-wide in every sample taken. Cost 0 today.
