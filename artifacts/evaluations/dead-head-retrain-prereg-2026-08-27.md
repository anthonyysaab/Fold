# Pre-registration — the degenerate-group retrain

**Frozen 2026-08-27, before any run exists.** Every threshold below is a
number fixed now. Nothing in this file may be recomputed from the runs it
judges; that is the whole point of writing it first.

This repo has been burned three times in two days by post-hoc reading — a
battery read at the wrong depth, a detector that could not fail, and a figure
conditional on a rolled-back configuration that propagated into three
documents. Two of those were mine.

## The hypothesis

`action_value` head constancy is caused, in part, by decision groups whose
within-group signal is exactly zero. On those groups the centred objective is
minimised by a constant head, and the ReLU tower makes that state absorbing.
Removing them from the action objective should reduce constancy.

## Arms — three, not two

| arm | `degenerate_group_filter` | what it is |
|---|---|---|
| **control** | `off` | the incumbent recipe, bit-identical to the pre-filter trainer |
| **treated** | `zero_weight` | zero-signal groups carry no action weight |
| **attribution** | `random` | a size-matched, uniformly drawn group set is muted instead |

The attribution arm is **mandatory**. Without it, "removing the attractor
helped" cannot be separated from "removing 29.66% of the groups helped".

`zero_weight` rather than `drop`: `drop` also strips those rows'
`state_value` and `residual_scale` supervision, cuts reward steps 10.1%, and
shifts the behaviour:reward trunk ratio +42%. `zero_weight` moves the action
objective's row set and nothing else — and the loss now normalises on the
**unfiltered** weight, so the filtered arm does not also run the action term
~1.38x hotter than `state_loss` and `residual_loss`.

## Primary readout

`heads.action_value.constant_pct` from `tools/head_degeneracy_audit.py`, over
the 4,795 stored journal rows carrying a 142-wide feature vector. One JSON per
arm per seed.

A retrained artifact scores on the same rows because `load_rows` filters on
feature width and `head_outputs` normalises with **the artifact's own** means
and stds — only the schema must match, not the normalisation.

**The statistic is the paired per-seed difference, never a level.**

## Thresholds — pinned now

From re-measuring all nine existing v7 artifacts: span 10.01%–46.36%, corpus
effect sd 12.31, seed effect sd 5.90, residual sd 6.52, giving a **paired
per-seed difference sd of 9.223 pp**.

| seeds | 2·SE boundary |
|---|---|
| n = 5 | **8.25 pp** |
| n = 8 | **6.52 pp** |

Let `D = mean(control − treated)` and `A = mean(random − treated)`, both paired
per seed.

- **CONFIRMED** — `D > boundary(n)` **and** `A > boundary(n)`. The targeted
  removal beats both doing nothing and removing the same number of groups at
  random.
- **REFUTED_BY_SIZE** — `D > boundary(n)` but `A ≤ boundary(n)`, **and** the
  half-width of the CI on `A` is below `boundary(n)`. Without that power
  condition a failure to reject zero would be reported as an affirmative
  causal claim, so absent it the verdict is UNRESOLVED_PENDING_ATTRIBUTION.
- **REFUTED** — `D ≤ 0`.
- **UNRESOLVED** — anything else. Escalate 5 → 8 seeds once. The boundary
  moves to 6.52 pp; it does not get recomputed from the data.

## Gates that VOID the run

A run failing any of these produces no verdict at all.

1. **Audit controls.** `tools/head_degeneracy_audit.py` must report
   `all_passed` on every artifact — zeroed `out_w` reads 100%, forced-live
   tower reads 0%, self-null identical.
2. **Seed variation.** The control arm's `constant_pct` values must **not all
   be equal**. Five identical values would otherwise confirm with a zero-width
   CI while proving only that the seeds never applied.
3. **Group counts, per split, exact.** These are what `reward_batch_loss`
   iterates, not whole-corpus figures:
   control **39,996** train groups; treated **28,143**; validation
   **10,045** / **7,056**.
4. **The attribution arm must not be inert.** Its recorded
   `*_muted_that_were_degenerate` must be strictly greater than 0 and strictly
   less than the zero-signal count. A random mask that caught all or none of
   them is not a control.
5. **Artifacts must be distinct.** Every arm×seed must differ from the
   incumbent and from each other in `weights_sha256`, and record its own
   `init_seed`. A run that silently reloaded old weights must be detectable.
6. **`min_spread`.** `constant_pct` tests bit-equality with `out_b`, so a head
   that went constant at a *non-bias* value would score a perfect 0.00%. Any
   arm whose `min_spread` is 0 while `constant_pct` is 0 is VOID.

## The falsifier, stated plainly

If `D ≤ 0` — the treated arm is no less constant than control — the
degenerate-group hypothesis is **wrong**, and the mechanism is elsewhere. The
named alternative is already on the table: the by-street cross-tab shows
degenerate groups at their *lowest* on the flop (26.6%) exactly where head
constancy jumps to 45.09%, so the `acts_first` correlation (68.96% of
no-price decisions are dead) is the leading rival explanation.

If `REFUTED_BY_SIZE`, the effect is about how many groups were removed, not
which — pointing at corpus size or batch composition, not the attractor.

## What this does NOT settle, even if CONFIRMED

- **Lower head constancy is not evidence of better play.** It is a property of
  the network, not of its results. Establishing better play needs a duel
  against `candidate-v7-0001c` on seat-swapped seeds clearing the known spread,
  plus the batteries — and per `DECISIONS.md` those remain a fit diagnostic.
- It does not explain `state_value`, which is 33.16% dead on an *uncentred*
  loss with the opposite street profile.
- It does not explain the flop.
- The filter's own bound: the head is dead on 51.17% of *live-group* corpus
  rows already, so removing zero-signal groups cannot by itself take corpus
  constancy below roughly that. **Expect it to help and not to suffice.**
- A clean win produces a candidate, nothing more. Promotion is a separate
  owner-authorised act, the bankroll is busted at 0, and none of this changes
  either.
