# Does the v8 Phase B value head die for the same reason? — partly

The v7 `action_value` head returns a constant on 35.6% of live decisions, and
that was **confirmed** to be caused by degenerate decision groups: removing
them cut constancy from 29.39% to 17.15% while removing the same number of
groups at random did nothing
(`dead-head-retrain-2026-08-27.json`).

`STATUS.md` records the v8 Phase B composed-value head sitting at the constant
predictor on held-out data — 0.68% / −1.38% / −0.86% better than predicting
the mean — attributed to corpus size. This tests whether the v7 mechanism is
also at work there.

Offline, read-only. Nothing here trains, promotes, or deploys.

## The objective is structurally identical

`v8_trainer_phase_b.value_loss` (line 672):

```python
centered = values - (values * mask).sum(dim=1, keepdim=True) / counts.unsqueeze(1)
errors   = (centered - data["target"][indexes]).square() * mask
```

Same construction as v7's `reward_batch_loss`. **A constant head gives
`centered == 0` exactly**, so the level cancels and only within-decision
differences are supervised. The corpus builder also centres the targets — it
asserts `centered rewards sum to ~0` — so a decision whose branches all carry
the same reward has all-zero centred targets, and a dead head is its exact
minimiser. The attractor exists here.

Instrument check, both forced by construction: a constant branch vector
centres to max|c| = 0.0e+00; a varying one to 1.8750.

## But it is three times smaller

| corpus | degenerate decisions |
|---|---|
| `candidate-v7-0001` | 14,842 / 50,041 = **29.7%** |
| `candidate-v8-0002.phase-b` | 489 / 5,189 = **9.4%** |

So **the attractor alone does not explain the Phase B constant predictor.**
My hypothesis was that the two failures shared a cause; on the numbers, the
shared cause is present but far weaker in v8, and the corpus-size explanation
`STATUS.md` gives is probably the dominant one. 5,189 decisions is very little
for a 413-input network.

## The street profile is the interesting part

| street | decisions | degenerate | rate |
|---|---|---|---|
| preflop | 4,017 | 260 | **6.5%** |
| flop | 864 | 120 | **13.9%** |
| turn | 182 | 52 | **28.6%** |
| river | 126 | 57 | **45.2%** |

The same steeply rising shape as v7, and the river rate (45.2%) is close to
v7's (42.5%). The overall figure is low **because the corpus is 79% preflop**
— exactly the skew `NEXT.md` item 1 already flags — and preflop is the least
degenerate street.

## What this changes about the re-harvest

`NEXT.md` item 1 targets 50k+ branch rows with deliberate street balance. This
says the target should be stated in **usable** decisions, not raw ones:

- The river has **126 decisions, of which 45.2% carry no within-decision
  signal at all** — about 69 usable.
- The turn has 182, about 130 usable.

Harvesting river decisions yields roughly half what the count suggests, and
the shortfall is worst exactly where the corpus is already thinnest. A
stratified sampler that hits a raw per-street quota will still under-supply
the turn and river by nearly a factor of two.

Recommendation: **filter degenerate decisions at harvest time**, or count the
quota after filtering. The v7 experiment showed removing them helps rather
than hurts, so there is no reason to spend corpus budget on rows the objective
cannot learn from.

## Caveats

- Branch counts differ from v7's uniform 4: here 20.1% of decisions have 2
  branches, 49.6% have 3, 30.3% have 4. A 2-branch decision is likelier to be
  degenerate by chance, so some of the rate is structural rather than about
  the game.
- This measures the corpus, not the trained head. It does **not** establish
  that Phase B's constant predictor was caused by these 9.4%; it establishes
  that the mechanism is present and too small to be the whole story.
- The Phase A dataset was not examined.
