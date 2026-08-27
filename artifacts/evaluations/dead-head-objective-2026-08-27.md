# Why the value head dies and the classifier doesn't — 2026-08-27

`candidate-v7-0001c`'s `action_value` head returns its output bias, bit for
bit, on **35.6%** of real decisions. `behavior_prior`, the same tower shape
on the same trunk, is **0.0%** dead. Retraining alone is ruled out — nine
artifacts across three corpora and three seeds all land between 10.0% and
46.4% — so the question is what the objective rewards.

This is an offline diagnosis. **Nothing here trains, promotes, or deploys.**

## The instrument, before the result

| check | reads | must be |
|---|---|---|
| a CONSTANT head, centred within a 4-branch group | max&#124;centred&#124; = 0.0e+00 | exactly 0 |
| a VARYING head, same centring | max&#124;centred&#124; = 0.4000 | non-zero |

The degeneracy figures they explain come from
`tools/head_degeneracy_audit.py`, whose own controls (zeroed `out_w` must
read 100%, forced-live tower must read 0%) both pass.

## The mechanism

`offline_trainer.reward_batch_loss` centres the prediction inside each
decision group before comparing it to the target:

```python
centered = predicted - (sums / counts)[group_ids]
action_loss = (weight * (centered - target).square()).sum() / weight.sum()
```

**A constant head produces `centered == 0` exactly.** The group mean of a
constant is that constant, so the level cancels identically — only
within-group *differences* are ever supervised. A dead tower produces
differences of exactly zero.

Now the corpus. 200,164 counterfactual rows over **50,041 decision groups**,
every group exactly 4 branches:

| quantity | value |
|---|---|
| targets exactly 0 | 59,811 / 200,164 (**29.9%**) |
| within-group target spread, p50 | 0.0908 |
| **groups where all four targets are identical** | **14,842 / 50,041 (29.7%)** |
| …of those, groups where the shared target is also 0 | **14,842 (29.7%)** |

**On 29.7% of decision groups, `centred == 0` is not a compromise — it is the
exact loss minimiser.** A dead tower is the correct answer there.

And the death is absorbing. The head tower is `linear -> ReLU -> linear`; when
every pre-activation is at or below zero the ReLU derivative is zero, so that
input contributes **no gradient to the tower at all**. A unit that dies on the
degenerate third of the corpus cannot be revived by it.

## Why the classifier is immune

`behavior_batch_loss` is `-log_softmax(logits)[chosen]`. Softmax couples all
four outputs, so every row pushes gradient into every one of them, and a
constant logit vector yields uniform probabilities at a fixed loss of
`log(4) ≈ 1.386` — strictly worse than any discriminating solution. **There is
no constant that minimises cross-entropy.** No attractor, hence 0.0% dead.

That is the whole asymmetry. It is not the tower shape, which both heads
share. It is that one objective has a constant solution and the other does
not.

## The objective is not signal-free — it is signal-free in patches

| | loss |
|---|---|
| dead head (`centred == 0`) | 0.046107 |
| optimal centred head | 0.000491 |
| removable by learning (within-group variance) | 0.045617 |

Globally the dead head sits at **94x** the achievable loss and forgoes
**98.9%** of what the objective can remove. So this is not "the targets carry
no information". Roughly 70% of the corpus carries plenty; the other 30%
actively rewards death, and death is one-way.

## What this implies for a fix

A LayerNorm on the head tower addresses the *absorbing* property but **not the
incentive** — the degenerate groups would still pay the head to be constant.
That is very likely why nine retrained artifacts all landed in 10–46%.

Ordered by how directly each attacks the mechanism:

1. **Drop or zero-weight the degenerate groups.** 14,842 groups supply
   literally zero within-group signal while rewarding the constant solution.
   They are 29.7% of the training signal and removing them removes the
   attractor. Cheapest to try, and it needs no architecture change.
2. **Make the level supervised**, so a constant is not free. The centring
   deliberately discards it; if it is discarded, the degenerate groups have to
   be discarded with it.
3. **LayerNorm / bias floor on the head tower** — worth doing, but as
   insurance against absorption, not as the fix.

## Caveats

- The 29.7% degenerate-group rate and the 35.6% live-dead rate are a close
  correspondence, **not a proven identity**: one is measured over training
  groups and the other over live decisions. Suggestive, not established.
- **The by-street cross-tab is now done, and it only partly supports the
  story.** Read from the feature matrix (street one-hot at indices 104-107,
  verified exactly one-hot on 5,000 rows):

  | street | groups | zero-signal | rate | live head dead |
  |---|---|---|---|---|
  | preflop | 29,804 | 8,557 | 28.7% | 27.4% |
  | flop | 11,185 | 2,971 | **26.6%** | **45.09%** |
  | turn | 5,467 | 1,789 | 32.7% | 46.58% |
  | river | 3,585 | 1,525 | **42.5%** | **55.21%** |

  Both profiles peak hard at the river, and preflop lines up almost exactly
  (28.7% vs 27.4%). **But the flop does not**: degenerate groups are at their
  *lowest* there (26.6%, below preflop) while the head is at 45.09%. So
  degenerate groups are part of the mechanism and are clearly implicated at
  the river, but **they do not explain the flop jump**. Something else drives
  that -- the `acts_first` correlation (68.96% of no-price decisions are dead,
  and hero acts first mostly postflop) is the obvious candidate and is not
  yet separated from this.
- `state_value` is 33.16% dead on an *uncentred* single-output loss, so
  centring is not the only route into the absorbing state. Its street profile
  runs the opposite way (35.85% preflop down to 7.72% river), which this
  diagnosis does not explain.
- **A hypothesis worth testing, not a claim**: `STATUS.md` records the v8
  Phase B composed-value head sitting at the constant predictor (0.68% better
  than the mean), attributed to corpus size. Two value regressions on two
  architectures both collapsing to a constant, while the classifier on the
  same trunk does not, suggests the objective may be implicated there too. If
  so the corpus-size explanation is incomplete.
