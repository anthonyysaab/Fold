# Does a less degenerate head make a less degenerate player? — no

The degenerate-group retrain cut `action_value` head constancy from 29.39% to
17.15%, confirmed against a pre-registered rule
(`dead-head-retrain-2026-08-27.json`). This asks the question that actually
matters: **did the play get less degenerate?**

The metric is the one this project retired the previous architecture over —
canonical **strength separation**, mean hand strength when the policy
aggresses minus when it folds. A player whose aggression says nothing about
its cards separates at zero, however lively its logits are.

Measured on `vs-p3` with the same `separation_report` and seed convention as
`p3-gate-2026-08-16`. 11 artifacts × 8 seeds × 1,000 hands. **0 chip
conservation violations.**

## The answer

| arm | separation | sd | per-artifact |
|---|---|---|---|
| control (`off`) | **0.1619** | 0.0388 | 0.1228, 0.1388, 0.1404, 0.2032, 0.2043 |
| treated (`zero_weight`) | **0.1653** | 0.0389 | 0.1172, 0.1413, 0.1599, 0.1957, 0.2122 |

**treated − control = +0.0034**, SE 0.0246, resolution threshold (2·SE)
**0.0491**. The difference is **fourteen times smaller than what this
measurement can resolve**. It is not a small effect; it is no effect.

Aggression rate is unmoved: control **0.698**, treated **0.688**, against the
real field's **0.215**. Both arms still act aggressively on roughly seven
decisions in ten.

And the reference points make it worse:

| | separation |
|---|---|
| real S14 field | **+0.386** |
| `candidate-v7-0001c` (incumbent) | **+0.2043** |
| control arm | +0.1619 |
| treated arm | +0.1653 |

**Both retrained arms separate *worse* than the incumbent.** Halving head
degeneracy did not move the play toward the field; it left it where it was,
slightly below the artifact already in production.

## The one real signal, which does not survive pooling

Per street the treated arm is better exactly where the head was deadest:

| arm | preflop | flop | turn | river |
|---|---|---|---|---|
| control | 0.1276 | 0.1848 | 0.2810 | 0.3591 |
| treated | 0.1190 | 0.1928 | **0.3260** | **0.4080** |

Turn +0.045 and river +0.049, against `action_value` constancy that ran
46.58% on the turn and 55.21% on the river. That is the right shape for the
mechanism. But preflop is slightly worse, preflop carries most of the
decisions, and pooled it washes out to nothing.

Worth pursuing as a lead, not reportable as a result.

## What this settles

**Do not promote any of these candidates.** They are not less degenerate
players. The treated arm is a better *network* by a confirmed margin and an
indistinguishable *player*.

It also settles the general question the session kept circling: head constancy
is a property of the model and does not transfer to play on its own. That is
now measured rather than argued — and it is the third time in two days that a
change which was demonstrably more correct produced no measurable improvement,
after the three gate repairs and the reverted gates that busted the bankroll.

## What it does not settle

- The mechanism is real and confirmed; only its *consequence for play* is
  null. Removing the attractor is still the right thing to do in any future
  training run — it costs nothing and the corpus budget is better spent on
  rows the objective can learn from.
- `vs-p3` is heads-up, and P3's aggression and sizing remain card-blind.
- The turn/river gain may be real and simply diluted. A probe weighted to
  postflop decisions, or a corpus that is not 79% preflop, could resolve it.
- Separation is not BB/100. Neither arm was duelled.
