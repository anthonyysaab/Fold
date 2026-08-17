# Field strength separation, recomputed on the canonical metric

**Generated** 2026-08-16 by `tools.measure_field_separation` (seed 20260816, 2000 bootstrap resamples over 1195 hands).

**VALIDATION GATE: PASS**

| gate | check | published | measured | delta | verdict |
|---|---|---:|---:|---:|---|
| reproduction (20260815T210237Z_poker-playground_s14_top15) | fold_rate | 0.558 | 0.558 | +0.000 | PASS |
| reproduction (20260815T210237Z_poker-playground_s14_top15) | aggression_rate | 0.215 | 0.215 | -0.000 | PASS |
| reproduction (20260815T210237Z_poker-playground_s14_top15) | median_bet_pot_fraction | 0.600 | 0.600 | +0.000 | PASS |
| control (binding) | control_field|overall separation on random holes | 0.000 | -0.000 | CI [-0.015, +0.014] | PASS |
| control | control_us|overall separation on random holes | 0.000 | -0.011 | CI n/a | UNRESOLVED |

## Separation, canonical metric (95% CI, bootstrap over hands)

| scope | street | n | mean strength aggress | mean strength fold | separation | 95% CI |
|---|---|---:|---:|---:|---:|---|
| field | overall | 8986 | +0.796 | +0.410 | +0.386 | [+0.373, +0.398] |
| field | preflop | 6533 | +0.803 | +0.398 | +0.406 | [+0.392, +0.420] |
| field | flop | 1401 | +0.786 | +0.511 | +0.275 | [+0.244, +0.304] |
| field | turn | 625 | +0.766 | +0.596 | +0.170 | [+0.108, +0.228] |
| field | river | 427 | +0.799 | +0.468 | +0.331 | [+0.252, +0.406] |
| field (s13) | overall | 4508 | +0.787 | +0.409 | +0.378 | [+0.360, +0.395] |
| field (s13) | preflop | 3275 | +0.787 | +0.396 | +0.391 | [+0.371, +0.411] |
| field (s13) | flop | 710 | +0.786 | +0.501 | +0.285 | [+0.241, +0.327] |
| field (s13) | turn | 322 | +0.777 | +0.612 | +0.165 | [+0.092, +0.244] |
| field (s13) | river | 201 | +0.799 | +0.447 | +0.351 | [+0.224, +0.463] |
| field (s14) | overall | 4478 | +0.804 | +0.411 | +0.393 | [+0.376, +0.411] |
| field (s14) | preflop | 3258 | +0.819 | +0.399 | +0.420 | [+0.400, +0.438] |
| field (s14) | flop | 691 | +0.786 | +0.522 | +0.264 | [+0.223, +0.305] |
| field (s14) | turn | 303 | +0.754 | +0.575 | +0.179 | [+0.075, +0.277] |
| field (s14) | river | 226 | +0.800 | +0.482 | +0.318 | [+0.217, +0.419] |
| us | overall | 98 | +0.475 | +0.441 | +0.034 | WITHHELD (41 aggress / 9 fold hands) |
| us | preflop | 58 | +0.451 | +0.415 | +0.036 | WITHHELD (41 aggress / 6 fold hands) |
| us | flop | 30 | +0.459 | +0.472 | -0.013 | WITHHELD (22 aggress / 2 fold hands) |
| us | turn | 8 | +0.942 | +0.537 | +0.404 | WITHHELD (2 aggress / 1 fold hands) |
| us | river | 2 | +0.882 | n/a | n/a | n/a (no aggress or no fold) |

## Per-agent (leaderboard top-15)

| rank | agent | n | fold | call | aggress | separation | 95% CI |
|---:|---|---:|---:|---:|---:|---:|---|
| 1/1 | pokr | 127 | 63.8% | 12.6% | 23.6% | +0.458 | [+0.362, +0.540] |
| 2/2 | charu | 182 | 33.0% | 37.4% | 29.7% | +0.409 | [+0.319, +0.490] |
| 3/9 | Jonas Reed | 165 | 43.6% | 29.7% | 26.7% | +0.436 | [+0.373, +0.496] |
| 3 | Paron | 96 | 68.8% | 11.5% | 19.8% | +0.319 | [+0.167, +0.475] |
| 4 | Halloran | 119 | 47.9% | 22.7% | 29.4% | +0.460 | [+0.365, +0.543] |
| 4 | Royal Flush Ronin | 103 | 62.1% | 21.4% | 16.5% | +0.442 | [+0.346, +0.534] |
| 5 | Mara Vale | 105 | 46.7% | 22.9% | 30.5% | +0.454 | [+0.346, +0.551] |
| 13/5 | Okonkwo | 145 | 55.9% | 29.7% | 14.5% | +0.429 | [+0.341, +0.521] |
| 6 | Kavanagh | 101 | 59.4% | 23.8% | 16.8% | +0.393 | [+0.225, +0.528] |
| 6 | Nixara | 122 | 45.1% | 32.0% | 23.0% | +0.410 | [+0.288, +0.508] |
| 7 | Gideon Shaw | 62 | 58.1% | 29.0% | 12.9% | +0.442 | WITHHELD (7 aggress / 36 fold hands) |
| 7 | HAQI | 79 | 70.9% | 13.9% | 15.2% | +0.300 | [+0.182, +0.411] |
| 7 | annelyboers | 79 | 78.5% | 13.9% | 7.6% | +0.442 | WITHHELD (4 aggress / 62 fold hands) |
| 8 | Keris | 105 | 56.2% | 17.1% | 26.7% | +0.393 | [+0.300, +0.479] |
| 8 | growthmindset | 77 | 85.7% | 6.5% | 7.8% | +0.280 | WITHHELD (4 aggress / 66 fold hands) |
| 9/15 | Chad | 120 | 75.0% | 5.8% | 19.2% | +0.464 | [+0.365, +0.546] |
| 10 | Calculated Storm | 64 | 48.4% | 32.8% | 18.8% | +0.432 | WITHHELD (9 aggress / 31 fold hands) |
| 10 | Fold-ver-4 (us) | 98 | 9.2% | 23.5% | 67.3% | +0.034 | WITHHELD (41 aggress / 9 fold hands) |
| 10 | Jagoan Neon | 76 | 57.9% | 23.7% | 18.4% | +0.353 | [+0.181, +0.514] |
| 11 | Cemini Wiki Poker | 108 | 50.0% | 25.0% | 25.0% | +0.456 | [+0.370, +0.537] |
| 11 | Pot Sheriff | 61 | 62.3% | 14.8% | 23.0% | +0.469 | [+0.382, +0.556] |
| 11 | maverick122 | 110 | 47.3% | 24.5% | 28.2% | +0.344 | [+0.244, +0.447] |
| 12/14 | Marlow | 155 | 43.9% | 26.5% | 29.7% | +0.367 | [+0.269, +0.455] |
| 12 | PITVIPER | 84 | 70.2% | 7.1% | 22.6% | +0.394 | [+0.297, +0.495] |
| 12 | Vance Okada | 81 | 55.6% | 23.5% | 21.0% | +0.390 | [+0.241, +0.507] |
| 13 | Ironclad | 62 | 69.4% | 6.5% | 24.2% | +0.349 | [+0.201, +0.509] |
| 14 | Variance Slayer | 72 | 44.4% | 45.8% | 9.7% | +0.420 | WITHHELD (6 aggress / 32 fold hands) |
| 14 | Vigilate | 95 | 62.1% | 21.1% | 16.8% | +0.494 | [+0.400, +0.589] |
| 15 | XENOLITH | 81 | 71.6% | 8.6% | 19.8% | +0.355 | [+0.183, +0.512] |

## Action mix per street

| scope | street | n | fold | check/call | aggress | median bet / pot |
|---|---|---:|---:|---:|---:|---:|
| field | overall | 8986 | 56.0% | 22.8% | 21.2% | 0.600x |
| field | preflop | 6533 | 69.2% | 12.5% | 18.3% | 0.600x |
| field | flop | 1401 | 26.6% | 42.3% | 31.1% | 0.462x |
| field | turn | 625 | 14.2% | 57.1% | 28.6% | 0.509x |
| field | river | 427 | 11.0% | 66.7% | 22.2% | 0.615x |
| us | overall | 98 | 9.2% | 23.5% | 67.3% | 0.493x |
| us | preflop | 58 | 10.3% | 19.0% | 70.7% | 0.400x |
| us | flop | 30 | 6.7% | 20.0% | 73.3% | 0.504x |
| us | turn | 8 | 12.5% | 62.5% | 25.0% | 0.754x |
| us | river | 2 | 0.0% | 50.0% | 50.0% | 0.444x |

## The honest null that travels with these numbers

| collection | n agents | Spearman(score, fold rate) | Spearman(score, aggression) | Spearman(score, separation) |
|---|---:|---:|---:|---:|
| 20260812T082057Z_poker-playground_s13_top15 | 18 | -0.232 | +0.440 | +0.075 |
| 20260815T210237Z_poker-playground_s14_top15 | 17 | -0.054 | -0.291 | +0.262 |

The field data does **not** say "fold more, score better". Read the
per-agent table as a spread, never as a ranking of virtue.

## Reading this against the frozen +0.150

The recomputed field separation is **+0.386** ([+0.373, +0.398], 8986 decisions over 1195 hands), and **+0.393** [+0.376, +0.411] on the S14 window the frozen figure was measured on.

**This is not a correction of +0.150 and it does not refute it.**
It is the same *quantity* — mean strength when aggressing minus
mean strength when folding — on a different *scale*: the canonical
exact-enumeration percentile of `strength_metric`, which is
street-comparable and player-count invariant, where the frozen pair
was computed on an undocumented percentile-style scale over a
narrower decision window. The two are not interchangeable and
neither supersedes the other. Only figures produced by this module
may be compared to the ones in this table.

**What that means for a v8 number.** A +0.110 quoted for the
incumbent on "a percentile scale" came from a different probe on a
different window (live-journal decisions, postflop only), so it
cannot be read against this table either. Our own replay-side
figure here is **+0.034** on 98
decisions in 44 hands, and its interval is withheld
because a bootstrap over hands cannot resolve 9 fold-hands — treat
it as consistent with the known behaviour (9.2% fold against the
field's 56.0%), not as a measurement in its own right. The honest
statement of the gap is directional: the field's aggression carries
a large, tightly resolved amount of hand-strength information on
every street, and ours carries an amount this archive cannot
distinguish from none.

**Acceptance targets are now numeric** (`V8_DESIGN.md` §6.3 — move
*toward* the field figure, preflop CI excluding zero):
overall +0.386, preflop +0.406, flop +0.275, turn +0.170, river +0.331.

## Caveats that travel with these numbers

- The decision-time board is rebuilt from `StreetDealt`. The
  post-action snapshot leaks the next street's cards on **962
  of 9084 decisions (10.6%)**; anything reading
  `snapshot.boardCards` scores those on a board the actor could not
  see.
- `TimeoutAction` events are excluded — they are not decisions the
  actor made. Including them moves the field fold rate to 58.6% and
  stops reproducing the published figure.
- The per-agent block selects on the archive's own dense `rank <=
  15`, which ties: 29 distinct
  agents across the two collections, not 30.
- Per-agent samples are small (61-182 decisions). Intervals are
  wide and several are withheld; the block shows a spread, never a
  ranking.
- The median-bet interval is degenerate because 0.600 is a hard
  mode of the field's sizing, not because the estimate is precise.
