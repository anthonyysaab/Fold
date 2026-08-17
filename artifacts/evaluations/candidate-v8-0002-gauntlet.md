# candidate-v8-0002 — Phase-B composed-value gauntlet (all three seeds)

**Generated** 2026-08-17. **Instrument**: `tools/evaluate_v8.py`, stages
`nullcheck,trivial,battery,champion,p3,duel,strength`. Carry-over sessions,
6,000-chip stacks at 50/100, `equity_trials` 80.

**No promotion. `artifacts/approved.json` untouched** (still
`candidate-v7-0001c`, approved 2026-08-14). All three artifacts are
`state: "candidate"`, `promotion: null`.

## Verdict

**NOT PROMOTABLE — and this time the loss is resolved, not unresolved.**

1. **All three seeds lose the duel decisively.** −32.75 / −21.44 / −31.55
   BB/100 over 16 seat-swapped seeds (t −10.46 / −6.97 / −10.53). Every
   margin exceeds the 16.78 BB/100 known seed spread, so DECISIONS §5's
   "enough seeds to clear the known spread" is satisfied *against* the
   candidates: this is a measured loss, not an UNRESOLVED. The across-seed
   spread is 11.31 BB/100 — below 16.78 — so the three seeds agree with each
   other; the result is a property of the training, not of the init seed.
2. **This is a regression against Phase A.** The Phase-A siblings scored
   +10.61 / +9.77 / +0.90 (all UNRESOLVED). Phase-B composed-value training
   moved the duel by roughly −35 BB/100 and turned an unresolved coin-flip
   into a resolved loss.
3. **Batteries are not held** on any channel against either freshly rebuilt
   reference arm. `vs-station` is the extreme: 10.27 / 43.46 / 16.19 against
   the champion's 250.82 and the incumbent's 273.95.
4. **The vs-p3 gate is failed, and it is the gate that was supposed to be
   informative.** All three lose to the incumbent on the repaired
   strength-aware channel by −13.89 / −15.81 / −8.79 BB/100, each resolved
   against its own 2·sd/√n (11.13 / 11.79 / 8.59). Member c's margin clears
   its threshold by only 0.20 BB/100 — call that one marginal.
5. **Strength separation moved the wrong way.** Overall +0.086 / +0.108 /
   +0.099 against the incumbent's +0.170 on the identical sims, and far from
   the canonical field benchmark +0.386. Preflop is positive with a CI
   excluding zero (+0.078 / +0.066 / +0.092), which is the design's minimal
   preflop acceptance — but the incumbent posts +0.153 on the same sims, so
   the acceptance is cleared while the comparison is lost.

**Why, mechanically — the objective did not train.** On the held-out split
the composed-value head is *at the constant predictor*: +0.68% for seed 401
and −1.38% / −0.86% for 402 / 403 (§6). A value head that cannot beat
"predict the branch-set mean" is contributing noise to the serve path, and
the serve path now consumes it for every action. That is the honest
explanation for a −35 BB/100 swing, and it is a **corpus-size result, not a
design refutation**: 16,094 branch rows from 5,189 decisions is roughly a
third of the 40–60k target, and 79% of it is preflop.

**What this does and does not falsify.** It does not falsify the composed
value design — the composition arithmetic is verified train/serve identical
to 4.1e-08, and the component heads still carry their Phase-A supervision.
It does establish that **Phase B at this corpus size is worse than no Phase
B**, which is a real and useful negative. The next honest step is corpus
volume and postflop coverage, not a hyperparameter search against a
16k-row, preflop-heavy signal.

## Reading caveats

- **The published battery means are stale and were not used.** §2 measures
  it like-for-like: the same policy, rebuilt, misses its published per-seed
  values by 24–155 BB/100 on every channel. Every battery number here is
  against arms rebuilt in this run.
- **The batteries still cannot referee aggression.** `always_aggress_large`
  beats every real policy on most card-blind channels, and it beats them on
  vs-p3 too (123.87 against the incumbent's 64.65) — so even the repaired
  P3 opponent is not a ceiling-free judge. Battery results remain a fit
  diagnostic (DECISIONS §5).
- **Turn/river value labels are thin.** The corpus holds 480 turn and 311
  river branch rows (validation: 16 and 15 decisions). Per-street value
  numbers on those streets in §6 are reported with their coverage and should
  not be read as evidence.
- **Two live documents still cite a superseded separation benchmark.**
  `DECISIONS.md` §5 and the `stage_strength` caveat string in
  `tools/evaluate_v8.py` both say the field benchmark is **+0.150**; the
  canonical recomputation
  (`field-separation-canonical-2026-08-16.json`) says **+0.386**. This report
  uses +0.386. Neither document was edited — flagged, not fixed.

## Seed → artifact map

| member | init seed | artifact | weights sha256 |
|---|---:|---|---|
| a | 401 | `candidate-v8-0002a` | `965eaa48fa7116236c41fb7b4f32a51178d949ee351dfa7a41ee16f9e8c6bb6a` |
| b | 402 | `candidate-v8-0002b` | `191dfb66ece1745e87de12e9fe93322e15de16ecd4c386327c78970317fab4fd` |
| c | 403 | `candidate-v8-0002c` | `e434ccd88775b43dd67b483b069b3b68989702e399b6826ecbae72666ed5fd63` |

## 1. Duel vs `candidate-v7-0001c` — the selector (16 seat-swapped seeds)

| member | seed | margin BB/100 | sd | se | t | 2·sd/√n | resolvable | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| a | 401 | **-32.75** | 12.52 | 3.13 | -10.46 | 6.26 | 16.78 | incumbent wins |
| b | 402 | **-21.44** | 12.30 | 3.08 | -6.97 | 6.15 | 16.78 | incumbent wins |
| c | 403 | **-31.55** | 11.98 | 3.00 | -10.53 | 5.99 | 16.78 | incumbent wins |

**Across-seed spread: 11.31 BB/100** (min -32.75, max -21.44). The known seed spread on
this project is 16.78 BB/100; a margin below the resolvable spread is
UNRESOLVED by rule, never a win or a loss.

## 2. Instrument gate — does the published champion still post its numbers?

The noise floor's champion is **`heuristic-aggressive-v6`**, not the duel
incumbent. `stage_champion` rebuilds that same policy on the same channels
and the same seeds, so fresh-minus-published isolates the *instrument*, not
a policy difference.

| channel | fresh | published mean | published MDE@8 | max per-seed diff | reproduces? |
|---|---:|---:|---:|---:|---|
| vs-median | +116.45 | +149.72 | 17.06 | 48.9100 | **no** |
| vs-nit | +62.30 | +79.72 | 3.22 | 24.2700 | **no** |
| vs-station | +250.82 | +311.54 | 42.13 | 131.1800 | **no** |
| vs-shover | +84.57 | +158.29 | 46.58 | 154.8900 | **no** |
| vs-textured | +111.12 | +165.61 | 17.13 | 96.1800 | **no** |
| five-max-lineup | +165.18 | +214.93 | 40.67 | 92.2400 | **no** |

**All channels reproduce: NO.**

## 3. Batteries vs the freshly rebuilt arms (8 seeds, seed-paired)

Both reference arms were rebuilt this run through the identical
`battery_tasks` seeds, so every paired difference is like-for-like.
`champ` = `heuristic-aggressive-v6` (the noise-floor champion);
`inc` = `candidate-v7-0001c` (the duel incumbent).

### member a (init seed 401)

| channel | candidate | champ | diff vs champ | t | MDE | held? | inc | diff vs inc | t |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| vs-median | +65.66 | +116.45 | -50.80 | -6.34 | 16.02 | **no** | +103.46 | -37.80 | -7.74 |
| vs-nit | +48.74 | +62.30 | -13.56 | -4.22 | 6.43 | **no** | +65.81 | -17.07 | -6.68 |
| vs-station | +10.27 | +250.82 | -240.54 | -8.64 | 55.67 | **no** | +273.95 | -263.68 | -13.13 |
| vs-shover | +36.11 | +84.57 | -48.46 | -3.85 | 25.17 | **no** | +50.54 | -14.43 | -0.68 |
| vs-textured | +66.19 | +111.12 | -44.93 | -3.72 | 24.18 | **no** | +102.71 | -36.52 | -12.97 |
| five-max-lineup | +74.52 | +165.18 | -90.66 | -3.79 | 47.81 | **no** | +113.16 | -38.64 | -3.37 |

### member b (init seed 402)

| channel | candidate | champ | diff vs champ | t | MDE | held? | inc | diff vs inc | t |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| vs-median | +76.81 | +116.45 | -39.64 | -4.64 | 17.09 | **no** | +103.46 | -26.65 | -5.87 |
| vs-nit | +51.14 | +62.30 | -11.17 | -4.00 | 5.59 | **no** | +65.81 | -14.67 | -6.67 |
| vs-station | +43.46 | +250.82 | -207.36 | -8.74 | 47.44 | **no** | +273.95 | -230.49 | -10.77 |
| vs-shover | +39.26 | +84.57 | -45.31 | -2.46 | 36.77 | **no** | +50.54 | -11.28 | -0.69 |
| vs-textured | +70.16 | +111.12 | -40.96 | -3.87 | 21.17 | **no** | +102.71 | -32.55 | -6.55 |
| five-max-lineup | +123.14 | +165.18 | -42.04 | -1.24 | 67.94 | yes | +113.16 | +9.98 | +0.43 |

### member c (init seed 403)

| channel | candidate | champ | diff vs champ | t | MDE | held? | inc | diff vs inc | t |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| vs-median | +74.19 | +116.45 | -42.27 | -6.02 | 14.04 | **no** | +103.46 | -29.28 | -8.32 |
| vs-nit | +56.04 | +62.30 | -6.26 | -1.80 | 6.97 | yes | +65.81 | -9.76 | -3.67 |
| vs-station | +16.19 | +250.82 | -234.62 | -9.96 | 47.14 | **no** | +273.95 | -257.76 | -13.30 |
| vs-shover | +24.12 | +84.57 | -60.45 | -4.35 | 27.79 | **no** | +50.54 | -26.42 | -1.56 |
| vs-textured | +71.94 | +111.12 | -39.18 | -3.89 | 20.16 | **no** | +102.71 | -30.77 | -3.40 |
| five-max-lineup | +94.88 | +165.18 | -70.31 | -3.34 | 42.09 | **no** | +113.16 | -18.28 | -1.31 |

## 4. vs-p3 channel (repaired strength-aware opponent, 16 seeds)

The P3 opponent carries the 2026-08-16 price-support clamp. Derived MDE at
16 seeds is ~9.6 BB/100; the run's own 2·sd/√n is given per row so the
effect is compared against the threshold this run actually supports.

| member | candidate | incumbent | diff (cand − inc) | t | 2·sd/√n | resolved? |
|---|---:|---:|---:|---:|---:|---|
| a | +50.76 | +64.65 | -13.89 | -2.49 | 11.13 | resolved |
| b | +48.84 | +64.65 | -15.81 | -2.68 | 11.79 | resolved |
| c | +55.86 | +64.65 | -8.79 | -2.05 | 8.59 | resolved |

Trivial floors on the same channel and seeds (member a's run):

| arm | BB/100 |
|---|---:|
| candidate-v7-0001c | +64.65 |
| candidate-v8-0002a | +50.76 |
| trivial-always_aggress_large | +123.87 |
| trivial-always_aggress_small | +120.65 |
| trivial-always_check_call | -11.31 |
| trivial-always_fold | -26.02 |
| trivial-uniform_random_legal | -135.89 |

## 5. Strength separation per street (canonical metric)

Field benchmark, `field-separation-canonical-2026-08-16.json`: overall
**+0.386** [+0.373, +0.398]; preflop +0.406, flop +0.275, turn +0.170,
river +0.331.

| member | scope | decisions | separation | 95% CI | field | gap to field |
|---|---|---:|---:|---|---:|---:|
| a | overall | 6069 | **+0.0859** | [+0.0661, +0.1057] | +0.386 | -0.3001 |
| a | preflop | 2761 | **+0.0775** | [+0.0499, +0.1051] | +0.406 | -0.3285 |
| a | flop | 1577 | **+0.0349** | [-0.0049, +0.0747] | +0.275 | -0.2401 |
| a | turn | 1014 | **+0.1267** | [+0.0716, +0.1818] | +0.170 | -0.0433 |
| a | river | 717 | **+0.2524** | [+0.1901, +0.3147] | +0.331 | -0.0786 |
| b | overall | 6252 | **+0.1076** | [+0.0889, +0.1262] | +0.386 | -0.2784 |
| b | preflop | 2774 | **+0.0660** | [+0.0377, +0.0944] | +0.406 | -0.3400 |
| b | flop | 1738 | **+0.0682** | [+0.0350, +0.1014] | +0.275 | -0.2068 |
| b | turn | 1054 | **+0.2056** | [+0.1583, +0.2528] | +0.170 | +0.0356 |
| b | river | 686 | **+0.2271** | [+0.1724, +0.2819] | +0.331 | -0.1039 |
| c | overall | 5780 | **+0.0987** | [+0.0789, +0.1186] | +0.386 | -0.2873 |
| c | preflop | 2682 | **+0.0922** | [+0.0652, +0.1192] | +0.406 | -0.3138 |
| c | flop | 1455 | **+0.0618** | [+0.0230, +0.1006] | +0.275 | -0.2132 |
| c | turn | 919 | **+0.1155** | [+0.0521, +0.1789] | +0.170 | -0.0545 |
| c | river | 724 | **+0.2818** | [+0.2144, +0.3491] | +0.331 | -0.0492 |

Incumbent `candidate-v7-0001c` on the identical sims (from member a's run):

| scope | decisions | separation | 95% CI |
|---|---:|---:|---|
| overall | 4523 | +0.1695 | [+0.1492, +0.1899] |
| preflop | 2554 | +0.1526 | [+0.1260, +0.1791] |
| flop | 1127 | +0.1586 | [+0.1193, +0.1980] |
| turn | 515 | +0.2226 | [+0.1540, +0.2912] |
| river | 327 | +0.3038 | [+0.2310, +0.3766] |

## 6. The composed-value objective did not learn (held-out, per street)

Measured on the stdlib serve path (`_forward_v3` + `compose_branch_values`),
scored with the trainer's own reduction, against the constant-predictor
baseline. Three impossible-by-construction gates passed first: targets are
centered per decision (max |sum| 2.7e-16); a constant prediction is
annihilated by the centering so its loss is independent of the constant;
and scoring the model's own composed values as targets gives exactly 0.0.
The stdlib path reproduces the trainer's reported validation `value_mse` to
all eight decimals.

| member | street | val decisions | model MSE | constant MSE | vs constant |
|---|---|---:|---:|---:|---:|
| a | ALL | 506 | 0.00919480 | 0.00925810 | +0.68% |
| | preflop | 388 | 0.00735995 | 0.00738443 | +0.33% |
| | flop | 87 | 0.01860897 | 0.01909523 | +2.55% |
| | turn | 16 | 0.01013935 | 0.00881315 | -15.05% |
| | river | 15 | 0.00104634 | 0.00114303 | +8.46% |
| b | ALL | 506 | 0.00938632 | 0.00925810 | -1.38% |
| | preflop | 388 | 0.00741178 | 0.00738443 | -0.37% |
| | flop | 87 | 0.01936480 | 0.01909523 | -1.41% |
| | turn | 16 | 0.01070102 | 0.00881315 | -21.42% |
| | river | 15 | 0.00118342 | 0.00114303 | -3.53% |
| c | ALL | 506 | 0.00933737 | 0.00925810 | -0.86% |
| | preflop | 388 | 0.00744400 | 0.00738443 | -0.81% |
| | flop | 87 | 0.01878009 | 0.01909523 | +1.65% |
| | turn | 16 | 0.01018866 | 0.00881315 | -15.61% |
| | river | 15 | 0.00263668 | 0.00114303 | -130.68% |

