# candidate-v8-0001 — evaluation gauntlet (2026-08-16)

**Instrument**: `tools/evaluate_v8.py` (additive wrapper; all measurement logic
imported unchanged from `tools/evaluate_policies.py`). Carry-over sessions,
6,000-chip stacks at 50/100, equity_trials 80 — the same configuration as the
published noise floor (`noise-floor-2026-08-15.json`). No promotion; this
artifact must not be deployed (its own manifest says so);
`artifacts/approved.json` untouched.

## Verdict

**Not promotable in this state — and not designed to be.** The reasons, most
binding first:

1. **Every battery channel is worse than the frozen champion beyond its
   published 8-seed MDE** (paired t from −2.71 to −14.53), and two channels
   are negative outright (vs-station −20.59, vs-shover −3.19, both below the
   `always_check_call` floor, though not resolvably: t −0.80 / −1.09).
   DECISIONS §5 requires holding the batteries; they are not held.
2. **The duel against `candidate-v7-0001c` is UNRESOLVED**: +9.77 BB/100 to
   v8 over 16 seat-swapped seeds (sd 17.21, se 4.30, t = +2.27). That clears
   the run's own 2·sd/√n (8.61) but sits below the 16.78 BB/100 known seed
   spread, and the rule is explicit: below the resolvable spread is neither a
   win nor a loss. Suggestive, not evidence. (For calibration: v7-0002c's
   +9.72 at 16 seeds was the same magnitude.)
3. **Strength separation moved the wrong way relative to the incumbent on the
   same instrument**: all-streets +0.110 [0.089, 0.131] for v8 against
   +0.170 [0.149, 0.190] for v7-0001c, with v8 aggressing *more* (71.5% vs
   64.6% of decisions). Preflop is +0.067 [0.041, 0.093] — positive with a CI
   excluding zero, which is the design's minimal preflop acceptance — but the
   incumbent posts +0.153 on the identical sims.
4. **Only one of the three trained init seeds has an artifact**, so §6.1
   ("every seed gauntleted") cannot be satisfied.

**Why this does not falsify the v8 design**: the artifact's own manifest says
Phase A trains component heads only and "must not be deployed." The measured
failure mechanism is precisely what Phase B exists to fix: `fold_through` is
an opponent-unconditional field average (~50% folds in the S14 archive), so
against archetypes that never fold (station: 5%) the composition keeps buying
fold equity that does not exist — a 300-hand probe shows v8 aggressing 71.4%
of vs-station decisions at a median wager of 0.98×(pot+call) with mean
strength 0.551. The batteries are exactly the out-of-distribution opponents
for field-fitted heads. What the gauntlet *does* establish: the composed
serve path plays (no fail-closed collapse — its behavior is far from the
heuristic fallback), does not lose to the incumbent head-to-head, busts less
than it in the duel (1.32 vs 1.48 busts/100h), and beats the meaningful
trivial floors on most channels. The honest next steps are the P3 opponent
(Phase B labels + a strength-aware battery leg) and exporting all three init
seeds before any promotion question is asked.

## Trivial-baseline floor (8 seeds, same channels/seeds/session mode as the candidate)

BB/100, seat "hero", carry-over sessions:

| floor | vs-median | vs-nit | vs-station | vs-shover | vs-textured | five-max |
|---|---|---|---|---|---|---|
| always_fold | -28.44 | -5.41 | -46.04 | -72.69 | -30.28 | -28.58 |
| always_check_call | -2.64 | 14.60 | 2.18 | +67.50 | 4.61 | 43.48 |
| always_aggress_small | +334.53 | +95.48 | +155.50 | +67.50 | +350.45 | +399.09 |
| always_aggress_large | +399.20 | +105.89 | +179.89 | +67.50 | +397.10 | +438.06 |
| uniform_random_legal | -107.65 | +1.20 | -489.70 | -341.11 | -141.38 | -491.12 |

Reading this table honestly: **`always_aggress_large` beats the published
champion mean on four of six channels while looking at nothing.** Against
card-blind archetypes maximum aggression is trivially optimal (constant fold
rates price nothing), so the aggress floors are a ceiling the batteries
cannot referee — they are the DECISIONS §4.1 defect made quantitative, and
they are why the batteries gate fit, not quality. The floors a candidate
must clear meaningfully are `always_fold`, `always_check_call`, and
`uniform_random_legal`; per-seed paired comparisons are in the JSON. (The
three identical +67.50 vs-shover entries are the same action sequence — every
hand ends all-in preflop against a permanent shover whether hero calls or
raises first — and +67.5 is ~1.2σ from the zero a coinflip predicts.)

## Batteries — candidate-v8-0001, 8 seeds, against the published references

Champion = `heuristic-aggressive-v6`, the frozen noise-floor reference,
paired seed-for-seed on identical match/opponent seeds. MDE from
`noise-floor-2026-08-15.json` at 8 seeds.

| channel | v8 BB/100 | champion | paired diff | MDE@8 | beyond MDE? |
|---|---|---|---|---|---|
| vs-median | +114.76 | +149.72 | -32.09 | 17.06 | **yes, worse** |
| vs-nit | +70.01 | +79.72 | -9.96 | 3.22 | **yes, worse** |
| vs-station | **-20.59** | +311.54 | -332.09 | 42.13 | **yes, worse** |
| vs-shover | **-3.19** | +158.29 | -164.23 | 46.58 | **yes, worse** |
| vs-textured | +105.67 | +165.61 | -62.30 | 17.13 | **yes, worse** |
| five-max-lineup | +92.62 | +214.93 | -113.93 | 40.67 | **yes, worse** |

**Every battery channel is broken beyond its published MDE**, and two are
*negative*: vs-station (-20.59 — below even the `always_check_call` floor's
+2.18) and vs-shover (-3.19, against `always_check_call`'s +67.50). Ruin is
also elevated where it matters: 4.04 busts/100h vs-station (champion 2.23),
1.47 on five-max (champion 0.65). Under DECISIONS §5 — "hold the batteries
within the published MDE" — this alone is disqualifying for promotion in this
state, before the duel is even read.

Paired per-seed t against the champion, per channel: vs-median −2.71,
vs-nit −2.91, vs-station −14.53, vs-shover −11.88, vs-textured −7.90,
five-max −10.30.

Floor comparisons (paired per-seed, same seeds): v8 beats `always_fold` and
`uniform_random_legal` on all six channels (weakest: vs-station over
always_fold, t = +1.05). It beats `always_check_call` on four channels
(t +1.16 to +10.37) and is *below* it on vs-shover (−70.69, t = −1.09) and
vs-station (−22.77, t = −0.80) — both unresolved at 8 seeds, both damning
directionally. It loses to both aggress floors everywhere, as does everything
measured on card-blind opponents, including the champion on 4/6 channels.

## Duel vs candidate-v7-0001c — 16 seat-swapped seeds, 2,000 hands per orientation

**+9.77 BB/100 to candidate-v8-0001** (per-seed seat-means; sd 17.21, se
4.30, **t = +2.27**). Both orientations of every seed share the simulator
seed; zero-sum and equal-hands invariants asserted in-run.

- Empirical MDE of this run: 2·sd/√16 = **8.61** — the margin clears it.
- Known seed spread: **16.78 BB/100** — the margin does not clear it.
- **Verdict: UNRESOLVED** (the binding rule reports any margin below the
  resolvable spread as neither win nor loss).
- Ruin inside the duel: v8 1.3234 busts/100h, 0.4639 busts/session vs the
  incumbent's 1.4797 and 0.5186 — the composed policy is the *less* ruinous
  seat.
- Per-seed seat-means: [18.22, 37.97, 1.93, 1.95, 30.37, −3.14, −2.90,
  15.96, 19.56, −19.82, −1.84, −12.09, 42.54, 7.69, 9.15, 10.73] — 11 of 16
  positive.

## Strength separation — canonical `strength_metric` percentile scale

Dedicated recording sims (2,000 HU hands vs median + 1,200 five-max hands,
identical seeds for both policies; ~4.5k decisions each). Separation =
mean strength when aggressing − mean strength when folding; the recorded
family is the submitted action's.

| street | v8-0001 | 95% CI | v7-0001c | 95% CI |
|---|---|---|---|---|
| preflop | +0.067 | [0.041, 0.093] | +0.153 | [0.126, 0.179] |
| flop | +0.135 | [0.087, 0.183] | +0.159 | [0.119, 0.198] |
| turn | +0.256 | [0.195, 0.317] | +0.223 | [0.154, 0.291] |
| river | +0.253 | [0.165, 0.341] | +0.304 | [0.231, 0.377] |
| **all** | **+0.110** | [0.089, 0.131] | **+0.170** | [0.149, 0.190] |

Readings: v8's preflop separation is positive with a CI excluding zero (the
design's minimal preflop acceptance) — but the incumbent is higher on the
same sims, and higher overall, while v8 aggresses more (71.5% vs 64.6% of
decisions). Note the incumbent measures +0.170 here against its **+0.020
live-field measurement**: the card-blind sim instrument and the replay-side
field instrument are not comparable, and neither of these numbers may be set
against the +0.150 field benchmark until that benchmark is recomputed on
`strength_metric` (V8_DESIGN §2). Within-instrument, the comparison stands:
**the composition did not buy more strength-aware aggression than the
incumbent already had.**


## Instrument validation before any result was read

- **Self-duel null check**: candidate-v8-0001 vs itself, 2 seeds both seat
  orders — paired diffs identically zero (exact rename-mirror), so the task
  wiring, seat swap, and label plumbing are sound.
- **Trivial-floor invariant**: `always_fold` vs the permanent shover must sit
  just above −75 BB/100 by construction (blind attrition, plus the forced
  final all-in-blind showdown it cannot decline). Measured: −72.69.

## Trainer context (three init seeds, one artifact)

Phase-A supervised run, three init seeds on byte-identical data:

| init seed | best epoch | val loss (total) |
|---|---|---|
| 101 | 6 | 2.302717 |
| **202 (exported)** | 7 | **2.300128** |
| 303 | 5 | 2.343328 |

Selection was minimum validation loss — a **gate, never a selector**
(V8_DESIGN §6.1) — and only seed 202 was persisted as an artifact. **This
gauntlet therefore covers one of the three trained seeds**; the design asks
for all three to be gauntleted, and validation loss has ranked seeds
backwards against duel results once already. Seeds 101/303 have no artifacts
to field.

## Caveats that bound every number here

1. **Every battery opponent is card-blind** (DECISIONS §5): the batteries are
   a fit diagnostic, not evidence of generalisation. Quantified in this run:
   `always_aggress_large` — a policy that looks at nothing — posts the
   highest battery numbers of anything measured.
2. The strength-separation field benchmark (+0.150) was measured on the S14
   replay window with a similar but not identical percentile metric; treat it
   as directional until recomputed on `strength_metric` per V8_DESIGN §2.
3. The duel verdict rule: a margin below the resolvable spread (max of the
   known 16.78 BB/100 seed spread and the run's own 2·sd/√n) is
   **UNRESOLVED**, never a win or a loss.
4. The residual head's output layer is exactly zero on this artifact
   (untrained in Phase A by design), verified on the weights file — the §6.5
   residual audit is trivially satisfied and a residual-off ablation is
   byte-identical.
