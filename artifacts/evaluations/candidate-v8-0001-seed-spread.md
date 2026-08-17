# candidate-v8-0001 — three-init-seed duel spread

**Deliverable: the init-seed spread is 9.71 BB/100** against a reference spread of 16.78 BB/100. **Verdict: UNRESOLVED.**

Instrument: `tools/evaluate_v8.py`, the same invocation as the first gauntlet — 16 seat-swapped duel seeds at 2,000 hands per orientation, 8 battery seeds, `--equity-trials 80`, `--workers 6`, 6,000-chip carry-over sessions at 50/100. Incumbent `candidate-v7-0001c` in every duel. No promotion; `artifacts/approved.json` untouched.

## Duel vs the incumbent, per init seed

| init seed | artifact | margin BB/100 | sd | se | t | own MDE (2·sd/√16) | verdict |
|---|---|---|---|---|---|---|---|
| 101 | `candidate-v8-0001a` | **+10.61** | 18.4 | 4.6 | +2.31 | 9.2 | UNRESOLVED |
| 202 | `candidate-v8-0001` | **+9.77** | 17.21 | 4.3 | +2.27 | 8.61 | UNRESOLVED |
| 303 | `candidate-v8-0001c` | **+0.90** | 8.98 | 2.24 | +0.40 | 4.49 | UNRESOLVED |

Spread = max − min = 9.71 BB/100 (best seed 101, worst seed 303). Reference: v7-0002 three-init-seed duel spread (a -7.06, b -1.88, c +9.72) on this instrument; DECISIONS.md §4.7.

## Paired cross-seed contrasts (common evaluation seeds)

Every gauntlet duels the same incumbent on the same 16 evaluation seeds, so the init seeds are paired by common random numbers and the shared evaluation noise cancels. This is the sensitive test of whether the init seeds differ *at all*.

| contrast | mean | sd | se | t |
|---|---|---|---|---|
| 101 − 202 | +0.84 | 13.17 | 3.29 | +0.26 |
| 101 − 303 | +9.71 | 17.72 | 4.43 | +2.19 |
| 202 − 303 | +8.86 | 15.85 | 3.96 | +2.24 |

## Batteries — channels held within the published MDE

A channel is *held* when the candidate is not worse than the frozen champion by more than the published 8-seed MDE (`noise-floor-2026-08-15.json`), which is the DECISIONS §5 / V8_DESIGN §6.2 condition.

| channel | MDE@8 | seed 101 | seed 202 | seed 303 |
|---|---|---|---|---|
| vs-median | 17.06 | -39.33 (t -3.78) **broken** | -32.09 (t -2.71) **broken** | -41.39 (t -3.40) **broken** |
| vs-nit | 3.22 | -10.32 (t -3.00) **broken** | -9.96 (t -2.91) **broken** | -10.71 (t -5.34) **broken** |
| vs-station | 42.13 | -320.18 (t -51.95) **broken** | -332.09 (t -14.53) **broken** | -291.65 (t -16.82) **broken** |
| vs-shover | 46.58 | -164.89 (t -10.99) **broken** | -164.23 (t -11.88) **broken** | -156.60 (t -8.70) **broken** |
| vs-textured | 17.13 | -70.31 (t -7.71) **broken** | -62.30 (t -7.90) **broken** | -66.87 (t -9.35) **broken** |
| five-max-lineup | 40.67 | -127.64 (t -3.13) **broken** | -113.93 (t -10.30) **broken** | -104.40 (t -4.61) **broken** |

| init seed | channels held | channels broken |
|---|---|---|
| 101 | 0 | 6 |
| 202 | 0 | 6 |
| 303 | 0 | 6 |

## Reading

- The spread (9.71) is 1.37x the mean margin across seeds (7.09). Seed-to-seed variation is the same size as the effect being claimed.
- Per-seed margins span +0.90 to +10.61; 2 of 3 seeds clear their own empirical MDE, so the head-to-head win is not reproducible across initialisations.
- All 3 per-seed duels are individually UNRESOLVED: True.
- Batteries: 18 of 18 seed-channel pairs are worse than the champion beyond the published MDE. The battery failure is the same on every seed, so it is structural, not initialisation luck.
- Multiple comparisons: three pairwise contrasts were computed. A Bonferroni-corrected two-sided 5% threshold at df=15 is |t| ≈ 2.69, which none of the contrasts reaches — so even the apparent separation between init seeds is not established at corrected significance.

## Wiring controls

- seed 101: self-duel mirror exact = `True` (paired diffs identically zero while the underlying match moves real chips)
- seed 202: self-duel mirror exact = `True` (paired diffs identically zero while the underlying match moves real chips)
- seed 303: self-duel mirror exact = `True` (paired diffs identically zero while the underlying match moves real chips)

Verdict rule: SEED-DEPENDENT if the init-seed spread reaches the 16.78 reference; SEED-ROBUST only if every seed's margin shares a sign and clears its own empirical MDE; UNRESOLVED otherwise.

