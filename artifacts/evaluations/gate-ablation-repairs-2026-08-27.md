# Gate ablation on a card-aware opponent

subject `artifacts/candidates/candidate-v7-0001c.approved.manifest.json` · 48 seeds/arm · 1152 matches · 4478.2s

> Every arm is the same artifact one gate edit apart. Nothing here trains, promotes or deploys.

## 0. The instrument, before any result

| check | result | verdict |
|---|---|---|
| null mirror (`live` under a second label) — every paired difference must be exactly 0 | 0 channels differ | PASS |
| chip conservation | 0 violations | PASS |
| reproduction gate — `live` vs the frozen `p3-gate-2026-08-16` incumbent arm | vs-p3 identical (16 seeds), vs-median identical (16 seeds), vs-station identical (16 seeds) | PASS |

Depth **60.0bb** (6000 chips at a 100 big blind), 80 equity trials, scale 1.0, fit `artifacts/p3/p3-fit.json`. Published instrument for comparison: {'starting_stack': 6000, 'equity_trials': 80, 'scale': 1.0, 'seeds': 16}.

## 1. Arms

| arm | `vs-p3` | `vs-median` | `vs-station` |
|---|---|---|---|
| live | +61.24 (sd 14.6) | +109.19 (sd 22.72) | +277.04 (sd 61.18) |
| revert-cap | +68.82 (sd 18.91) | +141.52 (sd 25.52) | +299.87 (sd 65.22) |
| revert-calls | +77.26 (sd 20.76) | +161.91 (sd 22.98) | +327.30 (sd 58.66) |
| revert-all | +77.73 (sd 21.22) | +175.63 (sd 28.04) | +345.22 (sd 61.65) |
| fix-a | +65.18 (sd 17.09) | +137.83 (sd 22.08) | +299.17 (sd 56.53) |
| winnable-price | +61.78 (sd 14.4) | +108.32 (sd 24.18) | +275.70 (sd 58.56) |
| condition-unpriced | +58.56 (sd 12.77) | +104.31 (sd 19.75) | +270.72 (sd 51.11) |
| live-mirror | +61.24 (sd 14.6) | +109.19 (sd 22.72) | +277.04 (sd 61.18) |

## 2. Every arm against `live`, paired on shared seeds

### `revert-cap`

| channel | BB/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | +7.58 | 2.99 | 5.08 | revert-cap ahead |
| `vs-median` | +32.33 | 9.22 | 7.01 | revert-cap ahead |
| `vs-station` | +22.83 | 4.13 | 11.06 | revert-cap ahead |

| channel | busts/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | +0.0200 | 1.81 | 0.0289 | UNRESOLVED |
| `vs-median` | +0.0300 | 2.42 | 0.0289 | revert-cap busts more |
| `vs-station` | +0.2700 | 8.14 | 0.0664 | revert-cap busts more |

### `revert-calls`

| channel | BB/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | +16.02 | 5.89 | 5.44 | revert-calls ahead |
| `vs-median` | +52.72 | 12.6 | 8.37 | revert-calls ahead |
| `vs-station` | +50.26 | 8.54 | 11.77 | revert-calls ahead |

| channel | busts/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | +0.0800 | 5.45 | 0.0318 | revert-calls busts more |
| `vs-median` | +0.0400 | 2.14 | 0.0404 | UNRESOLVED |
| `vs-station` | +0.3300 | 6.44 | 0.101 | revert-calls busts more |

### `revert-all`

| channel | BB/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | +16.49 | 5.34 | 6.18 | revert-all ahead |
| `vs-median` | +66.44 | 13.49 | 9.85 | revert-all ahead |
| `vs-station` | +68.18 | 10.72 | 12.72 | revert-all ahead |

| channel | busts/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | +0.0900 | 4.89 | 0.0375 | revert-all busts more |
| `vs-median` | +0.0700 | 3.66 | 0.0404 | revert-all busts more |
| `vs-station` | +0.5400 | 10.57 | 0.101 | revert-all busts more |

### `fix-a`

| channel | BB/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | +3.94 | 1.69 | 4.66 | UNRESOLVED |
| `vs-median` | +28.64 | 8.27 | 6.93 | fix-a ahead |
| `vs-station` | +22.14 | 3.96 | 11.18 | fix-a ahead |

| channel | busts/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | +0.0600 | 3.99 | 0.0318 | fix-a busts more |
| `vs-median` | +0.0200 | 1.21 | 0.0289 | UNRESOLVED |
| `vs-station` | +0.2700 | 7.04 | 0.0779 | fix-a busts more |

### `winnable-price`

| channel | BB/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | +0.54 | 1.27 | 0.85 | UNRESOLVED |
| `vs-median` | -0.88 | -1.18 | 1.48 | UNRESOLVED |
| `vs-station` | -1.33 | -1.06 | 2.52 | UNRESOLVED |

| channel | busts/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | -0.0000 | -0.81 | 0.0115 | UNRESOLVED |
| `vs-median` | -0.0000 | -1.0 | 0.0087 | UNRESOLVED |
| `vs-station` | -0.0000 | -0.26 | 0.0173 | UNRESOLVED |

### `condition-unpriced`

| channel | BB/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | -2.68 | -1.55 | 3.46 | UNRESOLVED |
| `vs-median` | -4.88 | -2.11 | 4.62 | live ahead |
| `vs-station` | -6.32 | -1.33 | 9.49 | UNRESOLVED |

| channel | busts/100 difference | t | paired MDE | verdict |
|---|---|---|---|---|
| `vs-p3` | +0.0100 | 0.66 | 0.026 | UNRESOLVED |
| `vs-median` | -0.0300 | -2.23 | 0.0289 | condition-unpriced busts less |
| `vs-station` | -0.1200 | -2.92 | 0.0837 | condition-unpriced busts less |

## Caveats

- P3's aggression and shove rate are still card-blind knobs. Only its folding is card-aware, so a policy can be punished for the prices it lays and still not for the hands it shows down.
- `vs-p3` is heads-up. Multiway strength-aware play is untested.
- A positive difference means the reverted arm scored higher, i.e. evidence *against* the change that arm reverts.
- Batteries remain a fit diagnostic. A trivial floor still beats every real policy on these channels, so no arm's absolute BB/100 is evidence of generalisation.
