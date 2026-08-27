# Head degeneracy audit

artifact `artifacts\candidates\candidate-v7-0001c.approved.manifest.json` · journal `.arena-training.jsonl` · 4795 decisions

> Predicate: does the head return `out_b` **bit-identically**? A constant output carries no information, and unlike a pre-activation test this stays measurable if the architecture changes.

## 0. The instrument, before any result

| control | reads | must be | verdict |
|---|---|---|---|
| zeroed `out_w` — can only emit the bias | 100.0% | 100.0% | PASS |
| forced-live tower — can never die | 0.0% | 0.0% | PASS |
| self-null — same weights twice | identical | identical | PASS |

## 1. Per head

| head | constant | acts first | facing a bet | folds among constant rows |
|---|---|---|---|---|
| `action_value` | **35.6%** (1707/4795) | 68.96% | 22.27% | 76 |
| `behavior_prior` | **0.0%** (0/4795) | 0.0% | 0.0% | 0 |
| `residual_scale` | **0.0%** (0/4795) | 0.0% | 0.0% | 0 |
| `state_value` | **33.16%** (1590/4795) | 28.56% | 35.0% | 282 |

## 2. By street

| head | preflop | flop | turn | river |
|---|---|---|---|---|
| `action_value` | 27.4% | 45.09% | 46.58% | 55.21% |
| `behavior_prior` | 0.0% | 0.0% | 0.0% | 0.0% |
| `residual_scale` | 0.0% | 0.0% | 0.0% | 0.0% |
| `state_value` | 35.85% | 35.35% | 24.94% | 7.72% |

