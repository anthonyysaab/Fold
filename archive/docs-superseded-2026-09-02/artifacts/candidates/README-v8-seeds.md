> **ARCHIVED 2026-09-02 — SUPERSEDED. NOT STATE, NOT PLAN, NOT RULES.**
> Do not use this file to learn what the system is, what its state is, or what to do.
> The live manual is `CLAUDE.md` → `.handoff/CONTEXT.md`. Open this file only for a
> specific fact that a live document cites here by path. Never edit it, never restore it.

# v8 Phase-A seed artifacts — seed → artifact map

Written 2026-08-16. Covers the three `v8-composed-value` Phase-A candidates
trained from `artifacts/phase_a/phase-a-dataset.jsonl.gz`.

> **2026-09-02**: a dated record. The trainer module is now `engine.v8_trainer`
> (the package was renamed from `devfun_poker_playground` on 2026-08-28), and
> `artifacts/approved.json` has pointed at `candidate-v9-0003b` since
> 2026-09-01. The sha256 map and the reproducibility claims below stand.

V8_DESIGN §6.1 requires **three init seeds, every seed gauntleted**, with
validation loss acting as a gate and never as a selector. The original run
(`--init-seeds 101 202 303`) trained all three but exported only the
minimum-validation-loss seed, which left §6.1 unsatisfiable: two of the three
seeds had no artifact to gauntlet. Seeds 101 and 303 were re-trained and
exported here so all three can enter the duel.

## The map

| artifact | init seed | member | weights sha256 |
|---|---|---|---|
| `candidate-v8-0001a` | 101 | a | `d9b2fff542579b6319d745dfe7b3505889dc6a95a7c128f8549711b6f6c84cb5` |
| `candidate-v8-0001`  | 202 | **b** (unsuffixed — the original export) | `4a0523533b3b3fa89d8350d7400f7ec833f7c54726c35a9071cd5c7be23f0f50` |
| `candidate-v8-0001c` | 303 | c | `289f016ceaf8c709d1c9006e96d9f0f57e1744c5fc0c957c2488363106cd924e` |

**`candidate-v8-0001` is the seed-202 'b' member of this triple.** It carries
no `b` suffix because it was exported before the other two existed; it was not
renamed, because its sha256 is cited by the gauntlet results already recorded
against it (duel vs `candidate-v7-0001c` +9.77 BB/100, t=+2.27, UNRESOLVED;
all six batteries worse than champion beyond MDE). Renaming it would silently
break those citations.

`weights_sha256` is `sha256` of the canonical weights encoding — the file
bytes with the trailing newline stripped, which is what each manifest records.

All three are `state: "candidate"`, `promotion: null`. `artifacts/approved.json`
still points at `candidate-v7-0001c` and was not touched.

## Validation losses (per head, held-out split)

Identical split for all three: `split_seed 17`, 8,276 train rows / 1,080 tables,
808 validation rows / 115 tables. 71,793 parameters.

| seed | artifact | fold_through | range | equity_called | **total** | best epoch |
|---|---|---|---|---|---|---|
| 101 | `-0001a` | **0.663203** | 1.588213 | 0.051301 | 2.302717 | 6 |
| 202 | `-0001`  | 0.678878 | **1.572724** | **0.048526** | **2.300128** | 7 |
| 303 | `-0001c` | 0.696233 | 1.593780 | 0.053316 | 2.343328 | 5 |

Train losses: 101 → 2.148702, 202 → 2.112162, 303 → 2.230826.

Two things worth carrying into the gauntlet:

- **The spread is 0.0432 total (1.9%), and 0.0018 (0.08%) between the top two.**
  On this project a 0.33% validation-loss difference once ranked seeds
  *backwards* against a 16.78 BB/100 duel spread. Nothing in this table is
  evidence about playing strength.
- **The heads disagree about the ordering.** Seed 202 wins the total, but seed
  101 has the best `fold_through` (0.663 vs 0.679, 2.3% better) — and
  `fold_through` is the exact head diagnosed as the mechanism of
  `candidate-v8-0001`'s battery losses (it was fitted on the real field's ~50%
  fold rate, so it prices fold equity that never-folding card-blind archetypes
  do not have). The total-loss selector picked the seed that is *worst but one*
  on the head under suspicion. This is a reason to gauntlet all three, not a
  prediction that 101 wins.

## Reproducibility (checked, not assumed)

Re-running seed 202 alone reproduces `candidate-v8-0001` **byte-for-byte**:
same weights file bytes, same sha256 `4a0523...0f50`, same per-head losses,
same `best_epoch` 7 and `optimizer_steps` 561. Seeds 101 and 303 likewise
reproduce the per-seed losses recorded in the original run's
`init_seeds_evaluated` block to all six decimals.

That matters beyond the determinism claim: seed 202 was the *second* of three
fits in the original process, and it reproduces when run *first and alone*.
Per-fit re-seeding is therefore complete — no RNG state leaks across fits — so
the single-seed 101 and 303 artifacts are the same models the original run
scored, not merely similar ones.

Controls run before those numbers were believed: recomputed sha256 matched each
manifest's recorded value (positive); the three seeds' weight payloads differ
pairwise with `model_version` stripped, so the hash is sensitive to weights and
not just to the version string (negative); `feature_normalization` is
byte-identical across all three, as it must be since it is computed from the
training split alone and cannot depend on `init_seed`.

Caveat on scope: determinism is established **on this machine** (Quadro RTX
3000, torch 2.13.0+cu130, Python 3.11.9) for this dataset and config. It is not
evidence of cross-GPU or cross-torch-version reproducibility.

## Commands

```
# from the repo root, CUDA interpreter
C:/Users/user/poker-nn-training/.venv/Scripts/python.exe -m devfun_poker_playground.v8_trainer \
    --model-version candidate-v8-0001a --init-seeds 101 --output-dir artifacts/candidates
C:/Users/user/poker-nn-training/.venv/Scripts/python.exe -m devfun_poker_playground.v8_trainer \
    --model-version candidate-v8-0001c --init-seeds 303 --output-dir artifacts/candidates
```

Every other config value is the trainer default, which is what the original
three-seed run used: 150 epochs (early stop patience 10), lr 1e-3, weight decay
0.01, dropout 0.2, warmup 200, batch 256, validation fraction 0.1, split seed
17, device cuda. The trainer was not modified; a single-seed `--init-seeds`
list makes its existing selection step a no-op.

Wall time: ~10 s per seed on the Quadro RTX 3000.

## Status

These are Phase-A component-head artifacts. Each manifest carries the standing
serve note: the composed-value serve path is a separate work item and **these
artifacts must not be deployed**. Promotion remains a separate, explicit,
human-authorised act.
