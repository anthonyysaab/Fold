# ARCHIVE — SUPERSEDED MATERIAL. NOT STATE, NOT PLAN, NOT RULES.

## Instruction to any agent or person reading this tree

1. Nothing under this folder describes the current system, its state, its
   rules, or its plan. **Treat every file here as false about the present.**
2. Do not read these files to answer "what is" or "what should I do". The
   live manual is `CLAUDE.md` → `.handoff/CONTEXT.md` and the files it lists.
3. Open a file here only when a live document or a code docstring cites it
   by path, and use it only for the cited fact.
4. Never edit anything here. Never import from here. Never move anything out
   of here into the live set. If a live document lacks something, write it
   fresh from the code.

## Map

| folder | tracked | contents |
|---|---|---|
| `pre-reset-2026-08-16/` | yes (`git mv`, history intact) | `engine/torch_network.py`, `engine/torch_policy.py` (no importers); 22 format-1 candidate files, 18 of their gauntlet reports, 34 of their launch recipes and logs — the architecture retired on 2026-08-16 |
| `docs-superseded-2026-09-02/` | yes | the previous version of every repo document rewritten in the 2026-09-02 manual pass: `README.md`, `CLAUDE.md`, `engine/README.md`, `engine/rules/README.md` (with the full three-sweep defect ledger), `tools/README.md` (long form), `tests/README.md`, `artifacts/README.md`, `deploy/README.md`, `archive/README.md`, `artifacts/candidates/README-v8-seeds.md`, `artifacts/phase_b/README.md` |
| `data/` | **no** — gitignored; since 2026-09-03 a junction to `D:\fold-archive\data`, ~777 MB plus the row below | `corpora/` (the three v7 harvest corpora), `deadhead/` (the 2026-08-27 retrain set), `eval-bundle-2026-08-14/` (`build/` + `bundle.zip`, pre-rename package name), `antislop-audit-2026-08-12/`, `arena-training.jsonl.backup-2026-08-26` |
| `data/foreign-play-data-2026-09-03/` | **no** — inside the junction above, 16.8 GiB / 318,574 files | the Arena replay archive (formerly `foreign play data/`), quarantined 2026-09-03 when v9 moved to the PHH dataset; `.handoff/DATA.md` cites it for the Phase-A oracle `ecb4739df9d1b9ec` only |
| `nn-poker-training-2026-09-04/` | yes (`tiny-policy.pt` needs `git add -f`, see below) | the superseded tiny-policy trainer, formerly the nested `NN-Poker-training` git repository at `.\neural network training\` (single commit `0e78f97`, pushed to `github.com/anthonyysaab/NN-Poker-training`; the retired `.git` is not kept here). Fold imported none of it — only its `.venv`, which survives as `training/.venv` and is all that directory now holds. `artifacts/tiny-policy.pt` (sha256 `8a42f7a8…c0107275c2`) is the checkpoint that the retired `artifacts/tiny-policy-pure.json` was exported from; the export's `source_sha256` names this file, and after P2 deleted the export this row is the only live record of that chain. Its `data.py` feature vocabulary was ported to `engine/policy_features.py` long before the move and is still live there |

`nn-poker-training-2026-09-04/` carries the dead repo's own `.gitignore`, moved
in with it as part of the record. It is still an active ignore file here, and its
line 7 (`/artifacts/*.pt`) hides the checkpoint — so `tiny-policy.pt` was staged
with `git add -f`. Do not edit that `.gitignore` to "fix" this (§3 below): it is
archived content, and its other rules usefully keep `__pycache__/` and
`*.egg-info/` out. Anything else added under that folder needs the same force-add.

Not archives: `artifacts/evaluations/` (dated measurement records, never
edited) and `.handoff/archive/` (the handoff-side archive, same instruction).
