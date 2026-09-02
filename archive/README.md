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
| `data/` | **no** — gitignored, ~777 MB | `corpora/` (the three v7 harvest corpora), `deadhead/` (the 2026-08-27 retrain set), `eval-bundle-2026-08-14/` (`build/` + `bundle.zip`, pre-rename package name), `antislop-audit-2026-08-12/`, `arena-training.jsonl.backup-2026-08-26` |

Not archives: `artifacts/evaluations/` (dated measurement records, never
edited) and `.handoff/archive/` (the handoff-side archive, same instruction).
