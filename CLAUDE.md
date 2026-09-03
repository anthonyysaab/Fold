# Fold-ver-4 — AGENT MANUAL

Revision 2026-09-03. This file plus `.handoff/` is the manual. Code wins over
docs. The Arena API wins over any state claim. Archives are not part of the
manual (§8). Previous version: `archive/docs-superseded-2026-09-02/CLAUDE.md`.

## 1. LIMITS — money. Not negotiable. Never part of any reset.

1. **Never pay an entry fee. Never rebuy.** `is_free_playground()` in
   `live_session.py` is the guard; never weaken it. **Never pass
   `--competition`** — it bypasses discovery and the guard.
2. **One live seat at a time.** Check for a running session BY PROCESS
   (`Get-Process python`), never by filename.
3. **Never smoke-test on a live session.**
4. **Never record, print, or paste the Arena API key** (`.arena-credentials`).
5. **Training never promotes, deploys, joins, or rebuys.** Promotion is an
   owner act through `tools/promote_candidate.py` (OLS gate for format 4; no
   gauntlet gate).
6. **Release the seat on every stop path.** There is no unclaim endpoint;
   `python -m tools.leave <competitionId>` is the recovery. Verify with
   `python tools/peek.py <competitionId>` → `activeTableCount 0`.
7. **Hard engine gates stay hard-coded.** Every live-path change ships as a
   default-OFF dial with a named ablation arm and a test that fails on the
   unfixed code. Nothing from a bust is fixed silently.
8. **A fresh season is a new join; rejoining the season you busted IS a rebuy.**

## 2. READ ORDER

1. `.handoff/CONTEXT.md` — environment, layout, key paths, machine move.
2. `.handoff/STATUS.md` — state. Verify against the API first (PROCEDURES §1).
3. `.handoff/DECISIONS.md` — binding rules, promotion bar, measured facts.
4. `.handoff/notes/V9_ARCHITECTURE.md` — the system as built.
5. `.handoff/NEXT.md` — the queue. `.handoff/PENDING_EDITS.md` — the defect log.
6. `.handoff/PROCEDURES.md` — how to run anything. `.handoff/DATA.md` — data.
7. Folder maps: `engine/README.md`, `engine/rules/README.md`, `tools/README.md`,
   `tests/README.md`, `artifacts/README.md`, `deploy/README.md`.

`.handoff/` is gitignored: no history, no remote. Edits there are
irreversible. On a machine move copy it first.

## 3. ENVIRONMENT (full table: CONTEXT §1)

| item | value |
|---|---|
| repo | `C:\Users\user\Fold-ver-4 (multiway)` — the ` (multiway)` is part of the name; quote it |
| tests / lint / live | `C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe` (3.11.9, **no torch** — correct). Plus `pokerkit==0.7.4` (`requirements-tools.txt`), imported by `tools/phh_*` only. |
| training | `C:\Users\user\Fold-ver-4 (multiway)\neural network training\.venv\Scripts\python.exe` (3.11.9, torch 2.13.0+cu130, CUDA available). Restored 2026-09-03 at this path, NOT the `C:\Users\user\poker-nn-training\` of earlier revisions. It is a **separate git repository** (`NN-Poker-training`) with its own `.venv`, living inside this working tree and gitignored. **Owner ruled 2026-09-03 that it stays nested** (`DECISIONS.md` §6); never commit it into Fold. Quote the path — it has spaces. |
| tools | `python -m tools.<name>` from the repo root |
| suite | 1,131 passed / 29 skipped / 370 subtests (2026-09-03, after the PHH build run). The 29 skips are 21 torch-availability and 8 that read the quarantined Arena archive. The torch half was RUN on the CUDA interpreter on 2026-09-03, not assumed: 104 passed / 2 skipped over the seven torch files, and those last 2 were a stale hardcoded venv path (now repo-relative), so nothing in that set skips silently. No pokerkit skips here — `pokerkit` 0.7.4 is installed, so the PHH tests really run. |
| ruff | 7 pre-existing errors (`tests/test_build_phase_b_corpus.py`, `tests/test_phase_a_dataset.py`, `tests/test_v8_trainer_phase_b.py`, `tools/build_phase_a_dataset.py`, `tools/measure_field_separation.py`, `tools/p3_gate.py`). Not a regression. |
| Arena, read-only | `python -m tools.api /api/arena/agent/me` — from PowerShell; Git Bash mangles the path |
| agent | **0Fold**, `@fold_ver_3`, `cmsnsh9er1ato12wxq5knep9d`. Quote and description stay exactly `Hello`. |
| git | `origin` = `github.com/anthonyysaab/Fold.git`; working branch `master` = `origin/master`; `v8-composed-value` merged (PRs #1–#6) |
| data | `phh-dataset/` → junction to `D:\phh-dataset` (PHH v3 `e2ec038`, `data/pluribus` only, 10,000 hands, MIT); `archive/data/` → junction to `D:\fold-archive\data`, holding the quarantined Arena archive |

## 4. STATE — one line, then verify

2026-09-03: **nothing runs; pointer `candidate-v9-0003b`; bankroll 0 on
Playground S17 (busted 2026-09-02 — the third bust); seat released; the
dataset switched to PHH/Pluribus, the retrain is queued for 2026-09-04.**
The board and the bust table: `.handoff/STATUS.md`. Never restate this from
memory; run PROCEDURES §1.

## 5. PROCEDURES INDEX (`.handoff/PROCEDURES.md`)

§1 verify live state · §2 tests and lint · §3 Phase-A dataset · §4 train
Phase A · §5 harvest Phase B · §6 train Phase B · §7 gauntlet and tripwire ·
§8 OLS gate · §9 promote / roll back (OWNER) · §10 start live play (OWNER) ·
§11 stop · §12 recover a seat · §13 move machines · §14 Linux deploy ·
§15 parallel agents · §16 Phase-A dataset from PHH.

## 6. RULES OF WORK — each one cost money or days

1. Measure the instrument before the result. Seed spread on one corpus was
   16.78 BB/100.
2. Validate every measurement script against an impossible-by-construction
   invariant before believing it.
3. A failed subagent is not a clean result. Check success counts.
4. A green suite is not evidence of safety on engine code. Sweep
   engine-touching changes adversarially; the L5 commit passed a green suite
   with two live-money holes.
5. Prefer estimated quantities to authored constants. A surviving constant is
   an explicit, ablatable parameter pinned by a test against its estimation
   artifact. A test that mirrors the implementation passes its bug.
6. Never build a path, vocabulary, or dial with no caller.
7. Offline metrics are gates, never selectors. The duel against
   `candidate-v7-0001c` is the selector; `v9-0003b` was promoted having lost
   it and busted within fourteen hours.
8. Frozen records (`artifacts/evaluations/*`) are never edited and keep their
   vocabulary. The package was `devfun_poker_playground/` before 2026-08-28.
9. Docs live next to the code they govern. One module per mechanism.
10. Owner questions about "the engine" mean the codebase, not the served
    artifact.
11. Commit and push only when the owner asks.

## 7. GIT HYGIENE

- Everything in-flight is tracked since `2696164` (2026-09-02); `git status`
  should be empty. Never `git add -A`: `.handoff/`, `archive/data/`,
  `phh-dataset/`, `runs/` and `artifacts/phase_a_v9/*pluribus*.jsonl.gz` are
  ignored bulk.
- Parallel agents: `.handoff/PROCEDURES.md` §15. Workers never write
  `.handoff/`, never run git, never touch the live path.
- Never `git add` a `*.weights.json` without a `.gitattributes … -text` rule:
  `core.autocrlf=true` rewrites the trailing LF and breaks the checksum.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## 8. ARCHIVES — DO NOT USE

`archive/` (repo) and `.handoff/archive/` hold superseded material: retired
rulesets and designs, session logs, previous versions of every document in
this manual, old code, old data. The Arena replay archive (`foreign play
data/`, obsolete 2026-09-03) is quarantined at
`archive/data/foreign-play-data-2026-09-03/`; `DATA.md` cites it for the
Phase-A oracle only.

1. Treat everything there as **false about the present**. Never read it to
   learn state, rules, plan, architecture, or procedure.
2. Open a file there only when a live document or a code docstring cites it
   by path, and use it only for the cited fact.
3. Never edit it. Never move anything from it into the live set. Never
   "restore" from it. If a live document lacks something, write it from the
   code.
4. `artifacts/evaluations/` is **not** an archive: dated measurement records
   that live documents cite for numbers. Read for the number, never for the
   plan. `gate-ablation-2026-08-26` (no depth suffix) is retracted; cite the
   `-60bb` record.
