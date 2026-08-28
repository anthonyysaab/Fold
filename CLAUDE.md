# Fold-ver-4 — start here

A Texas Hold'em agent for the dev.fun Arena. **The project was reset on
2026-08-16**: the previous architecture and its accumulated ruleset were retired.
Do not build on anything you find outside `.handoff/` without checking it against
the documents below — several subfolder READMEs are pre-reset and wrong.

## Read this order before doing anything

1. **`.handoff/CONTEXT.md`** — cold-start facts: paths, interpreters, live
   state. **Its git and test facts are stale** — it still says 344 tests, a
   dirty worktree with no remote, and three *uncommitted* gate changes. All
   three are wrong; this file has the current versions. Read it for paths and
   interpreters, not for state.
2. **`.handoff/notes/PROTOTYPE_POSTMORTEM.md`** — twenty candidates, one
   promotion, and why the previous architecture was abandoned. **Read this before
   proposing a new one.**
3. **`.handoff/DECISIONS.md`** — short. What is binding, including money safety.
4. **`.handoff/PLAYERS.md`** — the benchmark and opponent roster.
5. **`.handoff/STATUS.md`** — current state. 6. **`.handoff/NEXT.md`** — the plan.
7. **`.handoff/PENDING_EDITS.md`** — identified but unfixed work.

The retired ruleset and design docs are at `.handoff/archive/v6-v7-era/`, verbatim.
Frozen reports cite those rules by number. **Nothing in the live docs supersedes a
frozen report's own description of what it measured.**

## Money safety — never negotiable, and not part of the reset

- **Never pay an entry fee and never rebuy.** `is_free_playground()` refuses
  anything but the free Playground; Arena also hosts paid competitions. Never
  weaken that guard, never auto-join anything else.
- **One live seat at a time.** Check for a running `live_session.py` first.
- **Never use a live session as a smoke test.**
- **Never record or print the Arena API key** (`.arena-credentials`, gitignored).
- **Training never promotes, deploys, joins, or rebuys.** Promotion is a separate,
  explicit, human-authorised act — and note that `tools/promote_candidate.py`
  does **not** enforce a gauntlet gate, whatever the root README implies.
- **Release the seat on every stop path.** There is **no unclaim endpoint**;
  `python -m tools.leave <competitionId>` is the recovery, and a stop signal has
  been observed killing the supervisor without running its release path.

## Environment

- Work from `C:\Users\user\Fold-ver-4 (multiway)`. **The ` (multiway)` is part of
  the folder name** — quote the path everywhere.
- Tests / lint / live inference (stdlib-only, **no torch**, which is correct):
  `C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe`
- CUDA training (canonical):
  `C:\Users\user\poker-nn-training\.venv\Scripts\python.exe` — 3.11.9, torch
  2.13.0+cu130. A 3.14 install with torch 2.10.0+cu128 also works.
- Run repo tools as `python -m tools.<name>` so the root is importable.
- Baseline as of 2026-08-28: **766 passed, 12 skipped, 242 subtests passed**
  (~2.5-3.5 min) on the stdlib interpreter. **All twelve skips are
  torch-availability skips and all twelve pass on the CUDA interpreter**,
  which reports `778 passed` and zero skips (778 = 766 + 12). Seven say "CUDA
  PyTorch is unavailable" (six in `test_degenerate_group_filter.py`, one in
  `test_cuda_trainer.py`) and five say "PyTorch is unavailable" (all in
  `test_v8_trainer.py`). The older "344 tests, 1 CUDA skip" figure is pre-v8,
  and `tests/README.md` still prints it.
- **`ruff check .` is NOT clean**: 7 pre-existing errors in
  `tests/test_build_phase_b_corpus.py`, `tests/test_phase_a_dataset.py`,
  `tests/test_v8_trainer_phase_b.py`, `tools/build_phase_a_dataset.py`,
  `tools/measure_field_separation.py`, `tools/p3_gate.py` (unused imports
  and one unused local). Confirmed pre-existing by stashing all
  2026-08-26 work and re-running. Six are `--fix`-able.
- **`.handoff/` is gitignored** (`.gitignore:13`). The documents this file
  tells you to read first are **not version-controlled** — no history, no
  remote copy, and they do not survive a worktree clean. Treat edits to
  them as irreversible.
- **Seven commits** on `v8-composed-value` as of 2026-08-28, and a remote
  exists: `origin` = `github.com/anthonyysaab/Fold.git`. `v8-composed-value`
  is in sync with its remote and open as a PR against `master`. **Local
  `master` is three commits BEHIND `origin/master`** — it is not "pushed", it
  is stale; fetch before branching from it. The older "intentionally dirty, no
  remote" instruction is spent.
- There is a third local branch, `claude/sharp-bouman-d172f3`, with a checked
  out worktree under `.claude/worktrees/`. **It holds a second copy of
  `live_session.py`**, which matters for the "check for a running
  `live_session.py`" rule above — check by process, not by filename.

## Current state, in one line

**Nothing is running. The bankroll is BUSTED at 0.** A 2026-08-26 deployment
of the reverted gates ran 1.6h and went 1,000 -> 0 over 36 hands; it stopped
itself, did **not** rebuy (owner-gated), and released the seat. Verified
against the Arena, not a log: `chipState: busted`, `activeTableCount: 0`.
`artifacts/approved.json` was **rolled back** to `candidate-v7-0001c` and the
gate defaults with it, so the served configuration is the pre-2026-08-26 one
again. **Playing again requires an owner-authorised rebuy** — see the money
rules above.
`artifacts/approved.json` points at `candidate-v7-0001c` — a pointer, not a
deployment. The three 2026-08-15 gate changes were **measured, reverted,
deployed, and the deployment busted the bankroll**; both the pointer and the
gate defaults were then rolled back.

**So the served configuration today IS the 2026-08-15 one** —
`risk_cap_on_effective_stack` and `call_gates_on_effective_stack` True,
`reveal_expense_equity_slope` 0.12 (`decision_engine.py:252, :290, :291`), and
the manifest names none of them, so they are inherited from dataclass
defaults. That is the configuration `gate-decision-2026-08-26.md` priced at
**+16.49 BB/100 in favour of reverting**, and reverting it is what busted. The
tension is unresolved and it ships on the next restart either way. **Do not
treat this as settled.**

Four further repairs built on 2026-08-27 all ship **OFF** and were each
measured to be worth nothing in play, so nothing else moved. See `STATUS.md`
"Three null results, and one confirmation".

## Documents known to be stale — do not trust without checking

- **root `README.md`** — partly pre-reset, and it carries its own banner. Two
  of the four defects once listed here are **fixed**: it now states the
  hyper-aggression roll as 2% since 2026-08-15, and it now says outright that
  `promote_candidate.py` does not enforce a gauntlet gate. Still drifted: the
  archive size (says 16.638 GB, measured 16.839 GB).
- `tests/README.md` — **it already says 142-input**, correctly; the old
  "138-input" complaint here was itself wrong. The real defects: it prints the
  pre-v8 baseline "344 passed, 1 expected CUDA skip", and it documents 18 of
  the 49 `tests/test_*.py` files, so 31 are missing, not ten.
- `devfun_poker_playground/README.md` — describes the v6 trunk and three heads.
- `.handoff/notes/RED_TEAM.md` and `FEATURE_REGISTRY.md` — written pre-reset;
  their measurements stand, their framing and plans do not. Both carry banners.

## Working habits that were learned the hard way here

- **Measure the instrument before the result.** Seed spread on one corpus was
  16.78 BB/100 — larger than most effects worth chasing.
- **Validate any measurement script against an impossible-by-construction
  invariant before believing it.** Several have produced confident wrong answers.
- **A failed subagent is not a clean result.** Check success counts before
  believing a negative.
- **Prefer estimated quantities to guessed constants.** Six hand-authored
  constants were found wrong or inert in a single day, several running live.
