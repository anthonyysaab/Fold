# Fold-ver-4 — start here

A Texas Hold'em agent for the dev.fun Arena. **The project was reset on
2026-08-16**: the previous architecture and its accumulated ruleset were retired.
Do not build on anything you find outside `.handoff/` without checking it against
the documents below — several subfolder READMEs are pre-reset and wrong.

## Read this order before doing anything

1. **`.handoff/CONTEXT.md`** — cold-start facts: paths, interpreters, live state.
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
- Baseline as of 2026-08-26: **714 passed, 6 skipped, 207 subtests passed**
  (~4.5 min). The older "344 tests, 1 CUDA skip" figure is pre-v8.
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
- The worktree is **clean** as of 2026-08-26 with three commits, and a remote
  now exists: `origin` = `github.com/anthonyysaab/Fold.git`, with both `master`
  and `v8-composed-value` pushed. The older "intentionally dirty, no remote"
  instruction is spent — the v8 session committed everything in `d0113f4`.

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
deployment. **Three unmeasured live-path gate changes are committed in `d0113f4`
and ship on the next supervisor restart whether or not anyone intends it** (see
`PENDING_EDITS.md`). They no longer appear in `git status`, so nothing surfaces
them before a restart.

## Documents known to be stale — do not trust without checking

- **root `README.md`** — pre-reset. Its hyper-aggression rate (5%), archive size,
  CSV count, and promotion-gate claim are all wrong or refuted.
- `tests/README.md` — says 138-input; the tests assert 142, and it omits ten test
  files.
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
