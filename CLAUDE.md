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
- Baseline: **344 tests pass, 1 expected CUDA skip.**
- The worktree is **intentionally dirty** — one migration commit, no remote.
  Preserve uncommitted work; do not "clean up" git state.

## Current state, in one line

**Nothing is running.** Live play is stopped and the seat released.
`artifacts/approved.json` points at `candidate-v7-0001c` — a pointer, not a
deployment. **Three unmeasured live-path gate changes sit uncommitted and ship on
the next supervisor restart whether or not anyone intends it** (see
`PENDING_EDITS.md`).

## Documents known to be stale — do not trust without checking

- **root `README.md`** — pre-reset. Its hyper-aggression rate (5%), archive size,
  CSV count, and promotion-gate claim are all wrong or refuted.
- `tests/README.md` — says 138-input; the tests assert 142, and it omits ten test
  files.
- `devfun_poker_playground/README.md` — describes the v6 trunk and three heads.
- `tools/README.md` — omits `collect_foreign_play_data.py`.
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
