# Fold-ver-4 — start here

A Texas Hold'em agent for the dev.fun Arena. **The project was reset on
2026-08-16**: the previous architecture and its accumulated ruleset were retired.
Do not build on anything you find outside `.handoff/` without checking it against
the documents below — several subfolder READMEs are pre-reset and wrong.

## Read this order before doing anything

0. **If `.handoff/` is missing, STOP and read the warning at the bottom of
   this file** — it is gitignored, so a fresh clone does not have it, and
   most of this project's state lives there.
1. **`.handoff/CONTEXT.md`** — cold-start facts: paths, interpreters, live
   state, and (since 2026-08-31) a **MOVING MACHINES** block listing
   everything a `git clone` does not reproduce. Its environment facts are
   current again as of 2026-08-31.
1b. **`.handoff/notes/V9_RESTRUCTURE_PLAN.md`** — **the single most
   important document in the project right now.** The v9 restructure is
   the active workstream; that file's top box is the authoritative
   roadmap, and it also carries the pre-harvest decision pack, the
   measured harvest cost, and the record of which shapes were tried and
   reverted. Read it before touching `engine/` or `tools/`.
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
- Baseline as of **2026-08-31**: **924 passed, 18 skipped, 336 subtests
  passed** (~2-3 min) on the stdlib interpreter. **Every skip is a
  torch-availability skip and all of them pass on the CUDA interpreter.**
  The 18 are the 12 long-standing ones (six in
  `test_degenerate_group_filter.py`, one in `test_cuda_trainer.py`, five in
  `test_v8_trainer.py`) plus 6 added by the v9 trainers. Superseded
  figures you will still find quoted elsewhere: "766 / 12 / 242" was the
  2026-08-28 baseline before the v9 L3-L5 work, and the "344 tests, 1 CUDA
  skip" figure is pre-v8 — `tests/README.md` still prints that one.
- **On a machine without `foreign play data/` the pass count is LOWER**,
  because the archive-dependent tests skip rather than fail. That is not a
  regression; see the MOVING MACHINES block in `.handoff/CONTEXT.md`.
- **`ruff check .` is NOT clean**: 7 pre-existing errors in
  `tests/test_build_phase_b_corpus.py`, `tests/test_phase_a_dataset.py`,
  `tests/test_v8_trainer_phase_b.py`, `tools/build_phase_a_dataset.py`,
  `tools/measure_field_separation.py`, `tools/p3_gate.py` (unused imports
  and one unused local). Confirmed pre-existing by stashing all
  2026-08-26 work and re-running. Six are `--fix`-able.
- **`.handoff/` is gitignored** (`.gitignore:13`). The documents this file
  tells you to read first are **not version-controlled** — no history, no
  remote copy, and they do not survive a worktree clean or a fresh clone.
  Treat edits to them as irreversible, and see the warning at the bottom
  of this file before assuming a new checkout has them.
- **As of 2026-08-31: `v8-composed-value` is 23 commits AHEAD of its
  remote and NOT PUSHED.** That is the whole v9 L3/L4/L5 body of work.
  It is unpushed on purpose — the standing rule is push only when the
  owner asks — but it means a machine move loses it unless it is pushed
  first. `origin` = `github.com/anthonyysaab/Fold.git`; the branch is
  open as a PR against `master`. **Local `master` is 9 commits BEHIND
  `origin/master`** — stale, not pushed; fetch before branching from it.
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

## The active workstream: the v9 restructure (2026-08-31)

Everything above describes the SERVED state, which has not moved. The
work in progress is the **v9 restructure** — a four-branch action
contract (`fatal` / `passive` / `active` / `aggressive`) replacing the
old four, motivated by two measured defects: two of the old four slots
were aggression, so a dead value head argmaxed to aggression on a
2-in-4 coin flip (both bets of the bust hand were exactly that), and the
fixed 0.5/1.0 pot fractions deleted the engine's continuous sizer at
serve. **Authority: `.handoff/notes/V9_RESTRUCTURE_PLAN.md`.** Owner
waived the measurement gate for the restructure layers themselves;
it still applies to everything else, and absolutely to the gates.

- **DONE:** the contract, L1 (sizing function `g`, `schema4`, the
  extractor), L2 (composition, projection, format-4 loader, P3 belief
  provider wired, hyper roll off), L3 (both trainer forks), L4 (both
  dataset builders), L5 (engine coupling). The C1-C5 rule layer is
  built and wired **DARK** — every dial ships OFF and each needs its
  own measurement pass before it ever ships enabled.
- **LEFT:** L6 (telemetry + the v9 gauntlet floors), then the real 50k
  harvest → train → gauntlet → **owner** promotion.
- Proven end to end on real machinery, not just tests: archive →
  Phase-A dataset → CUDA Phase-A candidate → 6-leg harvest → schema-2
  corpus the trainer's own loader accepts → CUDA Phase-B candidate at
  3e-08 train/serve parity → served through `load_policy_v9`.
- Schema 4 is **FROZEN at 413 inputs** for that harvest. A feature
  change is free now and a full re-harvest afterwards — which is why
  five owner decisions gate the harvest (see the plan's PRE-HARVEST
  DECISION PACK).
- Harvest cost is **measured**, not extrapolated: 0.781 emitted
  decisions per hand, 1.577 s per decision on one core, so 50k
  decisions ≈ 64k hands ≈ 21.9 CPU-hours ≈ 3.6 h wall across six legs.
- **The hardest-won lesson of this work:** the L5 commit passed its own
  tests and a fully green suite and still shipped two live-money holes
  of the 2026-08-26 bust class (an ungated call answering the risk
  cap's refusal; a shove release with no equity term). An adversarial
  sweep caught both. **Sweep every engine-touching layer before
  believing it, and check the agents' success count** — one sweep
  returned an empty findings list purely because all five agents had
  died on a session limit. A green suite is not evidence of safety
  here.

## Documents known to be stale — do not trust without checking

- **root `README.md`** — partly pre-reset, and it carries its own banner. Two
  of the four defects once listed here are **fixed**: it now states the
  hyper-aggression roll as 2% since 2026-08-15, and it now says outright that
  `promote_candidate.py` does not enforce a gauntlet gate. Still drifted: the
  archive size (says 16.638 GB, measured 16.839 GB).
- `tests/README.md` — **it already says 142-input**, correctly; the old
  "138-input" complaint here was itself wrong. The real defects: it prints the
  pre-v8 baseline "344 passed, 1 expected CUDA skip", and it documents 18 of
  the **58** `tests/test_*.py` files (was 49 before the v9 work), so 40 are
  missing.
- `engine/README.md` — describes the v6 trunk and three heads.
- `.handoff/notes/RED_TEAM.md` and `FEATURE_REGISTRY.md` — written pre-reset;
  their measurements stand, their framing and plans do not. Both carry banners.
- `.handoff/STATUS.md` and `.handoff/NEXT.md` — their BODIES are the
  2026-08-28 post-bust record and queue, superseded by the v9 work. Both
  now carry a current block at the top; read that and treat the rest as
  history. `tools/README.md` is current (both v9 builders documented
  2026-08-31).

## Working habits that were learned the hard way here

- **Measure the instrument before the result.** Seed spread on one corpus was
  16.78 BB/100 — larger than most effects worth chasing.
- **Validate any measurement script against an impossible-by-construction
  invariant before believing it.** Several have produced confident wrong answers.
- **A failed subagent is not a clean result.** Check success counts before
  believing a negative.
- **Prefer estimated quantities to guessed constants.** Six hand-authored
  constants were found wrong or inert in a single day, several running live.
- **A green suite is not evidence of safety on engine code.** Two
  live-money holes shipped past one. Sweep adversarially.
- **Do not build a vocabulary, dial, or path with no caller.** Twice now
  the reverted work was unreachable *and* unsafe if reached.

## If `.handoff/` is missing (fresh clone, or a new machine)

`.gitignore:13` excludes `.handoff/`, so **a clone does not contain the
documents this file's read-order depends on** — including
`notes/V9_RESTRUCTURE_PLAN.md`, which is the authority on all current
work. They exist only in the working tree they were written in, with no
history and no remote copy.

Also excluded and not reproducible from the repo: `foreign play data/`
(~16.8 GB replay archive — the Phase-A builders, the P3 fit and several
measurement tools read it), `.arena-credentials` (recreate from your own
copy; **never** paste the key into any file here), `.arena-training.jsonl`
(the stored live journal), `runs/`, `artifacts/corpora/`,
`artifacts/deadhead/` and `.claude/`.

If you are holding a checkout without them: say so plainly rather than
reconstructing state from code, and ask for the `.handoff/` directory.
Do not infer the project's live state, money posture, or roadmap from
the source tree alone — the money rules in particular exist because of
events recorded only in those documents.
