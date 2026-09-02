> **ARCHIVED 2026-09-02 — SUPERSEDED. NOT STATE, NOT PLAN, NOT RULES.**
> Do not use this file to learn what the system is, what its state is, or what to do.
> The live manual is `CLAUDE.md` → `.handoff/CONTEXT.md`. Open this file only for a
> specific fact that a live document cites here by path. Never edit it, never restore it.

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
   state, the read order, and the **MOVING MACHINES** block listing
   everything a `git clone` does not reproduce. Rewritten 2026-09-02.
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

The retired ruleset and design docs are at `.handoff/archive/v6-v7-era/`, verbatim;
the pre-reset notes, the two session logs, the retros and the 2026-08-28
post-bust record are under `.handoff/archive/` as well (its `README.md` maps
them). Frozen reports cite those rules by number. **Nothing in the live docs supersedes a
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
  does **not** enforce a gauntlet gate (since 2026-09-02 it enforces the OLS
  baseline gate for format-4 candidates, `--ols-gate`, and nothing else).
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
- Baseline as of **2026-09-02**: **1007 passed, 21 skipped, 355 subtests
  passed** (~3 min) on the stdlib interpreter. **Every skip is a
  torch-availability skip and all of them pass on the CUDA interpreter.**
  Superseded figures you will still find quoted elsewhere: "958 / 18 /
  342" was the 2026-08-31 count, "956 / 18 / 340" was the
  count before the decision-pack sweep fixes, "947 / 18 / 340" before
  the pre-harvest decision pack landed, "936 / 18 / 340"
  before the rest of L6, "924 / 18 / 336" before the v9 L6
  floors, "766 / 12 / 242" was the
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
- **`v8-composed-value` is PUSHED and in sync with its remote; PRs #1–#6
  are MERGED to `master`** (the last, #6, on 2026-09-02). The working tree
  carries uncommitted 2026-09-01/02 work and every v9 artifact — see
  `.handoff/STATUS.md` "The tree". `origin` = `github.com/anthonyysaab/Fold.git`.
  Local `master` was fast-forwarded to `origin/master` (`b3abf5d`) on
  2026-09-02; fetch again before branching from it.
- There is a third local branch, `claude/sharp-bouman-d172f3`, with a checked
  out worktree under `.claude/worktrees/`. **It holds a second copy of
  `live_session.py`**, which matters for the "check for a running
  `live_session.py`" rule above — check by process, not by filename.
- **`archive/` exists since the 2026-09-02 janitor pass.**
  `archive/pre-reset-2026-08-16/` is tracked (`git mv`'d pre-reset code
  and the v2-era artifacts); `archive/data/` is gitignored bulk (the v7
  corpora, the deadhead set, the stale Eval bundle, the 2026-08-12 audit,
  the 2026-08-26 journal backup). Nothing in it is imported or read by
  the live path. `archive/README.md` maps every move and lists what was
  deliberately NOT archived — including the v7/v8 code, whose retirement
  is the owner-run post-promotion pass in the v9 plan, not a file move.

## Current state, in one line

**Nothing is running. The bankroll is BUSTED at 0 on Playground S17 — the
third bust — and the seat is released** (verified against the Arena API on
2026-09-02 ~19:00 UTC: `chipState: busted`, `activeTableCount: 0`).
`artifacts/approved.json` points at **`candidate-v9-0003b`**, promoted by the
owner on 2026-09-01 although it lost the seat-swapped duel to
`candidate-v7-0001c` by −41.14 BB/100 (every gauntleted v9 candidate lost
it). It was deployed to S17 the same night and went 1,000 -> 0 by
2026-09-02T13:28Z (last run: 977 -> 1,498 over 190 actions, then 1,498 -> 0
over 58). The supervisor did **not** rebuy. **No post-mortem of this bust
exists yet**, and whether to keep the pointer or `--rollback` to
`candidate-v7-0001c` is an open owner decision — `.handoff/NEXT.md`.
Playing again on S17 would be a rebuy (forbidden); a fresh season is a new
join.

**The served gate configuration is still the 2026-08-15 one** —
`risk_cap_on_effective_stack` and `call_gates_on_effective_stack` True,
`reveal_expense_equity_slope` 0.12 — inherited from dataclass defaults
because no manifest names them. `gate-decision-2026-08-26.md` priced
reverting them at **+16.49 BB/100**, and reverting them busted S16. The
tension is unresolved and ships on every restart. **Do not treat this as
settled**, and do not fix anything from the S17 bust silently: every
live-path change is a default-off dial with a named ablation arm.

The 2026-08-27 repairs still ship **OFF**, each measured to be worth nothing
in play (`.handoff/archive/2026-08-28-post-bust/STATUS-2026-08-28.md`).

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
  dataset builders), L5 (engine coupling), and **L6** (the v9
  trivial-baseline floors, fragment contract stamps, the additive
  `proposed_branch`/belief-degrade journal fields, and
  `head_degeneracy_audit --head`) — with one deferred item, the
  aggressive-minus-fatal separation key, which belongs to the future v9
  gauntlet recorder. The C1-C5 rule layer is
  built and wired **DARK** — every dial ships OFF and each needs its
  own measurement pass before it ever ships enabled.
- **DONE SINCE (2026-09-01/02):** the gauntlet wrapper fields format 4;
  every v9 candidate was gauntleted and every one lost the duel to
  `candidate-v7-0001c`; the owner promoted `0003b`; it busted S17; Phase 1
  of `artifacts/evaluations/v9-next-layer-plan-2026-09-02.md` ran
  (criterion 5 fired, the trainer default stays `raw`, Phase 2 = data is
  next). The v9 bust post-mortem is unwritten. The real pipeline
  has run (2026-08-31): Phase-A dataset 9,084 rows, Phase-A 0001a/b/c,
  the 50k harvest (106,961 branch rows / 40,012 decisions), and
  Phase-B 0002a/b/c with value_norm 0.878-0.887 — the value head is
  off the constant predictor, the corpus-size diagnosis is solved. A
  supplemental postflop harvest + loader-validated merge built the
  street-balanced corpus (56,043 decisions, postflop share 41%); its
  candidates 0003a/b/c score value_norm 0.939-0.946. The five
  pre-harvest owner decisions are settled and landed (2026-08-31)
  and swept (2 criticals found and fixed): `equity_vs_posterior` added
  (schema 414), block 7 size encoding fixed, lane costs on
  `contested_stack_chips`, normalization stamped, dormant features
  kept.
- Proven end to end on real machinery, not just tests: archive →
  Phase-A dataset → CUDA Phase-A candidate → 6-leg harvest → schema-2
  corpus the trainer's own loader accepts → CUDA Phase-B candidate at
  3e-08 train/serve parity → served through `load_policy_v9`.
- Schema 4 is **FROZEN at 414 inputs** (413 + `equity_vs_posterior`)
  for that harvest. A feature
  change is free now and a full re-harvest afterwards — which is why
  the five pre-harvest decisions (all now settled) gated the harvest
  (see the plan's PRE-HARVEST
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

- **root `README.md`** — the owner began rewriting its head on 2026-09-02
  (new title, a layer taxonomy, an empty "How To"); the body was swept the
  same day: pointer and live state current, archive size corrected to
  18.1 GB / 16.8 GiB, the live-runtime map on `engine/` paths, the OLS gate
  noted, `pytest` instead of `unittest`. Its "Simulate" and "Train"
  sections still describe the **frozen v7 line** and now say so at the top
  of each; the v9 pipeline is one paragraph there and fully in
  `tools/README.md`.
- **Swept and rewritten 2026-09-02**: `tests/README.md` (all 61 files,
  baseline 1007/21/355), `engine/README.md` (the current package map),
  `artifacts/README.md`, `tools/README.md` (re-verified flag by flag; the v9
  gauntlet fielding, the OLS gate, the harvester's `--postflop` / `--merge`
  and the five previously undocumented tools added), `.handoff/CONTEXT.md`,
  `STATUS.md`, `NEXT.md`, `PLAYERS.md`, `PENDING_EDITS.md` (audited item by
  item against the code) and `notes/README.md`. The pre-reset notes
  (`RED_TEAM`, `FEATURE_REGISTRY`, `SCRIPT_REVIEW`, `LEARNING_CONTRACT`,
  `FOREIGN_AUDIT`, `COUNTERFACTUAL_SIZING_DEFECT`, `CLOUD_HOSTING_RESEARCH`,
  the v7 reviews), the two session logs and the retros moved to
  `.handoff/archive/`; the 2026-08-28 STATUS/NEXT/PENDING_EDITS bodies are
  archived verbatim at `.handoff/archive/2026-08-28-post-bust/`.
- Still dated by design: `artifacts/candidates/README-v8-seeds.md`,
  `artifacts/phase_b/README.md`, `.handoff/notes/V8_DESIGN.md`,
  `SESSION_AUDIT_AND_LINUX_RUNBOOK.md` and `DATASET_CONTRACT.md` — each now
  carries a banner saying what in it is current. Frozen reports under
  `artifacts/evaluations/` are never edited.

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
(the replay archive — **18.1 GB / 16.8 GiB, measured 2026-08-31**; the
Phase-A builders, the P3 fit and several measurement tools read it),
`.arena-credentials` (recreate from your own copy; **never** paste the
key into any file here), `.arena-training.jsonl` (the stored live
journal, 12 MB), `archive/data/` (~777 MB: the v7 corpora, the
deadhead set, the stale Eval bundle, the 2026-08-12 audit and the
2026-08-26 journal backup — moved there 2026-09-02, see
`archive/README.md`), `runs/` and `.claude/`. Roughly
**18.9 GB in total**, of which `.handoff/` is 925 KB and is the only
part with no other source — copy it first.

If you are holding a checkout without them: say so plainly rather than
reconstructing state from code, and ask for the `.handoff/` directory.
Do not infer the project's live state, money posture, or roadmap from
the source tree alone — the money rules in particular exist because of
events recorded only in those documents.
