# 0FOLD : A multiway, multilayer poker agent

Intro: This project aims at assimilating notions of multiway Monte Carlo
equity simulations and engine wiring.

It is built in multiple layers :

| layer | owner's name | what implements it |
|---|---|---|
| Layer 0 | Situational Read | `engine/game_state.py` (snapshot → geometry), `risk_temperature.py` (the temperature read), `lead_position.py` (the lead gauge), `engine/opponent_model.py` (session aggression tracking) |
| Layer 1 | Interpreter | `engine/feature_extract_v9.py` + `engine/schema4.py` (the 414-input vector), `engine/p3_belief_provider.py` (the opponent-range posterior) |
| Layer 2 | Math Engine | `engine/hand_strength.py` + `engine/strength_metric.py` (equities), `engine/learned_policy_v9.py` (the composed value over four branches), `engine/aggression_sizing.py` (the continuous sizing function g) |
| Layer 3 | Range Model + personality functions and strategies | `engine/strength_aware_opponent.py` (the fitted P3 range model), `bluff.py` (the bluff advisor), `engine/rules/` (the C1–C5 candidate rules, dark), `engine/decision_engine.py` (the hard gates that every proposal passes) |

Revision 2026-09-02. Previous README: `archive/docs-superseded-2026-09-02/README.md`.

## What this is

A Texas Hold'em agent for the dev.fun Arena (agent **0Fold**). Live play is a
Python-stdlib process (`live_session.py` → `run_agent.py`) serving the
learned artifact named by `artifacts/approved.json`, inside hard money gates.
Training runs on CUDA from public complete-information replays and from a
simulator harvest against a fitted strength-aware opponent.

## State (2026-09-02, verified against the Arena)

Nothing runs. Pointer `candidate-v9-0003b` (promoted 2026-09-01). Bankroll 0
on Playground S17 — busted 2026-09-02, the third bust. Seat released.
Details, decisions pending, and the money rules: `CLAUDE.md`, then
`.handoff/STATUS.md` and `.handoff/NEXT.md`.

## The manual

`CLAUDE.md` is the quick-reference card. `.handoff/` is the manual:
`CONTEXT` · `STATUS` · `DECISIONS` · `notes/V9_ARCHITECTURE` · `NEXT` ·
`PENDING_EDITS` · `PROCEDURES` · `DATA` · `PLAYERS`. Each code folder has a
one-page map (`README.md`). `.handoff/` is gitignored — copy it by hand on a
machine move.

## Folder map

| path | contents |
|---|---|
| `live_session.py`, `run_agent.py` | the live supervisor and one session (`play.cmd` / `play.sh` launch it) |
| `bluff.py`, `lead_position.py`, `risk_temperature.py` | root modules the engine imports on every decision (each has `--self-test`) |
| `engine/` | the decision brain, serve runtimes, trainers, simulator, rule layer (`engine/README.md`) |
| `tools/` | dataset builders, harvesters, gauntlets, gates, estimators, Arena utilities (`tools/README.md`) |
| `tests/` | 61 test files, pytest (`tests/README.md`) |
| `artifacts/` | candidates, the approved pointer, datasets, corpora, P3 fit, frozen evaluation records (`artifacts/README.md`) |
| `deploy/` | Linux systemd deployment and the Eval-sandbox bundle (`deploy/README.md`) |
| `runs/` (ignored) | session archives of the current deployment; purged when the deployed version changes |
| `foreign play data/` (ignored, 18.1 GB) | the public replay archive (`.handoff/DATA.md`) |
| `archive/`, `.handoff/archive/` | superseded material — **not to be used** (`CLAUDE.md` §8) |
| `.arena-credentials`, `.arena-training.jsonl` (ignored) | the API key (never print it) and the live decision journal |

## How To

Full procedures with preconditions: `.handoff/PROCEDURES.md`. The six most
used, from the repo root with the stdlib interpreter:

```powershell
python -m pytest tests/ -q                                             # 1,145 / 29 / 370
python -m tools.api /api/arena/agent/me /api/arena/competition/list-active   # live state, read-only
python -m tools.evaluate_v8 --candidate-v9 artifacts/candidates/<v>.manifest.json --workers 6 --output artifacts/evaluations/<v>-gauntlet.json --workdir artifacts/evaluations/<v>-gauntlet.json.stages
python -m tools.ols_baseline --corpus artifacts/phase_b_v9/candidate-v9-phase-b-merged.phase-b.jsonl.gz --phase-a-dataset artifacts/phase_a_v9/phase-a-dataset-v9-pluribus-<date>.jsonl.gz --candidate artifacts/candidates/<v>.manifest.json
python -m tools.promote_candidate artifacts/candidates/<v>.manifest.json --reason "<evaluation summary>"   # OWNER
.\play.cmd                                                             # OWNER; Ctrl+C once to stop
```

Never pass `--competition`. Never rebuy. Never print the key.
