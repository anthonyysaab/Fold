> **ARCHIVED 2026-09-02 — SUPERSEDED. NOT STATE, NOT PLAN, NOT RULES.**
> Do not use this file to learn what the system is, what its state is, or what to do.
> The live manual is `CLAUDE.md` → `.handoff/CONTEXT.md`. Open this file only for a
> specific fact that a live document cites here by path. Never edit it, never restore it.

# 0FOLD : A multiway, multilayer poker agent

Intro: This project aims at assimilating notions of multiway Monte Carlo equity simulations and engine wiring.

It is built in multiple layers :

Layer 0 : Situational Read
Layer 1 : Interpreter
Layer 2 : Math Engine
Layer 3 : Range Model + personality functions and strategies


How To :


This project runs a Texas Hold'em policy against live dev.fun Arena tables. The live runtime needs only Python.

> **This file is partly pre-reset. Read `CLAUDE.md` first, then `.handoff/CONTEXT.md`.**
> The project was reset on 2026-08-16; `.handoff/` is authoritative for state, rules and plan.

Decisions come from a learned four-branch policy (`engine/learned_policy_v9.py`: a composed value head and a continuous sizing function) or, without an approved artifact, from deterministic equity rules — either way shaped a bounded amount by the situational risk temperature and kept inside hard safety gates. `artifacts/approved.json` points at `candidate-v9-0003b` (promoted 2026-09-01; `candidate-v7-0001c`, promoted 2026-08-14, is its rollback target). It was deployed to Playground S17 on 2026-09-01 and busted it on 2026-09-02; **nothing is running** (`.handoff/STATUS.md`, verified against the Arena API).

## What each root entry is or does

- `.arena-credentials` (ignored) is the Arena API key file; never commit or print it. `.arena-training.jsonl` (ignored) is the live decision journal, and `.arena-training.quarantine-2026-08-26.jsonl` holds the three fragments quarantined from it on 2026-08-26.
- `.claude/` stores machine-local assistant command permissions, plus a stale merged worktree under `.claude/worktrees/` that holds a second copy of `live_session.py` (check for a running session by process, not by filename).
- `.git/` is Git's internal version-history data and should not be edited manually.
- `.gitignore` lists generated and private paths Git should ignore.
- `.handoff/` stores local design decisions, current status, and next steps.
- `.ruff_cache/`, `__pycache__/` and `.pytest_cache/` are generated caches (ignored; absent until a lint or test run recreates them).
- `CLAUDE.md` is the agent-facing summary of the project's state, rules and read order; `README.md` is this project and root-folder map.
- `archive/` holds what the 2026-09-02 janitor pass moved out of the way: `pre-reset-2026-08-16/` (tracked, `git mv`'d pre-reset code and v2-era artifacts) and `data/` (ignored bulk: the v7 corpora, the deadhead set, the stale Eval bundle, the 2026-08-12 audit, the 2026-08-26 journal backup). Nothing in it is imported or read by the live path; `archive/README.md` maps every move.
- `artifacts/` stores trained policy data.
- `bluff.py` is the bluff advisor. It scores one situation's bluffing case (fold-equity price, draws, blockers, texture, discipline gates) and scales its mixing frequency with the learned bluff density, the lead gradient, and observed opponent wildness. The decision engine consults it in passive spots.
- `build/` and `bundle.zip` (both ignored, both absent until `deploy/devfun-arena/build_bundle.py` runs) are the regenerable Eval sandbox bundle and its staging folder; the 2026-08-14 build is in `archive/data/eval-bundle-2026-08-14/`.
- `deploy/` contains deployment adapters: the Eval sandbox bundle flow and the Linux systemd unit.
- `engine/` contains policy, game-state, and hand-strength code: the live decision brain plus the training modules. Both live play and training depend on it; never delete or move it.
- `foreign play data/` is the ignored public-replay corpus collected from Arena. The raw archive is 18.1 GB (16.8 GiB, measured 2026-08-31); the seven validated season CSVs are 0.288 GB (259,539 eligible rows). It also holds final-table, tournament, and Playground S14 collections.
- `lead_position.py` is the standalone -100..+100 player lead gauge: chip rank and chip share, accentuated by seat position. The engine computes it per decision and feeds it to the bluff advisor.
- `live_session.py` supervises continuous live play: back-to-back sessions, free-Playground discovery, money hard-stops, clean table release, and the deployment-scoped run archive.
- `play.cmd` / `play.sh` are the one-command launchers (Windows / Linux) for `live_session.py`.
- `risk_temperature.py` is the standalone 0-100 situational-risk gauge. The live engine imports its calculation, shifts its normal thresholds a bounded amount with the reading, and the runner logs each normal decision's reading and applied boldness.
- `run_agent.py` joins Arena, polls turns, asks the policy for actions, submits them, and leaves safely.
- `runs/` is the ignored, deployment-scoped archive of continuous-play sessions (purged when the deployed policy version changes).
- `tests/` contains local checks that do not contact Arena.
- `tools/` contains the offline dataset builders, harvesters, gauntlets, gates and estimators, plus a few manual Arena utilities (`tools/README.md`).

Every maintained code subfolder has its own `README.md` with an entry-by-entry map (`foreign play data/` has none; `.handoff/notes/DATASET_CONTRACT.md` describes it). `.handoff/CONTEXT.md` gives cold-start facts and the documented read order for the handoff folder. Generated folders such as `.git`, `build`, `.ruff_cache`, and `__pycache__` do not contain project notes.

## Live runtime

```text
live_session.py  (the supervisor: free-Playground discovery, money hard-stops, seat release, runs/ archive)
  -> run_agent.py  (one session: join, poll, act, leave)
     -> engine/learned_policy.py  (--learned: the approved artifact; format 4 -> engine/learned_policy_v9.py,
        with engine/aggression_sizing.py, engine/feature_extract_v9.py + schema4.py, engine/p3_belief_provider.py)
     -> engine/poker_policy.py  (the heuristic policies, and the fallback when nothing is approved)
        -> engine/decision_engine.py  (every proposal, learned or heuristic, passes through it: hard gates,
           sizing floors, the bluff hookup, the opponent tracker, the dark engine/rules/ dials)
           -> engine/game_state.py, engine/hand_strength.py, engine/opponent_model.py,
              engine/learning_contract.py, bluff.py, lead_position.py, risk_temperature.py
     -> engine/training_telemetry.py  (only with --telemetry)
```

The engine also keeps a session-scoped opponent model: each opponent's observed aggression frequency floors the range conditioning, blends the entry and stack-off thresholds back toward plain pot odds, and vetoes bluffs, so an all-in-every-hand attacker converges to a random range and gets called instead of folded through (simulated: -134 bb/100 before, breakeven within noise after). Because of that memory, live decisions are reproducible from the session history rather than from a single snapshot.

`run_agent.py --learned` plays the approved learned artifact from `artifacts/approved.json` instead of the heuristic, verifies its checksum and engine parameters at load, and refreshes it between hands when a new version is approved, keeping the session's opponent evidence.

Every deployed policy also carries a hardcoded anti-modeling dice roll (`HYPER_AGGRESSION_CHANCE`), **2% since 2026-08-15** (it was 5%; confirmed live at 20 fires in 1,073 decisions, with 5% ruled out at p = 7e-8). That fraction of decisions plays hyper-aggressively inside the hard gates, so pattern-learning opponents chase noise. The roll is salted and reproducible, marked in telemetry, and excluded from training labels.

`torch_policy.py` and `torch_network.py`, the optional checkpoint tools, were archived on 2026-09-02 to `archive/pre-reset-2026-08-16/engine/`; they were never on the live import path.

## Simulate, evaluate, improve

**The current line (v9, since 2026-08-30)** runs `python -m tools.build_phase_a_dataset_v9` -> `python -m engine.v9_trainer` (CUDA) -> `python -m tools.build_phase_b_corpus_v9` (`--postflop`, `--merge`) -> `python -m engine.v9_trainer_phase_b` -> `python -m tools.evaluate_v8 --candidate-v9 ...` -> `python -m tools.ols_baseline` (the gate `promote_candidate --ols-gate` enforces) -> `python -m tools.promote_candidate` (an owner act). `tools/README.md` documents every flag. The commands below are the **frozen v7 line**: `evaluate_policies` fields format 1/2 only and `self_play_cycle` trains through `engine.offline_trainer`.

```powershell
python -m tools.evaluate_policies --include-heuristic --candidate artifacts/candidates/<v>.manifest.json
python -m tools.evaluate_policies --candidate artifacts/candidates/<v>.manifest.json --ablate-sizing
python -m tools.self_play_cycle --foreign-csv <csv> --sparring artifacts/candidates/<v>.manifest.json --model-version <new-version> --device cuda --batch-size 1024
python -m tools.promote_candidate artifacts/candidates/<v>.manifest.json --reason "<evaluation summary>"
```

`engine/table_simulator.py` deals seeded Arena-shaped hands against scripted archetypes calibrated from the foreign audit (plus a permanent shover), scores BB/100, and captures self-play training examples with settled rewards. The improvement loop is: harvest (simulator or live telemetry) -> train a candidate -> gauntlet it against the incumbent -> promote only a winner -> the runner picks it up between hands. Promotion is always an explicit, reasoned step and nothing self-deploys, but note that **`tools/promote_candidate.py` does not enforce a gauntlet gate** — it validates the manifest and checksum, requires a free-form `--reason`, and since 2026-09-02 enforces the OLS baseline gate for format-4 candidates (`--ols-gate`, default `enforce`). The winner-only rule is a human discipline, not an automated one: `candidate-v9-0003b` was promoted having lost its duel, and busted S17.

## Run the agent

Credentials come from `ARENA_CREDENTIALS` when it is set, otherwise from this repository's gitignored `.arena-credentials`, so no environment setup is needed.

Continuous play is one command; it plays until you stop it. Windows:

```powershell
.\play.cmd
```

Linux or Raspberry Pi (Python 3.11+; the live runtime is stdlib-only, so nothing needs installing):

```bash
bash play.sh
```

For unattended Linux hosting, `deploy/live_session.service` is a ready systemd unit (`systemctl stop` releases the table cleanly via SIGTERM). Sessions default to six hours because every restart forfeits the lobby queue position and matchmaking, not decision speed, is the throughput bottleneck; tune with `--session-seconds`.

Every continuous run is archived for later study under the gitignored `runs/<launch-utc>/`: per-session console logs (`session-NNN.log`, the full runner output with temperature/bluff/range diagnostics), per-session manifests (`session-NNN.json` with chips before/after, timing, exit code, and the telemetry journal's byte offsets so that session's private records can be sliced out exactly), and a `run.json` launch summary rewritten at every boundary so even a hard kill leaves a readable record. Disable with `--no-archive`. The logs contain no hole cards; hole cards live only in the telemetry journal.

The archive always studies the current deployment: `runs/DEPLOYED` stamps the live policy version, and when a launch detects a different version (a promoted artifact or a heuristic bump), it purges the previous agent's run folders and starts fresh. Pass `--keep-old-runs` to adopt them across the change instead. The telemetry journal is never purged; it is the permanent training record and stamps every record with its own `policy_version`.

`live_session.py` behind it is a foreground console process, not a daemon or a service: it holds its window and ends with your login session. It restarts each finished session, re-discovers the free Playground competition every session (so a season rollover needs no new command), and releases the table on every exit route. Ctrl+C and Ctrl+Break arrive as signals, while window close, logoff, and Windows shutdown are caught through a console control handler, because CPython does not deliver those three as signals. It refuses to auto-join anything that costs money: the active `[Poker] Eval Open` bills per run and `[Poker] Tournament` needs a buy-in, so discovery accepts only the free Playground and an explicit `--competition` is required to override. It stops rather than retrying on a 402 entry fee, a 403 refusal, or a busted bankroll.

A single bounded session is still available:

```powershell
python run_agent.py <competition_id> --seconds 600 --aggressive
```

The runner stops instead of paying an entry fee or rebuying. A real Arena session is never used as a code smoke test.

## Record training data

Telemetry is opt-in:

```powershell
python run_agent.py <competition_id> --seconds 600 --aggressive --telemetry
```

This writes `.arena-training.jsonl`, which is ignored by Git. Pass a path after `--telemetry` to write elsewhere. The journal contains private hole cards, decisions, exact stack and legal-size data, and settled rewards. Keep it private. It never contains the Arena API key.

Only identity-verified, server-accepted, non-fallback decisions with a real choice between action families are marked suitable for training. During idle periods and shutdown, the runner joins those decisions to Arena settlement receipts by `tableId` and stores exact chip changes. New records also preserve the decision's starting purse: remaining stack plus chips already committed to the hand.

Telemetry collection does not train, explore, refresh weights, or change policy behavior yet.

## Train an offline candidate

**The current trainers are `engine/v9_trainer.py` (Phase A: supervised heads from the replay archive) and `engine/v9_trainer_phase_b.py` (Phase B: the composed value from the harvested corpus)**, run on the CUDA interpreter; their inputs come from the two v9 builders in `tools/`. Everything below describes the **frozen v6/v7 trainer**, kept servable for the rollback target and the frozen instruments.

After telemetry has settled hand results, the v7 trainer generates a local candidate artifact:

```powershell
python -m engine.offline_trainer .arena-training.jsonl
```

Foreign teacher decisions can be mixed through the validated CSV boundary as
behavior-only warm-up data. They cannot supply counterfactual value or sizing
targets, so use them with a simulator harvest:

```powershell
PowerShell -File archive/pre-reset-2026-08-16/artifacts/training-runs/candidate-v2-0015.ps1 -DryRun   # archived 2026-09-02; pre-reset recipe
```

The trainer z-scores inputs from counterfactual rows in the training split and stores the scales in the weights file. Public rows cannot redefine the value model's scale. Audit a corpus first with `python tools/audit_foreign_play_data.py <csv>`. The full row layout and training roles are recorded in `.handoff/notes/DATASET_CONTRACT.md`.

The trainer writes an immutable candidate manifest and checksummed JSON weights under `artifacts/candidates/`. For one selected decision per actor and hand, the simulator replays every legal family from the identical seeded state. The three existing action outputs predict each family's bounded signed-log purse advantage over the legal-family mean. Positive targets retain their true scale and receive 1.5x loss weight. Validation holds out whole hands and reports best-action accuracy, mean regret, and action-value error.

New value candidates use heuristic sizing by default; learned sizing is disabled until action selection passes. `evaluate_policies --ablate-sizing` separates action choice from sizing for older candidates. `--hybrid-min-advantage <value>` evaluates a conservative correction layer that keeps heuristic v5 unless an in-distribution learned alternative clears the requested value margin. The trainer does not approve, deploy, refresh, or roll back live weights.

Pass `--device cuda --batch-size 1024` from a CUDA-enabled PyTorch environment to train the neural network in vectorized GPU batches. Self-play hand generation remains CPU work.

## Measure risk separately

```powershell
python risk_temperature.py --hand-strength 40 --purse 973 --bet 20 --street preflop --players 6
```

Use `--json` for machine-readable output or `--self-test` for its built-in checks.

The live engine converts each reading into a bounded "boldness" via `TemperatureShaping`: cold readings (strong, cheap, late, short-handed) lower the aggression floor, shave call margins, and grow sizing; hot readings do the reverse. Hard safety gates never shift, and the reading is never sent to Arena. The shaping fields are future learned parameters.

## Ask the bluff advisor

```powershell
python bluff.py --hole Ah Kh --board 7h 2h 9c --street flop --pot 100 --stack 1000 --opponents 1 --hero-aggressions 1
```

Use `--json` for machine-readable output or `--self-test` for its built-in checks.

The advisor prices one bluff: estimated folds against the breakeven fold rate, semi-bluff equity from draws, blockers, board dryness, discipline gates, and a deterministic mixed-strategy roll. The roll is a salted hash of the situation (pass `--mix-key`, the table id in live play), so opponents cannot pattern it while every decision stays reproducible.

Bluffing frequency scales with `bluff_density`, tilted by the lead gradient through `lead_density_gain`:

```powershell
python lead_position.py --hero-stack 800 --opponents 1200 950 600 --position 0.8
python bluff.py --hole Ah Kh --board 7h 2h 9c --street flop --pot 100 --stack 1000 --opponents 1 --hero-aggressions 1 --lead -28.3
```

The gain's sign is the open question training will answer: positive bluffs the chip leader more, negative bluffs the trailer more, and the neutral default assumes neither.

The engine consults the advisor whenever a passive spot has a legal bet or raise: bluffs need a board the hero genuinely improves (never paired-board tiers), no live raising war, the advisor's pricing and mixed roll, and they size through the same legality and risk-cap clamps as any aggressive action. Executed bluffs submit as the aggress family, are logged with their kind, and are marked in telemetry so their outcomes can be scored separately. Every `BluffSettings` field is a learned parameter carried by candidate artifact manifests.

## Run local checks

```powershell
python -m pytest tests/ -q
python risk_temperature.py --self-test
python bluff.py --self-test
python lead_position.py --self-test
```
