# devfun-arena-live

Fresh project: an **aggressive, multiway live-play poker agent** for the
dev.fun Arena. Migrated from `arena-agent-original` on 2026-08-12. Self-contained
(the policy package + weights are vendored here).

The goal: take the proven-but-tight v3 policy and make it play a higher volume,
more aggressive game suited to 2–6-handed Playground tables (it currently
over-folds there — see "Why" below).

---

## What's here

```
devfun_poker_playground/   vendored torch-free policy package (rules, equity,
                           pure_model, snapshots, contract, _vendor/treys)
artifacts/tiny-policy-pure.json   exported net weights (125->64->3)
live_client.py             live-play client (join -> poll -> decide -> act)
multiway.py                MultiwayPolicy: the aggressive multiway variant (SCAFFOLD)
tools/
  submit.py                submit strategy.py/bundle.zip to a sandbox competition (Eval)
  poll.py                  poll a sandbox submission to a terminal status
  api.py                   authenticated GET helper
  peek.py                  dump live pending-actions state (diagnosis)
  leave.py                 leave a competition's queue/table cleanly
```

All tools read the API key from `$ARENA_CREDENTIALS` and never print it.

## Run it

```bash
# credentials were written by the arena client at registration; keep untracked
export ARENA_CREDENTIALS=C:/Users/user/devfun-poker-playground/.arena-credentials   # PowerShell: $env:ARENA_CREDENTIALS=...

# tight v3 policy (heads-up-tuned)
python live_client.py cmsg35zvs001hbagh1wdjc1me --seconds 2100

# aggressive multiway variant
python live_client.py cmsg35zvs001hbagh1wdjc1me --seconds 2100 --loose
```

Watch the agent live: **https://arena.dev.fun/agent/fold_ver_3**

---

## The agent (already live)

- **Fold-ver-3** — agentId `cmsnsh9er1ato12wxq5knep9d`, handle `fold_ver_3`,
  **CLAIMED / X-verified** to `@anthonyysaab`, status `Active`.
- Registered + claimed already; the sandbox_access gate is lifted. Only the API
  key (in `$ARENA_CREDENTIALS`) is needed.

## dev.fun Arena — how entry works (learned the hard way)

Two distinct models; the same agent can do both:

- **Live-play** (Playground / Tournament / 6-max): the agent connects and plays
  hands itself. Flow:
  `POST /texas/join {competitionId}` -> queue -> seated -> `GET
  /texas/pending-actions?competitionId=X` (**the competitionId query param is
  REQUIRED — omitting it 400s silently**) -> `tables[]` are the tables where it
  is your turn (each carries `allowedActions`) -> `POST /texas/action {tableId,
  action, amount, message, reasoning?}` -> `POST /texas/leave {competitionId}`.
  - `amount` is a **to-amount** (total street commitment). action enum:
    fold/check/call/bet/raise/all-in. cards are 2-char ("Ah"), streets are
    Capitalized ("Preflop"). **1-table concurrency** — you play one table at a
    time; a fresh join while seated returns `409 max_concurrent_tables`
    (live_client reconnects), and while queued returns `409 already_queued`.
  - **The live table IS the devfun_poker_playground snapshot contract** —
    `policy.decide(table)` consumes it directly (selfSeatNumber, seats[].holeCards,
    allowedActions.availableActions, callToAmount, raiseRange, ...).
  - HARD STOP on `402` (entry fee) or bust — rebuy is owner-gated. Playground
    S13 is **FREE** (1000 chips, no real money).
- **Sandbox submission** (Eval S1 / Heads-Up Ladder): upload a `strategy.py` or
  `bundle.zip`; the server hosts and runs it. Only these accept submissions;
  live comps return `400 "not configured for Texas benchmark submissions"`.

### Competition IDs (as of 2026-08-12)

| competition | id | type | notes |
|---|---|---|---|
| Playground S13 | `cmsg35zvs001hbagh1wdjc1me` | live-play | FREE, 2-6 handed, where live_client runs |
| Eval S1 | `seed_poker_eval_s1` | sandbox benchmark | PVE bb/100; Fold-ver-3 scored **+9.95** (partial; backend was degraded/timing out) |
| Tournament S12 | `cmslrboge8c2zmpfmv5adq4pd` | live-play | buy-in (possible 402) |
| 6-max final S2 | `cmsg3iqhm001o5d2mg1cbrl2t` | live-play | invite-only |
| Heads-Up Ladder | (not active) | sandbox PvP | would fit the tight policy perfectly; watch for it to reopen |

## Why the multiway variant

The v3 policy (range discipline + board-contribution discount) is **heads-up
tuned**: fold weak hands, commit only near the nuts. That won chipzen HU and
posted **+9.95 bb/100** on the Eval. But 6-handed it over-folds — it barely
defends blinds and its aggression floor (~0.72 equity to bet/raise) almost never
triggers multiway. Live result on Playground S13 over ~4 sessions: roughly
**break-even** (967 -> ~953 chips) but very **low volume** (~1 hand/table, folds
most). `multiway.py` loosens the open/raise floor and blind-defense margins while
keeping the stack-off gate. It's a **first pass** — tune it per the TODOs at the
bottom of `multiway.py` (test with `--loose`, compare chip delta + hands played).

## Where the rest lives (preserved, not migrated)

- `C:\Users\user\Documents\arena-agent-original` (Python, remote
  `arena.dev.fun-Agent-1-original`): `main`=vanilla `80c1da3`; `chipzen` branch
  (chipzen.ai Docker deploy); `arena-dev` branch (`deploy/devfun-arena/` = the
  sandbox strategy.py + bundle builder that got the +9.95 Eval run). 46 pytest
  tests. **The sandbox/Eval deployment tooling is on `arena-dev` there.**
- `C:\Users\user\devfun-poker-playground` (C11 native Arena client, remote
  `arena.dev.fun-Agent-1`): `main`=C baseline; `arena-dev` branch = the v2/v3 C
  port (strict `-Werror`, all C tests green). Also holds `.arena-credentials`.
- The chipzen.ai bot ("Fold-ver-2/3") is a separate deployment; see that repo.

## Next steps

1. Flesh out `multiway.py` (blind defense, c-betting, late-position steals,
   value-floor tuning) — TODO block in that file.
2. A/B it live on Playground S13: `--loose` vs tight, compare chip delta & VPIP.
3. Regenerate weights if needed: `arena-agent-original/tools/export_pure_weights.py`
   writes `artifacts/tiny-policy-pure.json` from the torch checkpoint.
