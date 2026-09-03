# OOD instrument — three designs and a judged recommendation (2026-09-03)

A **proposal, not a decision** (an owner decision on record, or a later dated
record, is what decides). Produced for Phase 0 of
`artifacts/evaluations/v9-next-layer-plan-2026-09-02.md` §1: the gauntlet
battery is in-distribution by construction (it shares scripted opponents with
the harvest panel) and the v7 duel is a single aggregate number, so river
behaviour is invisible.

Design-only lane of the 2026-09-03 parallel run. **No repo code was written**
and no module was created. Read-only probes were run from the scratchpad
against the existing engine; where a number below is measured, the probe, its
seeds and its literal output are named. Every load-bearing definition,
predicate, number and citation is in exactly one of three states, and each is
marked:

- **[VERIFIED]** — re-derived from the code or a frozen record in this
  session, with the file:line or the JSON key given;
- **[MEASURED]** — produced by a scratchpad probe described in §2.1, with the
  literal counts pasted;
- **[ASSUMPTION]** — not established; the pilot replaces it. An assumption
  is never used to justify a design choice without saying that it is one.

**Repair note — this record has been through two repair passes.** The first
draft (same date, uncommitted) was refuted by three reviewers: its named
slice was empty by construction, its completion predicate did not detect
completions, its recommended variance reduction contradicted the engine
contract it cited, and several numbers and citations were wrong. A first
repair pass rewrote it. **This second pass re-ran every probe from scratch
rather than trusting either the reviewers or the first repair**, and found
that the first repair had itself carried four wrong numbers (§10.C). Where a
count below differs from a reviewer's or from the first repair's, the number
here is the one this session measured, with its seeds. It is a single dated
record and is frozen at first commit (CLAUDE.md rule of work 8) — not before.

**Units — one scale only.** Every slice quantity in this record is in
**bb/decision** (big blinds per slice decision; big blind = 100 chips at the
instrument's 60 bb stack). The duel/battery scale is **BB/100** (big blinds
per 100 hands). The two are different scales and are never compared across
(`.handoff/DECISIONS.md` §3.2 — "another scale; never compare across"
[VERIFIED]; the first draft cited §3.4, which is the label/feature-change
rule). The first draft also named a reported quantity
`slice_bb_per_100_decisions` while stating every MDE in bb/decision, a silent
100x. **That name is withdrawn.** If a per-100-decision figure is ever wanted,
it is `100 x` the bb/decision figure and is labelled at the point of use.

---

## 0. Reference numbers the acceptance must beat (pasted, not re-derived)

The duel is the only OOD separation the project has, so it is the clarity bar.
From `artifacts/evaluations/candidate-v9-0003b-gauntlet.json`, key by key
[VERIFIED — I reloaded the JSON in this session and re-derived every ratio]:

| quantity | value | JSON key |
|---|---|---|
| duel paired mean (`candidate-v9-0003b` vs `candidate-v7-0001c`) | **-41.14 BB/100** | `duel.report.paired.mean` |
| paired sd / se / t | **15.6 / 3.9 / -10.55** | `duel.report.paired.{sd,se,t}` |
| empirical MDE at 16 seeds | **7.8 BB/100** | `duel.empirical_mde_bb_per_100`; `2*15.6/4 = 7.80` checks |
| resolvable seed spread | **16.78 BB/100** | `duel.resolvable_spread_bb_per_100` |
| construction | 16 seeds x 2 orientations x 2,000 hands = 64,000 hands | `duel.seeds` 16, `duel.report.hands_per_seed` 2000 |
| duel stage cost | **5,162.1 s at 6 workers** = 30,972.6 worker-s = **8.6035 CPU-h** | `config.stage_elapsed_seconds.duel`, `config.workers` |
| `\|Δ\|` / seed spread | 41.14 / 16.78 = **2.4517** | derived |
| `\|Δ\|` / own MDE | 41.14 / 7.8 = **5.2744** | derived |

**The verdict rule the evaluator actually enforces** [VERIFIED — quoted
verbatim from `duel.verdict_rule` in that JSON]: *"a margin below the
resolvable spread (max of the known 16.78 BB/100 seed spread and 2·sd/√n) is
UNRESOLVED, never a win/loss"*; `KNOWN_SEED_SPREAD_BB_PER_100 = 16.78` at
`tools/evaluate_v8.py:93`. The first draft's acceptance clause used
`max(MDE_own, 0)`, which drops the seed-spread half of the rule and makes the
`0` vacuous. Repaired in §7.

Measured cost rates used for every design's CPU arithmetic:

| rate | source | state |
|---|---|---|
| **0.371665 s/hand/core**, learned hero vs scripted opponent | `3221.1 s x 6 workers / 52,000 hero-hands`. 52,000 = 8 seeds x 6,500, and 6,500 = 1,000+1,000+1,000+1,500+1,000 (`_BATTERIES`, `tools/evaluate_policies.py:49-59`) + 1,000 five-max | [VERIFIED] |
| **0.483947 s/hand/core**, two learned seats | `5162.1 s x 6 workers / 64,000 hands` | [VERIFIED] |
| **0.024133 s/hand/core**, code policy vs code/scripted | `909 s x 6 workers / 226,000 hands` (`artifacts/evaluations/noise-floor-2026-08-15.json`, `config.elapsed_seconds` 909.0, **`config.workers` 6**). 226,000 = 24 duel seeds x 2 x 2,000 + 20 battery seeds x 6,500 = 96,000 + 130,000 [VERIFIED, both config keys re-read] | [VERIFIED] |
| **~0.089 s per learned `decide`** | `0.483947 / (2 seats x 2.72 decisions/hand)`. 2.72 is the mean of the two decisions-per-hand rates measured in §2.5 **for a scripted hero** — a proxy, and this figure is load-bearing nowhere below | [MEASURED input, proxy] |

> **Correction carried from the first draft.** It used *0.064 s/hand/core* for
> the code-policy rate, from "909 s x **16** workers". 16 is
> `config.cpu_count`, not `config.workers` (which is 6). `909*16/226000 =
> 0.064354`; `909*6/226000 = 0.024133`. The draft's rate was **2.667x** the
> record. Every code-policy cost line below uses 0.024133.

> **Correction carried from the first draft.** It used *0.25 s* per learned
> `decide` while stating a derivation that gives 0.097 — a 2.6x unexplained
> inflation with no basis. The repaired Design C needs no learned-`decide`
> probe at all (§6.2), so the figure is now decorative rather than
> load-bearing, and it is labelled a scripted-hero proxy.

**Worker count.** `.handoff/PROCEDURES.md` §7 runs the gauntlet at
`--workers 6`, and the frozen record's `config.workers` is 6 [VERIFIED].
§15 records 16 CPUs on this machine. Every cost below is therefore stated in
**CPU-hours** (portable, measured) with wall time at the **documented 6
workers**; a 12-worker wall time is given in parentheses and is an
[ASSUMPTION] about how the run would be launched, not a project setting. The
first draft asserted 12 workers as if it were established.

The **noise floor** contributes three things to every design: the MDE
definition (`MDE(n) = 2*sd/sqrt(n)`, the smallest true effect whose expected
`|t|` reaches 2 at n seeds — `config.mde_definition` [VERIFIED, verbatim]),
its self-duel null pattern (`duel_channel.mirror_exact: true`, paired diffs
identically zero) as the wiring check to copy, and its own invariant block as
the report shape. Its `invariants.battery_plan_structure` also states the
project's seed convention verbatim: *"match seeds 100+/200+ and opponent seeds
13+/23+"* [VERIFIED].

---

## 1. The binding rule this instrument has to answer first

`.handoff/DECISIONS.md` §3.6 [VERIFIED, quoted]: *"Every battery except
`vs-p3` is card-blind: a trivial floor beats every real policy there.
Batteries are fit diagnostics, not generalisation."*

Every slice-generating opponent proposed in §3 is card-blind
(`ScriptedAgent.reads_cards = False`, `engine/table_simulator.py:1341`
[VERIFIED]). The first draft never engaged this rule, and it is the rule that
most directly threatens the whole instrument: **an out-of-distribution claim
built from card-blind scripted archetypes is, by this project's own binding
rule, still a fit diagnostic.**

The honest position, and it must be stated in the instrument's own report:

- What the slice *can* establish is **held-out-opponent** generalisation:
  these archetypes are absent from the harvest panel (§3), so a v7-vs-v9
  difference on them is not a difference the harvest taught either policy.
- What the slice **cannot** establish is generalisation against an opponent
  that reads its own cards. A policy that overfolds a completing river will
  not be punished by an archetype that bets the river without looking at its
  hand, because that archetype's bet carries no information.
- DECISIONS §3.6's failure mode is specific and testable: *a trivial floor
  beats every real policy there.* The instrument therefore **must** carry the
  floor arms (Design B, §5) not as an optional invariant but as the DECISIONS
  §3.6 check. If a constant-intent floor produces the better slice number, the
  channel is a fit diagnostic and the v7-vs-v9 delta on it means nothing.
- `TexturedAgent` (`engine/table_simulator.py:1466`) is **not** card-reading
  — it prices the wager and the public board, and its docstring says so,
  keeping `reads_cards = False`.
- **Correction (made the same day, before first commit).** An earlier draft of this
  section said "no card-reading archetype exists in this codebase". That is
  **false**: `engine/strength_aware_opponent.py:508`
  `StrengthAwareAgent(ScriptedAgent)` — "an archetype whose folding responds
  to its hand" — sets `reads_cards = True` at :536, and
  `table_simulator.py:262` honours that by declining to resample its holding.
  It is the P3 archetype, which is why `.handoff/DECISIONS.md` §3.6 reads
  "every battery **except `vs-p3`** is card-blind". The correct statement is
  narrower and does not change this instrument's design: the archetypes **this
  slice arena uses** are card-blind, and P3 is excluded from the arena for the
  reason §3 gives (0 of 7,320 fitted rows exceed 1.1x pot, so any exploit
  found against it in this slice would be a P3 artefact) — not because no
  card-reading opponent exists.
- There is one compensation, and §6.2 depends on it: because the archetypes
  are card-blind, their holdings at a river decision are **uniform over the
  unseen cards** — their fold/bet choices cannot have selected them. That is
  what makes the villain-range average of §6.2 an exact conditional
  expectation rather than a modelling assumption. The same property that caps
  the OOD claim is what makes the variance reduction legitimate.

This is a genuine limit on what any of the three designs can claim, and it is
not repaired by any of them.

---

## 2. The named decision slice — repaired, and measured

Plan §1 names the slice as: *"Stack-committing calls facing greater-than-pot
bets on completing boards"* [VERIFIED, plan lines 57-58]. Rendering that
sentence into this engine's snapshot dialect is the whole difficulty, and the
first draft got all three clauses wrong. Each is repaired below with the probe
that establishes it.

### 2.1 The probes

Read-only scratchpad probes, project interpreter
(`C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe`), no repo
file created, no engine edit, no live path, no Arena call, no timed benchmark.
All matches are heads-up carry-over sessions at the instrument's 60 bb stack
(`run_sessions(..., starting_stack=6000, small_blind=50, big_blind=100)`).

- **P1 (predicate clauses)** — 3 opponents x 400 hands, match seeds 100 / 101
  / 102; hero `ScriptedAgent("hero", 0.226, 0.5, 0.0, 91)` wrapped in a
  recorder; opponents the battery's permanent shover
  (`ScriptedAgent("shover", 0.0, 0.0, 1.0, 13)`), a 0.9-aggression bettor, and
  the arena-median archetype. Logs the raw predicate inputs at every **priced**
  hero decision (`callChips > 0`). **Result: 1,377 priced hero decisions
  (491 / 570 / 316).**
- **P1b (invariant sweep)** — the same recorder over 5 opponents (the three
  above plus the two archetypes of §3) x 2 heroes (the scripted hero and the
  `always_check_call` floor) x 3 match seeds (100 / 313 / 727) x 1,000 hands.
  **Result: 42,145 priced hero decisions, of which 5,681 priced river
  decisions.**
- **P2 (archetypes and floors)** — the two archetypes this record proposes
  (§3) x three heroes (scripted median, `always_check_call` floor,
  `always_fold` floor) x 2,000 hands, match seed 200. **Result in §2.5.**
- **PB (board texture)** — `engine.table_simulator.board_coordination` called
  directly on 11 crafted turn/river pairs and on 20,000 random 5-card boards
  (seed 7).
- **PE (estimand)** — exact showdown equity at every slice decision: the
  hero's actual holding against **all C(45,2) = 990** villain holdings from the
  unseen cards, evaluated with the engine's own `_shared_evaluator`. Board is
  complete at a river slice decision, so this is an enumeration, not a
  simulation.
- **PL (served hero)** — the same recorder wrapping the **served learned
  policy**, `load_policy_v9("artifacts/candidates/candidate-v9-0003b.approved.manifest.json")`,
  300 hands per archetype, match seed 200. Small, and labelled as such; it
  exists to check that the scripted-hero proxy is the right order of magnitude.

### 2.2 Clause 1 — "greater-than-pot bet"

**First draft: `callChips > potChips`. This is unsatisfiable in this engine.**

`potChips` is `sum(seat.hand_commit for seat in seats)`
(`engine/table_simulator.py:573`) — it **already contains the villain's bet
and the hero's own commitment**. `callChips` is
`min(max(0, current_bet - actor.street_commit), actor.stack)` (`:554`)
[both VERIFIED].

Write `h`, `v` for the two seats' prior-street commitments, `s_h`, `s_v` for
their commitments on this street, `c` for `current_bet`. Where `callChips` is
not stack-capped,

`potChips - callChips = h + v + 2*s_h + (s_v - c)`.

In the ordinary case `c = s_v` and the gap is `h + v + 2*s_h >= 0`, strictly
positive once a blind is posted. In the one case where `c > s_v` — a villain
all-in for **less** than the posted blind — the gap collapses to `s_v`, which
is still at least one chip. The stack cap only makes `callChips` smaller.
**So the predicate can never fire, and the minimum gap is one chip.**

[MEASURED, P1] `callChips > potChips` fired **0 of 1,377** priced decisions.
[MEASURED, P1b] **0 of 42,145**. Maximum observed `callChips/potChips` over the sweep is
**0.9804** (0.9672 over P1’s narrower panel), on the literal row `street preflop, pot 51, call 50, seats
[(1, Active, 11949, 50), (2, Active, 0, 1)]` — the villain all-in for one chip
as the blind, which is exactly the bound above. Minimum observed
`potChips - callChips` = **1 chip**; never zero, never negative.

**Repair: `2 * callChips > potChips`.** A wager is "greater than pot" when it
exceeds the pot *as it stood before the wager*; that prior pot is
`potChips - callChips`, so the condition is `callChips > potChips - callChips`.
Equivalently: the hero is laid worse than 2:1 and needs more than 1/3 equity —
which is the sense in which the plan's sentence is a *pricing* claim.

[MEASURED, P1] `2 * callChips > potChips` fired **478 of 1,377** priced
decisions (231/491 vs the shover, 175/570 vs the 0.9-aggression bettor,
72/316 vs the arena-median).

### 2.3 Clause 2 — "stack-committing"

**First draft: `callChips >= 0.5 * effective_stack_chips(table)`. This is
worse than vacuous — it is inverted.**

`effective_stack_chips` (`engine/game_state.py:111`) returns
`min(hero_stack, max(opponent_stacks))`, counting chips *behind*. An all-in
opponent keeps `status: "Active"` with `stackChips: 0`, so the function
returns **0** the moment the last live opponent shoves — and the test
`callChips >= 0.5 * 0` becomes `callChips >= 0`, which every priced decision
satisfies. The clause admits the *smallest* calls precisely where it was
meant to admit the largest.

[MEASURED, P1] `effective_stack_chips == 0` on **293 of 1,377** priced
decisions (288 / 3 / 2 by opponent). The smallest call among those rows is
**50 chips = 0.50 bb**; literal row: `street preflop, call 50, pot 150, eff 0,
contested 100, seats [(1, Active, 11850, 50), (2, Active, 0, 100)]` — a
half-big-blind call that the clause as written classifies as
"stack-committing".

[MEASURED, P2] Against `completion-shover`, this record's own densest slice
generator, `effective_stack_chips == 0` on **377 of 377** priced river
decisions on a completing board. The clause is exactly 100% vacuous there.

This is a known, logged defect, not a discovery:
`.handoff/PENDING_EDITS.md` row **E**, status PARTLY — *"effective-stack
denominator collapses to 1 when every opponent is all-in ... Fix A =
`contested_stack_chips` behind `gate_stack_counts_committed_chips` (OFF)"*
[VERIFIED]. The first draft cited neither the defect nor the fix.

**Repair: `callChips >= 0.5 * contested_stack_chips(table)`**
(`engine/game_state.py:124`), which counts each active opponent's
`currentBetChips` alongside their remaining stack — "the pile that opponent
brought to this round".

[MEASURED, P1] `callChips >= 0.5 * effective_stack_chips` fires **609 of
1,377** (295 / 242 / 72); `callChips >= 0.5 * contested_stack_chips` fires
**512 of 1,377** (275 / 197 / 40). **97 rows are admitted by the effective
form and rejected by the contested form**; none the other way. It is a real
filter, not a rubber stamp.

> **The repaired clause is relative, and the owner should see that.** It asks
> whether the price is at least half of what the opponent has at stake, so a
> small absolute call against a short opponent qualifies (the row above is one:
> 0.50 bb, contested 100 chips, so `50 >= 0.5*100` holds). That is what
> "effective stack" means, and adding an absolute floor would be a new dial
> with no estimate behind it (CLAUDE.md rule of work 5). The instrument
> therefore **reports the pot and price distribution of the slice** (§2.6) so
> the owner can see how much of it is small-pot, rather than pinning a
> threshold in advance.

> **Caveat, and it is new — no reviewer found it, and the first repair pass
> got its counts wrong.** The `contested_stack_chips` docstring asserts an
> invariant: *"whenever `callChips` is positive the result is at least
> `callChips`."* [MEASURED, P1b] **That invariant fails: 60 violations in
> 42,145 priced decisions (0.142%).** Literal row: `street preflop, call 50,
> contested 25, pot 75, currentBet 100, seats [(1, Active, 11925, 50),
> (2, Active, 0, 25)]` — a villain all-in for *less than the posted big blind*,
> so the snapshot's `current_bet` (the nominal blind, 100) overstates what the
> villain can actually cover (25). **All 60 violations were preflop; 0 of the
> 5,681 priced river decisions in the same sweep violated it.** Heads-up on the
> river the villain's `currentBetChips` *is* `current_bet`, so the slice is
> unaffected in the heads-up construction proposed here — but the instrument
> must **assert** this per row rather than assume it (invariant I7, §5), and
> the docstring claim should be narrowed. This is outside this lane's write
> set; it is carried to `.handoff/PENDING_EDITS.md` as a proposed row.

### 2.4 Clause 3 — "completing board"

**First draft: `board_coordination(river) - board_coordination(turn) >= 0.30`,
OR 4+ cards of one suit; with the 0.30 dial claimed to be "pinned by fixture
tests (a four-flush completing on the river must classify completing; a blank
river must not)".**

**No value of the dial satisfies that fixture pair.** [MEASURED, PB, on
`engine.table_simulator.board_coordination` directly]:

| turn -> river | turn | river | delta | draw completing? |
|---|---|---|---|---|
| flush completes, `Ah 7h 2h Kd` -> `9h` | 0.7100 | 0.8000 | **+0.0900** | yes |
| flush completes, `Qs 8s 3s 2d` -> `6s` | 0.7100 | 0.8000 | **+0.0900** | yes |
| flush completes on a four-flush turn, `Ah 7h 2h Kh` -> `9h` | 0.8000 | 0.8000 | **+0.0000** | yes |
| straight completes (open-ended), `5h 6d 7c Ks` -> `8h` | 0.3500 | 0.5075 | +0.1575 | yes |
| straight completes (gutshot), `9s 8d 7c 2h` -> `6d` | 0.3500 | 0.5075 | +0.1575 | yes |
| straight completes, `Js Td 9c 2h` -> `8d` | 0.3500 | 0.5075 | +0.1575 | yes |
| **true blank**, `Ah 7d 2c Ks` -> `4s` | 0.3500 | 0.5075 | **+0.1575** | no |
| **true blank**, `As 9d 4c 2h` -> `7s` | 0.2625 | 0.4200 | **+0.1575** | no |
| third suit arrives, `Ah 7d 2c Ks` -> `4d` | 0.3500 | 0.5075 | +0.1575 | no |
| river **pairs** a dry board, `Ah 7d 2c Ks` -> `7s` | 0.3500 | 0.7075 | **+0.3575** | no (texture change) |
| flush completes **and** connects, `2h 7h Th Kd` -> `4h` | 0.5350 | 0.7125 | +0.1775 | yes |

Minimum draw-completing delta **0.0000**; maximum blank delta **0.1575**;
maximum non-completing delta **0.3575**. `separable by any threshold: False`.
A four-flush completion scores **strictly less than a blank**, and a straight
completion is **numerically identical** to a blank. At the proposed 0.30 the
predicate classifies **only the board-pairing river** as completing — the one
case in the table that is not a draw completing at all.

**Why**, which is the part that matters for any repair
(`board_coordination`, `engine/table_simulator.py:1424`,
`score = 0.45*suited + 0.35*connected + 0.20*paired`):

1. `connected` is a **saturating min-span** term: `max(0, 1 - (span-1)/4)` over
   the tightest gap between any two distinct board ranks. A new rank can only
   split an existing gap or fall outside the range, so the minimum gap never
   grows and the term **never falls**; it is already **1.00 on most turns**
   (`Ah 7d 2c Ks` scores 1.00 because K and A are adjacent ranks). A straight
   completing therefore moves it by **zero**. Likewise `suited` and `paired`
   are non-decreasing in the number of cards, so the whole score is monotone.
   [MEASURED, PB: over **20,000** random 5-card boards the turn->river delta
   was negative **0/20,000** times; min 0.0000, max 0.5525.]
2. The `suited` term carries the delta instead: a river that merely makes a
   *second* card of some suit moves `suited` 0.00 -> 0.35, i.e.
   `0.45 x 0.35 = +0.1575`. That is the blank's entire delta.
3. A genuine flush completion moves `suited` 0.80 -> 1.00, i.e.
   `0.45 x 0.20 = +0.0900` — *less* — and on an already-four-flush turn it
   moves nothing at all.
4. Only `paired` (weight 0.20, plus a suit move) can clear 0.30.

**The function is not broken.** It measures how coordinated a board *is*, and
its docstring says exactly that. It is simply not a completion detector, and
the delta of a monotone texture score is not a completion signal.

**Repair: an explicit structural predicate, no dial.** A river completes when
the fifth card makes a hand available to a range holding **one** specific
card that was not available on the turn:

- **flush** — the river's suit already appeared 3+ times on the turn (so the
  board now shows four of that suit); or
- **straight** — some window of five consecutive ranks now contains **four or
  more** distinct board ranks, and no window did on the turn (so a straight
  that needed two hole cards now needs one). The windows are the ten sets
  `{A,2,3,4,5}` (the wheel), `{2,3,4,5,6}`, ..., `{T,J,Q,K,A}` — stated
  explicitly because "a window the turn did not already contain" is exactly
  the ambiguity that made the first repair pass's version under-count
  (§10.C); or
- **pair** — the river's rank already appeared on the turn.

The first two are draw completions in the strict sense. The third is a
**texture change**, not a draw completing, and it is kept as a separately
named reason precisely so the owner can exclude it (§2.5b, §9.2). Three named,
separately-reportable booleans; no threshold, no `completion_delta` constant,
nothing to ablate — CLAUDE.md rule of work 5 is satisfied by removing the
constant rather than by pinning it.

[MEASURED, PB] Under this predicate all **11 of 11** crafted boards in the
table above classify as the "draw completing?" column states (with the pairing
river reported under `pair` and the two blanks and the third-suit river under
no reason at all). **This fixture set would have failed on the drafted
`board_coordination` predicate at every threshold** — which is the point, and
it is invariant I5 in §5.

### 2.5 What the repaired slice actually yields

[MEASURED, P2] Both proposed archetypes (§3), 2,000 hands each at 60 bb, match
seed 200, three heroes:

| hero | archetype | hero decisions (per hand) | priced | priced river | of those, completing | eff==0 among those | **slice, repaired** | **% of hands** | slice as first drafted |
|---|---|---|---|---|---|---|---|---|---|
| scripted median | `river-overbet` | 5,085 (**2.54**) | 2,100 | 323 | 216 | 91 | **28** | **1.40%** | **0** |
| `always_check_call` | `river-overbet` | 8,231 (4.12) | 3,186 | 886 | 612 | 265 | **125** | 6.25% | **0** |
| `always_fold` | `river-overbet` | 3,805 (1.90) | 1,649 | 202 | 136 | 0 | **12** | 0.60% | **0** |
| scripted median | `completion-shover` | 5,796 (**2.90**) | 1,409 | 377 | **377** | **377/377** | **150** | **7.50%** | **0** |
| `always_check_call` | `completion-shover` | 7,796 (3.90) | 1,788 | 647 | 647 | 647 | **507** | 25.35% | **0** |
| `always_fold` | `completion-shover` | 4,638 (2.32) | 1,294 | 297 | 297 | 297 | **98** | 4.90% | **0** |

Completion reasons over the completing rows, scripted hero (a row may carry
two): `river-overbet` 216 rows = **pair 138, flush 15, straight 66**;
`completion-shover` 377 rows = **pair 245, flush 40, straight 97**.

Pot and price at the slice decisions themselves, in bb [MEASURED, P2] — this
is the distribution §2.6 asks the instrument to report, and it is what the
§2.3 caveat about a *relative* stack clause looks like in numbers:

| arm | pot min / med / max | price min / med / max | swing `pot+call` med / max |
|---|---|---|---|
| scripted hero vs `river-overbet` (n=28) | 6.0 / 60.8 / 79.9 | 3.9 / 39.5 / 46.2 | 101.4 / 120.0 |
| scripted hero vs `completion-shover` (n=150) | 5.0 / 61.5 / 80.0 | 3.0 / 48.5 / 59.0 | 120.0 / 120.0 |
| `always_check_call` vs `river-overbet` (n=125) | 4.2 / 45.2 / 79.9 | 2.2 / 27.0 / 47.8 | 72.0 / 120.0 |
| `always_check_call` vs `completion-shover` (n=507) | 4.5 / 61.0 / 67.0 | 2.5 / 58.0 / 59.0 | 120.0 / 120.0 |

The median slice decision is a 60 bb pot at a 40-50 bb price — the deep-stack
stack-off the plan is aiming at — but the minimum is a 4-6 bb pot, so the
slice is **not** uniformly a big-pot event. The maximum swing is exactly
120.0 bb, the whole of a doubled 60 bb stack, in every arm.

**Three consequences the owner has to see before any design is chosen.**

**(a) The slice rate is a property of the HERO, not of the instrument
[MEASURED].** Across the three heroes it moves from 0.60% to 6.25% of hands on
`river-overbet` (**10.4x**) and from 4.90% to 25.35% on `completion-shover`
(**5.2x**), on identical opponents and seeds. The first draft's single "4-8%"
and the first
repair pass's single "3.35%" are both category errors: there is no
hero-independent slice rate. The two figures that matter are:

| hero | pooled over both channels, 4,000 hands | rate | **N per seed** at 2,000 hands/channel |
|---|---|---|---|
| scripted arena-median (P2) | 178 slice decisions | **4.45%** | **178** |
| **served `candidate-v9-0003b`** (PL, 600 hands) | 38 slice decisions | **6.33%** | **~253** |

[MEASURED, PL] The served learned policy generates the slice at **2.00%**
(`river-overbet`, 6 in 300 hands) and **10.67%** (`completion-shover`, 32 in
300 hands) — the same order as the scripted proxy and somewhat denser. 38
slice decisions is a small sample and it is quoted as an order-of-magnitude
check, not as the pilot's answer. `candidate-v7-0001c`'s rate is **not
measured here**; the pilot measures both arms.

**(b) "Completing board" is mostly *board-pairing*.** [MEASURED, P2, scripted
hero, both channels pooled] of **178** slice decisions, **57** carry a flush or
straight reason and **121** are pairing-only:

| definition | slice decisions per seed (2 channels x 2,000 hands) | rate | vs broad |
|---|---|---|---|
| broad (pair \| flush \| straight) | **178** | **4.45%** | — |
| narrow (flush \| straight only) | **57** | **1.43%** | **3.12x smaller** |

**Which definition the slice uses is an owner decision (§9.2), and it is a
3.1x decision.** (The first repair pass reported this as a 10x decision, on a
straight predicate that under-counted straights; see §10.C.)

**(c) The drafted predicate yields zero slice decisions in every single arm**
— 0 of 12,000 hands across six hero/archetype combinations. The slice the
whole instrument was to report as its own number did not exist.

### 2.6 The reported numbers

Each on its own line, never folded into an aggregate, all in bb/decision:

- `slice_call_frequency` — share of slice decisions the policy called;
- `slice_chips_per_decision` — realized chips over slice decisions, in
  bb/decision; ruin reported on per-agent denominators as always;
- `slice_equity_value_per_decision` — the §6.2 estimator over the same rows;
- `slice_opportunities`, the **completion-reason split** (pair / flush /
  straight) and the **pot and price distribution** (min / median / max, in bb)
  — so (b) and the relative-clause caveat of §2.3 are visible in every run;
- the **paired v7 - v9 delta** with `diffs / sd / se / t`, `MDE_own`, **and
  the instrument's own measured seed spread** — this delta is what the
  acceptance in §7 tests;
- a **strength stratification** — slice decisions bucketed by the hero's
  canonical `strength_percentile` (`engine/strength_metric.py:88` [VERIFIED])
  into bottom/middle/top terciles, so the "usually crushed, occasionally
  good" shape the plan names (plan **§0 item 2**, lines 34-36 [VERIFIED] —
  the first draft cited "§3b" for this, which is wrong; see §10) is a number
  rather than an anecdote.

This slice is the promotion wire for plan **§5 item 3** — *"The
out-of-distribution slice is non-negative"* [VERIFIED, plan line 205]. A
candidate's own slice number being non-negative is a **separate** bar from the
instrument's v7-vs-v9 acceptance here; the instrument must be accepted first.

---

## 3. Held-out opponents (shared roster)

Admissibility, plan §1 [VERIFIED, plan lines 53-56]: learned policies *in the
harvest* are excluded; evaluation has no hole-resampling requirement, so the
v7 line, the v6 champion and fresh archetypes are all admissible. The harvest
panel (`tools/build_phase_b_corpus.py:1216 _OPPONENT_KINDS`, imported by the
v9 harvester at `tools/build_phase_b_corpus_v9.py:154,161`) is exactly
`p3-median, p3-passive, p3-aggressive, median-bot, station-bot, tight-bot,
wild-bot, textured-bot` — plus learned v9 heroes. **Nothing below is in it**
[VERIFIED].

| opponent | what it is | why admissible / why useful |
|---|---|---|
| `candidate-v7-0001c` | format-2 learned, `artifacts/candidates/candidate-v7-0001c.approved.manifest.json` | never a v9 harvest seat; the duel incumbent, so the slice number sits next to the only existing OOD number |
| `heuristic-aggressive-v6` | code champion, `build_policy(aggressive=True)` (`engine/poker_policy.py:265`); `NOISE_FLOOR_CHAMPION_LABEL`, `tools/evaluate_v8.py:1012` | harvest never uses code heroes; near-free to run (0.024133 s/hand); the noise-floor reference |
| **`river-overbet`** (fresh) | `ScriptedAgent`-derived: pre-river (0.226, 0.15, 0.0); on a completing river bets/raises toward `callToAmount + 2 x pot` clamped into the legal range, else falls through to the base mixer | generates the slice with a sized wager; no P3 involved |
| **`completion-shover`** (fresh) | `ScriptedAgent`-derived: pre-river (0.0, 0.15, 0.0); on a completing river open-shoves | the densest generator — [MEASURED, P2] 377/377 priced completing-river decisions, 7.50% slice rate with the scripted hero, 10.67% with the served learned hero |

Both fresh archetypes key only on the public board and the legal-action
block, so `reads_cards = False` holds (they inherit it from `ScriptedAgent`).
**Read §1 before treating that as a virtue:** it is what makes the
villain-range average of §6.2 exact and it is also what caps what the
instrument can claim.

**P3 is deliberately excluded from the slice arena**: the P3 fit has 0 of
7,320 rows exceeding 1.1x pot (`.handoff/DATA.md` §5 [VERIFIED, quoted]), so
any candidate exploit found against P3 in the slice would be a P3 artefact,
not a finding. The battery's `vs-shover` channel is a shover archetype
*absent* from the harvest panel, but it shoves **every** street: [MEASURED,
P1] against it the hero took **491 priced decisions and 0 priced river
decisions in 400 hands** — it cannot produce this slice at all, which is why
`completion-shover` is a distinct archetype rather than a reuse of
`vs-shover`.

---

## 4. Design A — MVP-first

Angle: the smallest new surface that produces a slice number and a v7-v9
delta at all, using the realized (assumption-free) ledger, no counterfactuals.

**(i) Opponents.** `river-overbet` and `completion-shover` as two new OOD
channels; `candidate-v7-0001c` and `candidate-v9-0003b` as the two heroes;
`heuristic-aggressive-v6` through the same channels at half seeds as the
champion reference arm.

**(ii) The slice.** Recorded inline during ordinary carry-over-session matches
(`run_sessions`, `engine/table_simulator.py:1026`; `RecordingPolicy`, `:1111`
[VERIFIED]): a slice ledger on the recorder flags decisions meeting §2 and
accumulates realized chips. No new simulator path — the matches are the
existing battery-channel matches with a new opponent kind and a ledger,
exactly the `StrengthRecorder` pattern (`tools/evaluate_v8.py:714`) already
used for the strength stage. Seed construction copies `battery_tasks`
(`tools/evaluate_policies.py:261`): **`seed = 100 + index`,
`opponent_seed = 13 + index`** (200/23 for five-max) [VERIFIED at
`tools/evaluate_policies.py:282-284, 299-301`], so v7 and v9 arms are
seed-paired by construction and the paired delta is the existing
`paired_stats` (`:62`).

> **Correction carried from the first draft.** It said the slice "copies
> `battery_tasks` (hero seed 600+idx, opponent seed 60+idx)". **`600 + idx`
> and `60 + idx` appear nowhere in the tree** [VERIFIED by grep]. The real
> convention is 100+/200+ and 13+/23+, and the noise floor's own
> `invariants.battery_plan_structure` states it in those words.

**(iii) MDE and the clarity bar, from measured inputs.**

Chain: `sd_seed = sd_decision / sqrt(N)`; `sd_paired = sqrt(2) * sd_seed`;
`MDE_own(16) = 2 * sd_paired / 4`; clarity parity needs
`|Δ| = 10.55 * sd_paired / 4`.

`sd_decision` for the realized ledger is now **measured**, not assumed.
[MEASURED, PE] Conditional on the hero calling, a slice decision pays `+pot`
on a win and `-call` on a loss (the repo's own call value is
`eq*(pot + to_call) - to_call`: `engine/learned_policy_v8.py:238`, plan line
38 [VERIFIED]), so
`Var = E[e(1-e)(pot+call)^2] + Var(e*pot - (1-e)*call)` with `e` the exact
showdown equity of §6.2:

| slice row set | n | **sd_decision (realized)** |
|---|---|---|
| scripted hero, both channels, broad completion | 178 | **53.27 bb** |
| ... narrow completion only | 57 | 54.32 bb |
| ... pairing-only | 121 | 52.74 bb |
| all six P2 arms pooled | 920 | 54.03 bb |
| served `candidate-v9-0003b` hero (PL) | 38 | 47.30 bb |

The drafted **25 bb** is refuted: the measured value is **47-54 bb**, near the
whole 120 bb swing of a doubled 60 bb stack.

| arm | `sd_dec` | N/seed | `sd_paired` | `MDE_own(16)` | `\|Δ\|` for t = 10.55 |
|---|---|---|---|---|---|
| 25 (drafted), 120 (drafted) | 25.00 | 120 | 3.227 | 1.614 | **8.51** |
| **both channels pooled, scripted-hero N** | **53.27** | **178** | **5.647** | **2.823** | **14.89** |
| both channels pooled, served-hero N (PL) | 53.27 | 253 | 4.736 | 2.368 | **12.48** |
| `completion-shover` channel alone | 54.57 | 150 | 6.301 | 3.151 | **16.62** |
| `river-overbet` channel alone | 45.69 | 28 | 12.211 | 6.106 | **32.21** |
| narrow completion, pooled | 54.32 | 57 | 10.174 | 5.087 | **26.83** |

**Design A resolves a real effect but does not reach clarity parity on the
best arm.** It resolves at `MDE_own ~ 2.4-2.8 bb/decision` pooled, which is a
genuine diagnostic, and needs a true v7-v9 delta of **~12-15 bb/decision** —
a fifth to a quarter of a 60 bb stack per slice decision — for `t >= 10.55`.
Seeds needed at other true effects, pooled, `sd_paired = 5.647`:
`|Δ| = 20` -> **8.9 seeds**; `|Δ| = 10` -> **35.5 seeds**; `|Δ| = 5` ->
**142.0 seeds**.

> The first draft's "8.4, so a 5-6 delta resolves at t 6-8" rested on two
> unmeasured inputs (sd 25, N 120). The first repair pass replaced them with
> sd 40-50 and N 67 and concluded 18-23 — also wrong, because its N was
> measured on one hero with an under-counting straight predicate. Both are
> superseded by the table above.

**(iv) Cost.**

| work | hands | rate | CPU-h |
|---|---|---|---|
| v7 + v9 through 2 archetypes, 16 seeds x 2,000 | 128,000 | 0.371665 | **13.21** |
| v6 champion through 2 archetypes, 8 seeds x 2,000 | 32,000 | 0.024133 | **0.21** |
| slice ledger overhead | — | ~1% | 0.13 |
| **total** | | | **~13.6 CPU-h = 2.3 h wall at 6 workers** (68 min at 12) |

> The first draft priced the v6 row at **0.57 CPU-h** on the 2.667x-inflated
> code-policy rate. Corrected to 0.21.

**(v) Reuse.** `tools/evaluate_v8.py` (specs, task pools, `paired_stats`,
`_assert_chip_conservation` `:883`, fragment workdir, nullcheck);
`engine/table_simulator.py` (`run_sessions`, `ScriptedAgent` base,
`MatchResult`); `engine/strength_metric.py` (terciles); the noise floor (MDE
definition, seed conventions, report shape). New code: two opponent
subclasses + the ledger + the structural completion predicate, in one new
module `tools/evaluate_ood.py`, called from `evaluate_v8`'s stage list (one
module per mechanism; nothing built without a caller).

---

## 5. Design B — risk-first

Angle: the failure mode here is not sample size, it is believing a confident
wrong number. `.handoff/DECISIONS.md` §3.5 [VERIFIED]: *"Validate any
instrument against an impossible-by-construction invariant first; check
subagent success counts; a test that mirrors the implementation passes its
bug"*; §7.7: *"Three measurement scripts gave confident wrong answers before
invariant checks became mandatory."* **This record is now the fourth and fifth
data points**: three of the first draft's four load-bearing predicates were
wrong, and a single 400-hand probe found all three — and the first repair pass
then published four wrong counts of its own (§10.C), which a re-run found. B
is A's arena with the instrument validated against impossible-by-construction
checks *before* any v7-v9 number is read.

**(i) Opponents.** Identical to A (§4.i) plus two floor arms, run through the
same slice channels at 8 seeds. The modes are **`always_fold`** and
**`always_check_call`** — `TrivialAgent`, `tools/evaluate_v8.py:198`, whose
`TRIVIAL_MODES` (`:95`) are exactly `always_fold, always_check_call,
always_aggress_small, always_aggress_large, uniform_random_legal`, with
`__post_init__` raising `ValueError(f"unknown trivial mode {self.mode!r}")` on
anything else [VERIFIED].

> **Correction carried from the first draft.** It named **`always_call`**,
> which **is not a mode** and cannot be constructed. Note also that the v9
> gauntlet ran `config.floors: "v9"` [VERIFIED in the gauntlet JSON] — the
> four contract floors `always_fatal / always_passive / always_active /
> always_aggressive` plus `uniform_random_legal` (`V9_TRIVIAL_MODES`, `:111`;
> `FLOOR_SETS`, `:126`). Which floor set the slice channel runs is an owner
> choice (§9.5).

**(ii) The slice.** Identical definition and ledger to A; B adds the invariant
battery around it.

**(iii) Invariants, each pinned by a test that fails on the wrong
implementation.** This is the design's centre and it is also the §1 /
DECISIONS §3.6 check.

- **I1 — DECISIONS §3.6 floor check (replaces the drafted invariant 1).** No
  constant-intent floor may produce a better slice number than either learned
  policy. If one does, the channel is a fit diagnostic and the v7-v9 delta on
  it is not reported as generalisation. *This is the binding rule's own
  test*, and it is the reason the floors are not optional.

  > **The drafted invariant 1 is withdrawn: it was unsatisfiable on a correct
  > implementation.** It required the forced-fold floor's slice
  > *opportunities* to equal the forced-call floor's on identical seeds.
  > `always_fold` folds to any earlier-street wager (it only plays the free
  > check when checking costs nothing), so it reaches the river far less
  > often. [MEASURED, P2, 2,000 hands each, identical match seed 200]:
  >
  > | | priced river | completing | **slice opportunities** |
  > |---|---|---|---|
  > | `always_fold` vs `river-overbet` | 202 | 136 | **12** |
  > | `always_check_call` vs `river-overbet` | 886 | 612 | **125** |
  > | `always_fold` vs `completion-shover` | 297 | 297 | **98** |
  > | `always_check_call` vs `completion-shover` | 647 | 647 | **507** |
  >
  > A **10.4x** and a **5.2x** gap. Since any invariant failure quarantines
  > the run, the drafted battery would have quarantined every run.

- **I2 — detector completeness.** The forced-call floor records a *call* on
  every slice opportunity the ledger presents it; the forced-fold floor
  records **zero** slice calls and a **non-zero** opportunity count (measured
  above at 12 and 98, so the non-zero half is satisfiable). Catches a detector
  that counts folds as calls, or one that only fires when the hero calls.
  Opportunity counts are compared **within** an arm across seeds, never across
  the two arms.

  > **The drafted invariant 2 is withdrawn on two counts.** (a) It was a test
  > that mirrors the implementation — "the forced-call floor's slice ledger
  > must equal the empirical pot-odds outcome of calling everything" re-derives
  > the ledger from itself and cannot catch a detector bug (DECISIONS §3.5
  > names exactly this). (b) Its arithmetic was wrong: it said a caller
  > *"loses `pot+bet` on losses, wins the pot on wins"*. A caller loses only
  > `to_call` and wins `potChips` (which already contains the bet) — the
  > repo's own call value is `eq*(pot + to_call) - to_call`
  > (`engine/learned_policy_v8.py:238`; the same expression is quoted at plan
  > line 38) [VERIFIED]. A test pinned to the drafted sentence would fail on a
  > *correct* implementation.

- **I3 — self-duel slice null.** `candidate-v9-0003b` against itself
  (identical weights, seat-swapped) must produce paired slice diffs
  identically zero — the noise floor's `mirror_exact: true` pattern. A
  non-zero self-duel slice means the ledger reads seat or orientation.
- **I4 — arena purity.** The slice channel refuses any `StrengthAwareAgent`
  (P3) opponent at task build (test + runtime assert), per §3.
- **I5 — completion-predicate fixtures.** The eleven crafted turn/river pairs
  in §2.4 classify exactly as the "draw completing?" column states under the
  *structural* predicate, with the three reasons reported separately.
  [MEASURED: 11 of 11 pass.] **This test fails on the drafted
  `board_coordination` predicate at every threshold** — which is the point.
- **I6 — completeness of results.** Every seed match produces a finite
  per-seed value; a crashed worker is a failed result, not a missing point
  (DECISIONS §3.5, *"check subagent success counts"*).
- **I7 — chip conservation and the price floor.** `_assert_chip_conservation`
  across slice matches, per-agent ruin denominators, **and** a per-row assert
  that `contested_stack_chips >= callChips` at every slice decision — the
  invariant §2.3 measured failing 60 of 42,145 preflop rows and 0 of 5,681
  river rows. If it ever fails on a river row the slice is quarantined, not
  silently rescaled.

**(iv) Acceptance.** §7, with the additional pre-condition that all seven
invariants pass; any failure quarantines the run (UNRESOLVED, no number
enters a promotion argument).

**(v) Cost.** A's 13.6 CPU-h, plus **2** floor arms x 2 slice channels x
8 seeds x 2,000 hands = 64,000 hands. `TrivialAgent` is a **code policy**, so
it runs at the code rate 0.024133, not the learned rate:
`64,000 x 0.024133 / 3600` = **0.43 CPU-h**. Plus nullcheck ~0.1.
**Total ~14.1 CPU-h = 2.4 h wall at 6 workers** (71 min at 12).

> **Corrections carried from the first draft**, and one correction to a
> reviewer. The draft declared two floor arms and costed *three*
> ("3 arms x 16,000 hands ... 4.96 CPU-h"). One reviewer corrected this to
> 6.61 CPU-h — but that figure prices a code policy at the **learned-hero**
> rate 0.371665. On the correct code-policy rate the line is **0.43 CPU-h**,
> and B's invariant battery is therefore *nearly free* (+0.5 CPU-h on A),
> which strengthens the case for folding it in unconditionally.

**(vi) Reuse.** Everything A reuses, plus `TrivialAgent` / floor modes and
the nullcheck stage verbatim; the noise floor's `invariants` block as the
report shape.

---

## 6. Design C — statistics-first

Angle: start from the clarity bar and back out the sample size. §4 shows the
realized ledger lands short of parity, so the question is whether
per-decision variance can be removed rather than bought with hands.

### 6.1 The first draft's estimator was illegal, and is withdrawn

The draft proposed resampling **the hero's own hole cards** at each slice
decision and re-running the hero's `decide`, calling this "the exact contract
of `engine/table_simulator.py::_CounterfactualPoint`" and "reused, not
rebuilt". **The cited contract forbids exactly that operation.** [VERIFIED,
`engine/table_simulator.py:265-270`]:

```
resampled_seats = [
    seat
    for seat in seats
    if seat.agent_id != owner
    and getattr(seat.agent, "reads_cards", True) is False
]
```

with the comment at `:258-261`: *"The decision owner's hole cards and the
already-revealed board stay fixed; future board cards always resample, and a
card-blind opponent's hole cards resample too because its actions cannot
depend on them."* `owner` is `point.agent_id`. The hero is **hard-excluded**.

Two further facts make the draft's version unbuildable as "reuse":

- `_REVEALED_BOARD = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}`
  (`:112` [VERIFIED]). At a **river** decision `revealed = 5`, so
  `pool = board[5:]` is **empty** — there is no future board to resample
  either. At a river slice point the salt can move exactly one thing: the
  card-blind opponent's holes.
- The existing machinery re-runs `decide_forced` with a **pinned family**
  (`RecordingPolicy.decide_forced`, `:1125`), never `decide`. Re-running
  `decide` on a resampled holding is new machinery.

So the draft's Design C required an **`engine/` edit**, which under
`.handoff/PROCEDURES.md` §15 rule 4 would owe the §7.3 tripwire and — by
changing chance-salt semantics — puts the harvest corpus oracle
`79e61dbd4edf410a` at risk. The draft claimed the opposite. Withdrawn.

### 6.2 The repaired estimator — legal, cheaper, and needs no engine edit

At a river slice decision the hero's holding is fixed, the board is complete,
and the hero's action is the thing being scored. The **only** unknown that
drives the outcome is **the villain's holding**. Averaging over it is exactly
the conditional expectation that removes the showdown coin-flip:

> `slice_value(decision) = e * (potChips + callChips) - callChips`, where `e`
> is the hero's showdown equity with its **actual** holding against the
> villain's holdings drawn uniformly from the unseen cards.

This requires **no chance salt, no counterfactual replay, no engine edit, and
no second `decide` call.** The hero's holding, the board and the action are
all already known at the ledger; `e` is a pure hand-evaluation quantity. At a
five-card board `missing_board = 0`, so the estimand is a showdown equity over
**C(45,2) = 990** villain hands — **enumerated exactly** rather than sampled.
`engine/hand_strength.py:176 estimate_equity(...)` [VERIFIED] is the
Monte-Carlo form of the same quantity and is available if a sampled version is
ever wanted; the enumeration uses the engine's own `_shared_evaluator`.

Two honest costs of this estimator, stated rather than papered over:

1. **It changes the estimand.** The realized ledger measures "what the policy
   won"; this measures "what the policy's decision was worth against an
   unconditioned villain range". Against a card-blind archetype the villain's
   range genuinely *is* unconditioned — its actions cannot have selected its
   cards (§1) — so the two agree in expectation. That agreement is the
   cross-check, and it is valid **only** for `reads_cards = False` seats.
2. **It is defined conditional on the hero calling.** A slice decision the
   hero folds pays a deterministic 0 and adds no variance; the sds below are
   therefore the *conservative* (larger) figures, and the ledger reports the
   call frequency alongside so the two are never confused.

### 6.3 The variance reduction is now MEASURED, and it is ~0.60, not 0.45

The reduction factor is `sd(Y)/sd(X)` where `X` is the realized two-point
outcome and `Y = E[X | e]` the equity-averaged one, i.e.
`sqrt(Var(Y) / (Var(Y) + E[e(1-e)(pot+call)^2]))`. [MEASURED, PE — exact
enumeration of all 990 villain holdings at every slice decision]:

| slice row set | n | `sd_X` realized | `sd_Y` averaged | **reduction** |
|---|---|---|---|---|
| scripted hero, `river-overbet` | 28 | 45.69 | 26.98 | 0.5906 |
| scripted hero, `completion-shover` | 150 | 54.57 | 33.07 | 0.6060 |
| `always_check_call`, `river-overbet` | 125 | 39.19 | 24.23 | 0.6183 |
| `always_check_call`, `completion-shover` | 507 | 56.66 | 33.67 | 0.5944 |
| `always_fold`, `completion-shover` | 98 | 59.43 | 34.94 | 0.5880 |
| `always_fold`, `river-overbet` (degenerate — see below) | 12 | 4.92 | 2.41 | 0.4907 |
| **all six P2 arms pooled** | **920** | **54.03** | **32.42** | **0.6001** |
| scripted hero, both channels, broad | 178 | 53.27 | 32.19 | **0.6043** |
| ... narrow completion only | 57 | 54.32 | 34.45 | 0.6343 |
| ... pairing-only | 121 | 52.74 | 31.01 | 0.5880 |
| **served `candidate-v9-0003b` hero (PL)** | **38** | **47.30** | **27.22** | **0.5756** |

The factor is **stable at 0.58-0.63 across every hero, archetype and
completion definition probed**, on every arm that carries real pots. The one
outlier, 0.4907, is the `always_fold` / `river-overbet` arm, whose 12 slice
rows all sit in a 6.0 bb pot at a 3.5-4.0 bb price ([MEASURED, P2] pot min =
med = max = 6.0 bb): a forced folder never builds a pot, so that arm's absolute
sds (4.92 / 2.41 bb) are an order of magnitude below every other arm's and its
ratio is not comparable. The first repair pass's [ASSUMPTION: 0.45] is
refuted: the reduction is real but weaker than assumed, cutting sd by ~40%
rather than ~55%.

**MDE under the repaired estimator**, same chain as §4.iii:

| arm | `sd_dec` | N/seed | `sd_paired` | `MDE_own(16)` | `\|Δ\|` for t = 10.55 |
|---|---|---|---|---|---|
| A realized, both channels, scripted-hero N | 53.27 | 178 | 5.647 | 2.823 | **14.89** |
| **C averaged, both channels, scripted-hero N** | **32.19** | **178** | **3.412** | **1.706** | **9.00** |
| C averaged, both channels, served-hero N | 32.19 | 253 | 2.862 | 1.431 | **7.55** |
| C averaged, `completion-shover` alone | 33.07 | 150 | 3.819 | 1.909 | **10.07** |
| C averaged, `river-overbet` alone | 26.98 | 28 | 7.211 | 3.605 | **19.02** |
| C averaged, narrow completion | 34.45 | 57 | 6.454 | 3.227 | **17.02** |
| C as first drafted (sd 6.455, N 60) | 6.455 | 60 | 1.179 | 0.589 | **3.11** — and the draft printed **2.5** |

**C does not obviously reach parity either, but it is close.** It needs a true
delta of **7.6-9.0 bb/decision** on the pooled channels — roughly an eighth of
a 60 bb stack per slice decision. That is large, but not absurd for a policy
that busted three seasons on deep-stack stack-offs (`.handoff/DECISIONS.md`
§7.10). Seeds needed at other true effects, `sd_paired = 3.412`: `|Δ| = 10` ->
**13.0 seeds**; `|Δ| = 5` -> **51.8 seeds**. **Whether it clears is an
empirical question the pilot answers; this record does not claim it does.**

> **Correction carried from the first draft.** Its own chain (probe sd
> `25/sqrt(15) = 6.455`, per-seed `0.833`, paired `1.179`) gives
> `10.55 * 1.179 / 4 = 3.108`, not the **2.5** it printed; and its
> recommendation then justified itself with a "realistic 2.5-8 bb/decision"
> range whose low end **fails its own acceptance test**. Design A's parallel
> figure (8.51, printed 8.4) was computed correctly, which is how the slip is
> visible.

### 6.4 What is still unmeasured — the pilot's remaining job

Four inputs the designs rest on are measured above — the slice rate and `N`
(§2.5), the realized `sd_decision` (§4.iii), the variance-reduction factor
(§6.3) and the completion split (§2.5b). **Three things are not, and no
scratchpad probe can supply them:**

1. **The per-seed sd is modelled, not measured.** Every MDE above uses
   `sd_seed = sd_decision / sqrt(N)`, which assumes slice decisions inside a
   seed are independent. They are not: a carry-over session correlates stacks
   across hands, and one hand yields at most one river slice decision. Any
   positive correlation inflates `sd_seed` and every MDE above is then
   optimistic. **This is the single most important thing the pilot measures**,
   and it can only be measured by running seeds.
2. **`candidate-v7-0001c`'s slice rate and sd are not measured** — only
   `candidate-v9-0003b`'s (PL, 38 rows). If the two arms differ materially in
   `N`, the paired construction still works but the pooled `sd_decision` does
   not describe both.
3. **The instrument's own seed spread** — plan §1 demands it explicitly:
   *"must separate v7 from v9 at least as clearly as the duel does, and must
   report its own seed spread / MDE"* [VERIFIED, plan lines 61-62]. Nothing
   in this record measures it; it needs seeds.

**Pilot** at 2 seeds x 500 hands per archetype per policy, before any full
run, measuring exactly those three. The full run's acceptance is then
**pre-registered from measured inputs** before it starts, exactly as the
schedule sweep was. Cost: 2 policies x 2 archetypes x 2 seeds x 500 hands =
4,000 hands at 0.371665 = **0.41 CPU-h = ~4 min wall at 6 workers**.

**(iv) Full-run cost**, if the pilot supports it:

| work | quantity | rate | CPU-h |
|---|---|---|---|
| pilot | 4,000 hands | 0.371665 | **0.41** |
| full hands: 2 policies x 16 seeds x 2,000 x 2 archetypes | 128,000 | 0.371665 | **13.21** |
| floors (Design B, folded in) | 64,000 | 0.024133 | **0.43** |
| v6 champion reference | 32,000 | 0.024133 | **0.21** |
| ledger + nullcheck overhead | — | — | **0.23** |
| equity enumerations: 128,000 hands x 4.45%-6.33% = **5,700-8,100** slice decisions x 990 holdings = **5.6M-8.0M** hand evaluations | | [ASSUMPTION] | **unpriced** |
| **total, hands only** | | | **~14.5 CPU-h = 2.4 h wall at 6 workers** (73 min at 12) |

> **Corrections carried from the first draft.** It costed probes at
> `64,000 x 6% x 15` = 57,600 while specifying `K-1 = 14` probes per decision
> (which is 53,760), at an unsupported 0.25 s each. The repaired estimator
> makes learned-`decide` probes unnecessary entirely, so both errors dissolve.
> The enumeration cost is stated as an operation count and left **unpriced**:
> this lane runs no timed benchmark (`.handoff/PROCEDURES.md` §15 rule 3), so
> no wall figure for it is claimed here rather than invented.

**(v) Reuse.** The engine's `_shared_evaluator` / `_treys_card` (and
`engine/hand_strength.py::estimate_equity` if a sampled form is preferred);
`engine/strength_metric.py` for the tercile stratification;
`tools/evaluate_v8.py` for task/seed/paired machinery; the noise floor for the
null and MDE conventions. New code: the equity enumeration + slice ledger in
`tools/evaluate_ood.py`. **No `engine/` edit, therefore no tripwire and no
corpus-oracle obligation** — which is a real advantage over the drafted C, not
a restatement of it.

---

## 7. The acceptance test (shared; the designs differ only in how they reach it)

Let `Δ` be the paired v7-minus-v9 mean of the slice number over 16 seeds,
`sd_p` the paired sd, `MDE_own = 2*sd_p/sqrt(16)` the instrument's own MDE,
and `spread_own` the instrument's **own measured** seed spread (§6.4 item 3).
The instrument is **accepted** when all of:

1. every invariant of §5.iii passes — no number is believed without them
   (`.handoff/DECISIONS.md` §3.5);
2. `|Δ| >= max(spread_own, MDE_own)` — the evaluator's actual verdict rule
   (`duel.verdict_rule`; `tools/evaluate_v8.py:93`). Below that, UNRESOLVED —
   never a win or a loss (`.handoff/DECISIONS.md` §3.1). **`spread_own` is
   measured on this instrument's own scale; the duel's 16.78 BB/100 is a
   different scale and is not imported** (§ units, header);
3. **clarity parity with the duel**: `|t| = |Δ|/(sd_p/4) >= 10.55`, the duel's
   own t.

> **Withdrawn from the first draft:** the second half of clause 3,
> `|Δ|/MDE_own >= 2.45`. It was both **mis-derived** and **dead**.
> Mis-derived: the duel's 2.45 is `41.14/16.78` — `|Δ|` over the *seed
> spread* — while the duel's `|Δ|` over its *own MDE* is `41.14/7.8 =
> **5.2744**`. Two different denominators were presented as one ratio. Dead:
> since `MDE_own = 2*sd_p/4`, `|Δ|/MDE_own = |t|/2` identically, so
> `|t| >= 10.55` already forces `|Δ|/MDE_own >= 5.275` and the 2.45 clause
> could never bind. Clause 3 is now the single `t` test.

If (2) holds but (3) does not, the instrument exists and is a **diagnostic**,
but does not meet the plan's bar (*"separates v7 from v9 at least as clearly
as the duel"*) and must not gate a promotion on its own.

---

## 8. Judged synthesis

The three designs share one arena, one slice and one acceptance test; A, B
and C are three *angles* on it (MVP, risk, statistics), not three independent
instruments, and the table is scored on that basis. Saying so is more useful
than pretending otherwise: the recommendation below is a composition.

CPU-hours from §4-§6. Scored 1-5 (5 = best on that criterion).

| criterion | A MVP | B risk | C stats |
|---|---|---|---|
| separation clarity headroom (can it reach t >= 10.55?) | 2 (needs 12-15 bb/dec) | 2 (same ledger) | 4 (needs 7.6-9.0; measured factor 0.60) |
| CPU cost at 6 workers | 4 (13.6 h) | 4 (14.1 h) | 4 (14.5 h) |
| module reuse / new-code surface | 5 | 4 | 4 (no engine edit; +1 vs the drafted C) |
| resistance to silent wrongness | 1 | 5 | 2 without B, 5 with |
| slice readout richness | 2 | 3 | 4 |
| time-to-first-number | 5 | 4 | 3 |
| **total** | **19** | **22** | **21 (24 with B)** |

**Recommendation.**

1. **Run the pilot (§6.4) before choosing a design** — but the pilot is now a
   *smaller* question than the first repair pass made it. Four of its inputs
   are measured here, including the two the first repair pass left as
   assumptions (the per-decision sd, measured at **47-54 bb**, and the
   variance-reduction factor, measured at **0.60** where it was assumed 0.45).
   What remains is the per-seed correlation, `candidate-v7-0001c`'s
   arm, and the instrument's own seed spread — none of which can be got
   without running seeds (CLAUDE.md rule of work 1: *measure the instrument
   before the result*).
2. **Fold Design B in unconditionally, whichever design runs.** It is
   **+0.5 CPU-h** on the corrected code-policy rate, and it carries the
   DECISIONS §3.6 floor check (I1) that decides whether the channel measures
   generalisation at all. It is not a competing design; it is the admission
   test.
3. **Build the ledger so both estimators come off the same matches.** A's
   realized number and C's equity number are computed from the *same* slice
   rows at no extra match cost, so the pilot returns both and the full run
   reports both. The realized number is the assumption-free cross-check; the
   equity number is the statistic the acceptance would read. On measured
   inputs C's MDE is **1.71 bb/decision** against A's **2.82** — a 1.65x
   improvement for no additional hands.
4. **Pool the two channels or report them separately — decide before the run,
   not after.** Pooled, N = 178-253 per seed and parity needs 7.6-9.0
   bb/decision; on `river-overbet` alone N = 28 and parity needs 19.0. Plan §1
   says the slice is "reported as its own number, not folded into an
   aggregate", which is about not burying it in the gauntlet total, not about
   channel pooling — but the choice moves the bar by 2x and must be
   pre-registered (§9.7).
5. **If the pilot's per-seed sd comes in materially above the modelled
   `sd_decision/sqrt(N)`**, neither design clears parity. The instrument is
   then a **diagnostic** under §7 — reported, useful for plan §4 item 3b, and
   explicitly *not* a promotion gate. That outcome is acceptable and must be
   pre-registered as acceptable, so that a below-parity result is not
   retro-fitted into a win.

Sequencing (owner's call to start): pilot -> read the three remaining numbers
-> pre-register the acceptance -> invariants (B) -> full run (A+C ledgers from
one set of matches) -> freeze the record.

---

## 9. Open items only the owner can decide

1. Whether the instrument, once accepted, **gates promotion**. Plan §5 item 3
   says a candidate's slice must be non-negative; the v7-v9 clarity acceptance
   here is the *instrument's* bar, separate from that.
2. **The completion definition — broad or narrow.** [MEASURED] broad
   (pair | flush | straight) = 178 slice decisions per 4,000 hands with the
   scripted hero; narrow (flush | straight only) = 57 — a **3.12x** decision
   that moves the parity bar from 9.00 to 17.02 bb/decision. The record
   proposes **broad**, with the three reasons reported separately so the
   narrow number is always visible and the owner can re-cut after the pilot.
3. **Whether card-blind archetypes can carry an OOD claim at all** given
   `.handoff/DECISIONS.md` §3.6 (§1). If the answer is no, this instrument is
   downgraded to a held-out-opponent diagnostic. **This is the largest open
   question in the record.** Note the correction in §1: a card-reading
   archetype **already exists** (`StrengthAwareAgent`, `reads_cards = True`),
   so the prior work is **not** building one from scratch — it is deciding
   whether an archetype excluded for the 1.1x-pot reason in §3 can be
   re-admitted, re-fitted, or joined by a second card-reading archetype
   without that blind spot. That is a materially smaller and different piece
   of work than the earlier draft implied, and it should be weighed as such.
4. The pilot budget, and the per-seed-sd threshold (§8 item 5) at which the
   instrument is accepted as a diagnostic only.
5. Which floor set the slice channel runs — the v8 five (`TRIVIAL_MODES`) or
   the v9 four (`V9_TRIVIAL_MODES`, which is what the 0003b gauntlet ran).
6. Whether the v6-champion arms and the `river-overbet` channel stay in the
   standing instrument. `river-overbet` is the *sparse* channel (N = 28/seed
   against `completion-shover`'s 150) and costs half of the 13.21 CPU-h; it is
   also the only one that produces a *sized* wager rather than a shove, which
   is the case the plan's sentence actually describes.
7. Whether the slice number is pooled across the two channels or reported per
   channel (§8 item 4), and whether `tools/evaluate_ood.py` becomes a new
   stage of `tools/evaluate_v8.py` or a separate tool run beside it (one
   module per mechanism argues separate; the fragment workdir argues staged).
8. Whether the relative "stack-committing" clause (§2.3) should carry an
   absolute pot floor. The record proposes **no floor** (it would be an
   unestimated dial) and reporting the pot distribution instead.

---

## 10. What changed, where a reviewer was wrong, and where the first repair was wrong

### A. Refutations accepted and repaired (each verified independently here)

| # | defect | repair | my evidence, this session |
|---|---|---|---|
| 1 | `callChips > potChips` unsatisfiable | `2*callChips > potChips` | 0/1,377 (P1) and 0/42,145 (P1b) fired; max ratio 0.9804; min `pot-call` 1 chip; algebra in §2.2 |
| 2 | `0.5 * effective_stack_chips` vacuous/inverted | `0.5 * contested_stack_chips` | eff==0 on 293/1,377; **377/377** vs `completion-shover`; 97 rows differ between the two forms; PENDING_EDITS row E |
| 3 | C resamples the hero's holes; contradicts the cited contract | estimator replaced with villain-range averaging; no engine edit | `table_simulator.py:265-270`, `:258-261`, `:112`, `:1125` |
| 4 | `board_coordination` delta detects no completion | explicit structural predicate, dial removed | 11-case table, §2.4; 0/20,000 negative deltas |
| 5 | B invariant 1 unsatisfiable; invariant 2 mirrors the implementation and mis-states the call arithmetic | I1-I7 rewritten; the floor-parity requirement withdrawn | `always_fold` 12 vs `always_check_call` 125 (and 98 vs 507) slice opportunities |
| 6 | noise-floor rate 2.667x too high | 0.024133 s/hand/core | `config.workers = 6`, `cpu_count = 16` |
| 7 | C parity figure 2.5 vs its own chain's 3.11 | 3.108 shown; C re-derived from measured inputs to 9.00 | recomputed |
| 8 | `\|Δ\|/MDE_own >= 2.45` dead and mis-derived | clause withdrawn | `\|Δ\|/MDE_own = \|t\|/2`; duel's own ratio is 5.2744 |
| 9 | 100x unit ambiguity | one scale, bb/decision; `slice_bb_per_100_decisions` withdrawn | header |
| 10 | seed convention `600+idx / 60+idx` does not exist | `seed = 100+index`, `opponent_seed = 13+index` | `evaluate_policies.py:282-284, 299-301` |
| 11 | `always_call` is not a mode | `always_check_call` | `evaluate_v8.py:95, :198` |
| 12 | DECISIONS §3.4 cited for the units rule | §3.2 | `.handoff/DECISIONS.md` |
| 13 | probes specified at K-1=14, costed at x15, at an unsupported 0.25 s | estimator no longer uses learned-`decide` probes; the per-decide rate is re-derived to 0.089 and marked a proxy | §0, §2.5 |
| 14 | plan §1's "must report its own seed spread" not delivered | `spread_own` in the acceptance, and named as one of the three things only the pilot can measure | plan lines 61-62 |
| 15 | DECISIONS §3.6 (card-blind batteries) never engaged | §1, and invariant I1 | quoted |

### B. Where a reviewer was wrong, with evidence

- **"There is no plan §3b."** *Wrong.* Item **3b** exists — *"Distributional
  equity — conditional on the Phase 0 slice ... Do it only if the slice shows
  the call side actually leaking"* — at plan lines 187-191. It sits under
  plan **§4** (Phase 3 — architecture, heading at line 160), not §3 (Phase 2 —
  data, heading at line 114). The draft's error was the **section number**,
  not the item's existence, and this record cites plan §4 item 3b. (A second
  reviewer had this right.)
- **"`board_coordination` gives a four-flush completion +0.1775."**
  *Half right, and the case named does not produce it.* For the board the
  reviewer named (`Ah 7h 2h Kd -> 9h`) I measure **+0.0900**, matching the
  other two reviewers. +0.1775 **is** reachable — I reproduce it on
  `2h 7h Th Kd -> 4h` (0.5350 -> 0.7125), where the river completes the flush
  *and* tightens the min-span, moving `connected` by 0.25. So the figure is
  not fabricated; it is a different case. **The conclusion is unaffected** —
  a four-flush completion can still score below a blank, and on an
  already-four-flush turn it scores **+0.0000**.
- **The "vacuous stack clause" evidence used the wrong archetype.** The
  reviewer's 286/497 came from the battery's **permanent** shover, which is
  not this record's `completion-shover`, and which [MEASURED, P1] produces
  **0 priced river decisions in 400 hands** — it cannot generate the slice at
  all. Measured against the archetype this record actually proposes, the
  finding is **stronger**: 377/377, not 286/497.
- **"B's floors are 6.61 CPU-h, not 4.96."** *Both figures are wrong.*
  `TrivialAgent` is a **code policy**; priced at the code-policy rate the line
  is **0.43 CPU-h**. The reviewer corrected the hand count but kept the
  learned-hero rate.

### C. Where the FIRST REPAIR PASS was wrong (found by re-running, not by a reviewer)

The first repair fixed every reviewer finding, but four of its own [MEASURED]
numbers do not reproduce. All four are corrected above.

| its claim | what I measure | why it matters |
|---|---|---|
| "the `contested` form **excludes 223 / 389 / 275 rows** the `effective_stack` form wrongly admitted" | **97 rows** in total, pooled over all three opponents (20 / 45 / 32) | 223/389/275 are `priced - fired` for the contested clause, i.e. the rows that clause *rejects* — not the rows the effective form admitted. The sentence conflated two different sets and overstated the difference ~9x |
| "slice rate **3.35%**, N ~ 67 per seed" | the rate is **hero-dependent**: 0.60%-6.25% and 4.90%-25.35% across three heroes; **4.45%** pooled with the scripted hero (N = 178) and **6.33%** with the served learned hero (N ~ 253) | N drove every MDE. At the corrected N, Design A needs 12-15 bb/decision (not 18-23) and Design C needs 7.6-9.0 (not 9.2) |
| "completion reasons pair 437, **flush 58, straight 10**; narrow = 0.35%, a **10x** decision" | pair 138/245, flush 15/40, **straight 66/97** — straights are **26-31%** of completing rows, not 2%; narrow slice 57 vs broad 178, a **3.12x** decision | its straight clause ("the river's rank is new and creates a 5-in-a-row window that the turn did not already contain") is ambiguous — "contain" is never defined against a rank count — and I could not reproduce its implementation, so I state only the gap: on the explicit *one-card availability* test of §2.4, pinned by 11 fixtures, straights are an order of magnitude more common than it reported. Whichever reading was used, the 10x narrow/broad claim does not survive |
| "variance-reduction factor **[ASSUMPTION: 0.45]** ... the pilot measures this directly — it decides whether Design C exists" | **measured 0.5756-0.6343**, pooled **0.6001** over 920 slice decisions and 0.5756 on the served learned hero | it did not need the pilot: the board is complete at a river slice decision, so the equity is an exact 990-hand enumeration, not a simulation. C exists, and its advantage over A is 1.65x on sd, not 2.2x |
| "6 violations in 400 hands" of the `contested_stack_chips` docstring invariant | **60 violations in 42,145 priced decisions**, all preflop, **0 of 5,681 river rows** | the finding is real and is kept (I7); only the counts and the denominator were wrong, and the river-clean half is what makes the slice safe |
| "max blank delta 0.1575 / min completing 0.0900", "max 0.5525 over 400 random boards" | same 8 rows reproduce **exactly**; min completing is **0.0000** once an already-four-flush turn is included, and 0.5525 is the max over **20,000** boards (400 boards give 0.5075) | the conclusion is unchanged and strengthened |
| "at 12 workers" throughout | `.handoff/PROCEDURES.md` §7 and the frozen record both use **6** | wall times were halved against the project's own documented setting; CPU-h is now primary and 12 workers is marked an assumption |

### D. Found here, by neither the reviewers nor the first repair

- The slice rate has **no hero-independent value** (§2.5a) — the single most
  consequential correction in this pass, because N sets every MDE.
- The variance-reduction factor is **measurable without a pilot** (§6.3), and
  is 0.60.
- The served learned policy's own slice rate (§2.5a, PL) — 2.00% and 10.67%
  — which is the first evidence in this record that the scripted-hero proxy
  is the right order of magnitude.
- `board_coordination` is **provably monotone** in the number of board cards
  (§2.4 item 1), so *no* delta threshold can ever detect a flush completion on
  an already-four-flush turn: the delta there is exactly zero.
- The repaired stack clause is **relative**, so the slice legitimately
  contains small pots (§2.3); the instrument reports the pot distribution
  rather than pinning a floor.

---

## 11. Literal citations used above (each opened and checked in this session)

- Duel numbers: `artifacts/evaluations/candidate-v9-0003b-gauntlet.json` —
  `duel.report.paired` {mean -41.14, sd 15.6, se 3.9, t -10.55},
  `duel.report.hands_per_seed` 2000, `duel.seeds` 16,
  `duel.empirical_mde_bb_per_100` 7.8, `duel.resolvable_spread_bb_per_100`
  16.78, `duel.verdict_rule` (quoted verbatim in §0),
  `config.stage_elapsed_seconds` {battery 3221.1, duel 5162.1},
  `config.workers` 6, `config.floors` "v9".
- Noise floor: `artifacts/evaluations/noise-floor-2026-08-15.json` —
  `config.mde_definition` (`MDE(n) = 2*sd/sqrt(n)`), `config.elapsed_seconds`
  909.0, **`config.workers` 6**, `config.cpu_count` 16,
  `config.duel_seeds` 24, `config.battery_seeds` 20,
  `config.hands_per_duel_seed_per_orientation` 2000,
  `duel_channel.mirror_exact` true,
  `invariants.battery_plan_structure` (the 100+/200+ and 13+/23+ seed
  convention).
- Engine: `engine/table_simulator.py` — `:112` `_REVEALED_BOARD`, `:258-261`
  and `:265-270` the chance salt and the owner exclusion, `:554` `call_chips`,
  `:573` `pot`, `:1026` `run_sessions`, `:1111` `RecordingPolicy`, `:1125`
  `decide_forced`, `:1329`/`:1341` `ScriptedAgent` / `reads_cards = False`,
  `:1424` `board_coordination`, `:1466` `TexturedAgent`.
- Engine: `engine/game_state.py:111` `effective_stack_chips`, `:124`
  `contested_stack_chips` (and its docstring invariant, §2.3);
  `engine/hand_strength.py:176` `estimate_equity` and `_shared_evaluator` /
  `_treys_card`; `engine/strength_metric.py:88` `strength_percentile`;
  `engine/poker_policy.py:265` `build_policy(*, aggressive=False, ...)`;
  `engine/learned_policy_v8.py:238` the call value `eq*(pot+to_call)-to_call`;
  `engine/learned_policy_v9.py:589` `load_policy_v9`.
- Tools: `tools/evaluate_v8.py:93` `KNOWN_SEED_SPREAD_BB_PER_100`, `:95`
  `TRIVIAL_MODES`, `:111` `V9_TRIVIAL_MODES`, `:126` `FLOOR_SETS`, `:198`
  `TrivialAgent`, `:714` `StrengthRecorder`, `:883`
  `_assert_chip_conservation`, `:1012` `NOISE_FLOOR_CHAMPION_LABEL`;
  `tools/evaluate_policies.py:49-59` `_BATTERIES`, `:62` `paired_stats`,
  `:261` `battery_tasks`, `:282-284` and `:299-301` the seed convention;
  `tools/build_phase_b_corpus.py:1216` `_OPPONENT_KINDS` (8 kinds),
  imported by `tools/build_phase_b_corpus_v9.py:154,161`.
- Binding rules: `.handoff/DECISIONS.md` §3.1 (below MDE is UNRESOLVED), §3.2
  (another scale; never compare across), §3.5 (invariant first; subagent
  counts; a mirroring test passes its bug), **§3.6 (card-blind batteries are
  fit diagnostics)**, §7.7 (three wrong scripts), §7.10 (all three busts were
  deep-stack stack-offs).
- Procedures: `.handoff/PROCEDURES.md` §7 (gauntlet at `--workers 6`), §7.3
  (tripwire after any `engine/` edit), §15 rule 3 (no timed benchmarks) and
  rule 4 (`engine/` edit ⇒ tripwire; harvester ⇒ oracle `79e61dbd4edf410a`;
  Phase-A builder ⇒ `ecb4739df9d1b9ec`).
- Defect log: `.handoff/PENDING_EDITS.md` row **E** (effective-stack collapse;
  Fix A = `contested_stack_chips`, OFF; status PARTLY).
- P3 blind spot: `.handoff/DATA.md` §5 — 0 of 7,320 rows exceed 1.1× pot.
- Plan: `artifacts/evaluations/v9-next-layer-plan-2026-09-02.md` — §0 item 2
  ("usually crushed, occasionally good", lines 34-36), §0 item 3 (the call
  value, line 38), §1 (admissibility lines 53-56, the slice sentence lines
  57-58, the acceptance lines 61-62), §4 item 3b (distributional equity,
  lines 187-191), §5 item 3 (the promotion wire, line 205).
