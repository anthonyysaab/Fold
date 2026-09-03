# S17 bust post-mortem — `candidate-v9-0003b` — 2026-09-03

**Scope:** the RAILS questions only, per the lane prompt. The model is being
retrained on PHH/Pluribus data; nothing here asks what an OLS baseline or a
retrained policy would have done. Numbers come from
`artifacts/evaluations/s17-bust-postmortem-2026-09-03.json`, produced by
`python -m tools.session_postmortem` reading `.arena-training.jsonl` **raw**
(the official training reader refuses this journal — defect 25, the 104
duplicate `hand_result` rows — so the post-mortem reader is the raw one, with
every skip counted).

## Correction, 2026-09-03 — the central rails claim in the first draft was wrong

The first draft of this record said, twice, that **on the bust hand the rails
executed the composed proposal literally on all four streets, with no
override.** That is false. On the bust hand's **preflop** decision the composed
layer proposed the `active` branch — which at a price of 3 chips executes a
**call** — and the engine submitted a **raise to 20** instead, because the
bluff advisor fired: `bluff_kind == "steal"`
(`engine/decision_engine.py`, the bluff-advisor block at the end of
`DecisionEngine.decide_with_diagnostics` — lines 1339–1373 as of 2026-09-03).

The instrument missed it for two reasons, both now fixed in
`tools/session_postmortem.py`:

1. **The reader never read `bluff_kind`.** It was not parsed, not carried on
   the decision row, not counted, not in the JSON. `bluff_kind` is the *only*
   witness of a bluff override on the record.
2. **It never asked what the branch's action IS.** It printed a
   `proposed_branch → action` frequency table, which shows `active→raise 6`
   without saying that `active` at a positive price means *call*. The branch's
   action is state-dependent (`engine.branch_contract_v9.branch_action`:
   `active` = call at a price, bet unprovoked), so a fixed reading of the pair
   table cannot separate an override from a rendering.

A third trap the first draft walked into: **`proposed_family` is not a
proposal.** It is `DecisionResult.family`, which the bluff path rewrites to
`aggress` *before* the record is built (same block, line 1352 as of
2026-09-03), so on an overridden row it already agrees with the executed
action. On all five
bluff rows `proposed_family == "aggress"` while `proposed_branch == "active"`.

The record keeps its dated identity. It had not been committed and no live
document cites it, so the correction is made in place rather than in a second
dated record; the numbers below are re-derived from the journal bytes by the
repaired tool.

### Sources, and why it matters here

* **Journal** (`.arena-training.jsonl`): branches, actions, `bluff_kind`, pots,
  prices, stacks, equities, and **hero's own wager sizes** — the record's
  top-level `amount_to`, non-null on exactly 139 of 359 rows, which is exactly
  the 139 aggressive actions (bet 57 + raise 81 + all-in 1).
* **Session logs** (`.handoff/notes/evidence/2026-09-02-s17-bust/`): used to
  **cross-check** every hero size quoted below, and as the independent witness
  of the bluff (the supervisor prints `bluff=steal` on the line).
* **Not available from the journal:** opponents' wager sizes.
  `state.recent_actions[*].to_amount` is a schema-4 field and is absent from
  all 1,534 stored events on this schema-3 journal (defect 23, owned by the
  journal lane). No opponent size is quoted in this record.

## 0. Instrument checks (run first, reported first)

| check | expected | measured | verdict |
|---|---|---|---|
| journal parse | — | 7,627 lines, 0 unparsable, 104 duplicate `hand_result` rows (defect 25), **0 duplicates on this policy's 249 tables** | pass |
| third-launch hand counts, session 1 | log: 190 actions / ~132 table ids | **190 decisions / 132 tables** | pass |
| third-launch hand counts, session 2 | log: 58 actions / ~39 table ids | **58 decisions / 39 tables** | pass |
| per-table deltas, session 2 (1498 → 0) | −1498 | **−1498 exact** (39 tables: 38 net +138, bust table −1636) | pass |
| per-table deltas, session 1 (977 → 1498) | +521 | **+518** — a **3-chip gap**, 0.31% of 977 | gap, see below |
| third-launch total (977 → 0) | −977 | **−980** (518 − 1498) | same gap |
| override witnesses on the record | every field that can move execution off the proposal | **8 enumerated** from `make_decision_record`, all pinned by test to real record keys | pass |

The 3-chip gap: the journal carries a `hand_result` row only for tables where
hero acted, and there are **0** `hand_result` rows for tables with no hero
decision. One unobserved blind-walk (small + big blind = 3) inside session 1
explains the gap exactly and cannot be confirmed from the journal by
construction. Boundary residuals of the same class appear on the earlier
launches (launch1 +196 and launch2 −218 sum to 978 against the recorded 977).
The session logs' own "telemetry settled" accounting (133 and 39) matches the
journal rows (132 + one launch-2 straggler recorded at 05:33:19.648Z, delta 0,
and 39) — the supervisor saw nothing the journal does not hold.

## 1. Which hands lost the money (a)

Third launch = 171 tables, 248 decisions. Session 1 was a grind: **+518 net**,
largest single loss **−185** (92.5 bb). Session 2 was flat (+138 across 38
tables) and then **one hand lost −1636 chips (818 bb)** — the entire remaining
bankroll, 13:28:46Z. The ten largest losses of the run:

| table | delta chips | bb | launch |
|---|---|---|---|
| `cmtk4s7kymtizot1wknrcpjws` | **−1636** | −818.0 | 3 (the bust) |
| `cmtjkfmrdtcsxot1wrtvdoi7o` | −333 | −166.5 | 2 (04:00Z, Ah 8h on 8s7d4h, turn raise 352 over eff 102, equity 0.7125 ≥ near-nut release) |
| `cmtjzsxfcfhqkot1wiyh92bf3` | −185 | −92.5 | 3 |
| `cmtk1s1aeim1vot1wo4un1tm4` | −26 | −13.0 | 3 |
| `cmtjtfcoy6dsqot1w0tdw6c85` | −24 | −12.0 | 3 |
| `cmtjky282u0xwot1wyhc1u850` | −21 | −10.5 | 2 |
| `cmtk49t6xm2l1ot1woybq5z47` | −16 | −8.0 | 3 |
| `cmtjo5q1vyqdvot1wv8j7n104` | −15 | −7.5 | 3 |
| `cmtju0i9n73mrot1w6v0hp3t0` | −12 | −6.0 | 3 |
| `cmtjfgv79m1niot1wawflofee` | −6 | −3.0 | 1 |

**The bust hand, street by street** (hero `As 4h`; board `2h 3c 5c 9s Qc`).
`branch action` is what the proposed branch executes at that price
(`branch_contract_v9.branch_action`); sizes are the journal's `amount_to`, each
matching `session-002.log` act[55]–act[58]:

| street | proposed_branch | branch action | executed | amount_to | pot | call | hero stack | eff stack | equity (record) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| preflop | active | **call** | **raise** | 20 | 12 | 3 | 1634 | 1634 | 0.3567 | **OVERRIDE — `bluff_kind=steal`, a promotion** |
| flop | aggressive | raise | raise | 143 | 79 | 34 | 1616 | 1616 | 0.887 (wheel on 2-3-5) | literal |
| turn | active | bet | bet | 230 | 331 | 0 | 1473 | 1473 | 0.938 | literal |
| river | aggressive | raise | all-in | 1243 | 2429 | 1243 | 1243 | **1** | 0.7765 | rendering (stack-reaching realization, same family) |

So the hand is **one override, two literal executions, and one rendering** —
not "literal on all four streets". The preflop steal is what put hero in the
pot with `As 4h` at all.

Second-largest third-launch loss (`Qc 3d`, −185): preflop call 4 → flop bet 8
(0.767) → turn bet 19 (0.6275) → turn call 48 (0.462; price 0.3445, margin
+0.1175, no stack gate) → river bet 104 (0.492, cap ceiling 728) — lost at
showdown. All five decisions are literal executions of their branch; so are all
five on `cmtjkfmrdtcsxot1wrtvdoi7o` (−333).

## 2. What the composed layer proposed vs what the rails executed (b)

Every decision is now classified by comparing the executed action against
`branch_action(proposed_branch, to_call)`, and family-level differences are
attributed to the record field that witnesses them.

| verdict | n of 359 | meaning |
|---|---|---|
| literal | **300** | executed action IS the branch's action at that price |
| rendering | **2** | different action, same engine family — the branch still executed |
| override | **57** | different engine family — something else was submitted |

Overrides split **52 demotions / 5 promotions**, and by witness:

| witness | n | direction |
|---|---|---|
| `bluff_kind=steal` | **5** | promotion (`active`→call became a raise) |
| *no field on the record* | **52** | demotion |

The 52 unexplained demotions are the rails' own ladder — `active`→call
executed as fold ×30, `active`→bet executed as check ×17, `aggressive`→raise
executed as fold ×4 and as call ×1 — and they are safe-direction. They are
called *unexplained* here in the literal sense: the demotion ladder
(`_aggressive_action` → `_passive_action` → fold) leaves **nothing** on the
record saying it ran, so no reader can tell a rails demotion from a
mis-projected branch. That is a second instrument gap, distinct from the
`bluff_kind` one, and it is not fixed by this lane.

The 2 renderings: one `active`→bet executed as `raise` at a blind-option
preflop spot where the Arena names the unprovoked wager "raise" (the case
`branch_contract_v9.legal_branches` documents), and the bust river's
`aggressive`→raise executed as `all-in`.

### The five bluff overrides (5 of 359)

All five are preflop, all five have `proposed_branch == "active"` with a price
to call, and all five were submitted as a raise. Sizes are the journal's
`amount_to`; every one of them is the size printed in the session log on the
line that carries `bluff=steal`, so the two sources agree independently.

| table | UTC | hole | pot | call | raise to | equity | launch | session-log line |
|---|---|---|---|---|---|---|---|---|
| `cmtjcweexij4pot1wanotiufw` | 2026-09-02T00:27:17.670Z | Ah 9c | 12 | 3 | 20 | 0.4013 | 1 | `2026-09-01T235328Z/session-001.log` act[19] |
| `cmtjf3otilksaot1wlbvarvy6` | 2026-09-02T01:33:00.274Z | 3d 5d | 8 | 3 | 16 | 0.3825 | 1 | `2026-09-01T235328Z/session-001.log` act[41] |
| `cmtjrym2r4feaot1w71qzfymz` | 2026-09-02T07:28:51.862Z | Td As | 12 | 3 | 20 | 0.4605 | 3 (s1) | `2026-09-02T053317Z/session-001.log` act[64] |
| `cmtjxqw6fcqgjot1wceour347` | 2026-09-02T10:11:06.626Z | 8h As | 13 | 3 | 21 | 0.4060 | 3 (s1) | `2026-09-02T053317Z/session-001.log` act[152] |
| `cmtk4s7kymtizot1wknrcpjws` | 2026-09-02T13:28:10.596Z | As 4h | 12 | 3 | 20 | 0.3567 | 3 (s2) | `2026-09-02T053317Z/session-002.log` act[55] — **the bust hand** |

`bluff=steal` appears exactly 5 times across all four launch logs and
`bluff_kind == "steal"` on exactly 5 of 359 journal rows; launch 2 has none in
either source. The last of the five is the bust table. All five are at
`big_blind_chips` 2 with an effective stack of 878–1,634 (439–817 bb).

### What the record does and does not carry about the proposal

* `proposed_risk_fraction` is null on **359 of 359** rows: the sizing proposal
  was never journaled. Only the branch and the executed amount are recorded.
* `proposed_family` is **post-override** and must never be read as a proposal.
  It is asymmetric in exactly the wrong way: it *disagrees* with the executed
  action on the 52 demotions (so a family-vs-action check finds those) and
  *agrees* with it on the 5 bluff promotions, because the bluff path relabels
  the family before the record is written. A family-vs-action check is
  therefore blind to precisely the override class that built the bust pot.
  Measured: `proposed_family != action_family(action)` on 52 of 359 rows, and
  `proposed_family != branch_engine_family(proposed_branch, to_call)` on
  exactly the 5 bluff rows.
* `proposed_branch` is non-null on all 359 rows, so no decision fell to the
  deadline path or to a forced pin.

## 3. Safety gates: did anything pass by a thin margin? (c)

The record carries `rule_verdicts`, `proposed_branch`/`proposed_family`,
`bluff_kind`, `submitted_risk_fraction`, `equity`, stacks, pot, price. It does
**not** carry the temperature-shaping boldness or the tracker's wildness, so
the margins below are the neutral form; both quantities only move a margin.

* **`rule_verdicts`: 0 of 359 non-null.** C1–C5 are wired dark; no rule ever
  fired. Of the eight enumerated override channels, **six are provably idle**
  on this journal (`fallback_reason` 0, `rule_verdicts` 0, `hyper_aggression`
  0, null `proposed_branch` 0, non-2xx/`response_error` 0, `belief_degraded`
  0), **one is unobservable** because it was never journaled
  (`proposed_risk_fraction`, present on 0 of 359), and **one fired**:
  `bluff_kind`, 5 times. The first draft's "the rails ran clean" is therefore
  withdrawn — it was a claim about the channels the instrument had checked,
  stated as a claim about all of them.
* **Call stack gates never triggered.** 32 calls with a price; the largest
  call was 17.37% of the effective stack — below the 0.455 trigger, let alone
  0.78. Board stack-off gates: 0 triggered. No S16-style call-gate-by-0.00279
  can exist here because no call reached a gate.
* **Thinnest price clearance: +0.00538** (`cmtjrg8y73rxwot1wxibj7waa`, preflop
  call 5 into pot 8, Kh Ac, equity 0.39 vs pot odds 0.3846). A pot-odds
  clearance, not a safety gate.
* **Risk cap: 0 violations**, 109 sub-near-nut sized bets/raises, and **exactly
  1 bound at the ceiling** (`cmtjpwx7p12a6ot1w82vie0x2`, turn bet 60 = cap
  ceiling 60, eff 131, equity 0.348 — a hand hero won +114). The cap is alive
  but almost never touched: this policy played small-ball relative to stack
  depth.
* **The one all-in of the run — the bust river — passed the gated-shove
  near-nut release by +0.1225** (equity 0.7765 vs floor 0.654). That release
  is the *only* safety gate on that action path: it is keyed on the equity
  **estimate**, and the estimate was 0.7765 for a wheel on a four-straight
  board. The wheel lost.
* **The five steals cannot be re-priced from the record.** They fired at
  equities 0.3567–0.4605, raising to 20/16/20/21/20 for a submitted risk
  fraction of 0.011–0.017 of the effective stack. `rule_verdicts` is null and
  `proposed_risk_fraction` is null on all five, and the record carries no
  bluff-path threshold, so no margin can be reconstructed for them the way the
  call and cap margins above were. That is a gap in the record, not a
  measurement.

## 4. The all-in denominator collapse (d)

**Recurred, exactly once in 359 decisions: the bust river.**
`effective_stack_chips == 1` while `hero_stack_chips == 1243` — the
`max(1, ...)` clamp of `_gate_stack`/`effective_stack_chips` with every active
opponent all-in (defect E). On this decision **no gate outcome depended on
the collapsed denominator**: the action took the equity-keyed shove lane (the
risk cap is released above the near-nut floor, and the call stack gates were
not on an all-in action's path). The record's `submitted_risk_fraction 1.0`
is an artifact of the clamp, not a measured risk. Contrast with S16, where
the same collapse *tripped* a call gate accidentally-protectively on the
turn-call that busted that season: here the composed layer chose the
`aggressive` branch, so the call ladder (which would have denominated on the
collapsed 1) never ran. The collapse state itself is unaddressed and will
recur; `gate_stack_counts_committed_chips` still ships False.

## 5. What survives the retrain

The rails are implicated in *both* directions, not exonerated of one:

1. **The bust hand's preflop was an override, not an execution.** The composed
   layer's `active` branch would have called 3 into a pot of 12 with `As 4h`
   at equity 0.3567; the bluff advisor raised to 20 instead. Everything after
   it — the flop wheel, the turn barrel, the 818 bb river shove — is
   downstream of a pot the composition did not choose to build.
2. **Release, not mis-execution, on the river.** The stack-off rode the
   equity-estimate-keyed near-nut shove release at a ~820 bb depth that no
   instrument has ever run (all instruments run at 60 bb; live was 500–2,900
   bb), and it rendered as `all-in` because that is what an escalation at
   stack-reaching size renders as.

The four durable rails findings:

1. the bluff advisor can promote a non-aggressive composed branch into an
   opening raise, and `bluff_kind` is the only trace it leaves on the record —
   nothing else journaled distinguishes such a raise from one the composition
   chose;
2. the denominator collapse state recurs on the deep stack-off itself;
3. the gated shove lane trusts the Monte Carlo equity estimate without any
   stack-denominated brake;
4. `rule_verdicts` is dark on every row *and* the rails' demotion ladder
   journals nothing, so 52 of 57 family-level overrides in this run have no
   recorded cause — the thin-margin questions (c) can only be answered by
   reconstructing arithmetic from the fields the record happens to carry.

## 6. Method note, and what was re-checked

The JSON artifact is the tool's own output run over the real journal
(`python -m tools.session_postmortem --journal .arena-training.jsonl
--policy-version candidate-v9-0003b --windows launch1:… launch2:… launch3-s1:…
launch3-s2:… --output artifacts/evaluations/s17-bust-postmortem-2026-09-03.json`).
Its windows are launch1 2026-09-01T23:53:00Z–02:38:00Z, launch2 02:38:00Z–05:33:17Z,
session 1 05:33:17Z–11:33:22Z, session 2 11:33:22Z–13:29:00Z, from
`session-001.json`/`session-002.json`/`run.json` in
`.handoff/notes/evidence/2026-09-02-s17-bust/2026-09-02T053317Z/`.

On the 2026-09-03 correction pass, **8 "executed literally / no override"
style claims** from the first draft were re-checked against all eight
enumerated override fields:

* **3 refuted** — the bust hand executed literally on all four streets; "the
  only rails demotions in the whole run were safe-direction (6 + 17)" (it
  omitted 30 `active`→fold demotions, miscounted the aggressive non-raises as
  6 folded/called when 5 folded/called and the 6th was the river's `all-in`
  rendering, and omitted the 5 bluff promotions entirely); and the section-5
  conclusion that the bust was a proposal executed literally.
* **2 narrowed** — "the rails ran clean" becomes six channels idle, one
  unobservable, one fired; "only the branch, family and the executed amount"
  becomes only the branch and the executed amount, because `proposed_family`
  is post-override and is not a proposal.
* **3 confirmed unchanged** — `proposed_risk_fraction` null 359/359;
  `rule_verdicts` 0/359; the river's `aggressive` branch is why the call
  ladder never ran.

A further **9 numeric claims** outside that class (parse accounting, the two
other large-loss narratives, both call-gate claims, the thinnest price
clearance, the risk-cap counts, the near-nut margin, the single denominator
collapse) were re-verified against the regenerated JSON and are unchanged.

The repaired reader's regression tests are
`tests/test_session_postmortem.py::OverrideDetectionTests`, built on a
synthetic row of the bust hand's shape. They were checked against two
mutations of the reader — one that drops `bluff_kind` on load, one that
compares `proposed_family` to the executed family instead of projecting the
branch — and the second mutation makes that row report `literal`, i.e. it
reproduces the first draft's error and the tests catch it (4 of 4 fail).

The correction is recorded in place because this record had not been committed
and no live document cites it. From this landing it is frozen: numbers are
pinned to the journal bytes, and a later disagreement gets a new dated record,
not an edit.
