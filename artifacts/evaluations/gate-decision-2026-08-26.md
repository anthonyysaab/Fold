# The three unmeasured gate changes — measured, 2026-08-26

Closes the measurement half of `.handoff/NEXT.md` item 2: *"the effective-stack
risk cap, the effective-stack call gates, and `reveal_expense_equity_slope =
0.12` ship on the next supervisor restart whether or not anyone intends it.
Measure them or revert them."*

**They are now measured. The decision on what to do with the measurement is the
owner's and has not been taken.** Nothing in this work promoted, deployed, or
altered a live-path default; `artifacts/approved.json` is untouched.

## Sources

| artifact | what it prices |
|---|---|
| `gate-ablation-60bb-2026-08-26.{json,md}` | what each edit is worth in BB/100 and in ruin, on batteries |
| `gate-binding-audit-2026-08-26.{json,md}` | how often each edit changes a verdict on the **stored live journal**, and which real hands it lands on |

Neither is sufficient alone. The battery has an EV scale but scripted
opponents; the journal has real hands but cannot replay a hand that never
happened.

## A retraction that must not be quietly overwritten

An earlier run of `gate_ablation` (`gate-ablation-2026-08-26`, since deleted)
reported revert-cap +14.20 (t = 6.22), revert-calls +14.75, revert-all +19.52
with paired MDEs of 3.44–4.70. **Every one of those numbers is void.** It ran at
`--starting-stack 1000` — **10bb** against the simulator's fixed 50/100 blinds —
while every published instrument in this repo runs 6,000 (60bb), and it used 200
equity trials against the frozen 80. Both edits under test are *stack-denominator
swaps*, so depth is the treatment variable, not a nuisance parameter. It also
used a floor-free call-gate predicate that counted gate **arrival** as though it
were **refusal**, overstating that edit's live exposure by 72%.

Three instrument changes were made so this class of error fails loudly instead
of silently:

- the artifact now records `starting_stack`, `depth_bb`, `equity_trials`,
  `scale`, `seeds` and `p3_fit` — the absence of which is *why* the depth error
  went unnoticed;
- a **reproduction gate** was added: the `live` arm must reproduce
  `p3-gate-2026-08-16`'s incumbent per-seed BB/100 **exactly**, or the run is
  not the instrument the frozen reports were produced on. The two checks the
  tool originally carried — null mirror and chip conservation — are both
  invariant to stack depth by construction and could not have caught it;
- the audit now separates gate arrival from gate refusal.

## The instrument, before the result

| check | result |
|---|---|
| reproduction gate vs frozen `p3-gate-2026-08-16` incumbent | **PASS** — `vs-p3`, `vs-median`, `vs-station` all identical over 16 shared seeds |
| null mirror (`live` under a second label) | **PASS** — every paired difference exactly 0 |
| chip conservation | **PASS** — 0 violations in 720 matches |
| journal audit: `effective_stack <= hero_stack` on every record | **PASS** — 0 violations |
| journal audit: engine parity — every recorded sized bet at or under the hero-purse cap | **PASS** — 0 over, of 2,600 |

Config: 60.0bb (6,000 chips at a 100 big blind), 80 equity trials, scale 1.0,
48 seeds, 1,000 hands/seed, `artifacts/p3/p3-fit.json`.

## Result — `vs-p3`, the card-aware channel

A **positive** difference means the *reverted* arm scored higher, i.e. evidence
**against** the change that arm reverts.

| arm | BB/100 | t | paired MDE | seeds against | verdict |
|---|---|---|---|---|---|
| `revert-cap` | **+7.58** | 2.99 | 5.08 | 19 of 48 | resolved, but the weakest |
| `revert-calls` | **+16.02** | 5.89 | 5.44 | 9 of 48 | resolved |
| `revert-all` | **+16.49** | 5.34 | 6.18 | 10 of 48 | resolved |

The same direction, two to four times larger, on both card-blind controls
(`revert-all`: +66.44 on `vs-median`, +68.18 on `vs-station`).

**Card-awareness narrows the measured cost of the gates but does not reverse
it.** That was the open question — the prior −13.97 BB/100 was card-blind, and
a card-blind opponent cannot punish overcommitment. P3 can, and the gates still
lose. The shrinkage from +66.44 (`vs-median`) to +16.49 (`vs-p3`) is the size of
what card-awareness was worth: real, and not enough.

## Ruin — the one axis that favours the gates

| arm | busts/100 hands, `vs-p3` | difference | t | verdict |
|---|---|---|---|---|
| `live` | 0.0896 (43 busts) | — | — | — |
| `revert-cap` | — | +0.0200 | 1.81 | **UNRESOLVED** |
| `revert-calls` | — | +0.0800 | 5.45 | reverting busts more |
| `revert-all` | 0.1833 (88 busts) | +0.0900 | 4.89 | reverting busts more |

Reverting roughly **doubles** the bust rate. That is real and resolved for the
call gates and for the bundle; for the risk cap alone it does not resolve.

**It is already paid for.** `MatchResult.bb_per_100` is
`100 * chip_delta / (big_blind * hands)`, and a bust is a chip delta of minus
the stack — so the chips lost busting are *inside* the BB/100 figures above.
The +16.49 is net of the extra busts. Ruin here is a distribution-shape
diagnostic, not an additional cost to be added.

For the ruin reduction to justify the EV cost anyway, busting would have to be
an absorbing state worth more than its chips — the "never rebuy" rule making it
terminal. **It is not, in this format.** The arena reseats every hand (158
distinct `table_id`s for 157 hands, zero revisits), so a table bust costs that
table's stack, not the bankroll, and no rebuy is required to keep playing. The
live deployment ran ~1,000 hands from 1,000 chips to a 12,009 peak without
busting.

## What the live journal adds

| edit | changes a verdict on | hands it lands on |
|---|---|---|
| risk cap | 35 of 2,600 sub-near-nut sized bets (1.35%), 19 declined outright | 33 hands, **−1,193** total, median **+49**, 36% losers |
| call gates | 12 of 363 calls reach a gate (11 distinct), **at most 8 refused** | 7 hands, **−7,632**, worst −3,768, 6 of 7 losers |
| reveal slope | only the same calls; reverting removes 2 of the 8 refusals | max penalty 0.08 equity, median 0.001 |

Both edits barely act. Reverting therefore gives up very little of whatever
protection they provide — which is the reading that reconciles the journal with
the battery.

**A correction that matters**: the −5,000 hand that ended the deployment is
**not** refused by the call-gate edit. It clears the floor at 0.7525 equity
against 0.666 required, provably at any wildness. No argument in either
direction may attach it to this change. The −3,768 hand *is* refused, but it is
the hand the gate was derived from and is in-sample by construction.

## Recommendation

**Revert all three before any restart.** Every edit costs resolved EV on the
card-aware channel built specifically to test the counter-argument; that cost is
already net of the ruin the gates prevent; the ruin benefit does not carry the
absorbing-state value that would justify paying for it in this format; and the
live journal shows the edits act on so few decisions that little is given up.

If instead any part is **kept**, two things become mandatory:

1. **Fix the all-in denominator collapse first** (`.handoff/PENDING_EDITS.md`,
   2026-08-26). `_gate_stack(effective=True)` returns
   `max(1, effective_stack_chips(table))`, and an all-in opponent leaves that at
   0 → clamped to 1, tripping every call gate at any price *and* maxing the
   reveal expense. Verified against the incumbent's own manifest gates: it folds
   an 84-chip call into a 2,328 pot holding 0.69 equity. Live on 31 of 4,333
   decisions. Reverting the call gates to the hero purse incidentally avoids
   this, because hero's purse never collapses — keeping them does not.
2. **Write the choice into the manifest.** `candidate-v7-0001c`'s
   `engine_parameters.safety_gates` block names none of
   `reveal_expense_equity_slope`, `risk_cap_on_effective_stack`, or
   `call_gates_on_effective_stack`, so `SafetyGates.from_mapping` fills them
   from the dataclass defaults. **That gap is the mechanism that shipped three
   unmeasured changes under an approved artifact in the first place**, and it
   stays open until whatever is chosen is pinned explicitly.

Note the three are not fully separable: `revert-calls` and the slope must move
together, because the slope raises a call gate's floor and never creates a gate
— "revert the call gates, keep the slope" is a literal no-op on this journal.

## Caveats that travel with this

- **No instrument runs at live depth.** The journal is 1/2 blinds with median
  hero depth ~2,900bb and median effective depth ~900bb. The battery's 60bb is
  the repo's published-battery convention, not an arena match. This is the
  largest open gap in the result.
- **The simulator under-represents the condition these gates target.**
  `run_sessions` starts every seat at the same stack, so hero only comes to
  cover the table by winning. Live, hero covered on **84.4%** of decisions
  because its bankroll outgrew the table.
- **P3's aggression and sizing are still card-blind.** Only its folding is
  card-aware, so a policy can be punished for the prices it lays and still not
  for the hands it shows down. The blind spot is narrowed, not closed.
- **Refusal counts are ceilings.** Wildness is not recorded in the journal and
  `w > 0` strictly lowers the required equity; the honest form is "4 of 12
  provably never refuse; refusals in [0, 8]".
- **The journal figures are association, not counterfactual.** These gates fire
  on the largest prices, and large prices are mechanically where large losses
  live.
- **`vs-p3` is heads-up**, and batteries remain a fit diagnostic — a trivial
  floor still beats every real policy on these channels.
- **`revert-cap` is the weakest of the three results**: 19 of 48 seeds run
  against it, and its ruin difference does not resolve on `vs-p3`.
