# The post-bust work queue — proposed, verified, and none of it ready

2026-08-26. Five work items from `.handoff/NEXT.md` were each investigated and
then handed to a reviewer instructed to break the result. **All five authors
self-reported `ready_to_apply: false`, and all five reviewers returned
`sound: false`.** Nothing here has been applied to the live path.

That is the intended outcome, not a failure. The value delivered is a verified
diagnosis and an explicit list of what each change needs before it can ship.
On 2026-08-26 an insufficiently-caveated change busted the bankroll; the same
day, five successors were stopped at the gate.

## The requirement every one of them shares

Three conditions recurred in all five reviews, and they should be treated as
the standing bar for any live-path change here:

1. **A default-off dial.** Nothing lands as a bare edit. `SafetyGates` is the
   established place; a source comment saying "do not ship this" is not a
   guard.
2. **Tests that fail on the unfixed code.** Two reviewers demonstrated that
   the existing suite cannot distinguish a proposed patch from a deliberately
   wrong version of itself — both give `715 passed, 6 skipped`.
3. **A named battery arm.** A new dial that `tools/gate_ablation.GateArm` does
   not name will silently contaminate the shared `live` arm and break its
   reproduction gate against `p3-gate-2026-08-16`.

## A correction this review forced

An earlier revision of `NEXT.md` and `PENDING_EDITS.md` claimed the
`to_call <= 0` guard "accounts for 823 of the 1,043 chips" in the worst hand.
**That figure is conditional on `candidate-v7-0001c-gates-reverted`, which was
rolled back.** Under the approved gates the 602 turn call is *already refused*
— the effective-stack denominator collapses to 1 and trips the first call gate
— verified through `load_approved`. On that hand the guard is worth the
**221-chip turn lead**. Two reviewers caught this independently, which is why
it is written out rather than quietly amended.

## Item by item

### NEXT 1 — kill the `to_call <= 0` short-circuit
**Highest leverage by decision count. Blocked on an off-distribution risk.**

The diagnosis reproduced exactly under an independent re-run: 1,369 of 4,795
decisions have `to_call <= 0` and every one has `opponent_range_width` exactly
1.0; 940 carry aggression the guard discards. Counterfactual width median
0.750, mean equity delta −0.117, and on a validated replay the served family
changes on 11.3% and the action on 13.0% of reachable rows.

**Blocker, and it is serious.** `opponent_range_width` is learning-feature 138,
and under shipped code it is 1.0 whenever `call_effective_stack_fraction`
(feature 134) is 0 — by construction, since both derive from
`allowed["callChips"]`. **0 of 1,730,110 training rows violate that
invariant.** Enabling the change serves the action-value head a feature joint
with *zero* training support on roughly a third of decisions, and the
incumbent has no protection: `hybrid_min_margin_quantile` is null in its
manifest, so the OOD branch in `learned_policy._equity_family` never runs.

**Confounded by NEXT 2.** 69% of check-spots are exactly this population, and
rows whose head returns the bias term cannot respond to feature 138 at all —
so 11.3% is a floor, not an estimate.

### NEXT 3 — Fix A for the all-in denominator collapse
**Closest to applicable. Two concrete blockers.**

Correctly sited: a new `game_state.contested_stack_chips` reached only from
`_gate_stack`, **not** a change to `effective_stack_chips`, whose twelve other
consumers include two of the 142 learned inputs, the `schema3` v8 contract,
the telemetry field every frozen report reads, and two persisted corpora.

Verified loosening-only on the journal, reproduces both `PENDING_EDITS`
reference numbers exactly (denominator 823 and 73.1% on the −1,043 turn call;
CALL on the real 84-chip record), and removes 11 of 36 collapsed-denominator
refusals. Patch applied to a throwaway copy: suite green before and after.

**Blockers**: the dial ships defaulting to `True`, so applying it would change
behaviour on the next restart — it must default `False`. And it must be named
in `gate_ablation.GateArm` and `gate_binding_audit.load_gates`, or the shared
battery's `live` arm silently becomes Fix A.

### NEXT 5 — the −1,181 hand
**Attribution solved. The fix it implies is a new defect class.**

Reproduced bit-for-bit through the deployed artifact: 0 of 142 feature
mismatches on both decisions, action `call`, and a negative control diverges.

The path: the network's argmax returned `check_call`, `_passive_action`
consulted `_call_clears_margin`, and the `(0.78, 0.626)` gate tripped and
**passed by 0.00279** because the wildness blend at `w = 0.331` pulled the
requirement to 0.51221 against an estimated 0.515.

**The new finding**: `potChips` included **1,777 chips of an over-stack all-in
that hero could never win**, so the price read 0.2823 where the real price was
0.4943. Pot odds computed on unwinnable chips is a distinct defect from
anything previously tracked, and it makes every price-based threshold in the
engine optimistic in exactly the spots where someone is all-in for more than
hero holds.

**Blockers**: needs a default-off dial (`_pot_odds` is not currently
dial-addressable, so no ablation arm can be built for it), and tests that fail
on the unfixed code — the reviewer proved the current suite cannot tell the
patch from a deliberately wrong 2× overhang subtraction.

### NEXT 4 — size `equity_trials`
**Direction right, value not derived. Anchor invalid.**

What survives: `equity_trials` is an **unpinned serve parameter** that no
approved artifact records — the same hole that shipped three unmeasured gate
changes. Pinning it in `manifest.serve` is worth doing independent of the
value. The cost and safety analysis also survives: the worst of 29 real
decision shapes takes 293.8 ms at 2,048 trials, 6.6× margin under the 2.0 s
deadline floor.

**Blockers**: 2,048 was derived from margins measured against the *recorded
200-trial draw* rather than a high-precision equity; the true margins are
0.0196 / 0.0046 / …, not 0.0323 / 0.0290 / …. Worse, it is anchored on a
decision **the live policy does not take** — under approved gates the −1,043
turn call is folded at every equity within ±0.16. Re-anchor on a served-gates
decision, or sequence behind NEXT 3, which moves the same margin again.

### NEXT 2 — the dead `action_value` head
**Diagnosis excellent and it corrects two things I asserted. Proposed fix is circular.**

`1,707 / 4,795 = 35.60%` reproduces exactly, and "dead tower" ⟺ "bias output"
is an **exact biconditional** — both off-diagonals zero across all 4,795 rows.
Per head: `behavior_prior` 0.00%, `residual_scale` 0.00%, `state_value`
33.16%. The trunk is never fully dead.

**Two framings I gave were wrong:**
- It is **not distribution shift**. The training corpus is *worse* than live,
  at 46.12%.
- "Root cause is architectural" is only half true. `behavior_prior` uses the
  same linear→ReLU→linear tower on the same trunk and is **0.00% dead**. The
  shape provides an absorbing state; the counterfactual-value objective is
  what drives entry into it.
- **Retraining alone will not fix it**: nine draws across three corpora and
  three seeds all land between 10.0% and 46.4%.

**Blocker**: the proposed tripwire's predicate — "all tower pre-activations
≤ 0" — becomes unmeasurable once a LayerNorm sits in front of the ReLU, and it
reads 0.00% on a control that must read 100%. It would have passed the gate
for a fix that did nothing. Replace it with head-output constancy measured
against an `out_w = 0` control.

## Ordered plan

| # | item | verdict | next action |
|---|---|---|---|
| 1 | NEXT 3, Fix A | closest | flip default to `False`, name it in `GateArm` + `load_gates`, then it is applicable |
| 2 | NEXT 5, pot odds on unwinnable chips | new defect | add the dial and failing tests; the attribution itself needs nothing further |
| 3 | NEXT 4, pin `equity_trials` in the manifest | do the pinning | decouple from the value; pin first, re-derive the number separately |
| 4 | NEXT 1, short-circuit | blocked | needs an answer to the OOD question, and is confounded by NEXT 2 |
| 5 | NEXT 2, dead head | scoping only | fix the tripwire before any fix is trusted; retraining alone is ruled out |

**One battery run should follow, not five.** Once Fix A carries a proper
default-off dial and a named arm:

```
python -m tools.gate_ablation --seeds 48 --workers 14 --starting-stack 6000 \
    --output artifacts/evaluations/gate-ablation-fixa-2026-08-27.json
```

with a `fix-a` arm added alongside `live`. **Falsifier**: if `fix-a` does not
beat `live` on `vs-p3` by more than its paired MDE, the collapse repair is not
worth a live-path change on this evidence and the entry should say so.

**Do items 1 and 4 interact?** Yes. Both change what the gates see — the
short-circuit alters the equity fed to every threshold, Fix A alters the
denominator those thresholds use. They must not be measured in the same arm,
and NEXT 4's anchor moves under NEXT 3. Sequence, do not batch.
