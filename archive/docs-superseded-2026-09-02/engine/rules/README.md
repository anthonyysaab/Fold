> **ARCHIVED 2026-09-02 — SUPERSEDED. NOT STATE, NOT PLAN, NOT RULES.**
> Do not use this file to learn what the system is, what its state is, or what to do.
> The live manual is `CLAUDE.md` → `.handoff/CONTEXT.md`. Open this file only for a
> specific fact that a live document cites here by path. Never edit it, never restore it.

# engine/rules — the composed rule-layer candidates (C1–C5)

**Status: BUILT, ALL DIALS OFF — Phase 3 landed 2026-08-29; two
adversarial sweeps (2026-08-29 foundations, 2026-08-30 fixes) found and
closed 21 defects between them.** The five
modules and the composition below exist exactly to this spec; nothing
consults them until Phase-4 wiring, and importing the package changes no
behavior (the zero-diff invariant is fuzzed in
`tests/test_rules_composition.py`, 15 checks; full suite 781+12 with the
old 766 baseline intact). Per the owner's documentation rule, this README
is the authoritative design record and lives with the code it governs;
`.handoff/notes/V9_RESTRUCTURE_PLAN.md` keeps only the session pointer.

Phase-3 notes, recorded where they matter:

- The target arithmetic stayed single-sourced: `aggression_sizing` grew
  explicit-fraction forms (`aggressive_arms`,
  `aggressive_target_from_fractions`, `active_wager_from_fraction`) that
  the original functions now delegate to, so the composition never
  restates a formula. Retired-endpoint bit-exactness re-verified after
  the refactor.
- **Damper supremacy needed a construction, not just an assertion**: the
  C2 blend is non-monotone in b when the geometric size sits BELOW the
  lane band (low SPR), so a damped read could raise the blended
  fraction. The composition therefore emits
  `min(damped pipeline, undamped pipeline)` whenever d < 1 — in the
  non-monotone regime the damper has no effect, which is correct: the
  geometric blend already sized down further than the damper would.
  **Attribution follows the winning run** (`_LaneRun`): the first
  implementation attributed from the discarded damped evaluation, so a
  wager emitted by the undamped pot arm could be recorded as
  `stack-cap` with a geometric verdict whose `f_out` did not reproduce
  it. Fixed 2026-08-29.
- Attribution is a closed vocabulary: every composed wager names its
  setter — `g`, `C2`, `stack-cap`, or `C3A` — with the damper's verdict
  carried separately.

## Adversarial sweep, 2026-08-29 — nine defects found and fixed

Five reviewers with distinct lenses over the foundation commits, then
adversarial verification. **Every finding below was confirmed by
interpreter repro or by the live journal, never by argument alone**, and
each carries a regression test that fails on the unfixed code.

| # | where | defect |
|---|---|---|
| 1 | `commitment_gate` | **C1's denominator double-counted the outstanding bet.** `potChips` already contains the bet hero faces — verified on 1,042 of 1,097 first-in preflop live rows (`potChips == sb + bb`, none `== sb`) and it is the convention `_pot_odds` rests on. Was `pot + 2·to_call`, now `pot + to_call`; the gate had been firing ~33% past its derived boundary. The old test *mirrored* the bad formula, so it passed — it now derives the boundary forward from the convention. |
| 2 | `decision_engine` (C5 site) | **BLOCKER: the damper GREW wagers on hot reads.** The sizer arm is monotone *increasing* in boldness, so scaling a negative b toward zero raises the fraction (b=−1, d=0.1: 0.3050 → 0.4805). My wiring comment asserted the arm was safe and skipped the min. Now `min(damped, undamped)` — the same construction the composition already used, for the same reason. |
| 3 | `composition` | **Attribution came from the discarded pipeline.** When the damper's min picked the undamped run, `set_by`, the recorded C2 verdict and `boldness_used` all described the damped one — a wager emitted by the pot arm could journal as `stack-cap` with an `f_out` that could not reproduce it. `_LaneRun` now carries the winning run's explanation. |
| 4 | `geometric_sizing` | **C2 did the opposite of its accepted degradation.** Above the lane top it clamped the *blend* at `lane_top`, so at live SPR (~120) every mildly value-leaning read jumped to the band MAXIMUM. It now returns the lane fraction untouched and reports `fired: false` — the self-deactivation the spec was accepted on. |
| 5 | `coverage_targeting` | **The snap band was one-sided.** `to_amount >= (1−κ)·allin` admitted every covered all-in however far below, and `max()` then snapped a 5,000 wager DOWN onto a lone 100-chip all-in. Now two-sided: `(1−κ)·allin <= to_amount <= allin`. |
| 6 | `escalation_margin` | **κ_e was applied to a different quantity than it was measured on.** The estimator indexes by the STREET ordinal (hero's aggression included); the wiring counted opponents only, under-pricing exactly the bet-then-raised spot the margin exists for. The counter is now the street ordinal, and the docstring's pre-amendment "flows into neutral_price" wording (which described the *opposite* composition) is corrected. |
| 7 | `branch_contract_v9` + `feature_extract_v9` | **The free-spot `raise` shape was unhandled.** 27 live decisions offer `raise` at `to_call == 0` with `betRange` null and `raiseRange` stated (blind-option preflop). The contract now counts `raise` as an active-lane wager there, and the extractor falls back to `raiseRange`. |
| 8 | `aggression_sizing` | **A composed record could be half-loaded.** `parameters_from_record` accepted a composed record and silently dropped its `rules` block — right identity, wrong sizes. It now refuses and points at `parameters_and_rules_from_record`, which is the composed inverse. |
| 9 | verdict hygiene | The C4 verdict recorded the pre-wildness margin (now `margin_raw` / `wildness` / `margin_applied`); C1 journaled `fired` even with `call_stack_gates` empty, where nothing enforces (now recorded only when a gate exists); `_rule_verdicts` stayed a live list after a decision, contradicting its own contract (now drained). Doc drift: the README's verdict-field lists named fields that never existed, and `contested_stack_chips` still claimed no live consumer — g's depth-invariant read has used it unconditionally since 2026-08-29. |

The recurring lesson, now twice: **a test that mirrors the implementation
passes the implementation's bug.** Findings 1 and 4 both had green tests
written from the same wrong premise as the code. Regression tests here
derive their expectations from an independent source — the pot
convention, the live journal, or a hand-built state.

## Second sweep, 2026-08-30 — the FIXES reviewed, twelve more defects

The first sweep reviewed the foundations; this one reviewed **the fixes**,
after they were written fast and merged. 41 agents, zero failures, 36 raw
findings, **32 confirmed / 4 refuted / 0 unverified** (the earlier run
conflated dead agents with refutations — the script now separates them).
Distinct defects, after de-duplication:

| # | where | defect |
|---|---|---|
| 10 | `estimate_escalation_shift` → `escalation_margin` | **BLOCKER: κ_e was an artifact of `K_CAP`.** The k-bucket cap was documented as a *display* bucket but was also the regressor: refitting the same 1,903 rows at caps 2 / 3 / 5 / ∞ gives 0.0904 / 0.0671 / 0.0551 / 0.0490 — a **2.8× SE spread** on a constant nobody was treating as a parameter, because the relationship is concave and no single slope survives the choice. The slope is gone; C4 now reads a **measured step table** (+0.0799 at k=2, +0.1190 at k≥3) straight off the per-k means. |
| 11 | `escalation_margin` | The slope also **extrapolated past its support** — `count − 1` was unbounded over 30 rows above k=3, so a long street demanded more than the validator's own ceiling. The step table saturates by construction. |
| 12 | `decision_engine` | **C2 and C3A were inert at serve while the extractor applied them.** The engine consults only C1/C4/C5; the composition (C2/C3A) is reached solely by the v9 path, which does not exist until L2 — but `feature_extract_v9` *does* apply them to the branch-cost features. Enabling either would have taught a corpus sizes the engine never plays. The engine now **refuses** those dials rather than silently diverging. |
| 13 | `decision_engine` (C5) | The damper journaled `fired: "sizes cooled"` on every hot-regime decision where the min discards the damped fraction and the emitted size is byte-identical to dial-off — the same false-attribution class as finding 3, at the engine site. Recorded only when the damped arm actually wins. |
| 14 | `commitment_gate` | At the effective-stack collapse `spr_post` went **negative**, contradicting the docstring and journaling a nonsense ratio beside a reason about a shove nobody can make. Clamped at 0 (fully committed), which fires the gate identically. |
| 15 | `decision_engine` | The deadline path's drain carried the comment "nothing can have fired" — false, since that path reaches `_sized_action`, so a C5 verdict was silently discarded. It is now carried out on the result. |
| 16 | `ruin_damper_sweep` | The verdict boundary used `<` rather than `<=`, so a zero difference at a zero MDE read as a **resolved directional win** — which is precisely the null mirror's signature, meaning the instrument's own control could be labelled a win. Also: the reproduction gate passed **vacuously** when a frozen channel carried no seeds, despite its stated meaning being "this run IS the frozen instrument". |
| 17 | `estimate_snap_band` | A bin walk reaching bin 0 without ever rejecting was reported RESOLVED with κ = 1.0 — a value `SnapToCoverParams` refuses. That is the flat-null shape; it now reports "no behavioural edge found", and the selftest's flat gate asserts *nothing resolves* rather than "the band is everything". |
| 18 | tests | `test_free_spot_raise_is_an_active_wager` was **vacuous** — it passed verbatim on the pre-fix contract, because `all-in` alone already made the lane legal; it now omits `all-in` so only `raise` can. The `betRange`/`raiseRange` fallback had **no test at all**. `test_still_blends_inside_the_band` asserted an identity of the implementation; it now asserts the textbook 0.54. |
| 19 | docs | The fix commit updated the defect table and some docstrings but **not the normative spec sections**: C3A still specified the one-sided band it had just fixed, C4 still specified the opponent-only count and the `neutral_price` placement, C2's docstring still printed the clamp formula that caused its bug, and C4's "corrected" verdict-field list named the two fields that same commit deleted. All rewritten. |

**What this sweep is evidence for.** Reviewing fixes is not optional
politeness: the fix pass introduced or left twelve defects, one of them a
blocker in a *published parameter*, and one (finding 12) a train/serve
divergence that no test could have caught because the suite runs
dial-off. The three lessons now on the record — a test that mirrors the
implementation passes its bug; a dead agent is not a refutation; a
documented "display" constant can be a live regressor — were each paid
for once.

## Third sweep, 2026-08-30 — the fix-pass reviewed, and what the pattern means

Scoped to commit `f37d574` alone, with a machine-checked exit criterion
(no blockers, no non-note bugs on the L2 critical path, nothing
unverified). **It was not met**: 15 raw, 8 confirmed, 7 refuted, 0
unverified. The decisive finding:

| # | where | defect |
|---|---|---|
| 20 | `escalation_margin` | **The κ_e blocker's fix reproduced the blocker.** `ESCALATION_STEPS[3] = 0.1190` was the *pooled k≥3 mean* — but index 3 is read at exactly `count == 3`, where the measured step is **+0.0978**. So it over-priced the modal 3-bet spot by 22%, and the pooled value is itself a function of `K_CAP` (at any cap ≥ 4 that cell becomes 0.0978). The dependence was **relocated, not removed**, and the spec's claim that the table "saturates at the edge of the data" was false — the data runs to k = 9. Now genuinely per-k, with saturation restated as a deliberate conservative POLICY, and pinned by a test against the artifact's own `per_k_steps`. |
| 21 | `decision_engine` (C5) | The attribution guard compared pot **fractions**, but the big-blind floor, the integer round and the legal clamp all absorb small fractional differences — so it still journaled "sizes cooled" on decisions whose emitted amount was byte-identical to dial-off, and it recorded *before* the `return None` bail (1,521 of 3,000 probed states journaled a cooling for a wager the engine then abandoned). Now deferred to the end and compared on the **emitted amount**. |
| 22 | `estimate_snap_band` | The tightened flat-null gate lost the assertion that a real edge RESOLVES, so the battery could pass while the tool never reported a band. Gate 1 now asserts resolution. |
| 23 | `commitment_gate` | The `max(0, …)` clamp shipped with no test and both definitions still stated the unqualified ratio. Pinned and corrected. |
| 24 | records | The Phase-2 κ_e block still published the retracted slope as "ESTIMATED, resolved"; the snap-band gate description still described the behaviour its own fix inverted; the published κ_r artifact still carried verdicts computed under the old `<` rule. All three corrected — the artifact **annotated rather than regenerated**, and the correction verified: **all 8 changed verdicts are the null-mirror control; κ_r = 8.0 stands.** |

**The pattern, stated plainly.** Three sweeps found 9, then 12, then 8
defects — the rate is not converging, and each round reviewed the
previous round's *fixes*. Twice now a fix has reproduced the defect it
was fixing in a new place (finding 20 is the sharpest case: a constant
documented as cosmetic decided a shipped parameter, was removed from the
slope, and reappeared in a pooled cell).

What this says is narrower than "the code is unreliable": every one of
these lives in the **rules layer, which ships entirely dial-off**, and
the two engine-side defects were *over*-attribution in telemetry, not
wrong play. The L2 critical path — contract, schema 4, extractor, g —
took one finding across three sweeps. But it does say the fix-writing
pace has been outrunning review, and the mitigation is structural rather
than more sweeping: **every parameter that reaches a dial is now pinned
by a test against its own estimation artifact**, which is the guard that
would have caught findings 10 and 20 at the moment of writing.

Five mechanisms elevate money-geometry ratios and coverage into the rule
layer. Each is its **own module, own parameter dataclass, own default-off
dial, own typed verdict** — a future bust diagnosis must attribute every
decision to exactly one active handler. No candidate imports another;
only `composition.py` knows more than one exists.

```
commitment_gate.py     C1   gate_forward_commitment      call ladder
geometric_sizing.py    C2   sizing_geometric             g, both wager lanes
coverage_targeting.py  C3   sizing_snap_to_cover         g target / shove lane
                            sizing_cover_damp            g stack cap (deferrable)
escalation_margin.py   C4   margin_escalation_priced     call margins
ruin_damper.py         C5   ruin_damper                  boldness, applied last-wins
composition.py         —    the precedence table and the zero-diff invariant
```

Standing constraints, inherited unchanged: everything ships OFF;
integration and enabling are separate acts; the v7 serve path stays
byte-identical (`load_approved` tripwire); no promotion, ever, from this
package.

---

## C1 — forward-commitment gate (`commitment_gate.py`)

**Theory.** Commitment/SPR theory (Flynn–Mehta–Miller): a call is not
priced by this street alone — it creates next-street geometry. Facing a
shove of the remaining stack E′ into pot P′, the price is
`E′/(P′+2E′) = 1/(2 + 1/SPR′)`; at SPR′ = 1 that is 1/3, odds nearly any
hand "has". Below SPR′ ≈ 1 the call IS a stack-off in installments. The
diagnosed −1,043 bust hand is exactly this shape.

**Math.** `SPR′ = max(0, (gate_stack − to_call) / (pot + to_call))` — the
denominator is the post-call pot, and `potChips` ALREADY contains the
bet hero faces (verified 2026-08-29: 1,042 of 1,097 first-in preflop
live rows read `potChips == sb + bb`; it is the same convention
`_pot_odds` rests on). The first draft wrote `pot + 2·to_call`, which
double-counted the outstanding bet and fired the gate ~33% past its
derived boundary on a pot-sized bet. Where
`gate_stack` is **the call ladder's own denominator** (`_gate_stack`,
inheriting `call_gates_on_effective_stack` and, if ever enabled,
`gate_stack_counts_committed_chips` — one denominator authority, no new
choice) and `pot` is raw `potChips` (engine convention).

**Rule.** When `SPR′ ≤ 1`, the call is evaluated **as if the strictest
existing call stack gate had tripped**: same floor
(`call_stack_gates[0]`), same reveal penalty, same wildness slide, the
existing code path verbatim. C1 adds a trigger, never a floor or a blend.

**Parameters: none authored.** τ = 1 is derived from the 1/3-price
identity; the floor is reused.

**Firing/degeneracy.** Fires on priced calls only. At the effective-stack
collapse (gate_stack → 1) SPR′ → 0 and the gate demands the stack-off
floor — the correct behavior the collapsed denominator today produces
only by accident. `equity is None` (deadline path) passes, matching
`_call_clears_margin`.

**Verdict fields** (`CommitmentVerdict.as_mapping()`, the shape that
reaches schema-3 `rule_verdicts`): `rule`, `fired`, `spr_post`,
`reason`.

---

## C2 — geometric-leverage sizing (`geometric_sizing.py`)

**Theory.** Classical geometric pot growth: to be all-in by the river
betting the same pot fraction each street, `P·(1+2f)ⁿ = P + 2E`, so

```
f_geo = ((1 + 2·SPR)^(1/n) − 1) / 2,   SPR = eff/pot,
n = betting rounds remaining incl. this one (preflop 4 … river 1)
```

Checks against the literature: SPR 13, flop → f = 1.0 (three pot bets,
the textbook result, exact). SPR 4, flop → f ≈ 0.54. SPR 1, river →
f = 1.0.

**Rule.** Value-weighted blend with the boldness band, no new constants:

```
w = max(0, b)                      # value-heavy read ⇒ geometric
f_target = (1 − w)·f_lane(b) + w·f_geo,  clamped to (0, lane_top]
```

Applies to **both wager lanes** (aggressive raise and active bet); the
lane tops stay 1.0 / 0.695 — escalation-only, no overbets, per the v9
contract. **When `f_geo > lane_top` the rule returns the lane fraction
UNTOUCHED and reports `fired: false`** — above the band the geometric
plan is infeasible (even the band maximum every street cannot get
stacks in), so the rule has no advice. At live SPR (median ~120) that
is the normal case and the candidate self-deactivates, which is the
theoretically correct degradation and the reading this spec was
accepted on. NOTE: the first implementation clamped the *blend* at
`lane_top` instead, which did the opposite — every mildly value-leaning
read at live depth jumped to the band MAXIMUM. Fixed 2026-08-29.

**Parameters: none.** Closed form; the blend weight is the existing read.

**Verdict fields.** `rule`, `fired`, `spr`, `streets_remaining`,
`f_geo`, `weight`, `f_out`, `reason`.

---

## C3 — coverage targeting (`coverage_targeting.py`)

**Theory.** With ruin absorbing, loss against a covered opponent is
bounded by *their* stack while fold pressure on them is total; a
stack-off against a coverer risks ruin (Kelly asymmetry). The saturated
global lead score (live median +79.5, pinned ≥ +70 on 63.9% of
decisions) is NOT the signal; per-opponent, table-scoped cover margins
are.

**Rule A — snap-to-cover (primary).** For each active opponent j hero
covers, their all-in to-amount is `allin_j = currentBet_j + stack_j`.
When the composed target lands in the band BELOW that all-in —
`(1 − κ)·allin_j ≤ target_to ≤ allin_j` — snap `target_to := allin_j`
for the **largest covered** such j (covering it covers the smaller). A
raise that leaves a short stack 4bb behind buys the same fold decision
at worse leverage; the snap makes it clean. The band is **two-sided**:
the rule closes a small gap upward, and a one-sided test admits every
covered all-in however far below, letting `max()` snap a large wager
DOWN onto a tiny stack.

**Rule B — cover damp (deferrable).** When an active opponent covers
hero, scale the aggressive stack cap `s` down with the cover margin.
Nearly vacuous at live depth (hero covers everyone); real on the 60bb
instrument and in the shrunken-roll regime — which is C5's regime, so
Rule B ships as its own dial and MAY be deferred to the C5 design if the
estimation below finds no support.

**Parameters.** κ (snap band): **estimated** — recipe: in the
complete-information replay archive, take bets landing within x of an
opponent's remaining stack and measure the fold-response curve vs x; κ is
the band where the response is statistically indistinguishable from the
response to exact all-ins. If the archive lacks resolution, κ falls back
to owner-set (flagged; proposed 0.15) — recorded as which one it was.
Rule B's damp slope: estimated from the same archive or deferred.

**Verdict fields.** `rule`, `fired`, `to_amount`, `snapped_to`,
`candidates`, `reason`.

---

## C4 — escalation-priced margins (`escalation_margin.py`)

**Theory.** Each re-raise multiplicatively filters the opponent's range
toward its top (classical 3-bet/4-bet theory), so required calling
equity rises with the raise count. Today `_CALL_MARGINS` keys on street
alone; the third raise of a street prices like the first bet.

**Rule.** `margin += ESCALATION_STEPS[min(street_aggressions, 3)]`,
pre-scaled by `(1 − wildness)` inside the module. Two properties, both
learned the hard way:

- The count is the **street ordinal, hero's own aggression included** —
  the quantity the estimator indexes by. Excluding hero applies a
  measured number to a different quantity and under-prices the
  bet-then-raised spot the margin exists for.
- The scaled margin is added to the **call margin, never to
  `neutral_price`**. The gate blend is
  `required = (1 − w)·floor + w·neutral_price`, which slides TOWARD
  neutral_price as wildness rises — so a margin placed there would be
  *preserved* against a tracked maniac, exactly backwards.

**Parameters — a MEASURED STEP TABLE, not a fitted slope.** The per-k
means are the measurement: k=1 **0.6436** (n=1587), k=2 **0.7235**
(n=231), k≥3 **0.7626** (n=85), so the extra equity demanded is the step
from k=1: **+0.0799** at k=2, **+0.1190** at k≥3, and it **saturates**
there. An earlier version fitted a single slope (κ_e = 0.0671) and
multiplied an unbounded `count − 1`. That failed twice: the slope was an
artifact of the reporting cap (refitting the same rows at caps 2/3/5/∞
gives 0.0904/0.0671/0.0551/0.0490 — a 2.8× SE spread on a constant
documented as a display bucket, because the relationship is concave),
and it extrapolated past a support of 30 rows above k=3, demanding more
than the validator's own ceiling on a long street.

**Verdict fields.** `rule`, `fired`, `street_aggressions`, `margin_raw`,
`wildness`, `margin_applied`, `reason`.

---

## C5 — ruin damper (`ruin_damper.py`)

**Theory.** Kelly (1956): with an absorbing ruin barrier, tolerable risk
scales with bankroll. Every prior fix removed hero's raw stack from the
rules because it *decays* gates at a healthy roll; the unhandled
direction is the shrunken roll — the recorded death pattern (1,000 → 0
in 36 hands at roll ≈ table scale, every gate playing as if variance
were free). C5 reintroduces the raw stack deliberately, as a survival
term, in its own module — never inside g's depth-invariant read (g asks
what the table warrants; C5 asks what the roll affords).

**Rule.** Exposure unit = the deepest active opponent's total
(`stack + committed` — observable, table-scoped). Damping
`d = min(1, bankroll / (κ_r · exposure))`; effect **`b ← b·d`** before
the lanes (cools every size continuously). Gate-side tightening is
explicitly out of scope for this iteration — one effect, one dial,
attributable.

**Parameters.** Shape is Kelly-derived; the scale κ_r is **estimated
offline**: sweep κ_r in the battery and read the ruin-probability /
BB-per-100 frontier from the existing ruin column — no live cost, no
authored number. Initial sweep grid proposed 2–10.

**Verdict fields.** `rule`, `fired`, `d`, `bankroll`, `exposure`,
`reason`.

---

## Composition (`composition.py`)

Pipeline order, one place, never distributed:

```
read (g, depth-invariant) → C5: b ← b·d              (damper first, wins ties)
→ lane base f(b)          → C2: value-blend to f_geo
→ stack cap s(b)          → C3B: cover damp (if on)
→ target = min(pot arm, cap arm)
→ C3A: snap-to-cover within band                      (snap beats blend)
→ engine legalization (_sized_action — unchanged sole authority)
call side: pot odds + margins (C4 inside) + stack gates (C1 trigger added)
```

**Precedence rules (exhaustive):** (1) C5 applies first and cannot be
overridden — every downstream size is computed from the damped b. (2)
Within its band, C3A's snap overrides C2's blended target. (3) C1 adds a
trigger to the existing gate path; it introduces no floor. (4) C4's
margin rides the existing wildness blend; no rule adds a second blend.

**Invariants (tested, not asserted in prose):**
- **Zero-diff:** all five dials off ⇒ sizes bit-identical to base g and
  gate outcomes bit-identical to the current ladder, across a fuzzed
  state sweep. Integration without enablement must be a no-op.
- **Disjoint firing:** C1 gates calls; C2/C3 size wagers — they cannot
  fire on the same decision (different action families, by
  construction). Exactly one target-setter is attributed per wager.
- **Damper supremacy:** no emitted size exceeds the same state's size at
  d = 1.

**Identity.** When C2/C3 land in g's path (Phase 4), the sizing identity
bumps ONCE: `g-v9-linear-boldness-1` → `g-v9-composed-1`, and the sizing
record grows a `rules` block carrying every dial state. With all dials
off, `g-v9-composed-1` reproduces `g-v9-linear-boldness-1` bit-for-bit
(the zero-diff invariant is the proof). This lands **before** the
extractor bakes `cost_aggressive_eff`.

**Telemetry.** Verdicts are additive fields on the decision record
(schema-versioned per the standing telemetry rules); old journals are
unaffected; a diagnosis reads "this fold was C1 at spr_post 0.8", never
an inference.

## Phase 2 results (2026-08-29)

**κ_e (C4) — RETRACTED 2026-08-30; superseded by the per-k step table
in the C4 section above.** The slope below is a function of the
estimator's display cap (0.0904 / 0.0671 / 0.0551 / 0.0490 at caps
2 / 3 / 5 / uncapped) and must not be quoted. Its *census* stands — the
1,903 rows and the per-k means are what the step table is read from.
Kept, struck through, because a retracted number that vanishes is a
number that gets re-derived. ~~ESTIMATED, resolved.~~ `tools/estimate_escalation_shift.py`
over 1,196 replays, 1,903 aggressive events with true cards, zero parse
skips; gates: canonical-estimator sanity, planted-slope recovery, shuffle
null — all passed. Raiser equity vs one random holding by raise ordinal:
k=1 **0.6436** (n=1,587), k=2 **0.7235** (n=231), k=3+ **0.7626** (n=85);
canonical percentile 0.771 → 0.909 → 0.955.

    kappa_e = +0.0671 equity per extra raise (SE 0.0065, t ≈ 10.3)

Artifact: `artifacts/evaluations/escalation-shift-estimate-2026-08-29.json`.

**κ (C3) — UNRESOLVED: the archive cannot support the estimand.**
`tools/estimate_snap_band.py` over the same replays, 7,320 facing-a-wager
decisions; gates (planted step recovered at 0.85, flat null recovers full
range) passed — the instrument works; the data does not: **99.3% of field
decisions face bets under 20% of their own stack**, the informative
region c > 0.4 holds 34 rows, the all-in reference 15, and the band walk
terminates on an EMPTY bin, so the tool reports kappa = null rather than
dressing a corpus hole up as a measurement. This is the recorded P3
blind spot again ("0 of 7,320 rows exceed 1.1× pot"). Per the spec
fallback: **κ = owner-set, proposed 0.15, FLAGGED — owner to confirm or
amend before Phase 4 wires C3A.** The same hole removes any basis for
Rule B's damp slope: **C3B is DEFERRED into the C5 regime work.**
Artifact: `artifacts/evaluations/snap-band-estimate-2026-08-29.json`.

**κ_r (C5) — sequencing correction, recorded.** The spec's battery sweep
needs the damper dial to exist; the estimation therefore runs immediately
after Phase 3, not before it. Grid 2–10 unchanged.

## Phase plan

1. **This document** — owner veto gate. DONE (owner approved 2026-08-29).
2. Estimation scripts under `tools/`. DONE — results above; κ_r moved
   after Phase 3 by necessity.
3. The five modules + composition, to this spec, each with invariant
   checks. DONE 2026-08-29 — suite 781 passed / 12 skipped (old 766
   baseline intact), ruff clean.
4. Wiring. DONE 2026-08-29 (κ = 0.15 owner-confirmed same day):
   - `G_IDENTITY` bumped once, `g-v9-linear-boldness-1` →
     **`g-v9-composed-1`** (history in the g module; nothing was ever
     harvested against the old id). `composed_sizing_record` in
     `composition.py` emits the g block + all five dial states for v9
     manifests/corpus headers (lives rules-side — a record function in g
     would close an import cycle).
   - Engine: `DecisionEngine` takes `rule_layer: RuleLayerParams`
     (default all-off). C4 wired into `_call_clears_margin`'s margin —
     **amendment to this spec's wording**: the margin rides the wildness
     signal as `added·(1 − wildness)` rather than "flowing into
     neutral_price" (which is the target the floors slide TO — adding
     there would have exempted it from dissolution, the opposite of the
     intent). C1 wired as an extra trigger on the strictest call stack
     gate (enumerate + or-condition; floor/penalty/slide reused
     verbatim). C5 wired into `_sized_action`'s temperature arm
     (monotone there, so no min-with-undamped needed; the hyper branch
     deliberately bypasses it — the anti-modeling floor is not a tuning
     surface).
   - Every consulting site guards on its dial, so dial-off skips even
     the event walks. Byte-identity evidence: the entire pre-existing
     suite passes unmodified against the wired engine (781/12; the old
     766 baseline intact). The gate_ablation reproduction-gate tripwire
     remains the instrument-level check when the next battery runs.
5. Verdict telemetry fields. DONE 2026-08-29: `DecisionResult` gains
   `rule_verdicts` (fired attributions only, deduped, in firing order;
   None whenever no rule fired — every decision while the dials ship
   off, so records stay byte-identical to the pre-rules era).
   `TELEMETRY_SCHEMA_VERSION` 2 → **3** with readable {1,2,3} (old
   journals stay loadable); the journal record carries the optional
   `rule_verdicts` field. End-to-end proven in
   `tests/test_rules_composition.py::VerdictTelemetryTests`: a real
   decide-with-diagnostics under C1 records "C1-forward-commitment at
   spr_post <= 1" into the journal record; dial-off decisions carry
   null. Suite 785/12. One test lesson recorded there: the base engine
   never consults `_family` (a learned-backend hook) — the heuristic
   ladder routes by equity, so forcing the passive path in tests means
   removing the raise from the legal set, not stubbing `_family`.

   κ_r sweep. DONE 2026-08-29 (details above):
   `tools/ruin_damper_sweep.py` (sibling of gate_ablation on the frozen
   60bb instrument — mirror null, chip conservation, per-seed
   reproduction gate, grid {2,3,5,8,10}, channels trio + vs-shover).
   **Measured finding that re-targeted the sweep**: on the v7 learned
   subject the damper was consulted ZERO times in 1,000 probed hands —
   the v7 head pins `_branch_pot_fraction` on every aggressive decision,
   so the temperature arm C5 cools is never reached; a v7-subject sweep
   is structurally a null (this is the same head pathology v9 exists to
   fix, showing up as instrument blindness). The sweep therefore runs on
   the noise-floor champion `heuristic-aggressive-v6`, which sizes
   through the damped arm; its baseline reproduces the frozen
   `heuristic-aggressive-v6` arm of `p3-gate-2026-08-16` per-seed
   (verified in the smoke). A kappa_r chosen here prices the MECHANISM;
   **re-sweep on the v9 composed lanes before enabling the dial on any
   v9 artifact.**

   **Sweep RESULTS (32 seeds, 896 matches, all instrument gates PASS;
   `ruin-damper-sweep-2026-08-29.json`).** The Kelly frontier is real
   and measured. The ruin channels: vs-station (baseline 1.725
   busts/100) — resolved ruin reductions at kr >= 3, largest at kr-8
   (−0.39 busts/100), paid for with a resolved −29 to −35 BB/100 there
   (survival priced against callers, the classic trade); vs-shover
   (baseline 4.33 busts/100) — busts-less RESOLVED at kr-5 and kr-8
   with the EV delta actually trending POSITIVE (+1.3..+2.1, at/near
   mde): against a permanent shover, big sizes only donate variance.
   vs-p3 and vs-median: everything unresolved at 32 seeds. Chosen
   default: **kappa_r = 8.0** — the largest measured survival purchase
   with the EV cost resolved only on vs-station. At live scale kr-8
   engages when the roll falls under ~8x the deepest opponent's stake
   (~480bb vs a 60bb table) — inside the corridor the 2026-08-26 bust
   trajectory passed through. The dial still ships OFF.
