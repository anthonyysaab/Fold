# engine/rules — the composed rule-layer candidates (C1–C5)

**Status: BUILT, ALL DIALS OFF — Phase 3 landed 2026-08-29.** The five
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
- Attribution is a closed vocabulary: every composed wager names its
  setter — `g`, `C2`, `stack-cap`, or `C3A` — with the damper's verdict
  carried separately.

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

**Math.** `SPR′ = (gate_stack − to_call) / (pot + 2·to_call)`, where
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

**Verdict fields.** `fired`, `spr_post`, `floor_demanded`, `equity`,
`passed`.

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
contract. At live SPR (median ~120) f_geo exceeds the cap and the clamp
returns the lane band — the candidate self-deactivates at depths where
stacks cannot be gotten in, which is the theoretically correct
degradation.

**Parameters: none.** Closed form; the blend weight is the existing read.

**Verdict fields.** `fired` (w > 0 and clamp not binding), `spr`, `n`,
`f_geo`, `w`, `f_out`.

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
When the composed target satisfies `target_to ≥ (1 − κ)·allin_j`, snap
`target_to := allin_j` for the **largest covered** such j (covering it
covers the smaller). A raise that leaves a short stack 4bb behind buys
the same fold decision at worse leverage; the snap makes it clean.

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

**Verdict fields.** `fired`, `snap_target_seat`, `allin_to`, `band_x`,
`covered_count`, `covering_count`.

---

## C4 — escalation-priced margins (`escalation_margin.py`)

**Theory.** Each re-raise multiplicatively filters the opponent's range
toward its top (classical 3-bet/4-bet theory), so required calling
equity rises with the raise count. Today `_CALL_MARGINS` keys on street
alone; the third raise of a street prices like the first bet.

**Rule.** `margin += κ_e · max(0, opp_raises_this_street − 1)`, with the
count **rebuilt to exclude hero's own actions** (the current
`raises_current_street` includes them — it measures table escalation,
not opponent pressure). The addition flows into `neutral_price`, so the
existing wildness blend dissolves it against tracked maniacs
automatically — no second mechanism, by construction.

**Parameters.** κ_e: **estimated** — recipe: in the complete-information
replays, measure `E[opponent equity vs hero | k opponent raises]` as a
function of k; κ_e is the per-extra-raise shift. Direct, because those
replays carry the opponent's actual cards.

**Verdict fields.** `fired`, `opp_raises`, `margin_added`,
`wildness_dissolved_to`.

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

**Verdict fields.** `fired` (d < 1), `d`, `bankroll`, `exposure`.

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

**κ_e (C4) — ESTIMATED, resolved.** `tools/estimate_escalation_shift.py`
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
