# engine/rules — the composed rule layer (C1–C5)

Revision 2026-09-02. **Status: built, wired, every dial OFF.** Integration is
not enablement; each dial needs its own measurement pass (`DECISIONS.md` §2).
Design record, parameter derivations and the three-sweep defect ledger (29
defects): `archive/docs-superseded-2026-09-02/engine/rules/README.md`
(reference only). Invariants are tested in `tests/test_rules_composition.py`.

## The five mechanisms

| id | module / dial | rule | parameter (source) | wired into |
|---|---|---|---|---|
| C1 | `commitment_gate.py` — `gate_forward_commitment` | post-call `SPR′ = max(0, (gate_stack − to_call)/(pot + to_call))`; if `SPR′ ≤ 1` the call is judged as if the strictest call stack gate had tripped (same floor, penalty, wildness slide) | τ = 1, derived from the 1/3 shove-price identity; no authored number | `_call_clears_margin` trigger |
| C2 | `geometric_sizing.py` — `sizing_geometric` | `f_geo = ((1+2·SPR)^(1/n) − 1)/2`, n = streets left; blend `f = (1−w)·f_lane + w·f_geo`, `w = max(0, b)`; **above the lane top the rule returns the lane fraction and reports `fired: false`** (self-deactivates at live SPR) | none (closed form) | g, both wager lanes (v9 serve only) |
| C3A | `coverage_targeting.py` — `sizing_snap_to_cover` | if a composed `to_amount` lands in `[(1−κ)·allin_j, allin_j]` for a covered opponent j, snap to the largest such all-in (two-sided band) | κ = **0.15, owner-set** (archive lacked resolution: 99.3% of field decisions face < 20% of stack) | g target / shove lane (v9 serve only) |
| C3B | `sizing_cover_damp` | damp the aggressive cap when an opponent covers hero | deferred to the C5 regime | — |
| C4 | `escalation_margin.py` — `margin_escalation_priced` | call margin `+= step[k]·(1 − wildness)`, k = aggressions this street incl. hero's; added to the margin, never to `neutral_price` | measured per-k step table from 1,903 replay raises (k=2 +0.0799, k=3 +0.0978, per-k beyond); pinned by test against `escalation-shift-estimate-2026-08-29.json` | `_call_clears_margin` |
| C5 | `ruin_damper.py` — `ruin_damper` | `d = min(1, bankroll/(κ_r · exposure))`, exposure = deepest active opponent's stack + committed; `b ← b·d` before the lanes; emitted size never exceeds the undamped size | κ_r = **8.0** from the 32-seed frozen-instrument sweep on `heuristic-aggressive-v6` (`ruin-damper-sweep-2026-08-29.json`); **re-sweep on v9 lanes before enabling** | `_sized_action` temperature arm (hyper branch bypasses it) |

## Composition (`composition.py`) — one pipeline, one precedence table

```
read (g, depth-invariant) → C5 damps b → lane f(b) → C2 blend → cap s(b) → C3B (if on)
→ target = min(pot arm, cap arm) → C3A snap → engine legalization (_sized_action, sole authority)
call side: pot odds + margins (C4 inside) + stack gates (C1 trigger)
```

1. C5 first, never overridden; downstream sizes use the damped b; the
   emitted size is `min(damped, undamped)` and attribution follows the
   winning run.
2. C3A's snap overrides C2's blend inside the band.
3. C1 adds a trigger, never a floor. C4 rides the existing wildness blend.
4. Exactly one target-setter per wager (`g`, `C2`, `stack-cap`, `C3A`);
   C1 and C2/C3 cannot fire on the same decision.

Invariants (tested): **zero-diff** (all dials off ⇒ bit-identical sizes and
gate outcomes, fuzzed per state); **disjoint firing**; **damper supremacy**.
Sizing identity with the rules in the path: `g-v9-composed-1` (reproduces
`g-v9-linear-boldness-1` bit-for-bit with dials off). The engine refuses
C2/C3A unless the serve class declares `serves_composed_sizing` (only v9);
the extractor bakes the same dial states into the lane-cost features, and
the corpus header records them.

## Telemetry

`DecisionResult.rule_verdicts` (journal schema 3): fired attributions only,
in firing order, null when nothing fired — every decision while the dials
ship off. Verdict fields per rule: C1 `spr_post`; C2 `spr, streets_remaining,
f_geo, weight, f_out`; C3A `to_amount, snapped_to, candidates`; C4
`street_aggressions, margin_raw, wildness, margin_applied`; C5 `d, bankroll,
exposure` — recorded only when the damped arm actually wins on the emitted
amount.

## Estimators

`tools/estimate_escalation_shift.py` (C4), `tools/estimate_snap_band.py`
(C3), `tools/ruin_damper_sweep.py` (C5). Each runs an
impossible-by-construction self-test first; a parameter reaches a dial only
pinned by a test against its artifact. Lesson on record: a "display" cap was
a live regressor twice (κ_e); a test that mirrors the implementation passes
its bug.
