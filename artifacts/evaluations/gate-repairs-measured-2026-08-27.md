# The three gate repairs, measured — none of them earns its way in

Three repairs were added on 2026-08-27, each defaulting OFF, each with a
battery arm one field from `live`. This is the run that decides whether any of
them should be turned on.

**Verdict: none. All three stay dark.**

## The instrument, before the result

| check | verdict |
|---|---|
| reproduction gate vs frozen `p3-gate-2026-08-16` | **PASS** — `vs-p3`, `vs-median`, `vs-station` all per-seed identical |
| null mirror (`live` under a second label) | **PASS** — every paired difference exactly 0 |
| chip conservation | **PASS** — 0 violations |

Config: 60bb (6,000 chips at a 100 big blind), 80 equity trials, scale 1.0,
48 seeds, 1,000 hands/seed. The same instrument the frozen reports were
produced on.

The `revert-*` arms reproduce their 2026-08-26 values exactly (`revert-cap`
+7.58, `revert-calls` +16.02, `revert-all` +16.49 on `vs-p3`), which is a free
determinism check across two separate runs a day apart.

## The result, on `vs-p3`

Positive means the arm scored **higher than `live`**.

| arm | BB/100 | t | paired MDE | verdict | ruin |
|---|---|---|---|---|---|
| `fix-a` | +3.94 | 1.69 | 4.66 | **UNRESOLVED** | **busts MORE** (+0.060, t=3.99) |
| `winnable-price` | +0.54 | 1.27 | 0.85 | **UNRESOLVED** | unresolved |
| `condition-unpriced` | **−2.68** | −1.55 | 3.46 | **UNRESOLVED** | unresolved |

Not one of them resolves as an improvement. Two are flat and one trends
negative.

## What each result means

**`fix-a` — the all-in denominator repair.** The only arm with a *resolved*
effect, and it is the wrong sign: it **busts more** (+0.060 busts/100,
t=3.99). That is exactly what it should do. The repair is loosening-only — it
re-admits calls the collapsed denominator refused — and some of those calls
lose. The 84-chip fold into a 2,328 pot at 0.69 equity is still indefensible
and the repair still fixes it, but fixing it costs ruin and buys no resolved
EV. **This is a correctness fix, not a performance one, and it should be
argued on those terms or not at all.**

**`winnable-price` — pricing calls on winnable chips.** +0.54 BB/100 against a
0.85 MDE. It acts on so few decisions that the battery cannot see it; the
live-journal audit already said the overhang is legible on a small minority of
spots. Correct arithmetic, no measurable effect here.

**`condition-unpriced` — conditioning the range when hero acts first.** The
only arm trending *negative*, which is consistent with the reason it was
already blocked: feature 138 is 1.0 whenever feature 134 is 0 in **all
1,730,110 training rows**, so enabling it serves the action-value head a
feature joint with zero training support on about a third of decisions, and
the incumbent's OOD guard is disabled. The batteries appear to be paying for
that. It should not be enabled against this network at all — it needs a model
that can express the joint.

## The decision

All three remain `False`. That is where they already ship, so **nothing
changes and no live-path behaviour moves.**

This is the third time in two days that a plausible-sounding repair failed to
survive measurement — after the gate revert that busted the bankroll and the
first ablation that ran at the wrong depth. The pattern is worth naming: in
this codebase, a fix that is obviously correct on one hand has repeatedly
turned out to be worth nothing or worth less than nothing across a battery.
Shipping them off by default was the right call and this run is the reason.

## Caveats

- `vs-p3` is heads-up, and P3's aggression and sizing remain card-blind — only
  its folding is card-aware.
- 60bb is the repo's published-battery convention, **not** an arena match. Live
  play is 500–2,900bb deep. None of these arms has been measured in the regime
  they were designed for, which is the same gap that has been open since the
  gate decision.
- Batteries remain a fit diagnostic. A trivial floor still beats every real
  policy on these channels.
- UNRESOLVED is not "no effect". It means the effect, if any, is below what 48
  seeds can resolve on this instrument.
