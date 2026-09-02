# Harvest runtime — Phase 0/1/2 of the 2026-09-02 efficiency plan

**Instrument:** `python -m tools.bench_harvest` (micro + macro oracle), built
before any optimization per the measure-the-instrument-first rule. The macro
runs the real v9 harvester at `--hands-scale 0.01 --harvest-workers 1` with
fixed seeds (59 decisions / 162 branch rows), and the corpus writer already
pins gzip mtime to zero and sorts every JSON object, so a deterministic
harvest is byte-identical across runs. **The oracle is that identity**: the
pre-change corpus sha256 `79e61dbd4edf410a` must be reproduced after every
change, or the change is not an optimization.

## Phase 0 — the instrument, validated

Three unchanged runs: **byte-identical** (sha256 `79e61dbd4edf410a`, 16,410
bytes each), wall **87.2 / 89.3 / 95.7 s** — spread 8.5s = 9.5%. That spread
is the single-run resolution: a macro change below ~10% is invisible in one
run, so per-site decisions use the micro numbers.

Micro baseline (median): 7-card `Evaluator.evaluate` **11.5 µs**;
`estimate_equity` at 1,000 trials preflop 56.9 ms / flop 49.0 ms / turn
42.8 ms / river 42.3 ms; `copy.deepcopy(seats)` **20.9 ms**, `deepcopy(policy)`
**21.4 ms** per copy (~671 top-level copies per macro run).

## Phase 1 — deepcopy -> cheap clones

`DecisionEngine.__deepcopy__` (shallow copy + `AggressionTracker.clone()` +
rule-verdict list copy; the shared equity cache stays shared as its own
`__deepcopy__` already dictated), `LearnedPokerPolicyV9.__deepcopy__`
(isolated `P3BeliefProvider.clone()`), `AggressionTracker.clone()`. The
P3 provider clone exists because a shared provider would let a replay's
`last_degrade_reason` overwrite the original's per-decision telemetry.

Measured: policy deepcopy **21.4 ms -> 0.003 ms**, seats **20.9 -> 0.20 ms**.
Macro: **72.2 s** (-19%), oracle PASS, spread collapses 9.5% -> 0.6%.

## Phase 2 — the treys evaluator

1. **Structural best-five selection** in `_six`/`_seven`: one histogram
   pass classifies the hand into bit masks (quads/trips/pairs/seen ranks/
   suit counts, with up to four cards kept per rank), then a case mirroring
   the poker hand ordering selects the five cards and `_five` ranks them
   ONCE through the same lookup tables — exact by construction, verified
   against the in-module brute-force reference (`_seven_reference` /
   `_six_reference`) on 400,000 random deals plus every hand-class
   boundary (pinned in `tests/test_treys_evaluator.py`). A straight
   lookup table over all 13-bit rank masks replaces the per-call scan.
2. **Hero-eval caching in both equity estimators** (pure evaluations,
   bit-identical by construction): river board -> hoisted out of the
   trial loop; turn board -> memoized per river card.

Measured: `evaluate` 11.5 -> 4.2 µs; `estimate_equity` at 1,000 trials
preflop 56.9 -> 34.6 ms, flop 49.0 -> 26.0 ms, turn 42.8 -> 19.3 ms,
river 42.3 -> 18.0 ms. Macro **52.2 s**, then **43.9 s** after the flat
`_best_five`; oracle PASS at every step.

## Result

| stage | macro wall (median of 3) | vs baseline |
|---|---|---|
| baseline (2026-09-02, pre-change) | 89.3 s | 1.00x |
| Phase 1 (deepcopy clones) | 72.2 s | 1.24x |
| Phase 2 (evaluator + caching) | ~41 s | **~2.2x** |

The plan floated ~3x for Phases 1+2 combined; 2.2x is what was measured.
The remaining wall is dominated by the pinned 1,000-trial Monte Carlo
protocol itself (per-trial `random.sample` draws and 7-card showdown
evaluations), which cannot change without changing the RNG consumption
and therefore the corpus bytes. Production extrapolation: 21.9 CPU-hours
-> ~10 CPU-hours for a 50k-decision harvest.

**Every stage reproduced the pre-change corpus byte for byte** (sha256
`79e61dbd4edf410a`), the full suite is green (981 passed / 18 skipped /
342 subtests), and the frozen-instrument tripwire
(`tools.gate_ablation --seeds 16 --scale 1.0 --starting-stack 6000
--equity-trials 80`) still reproduces `p3-gate-2026-08-16`
bit-identically on all three channels — the engine edits did not drift
the frozen instrument.
