> **ARCHIVED 2026-09-02 — SUPERSEDED. NOT STATE, NOT PLAN, NOT RULES.**
> Do not use this file to learn what the system is, what its state is, or what to do.
> The live manual is `CLAUDE.md` → `.handoff/CONTEXT.md`. Open this file only for a
> specific fact that a live document cites here by path. Never edit it, never restore it.

# Phase B corpora — counterfactual value labels vs the repaired P3 opponent

Written 2026-08-17 by the session that built `tools/build_phase_b_corpus.py`.

> **2026-09-02**: this is the **v8** corpus format (corpus schema 1, 413 inputs).
> The served v9 line uses corpus schema 2 with the 414-input vector, written and
> validated by `tools/build_phase_b_corpus_v9.py` under `artifacts/phase_b_v9/`.
This directory holds **v8-native Phase B corpora** (V8_DESIGN §5 Phase B):
counterfactual branch values harvested with a v8 candidate acting as hero
against lineups that include the fitted, price-clamped
`StrengthAwareAgent` (the repaired P3 opponent, `artifacts/p3/p3-fit.json`).

## Why not `training_telemetry.save_training_corpus`

That writer hard-pins `feature_schema_version` to the v7 learning contract
(2, 142 features) in its header, and its loader refuses anything else
(`training_telemetry.py:675` / `:737`). Phase B rows carry the schema-3
vector (413 floats) plus per-decision structure `TrainingExample` cannot
carry (shared features per decision group, E6 branch sizing context, the
absorption map, per-decision P3 resample stats). So Phase B uses its own
documented format here rather than lying in a v7 header.

## Format — `<name>.phase-b.jsonl.gz` (corpus schema 1)

gzip JSONL, `mtime` pinned to 0 (byte-identical reruns produce
byte-identical files). Line 1 is a header object:

```json
{"kind": "phase-b-corpus", "corpus_schema_version": 1,
 "feature_schema_version": 3, "input_size": 413,
 "branch_labels": ["fold", "check_call", "aggress_small", "aggress_large"]}
```

Every following line is **one decision** (not one branch):

- `decision_id` — `sim-<seed>-<hand>:<actor>:<ordinal>`, unique.
- `street`, `big_blind`, `purse_bb`, `policy_version`, `harvest_leg`,
  `table_id`, `rollouts`.
- `inclusion_count` — how many eligible decisions this actor had in the
  hand (one was selected, per the v7 rule kept by V8_DESIGN §5).
- `context` — pot / to_call / contribution / effective_stack / purse /
  legal bet-raise range at the branch point.
- `features` — the 413-float schema-3 vector, shared by every branch of
  the decision.
- `branches` — the emitted E6 branch set (executable + distinct only):
  per branch `reward_bb` (centered within the decision; the per-decision
  sum is ~0 by construction), raw `outcome_bb`, `risk_fraction`, the
  `executed` (action, amount) pair, and for aggress branches the E6
  `pot_fraction`, `e6_target`, `e6_to_amount`.
- `branch_absorption` — candidate label -> surviving emitted label
  (an unreachable/duplicate target collapses into the branch that
  actually executes; survivors map to themselves).
- `p3` — per-decision conditional-resample counters (seats_resampled,
  tries, accepted, fallbacks, swaps_applied).

Reference reader/validator: `load_phase_b_corpus` /
`validate_phase_b_rows` in `tools/build_phase_b_corpus.py`, or
`python -m tools.build_phase_b_corpus --validate <corpus>`.

## The chance-salt rule (V8_DESIGN §5, for a card-aware opponent)

Hero cards + revealed board fixed; future board resampled per rollout;
card-blind opponents' holes resampled freely; **the P3 opponent's holes
also resample per rollout — never frozen** — by rejection sampling from
the unseen deck against the seat's own faced-price prefix decisions
(joint fitted continue/fold probability >= `--p3-accept-threshold`,
default 0.35, ablatable), with a counted uniform fallback after
`--p3-resample-tries` draws. Fallback rates are reported per leg and in
the summary sidecar.

## Sidecar

`<name>.phase-b.summary.json` — the harvest plan, per-leg diagnostics
(hands, rows, hero bb/100, emitted-branch-size histogram = the E6
collision measurement, P3 resample totals and fallback rate, wall time),
and the validator's aggregate report.

Tests: `tests/test_build_phase_b_corpus.py` (arranged-replay parity vs
the stock chance salt is the load-bearing validation).

Money safety: these corpora are offline simulation products. Nothing
here deploys, promotes, or touches `artifacts/approved.json`.
