# Artifacts folder

This folder contains trained data read by policy code.

- `README.md` is this folder map.
- `tiny-policy-pure.json` is the legacy 125-input, 64-hidden-unit, 3-output network. The live policy loads and validates it, but this export has no `table_sizes` metadata, so it does not currently choose live actions.
- `candidates/` holds multi-head candidate artifacts: a validated manifest plus a checksummed JSON weights file per version, written by the offline trainer and readable by `learned_policy.py`. V6 keeps the physical three-output action-head shape but records `counterfactual_value` semantics in the manifest.
- `training-runs/` holds immutable launch recipes and logs. `candidate-v2-0016.ps1` is the prepared 48,000-hand, four-rollout recipe; it does nothing unless called with `-DryRun` or `-Start`.
- `approved.json`, when present, is the atomic pointer to the approved live artifact, written only by `tools/promote_candidate.py` after the evaluation gate. `run_agent.py --learned` follows it and refreshes between hands. It is absent while no learned candidate has beaten the incumbent.
