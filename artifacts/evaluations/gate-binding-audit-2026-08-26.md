# Gate binding audit — what the three changes would have done live

`.arena-training.jsonl` · policy `candidate-v7-0001c` · gates from `artifacts\candidates\candidate-v7-0001c.approved.manifest.json`

> Read beside the battery number, never instead of it. This prices **how often** each edit changes a verdict and **which hands** it lands on. It is not a counterfactual EV.

## 0. The instrument, before any result

| check | result | verdict |
|---|---|---|
| `effective_stack <= hero_stack` on every record | 0 violations | PASS |
| equal denominators give equal verdicts | 0 differences over 674 records | PASS |
| recorded bets respect the hero-purse cap (validates this tool's arithmetic against the engine that ran) | 0 over, of 2600 sized bets (0 of them within the effective-stack cap) | PASS |
| reveal expense in [0,1], zero on the river | 0 out of range, 0 non-zero rivers | PASS |

Parse accounting: 6913 lines, 3 unparsable, 0 blank, 4737 decision records, 2173 hand results, 2069 hands with a chip delta.

**The manifest does not name `call_gates_on_effective_stack`, `reveal_expense_equity_slope`, `risk_cap_on_effective_stack`**, so the served policy takes the dataclass default. That is the mechanism by which an unmeasured gate change ships under an approved artifact.

## 1. How often each edit changes anything

Across **4333** recorded decisions the two denominators disagree on **3659** (84.44%) — hero covering the table. Median hero purse on those: 5937, median effective stack 1573.

| edit | population | changes the verdict | rate |
|---|---|---|---|
| effective-stack risk cap | 2600 sub-near-nut sized bets | 35 clipped, of which 19 declined outright | 1.35% |
| effective-stack call gates | 363 calls | 12 newly reach a stack gate (11 distinct decisions), of which **at most 8 are actually refused** | 2.2% |
| reveal-expense slope 0.12 | 360 calls with cards still to come | 12 where a stack gate also fires (the slope raises a floor, it never creates a gate) | — |

Chips the cap removes: **18673** total, median 269, largest single clip 3403. Chips in the calls the gate would actually refuse: **4389**, largest single call 2097.

> **Reaching a gate is not being refused by it.** The engine refuses only when `equity < required`; the stack-fraction trigger merely decides whether the floor is consulted at all. The refusal count is a **ceiling**: wildness is not recorded in the journal and w>0 strictly lowers the required equity, so refusals are evaluated at w=0.

## 2. The decay the change was made to fix

`PENDING_EDITS` claims the hero-purse cap went inert as the bankroll grew. Recomputed here from the journal:

> **This is corroboration, not validation, and the definition was chosen after seeing the alternative fail.** Under `_bounds_exposure` (the cap holds hero under the chips actually at risk) the decay is steep. Under the first definition tried (the cap sits below the legal raise maximum) it is **75.22 / 75.46 / 71.37 — flat, and identical for both denominators**, because a 0.455 fraction is below the legal max almost always. `_bounds_exposure` is the better definition and the reason is in its code comment, but a check picked after seeing the other one measure nothing cannot also serve as independent validation. The archived source buckets by *session*, not by these purse edges, so the agreement is directional only.

| hero purse | sub-near-nut sized bets | hero-purse cap binds | effective-stack cap binds |
|---|---|---|---|
| 0-4000 | 670 | 73.88% | 100.0% |
| 4000-10000 | 1703 | 35.29% | 99.88% |
| 10000+ | 227 | 3.08% | 100.0% |

## 3. Which hands the edits land on

> Association, not a counterfactual saving: clipping a bet changes how the hand continues and this journal cannot replay a hand that never happened. The selection is also not random -- these gates fire on the largest prices, and large prices are mechanically where large losses live, so some concentration is expected before any judgement about the gate.

| population | hands | total chips | mean | median | worst | losing |
|---|---|---|---|---|---|---|
| every hand with a recorded result | 1858 | 1348 | 0.7 | 3.0 | -5000 | 494 (26.59%) |
| hands the changed cap touches | 33 | -1193 | -36.2 | 49 | -1574 | 12 (36.36%) |
| hands the changed call gate would REFUSE (ceiling, w=0) | 7 | -7632 | -1090.3 | -1054 | -3768 | 6 (85.71%) |
| hands where the gate is merely REACHED | 10 | -13114 | -1311.4 | -1124.5 | -5000 | 8 (80.0%) |
| hands either edit touches | 38 | -8676 | -228.3 | 37.0 | -3768 | 17 (44.74%) |

