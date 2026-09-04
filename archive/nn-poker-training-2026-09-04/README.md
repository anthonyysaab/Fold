# Poker NN Training

A deliberately small, CPU-friendly behavior-cloning trainer for no-limit Texas Hold'em hand
histories. This project is independent from both the legacy `poker-agent` repository and the
dev.fun Playground runtime.

The initial data source is
[`takara-ai/poker_hands`](https://huggingface.co/datasets/takara-ai/poker_hands). It contains raw
Poker Hand History (PHH) action sequences rather than supplied labels, so the trainer replays each
hand and derives three next-action targets:

- `fold`
- `check_call`
- `aggress`

The network is one 64-unit hidden layer over 125 order-invariant card and state features. It is a
warm-start policy, not a complete poker strategy and not a profitability claim.

## Setup

```powershell
cd C:\Users\user\poker-nn-training
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Train

```powershell
poker-nn-train --max-hands 12000 --epochs 5
```

The command streams a pinned dataset revision, uses PokerKit for strict chronological replay, keeps
entire source groups in one train/validation/test split, and writes
`artifacts/tiny-policy.pt`. Most HandHQ hands hide every player's cards as `????`; those decisions
are skipped by default. Use `--include-unknown-holes` only for an intentionally card-blind
population baseline.

The current local checkpoint was trained from 79,832 train decisions, 3,802 validation decisions,
and 7,810 grouped test decisions. It reached 74.6% test accuracy and 0.651 macro-F1; the aggressive
class remains the weakest component.

## Consumer boundary

Runtime projects consume the checkpoint and its embedded metadata (`input_size`, `hidden_size`,
`feature_names`, and `labels`). They should not import code from the old `poker-agent` repository.

## Test

```powershell
pytest
ruff check .
```
