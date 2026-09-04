from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from poker_nn_training.data import (
    FEATURE_NAMES,
    LABELS,
    DecisionExample,
    InvalidHandHistory,
    extract_decisions,
    split_for_row,
)
from poker_nn_training.model import TinyPolicy, mask_illegal_logits

DATASET_ID = "takara-ai/poker_hands"
DATASET_REVISION = "6acb5afb6f43082e6a468fde578890d9188be393"
DATASET_COLUMNS = (
    "variant",
    "source_file",
    "source_type",
    "ante_trimming_status",
    "antes",
    "blinds_or_straddles",
    "min_bet",
    "starting_stacks",
    "actions",
)


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float
    macro_f1: float
    per_class_f1: tuple[float, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny next-action policy from streamed PHH hand histories."
    )
    parser.add_argument("--max-hands", type=int, default=12_000)
    parser.add_argument("--max-decisions", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--source-contains", default=None)
    parser.add_argument(
        "--include-unknown-holes",
        action="store_true",
        help="Include card-blind decisions; off by default because most HandHQ cards are ????.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/tiny-policy.pt"))
    return parser.parse_args(argv)


def collect_examples(args: argparse.Namespace) -> dict[str, list[DecisionExample]]:
    stream = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    ).select_columns(DATASET_COLUMNS)

    splits: dict[str, list[DecisionExample]] = {"train": [], "val": [], "test": []}
    hands_seen = 0
    hands_used = 0
    invalid_hands = 0

    for row in stream:
        if row["variant"] != "NT":
            continue
        if hands_seen >= args.max_hands:
            break
        hands_seen += 1

        source_file = str(row.get("source_file") or "")
        if args.source_contains and args.source_contains.lower() not in source_file.lower():
            continue

        try:
            decisions = extract_decisions(
                row,
                require_known_hole_cards=not args.include_unknown_holes,
            )
        except InvalidHandHistory:
            invalid_hands += 1
            continue
        if not decisions:
            continue

        split = split_for_row(row, seed=args.seed)
        room = args.max_decisions - sum(len(values) for values in splits.values())
        splits[split].extend(decisions[:room])
        hands_used += 1
        if room <= len(decisions):
            break

    counts = {name: len(examples) for name, examples in splits.items()}
    print(
        "data:",
        json.dumps(
            {
                "hands_scanned": hands_seen,
                "hands_used": hands_used,
                "invalid_hands": invalid_hands,
                "decisions": counts,
            },
            sort_keys=True,
        ),
    )
    if any(not splits[name] for name in ("train", "val", "test")):
        raise RuntimeError(
            "grouped split produced an empty train/val/test set; increase --max-hands or "
            "broaden --source-contains"
        )
    return splits


def to_tensors(examples: Iterable[DecisionExample]) -> tuple[Tensor, Tensor]:
    rows = list(examples)
    features = torch.tensor([row.features for row in rows], dtype=torch.float32)
    labels = torch.tensor([row.label for row in rows], dtype=torch.long)
    return features, labels


def evaluate(model: TinyPolicy, features: Tensor, labels: Tensor, batch_size: int) -> Metrics:
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    confusion = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            logits = mask_illegal_logits(model(batch_features), batch_features)
            total_loss += float(loss_function(logits, batch_labels))
            predictions = logits.argmax(dim=1)
            for truth, prediction in zip(batch_labels, predictions):
                confusion[int(truth), int(prediction)] += 1

    total = int(confusion.sum())
    accuracy = float(confusion.diag().sum()) / total
    per_class_f1: list[float] = []
    for class_index in range(len(LABELS)):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum()) - true_positive
        false_negative = int(confusion[class_index, :].sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return Metrics(
        loss=total_loss / total,
        accuracy=accuracy,
        macro_f1=sum(per_class_f1) / len(per_class_f1),
        per_class_f1=tuple(per_class_f1),
    )


def train(args: argparse.Namespace) -> Path:
    if args.max_hands < 1 or args.max_decisions < 1 or args.epochs < 1:
        raise ValueError("max-hands, max-decisions, and epochs must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    split_examples = collect_examples(args)
    train_features, train_labels = to_tensors(split_examples["train"])
    val_features, val_labels = to_tensors(split_examples["val"])
    test_features, test_labels = to_tensors(split_examples["test"])

    label_counts = Counter(int(label) for label in train_labels)
    majority_accuracy = max(label_counts.values()) / len(train_labels)
    print(
        "baseline:",
        json.dumps(
            {
                "majority_accuracy": majority_accuracy,
                "train_class_counts": {
                    LABELS[index]: label_counts.get(index, 0) for index in range(len(LABELS))
                },
            },
            sort_keys=True,
        ),
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = TinyPolicy(hidden_size=args.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    best_state: dict[str, Tensor] | None = None
    best_val_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = mask_illegal_logits(model(batch_features), batch_features)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_features, val_labels, args.batch_size)
        print("epoch:", json.dumps({"epoch": epoch, **asdict(val_metrics)}, sort_keys=True))
        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_features, test_labels, args.batch_size)
    print("test:", json.dumps(asdict(test_metrics), sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(FEATURE_NAMES),
            "hidden_size": args.hidden_size,
            "feature_names": FEATURE_NAMES,
            "labels": LABELS,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "seed": args.seed,
            "test_metrics": asdict(test_metrics),
        },
        args.output,
    )
    print(f"saved: {args.output}")
    return args.output


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
