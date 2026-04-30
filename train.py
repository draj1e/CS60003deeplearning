from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_eurosat
from src.train import train_model


def parse_hidden(value: str) -> tuple[int, int]:
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("hidden dims must look like 256,128")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pure NumPy 3-layer MLP on EuroSAT_RGB.")
    parser.add_argument("--data-dir", default="EuroSAT_RGB")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default="hw1")
    parser.add_argument("--hidden-dims", type=parse_hidden, default=(256, 128))
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    data = load_eurosat(
        args.data_dir,
        image_size=args.image_size,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    _, history, best_path = train_model(
        data=data,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        lr=args.lr,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
    )
    print(f"best model: {best_path}")
    print(f"best val acc: {max(row['val_acc'] for row in history):.4f}")


if __name__ == "__main__":
    main()
