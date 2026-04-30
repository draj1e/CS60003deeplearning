from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

from src.data import load_eurosat
from src.train import train_model


def parse_hidden_grid(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(";"):
        a, b = [int(v.strip()) for v in item.split(",")]
        result.append((a, b))
    return result


def parse_float_grid(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search MLP hyperparameters.")
    parser.add_argument("--data-dir", default="EuroSAT_RGB")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--hidden-grid", type=parse_hidden_grid, default=[(128, 64), (256, 128)])
    parser.add_argument("--lr-grid", type=parse_float_grid, default=[0.05, 0.01])
    parser.add_argument("--weight-decay-grid", type=parse_float_grid, default=[1e-4, 1e-3])
    parser.add_argument("--activation-grid", default="relu,tanh")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = load_eurosat(args.data_dir, image_size=args.image_size, seed=args.seed, max_per_class=args.max_per_class)
    output_dir = Path(args.output_dir)
    (output_dir / "search").mkdir(parents=True, exist_ok=True)
    activations = [v.strip() for v in args.activation_grid.split(",") if v.strip()]
    combos = list(itertools.product(args.hidden_grid, args.lr_grid, args.weight_decay_grid, activations))
    if args.limit is not None:
        combos = combos[: args.limit]

    records = []
    for index, (hidden_dims, lr, weight_decay, activation) in enumerate(combos, start=1):
        run_name = f"search_{index:03d}_h{hidden_dims[0]}-{hidden_dims[1]}_lr{lr}_wd{weight_decay}_{activation}"
        _, history, best_path = train_model(
            data=data,
            hidden_dims=hidden_dims,
            activation=activation,
            lr=lr,
            lr_decay=args.lr_decay,
            weight_decay=weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed + index,
            output_dir=output_dir,
            run_name=run_name,
        )
        best_row = max(history, key=lambda row: row["val_acc"])
        records.append({
            "run_name": run_name,
            "hidden_dims": f"{hidden_dims[0]},{hidden_dims[1]}",
            "lr": lr,
            "weight_decay": weight_decay,
            "activation": activation,
            "best_epoch": best_row["epoch"],
            "best_val_acc": best_row["val_acc"],
            "best_model": str(best_path),
        })
        (output_dir / "search" / "search_results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "search" / "search_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"search results: {csv_path}")


if __name__ == "__main__":
    main()
