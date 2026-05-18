#!/usr/bin/env python
"""汇总 outputs/classification_grid/summary_*.json 为 grid_summary.csv 与热力图。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", default="outputs/classification_grid")
    parser.add_argument("--out-csv", default="task1/results/grid_summary.csv")
    parser.add_argument("--out-fig", default="task1/figures/grid_heatmap.png")
    args = parser.parse_args()
    grid_dir = Path(args.grid_dir)
    rows = []
    for path in sorted(grid_dir.glob("summary_grid_*.json")):
        with path.open() as fh:
            data = json.load(fh)
        rows.append(
            {
                "run": data["run_name"],
                "lr_head": data["lr_head"],
                "lr_backbone": data["lr_backbone"],
                "epochs": data["epochs"],
                "best_val_acc": data["best_val_acc"],
                "test_acc": data["test_acc"],
                "test_loss": data["test_loss"],
                "duration_sec": data.get("duration_sec"),
            }
        )
    if not rows:
        raise SystemExit(f"no summaries under {grid_dir}")
    df = pd.DataFrame(rows).sort_values(["lr_head", "lr_backbone"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    pivot_val = df.pivot(index="lr_head", columns="lr_backbone", values="best_val_acc")
    pivot_test = df.pivot(index="lr_head", columns="lr_backbone", values="test_acc")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, pivot, title in [
        (axes[0], pivot_val, "best val acc"),
        (axes[1], pivot_test, "test acc"),
    ]:
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [f"{c:.0e}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{r:.0e}" for r in pivot.index])
        ax.set_xlabel("lr_backbone")
        ax.set_ylabel("lr_head")
        ax.set_title(title)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < pivot.values.max() * 0.6 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Task 1 hyperparameter grid (ResNet-18 baseline_pretrained, 30 epoch)")
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    print(df.to_string(index=False))
    print(f"\nsaved: {args.out_csv}\nsaved: {args.out_fig}")
    print("\nbest:")
    print(df.sort_values("best_val_acc", ascending=False).head(3).to_string(index=False))


if __name__ == "__main__":
    main()
