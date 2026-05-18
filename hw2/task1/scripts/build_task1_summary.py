#!/usr/bin/env python
"""汇总 Task 1 所有关键 run（scratch / baseline / baseline_best / SE_v1 / SE_v2）到一张表。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RUNS = [
    ("scratch", ROOT / "outputs/classification/summary_scratch.json", "ResNet-18 from scratch (no pretrain)"),
    ("baseline_pretrained_v1", ROOT / "outputs/classification/summary_baseline_pretrained.json", "Baseline (head 1e-3 / bb 1e-4, no sched)"),
    ("baseline_best", ROOT / "outputs/classification_grid/summary_grid_h3e-3_b1e-4.json", "Grid best (head 3e-3 / bb 1e-4)"),
    ("se_pretrained_v1", ROOT / "outputs/classification/summary_se_pretrained.json", "SE-ResNet18 (head 1e-3 / bb 1e-4, no sched)"),
    ("se_pretrained_v2_cosine", ROOT / "outputs/classification_se_v2/summary_se_pretrained_v2_cosine.json", "SE-ResNet18 (head 3e-3 / bb 1e-4, cosine)"),
]


def main() -> None:
    rows = []
    for tag, path, note in RUNS:
        with path.open() as fh:
            d = json.load(fh)
        rows.append(
            {
                "tag": tag,
                "note": note,
                "best_val_acc": d["best_val_acc"],
                "test_acc": d["test_acc"],
                "test_loss": d["test_loss"],
                "lr_head": d.get("lr_head"),
                "lr_backbone": d.get("lr_backbone"),
                "scheduler": d.get("scheduler", "none"),
                "epochs": d.get("epochs", 30),
            }
        )
    df = pd.DataFrame(rows)
    out = ROOT / "task1/results/task1_summary.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
