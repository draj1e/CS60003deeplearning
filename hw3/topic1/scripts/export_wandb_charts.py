#!/usr/bin/env python3
"""Log real training metrics to a local offline WandB run and export charts."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
WANDB_DIR = ROOT / "outputs" / "wandb"
B_METRICS = ROOT / "outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/csv_logs/version_0/metrics.csv"
C_METRICS = ROOT / "outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/csv_logs/version_0/metrics.csv"


def read_metric(path: Path, metric: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("step") or not row.get(metric):
                continue
            try:
                steps.append(int(float(row["step"])))
                values.append(float(row[metric]))
            except ValueError:
                continue
    return steps, values


def save_curve(path: Path, series: list[tuple[str, list[int], list[float]]], title: str) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for label, steps, values in series:
        if steps:
            plt.plot(steps, values, linewidth=1.0, label=label)
    plt.xlabel("Step")
    plt.ylabel("Logged value")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    WANDB_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = str(WANDB_DIR)

    b_steps, b_sds = read_metric(B_METRICS, "train/loss_sds")
    b_gn_steps, b_grad = read_metric(B_METRICS, "train/grad_norm")
    c_steps, c_loss = read_metric(C_METRICS, "train/loss")
    c_zero_steps, c_zero = read_metric(C_METRICS, "train/loss_zero123")

    run = wandb.init(
        project="hw3-topic1-real",
        name="real-training-curves",
        dir=str(WANDB_DIR),
        config={
            "object_a": "COLMAP + 3DGS, 2000 iterations",
            "object_b": "threestudio SDS, 1000 steps",
            "object_c": "Zero123 XL, 300 steps",
            "background": "T&T+DeepBlending + 3DGS, 2000 iterations",
        },
    )

    values_by_step: dict[int, dict[str, float]] = {}
    for steps, values, key in [
        (b_steps, b_sds, "object_b/sds_loss"),
        (b_gn_steps, b_grad, "object_b/grad_norm"),
        (c_steps, c_loss, "object_c/total_loss"),
        (c_zero_steps, c_zero, "object_c/zero123_loss"),
    ]:
        for step, value in zip(steps, values):
            values_by_step.setdefault(step, {})[key] = value

    for step in sorted(values_by_step):
        wandb.log(values_by_step[step], step=step)

    save_curve(
        REPORTS / "wandb_loss_curves.png",
        [
            ("B SDS loss", b_steps, b_sds),
            ("C Zero123 total loss", c_steps, c_loss),
            ("C Zero123 guidance loss", c_zero_steps, c_zero),
        ],
        "WandB Offline Exported Loss Curves",
    )
    save_curve(
        REPORTS / "wandb_validation_metrics.png",
        [
            ("B grad norm", b_gn_steps, b_grad),
            ("C Zero123 loss", c_zero_steps, c_zero),
        ],
        "WandB Offline Exported Validation/Diagnostic Metrics",
    )
    wandb.log(
        {
            "charts/loss_curves": wandb.Image(str(REPORTS / "wandb_loss_curves.png")),
            "charts/validation_metrics": wandb.Image(str(REPORTS / "wandb_validation_metrics.png")),
        }
    )
    run.finish()
    print("reports/wandb_loss_curves.png")
    print("reports/wandb_validation_metrics.png")


if __name__ == "__main__":
    main()
