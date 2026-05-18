#!/usr/bin/env python
"""画 swanlab 风格多-run 对比图，数据源：task1/results/history_*.csv 与 outputs/classification_grid/history_*.csv。
生成 3 张：
  1. fig_main_compare.png  — scratch vs baseline_v1 vs baseline_best vs SE_v2 的 train/val loss + val acc
  2. fig_grid_curves.png    — 9 组 grid runs 的 val acc 曲线（按 lr_head 分面板，颜色编码 lr_backbone）
  3. fig_se_vs_baseline.png — baseline_best vs SE_v1 vs SE_v2 的 train/val loss + val acc
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "task1/figures/swanlab"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)

PALETTE = ["#3a7ca5", "#d96c06", "#2a9d8f", "#9b2226", "#6f4e7c", "#7f7f7f", "#bc5090", "#003f5c", "#ffa600"]


def load(name: str, history_dir: Path = ROOT / "task1/results") -> pd.DataFrame:
    return pd.read_csv(history_dir / f"history_{name}.csv")


def panel(ax, dfs: dict[str, pd.DataFrame], ycol: str, title: str, ylabel: str) -> None:
    for (label, df), color in zip(dfs.items(), PALETTE):
        ax.plot(df["epoch"], df[ycol], color=color, lw=1.8, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)


def fig_main_compare() -> None:
    runs = {
        "scratch (no pretrain)": load("scratch"),
        "baseline v1 (h=1e-3, b=1e-4)": load("baseline_pretrained"),
        "baseline best (h=3e-3, b=1e-4)": load("baseline_best"),
        "SE-ResNet18 v2 (cosine)": load("se_pretrained_v2_cosine"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panel(axes[0], runs, "train_loss", "train loss", "loss")
    panel(axes[1], runs, "val_loss", "val loss", "loss")
    panel(axes[2], runs, "val_acc", "val accuracy", "acc")
    axes[2].set_ylim(0, 1)
    fig.suptitle("Task 1 main comparison — 102 Flowers, ResNet-18, 30 epoch", fontsize=12)
    fig.tight_layout()
    out = OUT / "fig_main_compare.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def fig_grid_curves() -> None:
    grid_dir = ROOT / "outputs/classification_grid"
    lr_heads = ["5e-4", "1e-3", "3e-3"]
    lr_bbs = ["3e-5", "1e-4", "3e-4"]
    colors = {"3e-5": "#3a7ca5", "1e-4": "#d96c06", "3e-4": "#2a9d8f"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, h in zip(axes, lr_heads):
        for b in lr_bbs:
            df = pd.read_csv(grid_dir / f"history_grid_h{h}_b{b}.csv")
            ax.plot(df["epoch"], df["val_acc"], color=colors[b], lw=1.6, label=f"bb={b}")
        ax.set_title(f"lr_head = {h}")
        ax.set_xlabel("epoch")
        ax.set_ylim(0.55, 0.95)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    axes[0].set_ylabel("val accuracy")
    fig.suptitle("Task 1 hyperparameter grid — val accuracy curves (9 runs)", fontsize=12)
    fig.tight_layout()
    out = OUT / "fig_grid_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def fig_se_vs_baseline() -> None:
    runs = {
        "baseline best": load("baseline_best"),
        "SE-ResNet18 v1 (no sched)": load("se_pretrained"),
        "SE-ResNet18 v2 (cosine)": load("se_pretrained_v2_cosine"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panel(axes[0], runs, "train_loss", "train loss", "loss")
    panel(axes[1], runs, "val_loss", "val loss", "loss")
    panel(axes[2], runs, "val_acc", "val accuracy", "acc")
    axes[2].set_ylim(0.55, 0.95)
    fig.suptitle("Task 1 attention ablation — baseline vs SE-ResNet18", fontsize=12)
    fig.tight_layout()
    out = OUT / "fig_se_vs_baseline.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    fig_main_compare()
    fig_grid_curves()
    fig_se_vs_baseline()
