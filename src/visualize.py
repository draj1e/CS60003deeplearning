from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_history(history: list[dict], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val acc", color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_first_layer_weights(w1: np.ndarray, image_shape: tuple[int, int, int], output_path: str | Path, max_units: int = 64) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    h, w, c = image_shape
    units = min(max_units, w1.shape[1])
    cols = int(np.ceil(np.sqrt(units)))
    rows = int(np.ceil(units / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.asarray(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= units:
            continue
        weight_img = w1[:, i].reshape(h, w, c)
        weight_img = weight_img - weight_img.min()
        weight_img = weight_img / (weight_img.max() + 1e-8)
        ax.imshow(weight_img)
        ax.set_title(str(i), fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_error_examples(
    paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    max_examples: int = 12,
) -> list[dict[str, str]]:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wrong = np.where(y_true != y_pred)[0][:max_examples]
    if len(wrong) == 0:
        return []
    cols = min(4, len(wrong))
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.asarray(axes).reshape(-1)
    records: list[dict[str, str]] = []
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, wrong):
        image = Image.open(paths[int(idx)]).convert("RGB")
        true_name = class_names[int(y_true[idx])]
        pred_name = class_names[int(y_pred[idx])]
        ax.imshow(image)
        ax.set_title(f"T:{true_name}\nP:{pred_name}", fontsize=8)
        records.append({"path": paths[int(idx)], "true": true_name, "pred": pred_name})
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return records
