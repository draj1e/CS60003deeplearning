from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import EuroSATData, batch_iter
from .model import MLPClassifier
from .optim import SGD, decayed_lr
from .visualize import plot_first_layer_weights, plot_history


def evaluate_loss(model: MLPClassifier, x: np.ndarray, y: np.ndarray, weight_decay: float, batch_size: int = 512) -> float:
    losses: list[float] = []
    weights: list[int] = []
    for start in range(0, len(y), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        loss, _ = model.loss_and_grads(xb, yb, weight_decay)
        losses.append(loss)
        weights.append(len(yb))
    return float(np.average(losses, weights=weights))


def train_model(
    data: EuroSATData,
    hidden_dims: tuple[int, int] = (256, 128),
    activation: str = "relu",
    lr: float = 0.05,
    lr_decay: float = 0.95,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    batch_size: int = 128,
    seed: int = 42,
    output_dir: str | Path = "outputs",
    run_name: str = "default",
) -> tuple[MLPClassifier, list[dict], Path]:
    output_dir = Path(output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "history").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    model = MLPClassifier(
        input_dim=data.train.x.shape[1],
        hidden_dims=hidden_dims,
        output_dim=len(data.class_names),
        activation=activation,
        seed=seed,
    )
    optimizer = SGD(model.params, lr)
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    best_acc = -1.0
    best_path = output_dir / "models" / f"{run_name}_best.npz"
    progress_path = output_dir / "history" / f"{run_name}_progress.json"
    history_path = output_dir / "history" / f"{run_name}_history.json"

    for epoch in range(1, epochs + 1):
        optimizer.lr = decayed_lr(lr, lr_decay, epoch)
        batch_losses: list[float] = []
        for xb, yb in batch_iter(data.train.x, data.train.y, batch_size, rng, shuffle=True):
            loss, grads = model.loss_and_grads(xb, yb, weight_decay)
            optimizer.step(grads)
            batch_losses.append(loss)

        train_loss = float(np.mean(batch_losses))
        val_loss = evaluate_loss(model, data.val.x, data.val.y, weight_decay)
        train_acc = model.accuracy(data.train.x, data.train.y)
        val_acc = model.accuracy(data.val.x, data.val.y)
        row = {
            "epoch": epoch,
            "lr": optimizer.lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_acc > best_acc:
            best_acc = val_acc
            model.save(best_path, metadata={"epoch": epoch, "val_acc": val_acc, "run_name": run_name})

        progress_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "last_finished_epoch": epoch,
                    "best_val_acc": best_acc,
                    "best_model": str(best_path),
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_history(history, output_dir / "figures" / f"{run_name}_curves.png")
    plot_first_layer_weights(model.params["W1"], data.image_shape, output_dir / "figures" / f"{run_name}_w1.png")
    return model, history, best_path
