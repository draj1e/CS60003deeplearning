from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from hw2.common.metrics import accuracy
from hw2.common.utils import append_csv, ensure_dir, get_device, load_yaml, save_json, set_seed
from hw2.classification.data import build_flower_loaders
from hw2.classification.models import build_classifier, parameter_groups


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train=False):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    for images, targets in tqdm(loader, leave=False):
        images, targets = images.to(device), targets.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast(enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, targets)
            if train and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif train:
                loss.backward()
                optimizer.step()
        batch = targets.size(0)
        total_loss += loss.item() * batch
        total_acc += accuracy(logits.detach(), targets) * batch
        total_count += batch
    return total_loss / total_count, total_acc / total_count


def _init_tracker(tracker: str, run_name: str, cfg: dict, project: str = "hw2-task1"):
    if tracker == "swanlab":
        import swanlab
        run = swanlab.init(project=project, experiment_name=run_name, config=cfg, mode=os.environ.get("SWANLAB_MODE", "local"))
        return ("swanlab", run, swanlab)
    if tracker == "wandb":
        import wandb
        run = wandb.init(project=project, name=run_name, config=cfg, mode=os.environ.get("WANDB_MODE", "offline"))
        return ("wandb", run, wandb)
    return ("none", None, None)


def _log_tracker(tracker, payload):
    name, _, module = tracker
    if name == "swanlab":
        module.log(payload)
    elif name == "wandb":
        module.log(payload)


def _finish_tracker(tracker):
    name, run, module = tracker
    if name == "swanlab":
        try:
            module.finish()
        except Exception:
            pass
    elif name == "wandb":
        try:
            run.finish()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classification.yaml")
    parser.add_argument("--variant", choices=["baseline_pretrained", "scratch", "se_pretrained"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default=None)
    parser.add_argument("--run-name", default=None, help="覆盖 summary/history/best 输出文件名后缀；不传则使用 variant")
    parser.add_argument("--tracker", choices=["none", "swanlab", "wandb"], default=None)
    parser.add_argument("--tracker-project", default="hw2-task1")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.variant:
        cfg["variant"] = args.variant
    for arg_name, cfg_name in [
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("image_size", "image_size"),
        ("num_workers", "num_workers"),
        ("output_dir", "output_dir"),
        ("lr_head", "lr_head"),
        ("lr_backbone", "lr_backbone"),
        ("weight_decay", "weight_decay"),
        ("scheduler", "scheduler"),
        ("tracker", "tracker"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            cfg[cfg_name] = value
    cfg.setdefault("scheduler", "none")
    cfg.setdefault("tracker", "none")
    if cfg["variant"] == "scratch":
        cfg["pretrained"] = False
        cfg["attention"] = "none"
    if cfg["variant"] == "se_pretrained":
        cfg["pretrained"] = True
        cfg["attention"] = "se"

    run_name = args.run_name or cfg["variant"]

    set_seed(int(cfg["seed"]))
    output_dir = ensure_dir(cfg["output_dir"])
    train_loader, val_loader, test_loader = build_flower_loaders(cfg["data_root"], cfg["image_size"], cfg["batch_size"], cfg["num_workers"])
    device = get_device()
    model = build_classifier(cfg["arch"], cfg["num_classes"], cfg["pretrained"], cfg["attention"]).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, cfg["lr_backbone"], cfg["lr_head"]), weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")
    scheduler = None
    if cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))

    history_path = output_dir / f"history_{run_name}.csv"
    best_acc = -1.0
    best_path = output_dir / f"best_{run_name}.pt"

    tracker = _init_tracker(cfg["tracker"], run_name, cfg, project=args.tracker_project)
    epoch_start = time.time()
    try:
        for epoch in range(1, int(cfg["epochs"]) + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, None, train=False)
            current_lrs = [group["lr"] for group in optimizer.param_groups]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr_backbone": current_lrs[0] if current_lrs else cfg["lr_backbone"],
                "lr_head": current_lrs[-1] if current_lrs else cfg["lr_head"],
            }
            append_csv(history_path, row, row.keys())
            _log_tracker(tracker, row)
            if scheduler is not None:
                scheduler.step()
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model": model.state_dict(), "config": cfg, "val_acc": best_acc}, best_path)
            print(row)

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, None, train=False)
        summary = {
            "run_name": run_name,
            "variant": cfg["variant"],
            "arch": cfg["arch"],
            "pretrained": cfg["pretrained"],
            "attention": cfg["attention"],
            "epochs": int(cfg["epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "lr_backbone": float(cfg["lr_backbone"]),
            "lr_head": float(cfg["lr_head"]),
            "weight_decay": float(cfg["weight_decay"]),
            "scheduler": cfg["scheduler"],
            "best_val_acc": best_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "duration_sec": time.time() - epoch_start,
            "checkpoint": str(best_path),
        }
        save_json(summary, output_dir / f"summary_{run_name}.json")
        _log_tracker(tracker, {"test_loss": test_loss, "test_acc": test_acc, "best_val_acc": best_acc})
    finally:
        _finish_tracker(tracker)


if __name__ == "__main__":
    main()
