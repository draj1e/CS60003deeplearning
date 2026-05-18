from __future__ import annotations

import argparse

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from hw2.common.metrics import mean_iou_from_logits
from hw2.common.utils import append_csv, ensure_dir, get_device, load_yaml, save_json, set_seed
from hw2.segmentation.data import build_segmentation_loaders
from hw2.segmentation.losses import build_loss
from hw2.segmentation.model import UNet


def run_epoch(model, loader, criterion, optimizer, device, num_classes, scaler=None, train=False):
    model.train(train)
    total_loss = 0.0
    total_miou = 0.0
    total_count = 0
    for images, masks in tqdm(loader, leave=False):
        images, masks = images.to(device), masks.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast(enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, masks)
            if train and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif train:
                loss.backward()
                optimizer.step()
        batch = images.size(0)
        total_loss += loss.item() * batch
        total_miou += mean_iou_from_logits(logits.detach(), masks, num_classes) * batch
        total_count += batch
    return total_loss / total_count, total_miou / total_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--loss", choices=["ce", "dice", "ce_dice"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.loss:
        cfg["loss"] = args.loss
    for arg_name, cfg_name in [
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("image_size", "image_size"),
        ("num_workers", "num_workers"),
        ("output_dir", "output_dir"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            cfg[cfg_name] = value
    set_seed(int(cfg["seed"]))
    output_dir = ensure_dir(cfg["output_dir"])
    train_loader, val_loader = build_segmentation_loaders(cfg["data_root"], cfg["image_size"], cfg["batch_size"], cfg["num_workers"], cfg["seed"])
    device = get_device()
    model = UNet(num_classes=cfg["num_classes"]).to(device)
    criterion = build_loss(cfg["loss"], cfg["num_classes"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")
    history_path = output_dir / f"history_{cfg['loss']}.csv"
    best_miou = -1.0
    best_path = output_dir / f"best_unet_{cfg['loss']}.pt"
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss, train_miou = run_epoch(model, train_loader, criterion, optimizer, device, cfg["num_classes"], scaler, train=True)
        val_loss, val_miou = run_epoch(model, val_loader, criterion, None, device, cfg["num_classes"], None, train=False)
        row = {"epoch": epoch, "train_loss": train_loss, "train_miou": train_miou, "val_loss": val_loss, "val_miou": val_miou}
        append_csv(history_path, row, row.keys())
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({"model": model.state_dict(), "config": cfg, "val_miou": best_miou}, best_path)
        print(row)
    save_json({"best_val_miou": best_miou, "checkpoint": str(best_path)}, output_dir / f"summary_{cfg['loss']}.json")


if __name__ == "__main__":
    main()
