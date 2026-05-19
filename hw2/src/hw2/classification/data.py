from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_flower_loaders(data_root: str | Path, image_size: int, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Flowers102 train/val/test loaders.

    增广策略与 torchvision ImageNet 微调教程对齐：
      train: RandomResizedCrop + RandomHorizontalFlip + ImageNet mean/std
      val / test: Resize(256) -> CenterCrop(224) + ImageNet mean/std
    image_size 仅控制裁剪/中心裁剪边长（256 是固定的短边 resize 尺寸）。
    """
    root = Path(data_root)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize_short = max(int(image_size * 256 / 224), image_size + 32)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(resize_short),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = datasets.Flowers102(root=root, split="train", download=True, transform=train_tf)
    val = datasets.Flowers102(root=root, split="val", download=True, transform=eval_tf)
    test = datasets.Flowers102(root=root, split="test", download=True, transform=eval_tf)
    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train, shuffle=True, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
    )
