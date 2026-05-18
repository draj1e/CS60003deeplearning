from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_flower_loaders(data_root: str | Path, image_size: int, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(data_root)
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
