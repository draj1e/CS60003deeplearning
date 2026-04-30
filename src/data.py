from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


@dataclass
class DatasetSplit:
    x: np.ndarray
    y: np.ndarray
    paths: list[str]


@dataclass
class EuroSATData:
    class_names: list[str]
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    mean: np.ndarray
    std: np.ndarray
    image_shape: tuple[int, int, int]


def scan_dataset(data_dir: str | Path, max_per_class: int | None = None) -> tuple[list[str], list[tuple[Path, int]]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"no class folders found under {root}")

    class_names = [p.name for p in class_dirs]
    items: list[tuple[Path, int]] = []
    for label, class_dir in enumerate(class_dirs):
        paths = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
        if max_per_class is not None:
            paths = paths[:max_per_class]
        if not paths:
            raise ValueError(f"no images found in {class_dir}")
        items.extend((path, label) for path in paths)
    return class_names, items


def stratified_split(
    items: list[tuple[Path, int]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = np.random.default_rng(seed)
    labels = sorted({label for _, label in items})
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    test: list[tuple[Path, int]] = []

    for label in labels:
        group = [item for item in items if item[1] == label]
        indices = rng.permutation(len(group))
        group = [group[i] for i in indices]
        n_total = len(group)
        n_test = max(1, int(round(n_total * test_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        test.extend(group[:n_test])
        val.extend(group[n_test : n_test + n_val])
        train.extend(group[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def load_images(items: list[tuple[Path, int]], image_size: int = 64) -> DatasetSplit:
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    for path, label in items:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (image_size, image_size):
                img = img.resize((image_size, image_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        features.append(arr.reshape(-1))
        labels.append(label)
        paths.append(str(path))
    return DatasetSplit(np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64), paths)


def standardize(split: DatasetSplit, mean: np.ndarray, std: np.ndarray) -> DatasetSplit:
    x = (split.x - mean) / std
    return DatasetSplit(x.astype(np.float32), split.y, split.paths)


def load_eurosat(
    data_dir: str | Path,
    image_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_per_class: int | None = None,
) -> EuroSATData:
    class_names, items = scan_dataset(data_dir, max_per_class=max_per_class)
    train_items, val_items, test_items = stratified_split(items, val_ratio, test_ratio, seed)

    train_raw = load_images(train_items, image_size)
    val_raw = load_images(val_items, image_size)
    test_raw = load_images(test_items, image_size)

    mean = train_raw.x.mean(axis=0, keepdims=True)
    std = train_raw.x.std(axis=0, keepdims=True) + 1e-6

    return EuroSATData(
        class_names=class_names,
        train=standardize(train_raw, mean, std),
        val=standardize(val_raw, mean, std),
        test=standardize(test_raw, mean, std),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        image_shape=(image_size, image_size, 3),
    )


def batch_iter(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator, shuffle: bool = True) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(y))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]
