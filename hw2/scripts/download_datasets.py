#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from torchvision.datasets import Flowers102

DATASETS = {
    "flowers102": {
        "target": "flowers102",
        "kind": "torchvision",
        "source": "Oxford VGG 102 Category Flower Dataset",
    },
    "stanford_background": {
        "target": "stanford_background_raw",
        "kind": "kagglehub",
        "handle": "balraj98/stanford-background-dataset",
    },
    "road_vehicle": {
        "target": "road_vehicle_raw",
        "kind": "kagglehub",
        "handle": "ashfakyeafi/road-vehicle-images-dataset",
    },
}


def copytree_merge(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copytree_merge(item, target)
        elif not target.exists():
            shutil.copy2(item, target)


def download_flowers(root: Path) -> Path:
    target = root / DATASETS["flowers102"]["target"]
    for split in ("train", "val", "test"):
        Flowers102(root=target, split=split, download=True)
    return target


def download_kaggle(handle: str, root: Path, target_name: str, copy_to_data: bool) -> Path:
    import kagglehub

    cache_path = Path(kagglehub.dataset_download(handle))
    target = root / target_name
    if copy_to_data:
        copytree_merge(cache_path, target)
        return target
    manifest = root / f"{target_name}.source.txt"
    manifest.write_text(f"kaggle_handle={handle}\ncache_path={cache_path}\n", encoding="utf-8")
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 HW2 所需公开数据集")
    parser.add_argument("--root", default="data", help="数据保存根目录")
    parser.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")
    parser.add_argument("--copy-kaggle", action="store_true", help="将 Kaggle 缓存复制到 data/；默认只记录缓存路径，避免重复占空间")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        spec = DATASETS[name]
        if spec["kind"] == "torchvision":
            path = download_flowers(root)
        else:
            path = download_kaggle(spec["handle"], root, spec["target"], args.copy_kaggle)
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
