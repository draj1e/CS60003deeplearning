#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 YOLOv8 可直接使用的 Road Vehicle dataset.yaml")
    parser.add_argument("--src", default="data/road_vehicle_raw/trafic_data/data_1.yaml")
    parser.add_argument("--out", default="data/road_vehicle_raw/trafic_data/dataset.yaml")
    args = parser.parse_args()
    src = Path(args.src).resolve()
    root = src.parent
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    output = {
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "nc": data["nc"],
        "names": data["names"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(output, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
