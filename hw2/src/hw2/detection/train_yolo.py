from __future__ import annotations

import argparse

from ultralytics import YOLO

from pathlib import Path

from hw2.common.utils import ensure_dir, load_yaml, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detection.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    output_dir = ensure_dir(cfg["output_dir"]).resolve()
    model = YOLO(cfg["base_model"])
    model.train(
        data=cfg["yolo_dataset_yaml"],
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["image_size"]),
        batch=int(cfg["batch_size"]),
        project=str(output_dir),
        name="train",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
