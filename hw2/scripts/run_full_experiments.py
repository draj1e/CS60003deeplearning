#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRESS_PATH = ROOT / "outputs" / "full_run_progress.json"

TASKS = [
    {
        "name": "classification_baseline_pretrained",
        "cmd": [sys.executable, "-m", "hw2.classification.train", "--config", "configs/classification.yaml", "--variant", "baseline_pretrained", "--epochs", "30", "--batch-size", "64", "--num-workers", "8"],
        "outputs": ["outputs/classification/summary_baseline_pretrained.json"],
    },
    {
        "name": "classification_scratch",
        "cmd": [sys.executable, "-m", "hw2.classification.train", "--config", "configs/classification.yaml", "--variant", "scratch", "--epochs", "30", "--batch-size", "64", "--num-workers", "8"],
        "outputs": ["outputs/classification/summary_scratch.json"],
    },
    {
        "name": "classification_se_pretrained",
        "cmd": [sys.executable, "-m", "hw2.classification.train", "--config", "configs/classification.yaml", "--variant", "se_pretrained", "--epochs", "30", "--batch-size", "64", "--num-workers", "8"],
        "outputs": ["outputs/classification/summary_se_pretrained.json"],
    },
    {
        "name": "detection_yolov8",
        "cmd": [sys.executable, "-m", "hw2.detection.train_yolo", "--config", "configs/detection.yaml"],
        "outputs": ["outputs/detection/train/weights/best.pt"],
    },
    {
        "name": "segmentation_ce",
        "cmd": [sys.executable, "-m", "hw2.segmentation.train", "--config", "configs/segmentation.yaml", "--loss", "ce"],
        "outputs": ["outputs/segmentation/summary_ce.json"],
    },
    {
        "name": "segmentation_dice",
        "cmd": [sys.executable, "-m", "hw2.segmentation.train", "--config", "configs/segmentation.yaml", "--loss", "dice"],
        "outputs": ["outputs/segmentation/summary_dice.json"],
    },
    {
        "name": "segmentation_ce_dice",
        "cmd": [sys.executable, "-m", "hw2.segmentation.train", "--config", "configs/segmentation.yaml", "--loss", "ce_dice"],
        "outputs": ["outputs/segmentation/summary_ce_dice.json"],
    },
    {
        "name": "tracking_demo",
        "cmd": [sys.executable, "-m", "hw2.detection.track_count", "--config", "configs/detection.yaml", "--weights", "outputs/detection/train/weights/best.pt", "--video", "data/videos/road_vehicle_demo.mp4"],
        "outputs": ["outputs/detection/tracking/count_events.json", "outputs/detection/tracking/tracked_counted.mp4"],
    },
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return {"started_at": now(), "tasks": {}}


def save_progress(progress: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = now()
    PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def outputs_exist(task: dict) -> bool:
    return all((ROOT / output).exists() for output in task["outputs"])


def main() -> None:
    parser = argparse.ArgumentParser(description="断点续跑 HW2 完整实验")
    parser.add_argument("--only", nargs="*", default=None, help="只运行指定任务名")
    parser.add_argument("--force", action="store_true", help="即使输出存在也重新运行")
    args = parser.parse_args()
    progress = load_progress()
    selected = set(args.only) if args.only else None
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    for task in TASKS:
        name = task["name"]
        if selected and name not in selected:
            continue
        status = progress["tasks"].get(name, {})
        if not args.force and status.get("status") == "completed" and outputs_exist(task):
            print(f"[skip] {name}")
            continue
        if not args.force and outputs_exist(task):
            progress["tasks"][name] = {"status": "completed", "skipped_existing": True, "updated_at": now(), "outputs": task["outputs"]}
            save_progress(progress)
            print(f"[mark-completed] {name}")
            continue
        print(f"[run] {name}: {' '.join(task['cmd'])}")
        progress["tasks"][name] = {"status": "running", "started_at": now(), "cmd": task["cmd"], "outputs": task["outputs"]}
        save_progress(progress)
        result = subprocess.run(task["cmd"], cwd=ROOT, env=env)
        if result.returncode != 0:
            progress["tasks"][name]["status"] = "failed"
            progress["tasks"][name]["returncode"] = result.returncode
            progress["tasks"][name]["failed_at"] = now()
            save_progress(progress)
            raise SystemExit(result.returncode)
        progress["tasks"][name]["status"] = "completed"
        progress["tasks"][name]["completed_at"] = now()
        save_progress(progress)
    print(f"progress: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()
