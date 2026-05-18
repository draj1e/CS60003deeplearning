#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def classification_summary() -> pd.DataFrame:
    rows = []
    for variant in ["baseline_pretrained", "scratch", "se_pretrained"]:
        data = load_json(f"outputs/classification/summary_{variant}.json")
        rows.append({
            "variant": variant,
            "best_val_acc": data["best_val_acc"],
            "test_acc": data["test_acc"],
            "test_loss": data["test_loss"],
            "checkpoint": data["checkpoint"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "classification_summary.csv", index=False)
    return frame


def segmentation_summary() -> pd.DataFrame:
    rows = []
    for loss in ["ce", "dice", "ce_dice"]:
        data = load_json(f"outputs/segmentation/summary_{loss}.json")
        rows.append({"loss": loss, "best_val_miou": data["best_val_miou"], "checkpoint": data["checkpoint"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "segmentation_summary.csv", index=False)
    return frame


def detection_summary() -> pd.DataFrame:
    results = pd.read_csv(ROOT / "outputs/detection/train/results.csv")
    results.columns = [column.strip() for column in results.columns]
    best_idx = results["metrics/mAP50(B)"].idxmax()
    best = results.loc[best_idx]
    row = {
        "best_epoch_by_mAP50": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "mAP50_95": float(best["metrics/mAP50-95(B)"]),
        "checkpoint": "outputs/detection/train/weights/best.pt",
    }
    frame = pd.DataFrame([row])
    frame.to_csv(TABLE_DIR / "detection_summary.csv", index=False)
    return frame


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    cls = classification_summary()
    seg = segmentation_summary()
    det = detection_summary()
    tracking = load_json("outputs/detection/tracking/count_events.json")
    md = []
    md.append("# HW2 实验结果汇总\n")
    md.append("## 分类\n")
    md.append(frame_to_markdown(cls))
    md.append("\n## 检测\n")
    md.append(frame_to_markdown(det))
    md.append("\n## 分割\n")
    md.append(frame_to_markdown(seg))
    md.append("\n## Tracking / 越线计数\n")
    md.append(f"- count: {tracking['count']}")
    md.append(f"- video: {tracking['output_video']}")
    (TABLE_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(TABLE_DIR / "summary.md")


if __name__ == "__main__":
    main()
