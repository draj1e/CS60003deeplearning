from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_eurosat
from src.metrics import accuracy, confusion_matrix
from src.model import MLPClassifier
from src.visualize import save_error_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved NumPy MLP on EuroSAT_RGB test split.")
    parser.add_argument("--data-dir", default="EuroSAT_RGB")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    data = load_eurosat(args.data_dir, image_size=args.image_size, seed=args.seed, max_per_class=args.max_per_class)
    model = MLPClassifier.load(args.model)
    pred = model.predict(data.test.x)
    acc = accuracy(data.test.y, pred)
    cm = confusion_matrix(data.test.y, pred, len(data.class_names))

    output_dir = Path(args.output_dir)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    errors = save_error_examples(
        data.test.paths,
        data.test.y,
        pred,
        data.class_names,
        output_dir / "figures" / "error_examples.png",
    )
    report = {
        "test_accuracy": acc,
        "class_names": data.class_names,
        "confusion_matrix": cm.tolist(),
        "error_examples": errors,
    }
    (output_dir / "reports" / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test accuracy: {acc:.4f}")
    print("classes:", data.class_names)
    print("confusion matrix rows=true cols=pred:")
    print(cm)
    print(f"evaluation report: {output_dir / 'reports' / 'evaluation.json'}")


if __name__ == "__main__":
    main()
