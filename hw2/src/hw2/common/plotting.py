from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def plot_history(csv_path: str | Path, output_path: str | Path, metrics: list[str]) -> None:
    data = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    for metric in metrics:
        if metric in data:
            plt.plot(data["epoch"], data[metric], label=metric)
    plt.xlabel("epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    ensure_dir(Path(output_path).parent)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
