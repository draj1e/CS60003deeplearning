#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hw2.common.plotting import plot_history


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制本地 CSV 训练曲线")
    parser.add_argument("--history", required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plot_history(Path(args.history), Path(args.out), args.metrics)
    print(args.out)


if __name__ == "__main__":
    main()
