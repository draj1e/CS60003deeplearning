#!/usr/bin/env bash
set -euo pipefail
python -m hw2.detection.train_yolo "$@"
