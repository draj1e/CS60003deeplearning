#!/usr/bin/env bash
# Task 1 超参网格 (lr_head × lr_backbone)，串行执行以避免显存竞争。
# 用法: bash task1/scripts/run_grid.sh [GPU_ID]
set -euo pipefail
GPU=${1:-1}
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES="$GPU"
export SWANLAB_MODE=local
mkdir -p task1/results task1/logs outputs/classification_grid

PY=/mnt/data/zjj/.conda/envs/zl2/bin/python
LR_HEAD=(5e-4 1e-3 3e-3)
LR_BB=(3e-5 1e-4 3e-4)
EPOCHS=30

for h in "${LR_HEAD[@]}"; do
  for b in "${LR_BB[@]}"; do
    NAME="grid_h${h}_b${b}"
    SUMMARY="outputs/classification_grid/summary_${NAME}.json"
    if [ -f "$SUMMARY" ]; then
      echo "[skip] $NAME already done"
      continue
    fi
    echo "[run ] $NAME"
    $PY -m hw2.classification.train \
      --config task1/configs/classification.yaml \
      --variant baseline_pretrained \
      --epochs $EPOCHS \
      --lr-head $h --lr-backbone $b \
      --scheduler none \
      --output-dir outputs/classification_grid \
      --run-name "$NAME" \
      --tracker swanlab \
      --tracker-project hw2-task1-grid \
      > "task1/logs/${NAME}.log" 2>&1
    METRICS=$($PY -c "import json; d=json.load(open('$SUMMARY')); print(f\"val={d['best_val_acc']:.4f} test={d['test_acc']:.4f} dur={d['duration_sec']:.0f}s\")")
    echo "[done] $NAME -> $METRICS"
  done
done
echo "all grid runs finished"
