#!/usr/bin/env bash
# 从 task1/results/grid_summary.csv 取最优 lr_head/lr_backbone，再用 cosine 重训 SE-ResNet18。
# 用法: bash task1/scripts/run_se_retrain.sh [GPU_ID]
set -euo pipefail
GPU=${1:-1}
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES="$GPU"
export SWANLAB_MODE=local
mkdir -p outputs/classification_se_v2

PY=/mnt/data/zjj/.conda/envs/zl2/bin/python
read LR_H LR_B <<< "$($PY - <<'EOF'
import pandas as pd
df = pd.read_csv('task1/results/grid_summary.csv')
top = df.sort_values('best_val_acc', ascending=False).iloc[0]
print(top['lr_head'], top['lr_backbone'])
EOF
)"
echo "[SE retrain] using best grid lr_head=$LR_H lr_backbone=$LR_B + cosine"

$PY -m hw2.classification.train \
  --config task1/configs/classification.yaml \
  --variant se_pretrained \
  --epochs 30 \
  --lr-head "$LR_H" --lr-backbone "$LR_B" \
  --scheduler cosine \
  --output-dir outputs/classification_se_v2 \
  --run-name se_pretrained_v2_cosine \
  --tracker swanlab \
  --tracker-project hw2-task1-se \
  2>&1 | tee task1/logs/se_pretrained_v2_cosine.log

echo "[done] $(cat outputs/classification_se_v2/summary_se_pretrained_v2_cosine.json)"
