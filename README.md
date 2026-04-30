# HW1：纯 NumPy 三层神经网络 EuroSAT 分类器

本项目按照 `hw1.pdf` 要求，从零实现三层 MLP 分类器，在 `EuroSAT_RGB` 遥感图像数据集上完成十分类训练、超参数搜索、测试评估、曲线可视化、第一层权重可视化和错例分析材料生成。

## 作业要求覆盖

| 要求 | 实现位置 |
| --- | --- |
| 不使用 PyTorch/TensorFlow/JAX | 全部模型与梯度位于 `src/model.py`，仅依赖 NumPy |
| 数据加载与预处理 | `src/data.py` |
| 模型定义 | `src/model.py` |
| 训练循环 | `src/train.py`、`train.py` |
| 测试评估 | `evaluate.py`、`src/metrics.py` |
| 超参数查找 | `search.py` |
| SGD / LR Decay / Cross-Entropy / L2 | `src/optim.py`、`src/model.py`、`src/train.py` |
| 最优权重保存 | `outputs/models/*_best.npz` |
| 混淆矩阵 | `evaluate.py` 输出并保存到 `outputs/reports/evaluation.json` |
| 曲线与权重可视化 | `src/visualize.py` 输出到 `outputs/figures/` |

## 环境依赖

```bash
python -m pip install -r requirements.txt
```

依赖很少：`numpy`、`Pillow`、`matplotlib`。不需要深度学习框架。

## 数据目录

默认数据目录为：

```text
EuroSAT_RGB/
  AnnualCrop/
  Forest/
  ...
```

代码会按类别分层划分为 70% 训练集、15% 验证集、15% 测试集。默认随机种子为 `42`，因此训练和评估会使用同一套划分。

## 快速冒烟测试

```bash
python train.py --max-per-class 20 --epochs 1 --hidden-dims 32,16 --batch-size 16 --run-name smoke
python evaluate.py --model outputs/models/smoke_best.npz --max-per-class 20
```

## 正式训练示例

```bash
python train.py \
  --data-dir EuroSAT_RGB \
  --run-name final_h256_128_lr001_wd001_relu \
  --hidden-dims 256,128 \
  --activation relu \
  --lr 0.01 \
  --lr-decay 0.95 \
  --weight-decay 0.001 \
  --batch-size 128 \
  --epochs 25
```

训练会自动保存：

- 最优模型：`outputs/models/final_h256_128_lr001_wd001_relu_best.npz`
- 训练历史：`outputs/history/final_h256_128_lr001_wd001_relu_history.json`
- Loss/Accuracy 曲线：`outputs/figures/final_h256_128_lr001_wd001_relu_curves.png`
- 第一层权重图：`outputs/figures/final_h256_128_lr001_wd001_relu_w1.png`

## 超参数搜索示例

```bash
python search.py \
  --hidden-grid '128,64;256,128' \
  --lr-grid '0.05,0.01' \
  --weight-decay-grid '0.0001,0.001' \
  --activation-grid 'relu,tanh' \
  --epochs 10
```

搜索结果保存到：

- `outputs/search/search_results.csv`
- `outputs/search/search_results.json`

## 测试评估示例

```bash
python evaluate.py --model outputs/models/final_h256_128_lr001_wd001_relu_best.npz
```

输出包括测试集 Accuracy、混淆矩阵，并保存：

- `outputs/reports/evaluation.json`
- `outputs/figures/error_examples.png`

## 报告生成建议

`outputs/reports/report.md` 是报告草稿模板。完成正式训练和评估后，把以下内容填入报告并转成 PDF：

- GitHub Repo 链接
- 模型权重下载地址（如 Google Drive）
- 最佳超参数和验证集表现
- 测试集 Accuracy 与 Confusion Matrix
- `outputs/figures/*_curves.png`
- `outputs/figures/*_w1.png`
- `outputs/figures/error_examples.png`
- 对第一层权重空间/颜色模式与错例原因的讨论
