# 任务 1：在 102 Category Flower Dataset 上微调 ImageNet 预训练 CNN

> **入口指南**
> - 📄 [`REPORT.md`](REPORT.md) — Task 1 实验报告（提交版；满足题目要求 (1)-(4) 与所有提交项）
> - 📘 [`EXPLAINED.md`](EXPLAINED.md) — 详细技术讲解（每个数字怎么解读、为什么这样设计、负结论分析）
> - 📁 本 README — 目录与复现说明

## 目录结构

```
task1/
├── README.md                       # 目录索引（本文件）
├── REPORT.md                       # 提交版实验报告
├── EXPLAINED.md                    # 详细技术讲解
├── code -> ../src/hw2/classification   # 训练/数据/模型代码（软链）
├── common -> ../src/hw2/common         # 通用工具（软链）
├── configs/classification.yaml     # 当前使用的配置快照
├── scripts/                        # 网格 / SE 重训 / 汇总 / 绘图脚本
│   ├── run_grid.sh
│   ├── run_se_retrain.sh
│   ├── summarize_grid.py
│   ├── build_task1_summary.py
│   └── make_swanlab_figs.py
├── checkpoints/                    # 5 个 best_*.pt 软链
├── results/                        # summary_*.json / history_*.csv / task1_summary.csv / grid_summary.csv
├── logs/                           # 9 组网格 + SE 重训的训练 log
└── figures/                        # 训练曲线 PNG + grid 热力图 + swanlab/ 子目录
    └── swanlab/                    # 多 run 风格对比图
```

## 题目要求（来自 HW2_深度学习与空间智能.pdf）

1. **Baseline**：在 102 Category Flower Dataset 上修改 ResNet-18/34 输出层；用 ImageNet 预训练参数初始化骨干，新输出层从零训练，骨干用较小学习率微调。
2. **超参数分析**：观察训练步数、学习率及其组合对性能的影响，尽量提升模型。
3. **预训练消融**：与从随机初始化训练的结果对比，量化预训练带来的提升。
4. **注意力机制**：在 baseline 基础上加入 SE-block / CBAM，或换用 ViT-Tiny / Swin-T，与 baseline 比较 Accuracy。

## 完成情况（全部 ✅）

| 子项 | 状态 | 关键指标 |
| --- | :---: | --- |
| (1) Baseline 预训练微调 | ✅ | ResNet-18 + ImageNet 预训练，head 3e-3 / bb 1e-4，30 epoch |
| (2) 超参数分析 | ✅ | 9 组 lr_head × lr_backbone 网格（初版增广），最优 val=0.9186；最优超参在最终增广下复跑 val=0.9284 |
| (3) 预训练 vs 随机初始化消融 | ✅ | 最终增广下 +52.58 pct test acc 提升 |
| (4) 注意力机制 | ✅ | SE-ResNet18 (v1 / v2 cosine)，结论：本场景不超 baseline |
| swanlab 可视化 | ✅ | 11 个 run 上报到 `swanlog/`，`swanlab watch swanlog` 截图 |
| 报告填充 | ✅ | `../reports/report.md` Task 1 段落已完整重写 |

## 最终结果汇总（`results/task1_summary.csv`）

| Tag | 配置 | Best Val | Test Acc | Test Loss |
| --- | --- | ---: | ---: | ---: |
| scratch (initial) | 随机初始化，初版增广 | 0.4196 | 0.3675 | 3.1271 |
| baseline_pretrained_v1 | 预训练，head 1e-3 / bb 1e-4，初版增广 | 0.9118 | 0.8839 | 0.5051 |
| baseline_best (initial) | 预训练，head 3e-3 / bb 1e-4，初版增广 | 0.9186 | 0.8920 | 0.4265 |
| se_pretrained_v1 | SE-ResNet18，head 1e-3 / bb 1e-4，初版增广 | 0.8892 | 0.8569 | 0.5790 |
| se_pretrained_v2_cosine (initial) | SE-ResNet18 + cosine，head 3e-3 / bb 1e-4，初版增广 | 0.8990 | 0.8699 | 0.5154 |
| **scratch_v2** | 随机初始化，最终增广 | 0.4314 | 0.3794 | 3.2474 |
| **baseline_best_v2** | **预训练，head 3e-3 / bb 1e-4，最终增广（REPORT.md 使用）** | **0.9284** | **0.9052** | **0.3703** |
| **se_v2_cosine_v2** | SE-ResNet18 + cosine，head 3e-3 / bb 1e-4，最终增广 | 0.9206 | 0.8866 | 0.4387 |

### 超参数网格（baseline，30 epoch，best val acc）

| lr_head \ lr_bb | 3e-5 | 1e-4 | 3e-4 |
| --- | ---: | ---: | ---: |
| 5e-4 | 0.9088 | 0.9108 | 0.9176 |
| 1e-3 | 0.9039 | 0.9147 | 0.9167 |
| 3e-3 | 0.9010 | **0.9186** | 0.9167 |

详图：`figures/grid_heatmap.png`。

## 复现命令

在仓库根目录 `/mnt/data/zjj/zl/hw2/`：

```bash
conda activate zl2
export PYTHONPATH=$PWD/src

# 1) 9 组超参网格 (~20 min on RTX 5090)
bash task1/scripts/run_grid.sh 1               # 参数: GPU index

# 2) 用 grid 最优超参 + cosine 训练 SE-ResNet18
bash task1/scripts/run_se_retrain.sh 1

# 3) 生成汇总表 + 热力图 + swanlab 风格对比图
python task1/scripts/summarize_grid.py
python task1/scripts/build_task1_summary.py
python task1/scripts/make_swanlab_figs.py

# 4) 单组手动训练示例（baseline 最优）
python -m hw2.classification.train \
  --config task1/configs/classification.yaml \
  --variant baseline_pretrained \
  --lr-head 3e-3 --lr-backbone 1e-4 \
  --tracker swanlab --run-name baseline_best

# 5) 看 swanlab dashboard（截图用）
swanlab watch swanlog
```

## 关键结论

1. **预训练是决定性的**：从零训练 test acc=0.3794（最终增广），预训练后 test acc=0.9052，提升 +52.58 pct。Flowers102 train 仅 1020 张，没有预训练完全没法学。
2. **lr 不对称**：头部新层 lr 应明显大于骨干 lr（3e-3 vs 1e-4 ≈ 30×）。骨干 lr 太小（3e-5）整行底排，太大与头部同量级又会冲刷预训练特征。
3. **SE-block 在本场景没赢**：在 ResNet-18 + Flowers102 + 30 epoch 上 SE 比 baseline 低约 2 pct test acc，即使用最优超参 + cosine 仍未追上。可能因为 BasicBlock 通道少（≤512）、SE 模块本身无预训练初始化，前期反而扰动了 ImageNet 特征。这是一个**有意义的负结论**，写进了报告。
