# HW2 实验报告草稿

**一句话总结：本报告比较预训练与注意力机制对花卉分类的影响，展示车辆检测跟踪与越线计数结果，并验证 U-Net 三种损失函数在语义分割上的 mIoU 差异。**

| 模块 | 方法 | 指标 | 结果文件 |
| --- | --- | --- | --- |
| 花卉分类 | ResNet baseline / scratch / SE-ResNet | Accuracy | `outputs/classification/summary_*.json` |
| 车辆检测跟踪 | YOLOv8 + tracking + line crossing | mAP / Tracking ID / count | `outputs/detection/` |
| 图像分割 | U-Net + CE/Dice/CE+Dice | mIoU | `outputs/segmentation/summary_*.json` |

## 不能做的
- 任务 3 不能使用任何预训练权重。
- 不能用测试集调参。
- 不能只给检测框而缺少稳定 Tracking ID。
- 不能缺少 Github repo 链接和模型权重下载地址。

## 小组信息
- 姓名 / 学号 / 分工：待填写。

## 任务 1：花卉分类

### 数据集与划分
- **102 Category Flower Dataset**（Oxford VGG），102 类花卉。
- 通过 `torchvision.datasets.Flowers102` 自动下载：train=1020 / val=1020 / test=6149。
- 训练增广：RandomResizedCrop(224)、RandomHorizontalFlip、ImageNet 均值方差归一化；val/test 仅 Resize+CenterCrop。

### 网络结构
- **Baseline**：`torchvision.models.resnet18(weights=IMAGENET1K_V1)`，替换 `fc` 为 `Linear(512, 102)`。
- **Scratch 消融**：同结构但 `weights=None`，所有参数随机初始化。
- **SE-ResNet18**：手动用 `SEBasicBlock`（在每个 `BasicBlock` 第二个 conv 后插入 SE-block，reduction=16）从头组装 ResNet，再将 ImageNet 预训练权重 strict=False 拷入兼容层。

### 训练设置（统一）
| 项 | 值 |
| --- | --- |
| Optimizer | AdamW (weight_decay=1e-4) |
| Image size | 224×224 |
| Batch size | 32 |
| Epoch | 30 |
| Mixed precision | AMP (`torch.cuda.amp`) |
| Loss | CrossEntropyLoss |
| 评价指标 | top-1 Accuracy |
| Tracker | swanlab（本地 dashboard） |
| 参数分组 | backbone / head 两组独立 lr |
| 硬件 | NVIDIA RTX 5090 × 1，约 132 s / 30 epoch |

### (2) 超参数分析：lr_head × lr_backbone 网格

在 `ResNet-18 + ImageNet 预训练` 上对 9 组组合各跑 30 epoch，得到 best val acc 表：

| lr_head \ lr_backbone | 3e-5 | 1e-4 | 3e-4 |
| --- | ---: | ---: | ---: |
| 5e-4 | 0.9088 | 0.9108 | 0.9176 |
| 1e-3 | 0.9039 | 0.9147 | 0.9167 |
| 3e-3 | 0.9010 | **0.9186** | 0.9167 |

观察：
- 骨干 lr 太小 (3e-5) 时整行均偏低 ≈ 0.90 → "微调"的关键是允许骨干以略小但非极小的 lr 适应新数据。
- 头部 lr 提升到 3e-3 时表现更好，说明新输出层确实需要较大 lr 从零拟合。
- 最优组合：**`lr_head=3e-3, lr_backbone=1e-4`**，best val=0.9186，test acc=0.8920。
- 详见 `task1/figures/grid_heatmap.png`、`task1/results/grid_summary.csv`。

### (3) 预训练消融

| 模型 | Best Val Acc | Test Acc | Δ vs scratch |
| --- | ---: | ---: | ---: |
| Scratch (随机初始化) | 0.4196 | 0.3675 | — |
| Baseline pretrained (v1, head 1e-3 / bb 1e-4) | 0.9118 | 0.8839 | +51.6 pct |
| **Baseline pretrained (grid 最优)** | **0.9186** | **0.8920** | **+52.5 pct** |

预训练让 test acc 提升超过 50 个百分点，验证了 ImageNet 预训练对小数据集（train 仅 1020 张）的关键作用。

### (4) 注意力机制 (SE-block) 对比

| 模型 | 超参 | Best Val | Test Acc |
| --- | --- | ---: | ---: |
| Baseline ResNet-18 (grid 最优) | head 3e-3 / bb 1e-4 | **0.9186** | **0.8920** |
| SE-ResNet18 (v1) | head 1e-3 / bb 1e-4 | 0.8892 | 0.8569 |
| SE-ResNet18 (v2, cosine) | head 3e-3 / bb 1e-4 + cosine | 0.8990 | 0.8699 |

观察：
- 直接在 ResNet-18 BasicBlock 中插入 SE-block 并未超过 baseline；最优超参 + cosine 调度让 SE 提升 +1.3 pct test，但仍低于 baseline 约 -2.2 pct。
- 可能原因：SE-block 在 BasicBlock 通道数较少 (64/128/256/512) 时增益有限；ImageNet 预训练权重无法覆盖 SE 模块本身的参数，导致 SE 前几个 epoch 需要从随机权重学习并对原有特征引入噪声。在更深/更宽的骨干（ResNet-50, EfficientNet）和更大数据集上 SE 收益通常更显著。

### 训练曲线（swanlab + matplotlib 双重记录）

- 网格热力图：`task1/figures/grid_heatmap.png`
- Baseline 最优 30-epoch：`task1/figures/baseline_best_curves.png`
- SE-ResNet18 v2 30-epoch：`task1/figures/se_v2_curves.png`
- Scratch（消融对照）：`task1/figures/classification_scratch_curves.png`
- swanlab 本地日志：`swanlog/`（用 `swanlab watch swanlog` 启 dashboard 截图）

### 结论
1. ImageNet 预训练对小数据集是决定性的：+52 pct。
2. 头部 lr 应明显大于骨干 lr（3e-3 vs 1e-4），与论文经验一致。
3. SE-block 在 ResNet-18 + Flowers102 上没体现优势，是一个**有意义的负结论**：注意力机制并非"加上就好"，与骨干容量、数据规模强相关。

## 任务 2：车辆检测跟踪
- 数据集：Road Vehicle Images Dataset。
- 测试视频：10-30 秒校园/路口视频。
- 方法：YOLOv8 微调，结合 tracking 输出 bbox、类别、Tracking ID。
- 越线计数：基于检测框中心点与虚拟线两侧符号变化，结合 Tracking ID 去重计数。
- 遮挡分析：待插入连续 3-4 帧可视化并分析 ID 维持/跳变原因。

## 任务 3：U-Net 分割
- 数据集：Stanford Background Dataset。
- 模型：手写 U-Net，随机初始化，编码器-解码器与 Skip Connection。
- 损失：Cross-Entropy、手写 Dice Loss、CE+Dice。
- 结果：待填三种损失的验证 mIoU 对比。

## 代码和模型
- Github repo：待填写。
- 模型权重网盘地址：待填写。

## 当前 smoke test 结果

| 模块 | 配置 | 结果 |
| --- | --- | --- |
| 分类 | scratch, 1 epoch, 64x64 | `outputs/smoke/classification/summary_scratch.json` |
| 分割 | U-Net + CE, 1 epoch, 64x64 | `outputs/smoke/segmentation/summary_ce.json` |
| 检测 | YOLOv8n, 1 epoch, 320px | `runs/detect/outputs/smoke/detection/train/results.csv` |
| 跟踪 | 12 秒合成视频 | `outputs/detection/tracking/tracked_counted.mp4` |

正式报告需要用完整 epoch 重跑，替换 smoke test 数值和截图。

## 完整实验结果（2026-04-26）

### 分类 Accuracy 对比

| 模型 | Best Val Acc | Test Acc | 结论 |
| --- | --- | --- | --- |
| ResNet18 ImageNet 预训练微调 | 0.9118 | 0.8839 | 最优，预训练显著提升收敛速度和测试精度 |
| ResNet18 随机初始化 | 0.4196 | 0.3675 | 明显低于预训练，说明小数据集从零训练泛化不足 |
| SE-ResNet18 预训练 | 0.8892 | 0.8569 | 注意力模型有效但本次略低于 baseline |

### 检测 mAP

| 模型 | Best Epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| YOLOv8n 微调 | 40 | 0.5540 | 0.4108 | 0.4072 | 0.2537 |

### 分割 mIoU 对比

| 损失配置 | Best Val mIoU | 结论 |
| --- | --- | --- |
| Cross-Entropy | 0.5466 | 稳定基线 |
| Dice Loss | 0.5268 | 能处理类别不均衡，但单独使用略低 |
| CE + Dice | 0.5508 | 本次最佳，兼顾像素分类和区域重叠 |

### 已生成素材

- 分类曲线：`outputs/figures/classification_*_curves.png`
- 分割曲线：`outputs/figures/segmentation_*_curves.png`
- YOLO 曲线：`outputs/detection/train/results.png`
- YOLO 权重：`outputs/detection/train/weights/best.pt`
- Tracking 视频：`outputs/detection/tracking/tracked_counted.mp4`
- 遮挡/连续帧：`outputs/detection/occlusion_frames/frame_*.jpg`
- 汇总表：`outputs/tables/summary.md`
