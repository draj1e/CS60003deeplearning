# Task 1 详细讲解：为什么这么做、每个数字代表什么

> 本文档配套 [`REPORT.md`](REPORT.md)，**报告负责"做了什么、得到什么"，本文负责"为什么这么做、结果该怎么解读"**。读完后你应该能：在面试或答辩时把所有数字、所有设计选择讲清楚，并解释 SE-block 为什么没赢。

---

## 目录

1. 题目在考什么
2. 数据集的"陷阱"：为什么 Flowers102 必须用迁移学习
3. 迁移学习的两个关键设计：参数分组 + 差分学习率
4. 超参数网格的解读：每个单元格在说什么
5. 预训练消融：+52 pct 是怎么来的
6. SE-block 为什么没赢：负结论的完整分析
7. 工程细节：AMP、CosineLR、swanlab、reproducibility
8. 常见追问与回答（FAQ）

---

## 1. 题目在考什么

题目四个要求，本质是把"现代图像分类的完整微调流程"拆成四份：

| 要求 | 在考的能力 |
| --- | --- |
| (1) Baseline 微调 | 会不会用预训练权重、能不能正确接 head、知不知道差分 lr |
| (2) 超参数分析 | 会不会做对照实验、会不会读热力图找最优 |
| (3) 预训练消融 | 知不知道迁移学习的边界 |
| (4) 注意力机制 | 能不能动手改网络（不只是调参） |

最容易忽略的点是 (2) 与 (4)：很多同学只跑一组超参就交、或者把 attention 当成"加上就有提升"的魔法。本作业**全部完成**且包含一个**反直觉的负结论**（SE 没超 baseline），这部分是核心讲解。

---

## 2. 数据集的"陷阱"：为什么 Flowers102 必须用迁移学习

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| 类别数 | 102 | 输出维度 |
| 每类训练样本 | 10 | **极少** |
| 训练集总图数 | 1020 | 一个 batch_size=32 训完只 32 iter |
| 测试集大小 | 6149 | 是训练集的 **6 倍** |

**这个数据集的训练集比测试集小 6 倍**——这是它故意设置的难度。把 102 类、每类 10 张样本拿去训一个 11M 参数的 ResNet-18，参数量是样本数的 11000 倍：必然严重过拟合，或者根本学不到结构。

**所以**：
- 从零训练 30 epoch 拿到 ~37% test acc 是"勉强收敛但泛化崩了"的典型样子。
- 预训练后只需要新输出层学"如何把 ImageNet 的 512 维特征映射到 102 类花卉"，仅 ~52K 参数 + ~1020 张样本，约 20:1 的样本-参数比，可以充分学到。

→ 这就是题目 (3) 一定要让你做消融的原因：让你**亲手感受**小数据集 + 大模型时迁移学习的不可替代性。

---

## 3. 迁移学习的两个关键设计：参数分组 + 差分学习率

### 3.1 参数分组（Parameter Groups）

PyTorch 的 optimizer 支持把参数分组，每组用独立的 hyperparams（lr、weight_decay、momentum）。代码：

```python
def parameter_groups(model, lr_backbone, lr_head):
    head_params = list(model.fc.parameters())     # 新接的 102 类输出层
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_head},
    ]
```

为什么必须这样：
- **backbone 已经训过**：在 ImageNet 1.28M 张图上训得很好；它的卷积核已经识别出花瓣的边缘、纹理、形状。我们只想**轻微调整**让它适应花卉数据集，所以 lr 要小。
- **head 完全没训过**：随机初始化的一个 Linear(512, 102)，需要从零拟合，lr 必须**显著大于** backbone。

**如果不分组**（所有参数同一个 lr）：
- lr 设得像 head 那样大 → backbone 的预训练权重被冲刷掉。
- lr 设得像 backbone 那样小 → head 30 epoch 都学不动。

### 3.2 差分学习率的实证

看 §5.2 的网格：

| lr_head \ lr_backbone | 3e-5 | 1e-4 | 3e-4 |
| --- | ---: | ---: | ---: |
| 5e-4 | 0.9088 | 0.9108 | 0.9176 |
| 1e-3 | 0.9039 | 0.9147 | 0.9167 |
| 3e-3 | 0.9010 | **0.9186** | 0.9167 |

- 第 1 列（bb=3e-5）整体偏低 → 骨干 lr 太小，"微调"成了"不调"。
- 第 3 列（bb=3e-4）整体最高，但与第 2 列（bb=1e-4）差距不大 → 骨干 lr 在 1e-4～3e-4 这个区间都行，再大就开始冲刷预训练特征。
- 第 3 行（head=3e-3）+ 第 2 列（bb=1e-4）= **最优**，比例 30:1。

**经验值**：head:backbone lr 比通常在 10:1～100:1，可作为后续任务的起点。

### 3.3 为什么不冻结 backbone（feature extractor）？

更激进的迁移学习是把 backbone 完全冻结（`requires_grad=False`），只训 head。这在数据非常少（<500）或者新任务与 ImageNet 极相似时管用。Flowers102 处于中间地带：花卉与 ImageNet 的"花"类别有重叠，但细粒度更高（102 类花 vs ImageNet 中可能只是 1-2 类），所以让 backbone 也微调一点效果更好——这就是我们用 lr_backbone=1e-4 而不是 0 的原因。

---

## 4. 超参数网格的解读：每个单元格在说什么

### 4.1 网格的几何形状

3×3 网格 × 30 epoch × 132s = 18 分钟。每个单元格是一次完整训练，**用同样的 seed=42、同样的 dataloader、同样的初始化**，唯一的变量是 lr_head、lr_backbone。这保证了对比的统计学合法性（无随机性混淆）。

### 4.2 网格的读法（不只是"找最大值"）

很多人看热力图只看最大那格。**更重要的是看趋势**：

1. **沿 lr_backbone 看**：固定 lr_head 时，lr_backbone 从 3e-5 → 1e-4 通常带来明显提升（约 +1 pct），从 1e-4 → 3e-4 边际变小甚至负向。说明骨干 lr 有一个"甜蜜区"，大约在 1e-4 附近。
2. **沿 lr_head 看**：固定 lr_backbone=1e-4 时，head 从 5e-4 → 3e-3 单调升 → head lr 还能再大试试（3e-3 是边界，应该试 1e-2）。
3. **对角线**：当 lr_head 和 lr_backbone 同向变化时，能看出训练动态有没有"协同"。本作业网格里，3e-3/3e-4 那格 (0.9167) 不如 3e-3/1e-4 (0.9186)，说明 head lr 已经偏大时，backbone lr 反而该收一点 → 这与"head 训不够时，backbone 要陪着多动"的直觉一致。

### 4.3 为什么没扫 lr_head=1e-2？

时间预算 + 经验。AdamW 在 ImageNet 风格的训练里 head lr 通常上限是 3e-3 ～ 1e-2，再大基本就发散。看到 3e-3 这一行的 val 还在涨，下一步可以补一组 head=1e-2 的实验（约 2 分钟）。

### 4.4 为什么没扫 epoch 数？

观察 `fig_grid_curves.png`：所有 9 条 val acc 曲线在 epoch 15 之后基本平稳。再训也无收益（甚至轻微过拟合）。**这是用曲线代替了 epoch 维度的扫表**——比扫 {30, 50, 100} 三个 epoch 数更省时间，也更直观。

---

## 5. 预训练消融：+52 pct 是怎么来的

### 5.1 数值

| 模型 | Test Acc | 说明 |
| --- | ---: | --- |
| ResNet-18 from scratch | 0.3675 | 30 epoch，head 1e-3 / bb 1e-4 |
| ResNet-18 ImageNet pretrained（grid best） | 0.8920 | 同样 30 epoch，最优超参 |
| Δ | **+0.5245** | **+52.45 pct** |

### 5.2 为什么这么巨大？

ImageNet 上预训练的 ResNet-18 已经把"卷积→ReLU→BN"的早期表示（边缘、纹理、颜色、低阶形状）学得非常好。这部分**对花卉同样有用**：花瓣边缘、对称性、纹理重复模式——这些是任何视觉数据都共享的低层结构。

从零训练时：
- 需要 1020 张样本同时学**所有 11M 参数**（卷积核 + BN + head），不可能。
- 训出来的卷积核是"刚好让这 1020 张图被记住"的过拟合解，对 6149 张测试图泛化崩塌。

预训练 + 微调时：
- 卷积核只需要**轻微调整**（lr=1e-4，30 epoch 累积调整量约 30 × batch_avg_grad × 1e-4 ≈ 很小）。
- head 在 1020 张样本上学一个 512 → 102 的线性分类器，足够。

### 5.3 把这个数字摆到上下文里

- ResNet-18 在 ImageNet 上的 top-1 acc 约 70%。
- Flowers102 task 上 89.2% test acc 大约等于："只要预训练特征"就能在新任务上拿到比 ImageNet 本身还高的准确率，因为 102 类 < 1000 类，分类难度本身就低。
- 即便如此，从零训也只能拿到 36.75%——比"随机猜"（1/102=0.98%）高很多但远不够实用。

→ **这就是迁移学习的实际价值，不只是理论**。

---

## 6. SE-block 为什么没赢：负结论的完整分析

这是本作业最有意思的部分。

### 6.1 数据

| 配置 | Best Val | Test Acc | Δ vs baseline best |
| --- | ---: | ---: | ---: |
| Baseline ResNet-18 | **0.9186** | **0.8920** | — |
| SE-ResNet18 v1（默认超参） | 0.8892 | 0.8569 | −3.51 pct |
| SE-ResNet18 v2（最优超参 + cosine） | 0.8990 | 0.8699 | −2.21 pct |

### 6.2 SE-block 是什么

通道注意力（channel attention）。每层特征图 (B, C, H, W) 做：
1. 全局平均池化 → (B, C)
2. Linear(C → C/r) → ReLU → Linear(C/r → C)
3. Sigmoid → 得到每个通道的权重 (B, C)
4. 乘回原特征图 (B, C, H, W) × (B, C, 1, 1)

直觉：让网络自己学"这一层的 256 个通道，哪些重要、哪些不重要"。

### 6.3 通常 SE-block 会赢，为什么这里没赢？

**原因 1：SE 模块本身没有预训练权重**
- ResNet-18 的 BasicBlock 数量是 [2, 2, 2, 2] = 8 个 block，每个加一个 SE 模块。
- 每个 SE 模块有 2 个 Linear 层，加起来 ~5K 参数（对 c=64-512 不等）。
- 这些参数**完全随机初始化**，需要从零学。
- 在 epoch 1-5 时，Sigmoid 输出大约在 0.5 附近（随机权重 → 输出接近 0），等于在 ImageNet 预训练好的特征上**乘了一堆 0.5**——信号衰减约 50%，需要 backbone 重新适应。

**原因 2：ResNet-18 通道数太少**
- SE 论文 (Hu et al., 2018) 的主实验是 SE-ResNet-50/101 在 ImageNet 上 +0.5～1.5 pct。
- ResNet-18 最大通道数 512，最小 64。SE 模块的 bottleneck 是 c/16 = 4～32，太窄，**通道注意力的表达能力受限**。

**原因 3：数据集太小**
- SE 引入了额外的参数和需要学的注意力权重。在 train=1020 的数据上，新增参数缺少足够样本去学好。
- 在大数据集（ImageNet 1.28M）上，SE 的优势会显现；在 Flowers102 上，反而是负担。

### 6.4 怎么让 SE 赢？（未做，但是可以讨论）

- **更深的骨干**：换 SE-ResNet-50。
- **更长的训练**：从 30 epoch 加到 100 epoch，让 SE 模块充分学习。
- **更好的初始化**：把 SE 的 Linear bias 初始化为 0、weight 初始化让 Sigmoid 输出接近 1（即"恒等门控"），让 SE 从"不影响"出发，学到优势再激活。
- **EMA / 蒸馏**：用 baseline 的输出作为 soft target，约束 SE 不偏离太远。

### 6.5 这个负结论怎么写进报告

不是"实验失败"，而是"经过严谨对比的工程经验"：
> 注意力机制并非"加上就好"，与骨干容量、数据规模、初始化策略强相关。在 ResNet-18 + Flowers102 + 30 epoch + 默认初始化的设置下，SE-block 反而降低了 test acc，原因在于 SE 模块缺少预训练初始化、骨干通道数较小、数据集训练样本不足以覆盖额外参数。

这是非常有价值的工程洞见——**评审看到负结论的真实分析，比一个无脑提升的"假赢"更可信**。

---

## 7. 工程细节

### 7.1 AMP（混合精度）

`torch.cuda.amp.autocast` + `GradScaler`：
- 前向用 float16 算（FP16 矩阵乘法在 5090 上有 Tensor Core 加速，约 2× 提速）。
- 梯度用 float32 缩放后再 unscale，避免 FP16 下溢（梯度太小变 0）。
- 在本作业中：30 epoch 训练时间从 ~250s 降到 ~132s，**约 1.9× 提速**，无精度损失。

### 7.2 CosineAnnealingLR（SE v2 用）

`lr(t) = lr_init * 0.5 * (1 + cos(π * t / T_max))`，t 是 epoch，T_max=30。
- epoch 0：lr = lr_init
- epoch 15（中点）：lr = lr_init / 2
- epoch 30：lr ≈ 0

为什么 SE 加 cosine？SE 模块在前期需要学（用大 lr），后期需要精调（用小 lr），cosine 自然实现了"warm 到 cool"。

### 7.3 swanlab 集成

`train.py` 里通过 `_init_tracker / _log_tracker / _finish_tracker` 三个函数封装 swanlab/wandb，每个 run 自动落到 `swanlog/run-<timestamp>-<hash>/`。要看：

```bash
swanlab watch swanlog
# 浏览器开 http://127.0.0.1:5092
```

可以选择多个 run 叠加 loss/acc 曲线。本作业 10 个 run 都在 swanlog 里。

### 7.4 可复现性

- `set_seed(42)`：torch / numpy / random / cuda 全部固定。
- `torch.backends.cudnn.benchmark = True`：会因 input shape 不同选不同 kernel，对吞吐有利但**理论上不严格 deterministic**。本作业 image_size 固定 224、batch_size 固定 32，输入 shape 不变，benchmark 不会跨 epoch 切换 kernel，可视为可复现。
- 严格 deterministic 需要 `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`，会牺牲 ~10% 吞吐，本作业未启用。

---

## 8. FAQ

**Q1：为什么不用 ResNet-34？**
A：题目允许 18 或 34。18 已经够：训练快、显存小、与"小数据集 + 简单骨干"原则一致。ResNet-34 通常额外 +0.5 pct，但训练时间翻倍，性价比低。

**Q2：为什么不用 ViT-Tiny？**
A：题目要求"加注意力机制 **或** 用轻量 ViT"——二选一。我们选了 SE-block 路线，并完成了 v1+v2 两组 SE 实验。ViT-Tiny 是后备方案，时间允许时可补做（预期 89-91% test acc，与 baseline 接近）。

**Q3：为什么 baseline_pretrained v1 与 best 都用 30 epoch？**
A：从 grid 曲线 (`fig_grid_curves.png`) 看，所有 9 组都在 epoch 15 后平稳。更长 epoch 不带来提升，反而占资源。**用曲线代替 epoch 网格扫描**是一种更高效的做法。

**Q4：测试集为什么是 6149 而不是 1020？**
A：Flowers102 的官方设定就是 train=1020 / val=1020 / test=6149（每类 train+val 各 10 张，剩下都是 test）。这是为了"训练数据稀缺、测试评估充分"的研究设定。

**Q5：为什么 SE v1 的 val_loss 一开始那么高（接近 4）？**
A：SE 模块随机初始化时，前期相当于把预训练特征乘以 ~0.5（Sigmoid 输出的初始期望），等于"破坏"预训练特征。需要 5-10 epoch 让 SE 学到"接近恒等"才能恢复。这正是 §6.3 原因 1 的可视化证据。看 `fig_se_vs_baseline.png`，SE v1 的橙色曲线在 epoch 1-5 显著高于 baseline 蓝色。

**Q6：如何把这个项目交出去？**
A：(1) 把 task1/ 目录连同 src/、scripts/、configs/ 推到 public GitHub repo；(2) 把 task1/checkpoints/ 里的几个 .pt 上传到百度云/Drive；(3) 把链接填进 `REPORT.md` §0 checklist 和 §9。

---

**作者**：_待填写_  
**完成时间**：2026-05-18  
**配套报告**：[`REPORT.md`](REPORT.md)
