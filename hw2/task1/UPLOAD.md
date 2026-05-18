# Task 1 提交清单：GitHub 与 网盘 分工

> 本文件列出 Task 1 提交时**哪些文件进 GitHub public repo**、**哪些进网盘（百度云/Google Drive）**、以及打包说明。报告 PDF 中只需粘两个 URL：repo 链接 + 网盘下载链接。

---

## 1. 总体策略

| 目标 | 内容 | 大小 | 上传位置 |
| --- | --- | ---: | --- |
| 可读可复现 | 代码 / 配置 / 脚本 / 文档 / 训练日志 / 训练曲线 PNG / 小型 CSV | ~1.9 MB | **GitHub** |
| 不适合 git 的二进制 | 5 个模型权重 (`.pt`) + swanlab 原始日志 | **200 MB** (tar.gz) | **网盘** |

理由：
- GitHub 单文件硬上限 100 MB；single `.pt` 已 43 MB，5 个权重总和远超 git 友好范围，且 `.pt` 是二进制，git diff/blame 没意义。
- swanlab `runs.swanlab` 是 SQLite 二进制，同理放网盘。
- 网盘包让评审一条 wget/下载链接就能拿到全部权重和原始日志，比 git-lfs 友好。

---

## 2. GitHub 上传清单

整个 `task1/` 几乎全部上传，**除了** `task1/checkpoints/`（软链指向大权重）和 `task1/upload/`（打包产物）。同时上传仓库根目录的工程文件。

```
仓库根/
├── README.md                       # 工程总入口（环境/数据/三任务说明）
├── DESIGN.md                       # 工程设计文档
├── PROGRESS.md                     # 进度日志
├── requirements.txt                # pip 依赖
├── environment.yml                 # conda 环境描述
├── docs_datasets.md                # 数据集来源说明
├── .gitignore                      # 已配置忽略 data/、outputs/、*.pt、swanlog/ 等
├── configs/classification.yaml     # 分类任务配置
├── src/hw2/                        # 全部源码 (本作业只涉及 classification/ + common/)
│   ├── __init__.py
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── data.py                 # Flowers102 dataloader
│   │   ├── models.py               # ResNet/SE-ResNet 构造
│   │   └── train.py                # 训练入口（支持 swanlab/wandb）
│   └── common/                     # utils / metrics / plotting
├── scripts/
│   ├── download_datasets.py
│   └── plot_histories.py
└── task1/                          # 本作业核心目录
    ├── README.md                   # task1 目录索引 + 复现命令
    ├── REPORT.md                   # 提交版实验报告
    ├── REPORT.pdf                  # ★ 评审主要看这个（pandoc 生成，与 .md 内容相同）
    ├── EXPLAINED.md                # 技术讲解（原理 / FAQ / 负结论分析）
    ├── UPLOAD.md                   # 本文件
    ├── configs/classification.yaml # 训练用配置快照
    ├── scripts/                    # 实验编排脚本（5 个）
    │   ├── run_grid.sh
    │   ├── run_se_retrain.sh
    │   ├── summarize_grid.py
    │   ├── build_task1_summary.py
    │   └── make_swanlab_figs.py
    ├── results/                    # 小型可读结果文件
    │   ├── task1_summary.csv       # 5 个核心 run 对比表
    │   ├── grid_summary.csv        # 9 组网格详表
    │   ├── summary_*.json (×5)     # 每个 run 的最终摘要
    │   └── history_*.csv (×5)      # 每个 run 的 per-epoch 曲线数据
    ├── logs/                       # 训练终端输出 (10 个 .log，每个 < 100KB)
    └── figures/                    # 训练曲线 + 热力图 (8 张 PNG，总 ~700KB)
        ├── baseline_best_curves.png
        ├── grid_heatmap.png
        ├── se_v2_curves.png
        ├── classification_*_curves.png (×3, 历史曲线)
        └── swanlab/
            ├── fig_main_compare.png
            ├── fig_grid_curves.png
            └── fig_se_vs_baseline.png
```

**不要 push** 这些：
- `task1/checkpoints/` — 全是软链，且目标是大 `.pt`
- `task1/upload/` — 网盘上传产物
- `task1/code -> ../src/hw2/classification`、`task1/common -> ../src/hw2/common` — 软链本身（git 会跟踪软链节点，但目标已经独立在 `src/` 里）
- `outputs/`、`swanlog/`、`data/`、`runs/` — `.gitignore` 已忽略
- `*.pt`、`*.pth` — `.gitignore` 已忽略

### 2.1 README.md 必须写的内容（题目要求）

`task1/README.md` 已经写了；如果想让仓库根 README 同样合规，确认它包含：
- 环境配置（conda 命令）
- 数据集下载（已经写了 `python scripts/download_datasets.py`）
- 训练命令（已经写了）
- 测试命令（用 ckpt 加载推理的示例代码，建议补一段）

---

## 3. 网盘上传清单

### 3.1 已打包文件

```
task1/upload/hw2_task1_weights_swanlog.tar.gz
```

- 大小：**200 MB**
- 格式：tar.gz
- SHA-256：`b489345ece6d0f3cb04d80b5f362872aaf1f134f70a554dd4c0adc42f146137f`
- 文件数：1640

### 3.2 内部结构

```
hw2_task1_weights_swanlog.tar.gz
├── MANIFEST.txt                                  # 内容/加载示例/对应代码 repo 提示
├── checkpoints/                                  # 5 个 PyTorch state_dict
│   ├── best_baseline_best.pt              43 MB  # ★ 推荐使用 (test acc 89.20%)
│   ├── best_baseline_pretrained.pt        43 MB  # baseline v1 (test 88.39%)
│   ├── best_scratch.pt                    43 MB  # 消融对照 (test 36.75%)
│   ├── best_se_pretrained.pt              44 MB  # SE v1 (test 85.69%)
│   └── best_se_pretrained_v2_cosine.pt    44 MB  # SE v2 最终版 (test 86.99%)
└── swanlog/                                      # swanlab 本地原始日志
    └── run-20260518_*/                           # 10 个 run (9 grid + 1 SE v2)
        ├── backup.swanlab                        # SQLite 训练数据
        ├── console/                              # 终端输出快照
        ├── files/                                # 该 run 的 cfg / metadata / requirements
        └── logs/                                 # 各 metric 的 per-step 数值
```

### 3.3 评审使用方法

下载并解压后：

```bash
tar xzf hw2_task1_weights_swanlog.tar.gz
ls checkpoints/        # 5 个 .pt
ls swanlog/            # 10 个 run + runs.swanlab
```

**加载模型推理**：
```python
import torch
from hw2.classification.models import build_classifier

ckpt = torch.load("checkpoints/best_baseline_best.pt", map_location="cpu")
model = build_classifier(arch="resnet18", num_classes=102,
                         pretrained=False, attention="none")
model.load_state_dict(ckpt["model"])
model.eval()
# 现在可以对 224×224 RGB tensor (已 ImageNet 归一化) 做 inference
```

**重现 swanlab dashboard**（看曲线截图）：
```bash
pip install 'swanlab[dashboard]'
swanlab watch swanlog
# 浏览器打开 http://127.0.0.1:5092
```

### 3.4 上传步骤

| 平台 | 步骤 |
| --- | --- |
| **百度云** | 网页登录 → 新建文件夹 `HW2_Task1` → 上传 `hw2_task1_weights_swanlog.tar.gz` → 分享，**勾选"提取码"**，有效期选 7 天或永久 → 复制"链接 + 提取码"两行 |
| **Google Drive** | 网页登录 → 新建文件夹 `HW2_Task1` → 上传 tar.gz → 右键"获取链接" → 改为"知道链接的任何人"→ 复制链接 |

上传完成后，把链接填进 `REPORT.md` 第 9 节"代码与权重"。

---

## 4. 已填好的链接（提交前的最后核对）

`task1/REPORT.md` 已经写入：

| 位置 | 内容 |
| --- | --- |
| §0 checklist & §9 | GitHub: <https://github.com/draj1e/CS60003deeplearning>（HW2 在 `hw2/` 子目录） |
| §0 checklist & §9 | 百度云: <https://pan.baidu.com/s/1_SVen8Jpwpx2-HYKtRuO8A?pwd=6666> （提取码 `6666`） |
| §1 小组信息表 | ⚠️ 仍需手动填写 姓名 / 学号 / 分工 |

填完组员信息后直接 `pandoc task1/REPORT.md -o task1/REPORT.pdf --pdf-engine=xelatex -V mainfont="Noto Sans CJK SC"` 转 PDF 提交。

---

## 5. 推荐提交流程

```bash
# 1) 把 task1/upload/hw2_task1_weights_swanlog.tar.gz 传到网盘
#    （手动操作；得到一个 share URL + 可能的提取码）

# 2) 把 URL 填进 task1/REPORT.md §1 / §9 / §0 三处

# 3) 生成 PDF
pandoc task1/REPORT.md -o task1/REPORT.pdf \
    --pdf-engine=xelatex -V mainfont="Noto Sans CJK SC"

# 4) 提交到 GitHub
git add task1/ src/hw2/ scripts/ configs/ requirements.txt environment.yml README.md DESIGN.md PROGRESS.md docs_datasets.md .gitignore
git commit -m "HW2 Task 1: classification with pretrained ResNet-18"
git push

# 5) 报告 PDF 单独提交到课程平台
```

---

## 6. .gitignore 检查项

确保仓库根 `.gitignore` 已包含（当前已配置）：

```gitignore
# 大文件 / 不可读二进制
*.pt
*.pth
*.onnx

# 数据 / 输出目录
data/
outputs/
runs/
swanlog/

# 网盘打包产物
task1/upload/

# Python 缓存
__pycache__/
*.pyc
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
```

---

## 7. 速查表

| 提交清单 | 在哪里 |
| --- | --- |
| 实验报告 (PDF) | 课程提交平台 |
| GitHub repo URL | 报告 §9 |
| 网盘 share URL + 提取码 | 报告 §9 |
| 组员姓名/学号/分工 | 报告 §1 |
| 代码 / 文档 / 配置 / 训练日志 / 曲线 PNG | GitHub repo |
| 5 个模型权重 + swanlab 原始日志 | 网盘 tar.gz (200 MB) |
| MANIFEST.txt（加载示例 / 对应 commit hash） | 在 tar.gz 内部 |

完成。
