# HW2 设计文档

**一句话总结：在独立 `zl2` 环境与 `hw2/` 工程中，完成分类微调、车辆检测跟踪计数、U-Net 分割三项实验的可复现代码与报告素材生成。**

| 任务 | 数据集 | 模型/方法 | 核心指标 | 主要输出 |
| --- | --- | --- | --- | --- |
| 任务 1 分类 | 102 Category Flower Dataset | ResNet-18/34 微调、随机初始化消融、SE-ResNet 注意力模型 | Accuracy | 训练曲线、验证/测试准确率、对比表 |
| 任务 2 检测跟踪 | Road Vehicle Images Dataset + 10-30 秒视频 | YOLOv8 微调 + ByteTrack/BoT-SORT + 越线计数 | mAP、稳定 Tracking ID、计数 | 检测权重、跟踪视频、遮挡帧可视化、计数 CSV |
| 任务 3 分割 | Stanford Background Dataset | 从零手写 U-Net，无预训练 | mIoU | 三种损失配置曲线、mIoU 对比、预测可视化 |

## 严格限制
- 任务 1 必须使用 ImageNet 预训练参数初始化 baseline，并对新输出层从零训练、骨干小学习率微调。
- 任务 1 必须做随机初始化 vs 预训练消融；必须加入注意力机制或轻量 ViT/Swin 对比。
- 任务 2 必须使用自训练/微调检测模型；视频结果必须包含 bbox、类别和稳定 Tracking ID。
- 任务 2 必须实现虚拟线越线计数，并分析遮挡/密集交汇导致的 ID 维持或跳变。
- 任务 3 禁止任何预训练权重；U-Net 必须用基础 PyTorch API 手写，包含编码器、解码器和 Skip Connection。
- 任务 3 Dice Loss 必须手动实现，并分别训练 CE、Dice、CE+Dice 三种配置。
- 最终提交仅 PDF 实验报告；报告需包含 Github repo 链接和模型权重网盘地址。
- 代码工程放在 `/mnt/data/zjj/zl/hw2`，不修改 HW1 根目录代码与文档。

## 工程结构
- `configs/`：任务默认 YAML 配置。
- `src/hw2/common/`：通用工具、指标、日志、可视化。
- `src/hw2/classification/`：花卉分类数据、模型、训练、评估。
- `src/hw2/detection/`：YOLO 数据转换、训练、视频跟踪、越线计数、遮挡帧导出。
- `src/hw2/segmentation/`：Stanford Background 数据、U-Net、Dice Loss、训练、评估。
- `scripts/`：命令行入口脚本。
- `reports/`：实验报告 Markdown 草稿与图表占位。
- `outputs/`：训练历史、模型、图像、视频和表格输出。

## 实现策略
1. 先搭建可运行工程和配置文件，所有脚本支持 `--config`、`--data-root`、`--output-dir` 覆盖。
2. 分类任务使用 `torchvision.models`；参数组区分 backbone/head 学习率；SE 模块手动插入 ResNet BasicBlock。
3. 检测任务复用 `ultralytics` 训练/跟踪能力，补充数据集转 YOLO 格式、计数逻辑和遮挡帧导出。
4. 分割任务完全自定义 PyTorch Dataset、U-Net、Dice Loss 和 mIoU，不调用预训练模型。
5. 日志同时落地本地 CSV/JSON；若配置启用 `wandb` 或 `swanlab`，再上传可视化曲线。

## 验证策略
- 先运行纯 Python 导入检查和轻量单元级 smoke test。
- 有数据后分别用极小 epoch/batch 做三任务端到端冒烟。
- 完整训练、报告截图和网盘权重地址由长训练完成后补齐。
