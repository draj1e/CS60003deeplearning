# HW2 进度文档

**一句话总结：已创建独立 Conda 环境 `zl2`，并在 `/mnt/data/zjj/zl/hw2` 初始化 HW2 工程，确保不与 HW1 冲突。**

| 时间 | 状态 | 内容 |
| --- | --- | --- |
| 2026-04-25 | 完成 | 读取 `HW2_深度学习与空间智能.pdf`，提取三项任务要求与提交限制 |
| 2026-04-25 | 完成 | 创建 Conda 环境 `zl2`，Python 版本为 3.10.20 |
| 2026-04-25 | 完成 | 确认施工目录为 `/mnt/data/zjj/zl/hw2`，不复用 HW1 根目录结构 |
| 2026-04-25 | 进行中 | 编写 HW2 设计文档与代码骨架 |

## 待办
- 补齐 `requirements.txt` 与环境安装说明。
- 搭建分类、检测跟踪、分割三套训练/评估脚本。
- 添加报告草稿模板和可视化输出说明。
- 安装依赖后执行导入与 smoke test。
| 2026-04-25 | 完成 | 写入 `requirements.txt`、`environment.yml`、三任务默认配置 |
| 2026-04-25 | 完成 | 实现花卉分类 ResNet 预训练/随机初始化/SE 注意力训练代码 |
| 2026-04-25 | 完成 | 实现 YOLOv8 训练、视频 Tracking ID、越线计数和遮挡帧导出代码 |
| 2026-04-25 | 完成 | 实现从零 U-Net、手写 Dice Loss、CE/Dice/CE+Dice 训练代码 |
| 2026-04-25 | 完成 | 编写 README 和报告 Markdown 草稿 |
| 2026-04-25 | 完成 | 在 `zl2` 安装依赖，并将 PyTorch 调整为 `cu128` 轮子以匹配本机 CUDA 12.8 驱动 |
| 2026-04-25 | 完成 | 通过导入、语法编译、分类/分割模型前向和 Dice Loss smoke test |
| 2026-04-25 | 完成 | 调研确认三个数据集均可公开下载，并新增统一下载脚本与数据说明 |
| 2026-04-25 | 完成 | 适配 Road Vehicle YOLOv8 dataset.yaml 与 Stanford `.regions.txt` 分割标签读取 |
| 2026-04-25 | 完成 | 为分类与分割训练入口增加 epoch、batch、image size、输出目录覆盖参数 |
| 2026-04-25 | 完成 | 完成 Flowers102/Stanford/YOLO 数据加载检查和三任务 1 epoch smoke test |
| 2026-04-25 | 完成 | 生成 12 秒 Road Vehicle 合成演示视频用于 tracking 脚本调试 |
| 2026-04-25 | 完成 | 使用合成 12 秒视频跑通 YOLO tracking、Tracking ID 绘制和越线计数输出 |
| 2026-04-25 | 完成 | 新增训练曲线绘图脚本并生成分类/分割 smoke 曲线图 |
| 2026-04-25 | 完成 | 更新 README 与报告草稿，补齐正式实验命令和 smoke test 输出位置 |
| 2026-04-25 | 完成 | 新增 `.gitignore`，避免提交数据、权重、输出和缓存文件 |
| 2026-04-26 | 完成 | 新增 `scripts/run_full_experiments.py`，用 `outputs/full_run_progress.json` 支持完整实验断点续跑 |
| 2026-04-26 | 完成 | 完整分类三组实验：baseline 预训练、scratch、SE 预训练 |
| 2026-04-26 | 完成 | 完整 YOLOv8 车辆检测训练 50 epoch，生成权重与 mAP 曲线 |
| 2026-04-26 | 完成 | 完整 U-Net 分割三组损失实验：CE、Dice、CE+Dice |
| 2026-04-26 | 完成 | 生成最终训练曲线、连续帧、tracking 视频和 `outputs/tables/summary.md` 指标汇总 |
