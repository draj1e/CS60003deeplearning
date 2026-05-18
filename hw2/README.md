# HW2 深度学习与空间智能

**一句话总结：本工程在 Conda 环境 `zl2` 中完成花卉分类微调、车辆检测跟踪计数、U-Net 分割三项实验。**

## 环境配置

```bash
conda activate zl2
cd /mnt/data/zjj/zl/hw2
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

如需新建环境：

```bash
conda create -y -n zl2 python=3.10
conda activate zl2
pip install -r requirements.txt
```


## 数据集下载

三项任务数据集都可从公开来源获取，统一脚本见 `scripts/download_datasets.py`：

```bash
conda activate zl2
cd /mnt/data/zjj/zl/hw2
export PYTHONPATH=$PWD/src
python scripts/download_datasets.py --dataset all
```

详细来源和注意事项见 `docs_datasets.md`。

## 任务 1：102 类花卉分类

```bash
export PYTHONPATH=$PWD/src
python -m hw2.classification.train --config configs/classification.yaml --variant baseline_pretrained
python -m hw2.classification.train --config configs/classification.yaml --variant scratch
python -m hw2.classification.train --config configs/classification.yaml --variant se_pretrained
```

输出：`outputs/classification/history_*.csv`、`best_*.pt`、`summary_*.json`。

## 任务 2：车辆检测、跟踪与越线计数

先将 Road Vehicle Images Dataset 转成 YOLO 格式，并在 `configs/detection.yaml` 中设置 `yolo_dataset_yaml`。

```bash
export PYTHONPATH=$PWD/src
python scripts/prepare_road_vehicle_yaml.py
python -m hw2.detection.train_yolo --config configs/detection.yaml
python -m hw2.detection.track_count --config configs/detection.yaml --weights outputs/detection/train/weights/best.pt --video data/videos/test.mp4
python -m hw2.detection.export_occlusion_frames --config configs/detection.yaml --video outputs/detection/tracking/tracked_counted.mp4 --frames 120 121 122 123
```

输出：YOLO 训练结果、带 Tracking ID 和计数的视频、遮挡连续帧、`count_events.json`。

## 任务 3：Stanford Background U-Net 分割

数据按如下结构放置：

```text
data/stanford_background/
  images/*.jpg
  labels/*.png
```

训练三种损失配置：

```bash
export PYTHONPATH=$PWD/src
python -m hw2.segmentation.train --config configs/segmentation.yaml --loss ce
python -m hw2.segmentation.train --config configs/segmentation.yaml --loss dice
python -m hw2.segmentation.train --config configs/segmentation.yaml --loss ce_dice
```

输出：`outputs/segmentation/history_*.csv`、`best_unet_*.pt`、`summary_*.json`。

## 报告要求检查清单

- 模型结构、数据集、实验结果介绍。
- 训练/测试划分、batch size、learning rate、optimizer、epoch、loss、指标。
- wandb 或 swanlab 的训练/验证 loss 曲线，以及 Accuracy/mAP/mIoU 曲线截图。
- Github public repo 链接。
- 模型权重网盘下载地址。
- 首页写明所有组员姓名、学号和具体分工。

## 已完成的本地验证

- Flowers102、Road Vehicle、Stanford Background 三个数据集已下载到 `data/`。
- 分类、分割、YOLO 检测均已完成 1 epoch smoke test。
- YOLOv8 数据集配置已生成：`data/road_vehicle_raw/trafic_data/dataset.yaml`。

## 建议正式实验命令

分类三组对比：

```bash
python -m hw2.classification.train --config configs/classification.yaml --variant baseline_pretrained --epochs 30 --batch-size 64
python -m hw2.classification.train --config configs/classification.yaml --variant scratch --epochs 30 --batch-size 64
python -m hw2.classification.train --config configs/classification.yaml --variant se_pretrained --epochs 30 --batch-size 64
```

检测训练与 tracking：

```bash
python scripts/prepare_road_vehicle_yaml.py
python -m hw2.detection.train_yolo --config configs/detection.yaml
python -m hw2.detection.track_count --config configs/detection.yaml --weights outputs/detection/train/weights/best.pt --video data/videos/road_vehicle_demo.mp4
```

分割三组损失：

```bash
python -m hw2.segmentation.train --config configs/segmentation.yaml --loss ce
python -m hw2.segmentation.train --config configs/segmentation.yaml --loss dice
python -m hw2.segmentation.train --config configs/segmentation.yaml --loss ce_dice
```

绘制训练曲线：

```bash
python scripts/plot_histories.py --history outputs/classification/history_baseline_pretrained.csv --metrics train_loss val_loss train_acc val_acc --out outputs/figures/classification_baseline_curves.png
python scripts/plot_histories.py --history outputs/segmentation/history_ce.csv --metrics train_loss val_loss train_miou val_miou --out outputs/figures/segmentation_ce_curves.png
```

## 当前已生成的 smoke test 结果

- 分类：`outputs/smoke/classification/summary_scratch.json`
- 分割：`outputs/smoke/segmentation/summary_ce.json`
- 检测：`runs/detect/outputs/smoke/detection/train/results.csv`
- Tracking 视频：`outputs/detection/tracking/tracked_counted.mp4`
