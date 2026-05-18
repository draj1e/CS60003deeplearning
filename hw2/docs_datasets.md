# HW2 数据集下载说明

**一句话总结：三项作业要求的数据集都能从公开网络来源下载，其中 Flowers102 可直接由 torchvision/牛津 VGG 下载，Road Vehicle 与 Stanford Background 可从 Kaggle 下载。**

| 作业任务 | 数据集 | 推荐来源 | 下载方式 | 注意事项 |
| --- | --- | --- | --- | --- |
| 任务 1 | 102 Category Flower Dataset | Oxford VGG / torchvision | `python scripts/download_datasets.py --dataset flowers102` | 官方图片压缩包约 329MB，`torchvision.datasets.Flowers102` 会自动处理 split |
| 任务 2 | Road Vehicle Images Dataset | Kaggle `ashfakyeafi/road-vehicle-images-dataset` | `python scripts/download_datasets.py --dataset road_vehicle` | 已是 YOLOv5 标注结构，需检查 `data_1.yaml` 路径后给 YOLOv8 使用 |
| 任务 3 | Stanford Background Dataset | Kaggle `balraj98/stanford-background-dataset` | `python scripts/download_datasets.py --dataset stanford_background` | Kaggle 包结构可能不是本工程默认 `images/ labels/`，下载后需整理或适配 Dataset |

## 运行命令

```bash
conda activate zl2
cd /mnt/data/zjj/zl/hw2
export PYTHONPATH=$PWD/src
python scripts/download_datasets.py --dataset all
```

默认 Kaggle 数据保留在 KaggleHub 缓存，并在 `data/*.source.txt` 记录真实路径，避免重复占用磁盘。若希望复制到工程目录：

```bash
python scripts/download_datasets.py --dataset all --copy-kaggle
```

## 来源
- Oxford VGG Flowers102: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
- Kaggle Stanford Background Dataset: https://www.kaggle.com/datasets/balraj98/stanford-background-dataset
- Kaggle Road Vehicle Images Dataset: https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset
