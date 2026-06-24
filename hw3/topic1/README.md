# HW3 Topic 1: 3DGS 与 AIGC 多源资产融合

本仓库完成题目一的真实运行链路：手机视频物体 A 使用 COLMAP+3DGS 重建，文本物体 B 使用 threestudio+SDS 生成，手机单图物体 C 使用 Zero123 生成，背景使用公开真实多视角数据训练 3DGS，并将四类真实产物统一为点/高斯式表达后生成多视角融合视频。

## 环境

当前验证环境：

- Conda 环境：`zl2`
- Python 3.10
- PyTorch 2.11.0 + CUDA 12.8
- GPU：RTX 5090，默认使用 `CUDA_VISIBLE_DEVICES=1`
- COLMAP 4.0.4
- ffmpeg 8.1.2

依赖记录见：

- `requirements.txt`
- `environment.yml`

## 第三方代码准备

GitHub 仓库不直接提交第三方大仓库。若从零复现训练，请先在仓库根目录准备：

```bash
mkdir -p third_party
git clone https://github.com/graphdeco-inria/gaussian-splatting third_party/gaussian-splatting
git clone https://github.com/threestudio-project/threestudio third_party/threestudio
```

本地验证时，PyTorch 2.6+ 需要将 threestudio 中 Zero123 checkpoint 的 `torch.load` 调用设置为 `weights_only=False`，否则官方 Zero123 XL 权重会因安全默认值变化加载失败。当前工作目录里的 `third_party/threestudio` 已做过该兼容修改；如果重新 clone，需要按报错位置对 `zero123_guidance.py` 和 `stable_zero123_guidance.py` 做同样修改。

## 输入数据

- `sources/video1.mp4`：物体 A 的手机环绕视频。
- `sources/pic1.jpg`：物体 C 的手机单图。
- `sources/HW3_深度学习与空间智能.pdf_by_PaddleOCR-VL-1.6.md`：作业要求 OCR 文本。

## 数据准备

GitHub 仓库只保存代码、报告和小型输入素材。训练产物、checkpoint、3DGS 点云和融合视频通过网盘压缩包提供。

1. 从网盘下载 `hw3_topic1_netdisk_core.zip`。
2. 在仓库根目录解压，保持压缩包内的相对目录结构：

```bash
unzip hw3_topic1_netdisk_core.zip -d .
```

3. 若要从零复现 Zero123 训练，还需要准备官方 Zero123 XL 权重：

```text
third_party/threestudio/load/zero123/zero123-xl.ckpt
```

4. 若要从零复现背景 3DGS 训练，需要准备公开 T&T+DeepBlending 数据：

```text
data/tandt_db/tandt/train
```

当前网盘核心包已经包含本次作业需要提交的训练 checkpoint、点云、视频、WandB offline run 和关键日志，不包含约 15GB 的 Zero123 XL 官方预训练权重。

## 关键真实产物

| 模块 | 方法 | 产物 |
| --- | --- | --- |
| 物体 A | COLMAP + 3DGS | `outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply` |
| 物体 B | threestudio + SDS | `outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/ckpts/last.ckpt` |
| 物体 B 导出 | 隐式体场采样点云 | `assets/object_b/object_b_dreamfusion_train1000_points.ply` |
| 物体 C | Zero123 XL | `outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/ckpts/last.ckpt` |
| 物体 C 导出 | 隐式体场采样点云 | `assets/object_c/object_c_zero123_points.ply` |
| 背景 | 开源 T&T+DeepBlending 数据 + 3DGS | `outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply` |
| 融合视频 | 真实产物统一点渲染 | `outputs/renders/fusion_real_multisource_render.mp4` |

最终视频：`outputs/renders/fusion_real_multisource_render.mp4`，96 帧，24 FPS，4 秒，H.264。

## 复现命令

以下命令均在仓库根目录执行，除 threestudio 命令需进入 `third_party/threestudio`。

### 物体 A

```bash
conda run -n zl2 python scripts/prepare_colmap_object_a.py

cd third_party/gaussian-splatting
CUDA_VISIBLE_DEVICES=1 conda run -n zl2 python train.py \
  -s ../../data/object_a_3dgs_undistorted \
  -m ../../outputs/3dgs_object_a \
  --iterations 2000 --test_iterations 2000 --save_iterations 2000 --quiet
CUDA_VISIBLE_DEVICES=1 conda run -n zl2 python render.py \
  -m ../../outputs/3dgs_object_a --iteration 2000
```

### 物体 B

```bash
cd third_party/threestudio
HF_HOME=../../.cache/huggingface TORCH_HOME=../../.cache/torch XDG_CACHE_HOME=../../.cache CUDA_VISIBLE_DEVICES=1 \
conda run -n zl2 python launch.py \
  --config configs/dreamfusion-sd.yaml --train --gpu 0 \
  exp_root_dir=../../outputs/threestudio name=object_b_dreamfusion tag=train1000_tinysd \
  system.prompt_processor.pretrained_model_name_or_path='segmind/tiny-sd' \
  system.guidance.pretrained_model_name_or_path='segmind/tiny-sd' \
  system.prompt_processor.prompt='a small blue ceramic deer figurine with white antlers, glossy product asset' \
  trainer.max_steps=1000 trainer.val_check_interval=250 checkpoint.every_n_train_steps=250 \
  data.width=64 data.height=64 data.eval_width=128 data.eval_height=128 \
  system.renderer.num_samples_per_ray=64

cd ../..
CUDA_VISIBLE_DEVICES=1 conda run -n zl2 python scripts/export_threestudio_volume_points.py \
  --config outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/configs/parsed.yaml \
  --ckpt outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/ckpts/last.ckpt \
  --out assets/object_b/object_b_dreamfusion_train1000_points.ply \
  --grid 96 --points 50000
```

### 物体 C

```bash
conda run -n zl2 python scripts/prepare_object_c_rgba.py

cd third_party/threestudio
HF_HOME=../../.cache/huggingface TORCH_HOME=../../.cache/torch XDG_CACHE_HOME=../../.cache CUDA_VISIBLE_DEVICES=1 \
conda run -n zl2 python launch.py \
  --config configs/zero123.yaml --train --gpu 0 \
  exp_root_dir=../../outputs/threestudio name=object_c_zero123 tag=train300_xl \
  data.image_path=../../data/object_c_zero123/pic1_rgba.png \
  trainer.max_steps=300 trainer.val_check_interval=100 checkpoint.every_n_train_steps=100 \
  data.height=64 data.width=64 data.random_camera.height=64 data.random_camera.width=64 \
  data.random_camera.batch_size=1 data.random_camera.eval_height=128 data.random_camera.eval_width=128 \
  data.random_camera.n_val_views=8 data.random_camera.n_test_views=24 \
  system.renderer.num_samples_per_ray=64

cd ../..
CUDA_VISIBLE_DEVICES=1 conda run -n zl2 python scripts/export_threestudio_volume_points.py \
  --config outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/configs/parsed.yaml \
  --ckpt outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/ckpts/last.ckpt \
  --out assets/object_c/object_c_zero123_points.ply \
  --grid 96 --points 50000
```

### 背景

```bash
cd third_party/gaussian-splatting
CUDA_VISIBLE_DEVICES=2 conda run -n zl2 python train.py \
  -s ../../data/tandt_db/tandt/train \
  -m ../../outputs/3dgs_background_train \
  --iterations 2000 --test_iterations 2000 --save_iterations 2000 --quiet
CUDA_VISIBLE_DEVICES=2 conda run -n zl2 python render.py \
  -m ../../outputs/3dgs_background_train --iteration 2000
```

### 融合渲染

```bash
conda run -n zl2 python scripts/render_real_fusion.py \
  --frames 96 --width 960 --height 540 \
  --background-points 140000 --object-a-points 45000
```

### WandB 图表与报告

```bash
cp docs/submission_info.template.json docs/submission_info.json
# 编辑 docs/submission_info.json，填写姓名、学号、分工、GitHub 和网盘链接
conda run -n zl2 python scripts/export_wandb_charts.py
conda run -n zl2 python scripts/make_report.py
```

报告源文件为 `reports/hw3_topic1_report.tex` 和 `reports/hw3_topic1_report.md`，PDF 版本为 `reports/hw3_topic1_report.pdf`。

## 日志

- `logs/3dgs_object_a_train.log`
- `logs/threestudio_dreamfusion_train1000_tinysd.log`
- `logs/zero123_object_c_train300_xl.log`
- `logs/3dgs_background_train2000.log`
- `logs/fusion_real_render.log`
- `outputs/wandb/wandb/offline-run-*/run-*.wandb`

## 提交提醒

报告中的姓名、学号、分工、GitHub public repo 链接和网盘链接需要提交前手动填写。GitHub 与网盘分别上传什么见 `docs/upload_split_guide.md`，GitHub 发布建议见 `docs/github_release_guide.md`，提交包清单见 `docs/submission_package_manifest.md`。旧的代理资产和旧视频只作为历史调试记录，不作为最终提交依据。
