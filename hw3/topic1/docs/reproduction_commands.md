# 题目一真实复现命令记录

本文档只记录最终题目一采用的真实路线。早期程序化代理资产和旧融合视频不计入最终提交。

## 1. 数据准备

```bash
unzip hw3_topic1_netdisk_core.zip -d .
```

若需要从零复现训练，还需要额外准备：

- Zero123 XL 官方预训练权重：`third_party/threestudio/load/zero123/zero123-xl.ckpt`
- 公开背景数据：`data/tandt_db/tandt/train`

## 2. 物体 A：手机视频 + COLMAP + 3DGS

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

关键产物：

- `outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply`
- `logs/3dgs_object_a_train.log`

## 3. 物体 B：threestudio + SDS

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

关键产物：

- `outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/ckpts/last.ckpt`
- `assets/object_b/object_b_dreamfusion_train1000_points.ply`

## 4. 物体 C：手机单图 + Zero123 XL

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

关键产物：

- `outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/ckpts/last.ckpt`
- `assets/object_c/object_c_zero123_points.ply`

## 5. 背景：公开真实数据 + 3DGS

```bash
cd third_party/gaussian-splatting
CUDA_VISIBLE_DEVICES=2 conda run -n zl2 python train.py \
  -s ../../data/tandt_db/tandt/train \
  -m ../../outputs/3dgs_background_train \
  --iterations 2000 --test_iterations 2000 --save_iterations 2000 --quiet

CUDA_VISIBLE_DEVICES=2 conda run -n zl2 python render.py \
  -m ../../outputs/3dgs_background_train --iteration 2000
```

关键产物：

- `outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply`
- `logs/3dgs_background_train2000.log`

## 6. 融合视频与图表

```bash
conda run -n zl2 python scripts/render_real_fusion.py \
  --frames 96 --width 960 --height 540 \
  --background-points 140000 --object-a-points 45000

conda run -n zl2 python scripts/export_wandb_charts.py
conda run -n zl2 python scripts/make_report.py
```

关键产物：

- `outputs/renders/fusion_real_multisource_render.mp4`
- `outputs/renders/fusion_real_preview.jpg`
- `reports/wandb_loss_curves.png`
- `reports/wandb_validation_metrics.png`
- `reports/hw3_topic1_report.pdf`
- `reports/hw3_topic1_report.md`
