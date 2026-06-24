# 题目一提交包清单

## 必交文档

- `reports/hw3_topic1_report.pdf`
- `reports/hw3_topic1_report.tex`
- `reports/hw3_topic1_report.md`
- `README.md`
- `environment.yml`
- `requirements.txt`
- `docs/submission_checklist.md`
- `docs/progress.md`
- `docs/submission_info.template.json`
- `docs/github_release_guide.md`
- `docs/upload_split_guide.md`
- `.gitignore`

## 代码

- `scripts/`
- `third_party/gaussian-splatting/`
- `third_party/threestudio/`

## 输入数据

- `sources/video1.mp4`
- `sources/pic1.jpg`
- `sources/HW3_深度学习与空间智能.pdf_by_PaddleOCR-VL-1.6.md`

## 关键训练结果与权重

- `outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply`
- `outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/ckpts/last.ckpt`
- `outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/ckpts/last.ckpt`
- `outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply`

## 可融合资产

- `assets/object_b/object_b_dreamfusion_train1000_points.ply`
- `assets/object_c/object_c_zero123_points.ply`

## 视频和图表

- `outputs/renders/fusion_real_multisource_render.mp4`
- `outputs/renders/fusion_real_preview.jpg`
- `reports/wandb_loss_curves.png`
- `reports/wandb_validation_metrics.png`
- `outputs/wandb/wandb/offline-run-20260624_041113-qk316ps0/run-qk316ps0.wandb`

## 大文件说明

- `third_party/threestudio/load/zero123/zero123-xl.ckpt` 是 Zero123 XL 官方预训练权重，约 15 GB。若网盘容量有限，可以在报告中说明该权重来自官方 Zero123 下载源，并上传本项目训练出的 `last.ckpt`。
- `data/tandt_db.zip` 和 `data/tandt_db/` 是公开背景数据集，可按课程要求决定是否上传；若不上传，应在 README 中保留下载/准备说明。

## 建议打包命令

```bash
zip -qr submit/hw3_topic1_netdisk_core.zip \
  README.md environment.yml requirements.txt .gitignore docs reports scripts sources \
  assets/object_b/object_b_dreamfusion_train1000_points.ply \
  assets/object_c/object_c_zero123_points.ply \
  outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply \
  outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply \
  outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/ckpts/last.ckpt \
  outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/save/it1000-test.mp4 \
  outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/ckpts/last.ckpt \
  outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/save/it300-test.mp4 \
  outputs/renders/fusion_real_multisource_render.mp4 \
  outputs/renders/fusion_real_preview.jpg \
  reports/wandb_loss_curves.png \
  reports/wandb_validation_metrics.png \
  outputs/wandb/wandb/offline-run-20260624_041113-qk316ps0/run-qk316ps0.wandb \
  logs/3dgs_object_a_train.log logs/3dgs_object_a_render.log \
  logs/threestudio_dreamfusion_train1000_tinysd.log \
  logs/zero123_object_c_train300_xl.log \
  logs/3dgs_background_train2000.log logs/3dgs_background_render2000.log \
  logs/fusion_real_render.log logs/object_b_export_points.log logs/object_c_export_points.log \
  outputs/progress.json
sha256sum submit/hw3_topic1_netdisk_core.zip > submit/hw3_topic1_netdisk_core.zip.sha256
```

提交前需要把压缩包上传到网盘，并把链接填入 `reports/hw3_topic1_report.pdf`。

报告首页信息由 `docs/submission_info.json` 控制。使用方式：

```bash
cp docs/submission_info.template.json docs/submission_info.json
# 编辑 docs/submission_info.json
conda run -n zl2 python scripts/make_report.py
```

## 已生成本地提交包

- 路径：`submit/hw3_topic1_netdisk_core.zip`
- 大小：约 201 MB
- 包内文件数：63
- SHA256 文件：`submit/hw3_topic1_netdisk_core.zip.sha256`

该包未包含约 15 GB 的 Zero123 XL 官方预训练权重 `third_party/threestudio/load/zero123/zero123-xl.ckpt`。如助教要求完整复现环境，可单独上传该官方权重或在报告中注明官方来源。
