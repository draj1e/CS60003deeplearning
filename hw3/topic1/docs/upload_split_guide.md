# 上传分工清单

## 上传到 GitHub Public Repository

GitHub 放代码、文档、小图和报告，不放大权重/大数据/训练输出。

建议上传：

- `.gitignore`
- `README.md`
- `environment.yml`
- `requirements.txt`
- `scripts/`
- `docs/`
- `reports/hw3_topic1_report.tex`
- `reports/hw3_topic1_report.md`
- `reports/hw3_topic1_report.pdf`
- `reports/wandb_loss_curves.png`
- `reports/wandb_validation_metrics.png`
- `reports/quality_comparison.png`
- `reports/runtime.png`
- `sources/HW3_深度学习与空间智能.pdf_by_PaddleOCR-VL-1.6.md`

不要上传到 GitHub：

- `.cache/`
- `third_party/`
- `data/`
- `outputs/`
- `assets/`
- `logs/`
- `submit/`
- `third_party/threestudio/load/zero123/zero123-xl.ckpt`

这些内容要么很大，要么是第三方仓库/下载缓存/训练产物，放网盘更合适。

## 上传到网盘

直接上传这个压缩包：

- `submit/hw3_topic1_netdisk_core.zip`

校验文件：

- `submit/hw3_topic1_netdisk_core.zip.sha256`

当前 SHA256 以 `submit/hw3_topic1_netdisk_core.zip.sha256` 为准。

该 zip 包包含：

- 报告 PDF、LaTeX/Markdown 报告源文件、README、环境文件、脚本、docs
- 手机输入素材
- A 的 3DGS 点云
- B 的 SDS checkpoint、测试视频、导出点云
- C 的 Zero123 checkpoint、测试视频、导出点云
- 背景 3DGS 点云
- 最终融合视频和预览图
- WandB offline run 和导出图表
- 关键日志

## 可选大文件

如助教要求完整复现，可单独上传：

- `third_party/threestudio/load/zero123/zero123-xl.ckpt`
- `data/tandt_db.zip`
- `data/tandt_db/`

这几个文件体积较大，当前核心网盘包没有包含它们。

## 填报告链接

拿到 GitHub 和网盘链接后：

```bash
cp docs/submission_info.template.json docs/submission_info.json
```

编辑 `docs/submission_info.json`，填写：

- 姓名
- 学号
- 分工
- GitHub Public 仓库链接
- 网盘链接和提取码

然后重新生成报告：

```bash
conda run -n zl2 python scripts/make_report.py
```

最后如果要重新生成网盘 zip，再运行：

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
  reports/wandb_loss_curves.png reports/wandb_validation_metrics.png \
  outputs/wandb/wandb/offline-run-20260624_041113-qk316ps0/run-qk316ps0.wandb \
  logs/3dgs_object_a_train.log logs/3dgs_object_a_render.log \
  logs/threestudio_dreamfusion_train1000_tinysd.log \
  logs/zero123_object_c_train300_xl.log \
  logs/3dgs_background_train2000.log logs/3dgs_background_render2000.log \
  logs/fusion_real_render.log logs/object_b_export_points.log logs/object_c_export_points.log \
  outputs/progress.json
sha256sum submit/hw3_topic1_netdisk_core.zip > submit/hw3_topic1_netdisk_core.zip.sha256
```
