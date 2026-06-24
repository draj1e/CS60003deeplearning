# 题目一最终完成审计

## 结论

题目一的本地技术链路已经真实完成：COLMAP+3DGS、threestudio+SDS、Zero123、开源背景 3DGS、真实融合视频、WandB offline 图表、报告和本地提交包均有可检查产物。

严格提交仍有三项外部信息不能由本地程序代办：姓名/学号/分工、Public GitHub 仓库链接、网盘下载链接。填写这些信息后，重新生成报告即可完成最终提交材料。

## 逐项要求与证据

| 要求 | 状态 | 证据 |
| --- | --- | --- |
| 手机视频物体 A | 完成 | `sources/video1.mp4`, `data/object_a/images/` |
| A 使用 COLMAP 位姿 | 完成 | `data/object_a_colmap_exhaustive/sparse/2`, `logs/colmap_object_a_exhaustive/` |
| A 使用 3DGS 重建 | 完成 | `outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply`, `logs/3dgs_object_a_train.log` |
| 文本物体 B 使用 threestudio+SDS | 完成 | `outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/ckpts/last.ckpt`, `logs/threestudio_dreamfusion_train1000_tinysd.log` |
| B 可融合资产 | 完成 | `assets/object_b/object_b_dreamfusion_train1000_points.ply`, `logs/object_b_export_points.log` |
| 单图物体 C 前景图 | 完成 | `data/object_c_zero123/pic1_rgba.png`, `logs/object_c_rgba_prepare.log` |
| C 使用 Zero123 | 完成 | `outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/ckpts/last.ckpt`, `logs/zero123_object_c_train300_xl.log` |
| C 可融合资产 | 完成 | `assets/object_c/object_c_zero123_points.ply`, `logs/object_c_export_points.log` |
| 开源背景数据 | 完成 | `data/tandt_db/tandt/train` |
| 背景 3DGS | 完成 | `outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply`, `logs/3dgs_background_train2000.log` |
| 三物体插入背景 | 完成 | `scripts/render_real_fusion.py`, `outputs/renders/fusion_real_preview.jpg` |
| 多视角漫游视频 | 完成 | `outputs/renders/fusion_real_multisource_render.mp4`，96 帧、24 FPS、4 秒 |
| 三种方式质量对比 | 完成 | `reports/hw3_topic1_report.pdf`, `reports/quality_comparison.png`, `reports/runtime.png` |
| 统一表达/合并渲染说明 | 完成 | `reports/hw3_topic1_report.pdf`, `scripts/render_real_fusion.py` |
| WandB/SwanLab 图表 | 完成 | `outputs/wandb/wandb/offline-run-20260624_041113-qk316ps0/run-qk316ps0.wandb`, `reports/wandb_loss_curves.png`, `reports/wandb_validation_metrics.png` |
| README 环境和命令 | 完成 | `README.md` |
| 本地提交包 | 完成 | `submit/hw3_topic1_netdisk_core.zip`, `submit/hw3_topic1_netdisk_core.zip.sha256` |
| GitHub Public 仓库 | 需用户操作 | 创建 Public 仓库并按 `docs/github_release_guide.md` 上传 |
| 网盘链接 | 需用户操作 | 上传 `submit/hw3_topic1_netdisk_core.zip` 后填写 |
| 姓名/学号/分工 | 需用户提供 | 填写 `docs/submission_info.json` 后重跑 `scripts/make_report.py` |

## 当前提交包

- 路径：`submit/hw3_topic1_netdisk_core.zip`
- SHA256 文件：`submit/hw3_topic1_netdisk_core.zip.sha256`

## 最后提交步骤

1. 复制并填写提交信息：

```bash
cp docs/submission_info.template.json docs/submission_info.json
```

2. 填写 `docs/submission_info.json` 中的姓名、学号、分工、GitHub 链接和网盘链接。
3. 重新生成报告：

```bash
conda run -n zl2 python scripts/make_report.py
```

4. 按 `docs/github_release_guide.md` 上传 Public GitHub 仓库。
5. 上传 `submit/hw3_topic1_netdisk_core.zip` 到网盘，并在报告中确认链接有效。
