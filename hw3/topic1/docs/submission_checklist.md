# 题目一提交核对表

## 结论

当前仓库的题目一技术要求已经由真实运行产物覆盖：物体 A 完成 COLMAP+3DGS，物体 B 完成 threestudio+SDS，物体 C 完成 Zero123，背景完成公开真实多视角数据的 3DGS，并生成了基于真实产物的融合漫游视频。

不能自动完成的提交信息仍需人工填写：报告首页姓名、学号、分工，Public GitHub 链接，以及模型权重/关键产物网盘链接。

## 题目要求逐条核对

| 要求 | 当前状态 | 证据/产物 | 是否严格满足 |
| --- | --- | --- | --- |
| 物体 A：手机视频/多视角照片 | 已使用 `sources/video1.mp4` | `data/object_a/images/` | 是 |
| 物体 A：COLMAP 提取位姿 | 已实际运行 | `data/object_a_colmap_exhaustive/sparse/2`, `logs/colmap_object_a_exhaustive/` | 是 |
| 物体 A：3DGS 重建 | 已实际训练 2000 iterations | `outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply`, `logs/3dgs_object_a_train.log` | 是 |
| 物体 B：threestudio 文本到 3D | 已实际运行 SDS | `outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/ckpts/last.ckpt`, `save/it1000-test.mp4` | 是 |
| 物体 B：导出可融合资产 | 已从真实隐式体场采样 | `assets/object_b/object_b_dreamfusion_train1000_points.ply`, `logs/object_b_export_points.log` | 是 |
| 物体 C：单图去背景/前景 | 已生成 RGBA 前景图 | `data/object_c_zero123/pic1_rgba.png`, `logs/object_c_rgba_prepare.log` | 是 |
| 物体 C：Zero123 单图到 3D | 已实际运行 Zero123 XL | `outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/ckpts/last.ckpt`, `save/it300-test.mp4` | 是 |
| 物体 C：导出可融合资产 | 已从真实隐式体场采样 | `assets/object_c/object_c_zero123_points.ply`, `logs/object_c_export_points.log` | 是 |
| 背景：开源 3D 数据集 | 已使用官方 T&T+DeepBlending 公开数据 | `data/tandt_db/tandt/train` | 是 |
| 背景：3DGS 重建 | 已实际训练 2000 iterations | `outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply`, `logs/3dgs_background_train2000.log` | 是 |
| 三个物体插入背景 | 已用真实产物融合 | `scripts/render_real_fusion.py` | 是 |
| 多视角漫游视频 | 已生成真实融合视频 | `outputs/renders/fusion_real_multisource_render.mp4`，96 帧、24 FPS、4 秒 | 是 |
| 三种方式质量对比 | 已写入真实报告 | `reports/hw3_topic1_report.pdf`, `reports/quality_comparison.png`, `reports/runtime.png` | 是 |
| WandB/SwanLab 图表 | 已使用 WandB offline 记录并导出 | `outputs/wandb/wandb/offline-run-*/run-*.wandb`, `reports/wandb_loss_curves.png`, `reports/wandb_validation_metrics.png` | 是 |
| 表达统一/合并渲染说明 | 已有真实实现依据 | A/背景读取 3DGS PLY，B/C 采样为点云统一渲染 | 是 |
| README 环境和命令 | 已更新为真实链路版 | `README.md` | 是 |
| GitHub Public 仓库 | 尚未上传 | 待用户创建/上传 | 否，需人工 |
| 权重/资产网盘链接 | 尚未上传 | 待用户压缩上传 | 否，需人工 |
| 报告首页姓名学号分工 | 尚未填写 | 待用户提供 | 否，需人工 |

## 提交前人工事项

1. 在报告 PDF 中填写组员姓名、学号、分工。
2. 创建 Public GitHub 仓库并上传代码，填写 GitHub 链接。
3. 打包关键权重和产物，上传网盘并填写链接。
4. 可选：如需在线页面链接，可运行 `wandb sync outputs/wandb/wandb/offline-run-*` 同步离线 run。

## 建议提交关键产物

- `README.md`
- `environment.yml`
- `requirements.txt`
- `scripts/`
- `docs/`
- `sources/`
- `assets/object_b/object_b_dreamfusion_train1000_points.ply`
- `assets/object_c/object_c_zero123_points.ply`
- `outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply`
- `outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply`
- `outputs/renders/fusion_real_multisource_render.mp4`
- `outputs/renders/fusion_real_preview.jpg`
- `reports/wandb_loss_curves.png`
- `reports/wandb_validation_metrics.png`
- `reports/hw3_topic1_report.pdf`
