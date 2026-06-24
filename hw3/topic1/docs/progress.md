# 题目一进度文档

## 当前状态

- 开始时间：2026-06-23
- 工作目录：仓库根目录
- 当前阶段：严格产物已完成；剩余提交外链和个人信息需人工填写
- GPU 策略：优先使用 GPU 1-4，默认 `CUDA_VISIBLE_DEVICES=1`

## 阶段清单

| 阶段 | 状态 | 产物 |
| --- | --- | --- |
| 需求解析 | done | `sources/HW3_深度学习与空间智能.pdf_by_PaddleOCR-VL-1.6.md` |
| 设计文档 | done | `docs/design.md` |
| 环境检查 | done | `requirements.txt`, `environment.yml` |
| 视频抽帧 | done | `data/object_a/images/` |
| 物体 A 重建 | done | `outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply` |
| 物体 B 生成 | done | `assets/object_b/object_b_dreamfusion_train1000_points.ply` |
| 物体 C 生成 | done | `assets/object_c/object_c_zero123_points.ply` |
| 背景场景 | done | `outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply` |
| 融合渲染 | done | `outputs/renders/fusion_real_multisource_render.mp4` |
| 报告与 README | done | `README.md`, `reports/hw3_topic1_report.pdf`, `reports/hw3_topic1_report.tex`, `reports/hw3_topic1_report.md` |

## 环境记录

- Python：3.10.x
- PyTorch：2.11.0+cu128
- CUDA：12.8，5 张 RTX 5090 可见
- ffmpeg：已安装并可用
- COLMAP：已安装并可用，版本 4.0.4
- Blender：未发现系统命令；最终融合优先使用本地 Python/可用渲染工具真实读取训练产物

## 决策记录

- 所有第三方代码、数据、输出均放在仓库根目录下。
- 不执行删除操作。
- 不使用程序化资产或旧代理资产冒充完成；失败阶段只记录日志并继续修复。

## 旧代理产物

以下产物只作为 baseline 或工程调试参考，不能作为严格题面完成结果：

- `assets/object_a/object_a_multiview_cup.glb`
- `assets/object_a/object_a_multiview_point_cloud.ply`
- `assets/object_b/object_b_text_to_3d_deer.glb`
- `assets/object_c/object_c_single_image_cup.glb`
- `assets/fusion_scene.glb`
- `outputs/renders/fusion_splat_multiview_render.mp4`
- `reports/hw3_topic1_report.pdf`

## 严格产物目标

- 物体 A：`outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply`，来源于 COLMAP 位姿 + 3DGS 训练。
- 物体 B：`outputs/threestudio_text_b/`，来源于 threestudio + SDS。
- 物体 C：`outputs/zero123_object_c/`，来源于 Zero123 或明确同源的单图到 3D 框架。
- 背景：`outputs/3dgs_background/.../point_cloud.ply`，来源于开源 3D 数据集 + 3DGS。
- 融合视频：使用以上真实产物重新生成。

## 严格补跑记录

- 2026-06-24：安装并验证 `colmap 4.0.4`、`ffmpeg 8.1.2`。
- 2026-06-24：`data/object_a_colmap_exhaustive` 使用 80 帧、exhaustive matcher 真实重建，最佳模型 `sparse/2` 注册 54/80 张图、2371 个稀疏点、平均重投影误差 0.754620 px。
- 2026-06-24：`data/object_a_3dgs_undistorted` 由 COLMAP `image_undistorter` 生成无畸变数据。
- 2026-06-24：官方 `graphdeco-inria/gaussian-splatting` 训练 2000 iterations，输出 34,576 个高斯，train L1 0.019615，PSNR 27.6767。
- 2026-06-24：threestudio DreamFusion 原始 `stabilityai/stable-diffusion-2-1-base` 因 Hugging Face 401 授权失败，日志为 `logs/threestudio_dreamfusion_smoke_retry.log`。
- 2026-06-24：验证公开 SD 兼容模型 `segmind/tiny-sd` 可加载，日志为 `logs/hf_probe_segmind_tiny_sd.log`。
- 2026-06-24：物体 B 使用 threestudio + SDS + `segmind/tiny-sd` 完成 1-step smoke test，输出 checkpoint 与 `outputs/threestudio/object_b_dreamfusion/smoke_tinysd@20260624-032434/save/it1-test.mp4`。
- 2026-06-24：物体 B 使用 threestudio + SDS + `segmind/tiny-sd` 完成 300-step 正式短训练，输出 `outputs/threestudio/object_b_dreamfusion/train300_tinysd@20260624-032529/ckpts/last.ckpt` 与 `save/it300-test.mp4`。
- 2026-06-24：物体 B 默认阈值 mesh 导出失败，原因是 300-step 密度场默认等值面为空；日志为 `logs/threestudio_dreamfusion_export_train300_tinysd.log`。
- 2026-06-24：物体 B 低阈值导出命令完成但 OBJ 为 0 顶点/0 面，不计入完成；日志为 `logs/threestudio_dreamfusion_export_train300_lowthr.log`。继续从 300-step checkpoint 续训。
- 2026-06-24：物体 B 从 300-step checkpoint 续训到 1000 steps，输出 `outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/ckpts/last.ckpt` 与 `save/it1000-test.mp4`，正在重新导出 mesh。
- 2026-06-24：物体 B 1000-step mesh 导出仍为 0 顶点/0 面，不计入 mesh 完成；日志为 `logs/threestudio_dreamfusion_export_train1000_tinysd.log`。
- 2026-06-24：物体 B 改用真实隐式体场采样路线，从 1000-step checkpoint 采样密度最高点并导出带颜色点云 `assets/object_b/object_b_dreamfusion_train1000_points.ply`，50,000 点，日志为 `logs/object_b_export_points.log`。
- 2026-06-24：物体 C 由手机单图生成 Zero123 输入 `data/object_c_zero123/pic1_rgba.png`，512x512，前景比例 0.4116。
- 2026-06-24：下载 3DGS 官方公开 T&T+DeepBlending COLMAP 数据 `data/tandt_db.zip`，解压到 `data/tandt_db/`。
- 2026-06-24：背景使用 `data/tandt_db/tandt/train` 真实多视角 COLMAP 数据训练官方 3DGS 2000 iterations，输出 `outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply`，日志为 `logs/3dgs_background_train2000.log`。
- 2026-06-24：背景 3DGS 渲染完成，`outputs/3dgs_background_train/train/ours_2000/renders/` 与 `gt/` 各 301 张图，日志为 `logs/3dgs_background_render2000.log`。
- 2026-06-24：新增真实融合脚本 `scripts/render_real_fusion.py`，读取 A/背景官方 3DGS PLY 与 B/C 真实导出点云，不再读取旧代理资产。
- 2026-06-24：Zero123 XL 权重 `third_party/threestudio/load/zero123/zero123-xl.ckpt` 断点下载完成，大小 15,465,973,531 bytes，日志为 `logs/zero123_xl_download.log`。
- 2026-06-24：物体 C 使用 Zero123 XL 完成 1-step smoke test，输出 `outputs/threestudio/object_c_zero123/smoke_xl_patch@20260624-035746/ckpts/last.ckpt` 与 `save/it1-test.mp4`。
- 2026-06-24：为兼容 PyTorch 2.6+，将 threestudio 的 Zero123/Stable-Zero123 官方 checkpoint 加载改为 `weights_only=False`，改动位于 `third_party/threestudio/threestudio/models/guidance/zero123_guidance.py` 与 `stable_zero123_guidance.py`。
- 2026-06-24：物体 C 使用手机单图 RGBA 输入和 Zero123 XL 完成 300-step 真实训练，输出 `outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/ckpts/last.ckpt`、`save/it300-test.mp4` 和 metrics CSV，日志为 `logs/zero123_object_c_train300_xl.log`。
- 2026-06-24：物体 C 从 300-step checkpoint 的隐式体场真实采样并导出带颜色点云 `assets/object_c/object_c_zero123_points.ply`，50,000 点，日志为 `logs/object_c_export_points.log`。
- 2026-06-24：真实融合渲染完成，输出 `outputs/renders/fusion_real_multisource_render.mp4` 与 `outputs/renders/fusion_real_preview.jpg`，视频 96 帧、24 FPS、4 秒，日志为 `logs/fusion_real_render.log`。
- 2026-06-24：使用 WandB offline 从真实训练 metrics 记录并导出图表，输出 `outputs/wandb/wandb/offline-run-20260624_041113-qk316ps0/run-qk316ps0.wandb`、`reports/wandb_loss_curves.png` 与 `reports/wandb_validation_metrics.png`。
- 2026-06-24：更新真实版 README、提交核对表和 PDF 报告 `reports/hw3_topic1_report.pdf`。
- 2026-06-24：新增题目一 Markdown 报告源文件 `reports/hw3_topic1_report.md`，并将正式网盘提交包统一为 `submit/hw3_topic1_netdisk_core.zip`。
- 2026-06-24：补充 README 中的第三方代码准备说明，明确 GitHub 不上传 `third_party/` 时的复现前置步骤。
- 2026-06-24：按正式实验报告要求新增双栏 LaTeX 报告源 `reports/hw3_topic1_report.tex`，加入中文作者信息、外部链接和参考文献。

## 验证记录

- 物体 A 真实 3DGS：`outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply`，34,576 高斯。
- 物体 B 真实 SDS 隐式体采样：`assets/object_b/object_b_dreamfusion_train1000_points.ply`，50,000 点。
- 物体 C 真实 Zero123 隐式体采样：`assets/object_c/object_c_zero123_points.ply`，50,000 点。
- 背景真实 3DGS：`outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply`，326,586 高斯。
- 真实融合视频：`outputs/renders/fusion_real_multisource_render.mp4`，H.264，960x544，96 帧，24 FPS，4 秒。
- WandB offline 图表：`reports/wandb_loss_curves.png`、`reports/wandb_validation_metrics.png`。
- 旧视频 `outputs/renders/fusion_splat_multiview_render.mp4` 只作为历史调试，不计入严格提交。

## 仍需人工填写

- 报告首页的姓名、学号和分工。
- 上传 GitHub 后填写仓库链接。
- 上传资产/权重压缩包到网盘后填写下载链接。

## 严格题面缺口

- GitHub public repo 链接、模型权重网盘链接、姓名学号分工仍需人工填写。
