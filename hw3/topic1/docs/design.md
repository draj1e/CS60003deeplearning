# 题目一严格完成设计文档

## 目标

完成“基于 3DGS 与 AIGC 的多源资产生成与真实场景融合”的可提交工程，包含：

- 物体 A：由手机环绕视频重建的真实物体资产。
- 物体 B：文本 Prompt 生成的 3D 虚拟资产。
- 物体 C：手机单图生成的 3D 资产。
- 背景：开源真实场景的 3DGS 重建结果。
- 融合：统一坐标、比例、材质后输出多视角漫游视频。
- 文档：README、环境说明、实验报告、关键指标与耗时记录。

## 输入

- `sources/video1.mp4`：手机拍摄的环绕视频，用作物体 A。
- `sources/pic1.jpg`：手机拍摄的瑞幸咖啡杯单图，用作物体 C。
- `sources/HW3_深度学习与空间智能.pdf_by_PaddleOCR-VL-1.6.md`：作业要求 OCR 文本。

## 约束与资源

- 使用当前 conda 环境 `zl2`。
- GPU 优先使用空闲的 `CUDA_VISIBLE_DEVICES=1,2,3,4`，单任务默认从 GPU 1 开始。
- 所有第三方代码、数据、输出放在仓库根目录下，并在文档中使用相对路径记录。
- 不删除任何已有文件或目录。
- 任务需要断点续跑：使用 `outputs/progress.json` 记录阶段产物和状态。
- 不使用程序化资产、代理 mesh、旧调试视频冒充题面要求产物；若某阶段失败，只记录失败日志并继续修复，不能标记完成。

## 技术路线

### 1. 环境与工具

使用真实训练和真实数据产物：

- 视频/图像处理：`ffmpeg` 或 conda-forge `ffmpeg`、OpenCV、Pillow。
- 几何处理：`trimesh`、`open3d`、`pymeshlab` 或 Blender Python。
- 渲染融合：Blender 后台渲染，输出 MP4。
- 3DGS 重建：优先使用 `graphdeco-inria/gaussian-splatting` 或兼容实现；若 COLMAP 不可用，安装 conda-forge `colmap`。
- AIGC 资产：
  - B：使用 threestudio 的 DreamFusion/SDS 配置真实优化，产出训练日志、checkpoint、validation/test 渲染和导出 mesh 或可采样资产。
  - C：使用 Zero123/Stable-Zero123 或同源单图到 3D 框架真实运行，产出训练日志、checkpoint、validation/test 渲染和导出 mesh 或可采样资产。

### 2. 物体 A：视频到 3DGS

步骤：

1. 用 `ffmpeg` 从 `video1.mp4` 均匀抽帧到 `data/object_a/images/`。
2. 用 COLMAP 进行特征提取、匹配、稀疏重建，得到相机位姿。
3. 运行 3DGS 训练，输出高斯点云和渲染结果。
4. 导出可融合表示：
   - 首选：3DGS `point_cloud.ply`。
   - 兼容：点云/mesh PLY，导入 Blender 作为高斯点或表面代理。

### 3. 物体 B：文本到 3D

Prompt 初稿：

> a small stylized blue ceramic deer figurine with smooth glossy surface, white antlers, product asset, centered, high detail

步骤：

1. 记录 threestudio/SDS 的环境、配置、启动命令。
2. 真实运行 SDS 优化，至少保留 smoke test、正式训练日志、checkpoint、导出日志。
3. 导出 mesh 或从隐式场采样为点云/高斯可用表示。

### 4. 物体 C：单图到 3D

步骤：

1. 对 `pic1.jpg` 做裁剪和背景移除，得到咖啡杯前景。
2. 调用 Zero123/Stable-Zero123 或同源单图到 3D 实现生成多视角、隐式场或 mesh。
3. 保留权重下载、预处理、训练/推理、导出日志。
4. 导出可融合 mesh/点云/高斯表示和预览图。

### 5. 背景场景

目标为开源 3D 场景背景。优先级：

1. 下载 Mip-NeRF 360 或 Tanks and Temples 等公开真实多视角数据。
2. 使用数据集自带位姿或 COLMAP 位姿训练 3DGS。
3. 输出背景 3DGS `point_cloud.ply`、训练日志和渲染预览。

### 6. 表达统一与融合

统一策略：

- 所有资产最终导出为可共同渲染的 mesh/point cloud/PLY。
- 3DGS 结果保留 PLY 表达；AIGC mesh 通过采样点云或直接 mesh 导入同一 Blender 场景。
- 完成尺度归一化、坐标对齐和多视角相机路径渲染。
- 报告中说明：A 和背景保留 3DGS PLY；B/C 若导出 mesh，则采样为带颜色点云并赋予高斯半径，或在同一渲染器中以 mesh/点云共同渲染。

## 输出结构

- `data/`：抽帧、COLMAP 数据、下载数据。
- `third_party/`：第三方仓库。
- `assets/object_a/`：物体 A 重建资产。
- `assets/object_b/`：文本生成资产。
- `assets/object_c/`：单图生成资产。
- `assets/background/`：背景重建或背景资产。
- `outputs/renders/`：预览图和最终视频。
- `outputs/progress.json`：断点进度。
- `reports/`：实验报告源文件和 PDF。
- `README.md`：项目说明。

## 风险与备选

- COLMAP/3DGS 在手机短视频上可能因模糊、反光和视角不足失败；备选为抽帧 SfM 点云 + Blender 点云代理。
- threestudio/Zero123 依赖重、下载权重大；失败时只记录失败日志和修复动作，不把代理资产计入完成。
- 背景数据下载可能慢；优先选择公开小规模真实多视角数据，必要时降低训练步数，但必须真实训练。
