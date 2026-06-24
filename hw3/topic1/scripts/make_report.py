#!/usr/bin/env python3
"""Generate charts and a PDF report for HW3 topic 1 real outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RENDERS = ROOT / "outputs" / "renders"
PROGRESS_JSON = ROOT / "outputs" / "progress.json"
B_METRICS = ROOT / "outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/csv_logs/version_0/metrics.csv"
C_METRICS = ROOT / "outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/csv_logs/version_0/metrics.csv"
SUBMISSION_INFO = ROOT / "docs" / "submission_info.json"
SUBMISSION_INFO_TEMPLATE = ROOT / "docs" / "submission_info.template.json"


def register_font() -> str:
    candidates = [
        "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
        "/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont("ReportFont", path))
                return "ReportFont"
            except Exception:
                continue
    return "Helvetica"


def read_metric(path: Path, metric: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    if not path.exists():
        return steps, values
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("step") or not row.get(metric):
                continue
            try:
                steps.append(int(float(row["step"])))
                values.append(float(row[metric]))
            except ValueError:
                continue
    return steps, values


def load_submission_info() -> dict:
    path = SUBMISSION_INFO if SUBMISSION_INFO.exists() else SUBMISSION_INFO_TEMPLATE
    if path.exists():
        return json.loads(path.read_text())
    return {
        "members": [
            {"name": "杨瑞欣", "student_id": "2521098012", "role": "题目一：3DGS 与 AIGC 多源资产生成、融合实验与报告整理"},
            {"name": "朱家杰", "student_id": "25210980147", "role": "题目一：实验复现、资产融合与报告整理"},
        ],
        "github_repo": "https://github.com/draj1e/CS60003deeplearning",
        "model_weights_url": "https://pan.baidu.com/s/1ECIzurYlQhJvwKDFMDASEA?pwd=6666",
        "model_weights_note": "百度网盘提取码：6666",
    }


def make_charts() -> dict[str, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    charts: dict[str, Path] = {}

    b_steps, b_sds = read_metric(B_METRICS, "train/loss_sds")
    c_steps, c_loss = read_metric(C_METRICS, "train/loss")
    plt.figure(figsize=(7.2, 4.2))
    if b_steps:
        plt.plot(b_steps, b_sds, linewidth=1.1, label="B SDS loss")
    if c_steps:
        plt.plot(c_steps, c_loss, linewidth=1.1, label="C Zero123 total loss")
    plt.xlabel("Step")
    plt.ylabel("Logged loss")
    plt.title("Local CSV Training Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    charts["curves"] = REPORTS / "loss_curves.png"
    plt.tight_layout()
    plt.savefig(charts["curves"], dpi=180)
    plt.close()

    methods = ["A COLMAP+3DGS", "B SDS", "C Zero123"]
    geom = [4.0, 2.5, 3.0]
    tex = [3.5, 2.5, 3.0]
    runtime = [35, 30, 4]
    x = range(len(methods))

    plt.figure(figsize=(7.2, 4.2))
    plt.bar([i - 0.18 for i in x], geom, width=0.36, label="Geometry")
    plt.bar([i + 0.18 for i in x], tex, width=0.36, label="Texture")
    plt.xticks(list(x), methods, rotation=8)
    plt.ylabel("Qualitative score (1-5)")
    plt.ylim(0, 5)
    plt.title("Method Quality Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    charts["quality"] = REPORTS / "quality_comparison.png"
    plt.tight_layout()
    plt.savefig(charts["quality"], dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.2))
    plt.bar(methods, runtime, color=["#4062bb", "#3f8f57", "#c06c35"])
    plt.ylabel("Approx. runtime (minutes)")
    plt.title("Measured Local Runtime")
    plt.grid(axis="y", alpha=0.25)
    charts["runtime"] = REPORTS / "runtime.png"
    plt.tight_layout()
    plt.savefig(charts["runtime"], dpi=180)
    plt.close()
    for name in ["wandb_loss_curves", "wandb_validation_metrics"]:
        path = REPORTS / f"{name}.png"
        if path.exists():
            charts[name] = path
    return charts


def table(data: list[list[str]], font: str, widths: list[float], font_size: float = 8.0) -> Table:
    cell_style = ParagraphStyle("table_cell", fontName=font, fontSize=font_size, leading=font_size + 2)
    wrapped = [[Paragraph(str(cell), cell_style) for cell in row] for row in data]
    return Table(
        wrapped,
        colWidths=[w * cm for w in widths],
        style=[
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ],
    )


def update_progress(report_pdf: Path) -> None:
    data = json.loads(PROGRESS_JSON.read_text()) if PROGRESS_JSON.exists() else {"stages": {}}
    stage = data.setdefault("stages", {}).setdefault("report", {})
    stage["status"] = "done"
    stage["artifacts"] = [
        str(report_pdf.relative_to(ROOT)),
        "reports/wandb_loss_curves.png",
        "reports/wandb_validation_metrics.png",
        "reports/loss_curves.png",
        "reports/quality_comparison.png",
        "reports/runtime.png",
        "docs/submission_checklist.md",
        "docs/submission_info.template.json",
    ]
    PROGRESS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def build_pdf(charts: dict[str, Path]) -> Path:
    info = load_submission_info()
    github_repo = info.get("github_repo", "https://github.com/draj1e/CS60003deeplearning")
    model_weights_url = info.get("model_weights_url", "https://pan.baidu.com/s/1ECIzurYlQhJvwKDFMDASEA?pwd=6666")
    model_weights_note = info.get("model_weights_note", "")

    font = register_font()
    styles = getSampleStyleSheet()
    normal = ParagraphStyle("normal_cn", parent=styles["Normal"], fontName=font, fontSize=9.5, leading=13)
    h1 = ParagraphStyle("h1_cn", parent=styles["Heading1"], fontName=font, fontSize=17, leading=22, spaceAfter=8)
    h2 = ParagraphStyle("h2_cn", parent=styles["Heading2"], fontName=font, fontSize=13, leading=17, spaceBefore=8, spaceAfter=5)
    small = ParagraphStyle("small_cn", parent=normal, fontSize=8, leading=10)

    pdf = REPORTS / "hw3_topic1_report.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm)
    story = []

    story.append(Paragraph("HW3 题目一：基于 3DGS 与 AIGC 的多源资产生成与真实场景融合", h1))
    story.append(Paragraph("杨瑞欣 2521098012", normal))
    story.append(Paragraph("朱家杰 25210980147", normal))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(f"GitHub 仓库链接：{github_repo}", normal))
    story.append(Paragraph(f"模型/资产权重网盘：{model_weights_url}；{model_weights_note}", normal))
    story.append(Spacer(1, 0.2 * cm))

    quick = [
        ["模块", "输入/方法", "关键输出", "状态"],
        ["物体 A", "手机视频 + COLMAP + 3DGS", "34,576 高斯", "完成"],
        ["物体 B", "文本 Prompt + threestudio SDS", "1000-step ckpt + 50,000 点云", "完成"],
        ["物体 C", "手机单图 RGBA + Zero123 XL", "300-step ckpt + 50,000 点云", "完成"],
        ["背景", "T&T+DeepBlending train + 3DGS", "326,586 高斯", "完成"],
        ["融合", "真实产物统一点渲染", "96 帧 MP4", "完成"],
    ]
    story.append(table(quick, font, [2.4, 4.5, 5.2, 1.8]))

    story.append(Paragraph("1. 任务背景", h2))
    story.append(Paragraph("题目要求从真实多视角、文本 Prompt、单张真实照片三种来源准备 3D 资产，并把它们插入开源真实场景 3DGS 背景中。本实验目标是比较三种资产生成方式的几何质量、纹理一致性和融合可用性，并验证不同来源资产在统一点/高斯式表达下的真实场景融合效果。", normal))

    story.append(Paragraph("2. 数据集描述", h2))
    dataset_rows = [
        ["数据", "来源", "用途", "路径"],
        ["物体 A 视频", "手机拍摄视频", "COLMAP 位姿估计与 3DGS 重建", "sources/video1.mp4"],
        ["物体 C 单图", "手机拍摄图片", "Zero123 XL 单图到 3D", "sources/pic1.jpg"],
        ["背景场景", "T&T+DeepBlending 公开数据", "背景 3DGS 训练", "data/tandt_db/tandt/train"],
        ["作业要求", "OCR Markdown", "需求核对", "sources/HW3_深度学习与空间智能.pdf_by_PaddleOCR-VL-1.6.md"],
    ]
    story.append(table(dataset_rows, font, [2.4, 3.0, 4.5, 4.5], font_size=7.2))

    story.append(Paragraph("3. 方法原理与实现", h2))
    method_rows = [
        ["模块", "实现细节", "日志/证据"],
        ["A", "80 帧手机视频抽帧，COLMAP exhaustive matcher，最佳模型注册 54/80 张图；官方 3DGS 训练 2000 iter。", "logs/colmap_object_a_exhaustive/, logs/3dgs_object_a_train.log"],
        ["B", "threestudio DreamFusion/SDS；原 SD2.1 因 401 不可用，改用公开 `segmind/tiny-sd`；训练到 1000 step。", "logs/threestudio_dreamfusion_*.log"],
        ["C", "单图裁剪为 RGBA，Zero123 XL 训练 300 step；PyTorch 2.6+ checkpoint 加载做兼容 patch。", "logs/zero123_object_c_train300_xl.log"],
        ["背景", "下载官方公开 T&T+DeepBlending COLMAP 数据，官方 3DGS 训练 2000 iter。", "logs/3dgs_background_train2000.log"],
        ["融合", "A/背景读取 3DGS PLY 的 SH DC 颜色，B/C 从隐式体场采样为 RGB 点云，统一归一化、缩放、插入背景并点渲染。", "scripts/render_real_fusion.py"],
    ]
    story.append(table(method_rows, font, [1.2, 7.4, 5.2], font_size=7.2))

    story.append(Paragraph("4. 实验设置、超参数与指标", h2))
    hyper = [
        ["模块", "Network Architecture / 方法", "Batch Size", "Learning Rate", "Optimizer", "Epochs / Steps", "Loss Function"],
        ["物体 A", "官方 3DGS", "官方默认", "官方默认", "Adam", "2000 iters", "L1 + SSIM 组合"],
        ["物体 B", "threestudio DreamFusion/SDS + tiny SD", "随机相机默认配置", "threestudio 默认", "Adam/Lightning 默认", "1000 steps", "SDS loss"],
        ["物体 C", "threestudio Zero123 XL", "1", "threestudio 默认", "Adam/Lightning 默认", "300 steps", "Zero123 guidance loss"],
        ["背景", "官方 3DGS", "官方默认", "官方默认", "Adam", "2000 iters", "L1 + SSIM 组合"],
        ["融合", "点/高斯统一渲染", "N/A", "N/A", "N/A", "96 frames", "N/A"],
    ]
    story.append(table(hyper, font, [1.4, 3.2, 2.0, 1.8, 2.2, 1.8, 2.4], font_size=6.3))
    metrics = [
        ["模块", "关键评价指标", "具体取值"],
        ["物体 A COLMAP", "注册图像 / 稀疏点 / 重投影误差", "54/80 images, 2371 points, 0.754620 px"],
        ["物体 A 3DGS", "高斯数量 / Train L1 / PSNR", "34,576, 0.019615, 27.6767"],
        ["物体 B SDS", "训练 checkpoint / 导出点云", "1000 steps, 50,000 points"],
        ["物体 C Zero123", "训练 checkpoint / 导出点云", "300 steps, 50,000 points"],
        ["背景 3DGS", "高斯数量 / 渲染 train views", "326,586, 301 views"],
        ["融合视频", "分辨率 / 帧数 / 帧率 / 时长", "960x544, 96 frames, 24 FPS, 4 seconds"],
    ]
    story.append(Spacer(1, 0.12 * cm))
    story.append(table(metrics, font, [3.0, 5.2, 5.8], font_size=7.2))

    story.append(Paragraph("5. 实验结果展示", h2))
    preview = RENDERS / "fusion_real_preview.jpg"
    if preview.exists():
        story.append(Image(str(preview), width=15 * cm, height=8.45 * cm))
        story.append(Paragraph("图 1：真实 A/B/C/背景产物融合渲染预览。", small))
    if "wandb_loss_curves" in charts:
        story.append(Image(str(charts["wandb_loss_curves"]), width=14.5 * cm, height=8.4 * cm))
        story.append(Paragraph("图 2：WandB offline run 导出的训练 Loss 曲线。", small))
    if "wandb_validation_metrics" in charts:
        story.append(Image(str(charts["wandb_validation_metrics"]), width=14.5 * cm, height=8.4 * cm))
        story.append(Paragraph("图 3：WandB offline run 导出的验证/诊断指标曲线。", small))
    story.append(Image(str(charts["curves"]), width=14.5 * cm, height=8.4 * cm))
    story.append(Paragraph("图 4：本地 CSV/TensorBoard 日志复核曲线。", small))
    story.append(Image(str(charts["quality"]), width=14.5 * cm, height=8.4 * cm))
    story.append(Image(str(charts["runtime"]), width=14.5 * cm, height=8.4 * cm))

    story.append(Paragraph("6. 深度现象分析与三种资产生成方式对比", h2))
    comp = [
        ["方式", "几何准确度", "纹理细节", "计算耗时", "现象分析"],
        ["多视角重建", "最好，几何来自真实视角和 COLMAP 位姿", "受手机模糊、反光和 3DGS 步数影响", "中等", "真实性最高，但对拍摄质量和位姿估计敏感"],
        ["文本到 3D", "语义形状可控但细节不稳定", "受公开 tiny SD 模型能力限制", "较长", "SDS 可从文本生成资产，但容易过平滑或漂浮"],
        ["单图到 3D", "正面一致性较好，背面依赖 Zero123 先验", "保留单图主视角外观", "较短", "单图信息不足，补全区域可信度低于多视角"],
    ]
    story.append(table(comp, font, [2.2, 3.3, 3.0, 2.0, 3.5]))

    story.append(Paragraph("7. 表达统一与合并渲染", h2))
    story.append(Paragraph("A 和背景保留官方 3DGS 输出的 `point_cloud.ply`，颜色从 SH 的 `f_dc_0/1/2` 恢复；B 和 C 的 threestudio/Zero123 mesh exporter 在本环境中可能导出空 mesh，因此采用真实 checkpoint 的隐式密度场采样，导出带 RGB 的 PLY。融合阶段把四个点集归一化到统一尺度，设置相对位置后用同一相机轨迹点渲染，避免用旧程序化资产冒充。", normal))

    story.append(Paragraph("8. 结论", h2))
    story.append(Paragraph("本实验完成了题目一要求的多源资产生成与真实场景融合流程。多视角视频重建在几何可信度上最好，文本到 3D 具有开放类别生成能力但细节依赖扩散模型能力，单图到 3D 输入成本最低但背面补全依赖先验。统一点/高斯式表达后，三类资产可以与真实背景 3DGS 共同渲染，形成多视角漫游视频。", normal))

    story.append(Paragraph("9. 外部链接", h2))
    links = [
        ["项目", "链接"],
        ["GitHub Public Repository", github_repo],
        ["模型权重与关键产物网盘", model_weights_url],
        ["提取码", model_weights_note.replace("百度网盘提取码：", "") or "6666"],
    ]
    story.append(table(links, font, [4.0, 10.0], font_size=8.0))

    story.append(PageBreak())
    story.append(Paragraph("附录：关键产物路径", h2))
    outputs = [
        "outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply",
        "outputs/threestudio/object_b_dreamfusion/train1000_tinysd@20260624-032851/ckpts/last.ckpt",
        "assets/object_b/object_b_dreamfusion_train1000_points.ply",
        "outputs/threestudio/object_c_zero123/train300_xl@20260624-035923/ckpts/last.ckpt",
        "assets/object_c/object_c_zero123_points.ply",
        "outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply",
        "outputs/renders/fusion_real_multisource_render.mp4",
    ]
    for item in outputs:
        story.append(Paragraph(item, normal))

    doc.build(story)
    return pdf


def main() -> None:
    charts = make_charts()
    pdf = build_pdf(charts)
    update_progress(pdf)
    print(pdf.relative_to(ROOT))


if __name__ == "__main__":
    main()
