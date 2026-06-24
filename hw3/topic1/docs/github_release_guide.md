# GitHub 发布说明

## 建议提交到 Public GitHub 的内容

- `README.md`
- `environment.yml`
- `requirements.txt`
- `scripts/`
- `docs/`
- `reports/hw3_topic1_report.md`
- `reports/hw3_topic1_report.pdf`
- `reports/wandb_loss_curves.png`
- `reports/wandb_validation_metrics.png`
- `reports/quality_comparison.png`
- `reports/runtime.png`
- `sources/HW3_深度学习与空间智能.pdf_by_PaddleOCR-VL-1.6.md`

## 不建议直接提交到 GitHub 的内容

- `.cache/`
- `third_party/`
- `data/`
- `outputs/`
- `assets/`
- `logs/`
- `submit/`
- `third_party/threestudio/load/zero123/zero123-xl.ckpt`

这些文件包含第三方仓库、大数据集、训练权重、渲染输出或本地缓存，应通过网盘链接提交。

## 初始化仓库示例

```bash
git init
git add README.md environment.yml requirements.txt scripts docs reports .gitignore sources/HW3_深度学习与空间智能.pdf_by_PaddleOCR-VL-1.6.md
git status
git commit -m "HW3 topic1 real 3DGS AIGC fusion"
git branch -M main
git remote add origin <your-public-github-repo-url>
git push -u origin main
```

## 外部产物

将 `submit/hw3_topic1_netdisk_core.zip` 上传到网盘，并把链接填写到 `docs/submission_info.json` 后重新生成报告：

```bash
cp docs/submission_info.template.json docs/submission_info.json
# 编辑 docs/submission_info.json
conda run -n zl2 python scripts/make_report.py
```
