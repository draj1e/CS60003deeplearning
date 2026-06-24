from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    src = Path("sources/pic1.jpg")
    out_dir = Path("data/object_c_zero123")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pic1_rgba.png"
    preview_path = out_dir / "pic1_rgba_preview.jpg"

    image_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(src)

    h, w = image_bgr.shape[:2]
    scale = 1024.0 / max(h, w)
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        h, w = image_bgr.shape[:2]

    margin_x = max(10, int(w * 0.08))
    margin_y = max(10, int(h * 0.08))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image_bgr, mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)

    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
        np.uint8
    )
    kernel = np.ones((5, 5), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        alpha = np.where(labels == largest, 255, 0).astype(np.uint8)

    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        raise RuntimeError("foreground segmentation produced an empty mask")

    pad = 32
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    crop_bgr = image_bgr[y0:y1, x0:x1]
    crop_alpha = alpha[y0:y1, x0:x1]

    side = max(crop_bgr.shape[:2])
    canvas_bgr = np.full((side, side, 3), 255, np.uint8)
    canvas_alpha = np.zeros((side, side), np.uint8)
    oy = (side - crop_bgr.shape[0]) // 2
    ox = (side - crop_bgr.shape[1]) // 2
    canvas_bgr[oy : oy + crop_bgr.shape[0], ox : ox + crop_bgr.shape[1]] = crop_bgr
    canvas_alpha[oy : oy + crop_alpha.shape[0], ox : ox + crop_alpha.shape[1]] = (
        crop_alpha
    )

    rgba = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGBA)
    rgba[..., 3] = canvas_alpha
    rgba = cv2.resize(rgba, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    preview_rgb = rgba[..., :3].copy()
    preview_rgb[rgba[..., 3] == 0] = 255
    cv2.imwrite(str(preview_path), cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))
    print(out_path)
    print(preview_path)


if __name__ == "__main__":
    main()
