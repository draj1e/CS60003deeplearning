#!/usr/bin/env python3
"""Render the final fusion video from real training outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from plyfile import PlyData


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RENDERS = OUTPUTS / "renders"
PROGRESS_JSON = OUTPUTS / "progress.json"
SH_C0 = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def read_ascii_rgb_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    v = ply["vertex"].data
    points = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    names = v.dtype.names or ()
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 190, dtype=np.uint8)
    return points, colors


def read_3dgs_ply(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ply = PlyData.read(path)
    v = ply["vertex"].data
    n = len(v)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
    else:
        idx = np.arange(n)
    points = np.column_stack([v["x"][idx], v["y"][idx], v["z"][idx]]).astype(np.float32)
    names = v.dtype.names or ()
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh = np.column_stack([v["f_dc_0"][idx], v["f_dc_1"][idx], v["f_dc_2"][idx]]).astype(np.float32)
        colors = np.clip((sh * SH_C0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    elif {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack([v["red"][idx], v["green"][idx], v["blue"][idx]]).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 190, dtype=np.uint8)
    return points, colors


def normalize(points: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(points, [2, 98], axis=0)
    center = (lo + hi) * 0.5
    scale = np.max(hi - lo)
    return (points - center) / max(float(scale), 1e-6)


def transform(
    points: np.ndarray,
    scale: float,
    translation: tuple[float, float, float],
    rot_y: float = 0.0,
) -> np.ndarray:
    c, s = math.cos(rot_y), math.sin(rot_y)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return points @ rot.T * scale + np.asarray(translation, dtype=np.float32)


def look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    up0 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(fwd, up0)
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def render_points(
    points: np.ndarray,
    colors: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    img = Image.new("RGB", (width, height), (220, 222, 218))
    draw = ImageDraw.Draw(img, "RGBA")
    right, up, fwd = look_at(eye, target)
    rel = points - eye[None, :]
    cam_x = rel @ right
    cam_y = rel @ up
    cam_z = rel @ fwd
    valid = cam_z > 0.08
    focal = 0.95 * width
    sx = width * 0.5 + focal * cam_x[valid] / cam_z[valid]
    sy = height * 0.5 - focal * cam_y[valid] / cam_z[valid]
    z = cam_z[valid]
    col = colors[valid]
    on = (sx > -24) & (sx < width + 24) & (sy > -24) & (sy < height + 24)
    sx, sy, z, col = sx[on], sy[on], z[on], col[on]
    order = np.argsort(z)[::-1]
    for x, y, zz, c in zip(sx[order], sy[order], z[order], col[order]):
        rad = max(0.85, min(3.4, 7.0 / float(zz)))
        draw.ellipse(
            (x - rad, y - rad, x + rad, y + rad),
            fill=(int(c[0]), int(c[1]), int(c[2]), 226),
        )
    return np.asarray(img.filter(ImageFilter.SMOOTH_MORE))


def update_progress(artifacts: list[str]) -> None:
    data = json.loads(PROGRESS_JSON.read_text()) if PROGRESS_JSON.exists() else {"stages": {}}
    stage = data.setdefault("stages", {}).setdefault("fusion_render", {})
    stage["status"] = "done"
    stage["artifacts"] = artifacts
    PROGRESS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--background-points", type=int, default=140000)
    parser.add_argument("--object-a-points", type=int, default=45000)
    parser.add_argument("--video", default="outputs/renders/fusion_real_multisource_render.mp4")
    parser.add_argument("--preview", default="outputs/renders/fusion_real_preview.jpg")
    args = parser.parse_args()

    bg_p, bg_c = read_3dgs_ply(
        ROOT / "outputs/3dgs_background_train/point_cloud/iteration_2000/point_cloud.ply",
        args.background_points,
        seed=11,
    )
    a_p, a_c = read_3dgs_ply(
        ROOT / "outputs/3dgs_object_a/point_cloud/iteration_2000/point_cloud.ply",
        args.object_a_points,
        seed=12,
    )
    b_p, b_c = read_ascii_rgb_ply(ROOT / "assets/object_b/object_b_dreamfusion_train1000_points.ply")
    c_p, c_c = read_ascii_rgb_ply(ROOT / "assets/object_c/object_c_zero123_points.ply")

    bg_p = normalize(bg_p)
    a_p = transform(normalize(a_p), 0.34, (-0.62, -0.18, 0.08), math.radians(18))
    b_p = transform(normalize(b_p), 0.33, (0.08, -0.18, -0.02), math.radians(-24))
    c_p = transform(normalize(c_p), 0.34, (0.66, -0.18, 0.08), math.radians(-15))

    points = np.concatenate([bg_p, a_p, b_p, c_p], axis=0)
    colors = np.concatenate([bg_c, a_c, b_c, c_c], axis=0)

    frames = []
    target = np.array([0.02, -0.03, 0.03], dtype=np.float32)
    for i in range(args.frames):
        ang = 2.0 * math.pi * i / args.frames
        eye = np.array(
            [1.85 * math.sin(ang), 0.36 + 0.06 * math.sin(2 * ang), -1.65 + 0.45 * math.cos(ang)],
            dtype=np.float32,
        )
        frames.append(render_points(points, colors, eye, target, args.width, args.height))

    video = ROOT / args.video
    preview = ROOT / args.preview
    video.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames[0]).save(preview, quality=94)
    imageio.mimsave(video, frames, fps=24, quality=8)
    update_progress([str(video.relative_to(ROOT)), str(preview.relative_to(ROOT))])
    print(f"wrote {video.relative_to(ROOT)} frames={len(frames)}")
    print(f"wrote {preview.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
