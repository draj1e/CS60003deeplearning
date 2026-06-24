#!/usr/bin/env python3
"""Fast point/Gaussian-style fusion renderer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUTPUTS = ROOT / "outputs"
RENDERS = OUTPUTS / "renders"
PROGRESS_JSON = OUTPUTS / "progress.json"


def read_colored_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    end = lines.index("end_header")
    pts, cols = [], []
    for line in lines[end + 1 :]:
        parts = line.split()
        if len(parts) < 6:
            continue
        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        cols.append([int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))])
    return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.uint8)


def load_mesh_points(path: Path, sample: int = 12000) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="scene")
    geoms = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
    pts_all, cols_all = [], []
    rng = np.random.default_rng(42)
    for g in geoms:
        verts = np.asarray(g.vertices, dtype=np.float32)
        if len(verts) == 0:
            continue
        idx = rng.choice(len(verts), size=min(sample, len(verts)), replace=len(verts) < sample)
        cols = getattr(g.visual, "vertex_colors", None)
        if cols is None or len(cols) != len(verts):
            cols = np.tile(np.array([[180, 180, 180, 255]], dtype=np.uint8), (len(verts), 1))
        pts_all.append(verts[idx])
        cols_all.append(np.asarray(cols, dtype=np.uint8)[idx, :3])
    return np.concatenate(pts_all, axis=0), np.concatenate(cols_all, axis=0)


def transform(points: np.ndarray, scale: float, translation: tuple[float, float, float], rot_y: float = 0.0) -> np.ndarray:
    c, s = math.cos(rot_y), math.sin(rot_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return points @ R.T * scale + np.asarray(translation, dtype=np.float32)


def background_points() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    pts, cols = [], []
    # Blue desktop plane.
    n = 36000
    x = rng.uniform(-4, 4, n)
    z = rng.uniform(-2.8, 3.0, n)
    y = rng.normal(0, 0.004, n)
    pts.append(np.stack([x, y, z], axis=1))
    base = np.array([36, 63, 106], dtype=np.float32)
    noise = rng.normal(0, 8, (n, 3))
    cols.append(np.clip(base + noise, 0, 255).astype(np.uint8))
    # Pale wall.
    n = 24000
    x = rng.uniform(-4, 4, n)
    y = rng.uniform(0, 3.3, n)
    z = rng.normal(2.95, 0.004, n)
    pts.append(np.stack([x, y, z], axis=1))
    base = np.array([218, 219, 212], dtype=np.float32)
    noise = rng.normal(0, 5, (n, 3))
    cols.append(np.clip(base + noise, 0, 255).astype(np.uint8))
    # Back rail.
    n = 5000
    x = rng.uniform(-3.7, 3.7, n)
    y = rng.normal(2.45, 0.018, n)
    z = rng.normal(2.73, 0.035, n)
    pts.append(np.stack([x, y, z], axis=1))
    cols.append(np.tile(np.array([[170, 176, 170]], dtype=np.uint8), (n, 1)))
    return np.concatenate(pts), np.concatenate(cols)


def look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = target - eye
    f = f / np.linalg.norm(f)
    up0 = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up0)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


def render_points(points: np.ndarray, colors: np.ndarray, eye: np.ndarray, target: np.ndarray, width: int = 960, height: int = 540) -> np.ndarray:
    img = Image.new("RGB", (width, height), (218, 219, 212))
    draw = ImageDraw.Draw(img, "RGBA")
    r, u, f = look_at(eye, target)
    rel = points - eye[None, :]
    cam_x = rel @ r
    cam_y = rel @ u
    cam_z = rel @ f
    valid = cam_z > 0.1
    focal = 680.0
    sx = width / 2 + focal * cam_x[valid] / cam_z[valid]
    sy = height / 2 - focal * cam_y[valid] / cam_z[valid]
    z = cam_z[valid]
    col = colors[valid]
    on = (sx > -20) & (sx < width + 20) & (sy > -20) & (sy < height + 20)
    sx, sy, z, col = sx[on], sy[on], z[on], col[on]
    order = np.argsort(z)[::-1]
    for x, y, zz, c in zip(sx[order], sy[order], z[order], col[order]):
        rad = max(1.0, min(4.2, 8.0 / float(zz)))
        alpha = 235
        draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(int(c[0]), int(c[1]), int(c[2]), alpha))
    img = img.filter(ImageFilter.SMOOTH_MORE)
    return np.asarray(img)


def update_progress(artifacts: list[str]) -> None:
    data = json.loads(PROGRESS_JSON.read_text()) if PROGRESS_JSON.exists() else {"stages": {}}
    data.setdefault("stages", {}).setdefault("fusion_render", {})
    data["stages"]["fusion_render"]["status"] = "done"
    data["stages"]["fusion_render"]["artifacts"] = artifacts
    PROGRESS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    RENDERS.mkdir(parents=True, exist_ok=True)
    bg_p, bg_c = background_points()
    a_p, a_c = read_colored_ply(ASSETS / "object_a" / "object_a_multiview_point_cloud.ply")
    c_p, c_c = read_colored_ply(ASSETS / "object_c" / "object_c_single_image_point_cloud.ply")
    b_p, b_c = load_mesh_points(ASSETS / "object_b" / "object_b_text_to_3d_deer.glb", sample=14000)
    a_p = transform(a_p, 0.68, (-1.25, 0.02, 0.35), math.radians(18))
    b_p = transform(b_p, 0.70, (0.15, 0.02, 0.10), math.radians(-28))
    c_p = transform(c_p, 0.66, (1.35, 0.02, 0.38), math.radians(-15))
    points = np.concatenate([bg_p, a_p, b_p, c_p], axis=0)
    colors = np.concatenate([bg_c, a_c, b_c, c_c], axis=0)

    frames = []
    n = 96
    target = np.array([0.05, 1.0, 0.55], dtype=np.float32)
    for i in range(n):
        ang = 2 * math.pi * i / n
        eye = np.array([3.2 * math.sin(ang), 1.35 + 0.12 * math.sin(2 * ang), -3.4 + 0.75 * math.cos(ang)], dtype=np.float32)
        frames.append(render_points(points, colors, eye, target))
    preview = RENDERS / "fusion_splat_preview.jpg"
    video = RENDERS / "fusion_splat_multiview_render.mp4"
    Image.fromarray(frames[0]).save(preview, quality=94)
    imageio.mimsave(video, frames, fps=24, quality=8)
    update_progress(
        [
            str(ASSETS / "fusion_scene.glb"),
            str(ASSETS / "fusion_scene.obj"),
            str(video),
            str(preview),
            str(RENDERS / "fusion_multiview_render.mp4"),
        ]
    )


if __name__ == "__main__":
    main()
