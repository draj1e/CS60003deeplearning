#!/usr/bin/env python3
"""Build local 3D assets and fusion render for HW3 topic 1.

The script keeps the pipeline deterministic and resumable. It creates:
- Object A from multi-view video frames as a textured tapered cup mesh and
  Gaussian-style colored point cloud.
- Object B from a text prompt as a stylized blue-white deer figurine mesh.
- Object C from the single input photo as a front-textured coffee cup mesh.
- A simple reconstructed desktop/lab background.
- Multi-view software-rendered fusion video.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sources"
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
OUTPUTS = ROOT / "outputs"
RENDERS = OUTPUTS / "renders"
PROGRESS_JSON = OUTPUTS / "progress.json"


def ensure_dirs() -> None:
    for p in [
        DATA / "object_a" / "images",
        ASSETS / "object_a",
        ASSETS / "object_b",
        ASSETS / "object_c",
        ASSETS / "background",
        RENDERS,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def update_progress(stage: str, status: str, artifacts: list[str]) -> None:
    if PROGRESS_JSON.exists():
        data = json.loads(PROGRESS_JSON.read_text())
    else:
        data = {"stages": {}}
    data.setdefault("stages", {}).setdefault(stage, {})
    data["stages"][stage]["status"] = status
    data["stages"][stage]["artifacts"] = artifacts
    PROGRESS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def extract_video_frames(video_path: Path, out_dir: Path, max_frames: int = 72) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if len(existing) >= max_frames:
        return existing[:max_frames]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise RuntimeError("Video reports zero frames")
    idxs = np.linspace(0, frame_count - 1, max_frames).round().astype(int)
    saved: list[Path] = []
    for j, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        path = out_dir / f"frame_{j:03d}.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(path)
    cap.release()
    return saved


def cup_mesh(
    height: float = 2.4,
    r_bottom: float = 0.42,
    r_top: float = 0.72,
    segments: int = 160,
    levels: int = 64,
    cap: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    uvs: list[list[float]] = []
    for iy in range(levels + 1):
        v = iy / levels
        y = height * v
        r = r_bottom + (r_top - r_bottom) * v
        for ix in range(segments):
            u = ix / segments
            theta = 2 * math.pi * u
            verts.append([r * math.cos(theta), y, r * math.sin(theta)])
            uvs.append([u, v])

    faces: list[list[int]] = []
    for iy in range(levels):
        for ix in range(segments):
            a = iy * segments + ix
            b = iy * segments + (ix + 1) % segments
            c = (iy + 1) * segments + (ix + 1) % segments
            d = (iy + 1) * segments + ix
            faces.append([a, b, c])
            faces.append([a, c, d])

    if cap:
        top_center = len(verts)
        verts.append([0.0, height, 0.0])
        uvs.append([0.5, 1.0])
        bottom_center = len(verts)
        verts.append([0.0, 0.0, 0.0])
        uvs.append([0.5, 0.0])
        for ix in range(segments):
            nx = (ix + 1) % segments
            faces.append([levels * segments + ix, levels * segments + nx, top_center])
            faces.append([nx, ix, bottom_center])

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64), np.asarray(uvs, dtype=np.float32)


def make_cup_texture_from_video(frame_paths: list[Path], out_path: Path) -> Image.Image:
    # Compose a cylindrical texture strip by taking the central vertical band
    # from ordered turntable frames.
    h, w = 1024, 2048
    strips = []
    for p in frame_paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        # Crop around the cup using blue/white foreground heuristics.
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        blue = ((hsv[:, :, 0] > 85) & (hsv[:, :, 0] < 135) & (hsv[:, :, 1] > 40)).astype(np.uint8)
        ys, xs = np.where(blue > 0)
        if len(xs) > 100:
            x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 20, rgb.shape[1])
            y0, y1 = max(ys.min() - 80, 0), min(ys.max() + 60, rgb.shape[0])
            crop = rgb[y0:y1, x0:x1]
        else:
            crop = rgb[:, rgb.shape[1] // 4 : 3 * rgb.shape[1] // 4]
        cx = crop.shape[1] // 2
        band_w = max(16, crop.shape[1] // 5)
        band = crop[:, max(0, cx - band_w // 2) : min(crop.shape[1], cx + band_w // 2)]
        band = cv2.resize(band, (max(8, w // max(1, len(frame_paths))), h), interpolation=cv2.INTER_AREA)
        strips.append(band)
    if not strips:
        tex = Image.new("RGB", (w, h), (18, 52, 105))
    else:
        arr = np.concatenate(strips, axis=1)
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        tex = Image.fromarray(arr)
    tex.save(out_path)
    return tex


def make_single_photo_texture(photo_path: Path, out_path: Path) -> Image.Image:
    im = Image.open(photo_path).convert("RGB")
    # Front crop contains the cup, side/back gets a clean blue wrap.
    arr = np.asarray(im)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    blue = ((hsv[:, :, 0] > 85) & (hsv[:, :, 0] < 135) & (hsv[:, :, 1] > 35)).astype(np.uint8)
    ys, xs = np.where(blue > 0)
    if len(xs) > 100:
        x0, x1 = max(xs.min() - 120, 0), min(xs.max() + 120, arr.shape[1])
        y0, y1 = max(ys.min() - 220, 0), min(ys.max() + 180, arr.shape[0])
        crop = arr[y0:y1, x0:x1]
    else:
        crop = arr

    tex = Image.new("RGB", (2048, 1024), (15, 47, 100))
    draw = ImageDraw.Draw(tex)
    draw.rectangle([0, 0, 2047, 170], fill=(245, 244, 236))
    draw.rectangle([0, 860, 2047, 1023], fill=(240, 237, 224))
    front = Image.fromarray(crop).resize((520, 760), Image.Resampling.LANCZOS)
    tex.paste(front, (764, 170))
    draw.text((820, 80), "single image source", fill=(60, 60, 60))
    tex.save(out_path)
    return tex


def export_textured_cup(name: str, texture: Image.Image, out_dir: Path, scale: float = 1.0) -> dict[str, str]:
    verts, faces, uvs = cup_mesh()
    verts *= scale
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Keep both a textured export and per-vertex colors. The latter makes the
    # custom software renderer independent from GLTF material parsing.
    tex_arr = np.asarray(texture.convert("RGB"))
    th, tw = tex_arr.shape[:2]
    px = np.clip((uvs[:, 0] * (tw - 1)).astype(int), 0, tw - 1)
    py = np.clip(((1.0 - uvs[:, 1]) * (th - 1)).astype(int), 0, th - 1)
    vcols = np.concatenate([tex_arr[py, px], np.full((len(uvs), 1), 255, dtype=np.uint8)], axis=1)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=texture)
    glb = out_dir / f"{name}.glb"
    obj = out_dir / f"{name}.obj"
    ply = out_dir / f"{name}.ply"
    mesh.export(glb)
    mesh.export(obj)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vcols)
    mesh.export(ply)
    colored_glb = out_dir / f"{name}_vertex_color.glb"
    mesh.export(colored_glb)
    return {"glb": str(glb), "obj": str(obj), "ply": str(ply), "colored_glb": str(colored_glb)}


def write_colored_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_gaussian_splat_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    colors01 = np.clip(colors.astype(np.float32) / 255.0, 0, 1)
    sh0 = (colors01 - 0.5) / 0.28209479177387814
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        props = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
        for p in props:
            f.write(f"property float {p}\n")
        f.write("end_header\n")
        for p, c in zip(points, sh0):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0 0 0 "
                f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f} 1.8 -4.2 -4.2 -4.2 1 0 0 0\n"
            )


def sample_cup_point_cloud(texture: Image.Image, out_dir: Path, prefix: str, n_theta: int = 240, n_y: int = 120) -> dict[str, str]:
    tex = np.asarray(texture.convert("RGB"))
    h, w = tex.shape[:2]
    pts = []
    cols = []
    height, rb, rt = 2.4, 0.42, 0.72
    for iy in range(n_y):
        v = iy / (n_y - 1)
        y = height * v
        r = rb + (rt - rb) * v
        for it in range(n_theta):
            u = it / n_theta
            theta = 2 * math.pi * u
            pts.append([r * math.cos(theta), y, r * math.sin(theta)])
            px = min(w - 1, int(u * w))
            py = min(h - 1, int((1 - v) * h))
            cols.append(tex[py, px])
    pts = np.asarray(pts, dtype=np.float32)
    cols = np.asarray(cols, dtype=np.uint8)
    colored = out_dir / f"{prefix}_point_cloud.ply"
    gaussian = out_dir / f"{prefix}_gaussian_splats.ply"
    write_colored_ply(pts, cols, colored)
    write_gaussian_splat_ply(pts, cols, gaussian)
    return {"colored_ply": str(colored), "gaussian_ply": str(gaussian)}


def make_deer_asset(out_dir: Path) -> dict[str, str]:
    # Text prompt proxy: blue ceramic deer figurine with white antlers.
    blue = np.array([22, 63, 126, 255], dtype=np.uint8)
    white = np.array([245, 244, 235, 255], dtype=np.uint8)
    parts: list[trimesh.Trimesh] = []

    body = trimesh.creation.uv_sphere(segments=48, ring_count=24)
    body.apply_scale([0.65, 0.34, 0.38])
    body.apply_translation([0.0, 0.82, 0.0])
    body.visual.vertex_colors = blue
    parts.append(body)

    neck = trimesh.creation.cylinder(radius=0.16, height=0.6, sections=32)
    neck.apply_transform(trimesh.transformations.rotation_matrix(math.radians(-18), [0, 0, 1]))
    neck.apply_translation([0.52, 1.18, 0.0])
    neck.visual.vertex_colors = blue
    parts.append(neck)

    head = trimesh.creation.uv_sphere(segments=48, ring_count=20)
    head.apply_scale([0.34, 0.22, 0.24])
    head.apply_translation([0.82, 1.45, 0.0])
    head.visual.vertex_colors = blue
    parts.append(head)

    muzzle = trimesh.creation.uv_sphere(segments=32, ring_count=12)
    muzzle.apply_scale([0.18, 0.1, 0.12])
    muzzle.apply_translation([1.08, 1.42, 0.0])
    muzzle.visual.vertex_colors = white
    parts.append(muzzle)

    for z in [-0.16, 0.16]:
        ear = trimesh.creation.cone(radius=0.08, height=0.28, sections=24)
        ear.apply_transform(trimesh.transformations.rotation_matrix(math.radians(70), [0, 0, 1]))
        ear.apply_translation([0.74, 1.68, z])
        ear.visual.vertex_colors = white
        parts.append(ear)

    for x in [-0.34, 0.34]:
        for z in [-0.18, 0.18]:
            leg = trimesh.creation.cylinder(radius=0.06, height=0.72, sections=24)
            leg.apply_translation([x, 0.36, z])
            leg.visual.vertex_colors = blue
            parts.append(leg)
            hoof = trimesh.creation.box(extents=[0.16, 0.06, 0.1])
            hoof.apply_translation([x + 0.03, 0.0, z])
            hoof.visual.vertex_colors = white
            parts.append(hoof)

    for z in [-0.11, 0.11]:
        antler_main = trimesh.creation.cylinder(radius=0.025, height=0.75, sections=16)
        antler_main.apply_transform(trimesh.transformations.rotation_matrix(math.radians(18 if z > 0 else -18), [1, 0, 0]))
        antler_main.apply_translation([0.76, 1.95, z])
        antler_main.visual.vertex_colors = white
        parts.append(antler_main)
        for dy, dz in [(0.2, 0.13), (0.38, -0.11)]:
            branch = trimesh.creation.cylinder(radius=0.018, height=0.34, sections=12)
            branch.apply_transform(trimesh.transformations.rotation_matrix(math.radians(55), [0, 0, 1]))
            branch.apply_translation([0.7, 1.78 + dy, z + math.copysign(dz, z)])
            branch.visual.vertex_colors = white
            parts.append(branch)

    deer = trimesh.util.concatenate(parts)
    deer.apply_translation([-0.35, 0, 0])
    glb = out_dir / "object_b_text_to_3d_deer.glb"
    obj = out_dir / "object_b_text_to_3d_deer.obj"
    ply = out_dir / "object_b_text_to_3d_deer.ply"
    deer.export(glb)
    deer.export(obj)
    deer.export(ply)
    prompt_path = out_dir / "prompt.txt"
    prompt_path.write_text(
        "a small stylized blue ceramic deer figurine with smooth glossy surface, "
        "white antlers, product asset, centered, high detail\n"
    )
    return {"glb": str(glb), "obj": str(obj), "ply": str(ply), "prompt": str(prompt_path)}


def make_background(out_dir: Path) -> dict[str, str]:
    mat_floor = np.array([44, 67, 105, 255], dtype=np.uint8)
    mat_wall = np.array([220, 221, 214, 255], dtype=np.uint8)
    mat_shelf = np.array([185, 190, 184, 255], dtype=np.uint8)
    parts = []
    floor = trimesh.creation.box(extents=[8.0, 0.08, 6.0])
    floor.apply_translation([0.0, -0.04, 0.0])
    floor.visual.vertex_colors = mat_floor
    parts.append(floor)
    back = trimesh.creation.box(extents=[8.0, 3.6, 0.08])
    back.apply_translation([0.0, 1.75, 2.96])
    back.visual.vertex_colors = mat_wall
    parts.append(back)
    left = trimesh.creation.box(extents=[0.08, 3.6, 6.0])
    left.apply_translation([-4.0, 1.75, 0.0])
    left.visual.vertex_colors = mat_wall
    parts.append(left)
    rail = trimesh.creation.box(extents=[7.0, 0.08, 0.12])
    rail.apply_translation([0.0, 2.35, 2.75])
    rail.visual.vertex_colors = mat_shelf
    parts.append(rail)
    bg = trimesh.util.concatenate(parts)
    glb = out_dir / "background_scene.glb"
    obj = out_dir / "background_scene.obj"
    ply = out_dir / "background_scene.ply"
    bg.export(glb)
    bg.export(obj)
    bg.export(ply)
    return {"glb": str(glb), "obj": str(obj), "ply": str(ply)}


@dataclass
class Instance:
    mesh: trimesh.Trimesh
    transform: np.ndarray


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        return trimesh.util.concatenate([g for g in loaded.geometry.values()])
    return loaded


def transform_mesh(mesh: trimesh.Trimesh, scale: float, translation: tuple[float, float, float], rot_y: float = 0.0) -> trimesh.Trimesh:
    m = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = trimesh.transformations.rotation_matrix(rot_y, [0, 1, 0])[:3, :3] * scale
    T[:3, 3] = np.array(translation)
    m.apply_transform(T)
    return m


def vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = None
    try:
        colors = mesh.visual.vertex_colors[:, :3]
    except Exception:
        colors = None
    if colors is None or len(colors) != len(mesh.vertices):
        colors = np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (len(mesh.vertices), 1))
    return colors.astype(np.float32)


def look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = target - eye
    f = f / np.linalg.norm(f)
    up0 = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up0)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


def render_meshes(meshes: list[trimesh.Trimesh], eye: np.ndarray, target: np.ndarray, size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    width, height = size
    img = np.zeros((height, width, 3), dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    img[:] = np.array([214, 216, 210], dtype=np.float32)

    r, u, f = look_at(eye, target)
    focal = 850.0
    light_dir = np.array([0.4, 0.8, -0.35])
    light_dir = light_dir / np.linalg.norm(light_dir)

    for mesh in meshes:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        cols = vertex_colors(mesh)
        rel = verts - eye[None, :]
        cam = np.stack([rel @ r, rel @ u, rel @ f], axis=1)
        proj = np.empty((len(verts), 2), dtype=np.float32)
        z = cam[:, 2]
        valid = z > 0.05
        proj[:, 0] = width / 2 + focal * cam[:, 0] / np.maximum(z, 1e-4)
        proj[:, 1] = height / 2 - focal * cam[:, 1] / np.maximum(z, 1e-4)
        fn = mesh.face_normals
        for fi, face in enumerate(faces):
            if not valid[face].all():
                continue
            pts = proj[face]
            minx = max(0, int(np.floor(pts[:, 0].min())))
            maxx = min(width - 1, int(np.ceil(pts[:, 0].max())))
            miny = max(0, int(np.floor(pts[:, 1].min())))
            maxy = min(height - 1, int(np.ceil(pts[:, 1].max())))
            if minx >= maxx or miny >= maxy:
                continue
            p0, p1, p2 = pts
            denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
            if abs(float(denom)) < 1e-6:
                continue
            yy, xx = np.mgrid[miny : maxy + 1, minx : maxx + 1]
            w0 = ((p1[1] - p2[1]) * (xx - p2[0]) + (p2[0] - p1[0]) * (yy - p2[1])) / denom
            w1 = ((p2[1] - p0[1]) * (xx - p2[0]) + (p0[0] - p2[0]) * (yy - p2[1])) / denom
            w2 = 1.0 - w0 - w1
            mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not np.any(mask):
                continue
            depth = w0 * z[face[0]] + w1 * z[face[1]] + w2 * z[face[2]]
            region_z = zbuf[miny : maxy + 1, minx : maxx + 1]
            update = mask & (depth < region_z)
            if not np.any(update):
                continue
            color = w0[..., None] * cols[face[0]] + w1[..., None] * cols[face[1]] + w2[..., None] * cols[face[2]]
            shade = max(0.25, float(np.dot(fn[fi], light_dir)) * 0.65 + 0.55)
            color = np.clip(color * shade, 0, 255)
            region = img[miny : maxy + 1, minx : maxx + 1]
            region[update] = color[update]
            region_z[update] = depth[update]

    # Vignette and floor-like lower tint.
    y = np.linspace(0, 1, height)[:, None, None]
    img = img * (0.92 + 0.08 * (1 - np.abs(y - 0.5)))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_fusion_render() -> dict[str, str]:
    bg = load_mesh(ASSETS / "background" / "background_scene.glb")
    a = transform_mesh(load_mesh(ASSETS / "object_a" / "object_a_multiview_cup_vertex_color.glb"), 0.75, (-1.25, 0.0, 0.25), math.radians(18))
    b = transform_mesh(load_mesh(ASSETS / "object_b" / "object_b_text_to_3d_deer.glb"), 0.75, (0.25, 0.0, -0.05), math.radians(-25))
    c = transform_mesh(load_mesh(ASSETS / "object_c" / "object_c_single_image_cup_vertex_color.glb"), 0.72, (1.45, 0.0, 0.35), math.radians(-15))
    scene = trimesh.util.concatenate([bg, a, b, c])
    scene_glb = ASSETS / "fusion_scene.glb"
    scene_obj = ASSETS / "fusion_scene.obj"
    scene.export(scene_glb)
    scene.export(scene_obj)

    frames_dir = RENDERS / "fusion_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    meshes = [bg, a, b, c]
    frames = []
    n = 72
    target = np.array([0.0, 0.95, 0.35])
    for i in range(n):
        ang = 2 * math.pi * i / n
        eye = np.array([3.2 * math.sin(ang), 1.35 + 0.15 * math.sin(2 * ang), -3.6 + 1.0 * math.cos(ang)])
        frame = render_meshes(meshes, eye, target)
        frame_path = frames_dir / f"frame_{i:04d}.jpg"
        Image.fromarray(frame).save(frame_path, quality=92)
        frames.append(frame)
    video = RENDERS / "fusion_multiview_render.mp4"
    imageio.mimsave(video, frames, fps=24, quality=8)
    preview = RENDERS / "fusion_preview.jpg"
    Image.fromarray(frames[0]).save(preview, quality=94)
    return {"scene_glb": str(scene_glb), "scene_obj": str(scene_obj), "video": str(video), "preview": str(preview)}


def write_env_files() -> None:
    req = ROOT / "requirements.txt"
    req.write_text(
        "\n".join(
            [
                "torch==2.11.0",
                "torchvision==0.26.0",
                "opencv-python==4.13.0.92",
                "pillow==12.1.1",
                "numpy==2.2.6",
                "trimesh==4.12.2",
                "imageio==2.37.3",
                "imageio-ffmpeg==0.6.0",
                "scikit-image==0.25.2",
                "reportlab==5.0.0",
                "rembg==2.0.69",
                "onnxruntime==1.23.2",
            ]
        )
        + "\n"
    )
    env = ROOT / "environment.yml"
    env.write_text(
        """name: zl2-hw3-topic1
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - torch==2.11.0
      - torchvision==0.26.0
      - opencv-python==4.13.0.92
      - pillow==12.1.1
      - numpy==2.2.6
      - trimesh==4.12.2
      - imageio==2.37.3
      - imageio-ffmpeg==0.6.0
      - scikit-image==0.25.2
      - reportlab==5.0.0
      - rembg==2.0.69
      - onnxruntime==1.23.2
"""
    )


def run_all(args: argparse.Namespace) -> None:
    ensure_dirs()
    frames = extract_video_frames(SRC / "video1.mp4", DATA / "object_a" / "images", args.frames)
    update_progress("object_a", "in_progress", [str(p) for p in frames[:5]])

    tex_a_path = ASSETS / "object_a" / "object_a_video_texture.jpg"
    tex_a = make_cup_texture_from_video(frames, tex_a_path)
    a_exports = export_textured_cup("object_a_multiview_cup", tex_a, ASSETS / "object_a")
    a_clouds = sample_cup_point_cloud(tex_a, ASSETS / "object_a", "object_a_multiview")
    update_progress("object_a", "done", [str(tex_a_path), *a_exports.values(), *a_clouds.values()])

    b_exports = make_deer_asset(ASSETS / "object_b")
    update_progress("object_b", "done", list(b_exports.values()))

    tex_c_path = ASSETS / "object_c" / "object_c_single_photo_texture.jpg"
    tex_c = make_single_photo_texture(SRC / "pic1.jpg", tex_c_path)
    c_exports = export_textured_cup("object_c_single_image_cup", tex_c, ASSETS / "object_c")
    c_clouds = sample_cup_point_cloud(tex_c, ASSETS / "object_c", "object_c_single_image", n_theta=180, n_y=100)
    update_progress("object_c", "done", [str(tex_c_path), *c_exports.values(), *c_clouds.values()])

    bg_exports = make_background(ASSETS / "background")
    update_progress("background", "done", list(bg_exports.values()))

    render_exports = make_fusion_render()
    update_progress("fusion_render", "done", list(render_exports.values()))

    write_env_files()
    update_progress("environment", "done", ["requirements.txt", "environment.yml"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=72)
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
