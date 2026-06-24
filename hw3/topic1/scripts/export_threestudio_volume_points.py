import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors_u8):
            f.write(
                f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", type=int, default=96)
    parser.add_argument("--points", type=int, default=50000)
    parser.add_argument("--chunk", type=int, default=262144)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "third_party" / "threestudio"))

    import threestudio  # noqa: WPS433
    import threestudio.models.geometry  # noqa: F401, WPS433
    import threestudio.models.materials  # noqa: F401, WPS433
    from threestudio.utils.misc import get_device  # noqa: WPS433

    device = get_device()
    cfg = OmegaConf.load(args.config)
    geometry = threestudio.find(cfg.system.geometry_type)(cfg.system.geometry).to(device)
    material = threestudio.find(cfg.system.material_type)(cfg.system.material).to(device)

    state = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    geometry.load_state_dict(
        {k.removeprefix("geometry."): v for k, v in state.items() if k.startswith("geometry.")},
        strict=True,
    )
    material.load_state_dict(
        {k.removeprefix("material."): v for k, v in state.items() if k.startswith("material.")},
        strict=True,
    )
    geometry.eval()
    material.eval()

    coords_1d = torch.linspace(-geometry.cfg.radius, geometry.cfg.radius, args.grid)
    mesh = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    points = torch.stack([m.reshape(-1) for m in mesh], dim=-1)

    densities = []
    with torch.no_grad():
        for start in range(0, len(points), args.chunk):
            chunk = points[start : start + args.chunk].to(device)
            densities.append(geometry.forward_density(chunk).reshape(-1).cpu())
    densities_t = torch.cat(densities)
    k = min(args.points, int((densities_t > 0).sum().item()), len(densities_t))
    values, indices = torch.topk(densities_t, k=k, largest=True)
    selected = points[indices].to(device)

    with torch.no_grad():
        geo = geometry.export(points=selected)
        mat = material.export(points=selected, **geo)
    colors = mat["albedo"].detach().cpu().numpy()
    selected_np = selected.detach().cpu().numpy()

    out = Path(args.out)
    write_ply(out, selected_np, colors)
    print(f"wrote {out} points={len(selected_np)} density_min={values[-1].item():.6f} density_max={values[0].item():.6f}")


if __name__ == "__main__":
    main()
