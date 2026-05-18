from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from hw2.common.utils import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detection.yaml")
    parser.add_argument("--video", default=None)
    parser.add_argument("--frames", nargs="*", type=int, default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    video_path = args.video or cfg["video_path"]
    frames = args.frames if args.frames else cfg.get("occlusion_frames", [])
    if not frames:
        raise ValueError("请通过 --frames 或 configs/detection.yaml 的 occlusion_frames 指定连续 3-4 帧")
    output_dir = ensure_dir(Path(cfg["output_dir"]) / "occlusion_frames")
    capture = cv2.VideoCapture(video_path)
    wanted = set(frames)
    index = -1
    saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if index in wanted:
            cv2.imwrite(str(output_dir / f"frame_{index:06d}.jpg"), frame)
            saved += 1
    capture.release()
    print(f"saved {saved} frames to {output_dir}")


if __name__ == "__main__":
    main()
