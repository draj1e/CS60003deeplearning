#!/usr/bin/env python3
"""Prepare real COLMAP image frames for object A from the phone video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="sources/video1.mp4")
    parser.add_argument("--out", default="data/object_a_colmap/images")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=720)
    args = parser.parse_args()

    video = ROOT / args.video
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // args.max_frames)
    saved = 0
    idx = 0
    while saved < args.max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = args.width / float(w)
        frame = cv2.resize(frame, (args.width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        path = out / f"frame_{saved:04d}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved += 1
        idx += step
    cap.release()
    print(f"saved {saved} frames to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
