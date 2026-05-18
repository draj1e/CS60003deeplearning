#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Road Vehicle 验证集图片合成 10-30 秒 tracking 演示视频")
    parser.add_argument("--images", default="data/road_vehicle_raw/trafic_data/valid/images")
    parser.add_argument("--out", default="data/videos/road_vehicle_demo.mp4")
    parser.add_argument("--seconds", type=int, default=12)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    images = sorted(Path(args.images).glob("*.jpg"))
    if not images:
        raise FileNotFoundError(args.images)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    frame_count = args.seconds * args.fps
    for index in range(frame_count):
        frame = cv2.imread(str(images[index % len(images)]))
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    print(output)


if __name__ == "__main__":
    main()
