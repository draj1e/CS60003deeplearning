from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from hw2.common.utils import ensure_dir, load_yaml, save_json


def side_of_line(point: tuple[float, float], line: tuple[int, int, int, int]) -> float:
    x, y = point
    x1, y1, x2, y2 = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def draw_line(frame, line, count: int) -> None:
    x1, y1, x2, y2 = line
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, f"cross count: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detection.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    video_path = args.video or cfg["video_path"]
    output_dir = ensure_dir(Path(cfg["output_dir"]) / "tracking")
    output_video = output_dir / "tracked_counted.mp4"
    line = tuple(int(value) for value in cfg["line"])
    model = YOLO(args.weights)
    track_history: dict[int, float] = {}
    counted_ids: set[int] = set()
    events: list[dict] = []
    writer = None
    frame_index = -1

    results = model.track(source=video_path, stream=True, persist=True, tracker=cfg["tracker"], conf=float(cfg["conf"]), iou=float(cfg["iou"]))
    for result in results:
        frame_index += 1
        frame = result.orig_img.copy()
        if writer is None:
            height, width = frame.shape[:2]
            fps = result.speed.get("fps", 25) if hasattr(result, "speed") else 25
            writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps or 25), (width, height))
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for track_id, box, class_id in zip(ids, xyxy, classes):
                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                current_side = side_of_line(center, line)
                previous_side = track_history.get(track_id)
                if previous_side is not None and previous_side * current_side < 0 and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    events.append({"frame": frame_index, "track_id": int(track_id), "class_id": int(class_id), "center": [float(center[0]), float(center[1])]})
                track_history[track_id] = current_side
                label = f"ID {track_id} C {class_id}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), max(20, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        draw_line(frame, line, len(counted_ids))
        writer.write(frame)
    if writer is not None:
        writer.release()
    save_json({"count": len(counted_ids), "events": events, "output_video": str(output_video)}, output_dir / "count_events.json")


if __name__ == "__main__":
    main()
