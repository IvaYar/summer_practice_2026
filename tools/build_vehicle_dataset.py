from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from time import perf_counter

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from car_detector.coco import COCO_NAMES

DEFAULT_VEHICLE_NAMES = ("car", "bus", "truck")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a small vehicle-only YOLO dataset from a video using pseudo-labels."
    )
    parser.add_argument("--video", required=True, help="Source video path.")
    parser.add_argument("--output", default="datasets/vehicle_pseudo", help="Output dataset directory.")
    parser.add_argument("--teacher", default="yolo26n.pt", help="Ultralytics model used for pseudo-labels.")
    parser.add_argument("--classes", default="car,bus,truck", help="Comma-separated COCO classes.")
    parser.add_argument("--label-imgsz", type=int, default=640, help="Teacher inference size for pseudo-labels.")
    parser.add_argument("--conf", type=float, default=0.18, help="Teacher confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Teacher IoU threshold.")
    parser.add_argument("--sample-every", type=int, default=15, help="Take one frame every N video frames.")
    parser.add_argument("--max-frames", type=int, default=800, help="Maximum sampled frames. Use 0 for no limit.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--keep-empty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser


def parse_class_names(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    missing = [name for name in names if name not in COCO_NAMES]
    if missing:
        raise ValueError(f"Unknown COCO class names: {missing}")
    return names


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Use --overwrite to replace it.")
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def yolo_line(box_xyxy, class_id: int, width: int, height: int) -> str:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(0.0, min(width - 1.0, x2))
    y2 = max(0.0, min(height - 1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return ""

    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}"


def write_dataset_yaml(output: Path, class_names: tuple[str, ...]) -> None:
    names = ", ".join(f"{index}: {name}" for index, name in enumerate(class_names))
    content = (
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {{{names}}}\n"
    )
    (output / "dataset.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    video_path = Path(args.video)
    output = Path(args.output)
    class_names = parse_class_names(args.classes)
    coco_ids = [COCO_NAMES.index(name) for name in class_names]
    coco_to_local = {coco_id: local_id for local_id, coco_id in enumerate(coco_ids)}

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install it with: python -m pip install -r requirements-training.txt"
        ) from exc

    prepare_output(output, args.overwrite)
    write_dataset_yaml(output, class_names)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    teacher = YOLO(args.teacher)
    random.seed(args.seed)

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    started = perf_counter()
    sampled = 0
    saved = 0
    labels_total = 0
    frame_index = -1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if args.sample_every > 1 and frame_index % args.sample_every != 0:
            continue
        if args.max_frames and sampled >= args.max_frames:
            break

        sampled += 1
        height, width = frame.shape[:2]
        result = teacher.predict(
            frame,
            imgsz=args.label_imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=coco_ids,
            verbose=False,
        )[0]

        label_lines: list[str] = []
        if result.boxes is not None:
            for box in result.boxes:
                coco_id = int(box.cls.item())
                if coco_id not in coco_to_local:
                    continue
                line = yolo_line(box.xyxy[0].tolist(), coco_to_local[coco_id], width, height)
                if line:
                    label_lines.append(line)

        if not label_lines and not args.keep_empty:
            continue

        split = "val" if random.random() < args.val_ratio else "train"
        stem = f"{video_path.stem}_{frame_index:08d}"
        image_path = output / "images" / split / f"{stem}.jpg"
        label_path = output / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        saved += 1
        labels_total += len(label_lines)

        if sampled % 25 == 0:
            elapsed = max(0.001, perf_counter() - started)
            print(
                f"sampled={sampled} saved={saved} labels={labels_total} "
                f"speed={sampled / elapsed:.2f} sampled_fps"
            )

    capture.release()

    metadata = {
        "video": str(video_path),
        "source_fps": source_fps,
        "source_frames": frame_count,
        "sample_every": args.sample_every,
        "sampled": sampled,
        "saved": saved,
        "labels": labels_total,
        "classes": class_names,
        "teacher": args.teacher,
        "label_imgsz": args.label_imgsz,
        "conf": args.conf,
        "iou": args.iou,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Dataset saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
