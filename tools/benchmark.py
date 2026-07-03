from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from car_detector.camera import create_source
from car_detector.config import parse_classes
from car_detector.detector import YoloOnnxDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure real detector FPS on this machine.")
    parser.add_argument("--model", default="models/yolo11n_320.onnx")
    parser.add_argument("--source", choices=["auto", "picamera2", "opencv"], default="auto")
    parser.add_argument("--video", default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--classes", default="car,bus,truck")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warmup", type=int, default=5)
    return parser


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, int(round((percent / 100.0) * (len(values) - 1))))
    return sorted(values)[index]


def main() -> int:
    args = build_parser().parse_args()
    source = create_source(args.source, args.width, args.height, args.fps, args.camera_index, args.video)
    detector = YoloOnnxDetector(
        model_path=args.model,
        input_size=args.input_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        class_names=parse_classes(args.classes),
        threads=args.threads,
    )

    try:
        for _ in range(args.warmup):
            frame = source.read()
            if frame is None:
                raise RuntimeError("No frames during warmup.")
            detector.detect(frame)

        infer_ms: list[float] = []
        frames = 0
        started = perf_counter()
        while perf_counter() - started < args.seconds:
            frame = source.read()
            if frame is None:
                break
            result = detector.detect_timed(frame, frames + 1)
            infer_ms.append(result.inference_ms)
            frames += 1
    finally:
        source.release()

    elapsed = perf_counter() - started
    summary = {
        "frames": frames,
        "elapsed_sec": round(elapsed, 3),
        "detector_fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
        "infer_ms_avg": round(statistics.mean(infer_ms), 2) if infer_ms else 0.0,
        "infer_ms_p50": round(statistics.median(infer_ms), 2) if infer_ms else 0.0,
        "infer_ms_p95": round(percentile(infer_ms, 95), 2) if infer_ms else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
