from __future__ import annotations

import argparse
from time import perf_counter

import cv2

from .async_infer import AsyncDetector
from .camera import create_source
from .config import merge_options, parse_classes
from .detector import InferenceResult, YoloOnnxDetector
from .overlay import age_ms, draw_detections, draw_status
from .rate import RateMeter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime vehicle detection for Raspberry Pi 5.")
    parser.add_argument("--config", default=None, help="YAML config path.")
    parser.add_argument("--model", default=None, help="Path to YOLO ONNX model.")
    parser.add_argument("--source", choices=["auto", "picamera2", "opencv"], default=None)
    parser.add_argument("--video", default=None, help="Use a video file instead of a live camera.")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None, help="Detector input size, e.g. 256 or 320.")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--classes", default=None, help="Comma-separated COCO classes, e.g. car,bus,truck.")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--async-inference", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save", default=None, help="Optional output video path.")
    parser.add_argument("--print-every", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    options = merge_options(args.config, vars(args))
    class_names = parse_classes(options["classes"])

    detector = YoloOnnxDetector(
        model_path=options["model"],
        input_size=options["input_size"],
        conf_threshold=options["conf"],
        iou_threshold=options["iou"],
        class_names=class_names,
        threads=options["threads"],
    )
    source = create_source(
        source=options["source"],
        width=options["width"],
        height=options["height"],
        fps=options["fps"],
        camera_index=options["camera_index"],
        video=options["video"],
    )

    async_worker = AsyncDetector(detector) if options["async_inference"] else None
    display_fps = RateMeter()
    latest_result = InferenceResult((), 0.0, perf_counter(), 0)
    writer = None
    frame_id = 0
    last_print = perf_counter()
    last_completed = 0
    det_fps = 0.0

    print(
        f"source={source.name} model={options['model']} input={options['input_size']} "
        f"classes={','.join(class_names)} async={bool(async_worker)}"
    )

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            frame_id += 1
            fps_now = display_fps.tick()

            if async_worker:
                async_worker.submit(frame, frame_id)
                latest_result = async_worker.latest()
            else:
                latest_result = detector.detect_timed(frame, frame_id)

            draw_detections(frame, latest_result.detections)
            draw_status(
                frame,
                [
                    f"display {fps_now:4.1f} FPS  detect {det_fps:4.1f} FPS",
                    f"infer {latest_result.inference_ms:5.1f} ms  age {age_ms(latest_result.timestamp):4.0f} ms",
                    f"vehicles {len(latest_result.detections)}",
                ],
            )

            if writer is None and options["save"]:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(options["save"], fourcc, options["fps"], (width, height))
            if writer:
                writer.write(frame)

            if not options["headless"]:
                cv2.imshow("Pi5 car detector", frame)
                if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                    break

            now = perf_counter()
            if now - last_print >= float(options["print_every"]):
                if async_worker:
                    completed = async_worker.completed
                    det_fps = (completed - last_completed) / (now - last_print)
                    last_completed = completed
                else:
                    det_fps = fps_now
                last_print = now
                print(
                    f"frame={frame_id} display_fps={fps_now:.1f} detect_fps={det_fps:.1f} "
                    f"infer_ms={latest_result.inference_ms:.1f} vehicles={len(latest_result.detections)}"
                )

            if options["max_frames"] and frame_id >= int(options["max_frames"]):
                break
    finally:
        if async_worker:
            async_worker.stop()
        if writer:
            writer.release()
        source.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
