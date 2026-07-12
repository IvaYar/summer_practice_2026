from __future__ import annotations

import argparse
from time import perf_counter

import cv2

from .async_infer import AsyncDetector
from .camera import create_source
from .config import merge_options, parse_classes
from .detector import InferenceResult, YoloOnnxDetector
from .overlay import age_ms, draw_detections, draw_roi, draw_status, draw_warning_line
from .rate import RateMeter

WINDOW_NAME = "Pi5 car detector"


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
    parser.add_argument(
        "--model-classes",
        default=None,
        help="Output class order. Use 'coco' for COCO models or comma-separated names for custom models.",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--async-inference", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--detect-every", type=int, default=None, help="Run detection every Nth frame.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save", default=None, help="Optional output video path.")
    parser.add_argument("--window-x", type=int, default=None, help="Display window X position.")
    parser.add_argument("--window-y", type=int, default=None, help="Display window Y position.")
    parser.add_argument("--geometry-filter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-box-area-ratio", type=float, default=None)
    parser.add_argument("--max-box-area-ratio", type=float, default=None)
    parser.add_argument("--max-box-width-ratio", type=float, default=None)
    parser.add_argument("--max-box-height-ratio", type=float, default=None)
    parser.add_argument("--min-box-aspect-ratio", type=float, default=None)
    parser.add_argument("--max-box-aspect-ratio", type=float, default=None)
    parser.add_argument("--edge-margin-ratio", type=float, default=None)
    parser.add_argument("--edge-min-conf", type=float, default=None)
    parser.add_argument("--roi", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--roi-x1-ratio", type=float, default=None)
    parser.add_argument("--roi-y1-ratio", type=float, default=None)
    parser.add_argument("--roi-x2-ratio", type=float, default=None)
    parser.add_argument("--roi-y2-ratio", type=float, default=None)
    parser.add_argument("--show-roi", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warning-line", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warning-line-y-ratio", type=float, default=None)
    parser.add_argument("--print-every", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    options = merge_options(args.config, vars(args))
    class_names = parse_classes(options["classes"])
    model_class_names = None if options["model_classes"] == "coco" else parse_classes(options["model_classes"])

    detector = YoloOnnxDetector(
        model_path=options["model"],
        input_size=options["input_size"],
        conf_threshold=options["conf"],
        iou_threshold=options["iou"],
        class_names=class_names,
        model_class_names=model_class_names,
        threads=options["threads"],
        geometry_filter=options["geometry_filter"],
        min_box_area_ratio=options["min_box_area_ratio"],
        max_box_area_ratio=options["max_box_area_ratio"],
        max_box_width_ratio=options["max_box_width_ratio"],
        max_box_height_ratio=options["max_box_height_ratio"],
        min_box_aspect_ratio=options["min_box_aspect_ratio"],
        max_box_aspect_ratio=options["max_box_aspect_ratio"],
        edge_margin_ratio=options["edge_margin_ratio"],
        edge_min_conf=options["edge_min_conf"],
        roi_enabled=options["roi"],
        roi_x1_ratio=options["roi_x1_ratio"],
        roi_y1_ratio=options["roi_y1_ratio"],
        roi_x2_ratio=options["roi_x2_ratio"],
        roi_y2_ratio=options["roi_y2_ratio"],
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
    detect_every = max(1, int(options["detect_every"]))
    display_fps = RateMeter()
    latest_result = InferenceResult((), 0.0, perf_counter(), 0)
    writer = None
    frame_id = 0
    last_print = perf_counter()
    last_completed = 0
    sync_completed = 0
    last_sync_completed = 0
    last_skipped = 0
    det_fps = 0.0

    print(
        f"source={source.name} model={options['model']} input={options['input_size']} "
        f"classes={','.join(class_names)} async={bool(async_worker)} detect_every={detect_every}"
    )

    if not options["headless"]:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(WINDOW_NAME, int(options["window_x"]), int(options["window_y"]))

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            frame_id += 1
            fps_now = display_fps.tick()
            should_detect = frame_id == 1 or (frame_id - 1) % detect_every == 0

            if async_worker:
                if should_detect:
                    async_worker.submit(frame, frame_id)
                latest_result = async_worker.latest()
            else:
                if should_detect:
                    latest_result = detector.detect_timed(frame, frame_id)
                    sync_completed += 1

            draw_detections(frame, latest_result.detections)
            if options["roi"] and options["show_roi"]:
                draw_roi(frame, detector.roi_box(frame.shape[:2]))
            if options["warning_line"]:
                draw_warning_line(frame, options["warning_line_y_ratio"], latest_result.detections)
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
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                    break

            now = perf_counter()
            if now - last_print >= float(options["print_every"]):
                if async_worker:
                    completed = async_worker.completed
                    det_fps = (completed - last_completed) / (now - last_print)
                    last_completed = completed
                    skipped = async_worker.skipped
                    skipped_delta = skipped - last_skipped
                    last_skipped = skipped
                else:
                    det_fps = (sync_completed - last_sync_completed) / (now - last_print)
                    last_sync_completed = sync_completed
                    skipped_delta = 0
                last_print = now
                print(
                    f"frame={frame_id} display_fps={fps_now:.1f} detect_fps={det_fps:.1f} "
                    f"infer_ms={latest_result.inference_ms:.1f} vehicles={len(latest_result.detections)} "
                    f"skipped={skipped_delta}"
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
