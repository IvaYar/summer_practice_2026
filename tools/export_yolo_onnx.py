from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an Ultralytics YOLO model to ONNX for OpenCV DNN.")
    parser.add_argument("--weights", default="yolo11n.pt", help="Ultralytics weights name/path.")
    parser.add_argument("--imgsz", type=int, default=320, help="Square image size used by the detector.")
    parser.add_argument("--output", default="models/yolo11n_320.onnx", help="Destination ONNX path.")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--simplify", action="store_true", help="Ask Ultralytics to simplify the ONNX graph.")
    parser.add_argument(
        "--end2end",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For YOLO26, use the default NMS-free head or --no-end2end for traditional YOLO output.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install it with: python -m pip install -r requirements-export.txt"
        ) from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    export_args = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "opset": args.opset,
        "simplify": args.simplify,
        "dynamic": False,
    }
    if args.end2end is not None:
        export_args["end2end"] = args.end2end

    exported = Path(model.export(**export_args))
    if exported.resolve() != output.resolve():
        shutil.copy2(exported, output)
    print(f"ONNX model saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
