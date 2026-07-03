from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and export a vehicle-only YOLO model.")
    parser.add_argument("--data", default="datasets/vehicle_pseudo/dataset.yaml", help="YOLO dataset YAML.")
    parser.add_argument("--weights", default="yolo26n.pt", help="Base Ultralytics weights.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="vehicle_yolo26n")
    parser.add_argument("--output", default="models/vehicle_yolo26n_320.onnx")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install it with: python -m pip install -r requirements-training.txt"
        ) from exc

    model = YOLO(args.weights)
    train_result = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )

    save_dir = Path(train_result.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = save_dir / "weights" / "last.pt"
    if not best_pt.exists():
        raise SystemExit(f"Training finished, but no weights were found in {save_dir / 'weights'}")

    trained = YOLO(str(best_pt))
    exported = Path(
        trained.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=12,
            simplify=False,
            dynamic=False,
            end2end=args.end2end,
        )
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output.resolve():
        shutil.copy2(exported, output)
    print(f"Best weights: {best_pt}")
    print(f"ONNX model saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
