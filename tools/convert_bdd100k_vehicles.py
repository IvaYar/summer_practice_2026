from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from car_detector.coco import COCO_NAMES

BDD_DET_CLASSES = {
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert BDD100K detection labels to vehicle-only YOLO format.")
    parser.add_argument("--bdd-root", required=True, help="Root containing BDD100K images and labels.")
    parser.add_argument("--output", default="datasets/bdd100k_vehicle", help="Output YOLO dataset directory.")
    parser.add_argument("--classes", default="car,bus,truck", help="Comma-separated BDD classes to keep.")
    parser.add_argument("--max-train", type=int, default=10000, help="Maximum train images. Use 0 for all.")
    parser.add_argument("--max-val", type=int, default=2000, help="Maximum val images. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--include-empty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-box-area", type=float, default=0.0, help="Minimum normalized box area.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser


def parse_classes(value: str) -> tuple[str, ...]:
    classes = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = [name for name in classes if name not in BDD_DET_CLASSES]
    if invalid:
        valid = ", ".join(sorted(BDD_DET_CLASSES))
        raise ValueError(f"Unknown BDD class names: {invalid}. Valid: {valid}")
    return classes


def find_first_existing(candidates: list[Path], kind: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    pretty = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {kind}. Tried:\n{pretty}")


def find_images_dir(root: Path, split: str) -> Path:
    return find_first_existing(
        [
            root / "images" / "100k" / split,
            root / "bdd100k" / "images" / "100k" / split,
            root / "bdd100k_images_100k" / "images" / "100k" / split,
            root / "bdd100k_images_100k" / "bdd100k" / "images" / "100k" / split,
        ],
        f"BDD100K {split} images directory",
    )


def find_label_json(root: Path, split: str) -> Path:
    return find_first_existing(
        [
            root / "labels" / "det_20" / f"det_{split}.json",
            root / "bdd100k" / "labels" / "det_20" / f"det_{split}.json",
            root / "bdd100k_labels_release" / "bdd100k" / "labels" / "det_20" / f"det_{split}.json",
            root / "bdd100k_labels_release" / "labels" / "det_20" / f"det_{split}.json",
        ],
        f"BDD100K {split} detection label JSON",
    )


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Use --overwrite to replace it.")
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def materialize_image(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required. Install requirements-training.txt first.") from exc

    with Image.open(image_path) as image:
        return image.size


def yolo_line(box: dict, class_id: int, width: int, height: int, min_area: float) -> str:
    x1 = float(box["x1"])
    y1 = float(box["y1"])
    x2 = float(box["x2"])
    y2 = float(box["y2"])
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(0.0, min(width - 1.0, x2))
    y2 = max(0.0, min(height - 1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return ""

    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    if box_w * box_h < min_area:
        return ""
    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
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


def convert_split(
    split: str,
    images_dir: Path,
    labels_json: Path,
    output: Path,
    class_names: tuple[str, ...],
    max_images: int,
    copy_mode: str,
    include_empty: bool,
    min_box_area: float,
) -> dict[str, int | dict[str, int]]:
    class_to_local = {name: index for index, name in enumerate(class_names)}
    records = json.loads(labels_json.read_text(encoding="utf-8"))
    random.shuffle(records)
    if max_images:
        records = records[:max_images]

    stats = Counter()
    class_counts: Counter[str] = Counter()
    for record in records:
        image_name = record["name"]
        src_image = images_dir / image_name
        if not src_image.exists():
            stats["missing_images"] += 1
            continue

        width, height = image_size(src_image)
        lines: list[str] = []
        for label in record.get("labels", []):
            category = label.get("category")
            box = label.get("box2d")
            if category not in class_to_local or not box:
                continue
            line = yolo_line(box, class_to_local[category], width, height, min_box_area)
            if not line:
                continue
            lines.append(line)
            class_counts[category] += 1

        if not lines and not include_empty:
            stats["empty_skipped"] += 1
            continue

        dst_image = output / "images" / split / image_name
        dst_label = output / "labels" / split / f"{Path(image_name).stem}.txt"
        materialize_image(src_image, dst_image, copy_mode)
        dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["boxes"] += len(lines)

    return {
        "images": stats["images"],
        "boxes": stats["boxes"],
        "empty_skipped": stats["empty_skipped"],
        "missing_images": stats["missing_images"],
        "class_counts": dict(class_counts),
    }


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.bdd_root)
    output = Path(args.output)
    class_names = parse_classes(args.classes)
    random.seed(args.seed)

    prepare_output(output, args.overwrite)
    write_dataset_yaml(output, class_names)

    summary = {
        "bdd_root": str(root),
        "output": str(output),
        "classes": class_names,
        "coco_equivalent_ids": {name: COCO_NAMES.index(name) for name in class_names if name in COCO_NAMES},
        "splits": {},
    }
    for split, max_images in (("train", args.max_train), ("val", args.max_val)):
        images_dir = find_images_dir(root, split)
        labels_json = find_label_json(root, split)
        print(f"{split}: images={images_dir} labels={labels_json}")
        split_stats = convert_split(
            split=split,
            images_dir=images_dir,
            labels_json=labels_json,
            output=output,
            class_names=class_names,
            max_images=max_images,
            copy_mode=args.copy_mode,
            include_empty=args.include_empty,
            min_box_area=args.min_box_area,
        )
        summary["splits"][split] = split_stats
        print(f"{split}: {split_stats}")

    (output / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"YOLO dataset saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
