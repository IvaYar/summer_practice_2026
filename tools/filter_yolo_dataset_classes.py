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

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install requirements-training.txt first.") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter a YOLO dataset down to selected classes and remap IDs.")
    parser.add_argument("--source", required=True, help="Source YOLO dataset root.")
    parser.add_argument("--output", default="datasets/bdd100k_yolo_vehicle", help="Output dataset root.")
    parser.add_argument("--classes", default="car,bus,truck", help="Comma-separated class names to keep.")
    parser.add_argument("--max-train", type=int, default=10000, help="Maximum train images. Use 0 for all.")
    parser.add_argument("--max-val", type=int, default=2000, help="Maximum val images. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--include-empty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser


def parse_names(raw_names) -> list[str]:
    if isinstance(raw_names, dict):
        return [raw_names[index] for index in sorted(raw_names)]
    if isinstance(raw_names, list):
        return list(raw_names)
    raise ValueError("data.yaml names must be a list or mapping.")


def load_data_yaml(source: Path) -> tuple[dict, list[str]]:
    data_path = source / "data.yaml"
    if not data_path.exists():
        data_path = source / "data.yml"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data.yaml in {source}")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    return data, parse_names(data["names"])


def split_images_dir(source: Path, data: dict, split: str) -> Path:
    value = data.get(split, f"{split}/images")
    path = Path(value)
    if not path.is_absolute():
        path = source / path
    if not path.exists() and "kaggle/input" in value:
        path = source / split / "images"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {split} images directory: {path}")
    return path


def labels_dir_from_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Use --overwrite to replace it.")
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def materialize(src: Path, dst: Path, mode: str) -> None:
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


def rewrite_label_file(src_label: Path, dst_label: Path, old_to_new: dict[int, int]) -> tuple[int, Counter[int]]:
    kept: list[str] = []
    counts: Counter[int] = Counter()
    if src_label.exists():
        for line in src_label.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_id = int(float(parts[0]))
            if old_id not in old_to_new:
                continue
            new_id = old_to_new[old_id]
            kept.append(" ".join([str(new_id), *parts[1:5]]))
            counts[new_id] += 1
    dst_label.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept), counts


def convert_split(
    source: Path,
    output: Path,
    data: dict,
    split: str,
    max_images: int,
    old_to_new: dict[int, int],
    copy_mode: str,
    include_empty: bool,
) -> dict:
    images_dir = split_images_dir(source, data, split)
    labels_dir = labels_dir_from_images(images_dir)
    images = [
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    random.shuffle(images)

    stats = Counter()
    class_counts: Counter[int] = Counter()
    for image_path in images:
        if max_images and stats["images"] >= max_images:
            break
        label_path = labels_dir / f"{image_path.stem}.txt"
        tmp_label = output / "labels" / split / f"{image_path.stem}.txt"
        boxes, counts = rewrite_label_file(label_path, tmp_label, old_to_new)
        if boxes == 0 and not include_empty:
            tmp_label.unlink(missing_ok=True)
            stats["empty_skipped"] += 1
            continue
        materialize(image_path, output / "images" / split / image_path.name, copy_mode)
        stats["images"] += 1
        stats["boxes"] += boxes
        class_counts.update(counts)
        if stats["images"] % 1000 == 0:
            print(f"{split}: images={stats['images']} boxes={stats['boxes']}")

    return {
        "images": stats["images"],
        "boxes": stats["boxes"],
        "empty_skipped": stats["empty_skipped"],
        "class_counts": dict(class_counts),
    }


def write_dataset_yaml(output: Path, class_names: list[str]) -> None:
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
    source = Path(args.source)
    output = Path(args.output)
    selected_names = [name.strip() for name in args.classes.split(",") if name.strip()]
    random.seed(args.seed)

    data, source_names = load_data_yaml(source)
    missing = [name for name in selected_names if name not in source_names]
    if missing:
        raise SystemExit(f"Class names not found in source data.yaml: {missing}. Source names: {source_names}")
    old_to_new = {source_names.index(name): index for index, name in enumerate(selected_names)}

    prepare_output(output, args.overwrite)
    write_dataset_yaml(output, selected_names)

    summary = {
        "source": str(source),
        "output": str(output),
        "source_names": source_names,
        "selected_names": selected_names,
        "old_to_new": old_to_new,
        "splits": {},
    }
    for split, max_images in (("train", args.max_train), ("val", args.max_val)):
        split_stats = convert_split(
            source=source,
            output=output,
            data=data,
            split=split,
            max_images=max_images,
            old_to_new=old_to_new,
            copy_mode=args.copy_mode,
            include_empty=args.include_empty,
        )
        summary["splits"][split] = split_stats
        print(f"{split}: {split_stats}")

    (output / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Filtered YOLO dataset saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
