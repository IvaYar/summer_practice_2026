from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - handled with a clear runtime error.
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "model": "models/yolo11n_320.onnx",
    "source": "auto",
    "video": None,
    "camera_index": 0,
    "width": 640,
    "height": 480,
    "fps": 30,
    "input_size": 320,
    "conf": 0.35,
    "iou": 0.45,
    "classes": "car,bus,truck",
    "threads": 4,
    "async_inference": True,
    "headless": False,
    "save": None,
    "window_x": 0,
    "window_y": 0,
    "print_every": 1.0,
    "max_frames": 0,
}


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config. Install requirements-runtime.txt first.")
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return data


def merge_options(config_path: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    options = dict(DEFAULT_CONFIG)
    options.update(load_config(config_path))
    for key, value in overrides.items():
        if key == "config" or value is None:
            continue
        options[key] = value
    return options


def parse_classes(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(value)
