from __future__ import annotations

from .detector import Detection


def filter_oncoming_detections(
    detections: tuple[Detection, ...],
    frame_shape: tuple[int, int],
    enabled: bool,
    side: str,
    split_x_ratio: float,
    min_y_ratio: float,
) -> tuple[Detection, ...]:
    if not enabled:
        return detections

    height, width = frame_shape
    split_x = width * _clamp_ratio(split_x_ratio)
    min_y = height * _clamp_ratio(min_y_ratio)
    side = side.lower().strip()
    if side not in {"left", "right"}:
        raise ValueError("oncoming_side must be 'left' or 'right'")

    filtered: list[Detection] = []
    for detection in detections:
        x1, _, x2, y2 = detection.box
        bottom_center_x = (x1 + x2) / 2.0
        if y2 < min_y:
            continue
        if side == "left" and bottom_center_x < split_x:
            filtered.append(detection)
        elif side == "right" and bottom_center_x >= split_x:
            filtered.append(detection)
    return tuple(filtered)


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
