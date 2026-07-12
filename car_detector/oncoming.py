from __future__ import annotations

from .detector import Detection


def filter_oncoming_detections(
    detections: tuple[Detection, ...],
    frame_shape: tuple[int, int],
    enabled: bool,
    side: str,
    split_x_ratio: float,
    min_y_ratio: float,
    boundary: str = "diagonal",
    line_x1_ratio: float = 0.42,
    line_y1_ratio: float = 1.00,
    line_x2_ratio: float = 0.84,
    line_y2_ratio: float = 0.05,
) -> tuple[Detection, ...]:
    if not enabled:
        return detections

    height, width = frame_shape
    min_y = height * _clamp_ratio(min_y_ratio)
    side = side.lower().strip()
    boundary = boundary.lower().strip()
    if side not in {"left", "right"}:
        raise ValueError("oncoming_side must be 'left' or 'right'")
    if boundary not in {"vertical", "diagonal"}:
        raise ValueError("oncoming_boundary must be 'vertical' or 'diagonal'")

    filtered: list[Detection] = []
    for detection in detections:
        x1, _, x2, y2 = detection.box
        if y2 < min_y:
            continue

        bottom_center_x = (x1 + x2) / 2.0
        boundary_x = _boundary_x_at_y(
            y2 / max(1, height),
            width,
            boundary,
            split_x_ratio,
            line_x1_ratio,
            line_y1_ratio,
            line_x2_ratio,
            line_y2_ratio,
        )
        if side == "left" and bottom_center_x < boundary_x:
            filtered.append(detection)
        elif side == "right" and bottom_center_x >= boundary_x:
            filtered.append(detection)
    return tuple(filtered)


def _boundary_x_at_y(
    y_ratio: float,
    frame_width: int,
    boundary: str,
    split_x_ratio: float,
    line_x1_ratio: float,
    line_y1_ratio: float,
    line_x2_ratio: float,
    line_y2_ratio: float,
) -> float:
    if boundary == "vertical":
        return frame_width * _clamp_ratio(split_x_ratio)

    x1 = _clamp_ratio(line_x1_ratio)
    y1 = _clamp_ratio(line_y1_ratio)
    x2 = _clamp_ratio(line_x2_ratio)
    y2 = _clamp_ratio(line_y2_ratio)
    if abs(y2 - y1) < 1e-6:
        return frame_width * x1

    t = (_clamp_ratio(y_ratio) - y1) / (y2 - y1)
    x_at_y = x1 + (x2 - x1) * t
    return frame_width * _clamp_ratio(x_at_y)


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
