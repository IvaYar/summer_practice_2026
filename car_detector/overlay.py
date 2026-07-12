from __future__ import annotations

from time import perf_counter

import cv2

from .detector import Detection

COLORS = {
    "car": (31, 173, 255),
    "bus": (53, 214, 133),
    "truck": (247, 184, 75),
    "motorcycle": (207, 135, 255),
}


def draw_detections(frame, detections: tuple[Detection, ...]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = COLORS.get(detection.class_name, (255, 255, 255))
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _draw_label(frame, label, x1, max(0, y1 - 8), color)


def draw_roi(frame, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), 1)


def draw_warning_line(
    frame,
    y_ratio: float,
    detections: tuple[Detection, ...] = (),
    label: str = "NO OVERTAKING",
) -> bool:
    height, width = frame.shape[:2]
    y = int(round(height * max(0.0, min(1.0, float(y_ratio)))))
    crossed = any(_box_bottom_crosses_line(detection.box, y) for detection in detections)
    color = (0, 0, 255) if crossed else (0, 255, 255)
    cv2.line(frame, (0, y), (width, y), color, 3, cv2.LINE_AA)
    _draw_label(frame, label, 10, max(24, y - 8), color)
    return crossed


def draw_status(frame, lines: list[str]) -> None:
    x, y = 10, 24
    line_height = 24
    for index, text in enumerate(lines):
        pos = (x, y + index * line_height)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)


def _box_bottom_crosses_line(box: tuple[int, int, int, int], line_y: int) -> bool:
    return box[3] >= line_y


def age_ms(timestamp: float) -> float:
    return max(0.0, (perf_counter() - timestamp) * 1000.0)


def _draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(height + baseline + 2, y)
    cv2.rectangle(frame, (x, y - height - baseline - 4), (x + width + 8, y + baseline), color, -1)
    cv2.putText(frame, text, (x + 4, y - 4), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
