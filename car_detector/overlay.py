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


def draw_status(frame, lines: list[str]) -> None:
    x, y = 10, 24
    line_height = 24
    for index, text in enumerate(lines):
        pos = (x, y + index * line_height)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)


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
