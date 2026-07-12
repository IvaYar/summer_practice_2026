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


def draw_oncoming_boundary(
    frame,
    side: str,
    split_x_ratio: float,
    min_y_ratio: float,
) -> None:
    height, width = frame.shape[:2]
    split_x = int(round(width * max(0.0, min(1.0, float(split_x_ratio)))))
    min_y = int(round(height * max(0.0, min(1.0, float(min_y_ratio)))))
    color = (0, 255, 255)
    cv2.line(frame, (split_x, min_y), (split_x, height), color, 1, cv2.LINE_AA)
    if side == "left":
        cv2.line(frame, (0, min_y), (split_x, min_y), color, 1, cv2.LINE_AA)
    else:
        cv2.line(frame, (split_x, min_y), (width, min_y), color, 1, cv2.LINE_AA)


def draw_warning_line(
    frame,
    y_ratio: float,
    detections: tuple[Detection, ...] = (),
) -> bool:
    height, width = frame.shape[:2]
    y = int(round(height * max(0.0, min(1.0, float(y_ratio)))))
    crossed = any(_box_bottom_crosses_line(detection.box, y) for detection in detections)
    color = (0, 0, 255) if crossed else (0, 255, 255)
    cv2.line(frame, (0, y), (width, y), color, 3, cv2.LINE_AA)
    return crossed


def draw_status(frame, lines: list[str], scale: float = 0.62, thickness: int = 1) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.2, float(scale))
    thickness = max(1, int(thickness))
    (_, text_height), baseline = cv2.getTextSize("Ag", font, scale, thickness)
    x = 10
    y = max(24, text_height + baseline + 8)
    line_height = text_height + baseline + 10
    outline = max(thickness + 3, int(round(thickness + scale * 4)))
    for index, text in enumerate(lines):
        pos = (x, y + index * line_height)
        cv2.putText(frame, text, pos, font, scale, (0, 0, 0), outline, cv2.LINE_AA)
        cv2.putText(frame, text, pos, font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


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
