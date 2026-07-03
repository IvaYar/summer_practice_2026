from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .coco import COCO_NAMES, class_ids_from_names


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class InferenceResult:
    detections: tuple[Detection, ...]
    inference_ms: float
    timestamp: float
    frame_id: int


class YoloOnnxDetector:
    def __init__(
        self,
        model_path: str,
        input_size: int,
        conf_threshold: float,
        iou_threshold: float,
        class_names: tuple[str, ...],
        threads: int = 0,
    ):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. Run tools/export_yolo_onnx.py first or pass --model."
            )
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.target_class_ids = class_ids_from_names(class_names)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_names = self.net.getUnconnectedOutLayersNames()
        if threads and threads > 0:
            cv2.setNumThreads(int(threads))

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        input_image, ratio, pad = letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (self.input_size, self.input_size), swapRB=True)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return self._postprocess(output, frame.shape[:2], ratio, pad)

    def detect_timed(self, frame: np.ndarray, frame_id: int = 0) -> InferenceResult:
        started = perf_counter()
        detections = self.detect(frame)
        inference_ms = (perf_counter() - started) * 1000.0
        return InferenceResult(detections, inference_ms, perf_counter(), frame_id)

    def _postprocess(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int],
        ratio: float,
        pad: tuple[float, float],
    ) -> tuple[Detection, ...]:
        predictions = np.squeeze(output)
        if predictions.ndim == 1:
            predictions = np.expand_dims(predictions, axis=0)
        if predictions.shape[0] <= 256 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []
        original_h, original_w = original_shape
        pad_x, pad_y = pad

        for row in predictions:
            if row.shape[0] < 6:
                continue
            box_xywh = row[:4]
            class_scores, objectness = self._split_scores(row)
            available_ids = [class_id for class_id in self.target_class_ids if class_id < len(class_scores)]
            if not available_ids:
                continue
            best_class_id = max(available_ids, key=lambda class_id: float(class_scores[class_id]))
            confidence = float(class_scores[best_class_id]) * objectness
            if confidence < self.conf_threshold:
                continue

            x_center, y_center, width, height = map(float, box_xywh)
            left = (x_center - width / 2.0 - pad_x) / ratio
            top = (y_center - height / 2.0 - pad_y) / ratio
            right = (x_center + width / 2.0 - pad_x) / ratio
            bottom = (y_center + height / 2.0 - pad_y) / ratio

            left = int(np.clip(left, 0, original_w - 1))
            top = int(np.clip(top, 0, original_h - 1))
            right = int(np.clip(right, 0, original_w - 1))
            bottom = int(np.clip(bottom, 0, original_h - 1))
            if right <= left or bottom <= top:
                continue

            boxes.append([left, top, right - left, bottom - top])
            confidences.append(confidence)
            class_ids.append(best_class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        if len(indices) == 0:
            return ()

        flat_indices = np.array(indices).reshape(-1)
        detections = []
        for index in flat_indices:
            x, y, w, h = boxes[int(index)]
            class_id = class_ids[int(index)]
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=COCO_NAMES[class_id],
                    confidence=confidences[int(index)],
                    box=(x, y, x + w, y + h),
                )
            )
        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return tuple(detections)

    @staticmethod
    def _split_scores(row: np.ndarray) -> tuple[np.ndarray, float]:
        coco_count = len(COCO_NAMES)
        if row.shape[0] == 5 + coco_count:
            return row[5:], float(row[4])
        if row.shape[0] == 4 + coco_count:
            return row[4:], 1.0
        if row.shape[0] > 85:
            return row[5:], float(row[4])
        return row[4:], 1.0


def letterbox(frame: np.ndarray, size: int, color: tuple[int, int, int] = (114, 114, 114)):
    height, width = frame.shape[:2]
    ratio = min(size / width, size / height)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_x = (size - new_width) / 2.0
    pad_y = (size - new_height) / 2.0
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, ratio, (left, top)
