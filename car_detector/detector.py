from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .coco import COCO_NAMES, class_ids_from_model_names


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
        model_class_names: tuple[str, ...] | None = None,
        threads: int = 0,
        geometry_filter: bool = True,
        min_box_area_ratio: float = 0.00002,
        max_box_area_ratio: float = 0.28,
        max_box_width_ratio: float = 0.78,
        max_box_height_ratio: float = 0.75,
        min_box_aspect_ratio: float = 0.20,
        max_box_aspect_ratio: float = 5.00,
        edge_margin_ratio: float = 0.02,
        edge_min_conf: float = 0.35,
        roi_enabled: bool = False,
        roi_x1_ratio: float = 0.00,
        roi_y1_ratio: float = 0.28,
        roi_x2_ratio: float = 1.00,
        roi_y2_ratio: float = 0.88,
    ):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. Run tools/export_yolo_onnx.py first or pass --model."
            )
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.model_class_names = model_class_names or COCO_NAMES
        self.target_class_ids = class_ids_from_model_names(class_names, self.model_class_names)
        self._target_class_ids_array = np.array(sorted(self.target_class_ids), dtype=np.int32)
        self.geometry_filter = bool(geometry_filter)
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.max_box_area_ratio = float(max_box_area_ratio)
        self.max_box_width_ratio = float(max_box_width_ratio)
        self.max_box_height_ratio = float(max_box_height_ratio)
        self.min_box_aspect_ratio = float(min_box_aspect_ratio)
        self.max_box_aspect_ratio = float(max_box_aspect_ratio)
        self.edge_margin_ratio = float(edge_margin_ratio)
        self.edge_min_conf = float(edge_min_conf)
        self.roi_enabled = bool(roi_enabled)
        self.roi_x1_ratio = float(roi_x1_ratio)
        self.roi_y1_ratio = float(roi_y1_ratio)
        self.roi_x2_ratio = float(roi_x2_ratio)
        self.roi_y2_ratio = float(roi_y2_ratio)
        cv2.setUseOptimized(True)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_names = self.net.getUnconnectedOutLayersNames()
        if threads and threads > 0:
            cv2.setNumThreads(int(threads))

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        detection_frame, offset = self._crop_to_roi(frame)
        input_image, ratio, pad = letterbox(detection_frame, self.input_size)
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (self.input_size, self.input_size), swapRB=True)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        detections = self._postprocess(output, detection_frame.shape[:2], ratio, pad)
        return self._offset_detections(detections, offset)

    def detect_timed(self, frame: np.ndarray, frame_id: int = 0) -> InferenceResult:
        started = perf_counter()
        detections = self.detect(frame)
        inference_ms = (perf_counter() - started) * 1000.0
        return InferenceResult(detections, inference_ms, perf_counter(), frame_id)

    def roi_box(self, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        height, width = frame_shape
        if not self.roi_enabled:
            return 0, 0, width, height

        x1 = int(np.clip(round(width * self.roi_x1_ratio), 0, width - 2))
        y1 = int(np.clip(round(height * self.roi_y1_ratio), 0, height - 2))
        x2 = int(np.clip(round(width * self.roi_x2_ratio), x1 + 2, width))
        y2 = int(np.clip(round(height * self.roi_y2_ratio), y1 + 2, height))
        return x1, y1, x2, y2

    def _crop_to_roi(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        x1, y1, x2, y2 = self.roi_box(frame.shape[:2])
        if x1 == 0 and y1 == 0 and x2 == frame.shape[1] and y2 == frame.shape[0]:
            return frame, (0, 0)
        return frame[y1:y2, x1:x2], (x1, y1)

    @staticmethod
    def _offset_detections(
        detections: tuple[Detection, ...],
        offset: tuple[int, int],
    ) -> tuple[Detection, ...]:
        offset_x, offset_y = offset
        if offset_x == 0 and offset_y == 0:
            return detections
        shifted = []
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            shifted.append(
                Detection(
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    box=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                )
            )
        return tuple(shifted)

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
        if predictions.ndim != 2:
            predictions = predictions.reshape(-1, predictions.shape[-1])

        rows, columns = predictions.shape
        if columns == 6:
            pass
        elif rows == 6 and columns != 6:
            predictions = predictions.T
        elif rows in {
            len(COCO_NAMES) + 4,
            len(COCO_NAMES) + 5,
            len(self.model_class_names) + 4,
            len(self.model_class_names) + 5,
        } and columns != rows:
            predictions = predictions.T

        fast_detections = self._postprocess_classic(predictions, original_shape, ratio, pad)
        if fast_detections is not None:
            return fast_detections

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []
        original_h, original_w = original_shape
        pad_x, pad_y = pad

        for row in predictions:
            if row.shape[0] < 6:
                continue

            parsed = self._parse_prediction(row)
            if parsed is None:
                continue
            best_class_id, confidence, box_mode = parsed
            if best_class_id not in self.target_class_ids:
                continue
            if confidence < self.conf_threshold:
                continue

            left, top, right, bottom = self._scale_box(row[:4], box_mode, ratio, pad_x, pad_y)

            left = int(np.clip(left, 0, original_w - 1))
            top = int(np.clip(top, 0, original_h - 1))
            right = int(np.clip(right, 0, original_w - 1))
            bottom = int(np.clip(bottom, 0, original_h - 1))
            if right <= left or bottom <= top:
                continue
            if self.geometry_filter and not self._passes_geometry_filter(
                left,
                top,
                right,
                bottom,
                confidence,
                original_w,
                original_h,
            ):
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
                    class_name=self.model_class_names[class_id],
                    confidence=confidences[int(index)],
                    box=(x, y, x + w, y + h),
                )
            )
        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return tuple(detections)

    def _postprocess_classic(
        self,
        predictions: np.ndarray,
        original_shape: tuple[int, int],
        ratio: float,
        pad: tuple[float, float],
    ) -> tuple[Detection, ...] | None:
        class_count = len(self.model_class_names)
        columns = predictions.shape[1]
        if columns == 4 + class_count:
            raw_boxes = predictions[:, :4].astype(np.float32, copy=False)
            class_scores = predictions[:, 4 : 4 + class_count].astype(np.float32, copy=False)
            objectness = None
        elif columns == 5 + class_count:
            raw_boxes = predictions[:, :4].astype(np.float32, copy=False)
            objectness = predictions[:, 4].astype(np.float32, copy=False)
            class_scores = predictions[:, 5 : 5 + class_count].astype(np.float32, copy=False)
        else:
            return None

        target_ids = self._target_class_ids_array
        target_ids = target_ids[target_ids < class_scores.shape[1]]
        if len(target_ids) == 0:
            return ()

        target_scores = class_scores[:, target_ids]
        best_target_indices = np.argmax(target_scores, axis=1)
        row_indices = np.arange(target_scores.shape[0])
        confidences = target_scores[row_indices, best_target_indices]
        if objectness is not None:
            confidences = confidences * objectness

        keep = confidences >= self.conf_threshold
        if not np.any(keep):
            return ()

        raw_boxes = raw_boxes[keep].copy()
        confidences = confidences[keep].astype(np.float32, copy=False)
        class_ids = target_ids[best_target_indices[keep]]

        if raw_boxes.size and float(np.max(raw_boxes)) <= 1.5:
            raw_boxes *= float(self.input_size)

        pad_x, pad_y = pad
        x_center = raw_boxes[:, 0]
        y_center = raw_boxes[:, 1]
        box_width = raw_boxes[:, 2]
        box_height = raw_boxes[:, 3]

        left = (x_center - box_width / 2.0 - pad_x) / ratio
        top = (y_center - box_height / 2.0 - pad_y) / ratio
        right = (x_center + box_width / 2.0 - pad_x) / ratio
        bottom = (y_center + box_height / 2.0 - pad_y) / ratio

        original_h, original_w = original_shape
        left = np.clip(left, 0, original_w - 1).astype(np.int32)
        top = np.clip(top, 0, original_h - 1).astype(np.int32)
        right = np.clip(right, 0, original_w - 1).astype(np.int32)
        bottom = np.clip(bottom, 0, original_h - 1).astype(np.int32)

        valid = (right > left) & (bottom > top)
        if self.geometry_filter:
            valid &= self._passes_geometry_filter_array(
                left,
                top,
                right,
                bottom,
                confidences,
                original_w,
                original_h,
            )
        if not np.any(valid):
            return ()

        left = left[valid]
        top = top[valid]
        right = right[valid]
        bottom = bottom[valid]
        confidences = confidences[valid]
        class_ids = class_ids[valid]

        nms_boxes = np.column_stack((left, top, right - left, bottom - top)).astype(np.int32)
        confidence_list = confidences.astype(float).tolist()
        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(),
            confidence_list,
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(indices) == 0:
            return ()

        detections = []
        for index in np.array(indices).reshape(-1):
            index = int(index)
            x, y, width, height = [int(value) for value in nms_boxes[index]]
            class_id = int(class_ids[index])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.model_class_names[class_id],
                    confidence=float(confidences[index]),
                    box=(x, y, x + width, y + height),
                )
            )
        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return tuple(detections)

    def _parse_prediction(self, row: np.ndarray) -> tuple[int, float, str] | None:
        if row.shape[0] == 6:
            confidence = float(row[4])
            class_id = int(round(float(row[5])))
            if class_id < 0 or class_id >= len(self.model_class_names):
                return None
            return class_id, confidence, "xyxy"

        class_scores, objectness = self._split_scores(row)
        available_ids = [class_id for class_id in self.target_class_ids if class_id < len(class_scores)]
        if not available_ids:
            return None
        class_id = max(available_ids, key=lambda candidate: float(class_scores[candidate]))
        confidence = float(class_scores[class_id]) * objectness
        return class_id, confidence, "xywh"

    def _scale_box(
        self,
        raw_box: np.ndarray,
        box_mode: str,
        ratio: float,
        pad_x: float,
        pad_y: float,
    ) -> tuple[float, float, float, float]:
        box = raw_box.astype(np.float32).copy()
        if float(np.max(box)) <= 1.5:
            box *= float(self.input_size)

        if box_mode == "xyxy":
            left = (float(box[0]) - pad_x) / ratio
            top = (float(box[1]) - pad_y) / ratio
            right = (float(box[2]) - pad_x) / ratio
            bottom = (float(box[3]) - pad_y) / ratio
            if right > left and bottom > top:
                return left, top, right, bottom

        x_center, y_center, width, height = map(float, box)
        left = (x_center - width / 2.0 - pad_x) / ratio
        top = (y_center - height / 2.0 - pad_y) / ratio
        right = (x_center + width / 2.0 - pad_x) / ratio
        bottom = (y_center + height / 2.0 - pad_y) / ratio
        return left, top, right, bottom

    def _passes_geometry_filter(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        confidence: float,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        width = right - left
        height = bottom - top
        frame_area = max(1, frame_width * frame_height)
        area_ratio = (width * height) / frame_area
        width_ratio = width / max(1, frame_width)
        height_ratio = height / max(1, frame_height)
        aspect_ratio = width / max(1, height)

        if area_ratio < self.min_box_area_ratio or area_ratio > self.max_box_area_ratio:
            return False
        if width_ratio > self.max_box_width_ratio or height_ratio > self.max_box_height_ratio:
            return False
        if aspect_ratio < self.min_box_aspect_ratio or aspect_ratio > self.max_box_aspect_ratio:
            return False

        margin_x = max(1, int(frame_width * self.edge_margin_ratio))
        margin_y = max(1, int(frame_height * self.edge_margin_ratio))
        touches_edge = (
            left <= margin_x
            or right >= frame_width - 1 - margin_x
            or bottom >= frame_height - 1 - margin_y
        )
        if touches_edge and confidence < self.edge_min_conf:
            return False

        return True

    def _passes_geometry_filter_array(
        self,
        left: np.ndarray,
        top: np.ndarray,
        right: np.ndarray,
        bottom: np.ndarray,
        confidence: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        width = right - left
        height = bottom - top
        frame_area = max(1, frame_width * frame_height)
        area_ratio = (width * height) / frame_area
        width_ratio = width / max(1, frame_width)
        height_ratio = height / max(1, frame_height)
        aspect_ratio = width / np.maximum(1, height)

        valid = (
            (area_ratio >= self.min_box_area_ratio)
            & (area_ratio <= self.max_box_area_ratio)
            & (width_ratio <= self.max_box_width_ratio)
            & (height_ratio <= self.max_box_height_ratio)
            & (aspect_ratio >= self.min_box_aspect_ratio)
            & (aspect_ratio <= self.max_box_aspect_ratio)
        )

        margin_x = max(1, int(frame_width * self.edge_margin_ratio))
        margin_y = max(1, int(frame_height * self.edge_margin_ratio))
        touches_edge = (
            (left <= margin_x)
            | (right >= frame_width - 1 - margin_x)
            | (bottom >= frame_height - 1 - margin_y)
        )
        valid &= ~(touches_edge & (confidence < self.edge_min_conf))
        return valid

    def _split_scores(self, row: np.ndarray) -> tuple[np.ndarray, float]:
        class_count = len(self.model_class_names)
        if row.shape[0] == 5 + class_count:
            return row[5:], float(row[4])
        if row.shape[0] == 4 + class_count:
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
