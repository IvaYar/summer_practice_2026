from __future__ import annotations

import threading
from time import perf_counter

from .detector import InferenceResult, YoloOnnxDetector


class AsyncDetector:
    def __init__(self, detector: YoloOnnxDetector):
        self.detector = detector
        self._condition = threading.Condition()
        self._frame = None
        self._frame_id = 0
        self._stopped = False
        self._completed = 0
        self._skipped = 0
        self._latest = InferenceResult((), 0.0, perf_counter(), 0)
        self._thread = threading.Thread(target=self._run, name="detector-worker", daemon=True)
        self._thread.start()

    @property
    def completed(self) -> int:
        with self._condition:
            return self._completed

    @property
    def skipped(self) -> int:
        with self._condition:
            return self._skipped

    def submit(self, frame, frame_id: int) -> bool:
        with self._condition:
            if self._frame is not None:
                self._skipped += 1
                return False
            self._frame = frame.copy()
            self._frame_id = frame_id
            self._condition.notify()
            return True

    def latest(self) -> InferenceResult:
        with self._condition:
            return self._latest

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._frame is None and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
                frame = self._frame
                frame_id = self._frame_id
                self._frame = None

            result = self.detector.detect_timed(frame, frame_id)

            with self._condition:
                self._latest = result
                self._completed += 1
