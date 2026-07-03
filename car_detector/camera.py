from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import cv2


class FrameSource(Protocol):
    name: str

    def read(self):
        ...

    def release(self) -> None:
        ...


class OpenCVSource:
    def __init__(self, camera_index: int, width: int, height: int, fps: int, video: str | None = None):
        self.name = video or f"opencv:{camera_index}"
        source = str(Path(video)) if video else camera_index
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {self.name}")
        if not video:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        self.capture.release()


class Picamera2Source:
    def __init__(self, width: int, height: int, fps: int):
        from picamera2 import Picamera2

        self.name = "picamera2"
        self.picam2 = Picamera2()
        main = {"size": (width, height), "format": "RGB888"}
        controls = {"FrameRate": fps}
        config = self.picam2.create_video_configuration(main=main, controls=controls, buffer_count=4)
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.0)

    def read(self):
        frame_rgb = self.picam2.capture_array("main")
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        self.picam2.stop()
        self.picam2.close()


def create_source(
    source: str,
    width: int,
    height: int,
    fps: int,
    camera_index: int = 0,
    video: str | None = None,
) -> FrameSource:
    if video:
        return OpenCVSource(camera_index=camera_index, width=width, height=height, fps=fps, video=video)

    if source in {"auto", "picamera2"}:
        try:
            return Picamera2Source(width=width, height=height, fps=fps)
        except Exception as exc:
            if source == "picamera2":
                raise RuntimeError(f"Picamera2 source failed: {exc}") from exc

    if source in {"auto", "opencv"}:
        return OpenCVSource(camera_index=camera_index, width=width, height=height, fps=fps)

    raise ValueError(f"Unknown source '{source}'. Use auto, picamera2, or opencv.")
