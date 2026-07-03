from __future__ import annotations

from collections import deque
from time import perf_counter


class RateMeter:
    def __init__(self, window_seconds: float = 1.5):
        self.window_seconds = window_seconds
        self.samples: deque[float] = deque()

    def tick(self) -> float:
        now = perf_counter()
        self.samples.append(now)
        while self.samples and now - self.samples[0] > self.window_seconds:
            self.samples.popleft()
        if len(self.samples) < 2:
            return 0.0
        return (len(self.samples) - 1) / (self.samples[-1] - self.samples[0])
