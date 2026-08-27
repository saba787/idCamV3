"""Scene stability detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from idcam.config.schema import DetectionConfig


@dataclass(frozen=True)
class StabilityDebug:
    """Latest stability metrics for debug overlay."""

    motion: float
    threshold: float
    stable_count: int
    stable_frames_required: int
    stable: bool


class StabilityDetector:
    """Tracks consecutive stable frames with enter/exit hysteresis."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._stable_count = 0
        self._stable_latched = False
        self._prev_gray: np.ndarray | None = None
        self._last_motion = 0.0

    def reset(self) -> None:
        self._stable_count = 0
        self._stable_latched = False
        self._prev_gray = None
        self._last_motion = 0.0

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    def _roi(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        frac = self.config.roi_center_fraction
        rh, rw = int(h * frac), int(w * frac)
        y0, x0 = (h - rh) // 2, (w - rw) // 2
        return frame[y0 : y0 + rh, x0 : x0 + rw]

    def update(self, frame: np.ndarray) -> bool:
        """True when document has been stable for required consecutive frames."""
        gray = self._to_gray(self._roi(frame))

        if self._prev_gray is None:
            self._prev_gray = gray
            self._stable_count = 0
            return False

        diff = cv2.absdiff(gray, self._prev_gray)
        motion = float(np.mean(diff))
        self._last_motion = motion
        self._prev_gray = gray

        enter = self.config.motion_threshold
        exit_threshold = enter * self.config.stability_exit_factor

        if motion < enter:
            self._stable_count += 1
            if self._stable_count >= self.config.stable_frames_required:
                self._stable_latched = True
        elif self._stable_latched and motion < enter * 1.5:
            # Hold latch through small preview noise once stable.
            pass
        elif motion >= exit_threshold:
            self._stable_count = 0
            self._stable_latched = False
        elif motion < enter * 1.5:
            self._stable_count = max(0, self._stable_count - 1)
            if self._stable_count == 0:
                self._stable_latched = False
        else:
            self._stable_count = 0
            self._stable_latched = False

        return self._stable_latched

    @property
    def stable_count(self) -> int:
        return self._stable_count

    def debug_info(self) -> StabilityDebug:
        return StabilityDebug(
            motion=self._last_motion,
            threshold=self.config.motion_threshold,
            stable_count=self._stable_count,
            stable_frames_required=self.config.stable_frames_required,
            stable=self._stable_latched,
        )
