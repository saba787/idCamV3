"""Document presence detection from preview frames."""

from __future__ import annotations

import cv2
import numpy as np

from idcam.config.schema import DetectionConfig


class DocumentDetector:
    """
    Detects document presence using frame differencing and edge density in ROI.

    Distinguishes: no document, entering, moving (handled by stability), stable.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._baseline: np.ndarray | None = None
        self._roi_slice: tuple[slice, slice] | None = None

    def reset_baseline(self) -> None:
        self._baseline = None

    def _get_roi(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        frac = self.config.roi_center_fraction
        rh, rw = int(h * frac), int(w * frac)
        y0, x0 = (h - rh) // 2, (w - rw) // 2
        self._roi_slice = (slice(y0, y0 + rh), slice(x0, x0 + rw))
        return frame[y0 : y0 + rh, x0 : x0 + rw]

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    def update_baseline(self, frame: np.ndarray) -> None:
        gray = self._to_gray(self._get_roi(frame))
        if self._baseline is None:
            self._baseline = gray.astype(np.float32)
        else:
            alpha = self.config.baseline_alpha
            cv2.accumulateWeighted(gray, self._baseline, alpha)

    def is_scene_empty(self, frame: np.ndarray) -> bool:
        """True when scene matches baseline (document removed)."""
        if self._baseline is None:
            return True
        gray = self._to_gray(self._get_roi(frame))
        diff = cv2.absdiff(gray, self._baseline.astype(np.uint8))
        motion = float(np.mean(diff))
        return motion < self.config.motion_threshold * 0.5

    def detect(self, frame: np.ndarray) -> bool:
        """
        Returns True if a document-like object is present in the ROI.
        Updates baseline slowly when scene is stable and empty.
        """
        gray = self._to_gray(self._get_roi(frame))

        if self._baseline is None:
            self._baseline = gray.astype(np.float32)
            return False

        diff = cv2.absdiff(gray, self._baseline.astype(np.uint8))
        motion = float(np.mean(diff))

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size

        present = motion > self.config.motion_threshold and edge_density > self.config.edge_density_threshold

        if not present and motion < self.config.motion_threshold * 0.5:
            alpha = self.config.baseline_alpha
            cv2.accumulateWeighted(gray, self._baseline, alpha)

        return present
