"""Document presence detection from preview frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from idcam.config.schema import DetectionConfig
from idcam.logging.setup import get_logger

logger = get_logger("vision.document_detector")


class DocumentPresence(str, Enum):
    NO_DOCUMENT = "no_document"
    DOCUMENT_ENTERING = "document_entering"
    DOCUMENT_MOVING = "document_moving"
    DOCUMENT_STABLE = "document_stable"
    DOCUMENT_READY = "document_ready"


@dataclass(frozen=True)
class DocumentDetectionDebug:
    """Latest detector metrics for debug overlay. Does not affect detection."""

    motion: float
    edge_density: float
    motion_threshold: float
    edge_density_threshold: float
    presence_frames: int
    stable_frames_required: int
    detected: bool


class DocumentDetector:
    """
    Detects document presence using frame differencing and edge density in ROI.

    Distinguishes: no document, entering, moving, stable, ready.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._baseline: np.ndarray | None = None
        self._roi_slice: tuple[slice, slice] | None = None
        self._presence_frames = 0
        self._last_motion = 0.0
        self._last_edge_density = 0.0

    @property
    def last_motion(self) -> float:
        return self._last_motion

    @property
    def last_edge_density(self) -> float:
        return self._last_edge_density

    def debug_info(self) -> DocumentDetectionDebug:
        """Return last computed metrics. Does not run detection or mutate state."""
        detected = (
            self._last_motion > self.config.motion_threshold
            and self._last_edge_density > self.config.edge_density_threshold
        )
        return DocumentDetectionDebug(
            motion=self._last_motion,
            edge_density=self._last_edge_density,
            motion_threshold=self.config.motion_threshold,
            edge_density_threshold=self.config.edge_density_threshold,
            presence_frames=self._presence_frames,
            stable_frames_required=self.config.stable_frames_required,
            detected=detected,
        )

    def reset_baseline(self) -> None:
        self._baseline = None
        self._presence_frames = 0

    def update_baseline(self, frame: np.ndarray) -> None:
        """Update baseline from a stable empty scene."""
        gray = self._to_gray(self._get_roi(frame))
        if self._baseline is None:
            self._baseline = gray.astype(np.float32)
        else:
            alpha = self.config.baseline_alpha
            cv2.accumulateWeighted(gray, self._baseline, alpha)

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

    def _metrics(self, frame: np.ndarray) -> tuple[float, float]:
        gray = self._to_gray(self._get_roi(frame))
        if self._baseline is None:
            self._baseline = gray.astype(np.float32)
            return 0.0, 0.0
        diff = cv2.absdiff(gray, self._baseline.astype(np.uint8))
        motion = float(np.mean(diff))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        self._last_motion = motion
        self._last_edge_density = edge_density
        return motion, edge_density

    def classify(self, frame: np.ndarray) -> DocumentPresence:
        motion, edge_density = self._metrics(frame)
        present = motion > self.config.motion_threshold and edge_density > self.config.edge_density_threshold

        if not present:
            self._presence_frames = 0
            if motion < self.config.motion_threshold * 0.5:
                gray = self._to_gray(self._get_roi(frame))
                alpha = self.config.baseline_alpha
                cv2.accumulateWeighted(gray, self._baseline, alpha)
            return DocumentPresence.NO_DOCUMENT

        self._presence_frames += 1
        if self._presence_frames < 2:
            return DocumentPresence.DOCUMENT_ENTERING
        if motion > self.config.motion_threshold * 1.5:
            return DocumentPresence.DOCUMENT_MOVING
        if self._presence_frames < self.config.stable_frames_required:
            return DocumentPresence.DOCUMENT_STABLE
        return DocumentPresence.DOCUMENT_READY

    def detect(self, frame: np.ndarray) -> bool:
        """Returns True when document is present and entering detection."""
        presence = self.classify(frame)
        return presence in {
            DocumentPresence.DOCUMENT_ENTERING,
            DocumentPresence.DOCUMENT_MOVING,
            DocumentPresence.DOCUMENT_STABLE,
            DocumentPresence.DOCUMENT_READY,
        }

    def is_document_present(self, frame: np.ndarray) -> bool:
        presence = self.classify(frame)
        return presence != DocumentPresence.NO_DOCUMENT

    def is_scene_empty(self, frame: np.ndarray) -> bool:
        """True when scene matches baseline (document removed)."""
        motion, edge_density = self._metrics(frame)
        empty = (
            motion < self.config.motion_threshold * self.config.removal_motion_factor
            and edge_density < self.config.edge_density_threshold * self.config.removal_edge_factor
        )
        if empty:
            logger.debug(
                "Scene empty: motion=%.2f edge=%.4f (thresholds motion<%.2f edge<%.4f)",
                motion,
                edge_density,
                self.config.motion_threshold * self.config.removal_motion_factor,
                self.config.edge_density_threshold * self.config.removal_edge_factor,
            )
        return empty

    def is_document_removed(self, frame: np.ndarray, consecutive_frames: int) -> bool:
        """True when document has been absent for required consecutive frames."""
        return consecutive_frames >= self.config.removal_frames_required
