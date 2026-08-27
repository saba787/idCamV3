"""Document presence detection from preview frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from idcam.config.schema import DetectionConfig
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import clip_roi, inner_roi

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
    reject_reason: str = ""
    roi_width: int = 0
    roi_height: int = 0
    frame_width: int = 0
    frame_height: int = 0
    gray_std: float = 0.0
    gray_mean: float = 0.0
    frame_motion: float = 0.0
    document_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    focus_roi: tuple[int, int, int, int] = (0, 0, 0, 0)


def gradient_edge_density(gray: np.ndarray, magnitude_threshold: float = 30.0) -> float:
    """Fraction of ROI pixels with Sobel magnitude above ``magnitude_threshold``."""
    if gray.size == 0:
        return 0.0
    work = np.ascontiguousarray(gray)
    if work.dtype != np.uint8:
        work = np.clip(work, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.count_nonzero(mag > magnitude_threshold)) / mag.size


class DocumentDetector:
    """
    Detects document presence using frame differencing and edge density in ROI.

    Uses temporal hysteresis: separate enter/lose thresholds and consecutive-frame
    counters so exposure noise does not flap detection state.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._baseline: np.ndarray | None = None
        self._roi_slice: tuple[slice, slice] | None = None
        self._present_frames = 0
        self._absent_frames = 0
        self._detected_latched = False
        self._last_motion = 0.0
        self._last_edge_density = 0.0
        self._last_reject_reason = "no_frame"
        self._last_roi_size = (0, 0)
        self._last_frame_size = (0, 0)
        self._last_gray_std = 0.0
        self._last_gray_mean = 0.0
        self._prev_gray: np.ndarray | None = None
        self._last_frame_motion = 0.0
        self._last_document_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._last_focus_roi: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def last_motion(self) -> float:
        return self._last_motion

    @property
    def last_edge_density(self) -> float:
        return self._last_edge_density

    @property
    def is_latched(self) -> bool:
        return self._detected_latched

    def roi_rect(self, frame_shape: tuple[int, ...] | None = None) -> tuple[int, int, int, int]:
        """Centered detection ROI as (x, y, width, height) in frame pixels."""
        if frame_shape is not None and len(frame_shape) >= 2:
            h, w = int(frame_shape[0]), int(frame_shape[1])
        else:
            w, h = self._last_frame_size
        frac = self.config.roi_center_fraction
        rh, rw = int(h * frac), int(w * frac)
        y0, x0 = (h - rh) // 2, (w - rw) // 2
        return int(x0), int(y0), int(rw), int(rh)

    def document_preview_roi(self, frame: np.ndarray | None = None) -> tuple[int, int, int, int]:
        """Document bounding box in preview pixels (x, y, width, height)."""
        if frame is None:
            if self._last_document_roi != (0, 0, 0, 0):
                return self._last_document_roi
            return self.roi_rect()
        center = self.roi_rect(frame.shape)
        tight = self._tight_document_box(frame, center)
        self._last_document_roi = clip_roi(tight, frame.shape)
        self._last_focus_roi = inner_roi(
            self._last_document_roi,
            inset_fraction=self.config.focus_roi_inset_fraction,
            frame_shape=frame.shape,
        )
        return self._last_document_roi

    def focus_preview_roi(self, frame: np.ndarray | None = None) -> tuple[int, int, int, int]:
        """Inner document ROI for autofocus/sharpness — preview pixels."""
        if frame is not None:
            self.document_preview_roi(frame)
        if self._last_focus_roi != (0, 0, 0, 0):
            return self._last_focus_roi
        if frame is not None:
            center = self.roi_rect(frame.shape)
            return inner_roi(
                center,
                inset_fraction=self.config.focus_roi_inset_fraction,
                frame_shape=frame.shape,
            )
        return self.roi_rect()

    def _tight_document_box(
        self,
        frame: np.ndarray,
        center: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x0, y0, rw, rh = center
        roi = self._to_gray(frame[y0 : y0 + rh, x0 : x0 + rw])
        if roi.size == 0:
            return center
        work = np.ascontiguousarray(roi)
        if work.dtype != np.uint8:
            work = np.clip(work, 0, 255).astype(np.uint8)
        blur = cv2.GaussianBlur(work, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return center
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 0.08 * float(rw * rh):
            return center
        bx, by, bw, bh = cv2.boundingRect(contour)
        pad_x = max(4, int(bw * 0.06))
        pad_y = max(4, int(bh * 0.06))
        return (
            x0 + bx - pad_x,
            y0 + by - pad_y,
            bw + 2 * pad_x,
            bh + 2 * pad_y,
        )

    def _strong_present(self, motion: float, edge_density: float) -> bool:
        return (
            motion > self.config.motion_threshold
            and edge_density > self.config.edge_density_threshold
        )

    def _clearly_absent(self, motion: float, edge_density: float) -> bool:
        return (
            motion < self.config.motion_threshold * self.config.lose_motion_factor
            and edge_density < self.config.edge_density_threshold * self.config.lose_edge_factor
        )

    def _update_latch(self, motion: float, edge_density: float) -> bool:
        """Update hysteresis latch. Returns current latched presence."""
        strong = self._strong_present(motion, edge_density)
        if strong:
            self._present_frames += 1
            self._absent_frames = 0
        else:
            self._present_frames = 0
            if self._detected_latched:
                if self._clearly_absent(motion, edge_density):
                    self._absent_frames += 1
                # Small AE noise: hold latch, do not increment absent.
            else:
                self._absent_frames += 1

        if not self._detected_latched and self._present_frames >= self.config.detect_frames_required:
            self._detected_latched = True
            self._absent_frames = 0
            logger.debug(
                "event=document_latch_on frames=%d motion=%.2f edge=%.4f",
                self._present_frames,
                motion,
                edge_density,
            )

        if self._detected_latched and self._absent_frames >= self.config.lose_frames_required:
            self._detected_latched = False
            self._present_frames = 0
            logger.debug(
                "event=document_latch_off frames=%d motion=%.2f edge=%.4f",
                self._absent_frames,
                motion,
                edge_density,
            )

        return self._detected_latched

    def debug_info(self) -> DocumentDetectionDebug:
        detected = self._detected_latched
        return DocumentDetectionDebug(
            motion=self._last_motion,
            edge_density=self._last_edge_density,
            motion_threshold=self.config.motion_threshold,
            edge_density_threshold=self.config.edge_density_threshold,
            presence_frames=self._present_frames,
            stable_frames_required=self.config.stable_frames_required,
            detected=detected,
            reject_reason="" if detected else self._last_reject_reason,
            roi_width=self._last_roi_size[0],
            roi_height=self._last_roi_size[1],
            frame_width=self._last_frame_size[0],
            frame_height=self._last_frame_size[1],
            gray_std=self._last_gray_std,
            gray_mean=self._last_gray_mean,
            frame_motion=self._last_frame_motion,
            document_roi=self._last_document_roi,
            focus_roi=self._last_focus_roi,
        )

    def reset_baseline(self) -> None:
        self._baseline = None
        self._present_frames = 0
        self._absent_frames = 0
        self._detected_latched = False
        self._prev_gray = None
        self._last_frame_motion = 0.0
        self._last_document_roi = (0, 0, 0, 0)
        self._last_focus_roi = (0, 0, 0, 0)
        self._last_reject_reason = "no_frame"

    def update_baseline(self, frame: np.ndarray) -> None:
        gray = self._to_gray(self._get_roi(frame))
        if self._baseline is None:
            self._baseline = gray.astype(np.float32)
        else:
            alpha = self.config.baseline_alpha
            cv2.accumulateWeighted(gray, self._baseline, alpha)

    def _get_roi(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._last_frame_size = (int(w), int(h))
        frac = self.config.roi_center_fraction
        rh, rw = int(h * frac), int(w * frac)
        y0, x0 = (h - rh) // 2, (w - rw) // 2
        self._roi_slice = (slice(y0, y0 + rh), slice(x0, x0 + rw))
        self._last_roi_size = (int(rw), int(rh))
        return frame[y0 : y0 + rh, x0 : x0 + rw]

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            if frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    def _metrics(self, frame: np.ndarray) -> tuple[float, float]:
        gray = self._to_gray(self._get_roi(frame))
        gray = np.ascontiguousarray(gray)
        edge_density = gradient_edge_density(gray)
        self._last_gray_std = float(np.std(gray))
        self._last_gray_mean = float(np.mean(gray))
        self._last_edge_density = edge_density
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            self._last_frame_motion = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
        else:
            self._last_frame_motion = 0.0
        self._prev_gray = gray.copy()
        if self._baseline is None:
            self._baseline = gray.astype(np.float32)
            self._last_motion = 0.0
            self._last_reject_reason = "baseline_initialized"
            return 0.0, edge_density
        diff = cv2.absdiff(gray, self._baseline.astype(np.uint8))
        motion = float(np.mean(diff))
        self._last_motion = motion
        if not self._strong_present(motion, edge_density):
            if motion <= self.config.motion_threshold and edge_density <= self.config.edge_density_threshold:
                self._last_reject_reason = "below_motion_and_edge"
            elif motion <= self.config.motion_threshold:
                self._last_reject_reason = "below_motion_threshold"
            elif edge_density <= self.config.edge_density_threshold:
                self._last_reject_reason = "below_edge_density_threshold"
        else:
            self._last_reject_reason = ""
        return motion, edge_density

    def classify(self, frame: np.ndarray) -> DocumentPresence:
        motion, edge_density = self._metrics(frame)
        present = self._update_latch(motion, edge_density)

        if not present:
            if motion < self.config.motion_threshold * self.config.lose_motion_factor:
                gray = self._to_gray(self._get_roi(frame))
                alpha = self.config.baseline_alpha
                cv2.accumulateWeighted(gray, self._baseline, alpha)
            return DocumentPresence.NO_DOCUMENT

        self.document_preview_roi(frame)
        if self._present_frames < self.config.detect_frames_required and not self._detected_latched:
            return DocumentPresence.DOCUMENT_ENTERING
        if self._last_frame_motion > self.config.motion_threshold * 1.5:
            return DocumentPresence.DOCUMENT_MOVING
        if self._present_frames < self.config.stable_frames_required:
            return DocumentPresence.DOCUMENT_STABLE
        return DocumentPresence.DOCUMENT_READY

    def detect(self, frame: np.ndarray) -> bool:
        """True when document presence latch is on (hysteresis-confirmed)."""
        motion, edge_density = self._metrics(frame)
        if self._update_latch(motion, edge_density):
            self.document_preview_roi(frame)
            return True
        return False

    def is_document_present(self, frame: np.ndarray) -> bool:
        motion, edge_density = self._metrics(frame)
        return self._update_latch(motion, edge_density)

    def is_scene_empty(self, frame: np.ndarray) -> bool:
        """True when document is clearly absent (hysteresis lose thresholds)."""
        motion, edge_density = self._metrics(frame)
        if self._detected_latched:
            return self._clearly_absent(motion, edge_density)
        return not self._strong_present(motion, edge_density)

    def is_document_removed(self, frame: np.ndarray, consecutive_frames: int) -> bool:
        return consecutive_frames >= self.config.removal_frames_required
