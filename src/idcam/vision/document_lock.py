"""Stable document lock with temporally smoothed ROI."""

from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

from idcam.config.schema import DetectionConfig
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import clip_roi, inner_roi

logger = get_logger("vision.document_lock")


@dataclass
class DocumentLockState:
    locked: bool = False
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    focus_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    locked_at: float = 0.0
    updates: int = 0


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame


def find_document_candidate(
    frame: np.ndarray,
    center_roi: tuple[int, int, int, int],
    min_area_fraction: float = 0.08,
) -> tuple[tuple[int, int, int, int], float] | None:
    """Return (bbox, confidence) in preview pixels or None."""
    x0, y0, rw, rh = center_roi
    patch = _to_gray(frame[y0 : y0 + rh, x0 : x0 + rw])
    if patch.size == 0:
        return None
    work = np.ascontiguousarray(patch)
    if work.dtype != np.uint8:
        work = np.clip(work, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(work, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    min_area = min_area_fraction * float(rw * rh)
    if area < min_area:
        return None
    bx, by, bw, bh = cv2.boundingRect(contour)
    pad_x = max(4, int(bw * 0.05))
    pad_y = max(4, int(bh * 0.05))
    bbox = (
        x0 + bx - pad_x,
        y0 + by - pad_y,
        bw + 2 * pad_x,
        bh + 2 * pad_y,
    )
    confidence = min(1.0, area / max(min_area, 1.0))
    return clip_roi(bbox, frame.shape), confidence


def smooth_roi(
    prev: tuple[int, int, int, int],
    new: tuple[int, int, int, int],
    alpha: float,
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """EMA smooth bbox corners to prevent per-frame jumps."""
    if prev == (0, 0, 0, 0):
        return clip_roi(new, frame_shape)
    a = min(max(float(alpha), 0.05), 1.0)
    px, py, pw, ph = prev
    nx, ny, nw, nh = new
    sx = int(round(px * (1 - a) + nx * a))
    sy = int(round(py * (1 - a) + ny * a))
    sw = int(round(pw * (1 - a) + nw * a))
    sh = int(round(ph * (1 - a) + nh * a))
    return clip_roi((sx, sy, sw, sh), frame_shape)


class DocumentLock:
    """One smoothed document ROI for the active capture pipeline."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._state = DocumentLockState()

    @property
    def state(self) -> DocumentLockState:
        return self._state

    @property
    def is_locked(self) -> bool:
        return self._state.locked

    @property
    def roi(self) -> tuple[int, int, int, int]:
        return self._state.roi

    @property
    def focus_roi(self) -> tuple[int, int, int, int]:
        return self._state.focus_roi

    def lock(self, frame: np.ndarray, candidate: tuple[int, int, int, int], confidence: float) -> None:
        roi = clip_roi(candidate, frame.shape)
        focus = inner_roi(
            roi,
            inset_fraction=self.config.focus_roi_inset_fraction,
            frame_shape=frame.shape,
        )
        self._state = DocumentLockState(
            locked=True,
            roi=roi,
            focus_roi=focus,
            confidence=confidence,
            locked_at=time.monotonic(),
            updates=1,
        )
        logger.info(
            "event=pipeline_stage stage=document_locked roi=%d,%d,%d,%d focus_roi=%d,%d,%d,%d confidence=%.2f",
            *roi,
            *focus,
            confidence,
        )

    def update(self, frame: np.ndarray, candidate: tuple[int, int, int, int] | None) -> None:
        if not self._state.locked:
            return
        if candidate is None:
            return
        smoothed = smooth_roi(
            self._state.roi,
            candidate,
            self.config.roi_smooth_alpha,
            frame.shape,
        )
        focus = inner_roi(
            smoothed,
            inset_fraction=self.config.focus_roi_inset_fraction,
            frame_shape=frame.shape,
        )
        self._state.roi = smoothed
        self._state.focus_roi = focus
        self._state.updates += 1

    def release(self) -> None:
        if self._state.locked:
            logger.info("event=pipeline_stage stage=document_lock_released")
        self._state = DocumentLockState()
