"""Image sharpness measurement."""

from __future__ import annotations

import cv2
import numpy as np


def roi_rect(frame_shape: tuple[int, ...], roi_fraction: float = 0.7) -> tuple[int, int, int, int]:
    """Centered ROI as (x, y, width, height) in frame pixels."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    frac = min(max(float(roi_fraction), 0.05), 1.0)
    rh, rw = max(1, int(h * frac)), max(1, int(w * frac))
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    return int(x0), int(y0), int(rw), int(rh)


def laplacian_variance(frame: np.ndarray, roi_fraction: float = 0.7) -> float:
    """
    Compute Laplacian variance as sharpness metric on the document ROI.
    Higher values indicate sharper images.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    x0, y0, rw, rh = roi_rect(gray.shape, roi_fraction)
    roi = gray[y0 : y0 + rh, x0 : x0 + rw]

    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    return float(laplacian.var())


def mean_brightness(frame: np.ndarray) -> float:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    return float(np.mean(gray))
