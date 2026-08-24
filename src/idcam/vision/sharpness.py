"""Image sharpness measurement."""

from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(frame: np.ndarray, roi_fraction: float = 0.7) -> float:
    """
    Compute Laplacian variance as sharpness metric.
    Higher values indicate sharper images.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    h, w = gray.shape
    rh, rw = int(h * roi_fraction), int(w * roi_fraction)
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    roi = gray[y0 : y0 + rh, x0 : x0 + rw]

    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    return float(laplacian.var())


def mean_brightness(frame: np.ndarray) -> float:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    return float(np.mean(gray))
