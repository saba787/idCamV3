"""Geometric document quadrilateral detection on preview frames."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from idcam.vision.motion import normalize_gray, to_gray
from idcam.vision.sharpness import clip_roi


@dataclass(frozen=True)
class DocumentGeometry:
    """Detected document quadrilateral in preview pixel coordinates."""

    corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    width: int
    height: int
    rotation_deg: float
    aspect_ratio: float
    area_fraction: float
    confidence: float
    edge_strength: float
    rectangularity: float


def order_corners(pts: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    work = pts.reshape(4, 2).astype(np.float32)
    s = work.sum(axis=1)
    diff = np.diff(work, axis=1).ravel()
    tl = work[np.argmin(s)]
    br = work[np.argmax(s)]
    tr = work[np.argmin(diff)]
    bl = work[np.argmax(diff)]
    return (
        (int(round(tl[0])), int(round(tl[1]))),
        (int(round(tr[0])), int(round(tr[1]))),
        (int(round(br[0])), int(round(br[1]))),
        (int(round(bl[0])), int(round(bl[1]))),
    )


def corners_to_bbox(
    corners: tuple[tuple[int, int], ...],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return clip_roi((x0, y0, x1 - x0, y1 - y0), frame_shape)


def smooth_corners(
    prev: tuple[tuple[int, int], ...] | None,
    new: tuple[tuple[int, int], ...],
    alpha: float,
    frame_shape: tuple[int, ...],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    if prev is None or len(prev) != 4:
        return order_corners(np.array(new, dtype=np.float32))
    a = min(max(float(alpha), 0.05), 1.0)
    smoothed: list[tuple[int, int]] = []
    for i in range(4):
        px, py = prev[i]
        nx, ny = new[i]
        sx = int(round(px * (1 - a) + nx * a))
        sy = int(round(py * (1 - a) + ny * a))
        smoothed.append((sx, sy))
    ordered = order_corners(np.array(smoothed, dtype=np.float32))
    return ordered


def corner_motion(
    prev: tuple[tuple[int, int], ...],
    curr: tuple[tuple[int, int], ...],
) -> float:
    if len(prev) != 4 or len(curr) != 4:
        return 0.0
    dist = 0.0
    for i in range(4):
        dist += math.hypot(curr[i][0] - prev[i][0], curr[i][1] - prev[i][1])
    return dist / 4.0


def detect_document_geometry(
    frame: np.ndarray,
    *,
    min_area_fraction: float = 0.05,
    max_area_fraction: float = 0.85,
    min_rectangularity: float = 0.55,
    min_edge_strength: float = 8.0,
) -> DocumentGeometry | None:
    """
    Detect a paper-like quadrilateral using edges, morphology, and contour geometry.

    Does not use motion or a fixed center ROI as the document definition.
    """
    if frame is None or frame.size == 0:
        return None
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h < 32 or w < 32:
        return None

    gray = normalize_gray(to_gray(frame))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blur))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    if upper <= lower:
        upper = min(255, lower + 1)
    edges = cv2.Canny(blur, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(w * h)
    min_area = frame_area * float(min_area_fraction)
    max_area = frame_area * float(max_area_fraction)
    candidates: list[DocumentGeometry] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = order_corners(approx)
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            corners = order_corners(box)

        bbox = corners_to_bbox(corners, frame.shape)
        bx, by, bw, bh = bbox
        if bw < w * 0.12 or bh < h * 0.12:
            continue

        aspect = float(bw) / max(float(bh), 1.0)
        if aspect < 0.35 or aspect > 2.8:
            continue

        rectangularity = area / max(float(bw * bh), 1.0)
        if rectangularity < min_rectangularity:
            continue

        mask = np.zeros(gray.shape, dtype=np.uint8)
        pts = np.array(corners, dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edge_strength = float(np.mean(mag[mask > 0]))
        if edge_strength < min_edge_strength:
            continue

        area_fraction = area / frame_area
        cx = bx + bw // 2
        cy = by + bh // 2
        center_dist = math.hypot(cx - w / 2.0, cy - h / 2.0) / max(math.hypot(w / 2.0, h / 2.0), 1.0)
        center_score = max(0.0, 1.0 - center_dist)

        tl, tr, br, bl = corners
        width_top = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
        width_bottom = math.hypot(br[0] - bl[0], br[1] - bl[1])
        height_left = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
        height_right = math.hypot(br[0] - tr[0], br[1] - tr[1])
        doc_width = max(width_top, width_bottom)
        doc_height = max(height_left, height_right)
        rotation = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))

        aspect_target = 1.414
        aspect_score = max(0.0, 1.0 - min(abs(math.log(max(aspect, 1e-3) / aspect_target)), 1.2))

        confidence = (
            0.30 * min(1.0, area_fraction / 0.22)
            + 0.25 * min(1.0, rectangularity)
            + 0.20 * min(1.0, edge_strength / 90.0)
            + 0.15 * center_score
            + 0.10 * aspect_score
        )
        confidence = min(1.0, max(0.0, confidence))

        candidates.append(
            DocumentGeometry(
                corners=corners,
                bbox=bbox,
                center=(cx, cy),
                width=int(round(doc_width)),
                height=int(round(doc_height)),
                rotation_deg=float(rotation),
                aspect_ratio=float(aspect),
                area_fraction=float(area_fraction),
                confidence=float(confidence),
                edge_strength=float(edge_strength),
                rectangularity=float(rectangularity),
            )
        )

    if not candidates:
        return None
    return max(candidates, key=lambda c: c.confidence)
