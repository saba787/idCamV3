"""Post-capture image validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from idcam.config.schema import CameraConfig, QualityConfig
from idcam.exceptions import ImageValidationError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import laplacian_variance, mean_brightness

logger = get_logger("camera.quality")


@dataclass
class ValidationResult:
    width: int
    height: int
    file_bytes: int
    sharpness: float
    brightness: float
    sha256: str


class ImageValidator:
    """Validates captured JPEG files before upload."""

    def __init__(self, quality_config: QualityConfig, camera_config: CameraConfig | None = None) -> None:
        self.config = quality_config
        self.camera_config = camera_config

    def validate(self, path: Path) -> ValidationResult:
        if not path.exists():
            raise ImageValidationError(f"Image file not found: {path}")
        if not path.is_file():
            raise ImageValidationError(f"Not a regular file: {path}")

        file_bytes = path.stat().st_size
        if file_bytes < self.config.min_file_bytes:
            raise ImageValidationError(f"File too small: {file_bytes} bytes")
        if file_bytes > self.config.max_file_bytes:
            raise ImageValidationError(f"File too large: {file_bytes} bytes")

        sha256 = self._compute_sha256(path)

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            raise ImageValidationError(f"JPEG corrupt or undecodable: {exc}") from exc

        try:
            with Image.open(path) as img:
                img.load()
                width, height = img.size
        except Exception as exc:
            raise ImageValidationError(f"JPEG truncated or incomplete: {exc}") from exc

        frame = cv2.imread(str(path))
        if frame is None:
            raise ImageValidationError("OpenCV could not decode image")

        cv_height, cv_width = frame.shape[:2]
        if cv_width != width or cv_height != height:
            raise ImageValidationError("Image dimension mismatch between PIL and OpenCV")

        if width < self.config.min_width or height < self.config.min_height:
            raise ImageValidationError(f"Dimensions too small: {width}x{height}")

        if self.config.require_expected_dimensions and self.camera_config is not None:
            tol = self.config.dimension_tolerance
            ew, eh = self.camera_config.capture_width, self.camera_config.capture_height
            if abs(width - ew) > tol or abs(height - eh) > tol:
                raise ImageValidationError(
                    f"Dimensions {width}x{height} do not match expected {ew}x{eh} (tol={tol})"
                )

        sharpness = laplacian_variance(frame)
        brightness = mean_brightness(frame)

        logger.info(
            "Sharpness check: score=%.1f min_required=%.1f",
            sharpness,
            self.config.min_sharpness,
        )

        if sharpness < self.config.min_sharpness:
            raise ImageValidationError(
                f"Image too blurry: sharpness={sharpness:.1f} min={self.config.min_sharpness:.1f}"
            )
        if brightness < self.config.min_brightness:
            raise ImageValidationError(f"Image too dark: brightness={brightness:.1f}")
        if brightness > self.config.max_brightness:
            raise ImageValidationError(f"Image too bright: brightness={brightness:.1f}")

        if float(np.std(frame)) < 5.0:
            raise ImageValidationError("Image appears uniform/corrupted")

        result = ValidationResult(
            width=width,
            height=height,
            file_bytes=file_bytes,
            sharpness=sharpness,
            brightness=brightness,
            sha256=sha256,
        )
        logger.info(
            "Image validated: %dx%d, %d bytes, sharpness=%.1f, sha256=%s",
            width,
            height,
            file_bytes,
            sharpness,
            sha256[:16],
        )
        return result

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
