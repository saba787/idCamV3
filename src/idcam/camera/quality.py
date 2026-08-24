"""Post-capture image validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from idcam.config.schema import QualityConfig
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


class ImageValidator:
    """Validates captured JPEG files before upload."""

    def __init__(self, config: QualityConfig) -> None:
        self.config = config

    def validate(self, path: Path) -> ValidationResult:
        if not path.exists():
            raise ImageValidationError(f"Image file not found: {path}")

        file_bytes = path.stat().st_size
        if file_bytes < self.config.min_file_bytes:
            raise ImageValidationError(f"File too small: {file_bytes} bytes")
        if file_bytes > self.config.max_file_bytes:
            raise ImageValidationError(f"File too large: {file_bytes} bytes")

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            raise ImageValidationError(f"JPEG corrupt or undecodable: {exc}") from exc

        # Re-open after verify() invalidates the image object
        frame = cv2.imread(str(path))
        if frame is None:
            raise ImageValidationError("OpenCV could not decode image")

        height, width = frame.shape[:2]
        if width < self.config.min_width or height < self.config.min_height:
            raise ImageValidationError(f"Dimensions too small: {width}x{height}")

        sharpness = laplacian_variance(frame)
        brightness = mean_brightness(frame)

        if sharpness < self.config.min_sharpness:
            raise ImageValidationError(f"Image too blurry: sharpness={sharpness:.1f}")
        if brightness < self.config.min_brightness:
            raise ImageValidationError(f"Image too dark: brightness={brightness:.1f}")
        if brightness > self.config.max_brightness:
            raise ImageValidationError(f"Image too bright: brightness={brightness:.1f}")

        # Basic corruption check: not a uniform image
        if float(np.std(frame)) < 5.0:
            raise ImageValidationError("Image appears uniform/corrupted")

        result = ValidationResult(
            width=width,
            height=height,
            file_bytes=file_bytes,
            sharpness=sharpness,
            brightness=brightness,
        )
        logger.info(
            "Image validated: %dx%d, %d bytes, sharpness=%.1f, brightness=%.1f",
            width,
            height,
            file_bytes,
            sharpness,
            brightness,
        )
        return result
