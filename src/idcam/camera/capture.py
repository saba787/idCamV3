"""Full-resolution image capture."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from idcam.config.schema import CameraConfig
from idcam.exceptions import CaptureError
from idcam.logging.setup import get_logger

logger = get_logger("camera.capture")


class CaptureCapableCamera(Protocol):
    def capture_still_to_file(self, path: Path) -> None: ...


class CaptureService:
    """Transactional still capture to disk."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config

    async def capture(self, camera: CaptureCapableCamera, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Capturing still image to %s", output_path)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(camera.capture_still_to_file, output_path),
                timeout=self.config.capture_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise CaptureError(f"Capture timed out after {self.config.capture_timeout_sec}s") from exc
        except Exception as exc:
            raise CaptureError(f"Capture failed: {exc}") from exc

        if not output_path.exists():
            raise CaptureError(f"Capture file not created: {output_path}")

        size = output_path.stat().st_size
        if size == 0:
            raise CaptureError("Capture file is empty")

        logger.info("Capture complete: %s (%d bytes)", output_path, size)
        return output_path
