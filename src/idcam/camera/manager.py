"""Picamera2 camera manager with mock fallback for development."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from idcam.config.schema import CameraConfig
from idcam.exceptions import CameraError, CameraUnavailableError
from idcam.logging.setup import get_logger

logger = get_logger("camera.manager")

try:
    from picamera2 import Picamera2
    from libcamera import controls

    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None  # type: ignore
    controls = None  # type: ignore


class CameraManager:
    """
    Manages Picamera2 lifecycle: preview stream and full-res capture.

  Thread-safe via lock. All public methods may be called from executor threads.
    """

    def __init__(self, config: CameraConfig, *, use_mock: bool = False) -> None:
        self.config = config
        self._use_mock = use_mock or not PICAMERA2_AVAILABLE
        self._lock = threading.RLock()
        self._picam2: Any = None
        self._started = False
        self._preview_config: Any = None
        self._still_config: Any = None

    @property
    def is_mock(self) -> bool:
        return self._use_mock

    @property
    def is_started(self) -> bool:
        return self._started

    def initialize(self) -> None:
        with self._lock:
            if self._started:
                return
            if self._use_mock:
                logger.warning("Using mock camera (Picamera2 not available)")
                self._started = True
                return

            try:
                self._picam2 = Picamera2()
                self._preview_config = self._picam2.create_preview_configuration(
                    main={"size": (self.config.preview_width, self.config.preview_height), "format": "RGB888"},
                    buffer_count=4,
                )
                self._still_config = self._picam2.create_still_configuration(
                    main={
                        "size": (self.config.capture_width, self.config.capture_height),
                        "format": "RGB888",
                    },
                    buffer_count=2,
                )
                self._picam2.configure(self._preview_config)
                self._picam2.start()
                self._started = True
                logger.info(
                    "Camera initialized: preview=%dx%d capture=%dx%d",
                    self.config.preview_width,
                    self.config.preview_height,
                    self.config.capture_width,
                    self.config.capture_height,
                )
            except Exception as exc:
                self._started = False
                self._picam2 = None
                raise CameraUnavailableError(f"Camera initialization failed: {exc}") from exc

    def shutdown(self) -> None:
        with self._lock:
            if not self._started:
                return
            if self._picam2 is not None:
                try:
                    self._picam2.stop()
                    self._picam2.close()
                except Exception as exc:
                    logger.error("Error closing camera: %s", exc)
                finally:
                    self._picam2 = None
            self._started = False
            logger.info("Camera shut down")

    def restart(self) -> None:
        logger.info("Restarting camera")
        self.shutdown()
        time.sleep(1.0)
        self.initialize()

    def capture_preview_array(self) -> np.ndarray:
        with self._lock:
            if not self._started:
                raise CameraError("Camera not started")
            if self._use_mock:
                return self._mock_frame()
            assert self._picam2 is not None
            return self._picam2.capture_array("main")

    def capture_still_to_file(self, path: Path) -> None:
        with self._lock:
            if not self._started:
                raise CameraError("Camera not started")
            if self._use_mock:
                self._mock_still(path)
                return

            assert self._picam2 is not None
            was_preview = True
            try:
                self._picam2.stop()
                was_preview = False
                self._picam2.configure(self._still_config)
                self._picam2.start()
                self._picam2.capture_file(str(path))
            finally:
                try:
                    self._picam2.stop()
                    self._picam2.configure(self._preview_config)
                    self._picam2.start()
                except Exception as exc:
                    logger.error("Failed to restore preview mode: %s", exc)
                    raise CameraError(f"Failed to restore preview after capture: {exc}") from exc

    def get_controls(self) -> dict[str, Any]:
        with self._lock:
            if self._use_mock:
                return {"AfMode": 1, "AfState": 2, "AfTrigger": 0, "LensPosition": 0.0}
            assert self._picam2 is not None
            return dict(self._picam2.camera_controls)

    def set_controls(self, control_dict: dict[str, Any]) -> None:
        with self._lock:
            if self._use_mock:
                return
            assert self._picam2 is not None
            # Filter to only advertised controls
            advertised = set(self._picam2.camera_controls.keys())
            filtered = {k: v for k, v in control_dict.items() if k in advertised}
            if not filtered:
                logger.warning("None of controls %s are advertised", list(control_dict.keys()))
                return
            self._picam2.set_controls(filtered)

    def autofocus_cycle(self) -> None:
        with self._lock:
            if self._use_mock:
                time.sleep(0.2)
                return
            assert self._picam2 is not None
            if hasattr(self._picam2, "autofocus_cycle"):
                self._picam2.autofocus_cycle()
            else:
                self.set_controls({"AfMode": 1, "AfTrigger": 0})

    def _mock_frame(self) -> np.ndarray:
        h, w = self.config.preview_height, self.config.preview_width
        frame = np.full((h, w, 3), 200, dtype=np.uint8)
        # Add some edges in center for detection tests
        ch, cw = h // 3, w // 3
        y0, x0 = h // 3, w // 3
        frame[y0 : y0 + ch, x0 : x0 + 10] = 50
        frame[y0 : y0 + 10, x0 : x0 + cw] = 50
        return frame

    def _mock_still(self, path: Path) -> None:
        from PIL import Image

        img = Image.new("RGB", (self.config.capture_width, self.config.capture_height), color=(220, 220, 220))
        # Draw document-like rectangle
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        margin = 200
        draw.rectangle(
            [margin, margin, self.config.capture_width - margin, self.config.capture_height - margin],
            outline=(30, 30, 30),
            width=20,
        )
        img.save(path, format="JPEG", quality=self.config.jpeg_quality)
