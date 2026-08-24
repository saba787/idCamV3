"""Camera autofocus workflow."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

from idcam.config.schema import CameraConfig
from idcam.exceptions import AutofocusError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import laplacian_variance

logger = get_logger("camera.autofocus")

# libcamera AfState values (when exposed)
AF_STATE_IDLE = 0
AF_STATE_SCANNING = 1
AF_STATE_FOCUSED = 2
AF_STATE_FAILED = 3


class FocusCapableCamera(Protocol):
  def get_controls(self) -> dict[str, Any]: ...
  def set_controls(self, controls: dict[str, Any]) -> None: ...
  def autofocus_cycle(self) -> None: ...
  def capture_preview_array(self) -> Any: ...


@dataclass
class AutofocusResult:
    success: bool
    sharpness: float
    attempts: int
    af_state: int | None
    method: str


class AutofocusController:
    """Manages autofocus with API probing and sharpness verification."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._af_available = False
        self._af_state_available = False

    def probe_capabilities(self, camera: FocusCapableCamera) -> dict[str, bool]:
        controls = camera.get_controls()
        self._af_available = "AfMode" in controls
        self._af_state_available = "AfState" in controls
        caps = {
            "AfMode": self._af_available,
            "AfTrigger": "AfTrigger" in controls,
            "AfState": self._af_state_available,
            "LensPosition": "LensPosition" in controls,
            "autofocus_cycle": hasattr(camera, "autofocus_cycle"),
        }
        logger.info("Autofocus capabilities: %s", caps)
        return caps

    async def run_autofocus(
        self,
        camera: FocusCapableCamera,
        *,
        sharpness_threshold: float,
        roi_fraction: float = 0.7,
    ) -> AutofocusResult:
        last_sharpness = 0.0
        last_af_state: int | None = None

        for attempt in range(1, self.config.af_max_retries + 1):
            logger.info("Autofocus attempt %d/%d", attempt, self.config.af_max_retries)
            try:
                if self._af_available:
                    await self._trigger_af(camera)
                    last_af_state = await self._wait_for_af(camera)
                    if last_af_state == AF_STATE_FAILED:
                        logger.warning("AF reported failed state on attempt %d", attempt)
                        continue
                elif hasattr(camera, "autofocus_cycle"):
                    await asyncio.to_thread(camera.autofocus_cycle)
                else:
                    logger.warning("No AF controls available; relying on sharpness check only")

                preview = await asyncio.to_thread(camera.capture_preview_array)
                last_sharpness = laplacian_variance(preview, roi_fraction)

                if last_sharpness >= sharpness_threshold:
                    return AutofocusResult(
                        success=True,
                        sharpness=last_sharpness,
                        attempts=attempt,
                        af_state=last_af_state,
                        method="af_controls" if self._af_available else "sharpness_only",
                    )

                logger.info(
                    "Sharpness %.1f below threshold %.1f after attempt %d",
                    last_sharpness,
                    sharpness_threshold,
                    attempt,
                )
            except Exception as exc:
                logger.error("Autofocus attempt %d failed: %s", attempt, exc)
                raise AutofocusError(f"Autofocus failed: {exc}") from exc

        raise AutofocusError(
            f"Autofocus did not reach sharpness {sharpness_threshold} "
            f"(best={last_sharpness:.1f}, attempts={self.config.af_max_retries})"
        )

    async def verify_focus(
        self,
        camera: FocusCapableCamera,
        *,
        sharpness_threshold: float,
        roi_fraction: float = 0.7,
    ) -> AutofocusResult:
        preview = await asyncio.to_thread(camera.capture_preview_array)
        sharpness = laplacian_variance(preview, roi_fraction)
        af_state = None
        if self._af_state_available:
            controls = camera.get_controls()
            af_state = controls.get("AfState")

        success = sharpness >= sharpness_threshold
        if not success:
            raise AutofocusError(f"Focus verification failed: sharpness={sharpness:.1f}")

        return AutofocusResult(
            success=True,
            sharpness=sharpness,
            attempts=0,
            af_state=af_state,
            method="verification",
        )

    async def _trigger_af(self, camera: FocusCapableCamera) -> None:
        def _do() -> None:
            # AfMode 1 = Auto, AfTrigger 0 = Start (Arducam/libcamera convention)
            camera.set_controls({"AfMode": 1, "AfTrigger": 0})

        await asyncio.to_thread(_do)

    async def _wait_for_af(self, camera: FocusCapableCamera) -> int:
        deadline = time.monotonic() + self.config.af_timeout_sec
        while time.monotonic() < deadline:
            controls = await asyncio.to_thread(camera.get_controls)
            state = controls.get("AfState", AF_STATE_IDLE)
            if state in (AF_STATE_FOCUSED, AF_STATE_FAILED):
                return int(state)
            await asyncio.sleep(0.1)
        logger.warning("AF state wait timed out after %.1fs", self.config.af_timeout_sec)
        return AF_STATE_FAILED
