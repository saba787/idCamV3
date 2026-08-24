"""Camera autofocus workflow."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from idcam.config.schema import CameraConfig
from idcam.exceptions import AutofocusError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import laplacian_variance

logger = get_logger("camera.autofocus")

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
class AutofocusCapabilities:
    af_mode: bool = False
    af_trigger: bool = False
    af_state: bool = False
    lens_position: bool = False
    autofocus_cycle: bool = False

    @property
    def hardware_af_available(self) -> bool:
        return self.af_mode or self.autofocus_cycle

    @property
    def degraded(self) -> bool:
        """True when only sharpness-based verification is available."""
        return not self.hardware_af_available

    def to_dict(self) -> dict[str, bool]:
        return {
            "AfMode": self.af_mode,
            "AfTrigger": self.af_trigger,
            "AfState": self.af_state,
            "LensPosition": self.lens_position,
            "autofocus_cycle": self.autofocus_cycle,
            "hardware_af_available": self.hardware_af_available,
            "degraded": self.degraded,
        }


@dataclass
class AutofocusResult:
    success: bool
    sharpness: float
    attempts: int
    af_state: int | None
    method: str
    min_required_sharpness: float = 0.0


class AutofocusController:
    """Manages autofocus with API probing and sharpness verification."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.capabilities = AutofocusCapabilities()

    @property
    def hardware_af_available(self) -> bool:
        return self.capabilities.hardware_af_available

    @property
    def is_degraded(self) -> bool:
        return self.capabilities.degraded

    def probe_capabilities(self, camera: FocusCapableCamera) -> AutofocusCapabilities:
        controls = camera.get_controls()
        self.capabilities = AutofocusCapabilities(
            af_mode="AfMode" in controls,
            af_trigger="AfTrigger" in controls,
            af_state="AfState" in controls,
            lens_position="LensPosition" in controls,
            autofocus_cycle=hasattr(camera, "autofocus_cycle"),
        )
        logger.info("Autofocus capabilities: %s", self.capabilities.to_dict())
        if self.capabilities.degraded:
            logger.warning(
                "Hardware autofocus unavailable; using sharpness-only verification (degraded mode)"
            )
        return self.capabilities

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
            logger.info(
                "event=autofocus_started attempt=%d/%d min_sharpness=%.1f degraded=%s",
                attempt,
                self.config.af_max_retries,
                sharpness_threshold,
                self.capabilities.degraded,
            )
            try:
                if self.capabilities.af_mode:
                    await self._trigger_af(camera)
                    last_af_state = await self._wait_for_af(camera)
                    if last_af_state == AF_STATE_FAILED:
                        logger.warning("AF reported failed state on attempt %d", attempt)
                        continue
                elif self.capabilities.autofocus_cycle:
                    await asyncio.to_thread(camera.autofocus_cycle)
                else:
                    logger.warning("No AF controls; sharpness-only mode")

                preview = await asyncio.to_thread(camera.capture_preview_array)
                if preview is None or getattr(preview, "size", 1) == 0:
                    raise AutofocusError("Invalid preview frame for sharpness check")
                last_sharpness = laplacian_variance(preview, roi_fraction)
                logger.info(
                    "event=autofocus_sharpness score=%.1f min_required=%.1f attempt=%d",
                    last_sharpness,
                    sharpness_threshold,
                    attempt,
                )

                if last_sharpness >= sharpness_threshold:
                    method = "af_controls" if self.capabilities.hardware_af_available else "sharpness_only"
                    return AutofocusResult(
                        success=True,
                        sharpness=last_sharpness,
                        attempts=attempt,
                        af_state=last_af_state,
                        method=method,
                        min_required_sharpness=sharpness_threshold,
                    )
            except AutofocusError:
                raise
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
        if preview is None or getattr(preview, "size", 1) == 0:
            raise AutofocusError("Invalid preview frame for focus verification")
        sharpness = laplacian_variance(preview, roi_fraction)
        af_state = None
        if self.capabilities.af_state:
            controls = camera.get_controls()
            af_state = controls.get("AfState")

        logger.info(
            "event=focus_verification score=%.1f min_required=%.1f af_state=%s",
            sharpness,
            sharpness_threshold,
            af_state,
        )

        if sharpness < sharpness_threshold:
            raise AutofocusError(
                f"Focus verification failed: sharpness={sharpness:.1f} min={sharpness_threshold:.1f}"
            )

        return AutofocusResult(
            success=True,
            sharpness=sharpness,
            attempts=0,
            af_state=af_state,
            method="verification",
            min_required_sharpness=sharpness_threshold,
        )

    async def _trigger_af(self, camera: FocusCapableCamera) -> None:
        def _do() -> None:
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
