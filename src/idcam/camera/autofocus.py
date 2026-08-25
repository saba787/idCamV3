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
    def autofocus_cycle(self) -> Any: ...
    def capture_preview_array(self) -> Any: ...
    def capture_metadata(self) -> dict[str, Any]: ...


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
                "event=autofocus_start attempt=%d/%d min_sharpness=%.1f degraded=%s",
                attempt,
                self.config.af_max_retries,
                sharpness_threshold,
                self.capabilities.degraded,
            )
            try:
                last_af_state = await self._run_hardware_af(camera, attempt)

                preview = await asyncio.to_thread(camera.capture_preview_array)
                if preview is None or getattr(preview, "size", 1) == 0:
                    raise AutofocusError("Invalid preview frame for sharpness check")
                last_sharpness = laplacian_variance(preview, roi_fraction)
                meta = self._read_metadata(camera)
                lens = meta.get("LensPosition")
                if last_af_state is None and "AfState" in meta:
                    last_af_state = int(meta["AfState"])
                logger.info(
                    "event=autofocus_result success=%s sharpness=%.1f min_required=%.1f "
                    "attempt=%d lens_position=%s af_state=%s",
                    last_sharpness >= sharpness_threshold,
                    last_sharpness,
                    sharpness_threshold,
                    attempt,
                    lens,
                    last_af_state,
                )

                if last_sharpness >= sharpness_threshold:
                    method = "af_controls" if self.capabilities.hardware_af_available else "sharpness_only"
                    if self.capabilities.autofocus_cycle:
                        method = "autofocus_cycle"
                    return AutofocusResult(
                        success=True,
                        sharpness=last_sharpness,
                        attempts=attempt,
                        af_state=last_af_state,
                        method=method,
                        min_required_sharpness=sharpness_threshold,
                    )
                logger.info("event=autofocus_retry attempt=%d reason=sharpness_below_threshold", attempt)
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
        meta = self._read_metadata(camera)
        af_state = meta.get("AfState")
        if af_state is not None:
            af_state = int(af_state)

        logger.info(
            "event=focus_verification score=%.1f min_required=%.1f af_state=%s lens_position=%s",
            sharpness,
            sharpness_threshold,
            af_state,
            meta.get("LensPosition"),
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

    def _read_metadata(self, camera: FocusCapableCamera) -> dict[str, Any]:
        reader = getattr(camera, "capture_metadata", None)
        if reader is None:
            return {}
        try:
            meta = reader()
        except Exception as exc:
            logger.debug("event=autofocus_metadata_unavailable error=%s", exc)
            return {}
        return dict(meta) if meta else {}

    async def _run_hardware_af(self, camera: FocusCapableCamera, attempt: int) -> int | None:
        """Trigger a real AF scan. Prefer Picamera2.autofocus_cycle (waits on metadata)."""
        if self.capabilities.autofocus_cycle:
            logger.info("event=autofocus_triggered method=autofocus_cycle attempt=%d", attempt)
            try:
                cycle_ok = await asyncio.wait_for(
                    asyncio.to_thread(camera.autofocus_cycle),
                    timeout=self.config.af_timeout_sec,
                )
            except TimeoutError:
                logger.warning(
                    "event=autofocus_timeout method=autofocus_cycle attempt=%d timeout_sec=%.1f",
                    attempt,
                    self.config.af_timeout_sec,
                )
                return None
            if cycle_ok is False:
                logger.warning("event=autofocus_result success=false method=autofocus_cycle attempt=%d", attempt)
            return None

        if self.capabilities.af_mode:
            logger.info(
                "event=autofocus_triggered method=af_controls AfMode=Auto AfTrigger=Start attempt=%d",
                attempt,
            )
            await self._trigger_af(camera)
            return await self._wait_for_af(camera)

        logger.warning("event=autofocus_triggered method=sharpness_only attempt=%d", attempt)
        return None

    async def _trigger_af(self, camera: FocusCapableCamera) -> None:
        def _do() -> None:
            # AfModeAuto=1, AfTriggerStart=0 (libcamera control_ids).
            camera.set_controls({"AfMode": 1, "AfTrigger": 0})

        await asyncio.to_thread(_do)

    async def _wait_for_af(self, camera: FocusCapableCamera) -> int | None:
        deadline = time.monotonic() + self.config.af_timeout_sec
        last_state: int | None = None
        saw_state = False
        while time.monotonic() < deadline:
            meta = await asyncio.to_thread(self._read_metadata, camera)
            if "AfState" not in meta:
                logger.info("event=autofocus_result af_state=unavailable source=metadata")
                return None
            saw_state = True
            last_state = int(meta["AfState"])
            if last_state in (AF_STATE_FOCUSED, AF_STATE_FAILED):
                logger.info(
                    "event=autofocus_result af_state=%s lens_position=%s",
                    last_state,
                    meta.get("LensPosition"),
                )
                return last_state
            await asyncio.sleep(0.1)
        logger.warning(
            "event=autofocus_timeout after_sec=%.1f last_af_state=%s",
            self.config.af_timeout_sec,
            last_state,
        )
        return AF_STATE_FAILED if saw_state else None
