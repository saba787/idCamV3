"""Camera autofocus workflow."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

from idcam.config.schema import CameraConfig
from idcam.exceptions import AutofocusError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import laplacian_variance, roi_rect

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
    af_windows: bool = False
    af_range: bool = False
    af_metering: bool = False

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
            "AfWindows": self.af_windows,
            "AfRange": self.af_range,
            "AfMetering": self.af_metering,
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
    before_sharpness: float = 0.0
    improved: bool = False
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class AutofocusDebug:
    running: bool = False
    method: str = ""
    attempt: int = 0
    focus_before: float = 0.0
    focus_after: float = 0.0
    focus_roi: str = ""


class AutofocusController:
    """Manages autofocus with API probing and document-ROI sharpness verification."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.capabilities = AutofocusCapabilities()
        self.debug = AutofocusDebug()

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
            af_windows="AfWindows" in controls,
            af_range="AfRange" in controls,
            af_metering="AfMetering" in controls,
        )
        logger.info("Autofocus capabilities: %s", self.capabilities.to_dict())
        if self.capabilities.degraded:
            logger.warning(
                "Hardware autofocus unavailable; using sharpness-only verification (degraded mode)"
            )
        if not self.capabilities.af_state:
            logger.info(
                "event=autofocus_result af_state=unavailable "
                "reason=not_advertised using=document_roi_sharpness"
            )
        return self.capabilities

    async def run_autofocus(
        self,
        camera: FocusCapableCamera,
        *,
        sharpness_threshold: float,
        roi_fraction: float = 0.7,
    ) -> AutofocusResult:
        last_after = 0.0
        last_before = 0.0
        last_af_state: int | None = None
        last_roi = (0, 0, 0, 0)
        last_lens: Any = None

        for attempt in range(1, self.config.af_max_retries + 1):
            self.debug = AutofocusDebug(running=True, method="hardware_cycle", attempt=attempt)
            try:
                last_roi = await self._apply_document_roi(camera, roi_fraction)
                preview, last_before, last_af_state, last_lens = await self._measure_focus(
                    camera, roi_fraction
                )
                self.debug.focus_before = last_before
                self.debug.focus_roi = "{},{},{},{}".format(*last_roi)
                logger.info(
                    "event=autofocus_start attempt=%d/%d roi=%s before_sharpness=%.1f "
                    "min_sharpness=%.1f degraded=%s af_state_advertised=%s",
                    attempt,
                    self.config.af_max_retries,
                    self.debug.focus_roi,
                    last_before,
                    sharpness_threshold,
                    self.capabilities.degraded,
                    self.capabilities.af_state,
                )

                last_af_state = await self._run_hardware_af(camera, attempt)
                preview, last_after, measured_state, last_lens = await self._measure_focus(
                    camera, roi_fraction
                )
                if last_af_state is None:
                    last_af_state = measured_state
                improved = last_after > last_before
                reached = last_after >= sharpness_threshold
                # AfState is not advertised on this Arducam; ROI sharpness is the authority.
                accepted = reached or (improved and last_after > last_before * 1.15)
                self.debug.focus_after = last_after
                self.debug.running = False
                logger.info(
                    "event=autofocus_result method=hardware_cycle success=%s "
                    "before_sharpness=%.1f after_sharpness=%.1f improved=%s "
                    "min_required=%.1f attempt=%d lens_position=%s af_state=%s roi=%s",
                    accepted,
                    last_before,
                    last_after,
                    improved,
                    sharpness_threshold,
                    attempt,
                    last_lens,
                    last_af_state,
                    self.debug.focus_roi,
                )
                if accepted:
                    method = "hardware_cycle" if self.capabilities.hardware_af_available else "sharpness_only"
                    return AutofocusResult(
                        success=True,
                        sharpness=last_after,
                        attempts=attempt,
                        af_state=last_af_state,
                        method=method,
                        min_required_sharpness=sharpness_threshold,
                        before_sharpness=last_before,
                        improved=improved,
                        roi=last_roi,
                    )
                logger.info("event=autofocus_retry attempt=%d reason=roi_sharpness_not_improved", attempt)
            except AutofocusError:
                self.debug.running = False
                raise
            except Exception as exc:
                self.debug.running = False
                logger.error("Autofocus attempt %d failed: %s", attempt, exc)
                raise AutofocusError(f"Autofocus failed: {exc}") from exc

        self.debug.running = False
        raise AutofocusError(
            f"Autofocus did not reach document ROI sharpness {sharpness_threshold} "
            f"(before={last_before:.1f} after={last_after:.1f}, attempts={self.config.af_max_retries})"
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
        roi = roi_rect(preview.shape, roi_fraction)
        meta = self._read_metadata(camera)
        af_state = meta.get("AfState")
        if af_state is not None:
            af_state = int(af_state)

        logger.info(
            "event=focus_verification score=%.1f min_required=%.1f af_state=%s "
            "lens_position=%s roi=%d,%d,%d,%d",
            sharpness,
            sharpness_threshold,
            af_state,
            meta.get("LensPosition"),
            *roi,
        )

        if sharpness < sharpness_threshold:
            raise AutofocusError(
                f"Focus verification failed: roi_sharpness={sharpness:.1f} min={sharpness_threshold:.1f}"
            )

        return AutofocusResult(
            success=True,
            sharpness=sharpness,
            attempts=0,
            af_state=af_state,
            method="verification",
            min_required_sharpness=sharpness_threshold,
            before_sharpness=self.debug.focus_before,
            improved=sharpness >= self.debug.focus_before,
            roi=roi,
        )

    async def _apply_document_roi(
        self, camera: FocusCapableCamera, roi_fraction: float
    ) -> tuple[int, int, int, int]:
        applier = getattr(camera, "apply_document_focus_window", None)
        if callable(applier):
            result = await asyncio.to_thread(applier, roi_fraction)
            if result and len(result) == 4:
                return int(result[0]), int(result[1]), int(result[2]), int(result[3])
        preview = await asyncio.to_thread(camera.capture_preview_array)
        if preview is None:
            return 0, 0, 0, 0
        return roi_rect(preview.shape, roi_fraction)

    async def _measure_focus(
        self,
        camera: FocusCapableCamera,
        roi_fraction: float,
    ) -> tuple[Any, float, int | None, Any]:
        preview = await asyncio.to_thread(camera.capture_preview_array)
        if preview is None or getattr(preview, "size", 1) == 0:
            raise AutofocusError("Invalid preview frame for sharpness check")
        sharpness = laplacian_variance(preview, roi_fraction)
        meta = self._read_metadata(camera)
        af_state = None
        if "AfState" in meta:
            try:
                af_state = int(meta["AfState"])
            except (TypeError, ValueError):
                af_state = None
        return preview, sharpness, af_state, meta.get("LensPosition")

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
                logger.warning(
                    "event=autofocus_result success=false method=autofocus_cycle attempt=%d",
                    attempt,
                )
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
            camera.set_controls({"AfMode": 1, "AfTrigger": 0, "AfSpeed": 1})

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
