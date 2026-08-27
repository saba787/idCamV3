"""Camera autofocus workflow — one hardware AF pass, sharpness-verified."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

from idcam.config.schema import CameraConfig
from idcam.exceptions import AutofocusError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import accept_preview_focus, clip_roi, laplacian_variance, roi_rect

logger = get_logger("camera.autofocus")

AF_STATE_IDLE = 0
AF_STATE_SCANNING = 1
AF_STATE_FOCUSED = 2
AF_STATE_FAILED = 3

PREVIEW_ABSOLUTE_FLOOR = 28.0


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
    lens_before: float | None = None
    lens_after: float | None = None
    improvement_ratio: float = 0.0
    sensor_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    preview_size: tuple[int, int] = (0, 0)


@dataclass
class AutofocusDebug:
    running: bool = False
    method: str = ""
    attempt: int = 0
    focus_before: float = 0.0
    focus_after: float = 0.0
    focus_roi: str = ""
    preview_roi: str = ""
    sensor_roi: str = ""
    preview_roi_wh: str = ""
    lens_before: float | None = None
    lens_after: float | None = None
    improvement_ratio: float = 0.0
    scaler_crop: str = ""
    preview_size: str = ""


def _fmt_roi(roi: tuple[int, int, int, int] | None) -> str:
    if not roi:
        return ""
    return "{},{},{},{}".format(*(int(v) for v in roi))


class AutofocusController:
    """Hardware AF on document ROI with before/after sharpness verification."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.capabilities = AutofocusCapabilities()
        self.debug = AutofocusDebug()
        self._last_preview_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._last_preview_shape: tuple[int, ...] = ()

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
            logger.warning("Hardware autofocus unavailable; sharpness-only mode")
        if not self.capabilities.af_state:
            logger.info(
                "event=autofocus_result af_state=unavailable using=lens_position_and_sharpness"
            )
        return self.capabilities

    def _resolve_preview_roi(
        self,
        frame: Any,
        roi_fraction: float,
        preview_roi: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int]:
        if preview_roi is not None:
            return clip_roi(preview_roi, frame.shape)
        return clip_roi(roi_rect(frame.shape, roi_fraction), frame.shape)

    def _update_debug_roi(
        self,
        roi: tuple[int, int, int, int],
        sensor: tuple[int, int, int, int],
        preview_size: tuple[int, int],
        scaler_crop: tuple[int, int, int, int],
    ) -> None:
        self.debug.focus_roi = _fmt_roi(roi)
        self.debug.preview_roi = _fmt_roi(roi)
        self.debug.sensor_roi = _fmt_roi(sensor)
        self.debug.preview_roi_wh = f"{roi[2]}x{roi[3]}"
        self.debug.scaler_crop = _fmt_roi(scaler_crop)
        self.debug.preview_size = f"{preview_size[0]}x{preview_size[1]}"

    async def run_autofocus(
        self,
        camera: FocusCapableCamera,
        *,
        sharpness_threshold: float,
        roi_fraction: float = 0.7,
        preview_roi: tuple[int, int, int, int] | None = None,
    ) -> AutofocusResult:
        preview_floor = min(float(sharpness_threshold), PREVIEW_ABSOLUTE_FLOOR)
        last_roi = (0, 0, 0, 0)
        last_sensor = (0, 0, 0, 0)
        preview_size = (0, 0)
        last_before = 0.0
        last_after = 0.0
        last_lens_before: Any = None
        last_lens_after: Any = None
        last_method = "none"
        last_ratio = 0.0

        preview = await asyncio.to_thread(camera.capture_preview_array)
        if preview is None or getattr(preview, "size", 1) == 0:
            raise AutofocusError("Invalid preview frame for sharpness check")
        h, w = int(preview.shape[0]), int(preview.shape[1])
        preview_size = (w, h)
        last_roi = self._resolve_preview_roi(preview, roi_fraction, preview_roi)
        mapping = await self._apply_document_roi(camera, last_roi, preview_size)
        last_sensor = tuple(mapping.get("sensor_roi", last_roi))
        last_before = laplacian_variance(preview, roi=last_roi)
        meta_before = self._read_metadata(camera)
        last_lens_before = meta_before.get("LensPosition")
        self._last_preview_roi = last_roi
        self._last_preview_shape = preview.shape
        self._update_debug_roi(
            last_roi,
            last_sensor,
            preview_size,
            tuple(mapping.get("scaler_crop", (0, 0, w, h))),
        )
        self.debug.focus_before = last_before
        self.debug.lens_before = float(last_lens_before) if last_lens_before is not None else None

        logger.info(
            "event=autofocus_start preview_roi=%s sensor_roi=%s before_sharpness=%.1f "
            "preview_floor=%.1f",
            self.debug.preview_roi,
            self.debug.sensor_roi,
            last_before,
            preview_floor,
        )

        # Already sharp — do not move the lens.
        if accept_preview_focus(
            before=max(last_before * 0.85, 1.0),
            after=last_before,
            absolute_floor=preview_floor,
        ):
            self.debug.method = "already_sharp"
            self.debug.focus_after = last_before
            self.debug.running = False
            logger.info(
                "event=autofocus_result method=already_sharp sharpness=%.1f",
                last_before,
            )
            return AutofocusResult(
                success=True,
                sharpness=last_before,
                attempts=0,
                af_state=None,
                method="already_sharp",
                min_required_sharpness=preview_floor,
                before_sharpness=last_before,
                improved=False,
                roi=last_roi,
                lens_before=self.debug.lens_before,
                lens_after=self.debug.lens_before,
                improvement_ratio=1.0,
                sensor_roi=last_sensor,
                preview_size=preview_size,
            )

        for attempt in range(1, self.config.af_max_retries + 1):
            self.debug = AutofocusDebug(running=True, method="hardware_trigger", attempt=attempt)
            self._update_debug_roi(
                last_roi,
                last_sensor,
                preview_size,
                tuple(mapping.get("scaler_crop", (0, 0, w, h))),
            )
            self.debug.focus_before = last_before
            self.debug.lens_before = float(last_lens_before) if last_lens_before is not None else None

            try:
                last_method, _ = await self._run_hardware_af(camera, attempt)
                self.debug.method = last_method

                after_frame = await asyncio.to_thread(camera.capture_preview_array)
                if after_frame is None or getattr(after_frame, "size", 1) == 0:
                    raise AutofocusError("Invalid preview frame after autofocus")
                last_after = laplacian_variance(after_frame, roi=last_roi)
                meta_after = self._read_metadata(camera)
                last_lens_after = meta_after.get("LensPosition")
                last_ratio = (last_after / last_before) if last_before > 1e-6 else (
                    2.0 if last_after > 0 else 0.0
                )
                improved = last_after > last_before
                accepted = accept_preview_focus(
                    before=last_before,
                    after=last_after,
                    absolute_floor=preview_floor,
                )
                self.debug.focus_after = last_after
                self.debug.lens_after = float(last_lens_after) if last_lens_after is not None else None
                self.debug.improvement_ratio = round(last_ratio, 3)
                self.debug.running = False
                logger.info(
                    "event=autofocus_result method=%s success=%s before=%.1f after=%.1f "
                    "ratio=%.3f attempt=%d lens=%s->%s preview_roi=%s",
                    last_method,
                    accepted,
                    last_before,
                    last_after,
                    last_ratio,
                    attempt,
                    last_lens_before,
                    last_lens_after,
                    self.debug.preview_roi,
                )
                if accepted:
                    pause = getattr(camera, "pause_autofocus", None)
                    if callable(pause):
                        await asyncio.to_thread(pause)
                    return AutofocusResult(
                        success=True,
                        sharpness=last_after,
                        attempts=attempt,
                        af_state=None,
                        method=last_method,
                        min_required_sharpness=preview_floor,
                        before_sharpness=last_before,
                        improved=improved,
                        roi=last_roi,
                        lens_before=self.debug.lens_before,
                        lens_after=self.debug.lens_after,
                        improvement_ratio=last_ratio,
                        sensor_roi=last_sensor,
                        preview_size=preview_size,
                    )
                last_before = last_after
                last_lens_before = last_lens_after
            except AutofocusError:
                self.debug.running = False
                raise
            except Exception as exc:
                self.debug.running = False
                raise AutofocusError(f"Autofocus failed: {exc}") from exc

        self.debug.running = False
        raise AutofocusError(
            f"Autofocus did not improve document ROI sharpness "
            f"(before={last_before:.1f} after={last_after:.1f} ratio={last_ratio:.2f}, "
            f"attempts={self.config.af_max_retries} method={last_method} "
            f"preview_roi={_fmt_roi(last_roi)})"
        )

    async def verify_focus(
        self,
        camera: FocusCapableCamera,
        *,
        sharpness_threshold: float,
        roi_fraction: float = 0.7,
        preview_roi: tuple[int, int, int, int] | None = None,
    ) -> AutofocusResult:
        preview = await asyncio.to_thread(camera.capture_preview_array)
        if preview is None or getattr(preview, "size", 1) == 0:
            raise AutofocusError("Invalid preview frame for focus verification")
        roi = preview_roi or self._last_preview_roi
        if roi == (0, 0, 0, 0):
            roi = roi_rect(preview.shape, roi_fraction)
        roi = clip_roi(roi, preview.shape)
        sharpness = laplacian_variance(preview, roi=roi)
        meta = self._read_metadata(camera)
        preview_floor = min(float(sharpness_threshold), PREVIEW_ABSOLUTE_FLOOR)
        last_after = self.debug.focus_after
        last_before = self.debug.focus_before
        held = last_after > 0 and sharpness >= last_after * 0.65
        relative = accept_preview_focus(
            before=last_before if last_before > 0 else sharpness,
            after=sharpness,
            absolute_floor=preview_floor,
        )
        ok = held or relative

        logger.info(
            "event=focus_verification score=%.1f last_after=%.1f held=%s relative=%s ok=%s "
            "preview_roi=%s",
            sharpness,
            last_after,
            held,
            relative,
            ok,
            _fmt_roi(roi),
        )

        if not ok:
            raise AutofocusError(
                f"Focus verification failed: roi_sharpness={sharpness:.1f} "
                f"last={last_after:.1f} preview_roi={_fmt_roi(roi)}"
            )

        self.debug.focus_after = sharpness
        return AutofocusResult(
            success=True,
            sharpness=sharpness,
            attempts=0,
            af_state=self._af_state_from_meta(meta),
            method="verification",
            min_required_sharpness=preview_floor,
            before_sharpness=last_before,
            improved=sharpness >= last_before,
            roi=roi,
            lens_before=self.debug.lens_before,
            lens_after=float(meta["LensPosition"]) if meta.get("LensPosition") is not None else self.debug.lens_after,
            improvement_ratio=self.debug.improvement_ratio,
            sensor_roi=tuple(int(x) for x in self.debug.sensor_roi.split(","))
            if self.debug.sensor_roi.count(",") == 3
            else (0, 0, 0, 0),
        )

    async def _apply_document_roi(
        self,
        camera: FocusCapableCamera,
        preview_roi: tuple[int, int, int, int],
        preview_size: tuple[int, int],
    ) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "preview_roi": preview_roi,
            "preview_size": preview_size,
            "scaler_crop": (0, 0, preview_size[0], preview_size[1]),
            "sensor_roi": preview_roi,
        }
        applier = getattr(camera, "apply_document_focus_window", None)
        if not callable(applier):
            return mapping
        try:
            result = await asyncio.to_thread(applier, preview_roi, preview_size)
        except TypeError:
            result = await asyncio.to_thread(applier, 0.7)
        if isinstance(result, dict):
            mapping.update(result)
            if result.get("sensor_roi"):
                mapping["sensor_roi"] = tuple(int(v) for v in result["sensor_roi"])
            if result.get("preview_roi"):
                mapping["preview_roi"] = tuple(int(v) for v in result["preview_roi"])
            return mapping
        if result and len(result) == 4:
            mapping["sensor_roi"] = (int(result[0]), int(result[1]), int(result[2]), int(result[3]))
        return mapping

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

    def _af_state_from_meta(self, meta: dict[str, Any]) -> int | None:
        if "AfState" not in meta:
            return None
        try:
            return int(meta["AfState"])
        except (TypeError, ValueError):
            return None

    async def _run_hardware_af(self, camera: FocusCapableCamera, attempt: int) -> tuple[str, int | None]:
        trigger = getattr(camera, "trigger_autofocus", None)
        settle = getattr(camera, "wait_for_lens_settle", None)
        if callable(trigger) and callable(settle):
            logger.info("event=autofocus_triggered method=af_trigger_lens_settle attempt=%d", attempt)
            await asyncio.to_thread(trigger)
            await asyncio.to_thread(settle, min(float(self.config.af_timeout_sec), 4.0))
            method = "hardware_trigger" if self.capabilities.hardware_af_available else "sharpness_only"
            return method, None

        if self.capabilities.autofocus_cycle:
            logger.info("event=autofocus_triggered method=autofocus_cycle attempt=%d", attempt)
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(camera.autofocus_cycle),
                    timeout=self.config.af_timeout_sec,
                )
            except TimeoutError:
                logger.warning("event=autofocus_timeout method=autofocus_cycle attempt=%d", attempt)
            return "hardware_cycle", None

        if self.capabilities.af_mode:
            await self._trigger_af(camera)
            state = await self._wait_for_af(camera)
            return "hardware_controls", state

        logger.warning("event=autofocus_triggered method=sharpness_only attempt=%d", attempt)
        return "sharpness_only", None

    async def _trigger_af(self, camera: FocusCapableCamera) -> None:
        def _do() -> None:
            camera.set_controls({"AfMode": 1, "AfTrigger": 0, "AfSpeed": 1})

        await asyncio.to_thread(_do)

    async def _wait_for_af(self, camera: FocusCapableCamera) -> int | None:
        settle = getattr(camera, "wait_for_lens_settle", None)
        if callable(settle):
            await asyncio.to_thread(settle, min(float(self.config.af_timeout_sec), 4.0))
            return None

        deadline = time.monotonic() + self.config.af_timeout_sec
        last_state: int | None = None
        last_lens: Any = None
        stable = 0
        while time.monotonic() < deadline:
            meta = await asyncio.to_thread(self._read_metadata, camera)
            lens = meta.get("LensPosition")
            if "AfState" not in meta:
                if last_lens is not None and lens is not None:
                    try:
                        if abs(float(lens) - float(last_lens)) < 0.05:
                            stable += 1
                            if stable >= 3:
                                return None
                        else:
                            stable = 0
                    except (TypeError, ValueError):
                        stable = 0
                last_lens = lens
                await asyncio.sleep(0.1)
                continue
            last_state = int(meta["AfState"])
            if last_state in (AF_STATE_FOCUSED, AF_STATE_FAILED):
                return last_state
            await asyncio.sleep(0.1)
        return last_state
