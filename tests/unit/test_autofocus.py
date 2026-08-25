"""Autofocus controller uses live metadata, not advertised control ranges."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from idcam.camera.autofocus import AutofocusController
from idcam.config.schema import CameraConfig
from idcam.exceptions import AutofocusError


def _sharp_frame() -> np.ndarray:
    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    frame[20:100, 20:30] = 0
    frame[20:30, 20:140] = 0
    frame[90:100, 20:140] = 0
    return frame


class FakeCamera:
    def __init__(
        self,
        *,
        advertised: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        cycle_result: Any = True,
        has_cycle: bool = True,
    ) -> None:
        self.advertised = advertised or {
            "AfMode": (0, 2, 0),
            "AfTrigger": (0, 1, 0),
            "LensPosition": (0.0, 10.0, 1.0),
        }
        self.metadata = metadata or {"AfState": 2, "LensPosition": 1.25}
        self.cycle_result = cycle_result
        self.has_cycle = has_cycle
        self.cycle_calls = 0
        self.set_calls: list[dict[str, Any]] = []
        self.metadata_calls = 0
        self.controls_calls = 0

    def get_controls(self) -> dict[str, Any]:
        self.controls_calls += 1
        return dict(self.advertised)

    def capture_metadata(self) -> dict[str, Any]:
        self.metadata_calls += 1
        return dict(self.metadata)

    def set_controls(self, controls: dict[str, Any]) -> None:
        self.set_calls.append(dict(controls))

    def autofocus_cycle(self) -> Any:
        if not self.has_cycle:
            raise AssertionError("autofocus_cycle should not be called")
        self.cycle_calls += 1
        return self.cycle_result

    def capture_preview_array(self) -> np.ndarray:
        return _sharp_frame()


class TestAutofocusController:
    @pytest.mark.asyncio
    async def test_already_sharp_skips_hardware_cycle(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert result.method == "already_sharp"
        assert camera.cycle_calls == 0

    @pytest.mark.asyncio
    async def test_prefers_autofocus_cycle_when_not_yet_sharp(self):
        class NeedsCycle(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                if self.cycle_calls:
                    return _sharp_frame()
                return np.full((120, 160, 3), 128, dtype=np.uint8)

        camera = NeedsCycle()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert camera.cycle_calls == 1
        assert result.method == "autofocus_cycle"

    @pytest.mark.asyncio
    async def test_wait_reads_af_state_from_metadata_not_control_ranges(self):
        class NeedsTrigger(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                if self.set_calls:
                    return _sharp_frame()
                return np.full((120, 160, 3), 128, dtype=np.uint8)

        camera = NeedsTrigger(has_cycle=False)
        camera.advertised["AfState"] = (0, 3, 0)
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        caps = ctl.probe_capabilities(camera)
        caps.autofocus_cycle = False
        ctl.capabilities = caps
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert camera.metadata_calls >= 1
        assert camera.set_calls
        assert camera.set_calls[0]["AfMode"] == 1
        assert camera.set_calls[0]["AfTrigger"] == 0

    @pytest.mark.asyncio
    async def test_missing_af_state_does_not_block_until_timeout(self):
        class NeedsTrigger(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                if self.set_calls:
                    return _sharp_frame()
                return np.full((120, 160, 3), 128, dtype=np.uint8)

        camera = NeedsTrigger(has_cycle=False, metadata={"LensPosition": 0.8})
        ctl = AutofocusController(CameraConfig(af_timeout_sec=2.0, af_max_retries=1))
        caps = ctl.probe_capabilities(camera)
        caps.autofocus_cycle = False
        ctl.capabilities = caps
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert camera.metadata_calls >= 1

    @pytest.mark.asyncio
    async def test_verify_focus_uses_metadata(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig())
        ctl.probe_capabilities(camera)
        result = await ctl.verify_focus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert result.af_state == 2
        assert camera.metadata_calls >= 1

    @pytest.mark.asyncio
    async def test_low_sharpness_retries_then_fails(self):
        class Blurry(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                return np.full((80, 80, 3), 128, dtype=np.uint8)

        camera = Blurry()
        ctl = AutofocusController(CameraConfig(af_max_retries=2, af_timeout_sec=1.0))
        ctl.probe_capabilities(camera)
        with pytest.raises(AutofocusError, match="did not reach sharpness"):
            await ctl.run_autofocus(camera, sharpness_threshold=500.0)
        assert camera.cycle_calls == 2
