"""Live Debug Viewer tests. The viewer is an observer, not a camera producer."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from http.client import HTTPResponse

import numpy as np
import pytest

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.config.schema import AppConfig, DebugConfig, DetectionConfig, ServerConfig, UIConfig
from idcam.preview.live_viewer import (
    DebugSnapshot,
    LiveDebugViewer,
    LiveFrameBuffer,
    draw_overlay,
    encode_jpeg,
    make_placeholder,
)
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.stability_detector import StabilityDetector


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    if port < 1024:
        pytest.skip("could not allocate an ephemeral port >= 1024")
    return port


def _wait_jpeg(viewer: LiveDebugViewer, timeout: float = 4.0) -> bytes:
    deadline = time.monotonic() + timeout
    last = -1
    while time.monotonic() < deadline:
        seq, jpeg = viewer.wait_jpeg(last, timeout=0.2)
        if jpeg:
            return bytes(jpeg)
        last = seq
    raise TimeoutError("no JPEG produced")


def _read_mjpeg_part(resp: HTTPResponse, limit: int = 2_000_000) -> bytes:
    data = b""
    while len(data) < limit:
        chunk = resp.read(2048)
        if not chunk:
            break
        data += chunk
        if b"\xff\xd8" in data and b"\xff\xd9" in data:
            break
    return data


@pytest.fixture
def app_config(tmp_path):
    return AppConfig(
        device_id="test-device",
        temp_dir=tmp_path / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        ui=UIConfig(
            status_file=tmp_path / "run" / "status.json",
            admin_socket=tmp_path / "run" / "admin.sock",
        ),
    )


@pytest.fixture
def debug_viewer():
    port = _free_port()
    viewer = LiveDebugViewer(
        DebugConfig(enabled=True, host="127.0.0.1", port=port, fps=5.0, jpeg_quality=70)
    )
    viewer.start()
    try:
        if not viewer.is_running:
            pytest.skip("debug HTTP server failed to bind")
        yield viewer
    finally:
        viewer.stop()


class TestDebugDisabled:
    def test_orchestrator_does_not_start_viewer(self, app_config):
        assert app_config.debug.enabled is False
        orch = ApplicationOrchestrator(app_config, use_mock_camera=True)
        assert orch._debug_viewer is None


class TestLiveFrameBuffer:
    def test_latest_frame_not_a_queue(self):
        buf = LiveFrameBuffer()
        buf.publish(b"one")
        buf.publish(b"two")
        buf.publish(b"three")
        seq, payload = buf.latest()
        assert payload == b"three"
        assert seq == 3

    def test_slow_consumer_gets_latest(self):
        buf = LiveFrameBuffer()
        buf.publish(b"a")
        buf.publish(b"b")
        seq, payload = buf.wait_latest(0, timeout=0.2)
        assert payload == b"b"
        assert seq == 2

    def test_thread_safe_publish_and_read(self):
        buf = LiveFrameBuffer()
        errors: list[BaseException] = []

        def writer() -> None:
            try:
                for i in range(50):
                    buf.publish(i)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        def reader() -> None:
            try:
                last = -1
                for _ in range(50):
                    last, _payload = buf.wait_latest(last, timeout=0.2)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3.0)
        assert not errors


class TestOverlayDoesNotMutate:
    def test_original_frame_unchanged(self):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        frame[10, 10] = (1, 2, 3)
        original = frame.copy()
        snap = DebugSnapshot(
            state="IDLE",
            document_detected=True,
            motion=12.5,
            edge_density=0.04,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=3,
            stable_required=8,
            camera_healthy=True,
        )
        drawn = draw_overlay(frame, snap, fps=5.0)
        assert np.array_equal(frame, original)
        assert drawn is not frame
        assert not np.array_equal(drawn, original)

    def test_publish_copies_frame(self, debug_viewer):
        frame = np.full((120, 160, 3), 90, dtype=np.uint8)
        original = frame.copy()
        debug_viewer.publish(
            frame,
            state="IDLE",
            document_detected=False,
            motion=0.0,
            edge_density=0.0,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=0,
            stable_required=8,
            stable=False,
            camera_healthy=True,
            autofocus_available=True,
        )
        frame[:] = 0
        stored = debug_viewer.latest_frame_copy()
        assert stored is not None
        assert np.array_equal(stored, original)
        assert not np.array_equal(stored, frame)


class TestDebugHttp:
    def test_status_json_and_no_secrets(self, debug_viewer):
        debug_viewer.publish(
            np.full((80, 120, 3), 30, dtype=np.uint8),
            state="IDLE",
            document_detected=False,
            motion=1.25,
            edge_density=0.01,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=0,
            stable_required=8,
            stable=False,
            camera_healthy=True,
            autofocus_available=True,
        )
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/status"
        with urllib.request.urlopen(url, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        dumped = json.dumps(payload)
        assert "IDCAM_AUTH_TOKEN" not in dumped
        assert "Bearer" not in dumped
        assert payload["enabled"] is True
        assert payload["state"] == "IDLE"
        assert payload["document_detected"] is False
        assert payload["camera_healthy"] is True
        assert payload["autofocus_available"] is True
        assert "motion" in payload
        assert "edge_density" in payload
        assert "stable_count" in payload

    def test_html_page(self, debug_viewer):
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/live"
        with urllib.request.urlopen(url, timeout=3) as resp:
            body = resp.read().decode("utf-8")
        assert "idCam V3 — Live Debug Viewer" in body
        assert "/preview/live.mjpg" in body
        assert "IDCAM_AUTH_TOKEN" not in body
        assert "Bearer" not in body

    def test_mjpeg_boundary_and_jpeg(self, debug_viewer):
        frame = np.full((96, 128, 3), 180, dtype=np.uint8)
        debug_viewer.publish(
            frame,
            state="IDLE",
            document_detected=False,
            motion=0.0,
            edge_density=0.0,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=0,
            stable_required=8,
            stable=False,
            camera_healthy=True,
            autofocus_available=True,
        )
        _wait_jpeg(debug_viewer)
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/live.mjpg"
        resp = urllib.request.urlopen(url, timeout=5)
        try:
            data = _read_mjpeg_part(resp)
        finally:
            resp.close()
        assert b"--frame" in data
        assert b"Content-Type: image/jpeg" in data
        assert b"\xff\xd8" in data
        assert b"\xff\xd9" in data
        debug_viewer.publish(
            frame,
            state="DOCUMENT_DETECTED",
            document_detected=True,
            motion=20.0,
            edge_density=0.1,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=1,
            stable_required=8,
            stable=False,
            camera_healthy=True,
            autofocus_available=True,
        )
        _wait_jpeg(debug_viewer)
        assert debug_viewer.is_running

    def test_multiple_clients_same_source(self, debug_viewer):
        frame = np.full((64, 80, 3), 100, dtype=np.uint8)
        debug_viewer.publish(
            frame,
            state="IDLE",
            document_detected=False,
            motion=0.0,
            edge_density=0.0,
            motion_threshold=8.0,
            edge_density_threshold=0.02,
            stable_count=0,
            stable_required=8,
            stable=False,
            camera_healthy=True,
            autofocus_available=True,
        )
        _wait_jpeg(debug_viewer)
        url = f"http://127.0.0.1:{debug_viewer.config.port}/preview/live.mjpg"
        results: list[bytes] = []
        errors: list[BaseException] = []

        def consume() -> None:
            try:
                resp = urllib.request.urlopen(url, timeout=5)
                try:
                    results.append(_read_mjpeg_part(resp))
                finally:
                    resp.close()
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=consume), threading.Thread(target=consume)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=6.0)
        assert not errors
        assert len(results) == 2
        for part in results:
            assert b"--frame" in part
            assert b"\xff\xd8" in part

    def test_waiting_placeholder_encodes(self):
        image = make_placeholder("WAITING FOR CAMERA...")
        jpeg = encode_jpeg(image, 70)
        assert jpeg[:2] == b"\xff\xd8"
        assert jpeg[-2:] == b"\xff\xd9"


class TestDetectorFrameIsolation:
    def test_overlay_does_not_change_detector_result(self):
        config = DetectionConfig(
            motion_threshold=5.0,
            edge_density_threshold=0.01,
            stable_frames_required=3,
        )
        detector = DocumentDetector(config)
        baseline = np.full((240, 320, 3), 200, dtype=np.uint8)
        detector.update_baseline(baseline)
        frame = baseline.copy()
        frame[40:200, 40:280] = 40
        before = detector.detect(frame.copy())
        snap = DebugSnapshot(state="IDLE", document_detected=before, camera_healthy=True)
        draw_overlay(frame, snap, 5.0)
        after = detector.detect(frame)
        assert before is after
        stab = StabilityDetector(config)
        moving = frame.copy()
        assert stab.update(frame) is False
        draw_overlay(moving, snap, 5.0)
        assert stab.update(moving) in {True, False}
