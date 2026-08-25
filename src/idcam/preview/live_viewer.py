"""Live Debug Viewer — observer of the existing preview pipeline.

ONE producer (the orchestrator preview path) publishes the latest frame.
Multiple HTTP clients consume the latest JPEG. There is no per-client camera
capture and no unbounded queue.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from idcam.config.schema import DebugConfig
from idcam.logging.setup import get_logger

logger = get_logger("preview.live_viewer")

MJPEG_BOUNDARY = "frame"
PLACEHOLDER_WIDTH = 1280
PLACEHOLDER_HEIGHT = 720
_MAX_CLIENT_LOGS_PER_MINUTE = 20


@dataclass
class DebugSnapshot:
    """Latest debug view of the capture pipeline. Frame may be None."""

    frame: np.ndarray | None = None
    timestamp: float = 0.0
    state: str = "BOOT"
    document_detected: bool = False
    motion: float = 0.0
    edge_density: float = 0.0
    motion_threshold: float = 8.0
    edge_density_threshold: float = 0.02
    stable_count: int = 0
    stable_required: int = 8
    stable: bool = False
    camera_healthy: bool = False
    autofocus_available: bool = False
    camera_error: str | None = None
    roi_center_fraction: float = 0.7
    waiting_for_camera: bool = True


class LiveFrameBuffer:
    """Thread-safe latest-slot buffer. Slow consumers skip to the newest item."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._seq = 0
        self._payload: Any = None

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._seq

    def publish(self, payload: Any) -> None:
        with self._cond:
            self._payload = payload
            self._seq += 1
            self._cond.notify_all()

    def latest(self) -> tuple[int, Any]:
        with self._lock:
            return self._seq, self._payload

    def wait_latest(self, last_seq: int, timeout: float) -> tuple[int, Any]:
        """Return the newest payload. Never queues older frames."""
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            return self._seq, self._payload

    def notify_all(self) -> None:
        with self._cond:
            self._cond.notify_all()


def resize_debug_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Resize only a debug copy. Never upscale."""
    if frame.ndim < 2:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width or max_width <= 0:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    bgr = _to_bgr(frame)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _draw_panel(image: np.ndarray, lines: list[str], *, banner: str | None = None) -> None:
    overlay = image.copy()
    line_h = 28
    pad = 12
    panel_w = 520
    panel_h = pad * 2 + line_h * len(lines)
    if banner:
        panel_h += line_h + 8
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, dst=image)

    y = 8 + pad + 18
    if banner:
        cv2.putText(
            image,
            banner,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 80, 80),
            2,
            cv2.LINE_AA,
        )
        y += line_h + 4
    for line in lines:
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_h


def draw_overlay(frame: np.ndarray, snapshot: DebugSnapshot, fps: float) -> np.ndarray:
    """Draw diagnostics on a copy of ``frame``. The input array is not modified."""
    debug_frame = frame.copy()
    if debug_frame.ndim == 2:
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2RGB)

    frac = snapshot.roi_center_fraction
    h, w = debug_frame.shape[:2]
    rh, rw = int(h * frac), int(w * frac)
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    cv2.rectangle(debug_frame, (x0, y0), (x0 + rw, y0 + rh), (0, 220, 80), 2)

    camera = "OK" if snapshot.camera_healthy else "ERROR"
    if not snapshot.camera_healthy and snapshot.waiting_for_camera and not snapshot.camera_error:
        camera = "WAITING"
    document = "YES" if snapshot.document_detected else "NO"
    banner = None
    if snapshot.waiting_for_camera and not snapshot.camera_error:
        banner = "WAITING FOR CAMERA..."
    if snapshot.camera_error or (not snapshot.camera_healthy and not snapshot.waiting_for_camera):
        banner = "CAMERA ERROR"

    lines = [
        f"STATE: {snapshot.state}",
        f"DOCUMENT: {document}",
        f"MOTION: {snapshot.motion:.2f}",
        f"MOTION THRESHOLD: {snapshot.motion_threshold:.2f}",
        f"EDGE: {snapshot.edge_density:.4f}",
        f"EDGE THRESHOLD: {snapshot.edge_density_threshold:.4f}",
        f"STABLE: {snapshot.stable_count}/{snapshot.stable_required}",
        f"FPS: {fps:.1f}",
        f"CAMERA: {camera}",
    ]
    if snapshot.camera_error and not snapshot.camera_healthy:
        err = snapshot.camera_error.replace("\n", " ")[:60]
        lines.append(f"DETAIL: {err}")
    _draw_panel(debug_frame, lines, banner=banner)
    return debug_frame


def make_placeholder(message: str) -> np.ndarray:
    image = np.full((PLACEHOLDER_HEIGHT, PLACEHOLDER_WIDTH, 3), 28, dtype=np.uint8)
    cv2.putText(
        image,
        message,
        (48, PLACEHOLDER_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return image


_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>idCam V3 — Live Debug Viewer</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif;
           background: #111; color: #eee; }
    header { padding: 16px 20px; background: #1b1b1b; border-bottom: 1px solid #333; }
    h1 { margin: 0; font-size: 1.25rem; font-weight: 600; }
    .note { color: #aaa; font-size: 0.85rem; margin-top: 4px; }
    main { display: grid; grid-template-columns: 1fr 320px; gap: 16px; padding: 16px; }
    img { width: 100%; max-width: 1280px; background: #000; border: 1px solid #333; }
    dl { margin: 0; }
    dt { color: #888; font-size: 0.75rem; text-transform: uppercase; margin-top: 12px; }
    dd { margin: 2px 0 0; font-variant-numeric: tabular-nums; font-size: 1.05rem; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>idCam V3 — Live Debug Viewer</h1>
    <div class="note">Debug only. Live camera preview — do not enable on untrusted networks.</div>
  </header>
  <main>
    <section>
      <img src="/preview/live.mjpg" alt="Live preview">
    </section>
    <aside>
      <dl>
        <dt>State</dt><dd id="state">—</dd>
        <dt>Camera</dt><dd id="camera">—</dd>
        <dt>Detection</dt><dd id="detection">—</dd>
        <dt>Motion</dt><dd id="motion">—</dd>
        <dt>Edge density</dt><dd id="edge">—</dd>
        <dt>Stable frames</dt><dd id="stable">—</dd>
        <dt>FPS</dt><dd id="fps">—</dd>
        <dt>Timestamp</dt><dd id="timestamp">—</dd>
      </dl>
    </aside>
  </main>
  <script>
    const el = (id) => document.getElementById(id);
    async function refresh() {
      try {
        const r = await fetch("/preview/status", { cache: "no-store" });
        if (!r.ok) throw new Error("status " + r.status);
        const d = await r.json();
        el("state").textContent = d.state || "—";
        el("camera").textContent = d.camera_healthy ? "healthy" : "unhealthy";
        el("detection").textContent = d.document_detected ? "detected" : "not detected";
        el("motion").textContent = Number(d.motion).toFixed(2);
        el("edge").textContent = Number(d.edge_density).toFixed(4);
        el("stable").textContent = (d.stable_count ?? 0) + " / " + (d.stable_required ?? 0);
        el("fps").textContent = Number(d.fps).toFixed(1);
        const ts = d.timestamp ? new Date(d.timestamp * 1000).toISOString() : "—";
        el("timestamp").textContent = ts;
      } catch (err) {
        el("camera").textContent = "unhealthy";
      }
    }
    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""


def _make_handler(viewer: LiveDebugViewer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/preview/live":
                self._serve_html()
            elif path == "/preview/live.mjpg":
                self._serve_mjpeg()
            elif path == "/preview/status":
                self._serve_status()
            else:
                self.send_error(404, "Not Found")

        def _serve_html(self) -> None:
            body = _HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_status(self) -> None:
            payload = json.dumps(viewer.status_dict(), separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _serve_mjpeg(self) -> None:
            viewer.client_connected(self.client_address)
            try:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
                )
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                last_seq = -1
                while viewer.is_running:
                    seq, jpeg = viewer.wait_jpeg(last_seq, timeout=1.0)
                    if jpeg is None or seq == last_seq:
                        continue
                    last_seq = seq
                    try:
                        self.wfile.write(
                            (
                                f"--{MJPEG_BOUNDARY}\r\n"
                                "Content-Type: image/jpeg\r\n"
                                f"Content-Length: {len(jpeg)}\r\n"
                                "\r\n"
                            ).encode("ascii")
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                        break
                    except OSError:
                        break
            finally:
                viewer.client_disconnected(self.client_address)

    return Handler


class LiveDebugViewer:
    """Background JPEG encoder + HTTP observer. Never captures from the camera."""

    def __init__(self, config: DebugConfig) -> None:
        self.config = config
        self._meta_lock = threading.Lock()
        self._frame_seq = 0
        self._latest_frame: np.ndarray | None = None
        self._snapshot = DebugSnapshot()
        self._jpeg_buffer = LiveFrameBuffer()
        self._stop = threading.Event()
        self._encoder_thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None
        self._server: ThreadingHTTPServer | None = None
        self._clients = 0
        self._clients_lock = threading.Lock()
        self._encode_times: deque[float] = deque(maxlen=12)
        self._actual_fps = 0.0
        self._client_log_times: deque[float] = deque(maxlen=_MAX_CLIENT_LOGS_PER_MINUTE)
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._started and not self._stop.is_set()

    def start(self) -> None:
        if self._started:
            return
        handler = _make_handler(self)
        try:
            self._server = ThreadingHTTPServer((self.config.host, self.config.port), handler)
            self._server.daemon_threads = True
            self._server.timeout = 0.5
        except OSError as exc:
            logger.error(
                "event=debug_viewer_bind_failed host=%s port=%s error=%s",
                self.config.host,
                self.config.port,
                exc,
            )
            self._server = None
            return

        self._stop.clear()
        self._encoder_thread = threading.Thread(
            target=self._encoder_loop, name="idcam-debug-encoder", daemon=True
        )
        self._http_thread = threading.Thread(
            target=self._http_loop, name="idcam-debug-http", daemon=True
        )
        self._encoder_thread.start()
        self._http_thread.start()
        self._started = True
        logger.info(
            "Live Debug Viewer enabled: http://%s:%s/preview/live",
            self.config.host,
            self.config.port,
        )
        logger.info(
            "event=debug_viewer_started host=%s port=%s fps=%.1f overlay=%s",
            self.config.host,
            self.config.port,
            self.config.fps,
            self.config.overlay,
        )

    def stop(self) -> None:
        if not self._started and self._server is None:
            return
        self._stop.set()
        self._jpeg_buffer.notify_all()
        server = self._server
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                logger.debug("event=debug_viewer_shutdown_error", exc_info=True)
            try:
                server.server_close()
            except Exception:
                logger.debug("event=debug_viewer_close_error", exc_info=True)
        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=2.0)
        self._server = None
        self._started = False
        logger.info("event=debug_viewer_stopped")

    def publish(
        self,
        frame: np.ndarray | None = None,
        *,
        state: str,
        document_detected: bool,
        motion: float,
        edge_density: float,
        motion_threshold: float,
        edge_density_threshold: float,
        stable_count: int,
        stable_required: int,
        stable: bool,
        camera_healthy: bool,
        autofocus_available: bool,
        camera_error: str | None = None,
        roi_center_fraction: float = 0.7,
    ) -> None:
        """Store the latest preview copy and metadata. Never blocks on HTTP clients."""
        with self._meta_lock:
            if frame is not None:
                self._latest_frame = frame.copy()
                self._frame_seq += 1
            self._snapshot = DebugSnapshot(
                frame=None,
                timestamp=time.time(),
                state=state,
                document_detected=document_detected,
                motion=motion,
                edge_density=edge_density,
                motion_threshold=motion_threshold,
                edge_density_threshold=edge_density_threshold,
                stable_count=stable_count,
                stable_required=stable_required,
                stable=stable,
                camera_healthy=camera_healthy,
                autofocus_available=autofocus_available,
                camera_error=camera_error,
                roi_center_fraction=roi_center_fraction,
                waiting_for_camera=self._latest_frame is None,
            )

    def snapshot_meta(self) -> DebugSnapshot:
        with self._meta_lock:
            return self._snapshot

    def latest_frame_copy(self) -> np.ndarray | None:
        with self._meta_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def status_dict(self) -> dict[str, Any]:
        snap = self.snapshot_meta()
        return {
            "enabled": True,
            "state": snap.state,
            "document_detected": snap.document_detected,
            "motion": snap.motion,
            "edge_density": snap.edge_density,
            "motion_threshold": snap.motion_threshold,
            "edge_density_threshold": snap.edge_density_threshold,
            "stable_count": snap.stable_count,
            "stable_required": snap.stable_required,
            "stable": snap.stable,
            "camera_healthy": snap.camera_healthy,
            "autofocus_available": snap.autofocus_available,
            "fps": round(self._actual_fps or self.config.fps, 2),
            "timestamp": snap.timestamp,
        }

    def wait_jpeg(self, last_seq: int, timeout: float) -> tuple[int, bytes | None]:
        seq, payload = self._jpeg_buffer.wait_latest(last_seq, timeout)
        jpeg = payload if isinstance(payload, (bytes, bytearray)) else None
        return seq, jpeg

    def client_connected(self, address: tuple[str, int]) -> None:
        with self._clients_lock:
            self._clients += 1
            count = self._clients
        self._log_client("event=debug_client_connected", address, count)

    def client_disconnected(self, address: tuple[str, int]) -> None:
        with self._clients_lock:
            self._clients = max(0, self._clients - 1)
            count = self._clients
        self._log_client("event=debug_client_disconnected", address, count)

    def _log_client(self, event: str, address: tuple[str, int], count: int) -> None:
        now = time.monotonic()
        recent = sum(1 for t in self._client_log_times if now - t < 60.0)
        if recent >= _MAX_CLIENT_LOGS_PER_MINUTE:
            logger.debug("%s clients=%d peer=%s", event, count, address[0])
        else:
            logger.info("%s clients=%d peer=%s", event, count, address[0])
        self._client_log_times.append(now)

    def _http_loop(self) -> None:
        server = self._server
        if server is None:
            return
        try:
            server.serve_forever(poll_interval=0.3)
        except Exception:
            if not self._stop.is_set():
                logger.exception("event=debug_http_server_failed")

    def _encoder_loop(self) -> None:
        interval = 1.0 / max(self.config.fps, 0.1)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                jpeg = self._encode_latest()
                if jpeg is not None:
                    self._jpeg_buffer.publish(jpeg)
                    now = time.monotonic()
                    self._encode_times.append(now)
                    if len(self._encode_times) >= 2:
                        span = self._encode_times[-1] - self._encode_times[0]
                        if span > 0:
                            self._actual_fps = (len(self._encode_times) - 1) / span
            except Exception:
                logger.exception("event=debug_encode_failed")
            remaining = interval - (time.monotonic() - t0)
            self._stop.wait(timeout=max(0.0, remaining))

    def _encode_latest(self) -> bytes | None:
        with self._meta_lock:
            frame = self._latest_frame
            snap = self._snapshot
        if frame is None:
            message = "WAITING FOR CAMERA..."
            if snap.camera_error:
                message = "CAMERA ERROR"
            image = make_placeholder(message)
            if self.config.overlay:
                image = draw_overlay(image, snap, self._actual_fps or self.config.fps)
            else:
                image = resize_debug_frame(image, self.config.max_width)
            return encode_jpeg(image, self.config.jpeg_quality)

        debug_copy = frame.copy()
        debug_copy = resize_debug_frame(debug_copy, self.config.max_width)
        if self.config.overlay:
            debug_copy = draw_overlay(debug_copy, snap, self._actual_fps or self.config.fps)
        return encode_jpeg(debug_copy, self.config.jpeg_quality)
