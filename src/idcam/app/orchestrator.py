"""Application orchestrator — single authority for state and transactions."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path

from idcam import __version__
from idcam.app.state_machine import AppState, StateMachine
from idcam.camera.autofocus import AutofocusController
from idcam.camera.capture import CaptureService
from idcam.camera.manager import CameraManager
from idcam.camera.quality import ImageValidator
from idcam.config.schema import AppConfig
from idcam.exceptions import (
    AutofocusError,
    CameraError,
    CaptureError,
    ImageValidationError,
    NetworkError,
    RecoveryExhaustedError,
    ServerRejectedError,
    ServerError,
    ShutdownRequested,
    TransactionError,
)
from idcam.logging.setup import get_logger, set_log_context
from idcam.network.api_client import ApiClient
from idcam.network.protocol import CaptureTransaction, UploadOutcome
from idcam.storage.temp_files import TempFileStore
from idcam.system.health import HealthStatus
from idcam.system.watchdog import Watchdog
from idcam.ui.status import StatusPublisher
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.stability_detector import StabilityDetector

logger = get_logger("orchestrator")


class ApplicationOrchestrator:
    """
    Main application loop with explicit state machine.

    Exactly one capture transaction at a time. No queue.
    """

    def __init__(self, config: AppConfig, *, use_mock_camera: bool = False) -> None:
        self.config = config
        self.sm = StateMachine()
        self.health = HealthStatus()
        self.status = StatusPublisher(config.ui.status_file, version=__version__)
        self.camera = CameraManager(config.camera, use_mock=use_mock_camera)
        self.autofocus = AutofocusController(config.camera)
        self.capture_service = CaptureService(config.camera)
        self.validator = ImageValidator(config.quality)
        self.detector = DocumentDetector(config.detection)
        self.stability = StabilityDetector(config.detection)
        self.temp_store = TempFileStore(config.temp_dir)
        self.api = ApiClient(config)
        self.watchdog = Watchdog(config.watchdog)
        self._shutdown_event = asyncio.Event()
        self._current_capture_id: str | None = None
        self._recovery_attempts = 0
        self._removal_count = 0
        self._document_captured = False

    async def run(self) -> int:
        """Run until shutdown. Returns exit code."""
        self._install_signal_handlers()
        try:
            await self._startup()
            await self._main_loop()
        except ShutdownRequested:
            logger.info("Shutdown requested")
        except Exception as exc:
            logger.exception("Fatal error: %s", exc)
            self._transition(AppState.ERROR, reason=str(exc))
            return 1
        finally:
            await self._shutdown()
        return 0

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._request_shutdown)
            except NotImplementedError:
                # Windows
                signal.signal(sig, lambda *_: self._request_shutdown())

    def _request_shutdown(self) -> None:
        logger.info("Signal received, requesting shutdown")
        self._shutdown_event.set()

    async def _startup(self) -> None:
        self._transition(AppState.INITIALIZING)
        self.temp_store.cleanup_stale()
        self.temp_store.ensure_space(100_000_000)

        await self.api.start()
        self.health.network_healthy = await self.api.check_connectivity()
        self.health.server_healthy = self.health.network_healthy

        self._transition(AppState.CAMERA_INITIALIZING)
        await asyncio.to_thread(self.camera.initialize)
        self.autofocus.probe_capabilities(self.camera)
        self.health.camera_healthy = True

        self._transition(AppState.CAMERA_READY)
        self._transition(AppState.IDLE)
        logger.info("Application ready")

    async def _main_loop(self) -> None:
        frame_interval = 1.0 / self.config.camera.preview_fps

        while not self._shutdown_event.is_set():
            if self.watchdog.check(
                self.sm,
                upload_timeout=self.config.server.total_timeout_sec,
            ):
                await self._enter_recovery("watchdog_timeout")
                continue

            if self.sm.state == AppState.IDLE:
                await self._idle_tick()
            elif self.sm.state == AppState.DOCUMENT_DETECTED:
                await self._document_detected_tick()
            elif self.sm.state == AppState.WAITING_FOR_STABILITY:
                await self._stability_tick()
            elif self.sm.state == AppState.FOCUSING:
                await self._focusing_tick()
            elif self.sm.state == AppState.FOCUS_VERIFICATION:
                await self._focus_verification_tick()
            elif self.sm.state == AppState.CAPTURING:
                await self._capturing_tick()
            elif self.sm.state == AppState.IMAGE_VALIDATION:
                await self._validation_tick()
            elif self.sm.state == AppState.UPLOADING:
                await self._uploading_tick()
            elif self.sm.state == AppState.WAITING_FOR_SERVER:
                await self._waiting_server_tick()
            elif self.sm.state == AppState.PROCESSING_RESPONSE:
                await self._processing_response_tick()
            elif self.sm.state == AppState.SUCCESS:
                await self._success_tick()
            elif self.sm.state == AppState.WAITING_FOR_REMOVAL:
                await self._removal_tick()
            elif self.sm.state == AppState.RECOVERY:
                await self._recovery_tick()
            elif self.sm.state == AppState.ERROR:
                await self._error_tick()
            else:
                await asyncio.sleep(0.1)

            await asyncio.sleep(frame_interval)

    async def _idle_tick(self) -> None:
        frame = await asyncio.to_thread(self.camera.capture_preview_array)
        if self.detector.detect(frame):
            logger.info("Document detected")
            self.stability.reset()
            self._document_captured = False
            self._transition(AppState.DOCUMENT_DETECTED)

    async def _document_detected_tick(self) -> None:
        if self.sm.is_timed_out():
            logger.info("Document detection timed out, returning to IDLE")
            self._transition(AppState.IDLE)
            return
        self._transition(AppState.WAITING_FOR_STABILITY)

    async def _stability_tick(self) -> None:
        frame = await asyncio.to_thread(self.camera.capture_preview_array)
        if self.stability.update(frame):
            logger.info("Document stable (%d frames)", self.stability.stable_count)
            self._current_capture_id = CaptureTransaction.generate_id()
            set_log_context(capture_id=self._current_capture_id)
            self._transition(AppState.FOCUSING)
        elif self.sm.is_timed_out():
            logger.info("Stability timeout, returning to IDLE")
            self._fail_transaction("stability_timeout")
            self._transition(AppState.IDLE)

    async def _focusing_tick(self) -> None:
        try:
            await self.autofocus.run_autofocus(
                self.camera,
                sharpness_threshold=self.config.quality.min_sharpness * 0.6,
                roi_fraction=self.config.detection.roi_center_fraction,
            )
            self._transition(AppState.FOCUS_VERIFICATION)
        except AutofocusError as exc:
            logger.error("Autofocus failed: %s", exc)
            await self._fail_transaction_and_wait("focus_failed", str(exc))

    async def _focus_verification_tick(self) -> None:
        try:
            await self.autofocus.verify_focus(
                self.camera,
                sharpness_threshold=self.config.quality.min_sharpness * 0.6,
                roi_fraction=self.config.detection.roi_center_fraction,
            )
            self._transition(AppState.CAPTURING)
        except AutofocusError:
            if self.sm.transition_count < 20:  # bounded refocus
                self._transition(AppState.FOCUSING, reason="refocus")
            else:
                await self._fail_transaction_and_wait("focus_failed", "Focus verification failed")

    async def _capturing_tick(self) -> None:
        assert self._current_capture_id is not None
        temp_path = self.temp_store.create_temp_path()
        try:
            await self.capture_service.capture(self.camera, temp_path)
            self._capture_path = temp_path
            self._transition(AppState.IMAGE_VALIDATION)
        except CaptureError as exc:
            logger.error("Capture failed: %s", exc)
            self.temp_store.cleanup(temp_path)
            await self._enter_recovery(str(exc))

    async def _validation_tick(self) -> None:
        try:
            self._validation_result = self.validator.validate(self._capture_path)
            self._transition(AppState.UPLOADING)
        except ImageValidationError as exc:
            logger.error("Image validation failed: %s", exc)
            self.temp_store.cleanup(self._capture_path)
            await self._fail_transaction_and_wait("image_invalid", str(exc))

    async def _uploading_tick(self) -> None:
        assert self._current_capture_id is not None
        transaction = CaptureTransaction(
            capture_id=self._current_capture_id,
            device_id=self.config.device_id,
            image_path=str(self._capture_path),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        try:
            self._pending_response = await self.api.upload_document(transaction)
        except (ServerRejectedError, ServerError) as exc:
            self._pending_response = None
            self.temp_store.cleanup(self._capture_path)
            await self._fail_transaction_and_wait("server_rejected", str(exc))
            return
        except NetworkError as exc:
            self._pending_response = None
            self.temp_store.cleanup(self._capture_path)
            await self._fail_transaction_and_wait("upload_failed", str(exc))
            return
        self._transition(AppState.WAITING_FOR_SERVER)

    async def _waiting_server_tick(self) -> None:
        self._transition(AppState.PROCESSING_RESPONSE)

    async def _processing_response_tick(self) -> None:
        response = self._pending_response
        if response.success and response.outcome == UploadOutcome.ACCEPTED:
            self.health.record_success()
            self.health.server_healthy = True
            self._document_captured = True
            self.temp_store.cleanup(self._capture_path)
            self._transition(AppState.SUCCESS)
        else:
            self.health.record_failure(response.message or "upload_failed")
            self.temp_store.cleanup(self._capture_path)
            failure_key = "upload_failed"
            if response.outcome == UploadOutcome.REJECTED:
                failure_key = "server_rejected"
            await self._fail_transaction_and_wait(failure_key, response.message or "Upload failed")

    async def _success_tick(self) -> None:
        self.status.update(
            AppState.SUCCESS,
            capture_id=self._current_capture_id,
            last_success_at=time.time(),
            camera_healthy=True,
            network_healthy=True,
            server_healthy=True,
        )
        await asyncio.sleep(2.0)
        self._removal_count = 0
        self._transition(AppState.WAITING_FOR_REMOVAL)

    async def _removal_tick(self) -> None:
        frame = await asyncio.to_thread(self.camera.capture_preview_array)
        if self.detector.is_scene_empty(frame):
            self._removal_count += 1
        else:
            self._removal_count = 0

        if self._removal_count >= self.config.detection.removal_frames_required:
            logger.info("Document removed, returning to IDLE")
            self.detector.reset_baseline()
            self._current_capture_id = None
            set_log_context(capture_id=None)
            self._transition(AppState.IDLE)
        elif self.sm.is_timed_out():
            logger.warning("Removal wait timed out, resetting to IDLE")
            self.detector.reset_baseline()
            self._current_capture_id = None
            set_log_context(capture_id=None)
            self._transition(AppState.IDLE)

    async def _recovery_tick(self) -> None:
        self._recovery_attempts += 1
        max_attempts = self.config.camera.recovery_max_attempts

        if self._recovery_attempts > max_attempts:
            logger.error("Recovery exhausted after %d attempts", max_attempts)
            self.health.record_failure("recovery_exhausted")
            self._transition(AppState.ERROR)
            raise RecoveryExhaustedError("Recovery attempts exhausted")

        logger.info("Recovery attempt %d/%d", self._recovery_attempts, max_attempts)
        self.health.record_recovery()

        try:
            await asyncio.sleep(self.config.camera.recovery_delay_sec)
            await asyncio.to_thread(self.camera.restart)
            self.health.camera_healthy = True
            self._recovery_attempts = 0
            self._transition(AppState.IDLE)
        except CameraError as exc:
            logger.error("Recovery failed: %s", exc)
            self.health.camera_healthy = False

    async def _error_tick(self) -> None:
        await asyncio.sleep(10.0)
        if not self._shutdown_event.is_set():
            self._transition(AppState.RECOVERY, reason="auto_recovery_from_error")

    async def _enter_recovery(self, reason: str) -> None:
        logger.warning("Entering recovery: %s", reason)
        self._cleanup_transaction_files()
        self._transition(AppState.RECOVERY, reason=reason)

    async def _fail_transaction_and_wait(self, failure_key: str, detail: str) -> None:
        self._fail_transaction(detail)
        self.status.update(
            AppState.WAITING_FOR_REMOVAL,
            capture_id=self._current_capture_id,
            failure_key=failure_key,
            last_error=detail,
        )
        self._transition(AppState.WAITING_FOR_REMOVAL)

    def _fail_transaction(self, detail: str) -> None:
        logger.warning("Transaction failed: %s", detail)
        self.health.record_failure(detail)
        self._cleanup_transaction_files()

    def _cleanup_transaction_files(self) -> None:
        path = getattr(self, "_capture_path", None)
        if path:
            self.temp_store.cleanup(Path(path))

    def _transition(self, target: AppState, *, reason: str = "") -> None:
        prev = self.sm.state
        self.sm.transition(target, reason=reason)
        set_log_context(state=target.value)
        logger.info("State %s -> %s%s", prev.value, target.value, f" ({reason})" if reason else "")
        self.status.update(
            target,
            capture_id=self._current_capture_id,
            camera_healthy=self.health.camera_healthy,
            network_healthy=self.health.network_healthy,
            server_healthy=self.health.server_healthy,
        )

    async def _shutdown(self) -> None:
        self._transition(AppState.SHUTDOWN)
        self._cleanup_transaction_files()
        await self.api.stop()
        await asyncio.to_thread(self.camera.shutdown)
        logger.info("Shutdown complete")
