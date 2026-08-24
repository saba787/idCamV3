"""Application orchestrator — single authority for state and transactions."""

from __future__ import annotations

import asyncio
import signal
import time
from pathlib import Path

from idcam import __version__
from idcam.app.state_machine import AppState, StateMachine
from idcam.app.transaction import ActiveCaptureTransaction, TransactionPhase
from idcam.camera.autofocus import AutofocusController
from idcam.camera.capture import CaptureService
from idcam.camera.manager import CameraManager
from idcam.camera.quality import ImageValidator
from idcam.config.schema import AppConfig
from idcam.exceptions import (
    AutofocusError,
    CameraError,
    CameraUnavailableError,
    CaptureError,
    ImageValidationError,
    NetworkError,
    ProtocolError,
    RecoveryExhaustedError,
    ServerRejectedError,
    ServerError,
    ShutdownRequested,
)
from idcam.logging.setup import get_logger, set_log_context
from idcam.network.api_client import ApiClient
from idcam.network.protocol import UploadOutcome, UploadRequest
from idcam.storage.disk_monitor import DiskMonitor, StorageHealth
from idcam.storage.temp_files import TempFileStore
from idcam.system.component_health import ComponentStatus, HealthLevel, SystemHealth
from idcam.system.watchdog import Watchdog
from idcam.ui.status import StatusPublisher
from idcam.vision.document_detector import DocumentDetector, DocumentPresence
from idcam.vision.stability_detector import StabilityDetector

logger = get_logger("orchestrator")

# States that run a single long-lived action per state entry (no re-entry per loop tick).
ONE_SHOT_STATES = frozenset({
    AppState.DOCUMENT_DETECTED,
    AppState.FOCUSING,
    AppState.FOCUS_VERIFICATION,
    AppState.CAPTURING,
    AppState.IMAGE_VALIDATION,
    AppState.UPLOADING,
    AppState.WAITING_FOR_SERVER,
    AppState.PROCESSING_RESPONSE,
    AppState.SUCCESS,
})


class ApplicationOrchestrator:
    """
    Main application loop with explicit state machine.

    Exactly one capture transaction at a time. No queue.
    """

    def __init__(self, config: AppConfig, *, use_mock_camera: bool = False) -> None:
        self.config = config
        self.sm = StateMachine()
        self.sm.set_timeouts(config.state_timeouts)
        self.health = SystemHealth()
        self.status = StatusPublisher(config.ui.status_file, version=__version__)
        self.camera = CameraManager(config.camera, use_mock=use_mock_camera)
        self.autofocus = AutofocusController(config.camera)
        self.temp_store = TempFileStore(config.temp_dir)
        self.capture_service = CaptureService(config.camera, self.temp_store)
        self.validator = ImageValidator(config.quality, config.camera)
        self.detector = DocumentDetector(config.detection)
        self.stability = StabilityDetector(config.detection)
        self.disk = DiskMonitor(config.storage, config.temp_dir)
        self.api = ApiClient(config)
        self.watchdog = Watchdog(config.watchdog, config.state_timeouts)
        self._shutdown_event = asyncio.Event()
        self._transaction: ActiveCaptureTransaction | None = None
        self._recovery_attempts = 0
        self._next_recovery_at: float = 0.0
        self._recovery_in_progress = False
        self._removal_count = 0
        self._state_tick_running = False
        self._state_action_claimed: AppState | None = None
        self._error_backoff_claimed = False
        self._capture_path: Path | None = None
        self._pending_response = None

    @property
    def has_active_transaction(self) -> bool:
        """True when the state machine is in any transactional state."""
        return self.sm.in_transaction()

    async def run(self) -> int:
        self._install_signal_handlers()
        try:
            await self._startup()
            await self._main_loop()
        except ShutdownRequested:
            logger.info("Shutdown requested")
        except RecoveryExhaustedError:
            return 1
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
                signal.signal(sig, lambda *_: self._request_shutdown())

    def _request_shutdown(self) -> None:
        logger.info("Signal received, requesting shutdown")
        self._shutdown_event.set()

    async def _startup(self) -> None:
        self._transition(AppState.INITIALIZING)
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        removed = self.temp_store.cleanup_stale(self.config.storage.stale_temp_max_age_sec)
        if removed:
            logger.info("Startup: removed %d stale temp files", removed)

        storage_status = self.disk.check()
        self._update_storage_health(storage_status)

        await self.api.start()
        network_ok = await self.api.check_connectivity()
        self.health.network = ComponentStatus(
            level=HealthLevel.HEALTHY if network_ok else HealthLevel.DEGRADED,
            message=None if network_ok else "Network unavailable",
            details={"healthy": network_ok},
        )
        self.health.server = ComponentStatus(
            level=HealthLevel.HEALTHY if network_ok else HealthLevel.DEGRADED,
            details={"configured": True, "reachable": network_ok},
        )

        self._transition(AppState.CAMERA_INITIALIZING)
        try:
            await asyncio.to_thread(self.camera.initialize)
        except CameraUnavailableError as exc:
            logger.error("event=camera_init_failed %s", exc)
            self.health.camera = ComponentStatus(
                level=HealthLevel.UNAVAILABLE,
                message=str(exc),
                details={"healthy": False},
            )
            self._transition(AppState.RECOVERY, reason="camera_init_failed")
            return

        caps = self.autofocus.probe_capabilities(self.camera)
        self.health.autofocus_available = caps.hardware_af_available
        self.health.autofocus_degraded = caps.degraded
        self.health.camera = ComponentStatus(
            level=HealthLevel.HEALTHY,
            details={
                "healthy": True,
                "autofocus_available": caps.hardware_af_available,
                "autofocus_degraded": caps.degraded,
            },
        )
        self.health.application = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})

        self._transition(AppState.CAMERA_READY)
        if storage_status.health != StorageHealth.HEALTHY:
            self._transition(AppState.ERROR, reason="storage_degraded")
            return
        self._transition(AppState.IDLE)
        logger.info("Application ready")

    async def _main_loop(self) -> None:
        frame_interval = 1.0 / self.config.camera.preview_fps

        while not self._shutdown_event.is_set():
            self._refresh_health()
            alert = self.watchdog.check(
                self.sm,
                capture_id=self._transaction.capture_id if self._transaction else None,
            )
            if alert is not None:
                await self._handle_watchdog(alert)
                await asyncio.sleep(frame_interval)
                continue

            if await self._handle_state_timeout():
                await asyncio.sleep(frame_interval)
                continue

            await self._dispatch_state_tick()
            await asyncio.sleep(frame_interval)

    async def _handle_state_timeout(self) -> bool:
        if not self.sm.is_timed_out():
            return False
        state = self.sm.state
        logger.warning(
            "event=state_timeout state=%s elapsed=%.1fs capture_id=%s",
            state.value,
            self.sm.elapsed(),
            self._transaction.capture_id if self._transaction else None,
        )
        if state == AppState.DOCUMENT_DETECTED:
            self._abort_transaction("document_detected_timeout")
            self._transition(AppState.IDLE, reason="document_detected_timeout")
            return True
        if state == AppState.WAITING_FOR_STABILITY:
            await self._fail_transaction_and_wait("stability_timeout", "Document not stable in time")
            return True
        if state in {AppState.FOCUSING, AppState.FOCUS_VERIFICATION}:
            await self._fail_transaction_and_wait("focus_failed", f"Focus timed out in {state.value}")
            return True
        if state in {AppState.UPLOADING, AppState.WAITING_FOR_SERVER, AppState.PROCESSING_RESPONSE}:
            await self._fail_transaction_and_wait("upload_failed", "Server did not respond in time")
            return True
        if state == AppState.SUCCESS:
            self._transition(AppState.WAITING_FOR_REMOVAL)
            return True
        if state == AppState.CAPTURING:
            await self._enter_recovery("capture_timeout")
            return True
        if state == AppState.IMAGE_VALIDATION:
            await self._fail_transaction_and_wait("image_invalid", "Image validation timed out")
            return True
        return False

    def _claim_state_action(self) -> bool:
        """Return True if this tick may run the state's action (one-shot per entry)."""
        if self.sm.state not in ONE_SHOT_STATES:
            return True
        if self._state_action_claimed == self.sm.state:
            return False
        self._state_action_claimed = self.sm.state
        return True

    async def _dispatch_state_tick(self) -> None:
        if self._state_tick_running:
            return
        self._state_tick_running = True
        try:
            handlers = {
                AppState.IDLE: self._idle_tick,
                AppState.DOCUMENT_DETECTED: self._document_detected_tick,
                AppState.WAITING_FOR_STABILITY: self._stability_tick,
                AppState.FOCUSING: self._focusing_tick,
                AppState.FOCUS_VERIFICATION: self._focus_verification_tick,
                AppState.CAPTURING: self._capturing_tick,
                AppState.IMAGE_VALIDATION: self._validation_tick,
                AppState.UPLOADING: self._uploading_tick,
                AppState.WAITING_FOR_SERVER: self._waiting_server_tick,
                AppState.PROCESSING_RESPONSE: self._processing_response_tick,
                AppState.SUCCESS: self._success_tick,
                AppState.WAITING_FOR_REMOVAL: self._removal_tick,
                AppState.RECOVERY: self._recovery_tick,
                AppState.ERROR: self._error_tick,
            }
            handler = handlers.get(self.sm.state)
            if handler:
                await handler()
        finally:
            self._state_tick_running = False

    async def _safe_preview(self):
        try:
            return await asyncio.to_thread(self.camera.capture_preview_array)
        except CameraError as exc:
            logger.error("event=camera_preview_failed %s", exc)
            await self._enter_recovery(str(exc))
            return None

    async def _check_motion_abort(self) -> bool:
        """Return False if document moved and transaction was failed."""
        frame = await self._safe_preview()
        if frame is None:
            return False
        if self.detector.classify(frame) == DocumentPresence.DOCUMENT_MOVING:
            logger.warning("event=document_moved_during_capture_phase state=%s", self.sm.state.value)
            await self._fail_transaction_and_wait("stability_timeout", "Document moved during capture")
            return False
        return True

    async def _idle_tick(self) -> None:
        if self.has_active_transaction:
            return
        if not self.disk.can_capture(self.config.storage.capture_reserve_mb):
            self._update_storage_health(self.disk.check())
            return
        frame = await self._safe_preview()
        if frame is None:
            return
        if self.detector.detect(frame):
            logger.info("event=document_detected")
            self.stability.reset()
            self._begin_transaction()
            self._transition(AppState.DOCUMENT_DETECTED)

    def _begin_transaction(self) -> ActiveCaptureTransaction:
        if self._transaction is not None and not self._transaction.is_terminal:
            raise RuntimeError("Cannot begin transaction: one already active")
        txn = ActiveCaptureTransaction(capture_id=ActiveCaptureTransaction.generate_id())
        txn.mark_document_detected()
        self._transaction = txn
        self._capture_path = None
        set_log_context(capture_id=txn.capture_id)
        logger.info("event=transaction_started capture_id=%s", txn.capture_id)
        return txn

    async def _document_detected_tick(self) -> None:
        if not self._claim_state_action():
            return
        self._transition(AppState.WAITING_FOR_STABILITY)

    async def _stability_tick(self) -> None:
        frame = await self._safe_preview()
        if frame is None:
            return
        presence = self.detector.classify(frame)
        if presence == DocumentPresence.DOCUMENT_MOVING:
            self.stability.reset()
            self.sm.touch_progress()
            return
        if self.stability.update(frame):
            logger.info("event=document_stable frames=%d", self.stability.stable_count)
            assert self._transaction is not None
            self._transaction.phase = TransactionPhase.STABILIZING
            self._transition(AppState.FOCUSING)
        else:
            self.sm.touch_progress()

    async def _focusing_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        if self._shutdown_event.is_set():
            return
        if not await self._check_motion_abort():
            return
        self._transaction.mark_focus_started()
        self._transaction.focus_attempts += 1
        min_sharp = self.config.quality.min_sharpness * 0.6
        try:
            result = await self.autofocus.run_autofocus(
                self.camera,
                sharpness_threshold=min_sharp,
                roi_fraction=self.config.detection.roi_center_fraction,
            )
            self._transaction.mark_focus_completed(result.sharpness, result.min_required_sharpness)
            self.sm.touch_progress()
            self._transition(AppState.FOCUS_VERIFICATION)
        except AutofocusError as exc:
            logger.error("event=autofocus_failed %s", exc)
            await self._fail_transaction_and_wait("focus_failed", str(exc))

    async def _focus_verification_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        if not await self._check_motion_abort():
            return
        min_sharp = self.config.quality.min_sharpness * 0.6
        self._transaction.focus_verify_attempts += 1
        try:
            result = await self.autofocus.verify_focus(
                self.camera,
                sharpness_threshold=min_sharp,
                roi_fraction=self.config.detection.roi_center_fraction,
            )
            self._transaction.mark_focus_completed(result.sharpness, result.min_required_sharpness)
            self.sm.touch_progress()
            self._transition(AppState.CAPTURING)
        except AutofocusError:
            if self._transaction.focus_verify_attempts < self.config.camera.af_verify_max_retries:
                self._transition(AppState.FOCUSING, reason="refocus")
            else:
                await self._fail_transaction_and_wait("focus_failed", "Focus verification failed")

    async def _capturing_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        if not await self._check_motion_abort():
            return
        if not self.disk.can_capture(self.config.storage.capture_reserve_mb):
            await self._fail_transaction_and_wait("storage_full", "Insufficient disk space")
            return
        output_path = self.temp_store.create_temp_path()
        self._transaction.mark_capture_started(str(output_path))
        try:
            self._capture_path = await self.capture_service.capture(self.camera, output_path)
            self.sm.touch_progress()
            self._transition(AppState.IMAGE_VALIDATION)
        except CaptureError as exc:
            logger.error("event=capture_failed %s", exc)
            self.temp_store.cleanup(output_path)
            await self._enter_recovery(str(exc))

    async def _validation_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        assert self._capture_path is not None
        try:
            result = self.validator.validate(self._capture_path)
            self._transaction.mark_validated(
                size=result.file_bytes,
                width=result.width,
                height=result.height,
                sha256=result.sha256,
                sharpness=result.sharpness,
            )
            logger.info(
                "event=image_validated sha256=%s sharpness=%.1f",
                result.sha256[:16],
                result.sharpness,
            )
            self.sm.touch_progress()
            self._transition(AppState.UPLOADING)
        except ImageValidationError as exc:
            logger.error("event=image_validation_failed %s", exc)
            self.temp_store.cleanup(self._capture_path)
            await self._fail_transaction_and_wait("image_invalid", str(exc))

    async def _uploading_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        assert self._capture_path is not None
        self._transaction.mark_upload_started()
        request = UploadRequest(
            capture_id=self._transaction.capture_id,
            device_id=self.config.device_id,
            image_path=str(self._capture_path),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        try:
            self._pending_response = await self.api.upload_document(request)
            self._transaction.mark_upload_completed(self._pending_response.http_status)
            self.sm.touch_progress()
            self._transition(AppState.WAITING_FOR_SERVER)
        except (ServerRejectedError, ServerError, ProtocolError) as exc:
            self.temp_store.cleanup(self._capture_path)
            key = "server_rejected" if isinstance(exc, ServerRejectedError) else "upload_failed"
            await self._fail_transaction_and_wait(key, str(exc))
        except NetworkError as exc:
            self.temp_store.cleanup(self._capture_path)
            await self._fail_transaction_and_wait("upload_failed", str(exc))

    async def _waiting_server_tick(self) -> None:
        if not self._claim_state_action():
            return
        self._transition(AppState.PROCESSING_RESPONSE)

    async def _processing_response_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        response = self._pending_response
        if response and response.success and response.outcome == UploadOutcome.ACCEPTED:
            self.health.record_success()
            self.health.server = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True, "reachable": True})
            self._transaction.mark_completed()
            self.temp_store.cleanup(self._capture_path)
            self._transition(AppState.SUCCESS)
        else:
            msg = response.message if response else "Upload failed"
            self.health.record_failure(msg)
            self.temp_store.cleanup(self._capture_path)
            failure_key = "upload_failed"
            if response and response.outcome == UploadOutcome.REJECTED:
                failure_key = "server_rejected"
            await self._fail_transaction_and_wait(failure_key, msg or "Upload failed")

    async def _success_tick(self) -> None:
        if not self._claim_state_action():
            return
        await asyncio.sleep(2.0)
        self._removal_count = 0
        self._transition(AppState.WAITING_FOR_REMOVAL)

    async def _removal_tick(self) -> None:
        frame = await self._safe_preview()
        if frame is None:
            return
        if self.detector.is_scene_empty(frame):
            self._removal_count += 1
        else:
            self._removal_count = 0
            self.sm.touch_progress()

        if self._removal_count >= self.config.detection.removal_frames_required:
            logger.info(
                "event=document_removed reason=scene_empty consecutive_frames=%d motion=%.2f",
                self._removal_count,
                self.detector.last_motion,
            )
            self._clear_transaction()
            self.detector.reset_baseline()
            self._transition(AppState.IDLE)

    async def _recovery_tick(self) -> None:
        if self._recovery_in_progress:
            return
        if time.monotonic() < self._next_recovery_at:
            return
        if not self._claim_state_action():
            return

        self._recovery_in_progress = True
        max_attempts = self.config.camera.recovery_max_attempts
        try:
            if self._recovery_attempts >= max_attempts:
                logger.error("Recovery exhausted after %d attempts", max_attempts)
                self.health.record_failure("recovery_exhausted")
                self.health.camera = ComponentStatus(level=HealthLevel.UNAVAILABLE, message="Camera recovery exhausted")
                self._transition(AppState.ERROR)
                raise RecoveryExhaustedError("Recovery attempts exhausted")

            self._recovery_attempts += 1
            logger.info("event=recovery_started attempt=%d/%d", self._recovery_attempts, max_attempts)
            self.health.record_recovery()
            self._abort_transaction("recovery")

            await asyncio.sleep(self.config.camera.recovery_delay_sec)
            await asyncio.to_thread(self.camera.restart)
            self.health.camera = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
            self._recovery_attempts = 0
            self._next_recovery_at = 0.0
            if self.disk.check().healthy:
                self._transition(AppState.IDLE)
            else:
                self._transition(AppState.ERROR, reason="storage_still_degraded")
        except CameraError as exc:
            logger.error("event=recovery_failed %s", exc)
            self.health.camera = ComponentStatus(level=HealthLevel.UNAVAILABLE, message=str(exc))
            self._next_recovery_at = time.monotonic() + self.config.camera.recovery_delay_sec
            self._state_action_claimed = None
        finally:
            self._recovery_in_progress = False

    async def _error_tick(self) -> None:
        storage = self.disk.check()
        self._update_storage_health(storage)
        if not storage.healthy or self._shutdown_event.is_set():
            return
        if self._error_backoff_claimed:
            return
        self._error_backoff_claimed = True
        try:
            await asyncio.sleep(10.0)
            if not self._shutdown_event.is_set():
                self._transition(AppState.RECOVERY, reason="auto_recovery_from_error")
        finally:
            self._error_backoff_claimed = False

    async def _handle_watchdog(self, alert) -> None:
        await self._enter_recovery(f"watchdog:{alert.reason}")

    async def _enter_recovery(self, reason: str) -> None:
        logger.warning(
            "event=enter_recovery reason=%s capture_id=%s",
            reason,
            getattr(self._transaction, "capture_id", None),
        )
        self._cleanup_transaction_files()
        if self._transaction and not self._transaction.is_terminal:
            self._transaction.mark_failed(reason)
        self._next_recovery_at = 0.0
        self._state_action_claimed = None
        self._transition(AppState.RECOVERY, reason=reason)

    def _abort_transaction(self, reason: str) -> None:
        if self._transaction and not self._transaction.is_terminal:
            self._transaction.mark_failed(reason)
        self._cleanup_transaction_files()

    async def _fail_transaction_and_wait(self, failure_key: str, detail: str) -> None:
        if self._transaction:
            self._transaction.mark_failed(detail)
        self.health.record_failure(detail)
        self._cleanup_transaction_files()
        self.status.update(
            AppState.WAITING_FOR_REMOVAL,
            capture_id=self._transaction.capture_id if self._transaction else None,
            failure_key=failure_key,
            last_error=detail,
            **self._health_status_kwargs(),
        )
        self._transition(AppState.WAITING_FOR_REMOVAL)

    def _cleanup_transaction_files(self) -> None:
        if self._capture_path:
            self.temp_store.cleanup(self._capture_path)
            self._capture_path = None

    def _clear_transaction(self) -> None:
        capture_id = self._transaction.capture_id if self._transaction else None
        logger.info("event=transaction_cleared capture_id=%s", capture_id)
        self._transaction = None
        self._capture_path = None
        self._pending_response = None
        set_log_context(capture_id=None)

    def _refresh_health(self) -> None:
        storage = self.disk.check()
        self._update_storage_health(storage)

    def _update_storage_health(self, storage_status) -> None:
        level = HealthLevel.HEALTHY
        if storage_status.health.value == "degraded":
            level = HealthLevel.DEGRADED
        elif storage_status.health.value == "unavailable":
            level = HealthLevel.UNAVAILABLE
        self.health.storage = ComponentStatus(
            level=level,
            message=storage_status.message,
            details=storage_status.to_dict(),
        )

    def _health_status_kwargs(self) -> dict:
        return {
            "camera": self.health.camera.to_dict(),
            "network": self.health.network.to_dict(),
            "server": self.health.server.to_dict(),
            "storage": self.health.storage.to_dict(),
            "application": self.health.application.to_dict(),
            "uptime_seconds": self.health.uptime_sec(),
        }

    def _transition(self, target: AppState, *, reason: str = "") -> None:
        if self._shutdown_event.is_set() and target not in {AppState.SHUTDOWN, AppState.RECOVERY}:
            target = AppState.SHUTDOWN
        prev = self.sm.state
        self.sm.transition(target, reason=reason)
        self._state_action_claimed = None
        if target != AppState.ERROR:
            self._error_backoff_claimed = False
        set_log_context(state=target.value)
        logger.info(
            "event=state_transition from=%s to=%s reason=%s capture_id=%s",
            prev.value,
            target.value,
            reason,
            self._transaction.capture_id if self._transaction else None,
        )
        ready = target == AppState.IDLE and self.health.overall_ready
        message_state = target
        if target == AppState.IDLE and not ready:
            message_state = AppState.ERROR if not self.health.storage.healthy else target

        self.status.update(
            message_state,
            capture_id=self._transaction.capture_id if self._transaction else None,
            last_success_at=self.health.last_success_at,
            last_error=self.health.last_error,
            **self._health_status_kwargs(),
        )

    async def _shutdown(self) -> None:
        self._transition(AppState.SHUTDOWN)
        self._cleanup_transaction_files()
        if self._transaction and not self._transaction.is_terminal:
            self._transaction.mark_failed("shutdown")
        self._clear_transaction()
        await self.api.stop()
        await asyncio.to_thread(self.camera.shutdown)
        logger.info("Shutdown complete")
