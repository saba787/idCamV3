"""Stuck-state watchdog."""

from __future__ import annotations

import time

from idcam.app.state_machine import AppState, StateMachine
from idcam.config.schema import WatchdogConfig
from idcam.logging.setup import get_logger

logger = get_logger("system.watchdog")

# States that must never block indefinitely
WATCHED_STATES = {
    AppState.FOCUSING,
    AppState.FOCUS_VERIFICATION,
    AppState.CAPTURING,
    AppState.IMAGE_VALIDATION,
    AppState.UPLOADING,
    AppState.WAITING_FOR_SERVER,
    AppState.PROCESSING_RESPONSE,
    AppState.RECOVERY,
    AppState.CAMERA_INITIALIZING,
}


class Watchdog:
    """Detects stuck states and signals recovery."""

    def __init__(self, config: WatchdogConfig) -> None:
        self.config = config
        self._last_check = time.monotonic()

    def check(self, sm: StateMachine, *, upload_timeout: float | None = None) -> bool:
        """
        Returns True if state appears stuck and recovery should be triggered.
        """
        if sm.state not in WATCHED_STATES:
            return False

        override = upload_timeout if sm.state in {
            AppState.UPLOADING,
            AppState.WAITING_FOR_SERVER,
        } else None

        timeout = override if override is not None else self.config.state_timeout_sec

        if sm.is_timed_out(timeout):
            logger.error(
                "Watchdog: state %s stuck for %.1fs (timeout=%.1fs)",
                sm.state.value,
                sm.elapsed(),
                timeout,
            )
            return True
        return False
