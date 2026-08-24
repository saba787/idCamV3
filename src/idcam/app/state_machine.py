"""Application state machine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from idcam.exceptions import StateMachineError


class AppState(str, Enum):
    BOOT = "BOOT"
    INITIALIZING = "INITIALIZING"
    CAMERA_INITIALIZING = "CAMERA_INITIALIZING"
    CAMERA_READY = "CAMERA_READY"
    IDLE = "IDLE"
    DOCUMENT_DETECTED = "DOCUMENT_DETECTED"
    WAITING_FOR_STABILITY = "WAITING_FOR_STABILITY"
    FOCUSING = "FOCUSING"
    FOCUS_VERIFICATION = "FOCUS_VERIFICATION"
    CAPTURING = "CAPTURING"
    IMAGE_VALIDATION = "IMAGE_VALIDATION"
    UPLOADING = "UPLOADING"
    WAITING_FOR_SERVER = "WAITING_FOR_SERVER"
    PROCESSING_RESPONSE = "PROCESSING_RESPONSE"
    SUCCESS = "SUCCESS"
    WAITING_FOR_REMOVAL = "WAITING_FOR_REMOVAL"
    RECOVERY = "RECOVERY"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


# States with bounded timeouts (seconds). None = no timeout.
STATE_TIMEOUTS: dict[AppState, float | None] = {
    AppState.BOOT: 10.0,
    AppState.INITIALIZING: 60.0,
    AppState.CAMERA_INITIALIZING: 120.0,
    AppState.CAMERA_READY: None,
    AppState.IDLE: None,
    AppState.DOCUMENT_DETECTED: 30.0,
    AppState.WAITING_FOR_STABILITY: 15.0,
    AppState.FOCUSING: 30.0,
    AppState.FOCUS_VERIFICATION: 10.0,
    AppState.CAPTURING: 60.0,
    AppState.IMAGE_VALIDATION: 10.0,
    AppState.UPLOADING: None,  # configured per-request
    AppState.WAITING_FOR_SERVER: None,
    AppState.PROCESSING_RESPONSE: 10.0,
    AppState.SUCCESS: 5.0,
    AppState.WAITING_FOR_REMOVAL: 120.0,
    AppState.RECOVERY: 180.0,
    AppState.ERROR: None,
    AppState.SHUTDOWN: 30.0,
}

ALLOWED_TRANSITIONS: dict[AppState, set[AppState]] = {
    AppState.BOOT: {AppState.INITIALIZING, AppState.SHUTDOWN},
    AppState.INITIALIZING: {AppState.CAMERA_INITIALIZING, AppState.ERROR, AppState.SHUTDOWN},
    AppState.CAMERA_INITIALIZING: {AppState.CAMERA_READY, AppState.RECOVERY, AppState.ERROR, AppState.SHUTDOWN},
    AppState.CAMERA_READY: {AppState.IDLE, AppState.RECOVERY, AppState.ERROR, AppState.SHUTDOWN},
    AppState.IDLE: {AppState.DOCUMENT_DETECTED, AppState.RECOVERY, AppState.ERROR, AppState.SHUTDOWN},
    AppState.DOCUMENT_DETECTED: {AppState.WAITING_FOR_STABILITY, AppState.IDLE, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.WAITING_FOR_STABILITY: {AppState.FOCUSING, AppState.IDLE, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.FOCUSING: {AppState.FOCUS_VERIFICATION, AppState.RECOVERY, AppState.ERROR, AppState.SHUTDOWN},
    AppState.FOCUS_VERIFICATION: {AppState.CAPTURING, AppState.FOCUSING, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.CAPTURING: {AppState.IMAGE_VALIDATION, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.IMAGE_VALIDATION: {AppState.UPLOADING, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.UPLOADING: {AppState.WAITING_FOR_SERVER, AppState.WAITING_FOR_REMOVAL, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.WAITING_FOR_SERVER: {AppState.PROCESSING_RESPONSE, AppState.WAITING_FOR_REMOVAL, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.PROCESSING_RESPONSE: {AppState.SUCCESS, AppState.WAITING_FOR_REMOVAL, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.SUCCESS: {AppState.WAITING_FOR_REMOVAL, AppState.SHUTDOWN},
    AppState.WAITING_FOR_REMOVAL: {AppState.IDLE, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.RECOVERY: {AppState.CAMERA_INITIALIZING, AppState.CAMERA_READY, AppState.IDLE, AppState.ERROR, AppState.SHUTDOWN},
    AppState.ERROR: {AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.SHUTDOWN: set(),
}


@dataclass
class StateMachine:
    """Explicit deterministic state machine."""

    state: AppState = AppState.BOOT
    entered_at: float = field(default_factory=time.monotonic)
    transition_count: int = 0

    def can_transition(self, target: AppState) -> bool:
        return target in ALLOWED_TRANSITIONS.get(self.state, set())

    def transition(self, target: AppState, *, reason: str = "") -> AppState:
        if not self.can_transition(target):
            raise StateMachineError(
                f"Invalid transition {self.state.value} -> {target.value}" + (f": {reason}" if reason else "")
            )
        self.state = target
        self.entered_at = time.monotonic()
        self.transition_count += 1
        return self.state

    def elapsed(self) -> float:
        return time.monotonic() - self.entered_at

    def is_timed_out(self, override_timeout: float | None = None) -> bool:
        timeout = override_timeout if override_timeout is not None else STATE_TIMEOUTS.get(self.state)
        if timeout is None:
            return False
        return self.elapsed() > timeout

    def in_transaction(self) -> bool:
        """True when a capture transaction is active."""
        transactional = {
            AppState.DOCUMENT_DETECTED,
            AppState.WAITING_FOR_STABILITY,
            AppState.FOCUSING,
            AppState.FOCUS_VERIFICATION,
            AppState.CAPTURING,
            AppState.IMAGE_VALIDATION,
            AppState.UPLOADING,
            AppState.WAITING_FOR_SERVER,
            AppState.PROCESSING_RESPONSE,
            AppState.SUCCESS,
        }
        return self.state in transactional
