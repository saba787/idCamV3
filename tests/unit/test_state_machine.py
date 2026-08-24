"""State machine unit tests."""

import time

import pytest

from idcam.app.state_machine import AppState, StateMachine
from idcam.exceptions import StateMachineError


class TestStateMachine:
    def test_initial_state(self):
        sm = StateMachine()
        assert sm.state == AppState.BOOT

    def test_valid_transition(self):
        sm = StateMachine()
        sm.transition(AppState.INITIALIZING)
        assert sm.state == AppState.INITIALIZING

    def test_invalid_transition_raises(self):
        sm = StateMachine()
        with pytest.raises(StateMachineError):
            sm.transition(AppState.CAPTURING)

    def test_idle_to_document_detected(self):
        sm = StateMachine(state=AppState.IDLE)
        sm.transition(AppState.DOCUMENT_DETECTED)
        assert sm.in_transaction()

    def test_full_happy_path_transitions(self):
        sm = StateMachine(state=AppState.IDLE)
        for target in [
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
            AppState.WAITING_FOR_REMOVAL,
            AppState.IDLE,
        ]:
            sm.transition(target)

    def test_recovery_from_focusing(self):
        sm = StateMachine(state=AppState.FOCUSING)
        sm.transition(AppState.RECOVERY)
        assert sm.state == AppState.RECOVERY

    def test_timeout_detection(self):
        sm = StateMachine(state=AppState.DOCUMENT_DETECTED)
        sm.entered_at = time.monotonic() - 100
        assert sm.is_timed_out()

    def test_no_timeout_for_idle(self):
        sm = StateMachine(state=AppState.IDLE)
        sm.entered_at = time.monotonic() - 10000
        assert not sm.is_timed_out()

    def test_in_transaction_states(self):
        sm = StateMachine(state=AppState.UPLOADING)
        assert sm.in_transaction()
        sm = StateMachine(state=AppState.IDLE)
        assert not sm.in_transaction()
