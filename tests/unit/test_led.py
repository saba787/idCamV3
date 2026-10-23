"""WS2812B status LED indicator."""

import time

from idcam.app.state_machine import AppState
from idcam.config.schema import LedConfig
from idcam.ui import led as led_mod
from idcam.ui.led import (
    DONE,
    FAULT,
    GREEN,
    OFF,
    READY,
    RED,
    STATE_PATTERNS,
    WHITE,
    LedIndicator,
    LedPattern,
)


class FakeBackend:
    def __init__(self, fail_on_show: bool = False) -> None:
        self.shown: list[tuple[int, int, int]] = []
        self.closed = False
        self.fail_on_show = fail_on_show

    def show(self, color):
        if self.fail_on_show:
            raise OSError("pio unavailable")
        self.shown.append(color)

    def close(self):
        self.closed = True


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_every_state_has_a_pattern():
    assert set(STATE_PATTERNS) == set(AppState)
    assert STATE_PATTERNS[AppState.SHUTDOWN].color == OFF


def test_disabled_by_default_never_touches_hardware(monkeypatch):
    def boom(_config):
        raise AssertionError("backend must not be created when disabled")

    monkeypatch.setattr(led_mod, "Pi5NeopixelBackend", boom)
    indicator = LedIndicator(LedConfig())
    indicator.start()
    indicator.set_state(AppState.IDLE)
    indicator.stop()
    assert not indicator.active


def test_missing_libraries_disable_indicator_without_raising(monkeypatch):
    def missing(_config):
        raise ModuleNotFoundError("No module named 'board'")

    monkeypatch.setattr(led_mod, "Pi5NeopixelBackend", missing)
    indicator = LedIndicator(LedConfig(enabled=True))
    indicator.start()
    assert not indicator.active
    indicator.set_state(AppState.ERROR)
    indicator.stop()


def test_state_changes_reach_the_strip_and_stop_turns_it_off():
    backend = FakeBackend()
    indicator = LedIndicator(LedConfig(enabled=True), backend=backend)
    indicator.start()
    try:
        indicator.set_state(AppState.IDLE)
        assert _wait_for(lambda: backend.shown and backend.shown[-1] == WHITE)
        indicator.set_state(AppState.WAITING_FOR_REMOVAL)
        assert _wait_for(lambda: backend.shown[-1] == GREEN)
    finally:
        indicator.stop()
    assert backend.closed
    assert not indicator.active


def test_write_failure_disables_indicator():
    backend = FakeBackend(fail_on_show=True)
    indicator = LedIndicator(LedConfig(enabled=True), backend=backend)
    indicator.start()
    assert _wait_for(lambda: not indicator.active)
    indicator.set_state(AppState.IDLE)
    indicator.stop()


def test_failure_flash_overrides_then_returns_to_state():
    indicator = LedIndicator(LedConfig(enabled=True, failure_flash_sec=0.5), backend=FakeBackend())
    indicator.set_state(AppState.IDLE)
    now = time.monotonic()
    indicator.flash_failure()
    assert indicator.current_color(now) == RED
    assert indicator.current_color(now + 1.0) == WHITE


def test_patterns():
    assert READY.color_at(123.4) == WHITE
    assert DONE.color_at(0.0) == GREEN
    assert FAULT.color_at(0.1) == RED
    assert FAULT.color_at(0.6) == OFF
    pulse = LedPattern((0, 0, 200), "pulse", 2.0)
    assert pulse.color_at(1.0) == (0, 0, 200)
    assert pulse.color_at(0.0)[2] < 50
