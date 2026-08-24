"""User and admin status publishing."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from idcam.app.state_machine import AppState
from idcam.ui.messages import FAILURE_MESSAGES, USER_MESSAGES


@dataclass
class StatusSnapshot:
    state: str
    message: str
    capture_id: str | None = None
    detail_message: str | None = None
    camera_healthy: bool = False
    network_healthy: bool = False
    server_healthy: bool = False
    last_success_at: float | None = None
    last_error: str | None = None
    version: str = "1.0.0"
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class StatusPublisher:
    """Publishes status to a JSON file for UI consumption."""

    def __init__(self, status_file: Path, version: str = "1.0.0") -> None:
        self.status_file = status_file
        self.version = version
        self._snapshot = StatusSnapshot(state=AppState.BOOT.value, message=USER_MESSAGES[AppState.BOOT])
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def snapshot(self) -> StatusSnapshot:
        return self._snapshot

    def update(
        self,
        state: AppState,
        *,
        capture_id: str | None = None,
        detail: str | None = None,
        camera_healthy: bool | None = None,
        network_healthy: bool | None = None,
        server_healthy: bool | None = None,
        last_success_at: float | None = None,
        last_error: str | None = None,
        failure_key: str | None = None,
    ) -> None:
        message = USER_MESSAGES.get(state, state.value)
        if failure_key and failure_key in FAILURE_MESSAGES:
            detail = FAILURE_MESSAGES[failure_key]

        snap = StatusSnapshot(
            state=state.value,
            message=message,
            capture_id=capture_id,
            detail_message=detail,
            camera_healthy=camera_healthy if camera_healthy is not None else self._snapshot.camera_healthy,
            network_healthy=network_healthy if network_healthy is not None else self._snapshot.network_healthy,
            server_healthy=server_healthy if server_healthy is not None else self._snapshot.server_healthy,
            last_success_at=last_success_at if last_success_at is not None else self._snapshot.last_success_at,
            last_error=last_error,
            version=self.version,
        )
        self._snapshot = snap
        self._write()

    def _write(self) -> None:
        tmp = self.status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._snapshot.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.status_file)

    def get_admin_info(self) -> dict:
        return self._snapshot.to_dict()
