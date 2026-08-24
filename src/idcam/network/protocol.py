"""Server API protocol models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import secrets
import string


class UploadOutcome(str, Enum):
    NEVER_SENT = "never_sent"
    SENT_UNKNOWN = "sent_unknown"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    SERVER_ERROR = "server_error"


@dataclass(frozen=True)
class CaptureTransaction:
    capture_id: str
    device_id: str
    image_path: str
    created_at: str

    @staticmethod
    def generate_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        suffix = "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        return f"{ts}-{suffix}"


@dataclass
class ServerResponse:
    success: bool
    capture_id: str | None
    result: str | None
    message: str | None
    http_status: int
    outcome: UploadOutcome

    @classmethod
    def from_http(cls, status: int, body: dict | None, capture_id: str) -> "ServerResponse":
        if body is None:
            if 200 <= status < 300:
                return cls(
                    success=True,
                    capture_id=capture_id,
                    result=None,
                    message=None,
                    http_status=status,
                    outcome=UploadOutcome.ACCEPTED,
                )
            if 400 <= status < 500:
                return cls(
                    success=False,
                    capture_id=capture_id,
                    result=None,
                    message="Server rejected request",
                    http_status=status,
                    outcome=UploadOutcome.REJECTED,
                )
            return cls(
                success=False,
                capture_id=capture_id,
                result=None,
                message="Server error",
                http_status=status,
                outcome=UploadOutcome.SERVER_ERROR,
            )

        return cls(
            success=bool(body.get("success", 200 <= status < 300)),
            capture_id=body.get("capture_id", capture_id),
            result=body.get("result"),
            message=body.get("message") or body.get("error"),
            http_status=status,
            outcome=UploadOutcome.ACCEPTED if body.get("success") else UploadOutcome.REJECTED,
        )
