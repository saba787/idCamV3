"""Configuration schema and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ServerConfig(BaseModel):
    url: HttpUrl
    path: str = "/api/document"
    connect_timeout_sec: float = Field(10.0, gt=0, le=120)
    read_timeout_sec: float = Field(60.0, gt=0, le=600)
    total_timeout_sec: float = Field(90.0, gt=0, le=900)
    verify_tls: bool = True
    max_upload_bytes: int = Field(50_000_000, gt=0)

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        if not value.startswith("/"):
            return f"/{value}"
        return value


class AuthConfig(BaseModel):
    type: Literal["bearer", "none"] = "none"
    token: str = ""

    def resolved_token(self) -> str | None:
        if self.type == "none" or not self.token:
            return None
        return self.token


class CameraConfig(BaseModel):
    preview_width: int = Field(1920, ge=320, le=3840)
    preview_height: int = Field(1080, ge=240, le=2160)
    capture_width: int = Field(9152, ge=640)
    capture_height: int = Field(6944, ge=480)
    jpeg_quality: int = Field(95, ge=50, le=100)
    preview_fps: float = Field(5.0, gt=0, le=30)
    af_max_retries: int = Field(3, ge=1, le=10)
    af_timeout_sec: float = Field(30.0, gt=0, le=120)
    capture_timeout_sec: float = Field(60.0, gt=0, le=300)
    recovery_max_attempts: int = Field(3, ge=1, le=10)
    recovery_delay_sec: float = Field(5.0, gt=0, le=60)


class DetectionConfig(BaseModel):
    roi_center_fraction: float = Field(0.7, gt=0.1, le=1.0)
    motion_threshold: float = Field(8.0, gt=0)
    edge_density_threshold: float = Field(0.02, gt=0, le=1.0)
    stable_frames_required: int = Field(8, ge=2, le=60)
    removal_frames_required: int = Field(10, ge=2, le=120)
    baseline_alpha: float = Field(0.05, gt=0, le=1.0)


class QualityConfig(BaseModel):
    min_sharpness: float = Field(80.0, gt=0)
    min_file_bytes: int = Field(100_000, gt=0)
    max_file_bytes: int = Field(50_000_000, gt=0)
    min_brightness: float = Field(30.0, ge=0, le=255)
    max_brightness: float = Field(225.0, ge=0, le=255)
    min_width: int = Field(640, gt=0)
    min_height: int = Field(480, gt=0)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = Path("/var/log/idcam")
    max_bytes: int = Field(10_485_760, gt=0)
    backup_count: int = Field(5, ge=1, le=50)
    json_format: bool = True


class UIConfig(BaseModel):
    status_file: Path = Path("/run/idcam/status.json")
    admin_socket: Path = Path("/run/idcam/admin.sock")
    refresh_interval_sec: float = Field(0.5, gt=0, le=5)


class WatchdogConfig(BaseModel):
    check_interval_sec: float = Field(1.0, gt=0, le=10)
    state_timeout_sec: float = Field(300.0, gt=0, le=3600)


class AppConfig(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=128)
    temp_dir: Path = Path("/var/lib/idcam/tmp")
    server: ServerConfig
    auth: AuthConfig = AuthConfig()
    camera: CameraConfig = CameraConfig()
    detection: DetectionConfig = DetectionConfig()
    quality: QualityConfig = QualityConfig()
    logging: LoggingConfig = LoggingConfig()
    ui: UIConfig = UIConfig()
    watchdog: WatchdogConfig = WatchdogConfig()

    @field_validator("device_id")
    @classmethod
    def strip_device_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("device_id cannot be empty")
        return stripped
