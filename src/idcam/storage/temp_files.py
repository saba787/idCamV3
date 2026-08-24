"""Temporary file management with atomic writes."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path


class TempFileStore:
    """Manages temporary capture files with atomic rename."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_temp_path(self, suffix: str = ".jpg") -> Path:
        name = f"capture-{uuid.uuid4().hex}{suffix}"
        return self.base_dir / name

    def atomic_write_finalize(self, temp_path: Path, final_path: Path) -> Path:
        """Atomically move temp file to final path."""
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_path, final_path)
        return final_path

    def cleanup(self, path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass

    def cleanup_stale(self, max_age_sec: float = 3600) -> int:
        """Remove temp files older than max_age_sec. Returns count removed."""
        import time

        removed = 0
        now = time.time()
        for entry in self.base_dir.glob("capture-*"):
            try:
                if now - entry.stat().st_mtime > max_age_sec:
                    entry.unlink()
                    removed += 1
            except OSError:
                continue
        return removed

    def ensure_space(self, required_bytes: int) -> None:
        """Raise OSError if insufficient disk space."""
        stat = os.statvfs(self.base_dir)
        free = stat.f_bavail * stat.f_frsize
        if free < required_bytes:
            raise OSError(f"Insufficient disk space: {free} bytes free, need {required_bytes}")
