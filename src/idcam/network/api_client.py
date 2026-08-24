"""HTTP API client for document upload."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from idcam.config.schema import AppConfig
from idcam.exceptions import NetworkError, ServerError, ServerRejectedError
from idcam.logging.setup import get_logger
from idcam.network.protocol import CaptureTransaction, ServerResponse, UploadOutcome
from idcam.network.retry_policy import RetryPolicy

logger = get_logger("network.api_client")


class ApiClient:
    """Dedicated HTTP client for document upload."""

    def __init__(self, config: AppConfig, retry_policy: RetryPolicy | None = None) -> None:
        self.config = config
        self.retry_policy = retry_policy or RetryPolicy()
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        timeout = httpx.Timeout(
            connect=self.config.server.connect_timeout_sec,
            read=self.config.server.read_timeout_sec,
            write=self.config.server.read_timeout_sec,
            pool=self.config.server.connect_timeout_sec,
        )
        self._client = httpx.AsyncClient(
            timeout=timeout,
            verify=self.config.server.verify_tls,
            follow_redirects=False,
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def check_connectivity(self) -> bool:
        """Lightweight connectivity check (DNS + TCP). Does not upload."""
        if not self._client:
            await self.start()
        assert self._client is not None
        base = str(self.config.server.url).rstrip("/")
        try:
            response = await self._client.head(base, timeout=5.0)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def upload_document(self, transaction: CaptureTransaction) -> ServerResponse:
        if not self._client:
            await self.start()
        assert self._client is not None

        image_path = Path(transaction.image_path)
        if not image_path.exists():
            raise NetworkError(f"Image not found for upload: {image_path}")

        file_size = image_path.stat().st_size
        if file_size > self.config.server.max_upload_bytes:
            raise NetworkError(f"Image exceeds max upload size: {file_size}")

        url = str(self.config.server.url).rstrip("/") + self.config.server.path
        headers = {
            "X-Device-ID": transaction.device_id,
            "X-Capture-ID": transaction.capture_id,
            "Idempotency-Key": transaction.capture_id,
        }
        token = self.config.auth.resolved_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        attempt = 0
        while True:
            attempt += 1
            logger.info(
                "Uploading capture_id=%s attempt=%d size=%d url=%s",
                transaction.capture_id,
                attempt,
                file_size,
                url,
            )
            try:
                with image_path.open("rb") as image_file:
                    files = {"image": (image_path.name, image_file, "image/jpeg")}
                    response = await self._client.post(url, headers=headers, files=files)

                body = None
                if response.content:
                    try:
                        body = response.json()
                    except json.JSONDecodeError:
                        logger.warning("Malformed JSON response for capture_id=%s", transaction.capture_id)
                        if 200 <= response.status_code < 300:
                            body = {"success": True, "capture_id": transaction.capture_id}

                result = ServerResponse.from_http(response.status_code, body, transaction.capture_id)

                if 400 <= response.status_code < 500:
                    logger.error(
                        "Server rejected upload capture_id=%s status=%d",
                        transaction.capture_id,
                        response.status_code,
                    )
                    raise ServerRejectedError(result.message or "Server rejected request")

                if response.status_code >= 500:
                    logger.error(
                        "Server error capture_id=%s status=%d",
                        transaction.capture_id,
                        response.status_code,
                    )
                    raise ServerError(result.message or "Server error")

                logger.info(
                    "Upload successful capture_id=%s status=%d outcome=%s",
                    transaction.capture_id,
                    response.status_code,
                    result.outcome.value,
                )
                return result

            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                logger.warning("Upload connection error capture_id=%s: %s", transaction.capture_id, exc)
                if self.retry_policy.should_retry_connection(attempt):
                    import asyncio

                    await asyncio.sleep(self.retry_policy.retry_delay_sec)
                    continue
                return ServerResponse(
                    success=False,
                    capture_id=transaction.capture_id,
                    result=None,
                    message=str(exc),
                    http_status=0,
                    outcome=UploadOutcome.NEVER_SENT,
                )
            except httpx.HTTPError as exc:
                logger.error("Upload HTTP error capture_id=%s: %s", transaction.capture_id, exc)
                return ServerResponse(
                    success=False,
                    capture_id=transaction.capture_id,
                    result=None,
                    message=str(exc),
                    http_status=0,
                    outcome=UploadOutcome.SENT_UNKNOWN,
                )
