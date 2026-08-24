"""Retry policy and protocol tests."""

import pytest

from idcam.network.protocol import CaptureTransaction, ServerResponse, UploadOutcome
from idcam.network.retry_policy import RetryPolicy


class TestRetryPolicy:
    def test_connection_retry_allowed(self):
        policy = RetryPolicy(max_connection_retries=2)
        assert policy.should_retry_connection(1)
        assert not policy.should_retry_connection(2)

    def test_no_post_retry_after_send(self):
        policy = RetryPolicy()
        assert not policy.should_retry_post_after_send()


class TestCaptureTransaction:
    def test_generate_unique_ids(self):
        ids = {CaptureTransaction.generate_id() for _ in range(100)}
        assert len(ids) == 100


class TestServerResponse:
    def test_from_http_success(self):
        resp = ServerResponse.from_http(200, {"success": True, "capture_id": "abc"}, "abc")
        assert resp.success
        assert resp.outcome == UploadOutcome.ACCEPTED

    def test_from_http_4xx(self):
        resp = ServerResponse.from_http(400, {"success": False}, "abc")
        assert not resp.success
        assert resp.outcome == UploadOutcome.REJECTED

    def test_from_http_5xx(self):
        resp = ServerResponse.from_http(503, None, "abc")
        assert not resp.success
        assert resp.outcome == UploadOutcome.SERVER_ERROR
