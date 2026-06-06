"""Tests for retry logic in AetherClient."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from aether import AetherClient
from aether.errors import AetherApiError, AetherNetworkError


def _ok_response():
    resp = MagicMock(spec=httpx.Response)
    resp.is_success = True
    resp.status_code = 200
    return resp


def _error_response(status, reason="Server Error", headers=None):
    resp = MagicMock(spec=httpx.Response)
    resp.is_success = False
    resp.status_code = status
    resp.reason_phrase = reason
    resp.headers = httpx.Headers(headers or {})
    resp.json.side_effect = Exception("no json")
    return resp


@pytest.fixture
def client():
    return AetherClient(
        base_url="http://localhost:9000",
        api_key="test-key",
        max_retries=2,
        retry_base_delay=0.01,
    )


class TestRetryOnServerError:
    def test_retries_on_502_then_succeeds(self, client):
        fail = _error_response(502)
        ok = _ok_response()

        with patch.object(client._client, "request", side_effect=[fail, ok]) as mock:
            resp = client._request_with_retry("GET", "/status")

        assert resp.status_code == 200
        assert mock.call_count == 2

    def test_retries_on_503_then_succeeds(self, client):
        fail = _error_response(503, "Service Unavailable")
        ok = _ok_response()

        with patch.object(client._client, "request", side_effect=[fail, ok]) as mock:
            resp = client._request_with_retry("GET", "/health")

        assert resp.status_code == 200
        assert mock.call_count == 2

    def test_exhausts_retries_then_raises(self, client):
        fail = _error_response(502)

        with patch.object(client._client, "request", return_value=fail):
            with pytest.raises(AetherApiError) as exc_info:
                client._request_with_retry("GET", "/status")

        assert exc_info.value.status_code == 502


class TestNoRetryOnClientError:
    def test_404_raises_immediately(self, client):
        resp = _error_response(404, "Not Found")

        with patch.object(client._client, "request", return_value=resp) as mock:
            with pytest.raises(AetherApiError) as exc_info:
                client._request_with_retry("GET", "/documents/missing")

        assert mock.call_count == 1
        assert exc_info.value.status_code == 404

    def test_401_raises_immediately(self, client):
        resp = _error_response(401, "Unauthorized")

        with patch.object(client._client, "request", return_value=resp) as mock:
            with pytest.raises(AetherApiError):
                client._request_with_retry("GET", "/documents")

        assert mock.call_count == 1


class TestRetryOnNetworkError:
    def test_retries_on_timeout_then_succeeds(self, client):
        ok = _ok_response()

        with patch.object(
            client._client, "request",
            side_effect=[httpx.TimeoutException("timed out"), ok],
        ) as mock:
            resp = client._request_with_retry("POST", "/documents")

        assert resp.status_code == 200
        assert mock.call_count == 2

    def test_retries_on_connection_error_then_succeeds(self, client):
        ok = _ok_response()

        with patch.object(
            client._client, "request",
            side_effect=[httpx.ConnectError("refused"), ok],
        ) as mock:
            resp = client._request_with_retry("GET", "/status")

        assert resp.status_code == 200
        assert mock.call_count == 2

    def test_network_error_exhausts_retries(self, client):
        with patch.object(
            client._client, "request",
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(AetherNetworkError):
                client._request_with_retry("GET", "/status")


class TestRetryOn429:
    def test_retries_on_rate_limit(self, client):
        rate_resp = _error_response(429, "Too Many Requests", {"retry-after": "0.01"})
        ok = _ok_response()

        with patch.object(client._client, "request", side_effect=[rate_resp, ok]) as mock:
            resp = client._request_with_retry("GET", "/search")

        assert resp.status_code == 200
        assert mock.call_count == 2


class TestNoRetryConfig:
    def test_zero_retries_raises_immediately(self):
        c = AetherClient(
            base_url="http://localhost:9000",
            api_key="test",
            max_retries=0,
        )
        fail = _error_response(502)

        with patch.object(c._client, "request", return_value=fail) as mock:
            with pytest.raises(AetherApiError):
                c._request_with_retry("GET", "/status")

        assert mock.call_count == 1
