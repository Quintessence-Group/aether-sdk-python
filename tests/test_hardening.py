"""Tests for production-hardening behavior: User-Agent, idempotency keys,
and insecure-URL enforcement."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from aether import AetherClient, DocumentPage
from aether._internal import USER_AGENT


def _ok_response(**kwargs):
    resp = MagicMock(spec=httpx.Response)
    resp.is_success = True
    resp.status_code = 200
    resp.json.return_value = {
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
    }
    for k, v in kwargs.items():
        setattr(resp, k, v)
    return resp


class TestUserAgent:
    def test_user_agent_on_default_headers(self):
        client = AetherClient(base_url="http://localhost:9000")
        assert client._client.headers["user-agent"] == USER_AGENT
        assert "aether-sdk-python/" in USER_AGENT


class TestIdempotencyKey:
    def test_post_gets_idempotency_key(self):
        client = AetherClient(base_url="http://localhost:9000", max_retries=0)
        with patch.object(client._client, "request", return_value=_ok_response()) as req:
            client.insert_text("hello")
        _, kwargs = req.call_args
        assert "Idempotency-Key" in kwargs["headers"]

    def test_idempotency_key_stable_across_retries(self):
        """A retried POST must reuse the same key so the server can dedupe."""
        client = AetherClient(base_url="http://localhost:9000", max_retries=2, retry_base_delay=0)
        retry_resp = _ok_response(is_success=False, status_code=503)
        retry_resp.headers = {}
        ok = _ok_response()
        with patch.object(client._client, "request", side_effect=[retry_resp, ok]) as req:
            client.insert_text("hello")
        keys = {c.kwargs["headers"]["Idempotency-Key"] for c in req.call_args_list}
        assert len(keys) == 1  # same key on every attempt

    def test_get_has_no_idempotency_key(self):
        client = AetherClient(base_url="http://localhost:9000", max_retries=0)
        with patch.object(client._client, "request", return_value=_ok_response()) as req:
            client.status()
        _, kwargs = req.call_args
        assert "Idempotency-Key" not in (kwargs.get("headers") or {})


class TestListPagination:
    def test_list_returns_documents_and_metadata(self):
        client = AetherClient(base_url="http://localhost:9000", max_retries=0)
        resp = _ok_response()
        resp.json.return_value = {
            "documents": [
                {"doc_id": "d1", "title": "A", "size_bytes": 10},
                {"doc_id": "d2", "title": "B", "size_bytes": 20},
            ],
            "count": 2,
            "total": 57,
            "has_more": True,
        }
        with patch.object(client._client, "request", return_value=resp):
            page = client.list(offset=0, limit=2)

        # Backward compatible: behaves like a list[DocumentRecord]
        assert isinstance(page, (list, DocumentPage))
        assert len(page) == 2
        assert [d.doc_id for d in page] == ["d1", "d2"]
        # New: pagination metadata
        assert page.total == 57
        assert page.has_more is True

    def test_metadata_defaults_when_absent(self):
        client = AetherClient(base_url="http://localhost:9000", max_retries=0)
        resp = _ok_response()
        resp.json.return_value = {"documents": [{"doc_id": "d1"}]}
        with patch.object(client._client, "request", return_value=resp):
            page = client.list()
        assert page.total == 1       # falls back to len(documents)
        assert page.has_more is False


class TestInsecureUrlEnforcement:
    def test_http_with_key_to_remote_host_raises(self):
        with pytest.raises(ValueError, match="insecure HTTP"):
            AetherClient(base_url="http://api.aetherdb.ai", api_key="secret")

    @pytest.mark.parametrize("url", [
        "http://localhost:9000",
        "http://127.0.0.1:9000",
        "https://api.aetherdb.ai",
    ])
    def test_allowed_configurations(self, url):
        AetherClient(base_url=url, api_key="secret")  # must not raise

    def test_http_without_key_allowed(self):
        AetherClient(base_url="http://api.aetherdb.ai")  # no key, no TLS needed
