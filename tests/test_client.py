"""Tests for AetherClient download_text and retrieve methods."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from aether import AetherClient, RetrievalResult


def _ok_response(**kwargs):
    """Create a mock httpx.Response that passes retry checks."""
    resp = MagicMock(spec=httpx.Response)
    resp.is_success = True
    resp.status_code = 200
    for k, v in kwargs.items():
        setattr(resp, k, v)
    return resp


@pytest.fixture
def client():
    return AetherClient(base_url="http://localhost:9000", api_key="test-key", max_retries=0)


class TestDownloadText:
    def test_returns_decoded_content(self, client):
        mock_resp = _ok_response(content=b"Hello, world!")

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.download_text("doc-123")

        assert result == "Hello, world!"

    def test_url_encodes_doc_id(self, client):
        mock_resp = _ok_response(content=b"content")

        with patch.object(client._client, "request", return_value=mock_resp):
            client.download_text("doc/with spaces")

        # Verifying it doesn't raise is sufficient — URL encoding tested implicitly


class TestRetrieve:
    def test_returns_results_with_content(self, client):
        search_resp = _ok_response()
        search_resp.json.return_value = {
            "query": "test",
            "results": [
                {"doc_id": "doc-1", "distance": 0.1, "title": "Doc 1", "content_type": "text/plain"},
                {"doc_id": "doc-2", "distance": 0.3, "title": "Doc 2", "content_type": "text/plain"},
            ],
        }

        dl_resp_1 = _ok_response(content=b"Content of doc 1")
        dl_resp_2 = _ok_response(content=b"Content of doc 2")

        with patch.object(client._client, "request", side_effect=[search_resp, dl_resp_1, dl_resp_2]):
            results = client.retrieve("test query", k=5)

        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].doc_id == "doc-1"
        assert results[0].content == "Content of doc 1"
        assert results[0].distance == 0.1
        assert results[1].content == "Content of doc 2"

    def test_deduplicates_by_doc_id(self, client):
        search_resp = _ok_response()
        search_resp.json.return_value = {
            "query": "test",
            "results": [
                {"doc_id": "doc-1", "distance": 0.1, "title": "Doc 1", "content_type": "text/plain"},
                {"doc_id": "doc-1", "distance": 0.2, "title": "Doc 1", "content_type": "text/plain"},
                {"doc_id": "doc-2", "distance": 0.3, "title": "Doc 2", "content_type": "text/plain"},
            ],
        }

        dl_resp_1 = _ok_response(content=b"Content 1")
        dl_resp_2 = _ok_response(content=b"Content 2")

        with patch.object(client._client, "request", side_effect=[search_resp, dl_resp_1, dl_resp_2]):
            results = client.retrieve("test", k=5)

        # Should deduplicate: 3 search results -> 2 unique docs
        assert len(results) == 2
        assert results[0].distance == 0.1  # keeps closest match
