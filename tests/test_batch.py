"""Tests for batch operations."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from aether import AetherClient, BatchInsertItem, BatchSearchQuery


@pytest.fixture
def client():
    return AetherClient(base_url="http://localhost:9000", api_key="test-key", max_retries=0)


def _ok_response(**kwargs):
    """Create a mock httpx.Response that passes retry checks."""
    resp = MagicMock(spec=httpx.Response)
    resp.is_success = True
    resp.status_code = 200
    for k, v in kwargs.items():
        setattr(resp, k, v)
    return resp


def test_batch_insert(client):
    mock_resp = _ok_response()
    mock_resp.json.return_value = {
        "results": [
            {"doc_id": "b1", "cid": "c1", "chunks": 2, "vectors": 2, "version": 1},
        ],
    }

    with patch.object(client._client, "request", return_value=mock_resp) as mock_req:
        results = client.batch_insert([BatchInsertItem(filename="a.txt", content="hello")])

    assert len(results) == 1
    assert results[0].doc_id == "b1"
    call_args = mock_req.call_args
    assert call_args[0][0] == "POST"  # method
    assert "/documents/batch" in call_args[0][1]  # url


def test_batch_search(client):
    mock_resp = _ok_response()
    mock_resp.json.return_value = {
        "results": [
            {
                "query": "test",
                "results": [
                    {"doc_id": "a", "distance": 0.1, "content_type": "text/plain"},
                ],
            },
        ],
    }

    with patch.object(client._client, "request", return_value=mock_resp) as mock_req:
        results = client.batch_search([BatchSearchQuery(q="test", k=5)])

    assert len(results) == 1
    assert results[0].query == "test"
    assert len(results[0].results) == 1
    call_args = mock_req.call_args
    assert call_args[0][0] == "POST"
    assert "/search/batch" in call_args[0][1]
