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


def test_batch_insert_tags_are_comma_joined_string(client):
    """The API batch deserializer expects tags as a comma-joined string, not a JSON array."""
    mock_resp = _ok_response()
    mock_resp.json.return_value = {
        "results": [
            {"doc_id": "b1", "cid": "c1", "chunks": 2, "vectors": 2, "version": 1},
        ],
    }

    with patch.object(client._client, "request", return_value=mock_resp) as mock_req:
        client.batch_insert(
            [BatchInsertItem(filename="a.txt", content="hello", tags=["alpha", "beta"])]
        )

    payload = mock_req.call_args.kwargs["json"]
    assert payload["documents"][0]["tags"] == "alpha,beta"


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


def test_batch_search_tags_are_comma_joined_string(client):
    """Batch search queries that carry tags must send them as a comma-joined string."""
    mock_resp = _ok_response()
    mock_resp.json.return_value = {
        "results": [
            {"query": "test", "results": []},
        ],
    }

    with patch.object(client._client, "request", return_value=mock_resp) as mock_req:
        client.batch_search([BatchSearchQuery(q="test", k=5, tags=["alpha", "beta"])])

    payload = mock_req.call_args.kwargs["json"]
    assert payload["queries"][0]["tags"] == "alpha,beta"


def test_batch_insert_serializes_entity_id(client):
    mock_resp = _ok_response()
    mock_resp.json.return_value = {
        "results": [
            {"doc_id": "b1", "cid": "c1", "chunks": 2, "vectors": 2, "version": 1,
             "entity_id": "user-123", "created_at": "2026-06-11T00:00:00Z"},
            {"doc_id": "b2", "cid": "c2", "chunks": 1, "vectors": 1, "version": 1},
        ],
    }

    with patch.object(client._client, "request", return_value=mock_resp) as mock_req:
        results = client.batch_insert([
            BatchInsertItem(filename="a.txt", content="hello", entity_id="user-123"),
            BatchInsertItem(filename="b.txt", content="world"),
        ])

    payload = mock_req.call_args[1]["json"]
    assert payload["documents"][0]["entity_id"] == "user-123"
    assert "entity_id" not in payload["documents"][1]
    # Mapper round-trips the full record returned by the server
    assert results[0].entity_id == "user-123"
    assert results[0].created_at == "2026-06-11T00:00:00Z"
    assert results[1].entity_id is None


def test_batch_search_serializes_filters(client):
    mock_resp = _ok_response()
    mock_resp.json.return_value = {
        "results": [
            {"query": "filtered", "results": []},
            {"query": "recent", "results": []},
            {"query": "plain", "results": []},
        ],
    }

    with patch.object(client._client, "request", return_value=mock_resp) as mock_req:
        client.batch_search([
            BatchSearchQuery(
                q="filtered",
                k=5,
                entity_id="user-123",
                since="2026-06-01T00:00:00Z",
                until="2026-06-10T23:59:59Z",
                max_distance=0.3,
            ),
            BatchSearchQuery(q="recent", last_n_days=7),
            BatchSearchQuery(q="plain"),
        ])

    queries = mock_req.call_args[1]["json"]["queries"]
    assert queries[0]["entity_id"] == "user-123"
    assert queries[0]["since"] == "2026-06-01T00:00:00Z"
    assert queries[0]["until"] == "2026-06-10T23:59:59Z"
    assert queries[0]["max_distance"] == 0.3
    assert "last_n_days" not in queries[0]
    assert queries[1]["last_n_days"] == 7
    assert "since" not in queries[1]
    for key in ("entity_id", "since", "until", "last_n_days", "max_distance"):
        assert key not in queries[2]
