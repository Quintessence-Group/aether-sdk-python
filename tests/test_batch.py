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
    assert "/v1/documents/batch" in call_args[0][1]  # url


def test_batch_search(client):
    mock_resp = _ok_response()
    mock_resp.json.return_value = {
        "results": [
            {
                "query": "test",
                "results": [
                    {"doc_id": "a", "score": 90, "content_type": "text/plain"},
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
    assert "/v1/search/batch" in call_args[0][1]


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


def test_batch_insert_serializes_source(client):
    mock_resp = _ok_response()
    mock_resp.json.return_value = {
        "results": [
            {"doc_id": "b1", "cid": "c1", "chunks": 2, "vectors": 2, "version": 1,
             "source": "slack", "tags": ["a"]},
            {"doc_id": "b2", "cid": "c2", "chunks": 1, "vectors": 1, "version": 1},
        ],
    }

    with patch.object(client._client, "request", return_value=mock_resp) as mock_req:
        results = client.batch_insert([
            BatchInsertItem(filename="a.txt", content="hello", source="slack"),
            BatchInsertItem(filename="b.txt", content="world"),
        ])

    payload = mock_req.call_args[1]["json"]
    assert payload["documents"][0]["source"] == "slack"
    assert "source" not in payload["documents"][1]
    # Mapper round-trips the full record returned by the server.
    assert results[0].source == "slack"
    assert results[0].tags == ["a"]
    assert results[1].source is None
    assert results[1].tags == []


def test_batch_search_serializes_facet_filters_csv(client):
    mock_resp = _ok_response()
    mock_resp.json.return_value = {
        "results": [
            {"query": "filtered", "results": []},
            {"query": "plain", "results": []},
        ],
    }

    with patch.object(client._client, "request", return_value=mock_resp) as mock_req:
        client.batch_search([
            BatchSearchQuery(
                q="filtered",
                tags=["a", "b"],
                any_tags=["x", "y"],
                content_types=["text/plain", "application/pdf"],
                sources=["slack", "email"],
            ),
            BatchSearchQuery(q="plain"),
        ])

    queries = mock_req.call_args[1]["json"]["queries"]
    # The batch endpoint takes comma-joined CSV strings (same convention as tags).
    assert queries[0]["tags"] == "a,b"
    assert queries[0]["any_tags"] == "x,y"
    assert queries[0]["content_type"] == "text/plain,application/pdf"
    assert queries[0]["source"] == "slack,email"
    for key in ("tags", "any_tags", "content_type", "source"):
        assert key not in queries[1]


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
                thread_id="support-42",
                since="2026-06-01T00:00:00Z",
                until="2026-06-10T23:59:59Z",
                max_distance=0.3,
            ),
            BatchSearchQuery(q="recent", last_n_days=7),
            BatchSearchQuery(q="plain"),
        ])

    queries = mock_req.call_args[1]["json"]["queries"]
    assert queries[0]["entity_id"] == "user-123"
    assert queries[0]["thread_id"] == "support-42"
    assert queries[0]["since"] == "2026-06-01T00:00:00Z"
    assert queries[0]["until"] == "2026-06-10T23:59:59Z"
    assert queries[0]["max_distance"] == 0.3
    assert "last_n_days" not in queries[0]
    assert queries[1]["last_n_days"] == 7
    assert "since" not in queries[1]
    for key in ("entity_id", "thread_id", "since", "until", "last_n_days", "max_distance"):
        assert key not in queries[2]


def test_batch_search_rejects_invalid_thread_id_before_request(client):
    with patch.object(client._client, "request") as mock_req:
        with pytest.raises(ValueError, match="thread_id"):
            client.batch_search([BatchSearchQuery(q="test", thread_id="bad\x00thread")])

    mock_req.assert_not_called()


def test_batch_search_serializes_freshness(client):
    mock_resp = _ok_response()
    mock_resp.json.return_value = {
        "results": [
            {"query": "fresh", "results": []},
            {"query": "plain", "results": []},
        ],
    }

    with patch.object(client._client, "request", return_value=mock_resp) as mock_req:
        client.batch_search([
            BatchSearchQuery(
                q="fresh",
                recency_weight=0.2,
                half_life_days=30.0,
                freshness_weight=0.4,
                freshness_half_life_days=7.0,
            ),
            BatchSearchQuery(q="plain"),
        ])

    queries = mock_req.call_args[1]["json"]["queries"]
    assert queries[0]["recency_weight"] == 0.2
    assert queries[0]["half_life_days"] == 30.0
    assert queries[0]["freshness_weight"] == 0.4
    assert queries[0]["freshness_half_life_days"] == 7.0
    for key in ("recency_weight", "half_life_days", "freshness_weight",
                "freshness_half_life_days"):
        assert key not in queries[1]
