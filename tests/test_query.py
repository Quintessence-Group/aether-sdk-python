"""Tests for the structured analytical query surface (client.query)."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from aether import AetherApiError, AetherClient, AggregateResult, DocumentPage


@pytest.fixture
def client():
    return AetherClient(base_url="http://localhost:9000", api_key="test-key", max_retries=0)


def _ok(payload):
    resp = MagicMock(spec=httpx.Response)
    resp.is_success = True
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


def test_query_mode_a_returns_document_page(client):
    resp = _ok(
        {
            "documents": [
                {"doc_id": "d1", "title": "t", "content_type": "text/plain",
                 "metadata": {"status": "paid"}}
            ],
            "total": 1,
            "has_more": False,
        }
    )
    with patch.object(client._client, "request", return_value=resp) as mock_req:
        page = client.query(
            filter={"field": "status", "op": "eq", "value": "paid"},
            sort=[{"by": "created_at", "dir": "desc"}],
            limit=10,
        )

    assert isinstance(page, DocumentPage)
    assert [d.doc_id for d in page] == ["d1"]
    assert page.total == 1 and page.has_more is False

    method, url = mock_req.call_args[0][0], mock_req.call_args[0][1]
    assert method == "POST" and "/v1/query" in url
    body = mock_req.call_args[1]["json"]
    assert body["filter"] == {"field": "status", "op": "eq", "value": "paid"}
    assert body["sort"] == [{"by": "created_at", "dir": "desc"}]
    assert body["limit"] == 10
    assert "aggregate" not in body and "group_by" not in body


def test_query_mode_b_returns_aggregate_result(client):
    resp = _ok(
        {
            "groups": [
                {"keys": {"status": "paid"}, "aggregates": {"count": 2, "total": 350.0}},
                {"keys": {"status": "open"}, "aggregates": {"count": 1, "total": 9.0}},
            ],
            "total_groups": 2,
            "scanned": 3,
        }
    )
    with patch.object(client._client, "request", return_value=resp) as mock_req:
        result = client.query(
            group_by=["status"],
            aggregate=[{"op": "count"}, {"op": "sum", "field": "amount", "as": "total"}],
            sort=[{"by": "total", "dir": "desc"}],
        )

    assert isinstance(result, AggregateResult)
    assert result.total_groups == 2 and result.scanned == 3
    assert result.groups[0].keys == {"status": "paid"}
    assert result.groups[0].aggregates["total"] == 350.0

    body = mock_req.call_args[1]["json"]
    assert body["group_by"] == ["status"]
    assert body["aggregate"][1] == {"op": "sum", "field": "amount", "as": "total"}


def test_query_scoped_handle_carries_partition(client):
    resp = _ok({"documents": [], "total": 0, "has_more": False})
    with patch.object(client._client, "request", return_value=resp) as mock_req:
        client.partition("acct-42").query(filter={"status": "paid"})
    assert mock_req.call_args[1]["json"]["partition"] == "acct-42"


def test_query_400_guardrail_raises_typed_error(client):
    err = MagicMock(spec=httpx.Response)
    err.is_success = False
    err.status_code = 400
    err.reason_phrase = "Bad Request"
    err.headers = {}
    err.json.return_value = {"error": "aggregation would produce more than 10000 groups"}
    with patch.object(client._client, "request", return_value=err):
        with pytest.raises(AetherApiError) as excinfo:
            client.query(group_by=["x"], aggregate=[{"op": "count"}])
    assert excinfo.value.status_code == 400
