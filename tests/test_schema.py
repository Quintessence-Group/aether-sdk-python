"""Tests for the field-schema facade (client.schema)."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from aether import AetherClient, FieldSchema


@pytest.fixture
def client():
    return AetherClient(base_url="http://localhost:9000", api_key="test-key", max_retries=0)


def _ok(payload):
    resp = MagicMock(spec=httpx.Response)
    resp.is_success = True
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


def test_declare_fields(client):
    resp = _ok(
        {
            "fields": [
                {
                    "name": "amount",
                    "type": "float",
                    "source": {"metadata": "amount"},
                    "coverage": 3,
                    "mismatch_count": 0,
                    "backfill": "complete",
                }
            ],
            "count": 1,
        }
    )
    with patch.object(client._client, "request", return_value=resp) as mock_req:
        fields = client.schema.declare_fields(
            [{"name": "amount", "type": "float", "source": {"metadata": "amount"}}]
        )

    assert len(fields) == 1 and isinstance(fields[0], FieldSchema)
    assert fields[0].name == "amount" and fields[0].type == "float"
    assert fields[0].coverage == 3

    method, url = mock_req.call_args[0][0], mock_req.call_args[0][1]
    assert method == "PUT" and "/v1/schema/fields" in url
    assert mock_req.call_args[1]["json"] == {
        "fields": [{"name": "amount", "type": "float", "source": {"metadata": "amount"}}]
    }


def test_list_fields(client):
    resp = _ok(
        {"fields": [{"name": "status", "type": "string", "source": {"metadata": "status"}}], "count": 1}
    )
    with patch.object(client._client, "request", return_value=resp) as mock_req:
        fields = client.schema.list_fields()
    assert [f.name for f in fields] == ["status"]
    assert mock_req.call_args[0][0] == "GET"


def test_delete_field(client):
    resp = _ok({"fields": [], "count": 0})
    with patch.object(client._client, "request", return_value=resp) as mock_req:
        remaining = client.schema.delete_field("amount")
    assert remaining == []
    method, url = mock_req.call_args[0][0], mock_req.call_args[0][1]
    assert method == "DELETE" and "/v1/schema/fields/amount" in url


def test_scoped_handle_schema_carries_partition(client):
    resp = _ok({"fields": [], "count": 0})
    with patch.object(client._client, "request", return_value=resp) as mock_req:
        client.partition("acct-9").schema.list_fields()
    assert mock_req.call_args[1]["params"].get("partition") == "acct-9"
