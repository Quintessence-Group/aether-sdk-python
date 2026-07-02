"""Partition lifecycle + trace / verify_isolation +
partition_required typed error — driven through a real client over
httpx.MockTransport so the genuine request/parse/error-mapping path runs."""

import httpx
import pytest

from aether import (
    AetherClient,
    PartitionRequiredError,
)


def _mock_client(handler) -> AetherClient:
    c = AetherClient(base_url="http://localhost:9000", api_key="test-key", max_retries=0)
    c._client = httpx.Client(
        base_url=c.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )
    return c


# ── list_partitions ──────────────────────────────────────────────────

def test_list_partitions_parses_counts_and_warnings():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/partitions"
        return httpx.Response(
            200,
            json={
                "partitions": [
                    {"id": "client-a", "document_count": 3},
                    {"id": "client-b", "document_count": 1},
                ],
                "count": 2,
                "warnings": [
                    {
                        "kind": "single_document",
                        "partitions": ["client-b"],
                        "detail": "holds a single document",
                    }
                ],
            },
        )

    listing = _mock_client(handler).list_partitions()
    assert [p.id for p in listing.partitions] == ["client-a", "client-b"]
    assert listing.partitions[0].document_count == 3
    assert listing.warnings[0].kind == "single_document"
    assert listing.warnings[0].partitions == ["client-b"]


# ── delete_partition ─────────────────────────────────────────────────

def test_delete_partition_returns_count_and_encodes_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        # The id is URL-encoded into the path segment (slash → %2F).
        assert request.url.raw_path == b"/v1/partitions/client%2F42"
        return httpx.Response(200, json={"status": "deleted", "partition": "client/42", "documents_deleted": 7})

    assert _mock_client(handler).delete_partition("client/42") == 7


def test_delete_partition_rejects_empty_id():
    client = _mock_client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        client.delete_partition("")


# ── search_trace + verify_isolation ──────────────────────────────────

def _trace_handler(partitions_touched, default_touched=False, results=1):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("trace") == "true"
        assert request.url.params.get("partition") == "client-a"
        return httpx.Response(
            200,
            json={
                "query": "q",
                "results": [
                    {"doc_id": "d1", "score": 90, "content_type": "text/plain"}
                ][:results],
                "trace": {
                    "scoped_to": "client-a",
                    "partitions_touched": partitions_touched,
                    "default_partition_touched": default_touched,
                    "results": results,
                    "candidates_in_scope": 1,
                    "boundary": "partition",
                },
            },
        )

    return handler


def test_search_trace_returns_results_and_trace():
    client = _mock_client(_trace_handler(["client-a"])).partition("client-a")
    traced = client.search_trace("returns policy")
    assert traced.trace.scoped_to == "client-a"
    assert traced.trace.partitions_touched == ["client-a"]
    assert traced.trace.candidates_in_scope == 1
    assert traced.trace.boundary == "partition"
    assert traced.results[0].doc_id == "d1"


def test_verify_isolation_ok_when_scope_holds():
    client = _mock_client(_trace_handler(["client-a"])).partition("client-a")
    check = client.verify_isolation("returns policy")
    assert check.ok is True
    assert check.leaked == []


def test_verify_isolation_flags_a_leak():
    client = _mock_client(_trace_handler(["client-a", "client-b"])).partition("client-a")
    check = client.verify_isolation("returns policy")
    assert check.ok is False
    assert check.leaked == ["client-b"]


def test_verify_isolation_requires_a_handle():
    client = _mock_client(_trace_handler(["client-a"]))
    with pytest.raises(ValueError):
        client.verify_isolation("returns policy")


# ── Typed partition_required error ───────────────────────────────────

def test_unscoped_multi_tenant_call_raises_partition_required():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": "This API key is multi-tenant, so every search must name a partition.",
                "code": "partition_required",
            },
        )

    client = _mock_client(handler)
    with pytest.raises(PartitionRequiredError) as exc:
        client.search("anything")
    assert exc.value.status_code == 400
    assert exc.value.error_code == "partition_required"
    assert not exc.value.is_retryable
