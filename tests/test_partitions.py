"""Partition lifecycle + trace / verify_isolation + ID-addressed guard +
move_document + partition echo + partition_required typed error — driven
through a real client over httpx.MockTransport so the genuine
request/parse/error-mapping path runs."""

import json

import httpx
import pytest

from aether import (
    AetherApiError,
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


# ── Partition guard on ID-addressed routes ───────────────────────────

_DOC_JSON = {"doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1}


def _guarded_handler(method: str, path: str, *, json_body=None, content=None):
    """Handler asserting the scoped handle sent the partition guard."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == method
        assert request.url.path == path
        assert request.url.params.get("partition") == "client-a"
        if content is not None:
            return httpx.Response(200, content=content)
        return httpx.Response(200, json=json_body if json_body is not None else {})

    return handler


def test_scoped_get_sends_partition_guard():
    handler = _guarded_handler("GET", "/v1/documents/d1", json_body=_DOC_JSON)
    record = _mock_client(handler).partition("client-a").get("d1")
    assert record.doc_id == "d1"


def test_scoped_download_sends_partition_guard(tmp_path):
    handler = _guarded_handler("GET", "/v1/documents/d1/download", content=b"payload")
    written = _mock_client(handler).partition("client-a").download("d1", tmp_path / "out.txt")
    assert written == len(b"payload")


def test_scoped_download_text_sends_partition_guard():
    handler = _guarded_handler("GET", "/v1/documents/d1/download", content=b"payload")
    assert _mock_client(handler).partition("client-a").download_text("d1") == "payload"


def test_scoped_soft_delete_sends_partition_guard():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/v1/documents/d1"
        assert request.url.params.get("partition") == "client-a"
        assert "hard" not in request.url.params  # soft by default
        return httpx.Response(200, json={})

    _mock_client(handler).partition("client-a").delete("d1")


def test_scoped_hard_delete_keeps_hard_and_partition():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("hard") == "true"
        assert request.url.params.get("partition") == "client-a"
        return httpx.Response(200, json={})

    _mock_client(handler).partition("client-a").delete("d1", hard=True)


def test_scoped_restore_sends_partition_guard():
    handler = _guarded_handler("POST", "/v1/documents/d1/restore", json_body={})
    _mock_client(handler).partition("client-a").restore("d1")


def test_scoped_backfill_sends_partition_param():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/documents/backfill-entity"
        assert request.url.params.get("partition") == "client-a"
        assert json.loads(request.content)["tag_prefix"] == "patient:"
        return httpx.Response(
            200,
            json={
                "scanned": 0, "updated": 0, "skipped_existing": 0,
                "skipped_no_match": 0, "skipped_ambiguous": 0, "skipped_invalid": 0,
            },
        )

    _mock_client(handler).partition("client-a").backfill_entity_from_tags("patient:")


def test_unscoped_by_id_call_sends_no_partition():
    def handler(request: httpx.Request) -> httpx.Response:
        # The base client's wire shape is unchanged by the guard support.
        assert "partition" not in request.url.params
        return httpx.Response(200, json=_DOC_JSON)

    _mock_client(handler).get("d1")


def test_wrong_guard_surfaces_the_plain_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": "document d1 not found", "code": "document_not_found"},
        )

    with pytest.raises(AetherApiError) as exc:
        _mock_client(handler).partition("client-b").get("d1")
    assert exc.value.status_code == 404
    assert exc.value.error_code == "document_not_found"


# ── move_document ────────────────────────────────────────────────────

def test_move_document_sends_both_body_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/documents/d1/move"
        body = json.loads(request.content)
        assert body == {"to_partition": "client-b", "expect_partition": "client-a"}
        return httpx.Response(200, json={**_DOC_JSON, "version": 2, "partition": "client-b"})

    record = _mock_client(handler).move_document(
        "d1", from_partition="client-a", to_partition="client-b"
    )
    assert record.partition == "client-b"
    assert record.version == 2


def test_move_document_null_names_the_default_partition():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # Both keys are always present; explicit null = the default partition.
        assert body["expect_partition"] is None
        assert body["to_partition"] == "client-b"
        return httpx.Response(200, json={**_DOC_JSON, "partition": "client-b"})

    record = _mock_client(handler).move_document(
        "d1", from_partition=None, to_partition="client-b"
    )
    assert record.partition == "client-b"


def test_move_document_is_never_scoped_by_a_handle():
    def handler(request: httpx.Request) -> httpx.Response:
        # A relocating call names its boundaries explicitly — the handle's
        # scope is not injected anywhere.
        assert "partition" not in request.url.params
        body = json.loads(request.content)
        assert body == {"to_partition": None, "expect_partition": "client-x"}
        return httpx.Response(200, json=_DOC_JSON)

    _mock_client(handler).partition("client-a").move_document(
        "d1", from_partition="client-x", to_partition=None
    )


def test_move_document_wrong_assertion_is_a_plain_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": "document d1 not found", "code": "document_not_found"},
        )

    with pytest.raises(AetherApiError) as exc:
        _mock_client(handler).move_document(
            "d1", from_partition="client-b", to_partition="client-c"
        )
    assert exc.value.status_code == 404
    assert exc.value.error_code == "document_not_found"


def test_move_document_rejects_empty_doc_id():
    client = _mock_client(lambda r: httpx.Response(200, json=_DOC_JSON))
    with pytest.raises(ValueError):
        client.move_document("", from_partition=None, to_partition="client-b")


def test_move_document_validates_partition_names_client_side():
    client = _mock_client(lambda r: httpx.Response(200, json=_DOC_JSON))
    # Empty/oversized names are rejected like the handle id; None stays valid.
    with pytest.raises(ValueError):
        client.move_document("d1", from_partition="", to_partition="client-b")
    with pytest.raises(ValueError):
        client.move_document("d1", from_partition=None, to_partition="x" * 257)


# ── Partition echo on responses ──────────────────────────────────────

def test_partition_echo_parsed_on_document_record():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**_DOC_JSON, "partition": "client-a"})

    assert _mock_client(handler).get("d1").partition == "client-a"


def test_partition_echo_null_means_default_partition():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**_DOC_JSON, "partition": None})

    assert _mock_client(handler).get("d1").partition is None


def test_partition_echo_parsed_on_search_hits():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "query": "q",
                "results": [
                    {"doc_id": "d1", "score": 90, "partition": "client-a"},
                    {"doc_id": "d2", "score": 80, "partition": None},
                ],
            },
        )

    results = _mock_client(handler).search("q")
    assert [r.partition for r in results] == ["client-a", None]


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


def test_unguarded_by_id_call_raises_partition_required():
    # A key minted with strict scoping rejects even ID-addressed calls that
    # carry no partition guard; the typed error is catchable as the base
    # API error too.
    def handler(request: httpx.Request) -> httpx.Response:
        assert "partition" not in request.url.params
        return httpx.Response(
            400,
            json={
                "error": "This API key requires every document call to name a partition.",
                "code": "partition_required",
            },
        )

    with pytest.raises(AetherApiError) as exc:
        _mock_client(handler).get("d1")
    assert isinstance(exc.value, PartitionRequiredError)
    assert exc.value.error_code == "partition_required"
