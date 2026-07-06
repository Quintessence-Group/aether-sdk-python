"""Tests for AsyncAetherClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aether import AsyncAetherClient, BatchSearchQuery, EntityBackfillReport, RetrievalResult
from aether.errors import CreditExhaustedError, TenantPausedError


@pytest.fixture
def client():
    return AsyncAetherClient(base_url="http://localhost:9000", api_key="test-key")


def make_async_response(json_data=None, content=None, status_code=200):
    """Create a mock httpx.Response for async client."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json.return_value = json_data
    if content is not None:
        resp.content = content
    return resp


@pytest.mark.asyncio
async def test_download_text(client):
    mock_resp = make_async_response(content=b"Hello async world!")

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        result = await client.download_text("doc-123")

    assert result == "Hello async world!"


@pytest.mark.asyncio
async def test_search_with_include_content(client):
    mock_resp = make_async_response(json_data={
        "query": "test",
        "results": [
            {"doc_id": "doc-1", "score": 90, "title": "Doc 1",
             "content_type": "text/plain", "content": "Inline content"},
        ],
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        results = await client.search("test", k=5, include_content=True)

    assert len(results) == 1
    assert results[0].content == "Inline content"


@pytest.mark.asyncio
async def test_retrieve_with_inline_content(client):
    """When server provides content inline, no downloads needed."""
    search_resp = make_async_response(json_data={
        "query": "test",
        "results": [
            {"doc_id": "doc-1", "score": 90, "title": "Doc 1",
             "content_type": "text/plain", "content": "Content 1", "passage": "Passage 1"},
            {"doc_id": "doc-2", "score": 70, "title": "Doc 2",
             "content_type": "text/plain", "content": "Content 2", "passage": None},
        ],
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=search_resp):
        results = await client.retrieve("test query", k=5)

    assert len(results) == 2
    assert isinstance(results[0], RetrievalResult)
    assert results[0].content == "Content 1"
    assert results[0].passage == "Passage 1"
    assert results[1].content == "Content 2"


@pytest.mark.asyncio
async def test_retrieve_fallback_to_download(client):
    """When server doesn't include content, falls back to parallel downloads."""
    search_resp = make_async_response(json_data={
        "query": "test",
        "results": [
            {"doc_id": "doc-1", "score": 90, "title": "Doc 1", "content_type": "text/plain"},
            {"doc_id": "doc-2", "score": 70, "title": "Doc 2", "content_type": "text/plain"},
        ],
    })
    dl_resp_1 = make_async_response(content=b"Downloaded 1")
    dl_resp_2 = make_async_response(content=b"Downloaded 2")

    with patch.object(
        client._client, "request", new_callable=AsyncMock,
        side_effect=[search_resp, dl_resp_1, dl_resp_2],
    ):
        results = await client.retrieve("test", k=5)

    assert len(results) == 2
    assert results[0].content == "Downloaded 1"
    assert results[1].content == "Downloaded 2"


@pytest.mark.asyncio
async def test_insert_stream(client):
    mock_resp = make_async_response(json_data={
        "doc_id": "stream-123",
        "cid": "streamhash",
        "chunks": 5,
        "vectors": 5,
        "version": 1,
    })

    import io
    stream = io.BytesIO(b"streamed data")

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
        result = await client.insert_stream(stream, filename="upload.pdf", content_type="application/pdf")

    assert result.doc_id == "stream-123"
    assert result.chunks == 5
    assert result.version == 1
    mock_post.assert_called_once()
    call_url = mock_post.call_args[0][0]
    assert "filename=upload.pdf" in call_url
    assert "content_type=application/pdf" in call_url


@pytest.mark.asyncio
async def test_retrieve_deduplicates(client):
    search_resp = make_async_response(json_data={
        "query": "test",
        "results": [
            {"doc_id": "doc-1", "score": 90, "content_type": "text/plain", "content": "C1"},
            {"doc_id": "doc-1", "score": 80, "content_type": "text/plain", "content": "C1"},
            {"doc_id": "doc-2", "score": 70, "content_type": "text/plain", "content": "C2"},
        ],
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=search_resp):
        results = await client.retrieve("test", k=5)

    assert len(results) == 2
    assert results[0].score == 90


@pytest.mark.asyncio
async def test_url_encoding_special_chars(client):
    """Verify doc_id with special characters is properly URL-encoded."""
    mock_resp = make_async_response(json_data={
        "doc_id": "doc/with spaces", "cid": "c1", "title": None,
        "content_type": "text/plain", "size_bytes": 0, "chunks": 0,
        "vectors": 0, "version": 1,
    })
    mock_resp.is_success = True

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.get("doc/with spaces")

    call_url = mock_req.call_args[0][1]
    assert "doc%2Fwith%20spaces" in call_url or "doc/with%20spaces" in call_url


@pytest.mark.asyncio
async def test_validation_empty_doc_id(client):
    with pytest.raises(ValueError, match="doc_id cannot be empty"):
        await client.get("")


@pytest.mark.asyncio
async def test_validation_search_k(client):
    with pytest.raises(ValueError, match="k must be at least 1"):
        await client.search("test", k=0)


def _wire_url(params: dict) -> str:
    """Render the final wire URL exactly as httpx would encode the params."""
    return str(httpx.URL("http://localhost:9000/x", params=params))


@pytest.mark.asyncio
async def test_search_passes_filters_as_url_params(client):
    mock_resp = make_async_response(json_data={"query": "q", "results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.search(
            "q",
            k=3,
            entity_id="user-123",
            since="2026-06-01T00:00:00Z",
            until="2026-06-10T23:59:59Z",
            max_distance=0.5,
        )

    method, url = mock_req.call_args[0]
    assert method == "GET"
    assert url == "/v1/search"
    params = mock_req.call_args[1]["params"]
    assert params["entity_id"] == "user-123"
    assert params["since"] == "2026-06-01T00:00:00Z"
    assert params["until"] == "2026-06-10T23:59:59Z"
    assert params["max_distance"] == 0.5
    wire = _wire_url(params)
    assert "entity_id=user-123" in wire
    assert "since=2026-06-01T00%3A00%3A00Z" in wire
    assert "until=2026-06-10T23%3A59%3A59Z" in wire
    assert "max_distance=0.5" in wire


@pytest.mark.asyncio
async def test_search_passes_last_n_days(client):
    mock_resp = make_async_response(json_data={"query": "q", "results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.search("q", last_n_days=30)

    params = mock_req.call_args[1]["params"]
    assert params["last_n_days"] == 30
    assert "since" not in params
    assert "until" not in params
    assert "last_n_days=30" in _wire_url(params)


@pytest.mark.asyncio
async def test_search_encodes_offset_timestamps(client):
    mock_resp = make_async_response(json_data={"query": "q", "results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.search("q", since="2026-06-01T00:00:00+02:00")

    wire = _wire_url(mock_req.call_args[1]["params"])
    assert "since=2026-06-01T00%3A00%3A00%2B02%3A00" in wire


@pytest.mark.asyncio
async def test_search_omits_unset_filters(client):
    mock_resp = make_async_response(json_data={"query": "q", "results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.search("q", k=2)

    params = mock_req.call_args[1]["params"]
    wire = _wire_url(params)
    for key in ("entity_id", "since", "until", "last_n_days", "max_distance"):
        assert key not in params
        assert key not in wire


@pytest.mark.asyncio
async def test_retrieve_forwards_filters(client):
    mock_resp = make_async_response(json_data={"query": "q", "results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.retrieve(
            "q",
            k=2,
            entity_id="user-123",
            since="2026-06-01T00:00:00Z",
            until="2026-06-10T23:59:59Z",
            max_distance=0.4,
        )

    params = mock_req.call_args[1]["params"]
    assert params["include_content"] == "true"
    assert params["entity_id"] == "user-123"
    assert params["since"] == "2026-06-01T00:00:00Z"
    assert params["until"] == "2026-06-10T23:59:59Z"
    assert params["max_distance"] == 0.4


@pytest.mark.asyncio
async def test_retrieve_forwards_last_n_days(client):
    mock_resp = make_async_response(json_data={"query": "q", "results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.retrieve("q", last_n_days=7)

    params = mock_req.call_args[1]["params"]
    assert params["last_n_days"] == 7
    assert "since" not in params


@pytest.mark.asyncio
async def test_list_passes_filters_as_url_params(client):
    mock_resp = make_async_response(json_data={
        "documents": [{"doc_id": "d1", "entity_id": "user-123", "created_at": "2026-06-02T08:00:00Z"}],
        "total": 1,
        "has_more": False,
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        records = await client.list(
            entity_id="user-123",
            since="2026-06-01T00:00:00Z",
            until="2026-06-10T23:59:59Z",
        )

    method, url = mock_req.call_args[0]
    assert method == "GET"
    assert url == "/v1/documents"
    params = mock_req.call_args[1]["params"]
    assert params == {
        "offset": 0,
        "limit": 50,
        "entity_id": "user-123",
        "since": "2026-06-01T00:00:00Z",
        "until": "2026-06-10T23:59:59Z",
    }
    wire = _wire_url(params)
    assert "since=2026-06-01T00%3A00%3A00Z" in wire
    assert "until=2026-06-10T23%3A59%3A59Z" in wire
    assert records[0].entity_id == "user-123"


@pytest.mark.asyncio
async def test_list_passes_last_n_days(client):
    mock_resp = make_async_response(json_data={"documents": [], "total": 0, "has_more": False})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.list(last_n_days=7)

    params = mock_req.call_args[1]["params"]
    assert params["last_n_days"] == 7
    assert "since" not in params


@pytest.mark.asyncio
async def test_list_omits_unset_filters(client):
    mock_resp = make_async_response(json_data={"documents": [], "total": 0, "has_more": False})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.list()

    assert mock_req.call_args[1]["params"] == {"offset": 0, "limit": 50}


@pytest.mark.asyncio
async def test_insert_sends_entity_id_param(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
        "entity_id": "user-123",
        "created_at": "2026-06-11T00:00:00Z",
        "updated_at": "2026-06-11T00:00:00Z",
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        record = await client.insert(f, entity_id="user-123")

    url = mock_req.call_args[0][1]
    assert "entity_id=user-123" in url
    # Mapper round-trips the full record returned by the server
    assert record.entity_id == "user-123"
    assert record.created_at == "2026-06-11T00:00:00Z"
    assert record.updated_at == "2026-06-11T00:00:00Z"


@pytest.mark.asyncio
async def test_insert_url_encodes_entity_id(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.insert(f, entity_id="customer:42")

    assert "entity_id=customer%3A42" in mock_req.call_args[0][1]


@pytest.mark.asyncio
async def test_insert_omits_entity_id_when_unset(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        record = await client.insert(f)

    assert "entity_id" not in mock_req.call_args[0][1]
    assert record.entity_id is None


@pytest.mark.asyncio
async def test_insert_with_embeddings_sends_entity_id_json(client):
    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
        "entity_id": "user-123",
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        record = await client.insert_with_embeddings("text", embedding=[0.1], entity_id="user-123")

    body = mock_req.call_args[1]["json"]
    assert body["entity_id"] == "user-123"
    assert record.entity_id == "user-123"


@pytest.mark.asyncio
async def test_search_by_vector_sends_filters_json(client):
    mock_resp = make_async_response(json_data={"results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.search_by_vector(
            [0.1, 0.2],
            k=3,
            entity_id="user-123",
            since="2026-06-01T00:00:00Z",
            until="2026-06-10T23:59:59Z",
            max_distance=0.5,
        )

    body = mock_req.call_args[1]["json"]
    assert body["entity_id"] == "user-123"
    assert body["since"] == "2026-06-01T00:00:00Z"
    assert body["until"] == "2026-06-10T23:59:59Z"
    assert body["max_distance"] == 0.5
    assert "last_n_days" not in body


@pytest.mark.asyncio
async def test_search_by_vector_sends_last_n_days_json(client):
    mock_resp = make_async_response(json_data={"results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.search_by_vector([0.1], last_n_days=30)

    body = mock_req.call_args[1]["json"]
    assert body["last_n_days"] == 30
    assert "since" not in body


@pytest.mark.asyncio
async def test_search_by_vector_omits_unset_filters(client):
    mock_resp = make_async_response(json_data={"results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.search_by_vector([0.1], k=2)

    body = mock_req.call_args[1]["json"]
    for key in ("entity_id", "since", "until", "last_n_days", "max_distance"):
        assert key not in body


@pytest.mark.asyncio
async def test_batch_search_serializes_filters(client):
    mock_resp = make_async_response(json_data={
        "results": [
            {"query": "filtered", "results": []},
            {"query": "recent", "results": []},
            {"query": "plain", "results": []},
        ],
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.batch_search([
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


@pytest.mark.asyncio
async def test_get_maps_entity_id(client):
    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "title": None, "content_type": "text/plain",
        "size_bytes": 5, "chunks": 1, "vectors": 1, "version": 1,
        "created_at": "2026-06-11T00:00:00Z", "updated_at": "2026-06-11T00:00:00Z",
        "entity_id": "user-123",
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        record = await client.get("d1")

    assert record.entity_id == "user-123"


def _backfill_report_json():
    return {
        "scanned": 10,
        "updated": 6,
        "skipped_existing": 2,
        "skipped_no_match": 1,
        "skipped_ambiguous": 1,
        "skipped_invalid": 0,
    }


@pytest.mark.asyncio
async def test_backfill_entity_from_tags_posts_default_body(client):
    mock_resp = make_async_response(json_data=_backfill_report_json())

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.backfill_entity_from_tags("patient:")

    method, url = mock_req.call_args[0]
    assert method == "POST"
    assert url == "/v1/documents/backfill-entity"
    assert mock_req.call_args[1]["json"] == {"tag_prefix": "patient:", "overwrite": False}


@pytest.mark.asyncio
async def test_backfill_entity_from_tags_forwards_overwrite(client):
    mock_resp = make_async_response(json_data=_backfill_report_json())

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.backfill_entity_from_tags("patient:", overwrite=True)

    assert mock_req.call_args[1]["json"] == {"tag_prefix": "patient:", "overwrite": True}


@pytest.mark.asyncio
async def test_backfill_entity_from_tags_parses_report(client):
    mock_resp = make_async_response(json_data=_backfill_report_json())

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        report = await client.backfill_entity_from_tags("patient:")

    assert isinstance(report, EntityBackfillReport)
    assert report.scanned == 10
    assert report.updated == 6
    assert report.skipped_existing == 2
    assert report.skipped_no_match == 1
    assert report.skipped_ambiguous == 1
    assert report.skipped_invalid == 0


@pytest.mark.asyncio
async def test_backfill_entity_from_tags_empty_prefix_raises(client):
    with pytest.raises(ValueError):
        await client.backfill_entity_from_tags("")


# ── Canonical billing errors through real client methods ─────────────────

CREDIT_EXHAUSTED_BODY = {
    "error": "Prepaid credit balance exhausted; top up to continue.",
    "code": "credit_exhausted",
    "request_id": "req-123",
    "resource": "vectors",
    "balance_cents": 0,
}
TENANT_PAUSED_BODY = {
    "error": "Tenant has been paused by the operator",
    "code": "tenant_paused",
    "request_id": "req-123",
}


def _mock_async_client(handler) -> AsyncAetherClient:
    """Real AsyncAetherClient backed by an httpx.MockTransport so requests
    flow through the genuine request/error-mapping path.
    """
    c = AsyncAetherClient(base_url="http://localhost:9000", api_key="test-key", max_retries=0)
    c._client = httpx.AsyncClient(
        base_url=c.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )
    return c


@pytest.mark.asyncio
async def test_insert_text_raises_credit_exhausted_on_402():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json=CREDIT_EXHAUSTED_BODY, headers={"x-request-id": "req-123"})

    client = _mock_async_client(handler)
    with pytest.raises(CreditExhaustedError) as exc_info:
        await client.insert_text("hello world")

    err = exc_info.value
    assert err.status_code == 402
    assert err.error_code == "credit_exhausted"
    assert err.request_id == "req-123"
    assert err.body["balance_cents"] == 0
    assert not err.is_retryable
    await client.close()


@pytest.mark.asyncio
async def test_search_raises_tenant_paused_on_403():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json=TENANT_PAUSED_BODY, headers={"x-request-id": "req-123"})

    client = _mock_async_client(handler)
    with pytest.raises(TenantPausedError) as exc_info:
        await client.search("anything")

    err = exc_info.value
    assert err.status_code == 403
    assert err.error_code == "tenant_paused"
    assert err.request_id == "req-123"
    assert not err.is_retryable
    await client.close()


# ── size_bytes / content_type parse regression ────────────────────────

_FULL_INSERT_BODY = {
    "doc_id": "d1",
    "cid": "c1",
    "title": "My Doc",
    "content_type": "application/pdf",
    "size_bytes": 4096,
    "chunks": 3,
    "vectors": 3,
    "version": 1,
    "created_at": "2026-06-11T00:00:00Z",
    "updated_at": "2026-06-11T00:00:00Z",
    "entity_id": "user-123",
}


@pytest.mark.asyncio
async def test_insert_text_parses_size_and_content_type(client):
    mock_resp = make_async_response(json_data=_FULL_INSERT_BODY, status_code=201)

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        record = await client.insert_text("hello world")

    assert record.size_bytes == 4096
    assert record.content_type == "application/pdf"
    assert record.title == "My Doc"


@pytest.mark.asyncio
async def test_insert_parses_size_and_content_type(client, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"data")
    mock_resp = make_async_response(json_data=_FULL_INSERT_BODY, status_code=201)

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        record = await client.insert(f)

    assert record.size_bytes == 4096
    assert record.content_type == "application/pdf"


@pytest.mark.asyncio
async def test_update_parses_size_and_content_type(client, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"data")
    mock_resp = make_async_response(json_data=_FULL_INSERT_BODY, status_code=201)

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        record = await client.update("d1", f)

    assert record.size_bytes == 4096
    assert record.content_type == "application/pdf"


@pytest.mark.asyncio
async def test_insert_stream_parses_size_and_content_type(client):
    import io

    mock_resp = make_async_response(json_data=_FULL_INSERT_BODY, status_code=201)

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        record = await client.insert_stream(io.BytesIO(b"data"))

    assert record.size_bytes == 4096
    assert record.content_type == "application/pdf"


@pytest.mark.asyncio
async def test_insert_text_defaults_size_when_absent(client):
    mock_resp = make_async_response(
        json_data={"doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1},
        status_code=201,
    )

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        record = await client.insert_text("hi")

    assert record.size_bytes == 0
    assert record.content_type == ""


# ── scoped-partition handle (async mirror) ──────────────────


def _async_partition_insert_resp():
    return make_async_response(
        json_data={"doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1},
    )


@pytest.mark.asyncio
async def test_partition_returns_distinct_scoped_object(client):
    scoped = client.partition("tenant-a")
    assert scoped is not client
    assert isinstance(scoped, AsyncAetherClient)
    assert scoped._partition == "tenant-a"
    assert client._partition is None
    # Shares transport + config, does not own it.
    assert scoped._client is client._client
    assert scoped._max_retries == client._max_retries
    assert scoped._owns_transport is False
    assert client._owns_transport is True


@pytest.mark.asyncio
async def test_async_rescoping_last_wins(client):
    scoped = client.partition("a").partition("b")
    assert scoped._partition == "b"
    assert scoped._client is client._client


@pytest.mark.asyncio
async def test_async_original_client_sends_no_partition(client):
    resp = make_async_response(json_data={"query": "q", "results": []})
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=resp) as mock_req:
        await client.search("q")

    assert "partition" not in mock_req.call_args[1]["params"]


@pytest.mark.asyncio
async def test_async_search_sends_partition_query(client):
    resp = make_async_response(json_data={"query": "q", "results": []})
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=resp) as mock_req:
        await client.partition("tenant-a").search("q")

    params = mock_req.call_args[1]["params"]
    assert params["partition"] == "tenant-a"
    assert "partition=tenant-a" in _wire_url(params)


@pytest.mark.asyncio
async def test_async_insert_text_sends_partition_query(client):
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=_async_partition_insert_resp()) as mock_req:
        await client.partition("tenant-a").insert_text("hello")

    assert "partition=tenant-a" in mock_req.call_args[0][1]


@pytest.mark.asyncio
async def test_async_list_sends_partition_query(client):
    resp = make_async_response(json_data={"documents": [], "total": 0, "has_more": False})
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=resp) as mock_req:
        await client.partition("tenant-a").list()

    assert mock_req.call_args[1]["params"]["partition"] == "tenant-a"


@pytest.mark.asyncio
async def test_async_search_by_vector_sends_partition_body(client):
    resp = make_async_response(json_data={"results": []})
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=resp) as mock_req:
        await client.partition("tenant-a").search_by_vector([0.1, 0.2])

    assert mock_req.call_args[1]["json"]["partition"] == "tenant-a"


@pytest.mark.asyncio
async def test_async_insert_with_embeddings_sends_partition_body(client):
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=_async_partition_insert_resp()) as mock_req:
        await client.partition("tenant-a").insert_with_embeddings("text", embedding=[0.1])

    assert mock_req.call_args[1]["json"]["partition"] == "tenant-a"


@pytest.mark.asyncio
async def test_async_batch_insert_sends_partition_per_item(client):
    from aether import BatchInsertItem

    resp = make_async_response(json_data={"results": []})
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=resp) as mock_req:
        await client.partition("tenant-a").batch_insert([
            BatchInsertItem(filename="a.txt", content="hello"),
            BatchInsertItem(filename="b.txt", content="world"),
        ])

    docs = mock_req.call_args[1]["json"]["documents"]
    assert [d["partition"] for d in docs] == ["tenant-a", "tenant-a"]


@pytest.mark.asyncio
async def test_async_batch_search_sends_partition_per_query(client):
    from aether import BatchSearchQuery

    resp = make_async_response(json_data={"results": []})
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=resp) as mock_req:
        await client.partition("tenant-a").batch_search([
            BatchSearchQuery(q="one"),
            BatchSearchQuery(q="two"),
        ])

    queries = mock_req.call_args[1]["json"]["queries"]
    assert [q["partition"] for q in queries] == ["tenant-a", "tenant-a"]


@pytest.mark.asyncio
async def test_async_update_sends_partition_query(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=_async_partition_insert_resp()) as mock_req:
        await client.partition("tenant-a").update("doc-1", f)

    method, url = mock_req.call_args[0]
    assert method == "PUT"
    assert "partition=tenant-a" in url


@pytest.mark.asyncio
async def test_async_insert_stream_sends_partition_query(client):
    import io

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=_async_partition_insert_resp()) as mock_post:
        await client.partition("tenant-a").insert_stream(io.BytesIO(b"data"))

    assert "partition=tenant-a" in mock_post.call_args[0][0]


@pytest.mark.asyncio
async def test_async_partition_value_is_url_encoded(client):
    resp = make_async_response(json_data={"query": "q", "results": []})
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=resp) as mock_req:
        await client.partition("tenant:a").search("q")

    assert "partition=tenant%3Aa" in _wire_url(mock_req.call_args[1]["params"])


@pytest.mark.asyncio
async def test_async_get_sends_partition_guard(client):
    resp = make_async_response(json_data={"doc_id": "d1", "cid": "c1"})
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=resp) as mock_req:
        await client.partition("tenant-a").get("d1")

    assert mock_req.call_args[0][1] == "/v1/documents/d1?partition=tenant-a"


@pytest.mark.asyncio
async def test_async_delete_sends_partition_guard(client):
    resp = make_async_response()
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=resp) as mock_req:
        await client.partition("tenant-a").delete("d1")

    assert mock_req.call_args[0][1] == "/v1/documents/d1?partition=tenant-a"


@pytest.mark.asyncio
async def test_async_unscoped_get_sends_no_partition(client):
    # The base client keeps the pre-handle wire shape byte-identical.
    resp = make_async_response(json_data={"doc_id": "d1", "cid": "c1"})
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=resp) as mock_req:
        await client.get("d1")

    assert mock_req.call_args[0][1] == "/v1/documents/d1"
    assert "params" not in mock_req.call_args[1]


@pytest.mark.asyncio
async def test_async_move_document_sends_both_body_fields(client):
    resp = make_async_response(
        json_data={"doc_id": "d1", "cid": "c1", "version": 2, "partition": "tenant-b"},
    )
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=resp) as mock_req:
        record = await client.move_document("d1", from_partition=None, to_partition="tenant-b")

    method, url = mock_req.call_args[0]
    assert method == "POST"
    assert url == "/v1/documents/d1/move"
    # Both keys are always present; explicit null = the default partition.
    assert mock_req.call_args[1]["json"] == {
        "to_partition": "tenant-b",
        "expect_partition": None,
    }
    assert record.partition == "tenant-b"


@pytest.mark.asyncio
async def test_async_empty_partition_rejected(client):
    with pytest.raises(ValueError):
        client.partition("   ")


@pytest.mark.asyncio
async def test_async_too_long_partition_rejected(client):
    with pytest.raises(ValueError):
        client.partition("x" * 257)


@pytest.mark.asyncio
async def test_async_invalid_partition_makes_no_http_call(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as mock_req:
        with pytest.raises(ValueError):
            client.partition("")
    mock_req.assert_not_called()


@pytest.mark.asyncio
async def test_async_closing_scoped_handle_keeps_parent_open():
    base = AsyncAetherClient(base_url="http://localhost:9000", api_key="k")
    scoped = base.partition("tenant-a")
    await scoped.close()
    assert not base._client.is_closed
    await base.close()
    assert base._client.is_closed


# ── metadata facet filters + source (async mirror) ──────────────────


@pytest.mark.asyncio
async def test_async_search_sends_facet_filters_csv(client):
    mock_resp = make_async_response(json_data={"query": "q", "results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.search(
            "q",
            tags=["a", "b"],
            any_tags=["x", "y"],
            content_types=["text/plain", "application/pdf"],
            sources=["slack", "email"],
        )

    params = mock_req.call_args[1]["params"]
    assert params["tags"] == "a,b"
    assert params["any_tags"] == "x,y"
    assert params["content_type"] == "text/plain,application/pdf"
    assert params["source"] == "slack,email"
    wire = _wire_url(params)
    assert "any_tags=x%2Cy" in wire
    assert "content_type=text%2Fplain%2Capplication%2Fpdf" in wire
    assert "source=slack%2Cemail" in wire


@pytest.mark.asyncio
async def test_async_search_omits_unset_facet_filters(client):
    mock_resp = make_async_response(json_data={"query": "q", "results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.search("q")

    params = mock_req.call_args[1]["params"]
    for key in ("any_tags", "content_type", "source"):
        assert key not in params


@pytest.mark.asyncio
async def test_async_list_sends_facet_filters_csv(client):
    mock_resp = make_async_response(json_data={"documents": [], "total": 0, "has_more": False})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.list(
            tags=["a", "b"],
            any_tags=["x"],
            content_types=["text/plain"],
            sources=["slack"],
        )

    params = mock_req.call_args[1]["params"]
    assert params["tags"] == "a,b"
    assert params["any_tags"] == "x"
    assert params["content_type"] == "text/plain"
    assert params["source"] == "slack"


@pytest.mark.asyncio
async def test_async_retrieve_forwards_facet_filters(client):
    mock_resp = make_async_response(json_data={"query": "q", "results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.retrieve(
            "q",
            any_tags=["x"],
            content_types=["text/plain"],
            sources=["slack"],
        )

    params = mock_req.call_args[1]["params"]
    assert params["include_content"] == "true"
    assert params["any_tags"] == "x"
    assert params["content_type"] == "text/plain"
    assert params["source"] == "slack"


@pytest.mark.asyncio
async def test_async_search_by_vector_sends_facet_filters_arrays(client):
    mock_resp = make_async_response(json_data={"results": []})

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.search_by_vector(
            [0.1, 0.2],
            any_tags=["x", "y"],
            content_types=["text/plain"],
            sources=["slack"],
        )

    body = mock_req.call_args[1]["json"]
    assert body["any_tags"] == ["x", "y"]
    assert body["content_type"] == ["text/plain"]
    assert body["source"] == ["slack"]


@pytest.mark.asyncio
async def test_async_insert_sends_source_and_round_trips(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
        "source": "slack", "tags": ["a", "b"],
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        record = await client.insert(f, source="slack")

    assert "source=slack" in mock_req.call_args[0][1]
    assert record.source == "slack"
    assert record.tags == ["a", "b"]


@pytest.mark.asyncio
async def test_async_insert_omits_source_when_unset(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        record = await client.insert(f)

    assert "source" not in mock_req.call_args[0][1]
    assert record.source is None
    assert record.tags == []


@pytest.mark.asyncio
async def test_async_insert_text_sends_source_param(client):
    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
    })
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.insert_text("hello", source="slack")

    assert "source=slack" in mock_req.call_args[0][1]


@pytest.mark.asyncio
async def test_async_update_sends_source_param(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
    })
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.update("doc-1", f, source="slack")

    method, url = mock_req.call_args[0]
    assert method == "PUT"
    assert "source=slack" in url


@pytest.mark.asyncio
async def test_async_insert_stream_sends_source_param(client):
    import io

    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
    })
    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
        await client.insert_stream(io.BytesIO(b"data"), source="slack")

    assert "source=slack" in mock_post.call_args[0][0]


@pytest.mark.asyncio
async def test_async_insert_async_sends_source_param(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    mock_resp = make_async_response(json_data={"job_id": "j1", "status": "queued", "poll_url": "/x"})
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.insert_async(f, source="slack")

    assert "source=slack" in mock_req.call_args[0][1]


@pytest.mark.asyncio
async def test_async_insert_with_embeddings_sends_source_json(client):
    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
        "source": "slack",
    })
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        record = await client.insert_with_embeddings("text", embedding=[0.1], source="slack")

    assert mock_req.call_args[1]["json"]["source"] == "slack"
    assert record.source == "slack"


@pytest.mark.asyncio
async def test_async_batch_insert_serializes_source(client):
    from aether import BatchInsertItem

    mock_resp = make_async_response(json_data={
        "results": [
            {"doc_id": "b1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1, "source": "slack"},
            {"doc_id": "b2", "cid": "c2", "chunks": 1, "vectors": 1, "version": 1},
        ],
    })
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        results = await client.batch_insert([
            BatchInsertItem(filename="a.txt", content="hello", source="slack"),
            BatchInsertItem(filename="b.txt", content="world"),
        ])

    docs = mock_req.call_args[1]["json"]["documents"]
    assert docs[0]["source"] == "slack"
    assert "source" not in docs[1]
    assert results[0].source == "slack"
    assert results[1].source is None


@pytest.mark.asyncio
async def test_async_batch_search_serializes_facet_filters_csv(client):
    mock_resp = make_async_response(json_data={
        "results": [
            {"query": "filtered", "results": []},
            {"query": "plain", "results": []},
        ],
    })
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp) as mock_req:
        await client.batch_search([
            BatchSearchQuery(
                q="filtered",
                tags=["a", "b"],
                any_tags=["x"],
                content_types=["text/plain"],
                sources=["slack"],
            ),
            BatchSearchQuery(q="plain"),
        ])

    queries = mock_req.call_args[1]["json"]["queries"]
    assert queries[0]["tags"] == "a,b"
    assert queries[0]["any_tags"] == "x"
    assert queries[0]["content_type"] == "text/plain"
    assert queries[0]["source"] == "slack"
    for key in ("tags", "any_tags", "content_type", "source"):
        assert key not in queries[1]


@pytest.mark.asyncio
async def test_async_search_parses_metadata_echo(client):
    mock_resp = make_async_response(json_data={
        "query": "q",
        "results": [
            {
                "doc_id": "d1", "score": 90, "content_type": "text/plain",
                "tags": ["a", "b"], "source": "slack",
                "created_at": "2026-06-11T00:00:00Z",
            },
        ],
    })
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        results = await client.search("q")

    assert results[0].tags == ["a", "b"]
    assert results[0].source == "slack"
    assert results[0].created_at == "2026-06-11T00:00:00Z"


@pytest.mark.asyncio
async def test_async_get_parses_tags_and_source(client):
    mock_resp = make_async_response(json_data={
        "doc_id": "d1", "cid": "c1", "tags": ["a"], "source": "slack",
    })
    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        record = await client.get("d1")

    assert record.tags == ["a"]
    assert record.source == "slack"


@pytest.mark.asyncio
async def test_async_ingest_files_reports_each_file(client, tmp_path):
    """Async batch ingest reports per-file outcomes and skips
    unsupported types instead of aborting."""
    from pathlib import Path

    from aether import DocumentRecord, IngestResult
    from aether.errors import AetherApiError

    good = tmp_path / "a.md"
    good.write_text("# hi")
    bad = tmp_path / "b.bin"
    bad.write_bytes(b"\x00\x01")

    async def _insert(path, content_type=None, **kwargs):
        if Path(path).name.endswith(".bin"):
            raise AetherApiError(422, "unsupported", error_code="unsupported")
        return DocumentRecord(doc_id=f"doc-{Path(path).name}", cid="c", content_type=content_type or "")

    with patch.object(client, "insert", side_effect=_insert):
        results = await client.ingest_files([good, bad])

    assert [r.status for r in results] == ["ingested", "skipped"]
    assert isinstance(results[0], IngestResult)
    assert results[0].doc_id == "doc-a.md"
    assert results[0].content_type == "text/markdown"


@pytest.mark.asyncio
async def test_async_search_parses_query_id_when_present(client):
    """Usage-feedback capture: the response-level query_id is stamped onto
    every hit; absent -> None (tolerant parse)."""
    mock_resp = make_async_response(json_data={
        "query": "q",
        "query_id": "qid-async",
        "results": [
            {"doc_id": "doc-1", "score": 90, "content_type": "text/plain"},
        ],
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        results = await client.search("q")

    assert results[0].query_id == "qid-async"


@pytest.mark.asyncio
async def test_async_search_query_id_none_when_absent(client):
    mock_resp = make_async_response(json_data={
        "query": "q",
        "results": [
            {"doc_id": "doc-1", "score": 90, "content_type": "text/plain"},
        ],
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_resp):
        results = await client.search("q")

    assert results[0].query_id is None


@pytest.mark.asyncio
async def test_async_send_search_feedback_posts_body(client):
    mock_resp = make_async_response(json_data={"recorded": True})

    with patch.object(
        client._client, "request", new_callable=AsyncMock, return_value=mock_resp
    ) as mock_req:
        result = await client.send_search_feedback("qid-1", "doc-1", "ignored")

    assert result is None
    method, url = mock_req.call_args[0]
    assert method == "POST"
    assert url == "/v1/search/feedback"
    assert mock_req.call_args.kwargs["json"] == {
        "query_id": "qid-1",
        "doc_id": "doc-1",
        "signal": "ignored",
    }


@pytest.mark.asyncio
async def test_async_send_search_feedback_validates_arguments(client):
    with pytest.raises(ValueError):
        await client.send_search_feedback("", "doc-1", "used")
    with pytest.raises(ValueError):
        await client.send_search_feedback("qid-1", "", "used")
    with pytest.raises(ValueError):
        await client.send_search_feedback("qid-1", "doc-1", "")
