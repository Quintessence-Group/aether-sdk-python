"""Tests for AsyncAetherClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aether import AsyncAetherClient, BatchSearchQuery, EntityBackfillReport, RetrievalResult


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
            {"doc_id": "doc-1", "distance": 0.1, "title": "Doc 1",
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
            {"doc_id": "doc-1", "distance": 0.1, "title": "Doc 1",
             "content_type": "text/plain", "content": "Content 1", "passage": "Passage 1"},
            {"doc_id": "doc-2", "distance": 0.3, "title": "Doc 2",
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
            {"doc_id": "doc-1", "distance": 0.1, "title": "Doc 1", "content_type": "text/plain"},
            {"doc_id": "doc-2", "distance": 0.3, "title": "Doc 2", "content_type": "text/plain"},
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
            {"doc_id": "doc-1", "distance": 0.1, "content_type": "text/plain", "content": "C1"},
            {"doc_id": "doc-1", "distance": 0.2, "content_type": "text/plain", "content": "C1"},
            {"doc_id": "doc-2", "distance": 0.3, "content_type": "text/plain", "content": "C2"},
        ],
    })

    with patch.object(client._client, "request", new_callable=AsyncMock, return_value=search_resp):
        results = await client.retrieve("test", k=5)

    assert len(results) == 2
    assert results[0].distance == 0.1


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
    assert url == "/search"
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
    assert url == "/documents"
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
    assert url == "/documents/backfill-entity"
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
