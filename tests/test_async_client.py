"""Tests for AsyncAetherClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aether import AsyncAetherClient, RetrievalResult


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
