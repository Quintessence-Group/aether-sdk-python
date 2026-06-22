"""Regression: insert-family methods must populate the full DocumentRecord.

Bug (pre-0.1.2): insert/insert_text/insert_stream/update/insert_with_embeddings/
batch_insert built DocumentRecord with only doc_id/cid/chunks/vectors/version, so
``size_bytes`` was always 0 and content_type/title/timestamps were defaulted even
though the server returns them. See get()/list() which parsed them correctly.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aether import AetherClient, AsyncAetherClient, BatchInsertItem

# A full insert/embed response body as the server actually returns it.
FULL = {
    "doc_id": "d-1",
    "cid": "aether:abc",
    "title": "report.txt",
    "content_type": "text/plain",
    "size_bytes": 86,
    "chunks": 1,
    "vectors": 1,
    "version": 1,
    "created_at": "2026-06-13T00:00:00+00:00",
    "updated_at": None,
}


def _resp(body, status=201):
    r = MagicMock(spec=httpx.Response)
    r.is_success = True
    r.status_code = status
    r.headers = {}
    r.json = MagicMock(return_value=body)
    return r


@pytest.fixture
def client():
    return AetherClient(base_url="http://localhost:9000", api_key="k", max_retries=0)


def _assert_full(rec):
    assert rec.size_bytes == 86, f"size_bytes dropped: {rec.size_bytes}"
    assert rec.content_type == "text/plain"
    assert rec.title == "report.txt"
    assert rec.created_at == "2026-06-13T00:00:00+00:00"
    assert rec.doc_id == "d-1" and rec.chunks == 1 and rec.version == 1


def test_insert_text_populates_size_bytes(client):
    with patch.object(client._client, "request", return_value=_resp(FULL)):
        _assert_full(client.insert_text("x" * 86, tags=["t"]))


def test_insert_with_embeddings_populates_size_bytes(client):
    with patch.object(client._client, "request", return_value=_resp(FULL)):
        _assert_full(client.insert_with_embeddings("content", embedding=[0.0] * 8))


def test_batch_insert_populates_size_bytes(client):
    with patch.object(client._client, "request", return_value=_resp({"results": [FULL, FULL]})):
        recs = client.batch_insert([BatchInsertItem(filename="a.txt", content="a"),
                                    BatchInsertItem(filename="b.txt", content="b")])
    assert len(recs) == 2
    for rec in recs:
        _assert_full(rec)


def test_update_populates_size_bytes(client):
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("new"); tmp = f.name
    try:
        with patch.object(client._client, "request", return_value=_resp(FULL, status=200)):
            _assert_full(client.update("d-1", tmp))
    finally:
        os.unlink(tmp)


@pytest.mark.asyncio
async def test_async_insert_text_populates_size_bytes():
    client = AsyncAetherClient(base_url="http://localhost:9000", api_key="k", max_retries=0)
    with patch.object(client._client, "request", new=AsyncMock(return_value=_resp(FULL))):
        _assert_full(await client.insert_text("x" * 86, tags=["t"]))
