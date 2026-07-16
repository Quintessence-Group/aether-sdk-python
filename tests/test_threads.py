"""Conversational-thread public SDK contract tests."""

import json

import httpx
import pytest

from aether import AetherClient, AsyncAetherClient
from aether.client import _validate_thread_id


def _sync_client(handler) -> AetherClient:
    client = AetherClient(
        base_url="http://localhost:9000", api_key="test-key", max_retries=0
    )
    client._client = httpx.Client(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )
    return client


def _async_client(handler) -> AsyncAetherClient:
    client = AsyncAetherClient(
        base_url="http://localhost:9000", api_key="test-key", max_retries=0
    )
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )
    return client


def _turn(doc_id: str, turn_index: int) -> dict:
    return {
        "doc_id": doc_id,
        "cid": f"blake3:{doc_id}",
        "chunks": 1,
        "vectors": 1,
        "version": 1,
        "content_type": "text/plain",
        "size_bytes": 5,
        "thread_id": "chat/42",
        "turn_index": turn_index,
    }


def test_append_thread_versions_escapes_and_sends_stable_caller_key():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["partition"] = request.url.params.get("partition")
        seen["idempotency"] = request.headers.get("Idempotency-Key")
        seen["content_type"] = request.headers.get("Content-Type")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json=_turn("turn-1", 0))

    client = _sync_client(handler).partition("tenant-a")
    turn = client.append_thread(
        "chat/42",
        "hello",
        tags=["support"],
        metadata={"role": "user"},
        idempotency_key="thread-turn-42-0",
    )
    try:
        assert seen == {
            "path": "/v1/threads/chat/42/append",
            "partition": "tenant-a",
            "idempotency": "thread-turn-42-0",
            "content_type": "application/json",
            "body": {"text": "hello", "metadata": {"role": "user"}, "tags": ["support"]},
        }
        assert turn.thread_id == "chat/42"
        assert turn.turn_index == 0
    finally:
        client.close()


def test_get_thread_forwards_window_and_partition():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"thread_id": "chat/42", "documents": [_turn("turn-2", 2)]})

    client = _sync_client(handler).partition("tenant-a")
    try:
        thread = client.get_thread("chat/42", last_n_turns=3, recent_first=True)
        assert seen == {
            "path": "/v1/threads/chat/42",
            "params": {"last_n_turns": "3", "recent_first": "true", "partition": "tenant-a"},
        }
        assert thread.documents[0].turn_index == 2
    finally:
        client.close()


def test_thread_validation_happens_before_transport():
    with AetherClient(base_url="http://localhost:9000") as client:
        with pytest.raises(ValueError, match="thread_id"):
            client.append_thread(" ", "hello")
        for thread_id in (".", ".."):
            with pytest.raises(ValueError, match="dot segment"):
                client.append_thread(thread_id, "hello")
        with pytest.raises(ValueError, match="control"):
            client.append_thread("safe\x00id", "hello")
        with pytest.raises(ValueError, match="control"):
            client.get_thread("safe\x85id")
        with pytest.raises(ValueError, match="surrogate"):
            client.append_thread("safe\ud800id", "hello")
        with pytest.raises(ValueError, match="surrogate"):
            client.get_thread("safe\udc00id")
        with pytest.raises(ValueError, match="256"):
            client.get_thread("😀" * 257)
        with pytest.raises(ValueError, match="last_n_turns"):
            client.get_thread("chat", last_n_turns=0)
        with pytest.raises(ValueError, match="last_n_turns"):
            client.get_thread("chat", last_n_turns=1001)
        with pytest.raises(ValueError, match="last_n_turns"):
            client.get_thread("chat", last_n_turns=1.5)

    assert _validate_thread_id("😀" * 256) == "😀" * 256


@pytest.mark.asyncio
async def test_async_thread_validation_rejects_surrogates_before_transport():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("invalid thread id reached transport")

    client = _async_client(handler)
    try:
        with pytest.raises(ValueError, match="surrogate"):
            await client.append_thread("safe\ud800id", "hello")
        with pytest.raises(ValueError, match="surrogate"):
            await client.get_thread("safe\udc00id")
    finally:
        await client.close()

    assert calls == []


def test_thread_filter_is_forwarded_to_search():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["thread_id"] = request.url.params.get("thread_id")
        return httpx.Response(200, json={"query": "hello", "results": []})

    with _sync_client(handler) as client:
        assert client.search("hello", thread_id="chat/42") == []
        assert seen["thread_id"] == "chat/42"


@pytest.mark.asyncio
async def test_async_thread_append_and_read_match_sync_contract():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(201, json=_turn("turn-1", 0))
        return httpx.Response(200, json={"thread_id": "chat/42", "documents": [_turn("turn-1", 0)]})

    client = _async_client(handler).partition("tenant-a")
    try:
        turn = await client.append_thread("chat/42", "hello", idempotency_key="stable-key")
        thread = await client.get_thread("chat/42", last_n_turns=1)
        assert turn.turn_index == 0
        assert thread.documents[0].thread_id == "chat/42"
        assert calls[0].url.path == "/v1/threads/chat/42/append"
        assert calls[0].headers["Idempotency-Key"] == "stable-key"
        assert calls[1].url.params["last_n_turns"] == "1"
    finally:
        await client.close()
