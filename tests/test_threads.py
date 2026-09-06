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


# ── Thread-lifecycle methods (restore / set ACL / move / delete) ──


def test_restore_thread_posts_with_idempotency_and_partition():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["partition"] = request.url.params.get("partition")
        seen["idempotency"] = request.headers.get("Idempotency-Key")
        return httpx.Response(
            200, json={"status": "restored", "thread_id": "chat/42", "turns": 3}
        )

    client = _sync_client(handler).partition("tenant-a")
    try:
        result = client.restore_thread("chat/42")
        # The engine's {status, thread_id, turns} body is parsed and returned.
        assert result.status == "restored"
        assert result.thread_id == "chat/42"
        assert result.turns == 3
        assert seen["method"] == "POST"
        assert seen["path"] == "/v1/threads/chat/42/restore"
        assert seen["partition"] == "tenant-a"
        # The POST retry transport mints a stable key when one is not supplied.
        assert seen["idempotency"]
    finally:
        client.close()


def test_set_thread_acl_puts_body_states_and_idempotency():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["partition"] = request.url.params.get("partition")
        seen["idempotency"] = request.headers.get("Idempotency-Key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"status": "acl_updated", "thread_id": "chat/42", "turns": 2}
        )

    client = _sync_client(handler).partition("tenant-a")
    try:
        result = client.set_thread_acl("chat/42", ["user:a", "group:b"])
        assert (result.status, result.thread_id, result.turns) == (
            "acl_updated",
            "chat/42",
            2,
        )
        assert seen["method"] == "PUT"
        assert seen["path"] == "/v1/threads/chat/42/acl"
        assert seen["partition"] == "tenant-a"
        # PUT is not auto-idempotent in the transport, so the method mints one.
        assert seen["idempotency"]
        assert seen["body"] == {"acl_readers": ["user:a", "group:b"]}

        # The three ACL states each travel explicitly on the wire.
        client.set_thread_acl("chat/42", None)  # unlabel / tenant-visible
        assert seen["body"] == {"acl_readers": None}
        client.set_thread_acl("chat/42", [])  # admin-only quarantine
        assert seen["body"] == {"acl_readers": []}
    finally:
        client.close()


def test_move_thread_posts_expect_and_to_partition_with_idempotency():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["partition"] = request.url.params.get("partition")
        seen["idempotency"] = request.headers.get("Idempotency-Key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"status": "moved", "thread_id": "chat/42", "turns": 4}
        )

    client = _sync_client(handler).partition("tenant-a")
    try:
        result = client.move_thread(
            "chat/42", to_partition="tenant-b", expect_partition="tenant-a"
        )
        assert (result.status, result.thread_id, result.turns) == (
            "moved",
            "chat/42",
            4,
        )
        assert seen["method"] == "POST"
        assert seen["path"] == "/v1/threads/chat/42/move"
        # Move is body-only: partition travels solely as expect_partition, so a
        # partition handle must NOT append a redundant ?partition= query (the
        # engine reads partition from the body only, matching TS/Go/.NET).
        assert seen["partition"] is None
        assert seen["idempotency"]
        assert seen["body"] == {
            "expect_partition": "tenant-a",
            "to_partition": "tenant-b",
        }
    finally:
        client.close()


def test_delete_thread_soft_by_default_and_hard_flag_with_idempotency():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        status = "hard_deleted" if request.url.params.get("hard") else "tombstoned"
        return httpx.Response(
            200, json={"status": status, "thread_id": "chat/42", "turns": 1}
        )

    client = _sync_client(handler).partition("tenant-a")
    try:
        soft_result = client.delete_thread("chat/42")
        assert soft_result.status == "tombstoned"
        assert soft_result.thread_id == "chat/42"
        soft = calls[-1]
        assert soft.method == "DELETE"
        assert soft.url.path == "/v1/threads/chat/42"
        assert "hard" not in soft.url.params  # recoverable tombstone by default
        assert soft.url.params.get("partition") == "tenant-a"
        # DELETE is not auto-idempotent in the transport, so the method mints one.
        assert soft.headers.get("Idempotency-Key")

        hard_result = client.delete_thread("chat/42", hard=True)
        assert hard_result.status == "hard_deleted"
        hard = calls[-1]
        assert hard.method == "DELETE"
        assert hard.url.params.get("hard") == "true"  # irreversible crypto purge
        assert hard.url.params.get("partition") == "tenant-a"
        assert hard.headers.get("Idempotency-Key")
    finally:
        client.close()


def test_thread_lifecycle_validation_happens_before_transport():
    with AetherClient(base_url="http://localhost:9000") as client:
        with pytest.raises(ValueError, match="thread_id"):
            client.restore_thread(" ")
        with pytest.raises(ValueError, match="thread_id"):
            client.set_thread_acl(" ", None)
        with pytest.raises(ValueError, match="thread_id"):
            client.move_thread(" ", to_partition=None)
        with pytest.raises(ValueError, match="thread_id"):
            client.delete_thread(" ")
        # A blank ACL label would silently collapse into the quarantine state
        # server-side, so it is rejected locally before any transport.
        with pytest.raises(ValueError, match="acl_readers"):
            client.set_thread_acl("chat/42", [" "])


def test_thread_facade_lifecycle_delegates_with_partition_expectation():
    from aether import Memory

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200, json={"status": "ok", "thread_id": "chat/42", "turns": 7}
        )

    client = _sync_client(handler).partition("tenant-a")
    try:
        thread = Memory("user-1", client=client).thread("chat/42")
        # The facade forwards the parsed ThreadLifecycleResult through.
        results = [
            thread.restore(),
            thread.set_acl(["user:a"]),
            thread.move("tenant-b"),
            thread.delete(hard=True),
        ]
    finally:
        client.close()

    for result in results:
        assert (result.status, result.thread_id, result.turns) == ("ok", "chat/42", 7)

    restore, acl, move, delete = calls
    assert (restore.method, restore.url.path) == (
        "POST",
        "/v1/threads/chat/42/restore",
    )
    assert (acl.method, acl.url.path) == ("PUT", "/v1/threads/chat/42/acl")
    assert json.loads(acl.content) == {"acl_readers": ["user:a"]}
    assert (move.method, move.url.path) == ("POST", "/v1/threads/chat/42/move")
    # The facade forwards the Memory client's current partition as expect_partition.
    assert json.loads(move.content) == {
        "expect_partition": "tenant-a",
        "to_partition": "tenant-b",
    }
    assert (delete.method, delete.url.path) == ("DELETE", "/v1/threads/chat/42")
    assert delete.url.params.get("hard") == "true"
    for req in calls:
        assert req.headers.get("Idempotency-Key")
    # The partition-guarded routes carry the handle's partition; move is body-only
    # and must NOT append a ?partition= query.
    assert restore.url.params.get("partition") == "tenant-a"
    assert acl.url.params.get("partition") == "tenant-a"
    assert delete.url.params.get("partition") == "tenant-a"
    assert move.url.params.get("partition") is None


@pytest.mark.asyncio
async def test_async_thread_lifecycle_methods_match_sync_contract():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200, json={"status": "ok", "thread_id": "chat/42", "turns": 5}
        )

    client = _async_client(handler).partition("tenant-a")
    try:
        results = [
            await client.restore_thread("chat/42"),
            await client.set_thread_acl("chat/42", ["user:a"]),
            await client.move_thread(
                "chat/42", to_partition="tenant-b", expect_partition="tenant-a"
            ),
            await client.delete_thread("chat/42", hard=True),
        ]
    finally:
        await client.close()

    # The async surface parses and returns the same {status, thread_id, turns}.
    for result in results:
        assert (result.status, result.thread_id, result.turns) == ("ok", "chat/42", 5)

    restore, acl, move, delete = calls
    assert (restore.method, restore.url.path) == (
        "POST",
        "/v1/threads/chat/42/restore",
    )
    assert (acl.method, acl.url.path) == ("PUT", "/v1/threads/chat/42/acl")
    assert json.loads(acl.content) == {"acl_readers": ["user:a"]}
    assert (move.method, move.url.path) == ("POST", "/v1/threads/chat/42/move")
    assert json.loads(move.content) == {
        "expect_partition": "tenant-a",
        "to_partition": "tenant-b",
    }
    assert (delete.method, delete.url.path) == ("DELETE", "/v1/threads/chat/42")
    assert delete.url.params.get("hard") == "true"
    for req in calls:
        assert req.headers["Idempotency-Key"]
    # Move is body-only (no ?partition= guard); the other routes carry the handle.
    assert restore.url.params.get("partition") == "tenant-a"
    assert acl.url.params.get("partition") == "tenant-a"
    assert delete.url.params.get("partition") == "tenant-a"
    assert move.url.params.get("partition") is None
