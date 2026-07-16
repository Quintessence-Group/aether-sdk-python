"""Read-ACLs, acting-principal scoping, and the access-audit surface —
``acl_readers`` wire encoding on the insert family, ``as_principal()``
assertion injection (and its composition with ``partition()``),
``client.audit.access()`` parsing, and the typed pinned-key mismatch error —
driven through a real client over httpx.MockTransport so the genuine
request/parse/error-mapping path runs."""

import json

import httpx
import pytest

from aether import (
    AccessAuditPage,
    AetherClient,
    AsyncAetherClient,
    BatchInsertItem,
    PrincipalPinMismatchError,
)


def _mock_client(handler) -> AetherClient:
    c = AetherClient(base_url="http://localhost:9000", api_key="test-key", max_retries=0)
    c._client = httpx.Client(
        base_url=c.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )
    return c


def _mock_async_client(handler) -> AsyncAetherClient:
    c = AsyncAetherClient(base_url="http://localhost:9000", api_key="test-key", max_retries=0)
    c._client = httpx.AsyncClient(
        base_url=c.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )
    return c


_DOC = {"doc_id": "d1", "cid": "c1", "content_type": "text/plain"}


# ── acl_readers wire encoding ────────────────────────────────────────

def test_insert_text_sends_acl_readers_csv():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.params["acl_readers"] == "user:alice,group:eng"
        return httpx.Response(200, json=_DOC)

    _mock_client(handler).insert_text(
        "hello", acl_readers=["user:alice", "group:eng"]
    )


def test_insert_text_empty_acl_readers_is_explicit_empty_param():
    # [] is meaningful (admin-only quarantine): the param is present but empty.
    def handler(request: httpx.Request) -> httpx.Response:
        assert "acl_readers" in request.url.params
        assert request.url.params["acl_readers"] == ""
        return httpx.Response(200, json=_DOC)

    _mock_client(handler).insert_text("hello", acl_readers=[])


def test_insert_text_omitted_acl_readers_sends_no_param():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "acl_readers" not in request.url.params
        return httpx.Response(200, json=_DOC)

    _mock_client(handler).insert_text("hello")


def test_update_sends_acl_readers(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("updated")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert request.url.params["acl_readers"] == "user:bob"
        return httpx.Response(200, json=_DOC)

    _mock_client(handler).update("d1", p, acl_readers=["user:bob"])


def test_insert_async_sends_acl_readers(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("queued")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/documents/async"
        assert request.url.params["acl_readers"] == "user:alice"
        return httpx.Response(200, json={"job_id": "j1", "status": "queued", "poll_url": "/x"})

    _mock_client(handler).insert_async(p, acl_readers=["user:alice"])


def test_ingest_files_applies_acl_readers_to_every_file(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("aaa")
    b = tmp_path / "b.txt"
    b.write_text("bbb")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params.get("acl_readers"))
        return httpx.Response(200, json=_DOC)

    results = _mock_client(handler).ingest_files([a, b], acl_readers=["group:eng"])
    assert [r.status for r in results] == ["ingested", "ingested"]
    assert seen == ["group:eng", "group:eng"]


def test_batch_insert_item_carries_acl_readers():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        docs = body["documents"]
        assert docs[0]["acl_readers"] == "user:alice,group:eng"
        assert docs[1]["acl_readers"] == ""  # explicit quarantine survives
        assert "acl_readers" not in docs[2]  # omitted stays omitted
        return httpx.Response(200, json={"results": [_DOC, _DOC, _DOC]})

    _mock_client(handler).batch_insert(
        [
            BatchInsertItem(
                filename="a.txt", content="a", acl_readers=["user:alice", "group:eng"]
            ),
            BatchInsertItem(filename="b.txt", content="b", acl_readers=[]),
            BatchInsertItem(filename="c.txt", content="c"),
        ]
    )


def _no_http_client() -> AetherClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call expected")

    return _mock_client(handler)


def test_acl_readers_label_with_comma_is_rejected_client_side():
    # A comma inside one label would silently split it into several on the
    # CSV wire — widening the ACL (e.g. smuggling in an extra group grant).
    c = _no_http_client()
    with pytest.raises(ValueError, match="comma"):
        c.insert_text("hello", acl_readers=["user:bob,group:everyone"])
    with pytest.raises(ValueError, match="comma"):
        c.batch_insert(
            [BatchInsertItem(filename="a.txt", content="a", acl_readers=["a,b"])]
        )


def test_acl_readers_blank_label_is_rejected_client_side():
    # [""] would encode identically to [] — an accidental quarantine.
    c = _no_http_client()
    with pytest.raises(ValueError, match="empty"):
        c.insert_text("hello", acl_readers=[""])
    with pytest.raises(ValueError, match="empty"):
        c.insert_text("hello", acl_readers=["user:alice", "   "])


def test_insert_with_embeddings_sends_acl_readers_json_array():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/documents/embed"
        body = json.loads(request.content)
        assert body["acl_readers"] == ["user:alice", "group:eng"]
        return httpx.Response(200, json=_DOC)

    _mock_client(handler).insert_with_embeddings(
        "hello", embedding=[0.1, 0.2], acl_readers=["user:alice", "group:eng"]
    )


def test_insert_with_embeddings_acl_readers_empty_and_omitted_are_distinct():
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body.get("acl_readers", "ABSENT"))
        return httpx.Response(200, json=_DOC)

    c = _mock_client(handler)
    c.insert_with_embeddings("hello", embedding=[0.1], acl_readers=[])
    c.insert_with_embeddings("hello", embedding=[0.1])
    assert seen == [[], "ABSENT"]


# ── as_principal scoping ─────────────────────────────────────────────

def test_as_principal_injects_assertion_on_reads():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["acting_principal"] == "user:alice"
        assert request.url.params["acting_groups"] == "group:eng,group:board"
        return httpx.Response(200, json={"query": "q", "results": []})

    scoped = _mock_client(handler).as_principal(
        "user:alice", groups=["group:eng", "group:board"]
    )
    scoped.search("q")


def test_as_principal_without_groups_sends_no_groups_param():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["acting_principal"] == "user:alice"
        assert "acting_groups" not in request.url.params
        return httpx.Response(200, json=_DOC)

    _mock_client(handler).as_principal("user:alice").get("d1")


def test_as_principal_applies_to_writes_too():
    # The assertion rides on every request through the handle, so the
    # access-audit actor is consistent for reads AND writes.
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["acting_principal"] == "user:alice"
        return httpx.Response(200, json=_DOC)

    _mock_client(handler).as_principal("user:alice").insert_text("hello")


def test_as_principal_composes_with_partition():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["partition"] == "client-a"
        assert request.url.params["acting_principal"] == "user:alice"
        return httpx.Response(200, json={"query": "q", "results": []})

    _mock_client(handler).partition("client-a").as_principal("user:alice").search("q")


def test_as_principal_rescope_is_last_wins():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["acting_principal"] == "user:b"
        return httpx.Response(200, json={"query": "q", "results": []})

    _mock_client(handler).as_principal("user:a").as_principal("user:b").search("q")


def test_as_principal_leaves_base_client_unscoped():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "acting_principal" not in request.url.params
        return httpx.Response(200, json={"query": "q", "results": []})

    base = _mock_client(handler)
    base.as_principal("user:alice")  # scoped clone unused
    base.search("q")


def test_as_principal_rejects_empty_principal():
    client = _mock_client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        client.as_principal("")
    with pytest.raises(ValueError):
        client.as_principal("   ")


def test_as_principal_drops_blank_groups():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["acting_groups"] == "group:eng"
        return httpx.Response(200, json=_DOC)

    _mock_client(handler).as_principal("user:alice", groups=["", " ", "group:eng"]).get("d1")


def test_as_principal_group_with_comma_is_rejected():
    # Groups are comma-joined on the wire; a comma inside one label would
    # silently widen the asserted read scope with extra groups.
    with pytest.raises(ValueError, match="comma"):
        _no_http_client().as_principal("user:alice", groups=["group:a,group:admin"])


def test_as_principal_url_encodes_labels():
    def handler(request: httpx.Request) -> httpx.Response:
        # httpx decodes params; the raw query must be percent-encoded.
        assert b"acting_principal=user%3Aal%20ice" in request.url.query
        return httpx.Response(200, json=_DOC)

    _mock_client(handler).as_principal("user:al ice").get("d1")


# ── audit.access ─────────────────────────────────────────────────────

_ACCESS_BODY = {
    "records": [
        {
            "at": "2026-07-01T00:00:00Z",
            "actor": "user:bob",
            "action": "denied",
            "resource": "document:d1",
            "outcome": "denied",
            "source": "access",
        },
        {
            "at": "2026-07-01T00:00:01Z",
            "actor": "user:alice",
            "action": "read",
            "resource": "document:d1",
            "outcome": "ok",
            "source": "access",
        },
    ],
    "total": 7,
}


def test_audit_access_parses_records_and_total():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/audit/access"
        return httpx.Response(200, json=_ACCESS_BODY)

    page = _mock_client(handler).audit.access()
    assert isinstance(page, AccessAuditPage)
    assert page.total == 7
    assert len(page) == 2
    assert page[0].actor == "user:bob"
    assert page[0].action == "denied"
    assert page[0].outcome == "denied"
    assert page[0].source == "access"
    assert page[0].proof is None
    assert page[1].outcome == "ok"


def test_audit_access_sends_filters():
    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        assert p["actor"] == "user:alice"
        assert p["resource"] == "document:d1"
        assert p["action"] == "read"
        assert p["since"] == "2026-07-01T00:00:00Z"
        assert p["until"] == "2026-07-02T00:00:00Z"
        assert p["limit"] == "10"
        assert p["offset"] == "20"
        return httpx.Response(200, json={"records": [], "total": 0})

    page = _mock_client(handler).audit.access(
        actor="user:alice",
        resource="document:d1",
        action="read",
        since="2026-07-01T00:00:00Z",
        until="2026-07-02T00:00:00Z",
        limit=10,
        offset=20,
    )
    assert page.total == 0
    assert list(page) == []


def test_audit_access_carries_principal_scope():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["acting_principal"] == "user:alice"
        return httpx.Response(200, json={"records": [], "total": 0})

    _mock_client(handler).as_principal("user:alice").audit.access()


# ── pinned-key mismatch typed error ──────────────────────────────────

def test_principal_pin_mismatch_maps_to_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "error": "This API key is pinned to a principal and cannot assert a different acting_principal",
                "code": "principal_pin_mismatch",
            },
        )

    client = _mock_client(handler).as_principal("user:mallory")
    with pytest.raises(PrincipalPinMismatchError) as exc_info:
        client.search("q")
    assert exc_info.value.status_code == 403
    assert exc_info.value.error_code == "principal_pin_mismatch"
    assert exc_info.value.is_retryable is False


def test_plain_403_stays_base_api_error():
    from aether import AetherApiError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "forbidden"})

    with pytest.raises(AetherApiError) as exc_info:
        _mock_client(handler).search("q")
    assert not isinstance(exc_info.value, PrincipalPinMismatchError)


# ── async client mirrors ─────────────────────────────────────────────

async def test_async_as_principal_and_audit_access():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/audit/access":
            assert request.url.params["acting_principal"] == "user:alice"
            return httpx.Response(200, json=_ACCESS_BODY)
        assert request.url.params["acting_principal"] == "user:alice"
        assert request.url.params["acl_readers"] == "user:alice"
        return httpx.Response(200, json=_DOC)

    client = _mock_async_client(handler).as_principal("user:alice")
    await client.insert_text("hello", acl_readers=["user:alice"])
    page = await client.audit.access()
    assert page.total == 7
    assert page[0].source == "access"
    await client.close()


async def test_async_principal_pin_mismatch_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"error": "pinned", "code": "principal_pin_mismatch"}
        )

    client = _mock_async_client(handler).as_principal("user:mallory")
    with pytest.raises(PrincipalPinMismatchError):
        await client.search("q")
    await client.close()
