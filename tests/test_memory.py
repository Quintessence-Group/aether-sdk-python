"""Contract test for the ``Memory`` / ``AsyncMemory`` facade (AET-145).

Mocked at the same transport layer as the existing client tests
(``client._client.request``) with the *real* raw client underneath, constructed
via the DI path (``Memory(entity_id, client=...)``).

This pins the facade to the **shipped 0.3.x search surface**: hits carry a
calibrated ``score`` (0–100, higher = better) and a ``passage`` — there is no
``distance`` field and ``retrieve`` fetches each matched document's text with a
follow-up ``GET /documents/{id}/download`` (search no longer inlines content).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aether import (
    AetherClient,
    AsyncAetherClient,
    AsyncMemory,
    CreditExhaustedError,
    Memory,
    MemoryItem,
)

FIXED_NOW = datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone.utc)


# ── transport stubs ──────────────────────────────────────────────────


def _resp(*, json_data=None, content=None, status_code=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.reason_phrase = "OK" if resp.is_success else "Error"
    resp.headers = {}
    if json_data is not None:
        resp.json.return_value = json_data
    if content is not None:
        resp.content = content
    return resp


class Transport:
    """Records every ``request(method, url, **kwargs)`` and dispatches a scripted
    response keyed by ``(method, path)``. A list value pops per call; a scalar is
    reused."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def _lookup(self, method, url):
        # url may carry a query string (inserts build it into the path); match on
        # the path prefix so routes stay readable.
        path = url.split("?", 1)[0]
        for key in ((method, url), (method, path)):
            if key in self.routes:
                value = self.routes[key]
                return value.pop(0) if isinstance(value, list) else value
        raise AssertionError(f"unexpected request: {method} {url}")

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._lookup(method, url)

    def methods(self):
        return [c[0] for c in self.calls]

    def paths(self):
        return [c[1].split("?", 1)[0] for c in self.calls]

    def calls_to(self, method, path):
        return [c for c in self.calls if c[0] == method and c[1].split("?", 1)[0] == path]


def _sync_memory(transport, entity_id="user-42", **kw):
    client = AetherClient(base_url="http://localhost:9000", api_key="k", max_retries=0)
    patch.object(client._client, "request", side_effect=transport).start()
    return Memory(entity_id, client=client, now=lambda: FIXED_NOW, **kw)


def _async_transport(transport):
    async def _call(method, url, **kwargs):
        return transport(method, url, **kwargs)

    return _call


def _async_memory(transport, entity_id="user-42", **kw):
    client = AsyncAetherClient(base_url="http://localhost:9000", api_key="k", max_retries=0)
    patch.object(
        client._client, "request", new=AsyncMock(side_effect=_async_transport(transport))
    ).start()
    return AsyncMemory(entity_id, client=client, now=lambda: FIXED_NOW, **kw)


@pytest.fixture(autouse=True)
def _stop_patches():
    yield
    patch.stopall()


def _insert_resp(doc_id="doc-new", created_at="2026-06-15T00:00:00Z", entity_id="user-42"):
    return _resp(json_data={
        "doc_id": doc_id, "cid": "cid-1", "chunks": 1, "vectors": 1,
        "version": 1, "created_at": created_at, "entity_id": entity_id,
    })


def _search_resp(results):
    # 0.3.x wire shape: score + passage, no distance, no content.
    return _resp(json_data={"query": "q", "results": results})


def _hit(doc_id, score, passage="p"):
    return {"doc_id": doc_id, "score": score, "content_type": "text/plain", "passage": passage}


def _download_resp(text):
    return _resp(content=text.encode("utf-8"))


def _doc_resp(doc_id, created_at):
    return _resp(json_data={
        "doc_id": doc_id, "cid": "c", "content_type": "text/plain",
        "size_bytes": 1, "version": 1, "created_at": created_at, "entity_id": "user-42",
    })


def _list_resp(documents):
    return _resp(json_data={"documents": documents, "total": len(documents), "has_more": False})


# ── scoping ──────────────────────────────────────────────────────────


class TestScoping:
    def test_remember_sends_entity_id_field(self):
        t = Transport({("POST", "/documents"): _insert_resp()})
        _sync_memory(t).remember("hello")
        method, url, _ = t.calls[0]
        assert method == "POST"
        assert "entity_id=user-42" in url

    def test_recall_sends_entity_id_filter(self):
        t = Transport({("GET", "/search"): _search_resp([])})
        _sync_memory(t).recall("anxiety")
        assert t.calls[0][2]["params"]["entity_id"] == "user-42"

    def test_list_sends_entity_id_filter(self):
        t = Transport({("GET", "/documents"): _list_resp([])})
        _sync_memory(t).list()
        assert t.calls[0][2]["params"]["entity_id"] == "user-42"


# ── remember round-trip ──────────────────────────────────────────────


class TestRememberRoundTrip:
    def test_returns_memory_item(self):
        t = Transport({("POST", "/documents"): _insert_resp(doc_id="doc-7", created_at="2026-06-15T09:30:00Z")})
        item = _sync_memory(t).remember("anxious about flying")
        assert isinstance(item, MemoryItem)
        assert (item.id, item.text, item.entity_id, item.score) == (
            "doc-7", "anxious about flying", "user-42", None)
        assert item.created_at == "2026-06-15T09:30:00Z"

    def test_empty_text_is_client_side_error(self):
        t = Transport({})
        with pytest.raises(ValueError):
            _sync_memory(t).remember("   ")
        assert t.calls == []


# ── metadata → tags (write-only) ─────────────────────────────────────


class TestMetadataTags:
    def test_metadata_encoded_as_tags(self):
        t = Transport({("POST", "/documents"): _insert_resp()})
        _sync_memory(t).remember("breathing helps", {"topic": "anxiety"})
        assert "tags=topic%3Aanxiety" in t.calls[0][1]

    def test_multiple_metadata_sorted_by_key(self):
        t = Transport({("POST", "/documents"): _insert_resp()})
        _sync_memory(t).remember("x", {"topic": "anxiety", "score": "5", "active": "yes"})
        assert "tags=active%3Ayes%2Cscore%3A5%2Ctopic%3Aanxiety" in t.calls[0][1]

    def test_prefix_keys_sorted_by_key_not_tag(self):
        t = Transport({("POST", "/documents"): _insert_resp()})
        _sync_memory(t).remember("x", {"a0": "w", "a": "v"})
        assert "tags=a%3Av%2Ca0%3Aw" in t.calls[0][1]

    def test_value_with_first_colon_split(self):
        t = Transport({("POST", "/documents"): _insert_resp()})
        _sync_memory(t).remember("x", {"time": "12:30"})
        assert "tags=time%3A12%3A30" in t.calls[0][1]

    @pytest.mark.parametrize("metadata", [
        {"topic": "a,b"}, {"": "v"}, {"a,b": "v"}, {"a:b": "v"}])
    def test_bad_metadata_raises_no_http(self, metadata):
        t = Transport({})
        with pytest.raises(ValueError):
            _sync_memory(t).remember("x", metadata)
        assert t.calls == []


# ── recall (default: recency_weight=0) ───────────────────────────────


class TestRecallDefault:
    def test_search_then_download_server_order(self):
        t = Transport({
            ("GET", "/search"): _search_resp([_hit("d1", 95), _hit("d2", 70)]),
            ("GET", "/documents/d1/download"): _download_resp("first"),
            ("GET", "/documents/d2/download"): _download_resp("second"),
        })
        items = _sync_memory(t).recall("query", k=5)
        # one search + one download per unique hit; NO metadata get() calls
        assert len(t.calls_to("GET", "/search")) == 1
        assert len(t.calls_to("GET", "/documents/d1/download")) == 1
        assert len(t.calls_to("GET", "/documents/d2/download")) == 1
        assert [i.id for i in items] == ["d1", "d2"]
        assert [i.text for i in items] == ["first", "second"]
        assert all(i.created_at is None for i in items)
        # score normalized from the 0–100 wire score; higher = better
        assert items[0].score == pytest.approx(0.95)
        assert items[1].score == pytest.approx(0.70)
        # no removed include_content flag; entity filter + k forwarded
        params = t.calls[0][2]["params"]
        assert "include_content" not in params
        assert params["entity_id"] == "user-42"
        assert params["k"] == 5

    def test_empty_query_is_client_side_error(self):
        t = Transport({})
        with pytest.raises(ValueError):
            _sync_memory(t).recall("   ")
        assert t.calls == []

    def test_k_below_one_is_client_side_error(self):
        t = Transport({})
        with pytest.raises(ValueError):
            _sync_memory(t).recall("query", k=0)
        assert t.calls == []


# ── recall (recency_weight>0: blended re-ranking) ────────────────────
#
# recency_weight=0.5, half_life_days=30, now=2026-06-15. similarity = score/100,
# recency = 0.5 ** (age_days / 30). blended = 0.5*sim + 0.5*recency:
#   docA score=90 age=0d   -> 0.5*0.90 + 0.5*1.0 = 0.95
#   docB score=80 age=30d  -> 0.5*0.80 + 0.5*0.5 = 0.65
#   docC score=100 created=null (recency 0) -> 0.5*1.00 + 0.5*0.0 = 0.50
# Pure score order is [docC, docA, docB]; recency reorders to [docA, docB, docC].
class TestRecallRecency:
    def _transport(self):
        return Transport({
            ("GET", "/search"): _search_resp([_hit("docA", 90), _hit("docB", 80), _hit("docC", 100)]),
            ("GET", "/documents/docA/download"): _download_resp("A"),
            ("GET", "/documents/docB/download"): _download_resp("B"),
            ("GET", "/documents/docC/download"): _download_resp("C"),
            ("GET", "/documents/docA"): _doc_resp("docA", "2026-06-15T00:00:00Z"),
            ("GET", "/documents/docB"): _doc_resp("docB", "2026-05-16T00:00:00Z"),
            ("GET", "/documents/docC"): _doc_resp("docC", None),
        })

    def test_blended_reorder(self):
        items = _sync_memory(self._transport()).recall("q", k=5, recency_weight=0.5)
        assert [i.id for i in items] == ["docA", "docB", "docC"]
        assert [i.score for i in items] == pytest.approx([0.95, 0.65, 0.50])
        # recency mode resolves created_at, so it is populated
        assert items[0].created_at == "2026-06-15T00:00:00Z"

    def test_top_k_truncation(self):
        items = _sync_memory(self._transport()).recall("q", k=2, recency_weight=0.5)
        assert [i.id for i in items] == ["docA", "docB"]


# ── list (chronological) ─────────────────────────────────────────────


class TestList:
    def test_newest_first_text_downloaded_score_none(self):
        t = Transport({
            ("GET", "/documents"): _list_resp([
                {"doc_id": "m1", "content_type": "text/plain", "created_at": "2026-06-15T00:00:00Z"},
                {"doc_id": "m2", "content_type": "text/plain", "created_at": "2026-06-01T00:00:00Z"},
            ]),
            ("GET", "/documents/m1/download"): _download_resp("newest"),
            ("GET", "/documents/m2/download"): _download_resp("older"),
        })
        items = _sync_memory(t).list()
        assert [i.id for i in items] == ["m1", "m2"]
        assert [i.text for i in items] == ["newest", "older"]
        assert all(i.score is None for i in items)


# ── forget ───────────────────────────────────────────────────────────


class TestForget:
    def test_forget_issues_one_delete(self):
        t = Transport({("DELETE", "/documents/doc-x"): _resp(json_data={})})
        _sync_memory(t).forget("doc-x")
        assert t.calls_to("DELETE", "/documents/doc-x")

    def test_forget_empty_id_raises(self):
        t = Transport({})
        with pytest.raises(ValueError):
            _sync_memory(t).forget("")
        assert t.calls == []

    def test_forget_all_deletes_every_listed_and_returns_count(self):
        t = Transport({
            ("GET", "/documents"): [
                _list_resp([{"doc_id": "a", "content_type": "text/plain"},
                            {"doc_id": "b", "content_type": "text/plain"}]),
                _list_resp([]),
            ],
            ("DELETE", "/documents/a"): _resp(json_data={}),
            ("DELETE", "/documents/b"): _resp(json_data={}),
        })
        assert _sync_memory(t).forget_all() == 2


# ── error passthrough ────────────────────────────────────────────────


class TestErrorPassthrough:
    def test_credit_exhausted_surfaces_typed_error(self):
        t = Transport({("POST", "/documents"): _resp(
            json_data={"error": "out of credit", "code": "credit_exhausted"},
            status_code=402)})
        with pytest.raises(CreditExhaustedError):
            _sync_memory(t).remember("x")


# ── invalid construction ─────────────────────────────────────────────


class TestInvalidConstruction:
    @pytest.mark.parametrize("entity_id", ["", "   ", "\t"])
    def test_empty_or_whitespace_entity_id_raises(self, entity_id):
        with pytest.raises(ValueError):
            Memory(entity_id, client=MagicMock())

    def test_oversized_entity_id_raises(self):
        with pytest.raises(ValueError):
            Memory("x" * 257, client=MagicMock())

    def test_max_length_entity_id_ok(self):
        Memory("x" * 256, client=MagicMock())

    def test_non_positive_half_life_raises(self):
        with pytest.raises(ValueError):
            Memory("u", client=MagicMock(), half_life_days=0)


# ── async parity ─────────────────────────────────────────────────────


class TestAsyncParity:
    @pytest.mark.asyncio
    async def test_remember_round_trip(self):
        t = Transport({("POST", "/documents"): _insert_resp(doc_id="d9")})
        item = await _async_memory(t).remember("hello")
        assert item.id == "d9" and item.score is None

    @pytest.mark.asyncio
    async def test_recall_default(self):
        t = Transport({
            ("GET", "/search"): _search_resp([_hit("d1", 88)]),
            ("GET", "/documents/d1/download"): _download_resp("text"),
        })
        items = await _async_memory(t).recall("q", k=3)
        assert [i.id for i in items] == ["d1"]
        assert items[0].text == "text"
        assert items[0].score == pytest.approx(0.88)
