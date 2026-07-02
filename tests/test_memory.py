"""Contract test for the ``Memory`` / ``AsyncMemory`` facade (MEMORY_CONTRACT.md §8).

Mocked at the **same transport layer as the existing client tests** —
``client._client.request`` (the underlying httpx ``Client``/``AsyncClient``) —
with the *real* raw client running underneath, constructed via the DI path
(``Memory(entity_id, client=...)``). This asserts the nine observable behaviors
from ``docs/MEMORY_CONTRACT.md`` §8 cases 1–9.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aether import (
    AetherApiError,
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
    """Mock httpx.Response that passes the raw client's retry/status checks."""
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
    """Records every ``request(method, url, **kwargs)`` and dispatches a
    scripted response keyed by ``(method, path)``. Per-key responses pop off a
    queue when a list is supplied, else the single value is reused.
    """

    def __init__(self, routes):
        # routes: {(method, path): response | [responses...]}
        self.routes = routes
        self.calls = []  # list of (method, url, kwargs)

    def _lookup(self, method, url):
        key = (method, url)
        if key not in self.routes:
            raise AssertionError(f"unexpected request: {method} {url}")
        value = self.routes[key]
        if isinstance(value, list):
            return value.pop(0)
        return value

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._lookup(method, url)

    # request-introspection helpers
    def methods(self):
        return [c[0] for c in self.calls]

    def calls_to(self, method, path):
        return [c for c in self.calls if c[0] == method and c[1] == path]


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
        "doc_id": doc_id,
        "cid": "cid-1",
        "chunks": 1,
        "vectors": 1,
        "version": 1,
        "created_at": created_at,
        "entity_id": entity_id,
    })


def _search_resp(results):
    return _resp(json_data={"query": "q", "results": results})


def _list_resp(documents):
    return _resp(json_data={"documents": documents, "total": len(documents), "has_more": False})


# ── §8.1 scoping ─────────────────────────────────────────────────────


class TestScoping:
    def test_remember_sends_entity_id_field(self):
        t = Transport({("POST", "/v1/documents?filename=text.txt&content_type=text%2Fplain&entity_id=user-42"): _insert_resp()})
        mem = _sync_memory(t)
        mem.remember("hello")
        method, url, _ = t.calls[0]
        assert method == "POST"
        assert "entity_id=user-42" in url

    def test_recall_sends_entity_id_filter(self):
        t = Transport({("GET", "/v1/search"): _search_resp([])})
        mem = _sync_memory(t)
        mem.recall("anxiety")
        params = t.calls[0][2]["params"]
        assert params["entity_id"] == "user-42"

    def test_list_sends_entity_id_filter(self):
        t = Transport({("GET", "/v1/documents"): _list_resp([])})
        mem = _sync_memory(t)
        mem.list()
        params = t.calls[0][2]["params"]
        assert params["entity_id"] == "user-42"


# ── §8.2 remember round-trip ─────────────────────────────────────────


class TestRememberRoundTrip:
    def test_returns_memory_item_with_id_and_created_at(self):
        t = Transport({("POST", "/v1/documents?filename=text.txt&content_type=text%2Fplain&entity_id=user-42"): _insert_resp(doc_id="doc-7", created_at="2026-06-15T09:30:00Z")})
        mem = _sync_memory(t)
        item = mem.remember("anxious about flying")
        assert isinstance(item, MemoryItem)
        assert item.id == "doc-7"
        assert item.created_at == "2026-06-15T09:30:00Z"
        assert item.text == "anxious about flying"
        assert item.entity_id == "user-42"
        assert item.score is None

    def test_empty_text_is_client_side_error(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.remember("   ")
        assert t.calls == []  # no HTTP call


# ── §8.3 metadata → tags ─────────────────────────────────────────────


class TestMetadataTags:
    def test_metadata_encoded_as_tags(self):
        expected_url = (
            "/v1/documents?filename=text.txt&content_type=text%2Fplain"
            "&tags=topic%3Aanxiety&entity_id=user-42"
            "&metadata=%7B%22topic%22%3A%22anxiety%22%7D"
        )
        t = Transport({("POST", expected_url): _insert_resp()})
        mem = _sync_memory(t)
        mem.remember("breathing helps", {"topic": "anxiety"})
        url = t.calls[0][1]
        assert "tags=topic%3Aanxiety" in url  # 'topic:anxiety' url-encoded
        assert "metadata=%7B%22topic%22%3A%22anxiety%22%7D" in url

    def test_multiple_metadata_sorted_by_key(self):
        # string-only values; first ':' is the separator; tags emitted SORTED BY
        # KEY ascending so the wire string is byte-identical across languages.
        # sorted keys: active, score, topic
        expected_url = (
            "/v1/documents?filename=text.txt&content_type=text%2Fplain"
            "&tags=active%3Ayes%2Cscore%3A5%2Ctopic%3Aanxiety&entity_id=user-42"
            "&metadata=%7B%22topic%22%3A%22anxiety%22%2C%22score%22%3A%225%22%2C%22active%22%3A%22yes%22%7D"
        )
        t = Transport({("POST", expected_url): _insert_resp()})
        mem = _sync_memory(t)
        # insertion order deliberately NOT sorted, to prove the encoder sorts
        mem.remember("x", {"topic": "anxiety", "score": "5", "active": "yes"})
        url = t.calls[0][1]
        # byte-identical sorted wire order
        assert "tags=active%3Ayes%2Cscore%3A5%2Ctopic%3Aanxiety" in url

    def test_metadata_prefix_keys_sorted_by_key(self):
        # Regression: keys are sorted, NOT the assembled "key:value" tags. With a
        # prefix key, key-sort gives a:v,a0:w; a tag-string sort would give
        # a0:w,a:v ('0' 0x30 < ':' 0x3A). Must match the other three SDKs.
        expected_url = (
            "/v1/documents?filename=text.txt&content_type=text%2Fplain"
            "&tags=a%3Av%2Ca0%3Aw&entity_id=user-42"
            "&metadata=%7B%22a0%22%3A%22w%22%2C%22a%22%3A%22v%22%7D"
        )
        t = Transport({("POST", expected_url): _insert_resp()})
        mem = _sync_memory(t)
        mem.remember("x", {"a0": "w", "a": "v"})
        assert "tags=a%3Av%2Ca0%3Aw" in t.calls[0][1]

    def test_value_with_first_colon_split(self):
        # value may contain ':'; only the FIRST ':' separates key from value
        expected_url = (
            "/v1/documents?filename=text.txt&content_type=text%2Fplain"
            "&tags=time%3A12%3A30&entity_id=user-42"
            "&metadata=%7B%22time%22%3A%2212%3A30%22%7D"
        )
        t = Transport({("POST", expected_url): _insert_resp()})
        mem = _sync_memory(t)
        mem.remember("x", {"time": "12:30"})
        assert "tags=time%3A12%3A30" in t.calls[0][1]  # 'time:12:30'

    def test_comma_in_value_raises_no_http(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.remember("x", {"topic": "a,b"})
        assert t.calls == []  # no HTTP call made

    def test_empty_key_raises_no_http(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.remember("x", {"": "value"})
        assert t.calls == []  # no HTTP call made

    def test_comma_in_key_raises_no_http(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.remember("x", {"a,b": "value"})
        assert t.calls == []  # no HTTP call made

    def test_colon_in_key_raises_no_http(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.remember("x", {"a:b": "value"})
        assert t.calls == []  # no HTTP call made


# ── §8.4 recall default (recency_weight=0) ───────────────────────────


class TestRecallDefault:
    def test_single_call_null_created_at_server_order(self):
        results = [
            {"doc_id": "d1", "score": 90, "content_type": "text/plain", "content": "first"},
            {"doc_id": "d2", "score": 60, "content_type": "text/plain", "content": "second"},
        ]
        t = Transport({("GET", "/v1/search"): _search_resp(results)})
        mem = _sync_memory(t)
        items = mem.recall("query", k=5)
        # exactly one retrieve (search) call
        assert t.methods() == ["GET"]
        assert len(t.calls_to("GET", "/v1/search")) == 1
        # created_at null on every item; order == server order
        assert [i.id for i in items] == ["d1", "d2"]
        assert all(i.created_at is None for i in items)
        # score normalized from the 0-100 wire score; higher = better
        assert items[0].score == pytest.approx(0.90)
        assert items[1].score == pytest.approx(0.60)
        # include_content forwarded, entity_id filter present
        params = t.calls[0][2]["params"]
        assert params["include_content"] == "true"
        assert params["entity_id"] == "user-42"
        assert params["k"] == 5

    def test_empty_query_is_client_side_error(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.recall("   ")
        assert t.calls == []  # no HTTP call made

    def test_k_below_one_is_client_side_error(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.recall("query", k=0)
        assert t.calls == []  # no HTTP call made


# ── §8.5 recall recency (golden ordering) ────────────────────────────

# Canonical recency golden vector (MEMORY_CONTRACT.md §8.1) — IDENTICAL inputs
# and asserted order in all four SDKs. Params: recency_weight=0.5,
# half_life_days=30, injected now=2026-06-15T00:00:00Z, k=5. retrieve returns the
# candidates in server order (descending score); get(doc_id) yields each
# created_at (doc-e: null). blended = 0.5*(score/100) + 0.5*0.5^(age/30):
#   doc-e score=95 created null        blended=0.475000  (best score but recency 0)
#   doc-a score=90 created 2026-01-01  blended=0.461049  (165 days old)
#   doc-b score=80 created 2026-06-14  blended=0.888580  (1 day — freshest wins)
#   doc-c score=70 created 2026-06-10  blended=0.795449  (5 days)
#   doc-d score=60 created 2026-05-16  blended=0.550000  (30 days = one half-life)
# sorted (blended DESC, score DESC, doc_id ASC) -> [doc-b, doc-c, doc-d, doc-e, doc-a]
GOLDEN_RESULTS = [
    {"doc_id": "doc-e", "score": 95, "content_type": "text/plain", "content": "E"},
    {"doc_id": "doc-a", "score": 90, "content_type": "text/plain", "content": "A"},
    {"doc_id": "doc-b", "score": 80, "content_type": "text/plain", "content": "B"},
    {"doc_id": "doc-c", "score": 70, "content_type": "text/plain", "content": "C"},
    {"doc_id": "doc-d", "score": 60, "content_type": "text/plain", "content": "D"},
]
GOLDEN_CREATED = {
    "doc-e": None,
    "doc-a": "2026-01-01T00:00:00Z",
    "doc-b": "2026-06-14T00:00:00Z",
    "doc-c": "2026-06-10T00:00:00Z",
    "doc-d": "2026-05-16T00:00:00Z",
}
GOLDEN_ORDER = ["doc-b", "doc-c", "doc-d", "doc-e", "doc-a"]
GOLDEN_BLENDED = {
    "doc-b": 0.888580,
    "doc-c": 0.795449,
    "doc-d": 0.550000,
    "doc-e": 0.475000,
    "doc-a": 0.461049,
}


def _doc_get_resp(doc_id):
    return _resp(json_data={
        "doc_id": doc_id,
        "cid": "c",
        "content_type": "text/plain",
        "size_bytes": 1,
        "version": 1,
        "created_at": GOLDEN_CREATED[doc_id],
        "entity_id": "user-42",
    })


class TestRecallRecency:
    def test_golden_blended_order(self):
        routes = {("GET", "/v1/search"): _search_resp(list(GOLDEN_RESULTS))}
        for doc_id in GOLDEN_CREATED:
            routes[("GET", f"/v1/documents/{doc_id}")] = _doc_get_resp(doc_id)
        t = Transport(routes)
        mem = _sync_memory(t, half_life_days=30.0)
        items = mem.recall("query", k=5, recency_weight=0.5)
        # §8.1 canonical order
        assert [i.id for i in items] == GOLDEN_ORDER
        # blended scores match the shared vector within 1e-6
        for item in items:
            assert item.score == pytest.approx(GOLDEN_BLENDED[item.id], abs=1e-6)
        # created_at populated in recency mode
        assert items[0].created_at == "2026-06-14T00:00:00Z"
        # doc-e has a null created_at (recency 0 sank it to rank 4)
        assert items[3].id == "doc-e"
        assert items[3].created_at is None
        # blended score is monotonic non-increasing (sorted DESC)
        scores = [i.score for i in items]
        assert scores == sorted(scores, reverse=True)
        # overfetch: k*4 = 20 requested on the search call
        assert t.calls_to("GET", "/v1/search")[0][2]["params"]["k"] == 20

    def test_top_k_truncation(self):
        routes = {("GET", "/v1/search"): _search_resp(list(GOLDEN_RESULTS))}
        for doc_id in GOLDEN_CREATED:
            routes[("GET", f"/v1/documents/{doc_id}")] = _doc_get_resp(doc_id)
        t = Transport(routes)
        mem = _sync_memory(t, half_life_days=30.0)
        items = mem.recall("query", k=2, recency_weight=0.5)
        assert [i.id for i in items] == GOLDEN_ORDER[:2]


# ── §8.6 list ────────────────────────────────────────────────────────


class TestList:
    def test_newest_first_text_downloaded(self):
        # server returns filtered listing newest-first already
        docs = [
            {"doc_id": "n1", "created_at": "2026-06-14T00:00:00Z", "entity_id": "user-42"},
            {"doc_id": "n2", "created_at": "2026-06-10T00:00:00Z", "entity_id": "user-42"},
        ]
        routes = {
            ("GET", "/v1/documents"): _list_resp(docs),
            ("GET", "/v1/documents/n1/download"): _resp(content=b"newest text"),
            ("GET", "/v1/documents/n2/download"): _resp(content=b"older text"),
        }
        t = Transport(routes)
        mem = _sync_memory(t)
        items = mem.list()
        assert [i.id for i in items] == ["n1", "n2"]
        assert items[0].text == "newest text"
        assert items[1].text == "older text"
        assert items[0].created_at == "2026-06-14T00:00:00Z"
        assert all(i.score is None for i in items)
        # entity_id filter sent on the listing
        assert t.calls_to("GET", "/v1/documents")[0][2]["params"]["entity_id"] == "user-42"
        # 1 + N: one listing + one download per item
        assert len(t.calls_to("GET", "/v1/documents")) == 1
        assert len(t.calls_to("GET", "/v1/documents/n1/download")) == 1
        assert len(t.calls_to("GET", "/v1/documents/n2/download")) == 1


# ── §8.7 forget / forget_all ─────────────────────────────────────────


class TestForget:
    def test_forget_issues_one_delete(self):
        t = Transport({("DELETE", "/v1/documents/doc-x"): _resp()})
        mem = _sync_memory(t)
        mem.forget("doc-x")
        assert t.methods() == ["DELETE"]
        assert t.calls[0][1] == "/v1/documents/doc-x"

    def test_forget_empty_id_raises(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.forget("")
        assert t.calls == []

    def test_forget_all_deletes_every_listed_and_returns_count(self):
        # first listing returns 2 docs; tombstones excluded -> second listing empty
        docs = [
            {"doc_id": "g1", "created_at": "2026-06-14T00:00:00Z", "entity_id": "user-42"},
            {"doc_id": "g2", "created_at": "2026-06-10T00:00:00Z", "entity_id": "user-42"},
        ]
        routes = {
            ("GET", "/v1/documents"): [_list_resp(docs), _list_resp([])],
            ("DELETE", "/v1/documents/g1"): _resp(),
            ("DELETE", "/v1/documents/g2"): _resp(),
        }
        t = Transport(routes)
        mem = _sync_memory(t)
        count = mem.forget_all()
        assert count == 2
        assert len(t.calls_to("DELETE", "/v1/documents/g1")) == 1
        assert len(t.calls_to("DELETE", "/v1/documents/g2")) == 1
        # paged until the listing was empty (two listing calls)
        assert len(t.calls_to("GET", "/v1/documents")) == 2
        # listing scoped to the entity with limit 1000
        assert t.calls_to("GET", "/v1/documents")[0][2]["params"]["entity_id"] == "user-42"
        assert t.calls_to("GET", "/v1/documents")[0][2]["params"]["limit"] == 1000


# ── §8.8 error passthrough ───────────────────────────────────────────


class TestErrorPassthrough:
    def test_credit_exhausted_surfaces_typed_error(self):
        err = _resp(
            json_data={"error": "out of credit", "code": "credit_exhausted"},
            status_code=402,
        )
        t = Transport({("POST", "/v1/documents?filename=text.txt&content_type=text%2Fplain&entity_id=user-42"): err})
        mem = _sync_memory(t)
        with pytest.raises(CreditExhaustedError) as exc:
            mem.remember("hello")
        assert exc.value.status_code == 402
        assert exc.value.error_code == "credit_exhausted"

    def test_generic_api_error_surfaces_unchanged(self):
        err = _resp(json_data={"error": "boom"}, status_code=400)
        t = Transport({("GET", "/v1/search"): err})
        mem = _sync_memory(t)
        with pytest.raises(AetherApiError) as exc:
            mem.recall("q")
        assert exc.value.status_code == 400
        assert not isinstance(exc.value, CreditExhaustedError)


# ── §8.9 invalid construction ────────────────────────────────────────


class TestInvalidConstruction:
    def test_empty_entity_id_raises(self):
        with pytest.raises(ValueError):
            Memory("")

    @pytest.mark.parametrize("entity_id", ["   ", "\t\n"])
    def test_whitespace_only_entity_id_raises(self, entity_id):
        # Regression: a whitespace-only entity_id must be
        # rejected client-side — the raw layer silently drops it, which would
        # leak across the whole tenant. No HTTP client is constructed because
        # validation runs before any transport setup in the constructor.
        with pytest.raises(ValueError):
            Memory(entity_id)

    @pytest.mark.parametrize("entity_id", ["   ", "\t\n"])
    def test_async_whitespace_only_entity_id_raises(self, entity_id):
        with pytest.raises(ValueError):
            AsyncMemory(entity_id)

    def test_oversized_entity_id_raises(self):
        with pytest.raises(ValueError):
            Memory("x" * 257)

    def test_max_length_entity_id_ok(self):
        # 256 chars is the boundary and must be accepted (DI path, no network)
        client = AetherClient(base_url="http://localhost:9000", api_key="k")
        mem = Memory("x" * 256, client=client)
        assert mem.entity_id == "x" * 256

    def test_async_empty_entity_id_raises(self):
        with pytest.raises(ValueError):
            AsyncMemory("")


# ── Async parity (mirrors the sync cases at the async transport) ──────


class TestAsyncParity:
    async def test_remember_round_trip(self):
        t = Transport({("POST", "/v1/documents?filename=text.txt&content_type=text%2Fplain&entity_id=user-42"): _insert_resp(doc_id="ad1", created_at="2026-06-15T01:00:00Z")})
        mem = _async_memory(t)
        item = await mem.remember("async hello")
        assert item.id == "ad1"
        assert item.created_at == "2026-06-15T01:00:00Z"
        assert item.entity_id == "user-42"

    async def test_recall_default_single_call(self):
        results = [
            {"doc_id": "d1", "score": 90, "content_type": "text/plain", "content": "x"},
        ]
        t = Transport({("GET", "/v1/search"): _search_resp(results)})
        mem = _async_memory(t)
        items = await mem.recall("q")
        assert len(t.calls_to("GET", "/v1/search")) == 1
        assert items[0].created_at is None
        assert items[0].score == pytest.approx(0.90)

    async def test_recall_recency_golden_order(self):
        routes = {("GET", "/v1/search"): _search_resp(list(GOLDEN_RESULTS))}
        for doc_id in GOLDEN_CREATED:
            routes[("GET", f"/v1/documents/{doc_id}")] = _doc_get_resp(doc_id)
        t = Transport(routes)
        mem = _async_memory(t, half_life_days=30.0)
        items = await mem.recall("q", k=5, recency_weight=0.5)
        assert [i.id for i in items] == GOLDEN_ORDER
        for item in items:
            assert item.score == pytest.approx(GOLDEN_BLENDED[item.id], abs=1e-6)

    async def test_list_text_downloaded(self):
        docs = [{"doc_id": "n1", "created_at": "2026-06-14T00:00:00Z", "entity_id": "user-42"}]
        routes = {
            ("GET", "/v1/documents"): _list_resp(docs),
            ("GET", "/v1/documents/n1/download"): _resp(content=b"hi"),
        }
        t = Transport(routes)
        mem = _async_memory(t)
        items = await mem.list()
        assert items[0].text == "hi"

    async def test_forget_all_returns_count(self):
        docs = [{"doc_id": "g1", "created_at": "2026-06-14T00:00:00Z", "entity_id": "user-42"}]
        routes = {
            ("GET", "/v1/documents"): [_list_resp(docs), _list_resp([])],
            ("DELETE", "/v1/documents/g1"): _resp(),
        }
        t = Transport(routes)
        mem = _async_memory(t)
        count = await mem.forget_all()
        assert count == 1

    async def test_error_passthrough(self):
        err = _resp(json_data={"error": "out", "code": "credit_exhausted"}, status_code=402)
        t = Transport({("POST", "/v1/documents?filename=text.txt&content_type=text%2Fplain&entity_id=user-42"): err})
        mem = _async_memory(t)
        with pytest.raises(CreditExhaustedError):
            await mem.remember("hello")

    async def test_empty_query_is_client_side_error(self):
        t = Transport({})
        mem = _async_memory(t)
        with pytest.raises(ValueError):
            await mem.recall("   ")
        assert t.calls == []  # no HTTP call made

    async def test_k_below_one_is_client_side_error(self):
        t = Transport({})
        mem = _async_memory(t)
        with pytest.raises(ValueError):
            await mem.recall("query", k=0)
        assert t.calls == []  # no HTTP call made

    async def test_metadata_validation_no_http(self):
        t = Transport({})
        mem = _async_memory(t)
        with pytest.raises(ValueError):
            await mem.remember("x", {"a,b": "v"})
        assert t.calls == []  # no HTTP call made


# ── extract_facts constructor default + per-call override (contract §3) ──

_INSERT_URL_PLAIN = "/v1/documents?filename=text.txt&content_type=text%2Fplain&entity_id=user-42"
_INSERT_URL_EXTRACT = _INSERT_URL_PLAIN + "&extract_facts=true"


class TestExtractFactsDefault:
    """The constructor flag sets the default for ``remember``'s server-side
    fact extraction; an explicit per-call ``extract=`` overrides it.
    Still exactly one HTTP call — the fact fan-out happens server-side."""

    def test_constructor_flag_sets_default(self):
        t = Transport({("POST", _INSERT_URL_EXTRACT): _insert_resp()})
        mem = _sync_memory(t, extract_facts=True)
        item = mem.remember("fact one. fact two.")
        assert item.text == "fact one. fact two."
        assert len(t.calls) == 1  # single insert; extraction is server-side

    def test_per_call_false_overrides_constructor_true(self):
        t = Transport({("POST", _INSERT_URL_PLAIN): _insert_resp()})
        mem = _sync_memory(t, extract_facts=True)
        mem.remember("fact one. fact two.", extract=False)
        assert len(t.calls) == 1

    def test_per_call_true_overrides_constructor_default_off(self):
        t = Transport({("POST", _INSERT_URL_EXTRACT): _insert_resp()})
        mem = _sync_memory(t)
        mem.remember("fact one. fact two.", extract=True)
        assert len(t.calls) == 1

    def test_default_off_sends_no_extract_flag(self):
        t = Transport({("POST", _INSERT_URL_PLAIN): _insert_resp()})
        mem = _sync_memory(t)
        mem.remember("fact one. fact two.")
        assert len(t.calls) == 1

    async def test_async_constructor_flag_sets_default(self):
        t = Transport({("POST", _INSERT_URL_EXTRACT): _insert_resp()})
        mem = _async_memory(t, extract_facts=True)
        await mem.remember("fact one. fact two.")
        assert len(t.calls) == 1

    async def test_async_per_call_false_overrides_constructor_true(self):
        t = Transport({("POST", _INSERT_URL_PLAIN): _insert_resp()})
        mem = _async_memory(t, extract_facts=True)
        await mem.remember("fact one. fact two.", extract=False)
        assert len(t.calls) == 1


# ── Memory composed on a partition handle ───────────────────


class TestMemoryOnPartitionHandle:
    """A Memory built on a partition-scoped client is automatically scoped to
    BOTH partition and entity — the Memory constructor is unchanged."""

    def test_remember_and_recall_send_partition(self):
        insert_url = (
            "/v1/documents?filename=text.txt&content_type=text%2Fplain"
            "&entity_id=user-42&partition=tenant-x"
        )
        t = Transport({
            ("POST", insert_url): _insert_resp(),  # remember → POST /documents
            ("GET", "/v1/search"): _search_resp([]),  # recall → GET /search
        })
        client = AetherClient(base_url="http://localhost:9000", api_key="k", max_retries=0)
        patch.object(client._client, "request", side_effect=t).start()

        # DI path on a partition handle — no Memory constructor change.
        mem = Memory("user-42", client=client.partition("tenant-x"), now=lambda: FIXED_NOW)

        mem.remember("hello")
        # remember carries partition in the POST /documents URL.
        method, url, _ = t.calls[0]
        assert method == "POST"
        assert "partition=tenant-x" in url
        assert "entity_id=user-42" in url

        mem.recall("anxiety")
        # recall carries partition (and entity) on the search query.
        params = t.calls[1][2]["params"]
        assert params["partition"] == "tenant-x"
        assert params["entity_id"] == "user-42"


# ── Part II — memory graph (MEMORY_CONTRACT.md §14) ───────────────────


def _entity_dict(**over):
    base = {
        "memory_entity_id": "ent-1",
        "entity_id": "user-42",
        "partition": None,
        "entity_type": "person",
        "display_name": "John",
        "aliases": [],
        "attributes": {},
        "created_at": "2026-06-15T00:00:00Z",
        "updated_at": "2026-06-15T00:00:00Z",
    }
    base.update(over)
    return base


def _relationship_dict(**over):
    base = {
        "relationship_id": "rel-1",
        "entity_id": "user-42",
        "partition": None,
        "from_entity_id": "ent-1",
        "to_entity_id": "ent-2",
        "relationship_type": "works_at",
        "attributes": {},
        "valid_from": None,
        "observed_at": "2026-06-15T00:00:00Z",
        "invalid_from": None,
        "created_at": "2026-06-15T00:00:00Z",
        "updated_at": "2026-06-15T00:00:00Z",
    }
    base.update(over)
    return base


def _fact_dict(**over):
    base = {
        "fact_id": "fact-1",
        "entity_id": "user-42",
        "partition": None,
        "subject_type": "owner",
        "subject_id": None,
        "predicate": "favorite_color",
        "value": "blue",
        "cardinality": "single",
        "valid_from": None,
        "observed_at": "2026-06-15T00:00:00Z",
        "invalid_from": None,
        "supersedes_fact_id": None,
        "created_at": "2026-06-15T00:00:00Z",
        "updated_at": "2026-06-15T00:00:00Z",
    }
    base.update(over)
    return base


def _entity_resp(**over):
    return _resp(json_data=_entity_dict(**over))


def _entities_resp(entities):
    return _resp(json_data={"entities": entities, "count": len(entities)})


def _relationship_resp(**over):
    return _resp(json_data=_relationship_dict(**over))


def _relationships_resp(rels):
    return _resp(json_data={"relationships": rels, "count": len(rels)})


def _fact_resp(**over):
    return _resp(json_data=_fact_dict(**over))


def _facts_resp(facts):
    return _resp(json_data={"facts": facts, "count": len(facts)})


def _consolidate_resp(before=3, after=2, retracted=1):
    return _resp(json_data={
        "active_facts_before": before,
        "active_facts_after": after,
        "retracted": retracted,
    })


class TestGraphEntities:
    def test_upsert_entity_round_trip(self):
        t = Transport({("POST", "/v1/memory/entities"): _entity_resp(memory_entity_id="ent-9")})
        mem = _sync_memory(t)
        ent = mem.upsert_entity(
            "person", display_name="John", attributes={"age": 30, "vip": True}
        )
        method, url, kwargs = t.calls[0]
        assert (method, url) == ("POST", "/v1/memory/entities")
        assert kwargs["params"]["entity_id"] == "user-42"
        body = kwargs["json"]
        assert body["entity_type"] == "person"
        assert body["display_name"] == "John"
        assert body["attributes"] == {"age": 30, "vip": True}
        assert "memory_entity_id" not in body  # minted server-side
        from aether import MemoryEntity
        assert isinstance(ent, MemoryEntity)
        assert ent.memory_entity_id == "ent-9"
        assert ent.entity_id == "user-42"

    def test_upsert_entity_with_id_sends_it(self):
        t = Transport({("POST", "/v1/memory/entities"): _entity_resp()})
        mem = _sync_memory(t)
        mem.upsert_entity("person", memory_entity_id="ent-fixed")
        assert t.calls[0][2]["json"]["memory_entity_id"] == "ent-fixed"

    def test_get_entity(self):
        t = Transport({("GET", "/v1/memory/entities/ent-1"): _entity_resp()})
        mem = _sync_memory(t)
        ent = mem.get_entity("ent-1")
        assert t.calls[0][2]["params"]["entity_id"] == "user-42"
        assert ent.memory_entity_id == "ent-1"

    def test_list_entities_filters(self):
        t = Transport({("GET", "/v1/memory/entities"): _entities_resp([_entity_dict(), _entity_dict(memory_entity_id="ent-2")])})
        mem = _sync_memory(t)
        ents = mem.list_entities(entity_type="person", limit=10)
        params = t.calls[0][2]["params"]
        assert params["entity_id"] == "user-42"
        assert params["entity_type"] == "person"
        assert params["limit"] == 10
        assert len(ents) == 2

    def test_list_entities_omits_unset_filters(self):
        t = Transport({("GET", "/v1/memory/entities"): _entities_resp([])})
        mem = _sync_memory(t)
        mem.list_entities()
        params = t.calls[0][2]["params"]
        assert "entity_type" not in params
        assert "limit" not in params


class TestGraphPartition:
    def test_graph_call_carries_partition_and_entity(self):
        t = Transport({("POST", "/v1/memory/entities"): _entity_resp()})
        client = AetherClient(base_url="http://localhost:9000", api_key="k", max_retries=0)
        patch.object(client._client, "request", side_effect=t).start()
        mem = Memory("user-42", client=client.partition("tenant-x"), now=lambda: FIXED_NOW)
        mem.upsert_entity("person")
        params = t.calls[0][2]["params"]
        assert params["entity_id"] == "user-42"
        assert params["partition"] == "tenant-x"

    def test_unscoped_graph_call_omits_partition(self):
        t = Transport({("POST", "/v1/memory/entities"): _entity_resp()})
        mem = _sync_memory(t)
        mem.upsert_entity("person")
        assert "partition" not in t.calls[0][2]["params"]


class TestGraphRelationships:
    def test_relate_round_trip(self):
        t = Transport({("POST", "/v1/memory/relationships"): _relationship_resp(relationship_id="rel-9")})
        mem = _sync_memory(t)
        rel = mem.relate("ent-1", "ent-2", "works_at", valid_from="2026-01-01T00:00:00Z")
        body = t.calls[0][2]["json"]
        assert body["from_entity_id"] == "ent-1"
        assert body["to_entity_id"] == "ent-2"
        assert body["relationship_type"] == "works_at"
        assert body["valid_from"] == "2026-01-01T00:00:00Z"
        assert rel.relationship_id == "rel-9"

    def test_list_relationships_active_filter(self):
        t = Transport({("GET", "/v1/memory/relationships"): _relationships_resp([_relationship_dict()])})
        mem = _sync_memory(t)
        mem.list_relationships(include_inactive=True, as_of="2026-06-01T00:00:00Z", from_entity_id="ent-1")
        params = t.calls[0][2]["params"]
        assert params["include_inactive"] == "true"
        assert params["as_of"] == "2026-06-01T00:00:00Z"
        assert params["from_entity_id"] == "ent-1"

    def test_list_relationships_default_omits_include_inactive(self):
        t = Transport({("GET", "/v1/memory/relationships"): _relationships_resp([])})
        mem = _sync_memory(t)
        mem.list_relationships()
        assert "include_inactive" not in t.calls[0][2]["params"]


class TestGraphFacts:
    def test_remember_fact_owner_default(self):
        t = Transport({("POST", "/v1/memory/facts"): _fact_resp()})
        mem = _sync_memory(t)
        fact = mem.remember_fact("favorite_color", "blue")
        body = t.calls[0][2]["json"]
        assert body["subject_type"] == "owner"
        assert "subject_id" not in body
        assert body["predicate"] == "favorite_color"
        assert body["value"] == "blue"
        assert fact.fact_id == "fact-1"

    def test_remember_fact_entity_subject(self):
        t = Transport({("POST", "/v1/memory/facts"): _fact_resp(subject_type="entity", subject_id="ent-1")})
        mem = _sync_memory(t)
        mem.remember_fact("role", "engineer", subject_type="entity", subject_id="ent-1")
        body = t.calls[0][2]["json"]
        assert body["subject_type"] == "entity"
        assert body["subject_id"] == "ent-1"

    def test_remember_fact_scalar_value_types(self):
        t = Transport({("POST", "/v1/memory/facts"): [_fact_resp(value=7), _fact_resp(value=None)]})
        mem = _sync_memory(t)
        mem.remember_fact("count", 7)
        assert t.calls[0][2]["json"]["value"] == 7
        mem.remember_fact("nickname", None)
        assert t.calls[1][2]["json"]["value"] is None  # null value sent explicitly

    def test_remember_fact_supersedes_and_cardinality(self):
        t = Transport({("POST", "/v1/memory/facts"): _fact_resp()})
        mem = _sync_memory(t)
        mem.remember_fact("color", "red", cardinality="multi", supersedes_fact_id="fact-0")
        body = t.calls[0][2]["json"]
        assert body["cardinality"] == "multi"
        assert body["supersedes_fact_id"] == "fact-0"

    def test_entity_subject_without_id_raises_no_http(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.remember_fact("role", "x", subject_type="entity")
        assert t.calls == []

    def test_list_facts_filters(self):
        t = Transport({("GET", "/v1/memory/facts"): _facts_resp([_fact_dict()])})
        mem = _sync_memory(t)
        mem.list_facts(subject_type="entity", subject_id="ent-1", predicate="role", include_inactive=True)
        params = t.calls[0][2]["params"]
        assert params["subject_type"] == "entity"
        assert params["subject_id"] == "ent-1"
        assert params["predicate"] == "role"
        assert params["include_inactive"] == "true"
        assert "history" not in params

    def test_fact_history(self):
        t = Transport({("GET", "/v1/memory/facts"): _facts_resp([_fact_dict(invalid_from="2026-06-10T00:00:00Z"), _fact_dict(fact_id="fact-2")])})
        mem = _sync_memory(t)
        facts = mem.fact_history("favorite_color", subject_type="entity", subject_id="ent-1")
        params = t.calls[0][2]["params"]
        assert params["history"] == "true"
        assert params["subject_type"] == "entity"
        assert params["subject_id"] == "ent-1"
        assert params["predicate"] == "favorite_color"
        assert len(facts) == 2


class TestGraphConsolidate:
    def test_consolidate(self):
        t = Transport({("POST", "/v1/memory/consolidate"): _consolidate_resp(before=5, after=3, retracted=2)})
        mem = _sync_memory(t)
        report = mem.consolidate()
        assert t.calls[0][2]["params"]["entity_id"] == "user-42"
        assert report.active_facts_before == 5
        assert report.active_facts_after == 3
        assert report.retracted == 2


class TestGraphValidation:
    def test_empty_entity_type_raises_no_http(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.upsert_entity("  ")
        assert t.calls == []

    def test_empty_predicate_raises_no_http(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.remember_fact("", "x")
        assert t.calls == []

    def test_bad_subject_type_raises_no_http(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.remember_fact("p", "x", subject_type="bogus")
        assert t.calls == []

    def test_bad_cardinality_raises_no_http(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.remember_fact("p", "x", cardinality="bogus")
        assert t.calls == []

    def test_empty_relate_args_raise_no_http(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.relate("", "ent-2", "works_at")
        assert t.calls == []

    def test_empty_get_entity_raises_no_http(self):
        t = Transport({})
        mem = _sync_memory(t)
        with pytest.raises(ValueError):
            mem.get_entity("")
        assert t.calls == []


class TestGraphErrorPassthrough:
    def test_402_surfaces_typed_error(self):
        err = _resp(json_data={"error": "credits exhausted", "code": "credit_exhausted"}, status_code=402)
        t = Transport({("POST", "/v1/memory/entities"): err})
        mem = _sync_memory(t)
        with pytest.raises(CreditExhaustedError):
            mem.upsert_entity("person")


class TestGraphAsync:
    async def test_async_upsert_entity(self):
        t = Transport({("POST", "/v1/memory/entities"): _entity_resp(memory_entity_id="ent-async")})
        mem = _async_memory(t)
        ent = await mem.upsert_entity("person", attributes={"k": "v"})
        assert ent.memory_entity_id == "ent-async"
        assert t.calls[0][2]["params"]["entity_id"] == "user-42"
        assert t.calls[0][2]["json"]["attributes"] == {"k": "v"}

    async def test_async_remember_fact_validation(self):
        t = Transport({})
        mem = _async_memory(t)
        with pytest.raises(ValueError):
            await mem.remember_fact("role", "x", subject_type="relationship")
        assert t.calls == []

    async def test_async_consolidate(self):
        t = Transport({("POST", "/v1/memory/consolidate"): _consolidate_resp()})
        mem = _async_memory(t)
        report = await mem.consolidate()
        assert report.retracted == 1
