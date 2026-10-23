"""``get_text`` and the recall/list hydration rule (MEMORY_CONTRACT.md §25.6,
golden table §27.5).

Mocked at the transport layer with ``httpx.MockTransport`` so the real
request, retry and error-mapping code runs for both the sync and async
clients. The handler answers by route, so parallel async reads may arrive in
any order.
"""

import json

import httpx
import pytest

from aether import (
    AetherApiError,
    AetherClient,
    AsyncAetherClient,
    AsyncMemory,
    DocumentText,
    Memory,
    MemoryFileSection,
)
from aether._internal import is_text_like_content_type

PARSED = "# Master Services Agreement\n\nParsed text."


def _document_text(**overrides):
    body = {
        "doc_id": "doc-1",
        "content_type": "application/pdf",
        "source": "passages",
        "state": "ready",
        "text": PARSED,
        "offset": 0,
        "next_offset": None,
        "total_bytes": len(PARSED),
        "truncated": False,
        "section": None,
        "sections": None,
    }
    body.update(overrides)
    return body


def _hit(content_type, **extra):
    hit = {
        "doc_id": "doc-1",
        "score": 90,
        "content_type": content_type,
        "passage": "the matched passage",
        "tags": [],
        "source": None,
        "metadata": {},
        "partition": None,
    }
    hit.update(extra)
    return hit


def _record(content_type, **extra):
    record = {
        "doc_id": "doc-1",
        "cid": "",
        "content_type": content_type,
        "size_bytes": 10,
        "version": 1,
        "created_at": "2026-10-01T00:00:00Z",
        "entity_id": "acme",
    }
    record.update(extra)
    return record


class Router:
    """A route-keyed fake server that records the document reads it served."""

    def __init__(self, *, search=None, listing=None, download="original text", text=None):
        self.search = search
        self.listing = listing
        self.download = download
        # text: (status, body) or None for the default ready response; the
        # string "forbidden" makes any /text call fail the test.
        self.text = text
        self.downloads = []
        self.texts = []
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/search":
            return httpx.Response(200, json={"query": "q", "results": [self.search]})
        if path == "/v1/documents":
            return httpx.Response(
                200, json={"documents": [self.listing], "total": 1, "has_more": False}
            )
        if path.endswith("/download"):
            self.downloads.append(path.split("/")[3])
            return httpx.Response(200, content=self.download.encode("utf-8"))
        if path.endswith("/text"):
            if self.text == "forbidden":
                raise AssertionError("/text must not be called")
            self.texts.append(request.url)
            status, body = self.text or (200, _document_text())
            return httpx.Response(status, json=body)
        raise AssertionError(f"unexpected request {request.method} {request.url}")


def _client(router) -> AetherClient:
    c = AetherClient(base_url="http://localhost:9000", api_key="k", max_retries=0)
    c._client = httpx.Client(
        base_url=c.base_url,
        transport=httpx.MockTransport(router),
        headers={"Authorization": "Bearer k"},
    )
    return c


def _async_client(router) -> AsyncAetherClient:
    c = AsyncAetherClient(base_url="http://localhost:9000", api_key="k", max_retries=0)
    c._client = httpx.AsyncClient(
        base_url=c.base_url,
        transport=httpx.MockTransport(router),
        headers={"Authorization": "Bearer k"},
    )
    return c


# ── get_text ─────────────────────────────────────────────────────────


def test_get_text_maps_options_and_parses_the_result():
    router = Router(
        text=(
            200,
            _document_text(
                text="## Agreed terms\n- Net 30\n",
                next_offset=25,
                truncated=True,
                section={
                    "heading": "Agreed terms",
                    "level": 2,
                    "slug": "agreed-terms",
                    "start_line": 4,
                    "end_line": 6,
                    "bytes": 71,
                },
            ),
        )
    )
    client = _client(router).partition("client-a")
    text = client.get_text("doc-1", section="Agreed terms", offset=0, max_bytes=25)

    url = router.texts[0]
    assert url.path == "/v1/documents/doc-1/text"
    assert url.params["section"] == "Agreed terms"
    assert url.params["offset"] == "0"
    assert url.params["max_bytes"] == "25"
    assert url.params["partition"] == "client-a"
    assert "outline" not in url.params
    assert router.requests[0].method == "GET"

    assert isinstance(text, DocumentText)
    assert text.source == "passages"
    assert text.next_offset == 25
    assert text.truncated is True
    assert text.section == MemoryFileSection(
        heading="Agreed terms", level=2, slug="agreed-terms", start_line=4, end_line=6, bytes=71
    )
    assert text.sections is None


def test_get_text_outline_parses_sections_and_null_next_offset():
    router = Router(
        text=(
            200,
            _document_text(
                text="",
                sections=[
                    {
                        "heading": "Acme memory",
                        "level": 1,
                        "slug": "acme-memory",
                        "start_line": 1,
                        "end_line": 9,
                        "bytes": 122,
                    }
                ],
            ),
        )
    )
    text = _client(router).get_text("doc-1", outline=True)
    url = router.texts[0]
    assert url.params["outline"] == "true"
    assert "offset" not in url.params and "max_bytes" not in url.params
    assert text.next_offset is None
    assert [s.slug for s in text.sections] == ["acme-memory"]


def test_get_text_rejects_an_empty_doc_id_before_any_http_call():
    router = Router()
    with pytest.raises(ValueError):
        _client(router).get_text("")
    assert router.requests == []


def test_get_text_surfaces_section_errors_with_the_server_code():
    router = Router(
        text=(
            409,
            {
                "error": "ambiguous",
                "code": "document_section_ambiguous",
                "candidates": [{"heading": "Notes", "slug": "notes", "start_line": 16}],
            },
        )
    )
    with pytest.raises(AetherApiError) as exc:
        _client(router).get_text("doc-1", section="notes")
    assert exc.value.status_code == 409
    assert exc.value.error_code == "document_section_ambiguous"
    assert exc.value.body["candidates"][0]["slug"] == "notes"


@pytest.mark.asyncio
async def test_async_get_text_maps_options():
    router = Router()
    client = _async_client(router)
    text = await client.get_text("doc-1", max_bytes=1048576)
    assert router.texts[0].params["max_bytes"] == "1048576"
    assert text.text == PARSED
    await client.close()


# ── hydration (§25.6 / §27.5) ────────────────────────────────────────

DOWNLOAD_ROWS = [
    "text/plain",
    "text/markdown; charset=utf-8",
    "TEXT/MARKDOWN",
    "text/x-python",
    "application/json",
    "application/x-yaml",
]
TEXT_ROWS = [
    "text/html",
    "text/csv",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/octet-stream",
    "",
]


def test_text_like_classification_matches_the_golden_table():
    for ct in DOWNLOAD_ROWS:
        assert is_text_like_content_type(ct), ct
    for ct in TEXT_ROWS:
        assert not is_text_like_content_type(ct), ct
    assert not is_text_like_content_type(None)


@pytest.mark.parametrize("content_type", DOWNLOAD_ROWS)
def test_retrieve_downloads_text_like_types(content_type):
    router = Router(search=_hit(content_type), text="forbidden")
    [result] = _client(router).retrieve("q")
    assert result.content == "original text"
    assert router.downloads == ["doc-1"]


@pytest.mark.parametrize("content_type", TEXT_ROWS)
def test_retrieve_reads_other_types_through_text_never_download(content_type):
    router = Router(search=_hit(content_type))
    [result] = _client(router).retrieve("q")
    assert result.content == PARSED
    assert router.downloads == []
    assert [u.params["max_bytes"] for u in router.texts] == ["1048576"]


@pytest.mark.parametrize("content_type", TEXT_ROWS)
def test_memory_list_reads_other_types_through_text(content_type):
    router = Router(listing=_record(content_type))
    [item] = Memory("acme", client=_client(router)).list()
    assert item.text == PARSED
    assert router.downloads == []


@pytest.mark.parametrize("content_type", DOWNLOAD_ROWS)
def test_memory_list_downloads_text_like_types(content_type):
    router = Router(listing=_record(content_type), text="forbidden")
    [item] = Memory("acme", client=_client(router)).list()
    assert item.text == "original text"


def test_unsupported_pdf_falls_back_to_the_passage():
    router = Router(
        search=_hit("application/pdf"),
        text=(200, _document_text(state="unsupported", source=None, text="")),
    )
    [result] = _client(router).retrieve("q")
    assert result.content == "the matched passage"


def test_404_from_an_older_server_falls_back_to_the_passage():
    router = Router(
        search=_hit("application/pdf"),
        text=(404, {"error": "Not Found", "code": "document_not_found"}),
    )
    [result] = _client(router).retrieve("q")
    assert result.content == "the matched passage"
    assert router.downloads == []


def test_missing_passage_falls_back_to_empty_string():
    hit = _hit("application/pdf")
    del hit["passage"]
    router = Router(search=hit, text=(404, {"error": "Not Found"}))
    [result] = _client(router).retrieve("q")
    assert result.content == ""


def test_memory_recall_returns_parsed_text_for_a_pdf_hit():
    router = Router(search=_hit("application/pdf"))
    [item] = Memory("acme", client=_client(router)).recall("terms")
    assert item.text == PARSED
    assert router.downloads == []


def test_memory_list_falls_back_to_empty_string():
    for text in (
        (200, _document_text(state="unsupported", source=None, text="")),
        (404, {"error": "Not Found"}),
    ):
        router = Router(listing=_record("application/pdf"), text=text)
        [item] = Memory("acme", client=_client(router)).list()
        assert item.text == ""


def test_other_errors_propagate():
    router = Router(search=_hit("application/pdf"), text=(500, {"error": "boom"}))
    with pytest.raises(AetherApiError) as exc:
        _client(router).retrieve("q")
    assert exc.value.status_code == 500


@pytest.mark.parametrize("content_type,modality", [("image/png", "image"), ("audio/mpeg", "audio")])
def test_media_is_never_read(content_type, modality):
    router = Router(search=_hit(content_type, modality=modality), text="forbidden")
    [result] = _client(router).retrieve("q")
    assert result.content == "the matched passage"
    router = Router(
        listing=_record(content_type, modality=modality, derived_text="a caption"),
        text="forbidden",
    )
    [item] = Memory("acme", client=_client(router)).list()
    assert item.text == "a caption"
    assert router.downloads == []


# ── async twins ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", TEXT_ROWS)
async def test_async_retrieve_reads_other_types_through_text(content_type):
    router = Router(search=_hit(content_type))
    client = _async_client(router)
    [result] = await client.retrieve("q")
    assert result.content == PARSED
    assert router.downloads == []
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", DOWNLOAD_ROWS)
async def test_async_retrieve_downloads_text_like_types(content_type):
    router = Router(search=_hit(content_type), text="forbidden")
    client = _async_client(router)
    [result] = await client.retrieve("q")
    assert result.content == "original text"
    await client.close()


@pytest.mark.asyncio
async def test_async_fallbacks_and_memory_list():
    router = Router(search=_hit("application/pdf"), text=(404, {"error": "Not Found"}))
    client = _async_client(router)
    [result] = await client.retrieve("q")
    assert result.content == "the matched passage"
    await client.close()

    router = Router(
        listing=_record("application/pdf"),
        text=(200, _document_text(state="unsupported", source=None, text="")),
    )
    client = _async_client(router)
    [item] = await AsyncMemory("acme", client=client).list()
    assert item.text == ""
    await client.close()

    router = Router(listing=_record("application/pdf"))
    client = _async_client(router)
    [item] = await AsyncMemory("acme", client=client).list()
    assert item.text == PARSED
    assert router.downloads == []
    await client.close()

    router = Router(search=_hit("application/pdf"), text=(503, {"error": "busy"}))
    client = _async_client(router)
    with pytest.raises(AetherApiError):
        await client.retrieve("q")
    await client.close()


def test_document_text_round_trips_through_json():
    # The parsed dataclass carries every wire field.
    body = _document_text(next_offset=10, truncated=True)
    text = _client(Router(text=(200, body))).get_text("doc-1")
    assert {
        "doc_id": text.doc_id,
        "content_type": text.content_type,
        "source": text.source,
        "state": text.state,
        "text": text.text,
        "offset": text.offset,
        "next_offset": text.next_offset,
        "total_bytes": text.total_bytes,
        "truncated": text.truncated,
        "section": text.section,
        "sections": text.sections,
    } == json.loads(json.dumps(body))
