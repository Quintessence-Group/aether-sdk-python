"""Memory files (``Memory.files``, ``list_memory_files``, ``grep_memory_files``),
document summaries, the ``summarize`` ingest option and the typed
``VersionConflictError``.

Mocked at the transport layer with ``httpx.MockTransport`` so the real
request building, retry and error-mapping code runs for both the sync and
async clients. Each fake server answers with scripted responses in order and
records every request it saw.
"""

import json
import uuid

import httpx
import pytest

from aether import (
    AetherApiError,
    AetherClient,
    AsyncAetherClient,
    AsyncMemory,
    AsyncMemoryFiles,
    BatchInsertItem,
    BatchSearchQuery,
    CreditExhaustedError,
    DocumentSummary,
    Memory,
    MemoryFile,
    MemoryFileDeleteResult,
    MemoryFileDiff,
    MemoryFileHit,
    MemoryFileInfo,
    MemoryFilePage,
    MemoryFiles,
    MemoryFileSection,
    MemoryFileVersions,
    MemoryFileWriteResult,
    MemoryGrepResult,
    VersionConflictError,
    aether_api_error_from_response,
    validate_memory_file_name,
)
from aether._internal import parse_etag_version

OWNER = "scoped:ws_42:company:c_7"
PARTITION = "scoped-ws_42"
PREFIX = "scoped:ws_42:company:"

AGREED_TERMS = {
    "heading": "Agreed terms",
    "level": 2,
    "slug": "agreed-terms",
    "start_line": 4,
    "end_line": 6,
    "bytes": 71,
}
VECTOR_A = [
    {"heading": "Acme memory", "level": 1, "slug": "acme-memory", "start_line": 1, "end_line": 9, "bytes": 122},
    AGREED_TERMS,
    {"heading": "Exceptions", "level": 2, "slug": "exceptions", "start_line": 7, "end_line": 8, "bytes": 15},
    {"heading": "Notes", "level": 2, "slug": "notes", "start_line": 9, "end_line": 9, "bytes": 9},
]
AGREED_CONTENT = "## Agreed terms\n- 2026-09-01 — Net 30 payment terms (alice, ISS-12)\n\n"

INFO = {
    "entity_id": OWNER,
    "partition": PARTITION,
    "name": "memory.md",
    "file_id": "3d1f2c4e-0000-4000-8000-000000000001",
    "doc_id": "a8c2b7d4-0000-4000-8000-000000000007",
    "version": 7,
    "bytes": 122,
    "content_hash": "9f1c7a",
    "author": "scoped:ws_42:user:u_3",
    "principal": "key:ak_1a2b3c4d",
    "op": "append_section",
    "created_at": "2026-09-01T10:00:00Z",
    "updated_at": "2026-10-10T09:12:44Z",
}
READ_BODY = {
    **INFO,
    "is_current": True,
    "content": AGREED_CONTENT,
    "section": AGREED_TERMS,
    "offset": 0,
    "next_offset": None,
    "total_bytes": 71,
    "truncated": False,
    "sections": VECTOR_A,
}

NAME_128 = "abcdefghijklmnopqrstuvwxyz" * 4 + "abcdefghijklmnopqrstuvwx"
ACCEPTED_NAMES = ["memory.md", "a", "A", "0", "a.", "Grep", "grep.md", "notes_2026-10.md", "x.y.z", NAME_128]
REJECTED_NAMES = [
    "",
    ".hidden",
    "-x",
    "_x",
    ".",
    "..",
    "a..b",
    "a...b",
    "a/b",
    "a b",
    "a\\b",
    "memory.md ",
    "naïve.md",
    "grep",
    NAME_128 + "y",
    "memory.md\n",
]


def _write_body(**overrides):
    body = {
        **INFO,
        "version": 8,
        "unchanged": False,
        "previous_version": 7,
        "file_created": False,
        "section": AGREED_TERMS,
        "section_created": False,
        "sections": VECTOR_A,
    }
    body.update(overrides)
    return body


def _summary_body(**overrides):
    body = {
        "doc_id": "doc-1",
        "state": "ready",
        "reason": None,
        "error": None,
        "summary_doc_id": "sum-1",
        "content": "## Summary\n- Net 30.\n",
        "revision": "r3f9a1c2b7d4e5f60",
        "authored_by": "aether",
        "generator": "aether:model-x:2026-10-10.v1",
        "prompt_version": "2026-10-10.v1",
        "generated_at": "2026-10-10T09:00:00Z",
        "source_cid": "cid-1",
        "stale": False,
        "truncated_input": False,
    }
    body.update(overrides)
    return body


def _record(**overrides):
    record = {
        "doc_id": "doc-1",
        "cid": "cid-1",
        "content_type": "text/plain",
        "chunks": 1,
        "vectors": 1,
        "version": 1,
        "created_at": "2026-10-10T00:00:00Z",
        "entity_id": OWNER,
    }
    record.update(overrides)
    return record


def ok(body=None, status=200, etag=None, headers=None):
    h = dict(headers or {})
    if etag is not None:
        h["ETag"] = etag
    return httpx.Response(status, json=body if body is not None else {}, headers=h)


def err(status, code, **fields):
    body = {"error": f"{code} happened", "code": code, "request_id": "req-1", **fields}
    return httpx.Response(status, json=body, headers={"x-request-id": "req-1"})


class Server:
    """Answers each request with the next scripted response and records it."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        return self.responses.pop(0)

    @property
    def only(self) -> httpx.Request:
        assert len(self.requests) == 1, [str(r.url) for r in self.requests]
        return self.requests[0]


def _json(request: httpx.Request):
    return json.loads(request.content)


def _client(server, *, max_retries=0) -> AetherClient:
    c = AetherClient(
        base_url="http://localhost:9000", api_key="k", max_retries=max_retries, retry_base_delay=0
    )
    c._client = httpx.Client(
        base_url=c.base_url,
        transport=httpx.MockTransport(server),
        headers={"Authorization": "Bearer k"},
    )
    return c


def _async_client(server, *, max_retries=0) -> AsyncAetherClient:
    c = AsyncAetherClient(
        base_url="http://localhost:9000", api_key="k", max_retries=max_retries, retry_base_delay=0
    )
    c._client = httpx.AsyncClient(
        base_url=c.base_url,
        transport=httpx.MockTransport(server),
        headers={"Authorization": "Bearer k"},
    )
    return c


def _memory(server, *, partition=True, max_retries=0, **kw) -> Memory:
    client = _client(server, max_retries=max_retries)
    if partition:
        client = client.partition(PARTITION)
    return Memory(OWNER, client=client, **kw)


def _async_memory(server, *, partition=True, max_retries=0, **kw) -> AsyncMemory:
    client = _async_client(server, max_retries=max_retries)
    if partition:
        client = client.partition(PARTITION)
    return AsyncMemory(OWNER, client=client, **kw)


def _is_uuid4(value) -> bool:
    try:
        return uuid.UUID(value).version == 4
    except (TypeError, ValueError):
        return False


# Every per-file facade op, called with a given name and otherwise valid args.
PER_FILE_OPS = {
    "read": lambda f, n: f.read(n),
    "write": lambda f, n: f.write(n, "x", create=True),
    "append": lambda f, n: f.append(n, "x"),
    "append_section": lambda f, n: f.append_section(n, "Notes", "x"),
    "replace_section": lambda f, n: f.replace_section(n, "Notes", "x", expected_version=1),
    "versions": lambda f, n: f.versions(n),
    "diff": lambda f, n: f.diff(n),
    "delete": lambda f, n: f.delete(n),
}


# ── 1. Names ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ACCEPTED_NAMES)
def test_accepted_names_pass_the_validator_and_reach_the_path(name):
    assert validate_memory_file_name(name) == name
    server = Server(ok(READ_BODY))
    _memory(server).files.read(name)
    assert server.only.url.path == f"/v1/memory/files/{name}"


@pytest.mark.parametrize("name", REJECTED_NAMES)
def test_rejected_names_raise_before_any_http_call(name):
    with pytest.raises(ValueError, match="invalid memory file name"):
        validate_memory_file_name(name)
    server = Server()
    with pytest.raises(ValueError, match="invalid memory file name"):
        _memory(server).files.read(name)
    assert server.requests == []


@pytest.mark.parametrize("op", sorted(PER_FILE_OPS))
def test_every_per_file_op_validates_the_name(op):
    server = Server()
    files = _memory(server).files
    for bad in ("grep", "a..b", "memory.md\n", None):
        with pytest.raises(ValueError):
            PER_FILE_OPS[op](files, bad)
    assert server.requests == []


def test_validator_message_explains_the_rule():
    with pytest.raises(ValueError) as exc:
        validate_memory_file_name("a b")
    message = str(exc.value)
    assert "'a b'" in message
    assert "1-128" in message and '"grep"' in message


# ── 2. Scoping ───────────────────────────────────────────────────────


def test_facade_calls_send_the_owner_and_the_partition():
    server = Server(
        ok(READ_BODY, etag='"7"'),
        ok(_write_body(), etag='"8"'),
        ok({"files": [], "next_cursor": None}),
        ok({"matches": [], "truncated": False, "truncated_reason": None, "files_scanned": 0, "bytes_scanned": 0}),
        ok({**{k: INFO[k] for k in ("entity_id", "partition", "name", "file_id")}, "version": 8, "hard": False, "versions_deleted": 1}),
    )
    files = _memory(server).files
    files.read("memory.md")
    files.write("memory.md", "# Acme\n", expected_version=7)
    files.list()
    files.grep("Net 30")
    files.delete("memory.md")

    read, write, listing, grep, delete = server.requests
    for request in (read, write, listing, delete):
        assert request.url.params["entity_id"] == OWNER
        assert request.url.params["partition"] == PARTITION
    assert (listing.method, listing.url.path) == ("GET", "/v1/memory/files")
    assert (grep.method, grep.url.path) == ("POST", "/v1/memory/files/grep")
    assert grep.url.params["partition"] == PARTITION
    assert "entity_id" not in grep.url.params
    assert _json(grep)["entity_id"] == OWNER
    assert "entity_prefix" not in _json(grep)


def test_unscoped_memory_sends_no_partition():
    server = Server(ok(READ_BODY), ok({"files": []}))
    files = _memory(server, partition=False).files
    files.read("memory.md")
    files.list()
    for request in server.requests:
        assert "partition" not in request.url.params
        assert request.url.params["entity_id"] == OWNER


def test_files_accessor_is_stable_and_scoped_to_the_entity():
    mem = _memory(Server())
    assert isinstance(mem.files, MemoryFiles)
    assert mem.files is mem.files
    assert mem.files.entity_id == OWNER


def test_list_memory_files_by_entity_prefix():
    server = Server(ok({"files": [INFO], "next_cursor": "c2"}))
    page = _client(server).partition(PARTITION).list_memory_files(
        entity_prefix=PREFIX, prefix="mem", limit=50, cursor="c1"
    )
    params = server.only.url.params
    assert params["entity_prefix"] == PREFIX
    assert "entity_id" not in params
    assert (params["prefix"], params["limit"], params["cursor"]) == ("mem", "50", "c1")
    assert params["partition"] == PARTITION
    assert isinstance(page, MemoryFilePage)
    assert page.next_cursor == "c2"
    assert page.files == [MemoryFileInfo(**INFO)]


def test_list_memory_files_treats_empty_strings_as_unset():
    server = Server(ok({"files": [], "next_cursor": None}))
    _client(server).list_memory_files(entity_id="", entity_prefix=PREFIX, prefix="", cursor="")
    params = server.only.url.params
    assert dict(params) == {"entity_prefix": PREFIX}


def test_list_memory_files_with_no_scope_sends_no_entity():
    server = Server(ok({"files": []}))
    _client(server).list_memory_files()
    assert dict(server.only.url.params) == {}


def test_grep_memory_files_by_entity_prefix():
    server = Server(ok({"matches": []}))
    _client(server).grep_memory_files("net", entity_prefix=PREFIX)
    body = _json(server.only)
    assert body["entity_prefix"] == PREFIX
    assert "entity_id" not in body


def test_cross_entity_calls_reject_both_scopes_before_any_http_call():
    server = Server()
    client = _client(server)
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.list_memory_files(entity_id=OWNER, entity_prefix=PREFIX)
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.grep_memory_files("net", entity_id=OWNER, entity_prefix=PREFIX)
    assert server.requests == []


# ── 3. Preconditions ─────────────────────────────────────────────────


@pytest.mark.parametrize("version,header", [(7, '"7"'), (1, '"1"'), (123456, '"123456"')])
def test_write_expected_version_sends_a_quoted_if_match(version, header):
    server = Server(ok(_write_body()))
    _memory(server).files.write("memory.md", "x", expected_version=version)
    headers = server.only.headers
    assert headers["If-Match"] == header
    assert "If-None-Match" not in headers


def test_write_create_sends_if_none_match_star():
    server = Server(ok(_write_body(version=1, previous_version=None, file_created=True), status=201))
    _memory(server).files.write("memory.md", "x", create=True)
    headers = server.only.headers
    assert headers["If-None-Match"] == "*"
    assert "If-Match" not in headers


@pytest.mark.parametrize("kwargs", [{}, {"expected_version": 7, "create": True}])
def test_write_needs_exactly_one_precondition(kwargs):
    server = Server()
    with pytest.raises(ValueError, match="exactly one of expected_version or create"):
        _memory(server).files.write("memory.md", "x", **kwargs)
    assert server.requests == []


@pytest.mark.parametrize("bad", [0, -1, True, "7", 1.5])
def test_expected_version_must_be_a_positive_integer(bad):
    server = Server()
    files = _memory(server).files
    with pytest.raises(ValueError, match="expected_version"):
        files.write("memory.md", "x", expected_version=bad)
    with pytest.raises(ValueError, match="expected_version"):
        files.append("memory.md", "x", expected_version=bad)
    with pytest.raises(ValueError, match="expected_version"):
        files.delete("memory.md", expected_version=bad)
    assert server.requests == []


def test_append_sends_if_match_only_when_given():
    server = Server(ok(_write_body()), ok(_write_body()), ok(_write_body()))
    files = _memory(server).files
    files.append("memory.md", "- a")
    files.append("memory.md", "- b", expected_version=3)
    files.append_section("memory.md", "Notes", "- c", expected_version=5)
    plain, conditional, section = server.requests
    assert "If-Match" not in plain.headers and "If-None-Match" not in plain.headers
    assert conditional.headers["If-Match"] == '"3"'
    assert section.headers["If-Match"] == '"5"'


def test_replace_section_needs_exactly_one_precondition():
    server = Server(ok(_write_body()), ok(_write_body(file_created=True), status=201))
    files = _memory(server).files
    with pytest.raises(ValueError, match="exactly one of expected_version or create"):
        files.replace_section("memory.md", "Notes", "- n1")
    with pytest.raises(ValueError, match="exactly one of expected_version or create"):
        files.replace_section("memory.md", "Notes", "- n1", expected_version=7, create=True)
    assert server.requests == []
    files.replace_section("memory.md", "Notes", "- n1", expected_version=7)
    files.replace_section("memory.md", "Notes", "- n1", create=True)
    by_version, created = server.requests
    assert by_version.headers["If-Match"] == '"7"' and "If-None-Match" not in by_version.headers
    assert created.headers["If-None-Match"] == "*" and "If-Match" not in created.headers


def test_delete_expected_version_sends_if_match():
    server = Server(ok({"version": 4, "hard": False, "versions_deleted": 1}), ok({"versions_deleted": 1}))
    files = _memory(server).files
    files.delete("memory.md", expected_version=4)
    files.delete("memory.md")
    conditional, plain = server.requests
    assert conditional.headers["If-Match"] == '"4"'
    assert "If-Match" not in plain.headers


def test_put_summary_expected_revision_sends_a_quoted_if_match():
    server = Server(ok(_summary_body(authored_by="user")))
    _client(server).put_summary("doc-1", "My summary", expected_revision="r3f9a1c2b7d4e5f60")
    assert server.only.headers["If-Match"] == '"r3f9a1c2b7d4e5f60"'
    assert "If-None-Match" not in server.only.headers


# ── 4. Idempotency-Key ───────────────────────────────────────────────

WRITE_OPS = {
    "write": lambda f, **kw: f.write("memory.md", "# Acme\n", expected_version=7, **kw),
    "append": lambda f, **kw: f.append("memory.md", "- a", **kw),
    "append_section": lambda f, **kw: f.append_section("memory.md", "Notes", "- a", **kw),
    "replace_section": lambda f, **kw: f.replace_section("memory.md", "Notes", "- a", expected_version=7, **kw),
}


@pytest.mark.parametrize("op", sorted(WRITE_OPS))
def test_every_write_sends_an_idempotency_key(op):
    server = Server(ok(_write_body()), ok(_write_body()))
    files = _memory(server).files
    WRITE_OPS[op](files)
    WRITE_OPS[op](files)
    first, second = server.requests
    assert _is_uuid4(first.headers["Idempotency-Key"])
    assert _is_uuid4(second.headers["Idempotency-Key"])
    assert first.headers["Idempotency-Key"] != second.headers["Idempotency-Key"]


@pytest.mark.parametrize("op", sorted(WRITE_OPS))
def test_a_retried_busy_write_reuses_its_idempotency_key(op):
    busy = httpx.Response(
        503,
        json={"error": "file is busy", "code": "memory_file_busy"},
        headers={"Retry-After": "0"},
    )
    server = Server(busy, ok(_write_body(), etag='"8"'))
    result = WRITE_OPS[op](_memory(server, max_retries=1).files)
    first, second = server.requests
    assert first.method == second.method
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]
    assert _is_uuid4(first.headers["Idempotency-Key"])
    assert result.version == 8


@pytest.mark.parametrize("op", sorted(WRITE_OPS))
def test_a_caller_supplied_idempotency_key_is_sent_verbatim(op):
    server = Server(ok(_write_body()))
    WRITE_OPS[op](_memory(server).files, idempotency_key="my-key-1")
    assert server.only.headers["Idempotency-Key"] == "my-key-1"


def test_an_empty_idempotency_key_gets_a_fresh_one():
    server = Server(ok(_write_body()))
    _memory(server).files.append("memory.md", "- a", idempotency_key="")
    assert _is_uuid4(server.only.headers["Idempotency-Key"])


def test_reads_deletes_and_summary_puts_send_no_idempotency_key():
    server = Server(
        ok(READ_BODY),
        ok({"versions": []}),
        ok({"diff": ""}),
        ok({"versions_deleted": 1}),
        ok(_summary_body()),
        ok(_summary_body()),
    )
    mem = _memory(server)
    mem.files.read("memory.md")
    mem.files.versions("memory.md")
    mem.files.diff("memory.md")
    mem.files.delete("memory.md")
    client = mem.client
    client.get_summary("doc-1")
    client.put_summary("doc-1", "x")
    for request in server.requests:
        assert "Idempotency-Key" not in request.headers, request.url


# ── 5. Typed 412 ─────────────────────────────────────────────────────


def test_version_conflict_is_typed_with_current_version():
    server = Server(err(412, "memory_file_version_conflict", current_version=8))
    with pytest.raises(VersionConflictError) as exc:
        _memory(server).files.write("memory.md", "x", expected_version=7)
    e = exc.value
    assert isinstance(e, AetherApiError)
    assert e.status_code == 412
    assert e.error_code == "memory_file_version_conflict"
    assert e.current_version == 8
    assert e.request_id == "req-1"
    assert not e.is_retryable


@pytest.mark.parametrize("fields", [{"current_version": None}, {}, {"current_version": "8"}, {"current_version": True}])
def test_version_conflict_without_an_integer_current_version_is_none(fields):
    server = Server(err(412, "memory_file_version_conflict", **fields))
    with pytest.raises(VersionConflictError) as exc:
        _memory(server).files.write("memory.md", "x", create=True)
    assert exc.value.current_version is None


def test_version_conflict_on_delete_and_from_the_factory():
    server = Server(err(412, "memory_file_version_conflict", current_version=5))
    with pytest.raises(VersionConflictError) as exc:
        _memory(server).files.delete("memory.md", expected_version=4)
    assert exc.value.current_version == 5

    e = aether_api_error_from_response(
        412, "conflict", error_code="memory_file_version_conflict", body={"current_version": 3}
    )
    assert type(e) is VersionConflictError and e.current_version == 3
    other = aether_api_error_from_response(412, "conflict", error_code="summary_revision_conflict")
    assert type(other) is AetherApiError


# ── 6. PATCH bodies ──────────────────────────────────────────────────


def test_append_section_sends_the_exact_patch_body():
    server = Server(ok(_write_body()))
    _memory(server).files.append_section("memory.md", "Agreed terms", "x")
    request = server.only
    assert request.method == "PATCH"
    assert request.url.path == "/v1/memory/files/memory.md"
    assert request.headers["Content-Type"] == "application/json"
    assert _json(request) == {
        "op": "append_section",
        "section": "Agreed terms",
        "content": "x",
        "level": 2,
        "create_if_missing": True,
    }


def test_append_sends_no_section_or_level():
    server = Server(ok(_write_body()))
    _memory(server).files.append("memory.md", "trailing", author="scoped:ws_42:user:u_3")
    request = server.only
    assert request.method == "PATCH"
    assert _json(request) == {"op": "append", "content": "trailing", "create_if_missing": True}
    assert request.url.params["author"] == "scoped:ws_42:user:u_3"


def test_replace_section_maps_level_and_create_if_missing():
    server = Server(ok(_write_body()))
    _memory(server).files.replace_section(
        "memory.md", "Key decisions", "", expected_version=7, level=3, create_if_missing=False
    )
    assert _json(server.only) == {
        "op": "replace_section",
        "section": "Key decisions",
        "content": "",
        "level": 3,
        "create_if_missing": False,
    }


def test_golden_p1_append_section():
    server = Server(ok(_write_body(bytes=175)))
    result = _memory(server).files.append_section(
        "memory.md", "Agreed terms", "- 2026-10-10 — Net 45 for EU entities (bob, PRJ-3)"
    )
    assert _json(server.only) == {
        "op": "append_section",
        "section": "Agreed terms",
        "content": "- 2026-10-10 — Net 45 for EU entities (bob, PRJ-3)",
        "level": 2,
        "create_if_missing": True,
    }
    assert result.bytes == 175


def test_golden_p3_append_section_creates_a_missing_section():
    key_decisions = {"heading": "Key decisions", "level": 2, "slug": "key-decisions", "start_line": 11, "end_line": 12, "bytes": 43}
    server = Server(
        ok(_write_body(bytes=165, section=key_decisions, section_created=True, sections=VECTOR_A + [key_decisions]), status=201)
    )
    result = _memory(server).files.append_section("memory.md", "Key decisions", "- Use Stripe for billing")
    assert _json(server.only) == {
        "op": "append_section",
        "section": "Key decisions",
        "content": "- Use Stripe for billing",
        "level": 2,
        "create_if_missing": True,
    }
    assert result.section_created is True
    assert result.section == MemoryFileSection(**key_decisions)
    assert [s.slug for s in result.sections][-1] == "key-decisions"


def test_golden_p4_replace_section_by_slug():
    server = Server(ok(_write_body(bytes=77, op="replace_section")))
    result = _memory(server).files.replace_section("memory.md", "agreed-terms", "- Net 60", expected_version=7)
    request = server.only
    assert request.method == "PATCH"
    assert request.headers["If-Match"] == '"7"'
    assert _json(request) == {
        "op": "replace_section",
        "section": "agreed-terms",
        "content": "- Net 60",
        "level": 2,
        "create_if_missing": True,
    }
    assert (result.bytes, result.op) == (77, "replace_section")


# ── 7. read ──────────────────────────────────────────────────────────


def test_read_parses_the_memory_file():
    server = Server(ok(READ_BODY, etag='"7"'))
    file = _memory(server).files.read("memory.md", section="Agreed terms")
    request = server.only
    assert (request.method, request.url.path) == ("GET", "/v1/memory/files/memory.md")
    assert request.url.params["section"] == "Agreed terms"
    for absent in ("version", "outline", "offset", "max_bytes"):
        assert absent not in request.url.params

    assert isinstance(file, MemoryFile) and isinstance(file, MemoryFileInfo)
    assert (file.entity_id, file.partition, file.name, file.version) == (OWNER, PARTITION, "memory.md", 7)
    assert file.is_current is True
    assert file.content == AGREED_CONTENT
    assert file.section == MemoryFileSection(**AGREED_TERMS)
    assert file.sections == [MemoryFileSection(**s) for s in VECTOR_A]
    assert (file.offset, file.next_offset, file.total_bytes, file.truncated) == (0, None, 71, False)
    assert (file.author, file.principal, file.op) == ("scoped:ws_42:user:u_3", "key:ak_1a2b3c4d", "append_section")
    assert (file.created_at, file.updated_at) == ("2026-09-01T10:00:00Z", "2026-10-10T09:12:44Z")


def test_read_maps_every_option_to_the_query():
    server = Server(ok(READ_BODY), ok({**READ_BODY, "content": "", "section": None}))
    files = _memory(server).files
    files.read("memory.md", version=6, offset=10, max_bytes=4096)
    files.read("memory.md", outline=True)
    paged, outline = server.requests
    assert (paged.url.params["version"], paged.url.params["offset"], paged.url.params["max_bytes"]) == ("6", "10", "4096")
    assert "outline" not in paged.url.params
    assert outline.url.params["outline"] == "true"


def test_read_takes_a_missing_version_from_the_etag():
    body = {k: v for k, v in READ_BODY.items() if k != "version"}
    server = Server(ok(body, etag='"7"'))
    assert _memory(server).files.read("memory.md").version == 7


def test_read_keeps_the_body_version_over_the_etag():
    server = Server(ok(READ_BODY, etag='"9"'))
    assert _memory(server).files.read("memory.md").version == 7


def test_read_tolerates_missing_optional_fields():
    server = Server(ok({"name": "memory.md", "version": 2, "content": "hi\n"}))
    file = _memory(server).files.read("memory.md")
    assert (file.version, file.content, file.sections, file.section, file.next_offset) == (2, "hi\n", [], None, None)
    assert file.partition is None and file.author is None


@pytest.mark.parametrize(
    "etag,expected",
    [('"7"', 7), ('"123456"', 123456), ("7", 7), ('W/"7"', 7), ('"abc"', None), ("", None), (None, None), ('"-1"', None)],
)
def test_parse_etag_version(etag, expected):
    assert parse_etag_version(etag) == expected


# ── 8. write ─────────────────────────────────────────────────────────


def test_write_puts_raw_markdown_and_parses_a_create():
    created = _write_body(version=1, previous_version=None, file_created=True, op="put", section=None, sections=[])
    server = Server(ok(created, status=201, etag='"1"'))
    result = _memory(server).files.write(
        "memory.md", "# Acme memory\nIntro — line.\n", create=True, author="scoped:ws_42:user:u_3"
    )
    request = server.only
    assert (request.method, request.url.path) == ("PUT", "/v1/memory/files/memory.md")
    assert request.content == "# Acme memory\nIntro — line.\n".encode("utf-8")
    assert request.headers["Content-Type"] == "text/markdown; charset=utf-8"
    assert request.url.params["author"] == "scoped:ws_42:user:u_3"
    assert isinstance(result, MemoryFileWriteResult)
    assert (result.version, result.file_created, result.previous_version, result.unchanged) == (1, True, None, False)
    assert result.section is None and result.sections == []


def test_write_parses_an_unchanged_result_and_omits_an_empty_author():
    server = Server(ok(_write_body(version=7, unchanged=True, previous_version=7)))
    result = _memory(server).files.write("memory.md", "same", expected_version=7, author="")
    assert "author" not in server.only.url.params
    assert result.unchanged is True and result.version == 7


def test_write_takes_a_missing_version_from_the_etag():
    body = {k: v for k, v in _write_body().items() if k != "version"}
    server = Server(ok(body, etag='"8"'))
    assert _memory(server).files.write("memory.md", "x", expected_version=7).version == 8


def test_write_allows_empty_content_client_side():
    server = Server(err(400, "invalid_input"))
    with pytest.raises(AetherApiError):
        _memory(server).files.write("memory.md", "", create=True)
    assert server.only.content == b""


# ── 9. list / versions / diff / grep / delete ────────────────────────


def test_list_maps_options_and_parses_the_page():
    server = Server(ok({"files": [INFO, {**INFO, "name": "notes.md", "author": None}], "next_cursor": None}))
    page = _memory(server).files.list(prefix="m", limit=10, cursor="c1")
    params = server.only.url.params
    assert (params["prefix"], params["limit"], params["cursor"]) == ("m", "10", "c1")
    assert [f.name for f in page.files] == ["memory.md", "notes.md"]
    assert page.files[1].author is None
    assert page.next_cursor is None


def test_versions_maps_limit_and_parses_newest_first():
    body = {
        **{k: INFO[k] for k in ("entity_id", "partition", "name", "file_id")},
        "current_version": 7,
        "versions": [
            {"doc_id": "d7", "version": 7, "bytes": 122, "content_hash": "h7", "author": None,
             "principal": "key:ak_1", "op": "append_section", "updated_at": "2026-10-10T09:12:44Z", "is_current": True},
            {"doc_id": "d6", "version": 6, "bytes": 70, "content_hash": "h6", "author": "bob",
             "principal": "key:ak_1", "op": "put", "updated_at": "2026-10-09T09:00:00Z", "is_current": False},
        ],
    }
    server = Server(ok(body))
    versions = _memory(server).files.versions("memory.md", 2)
    request = server.only
    assert (request.method, request.url.path) == ("GET", "/v1/memory/files/memory.md/versions")
    assert request.url.params["limit"] == "2"
    assert isinstance(versions, MemoryFileVersions)
    assert versions.current_version == 7
    assert [(v.version, v.is_current, v.author) for v in versions.versions] == [(7, True, None), (6, False, "bob")]


def test_diff_sends_context_by_default_and_versions_when_given():
    body = {
        **{k: INFO[k] for k in ("entity_id", "partition", "name", "file_id")},
        "from_version": 6,
        "to_version": 7,
        "context": 3,
        "diff": "--- memory.md@v6\n+++ memory.md@v7\n@@ -4,3 +4,4 @@\n",
        "added_lines": 1,
        "removed_lines": 0,
        "approximate": False,
        "truncated": False,
    }
    server = Server(ok(body), ok(body))
    files = _memory(server).files
    diff = files.diff("memory.md")
    files.diff("memory.md", from_version=0, to_version=7, context=0)
    default, explicit = server.requests
    assert (default.method, default.url.path) == ("GET", "/v1/memory/files/memory.md/diff")
    assert default.url.params["context"] == "3"
    assert "from" not in default.url.params and "to" not in default.url.params
    assert (explicit.url.params["from"], explicit.url.params["to"], explicit.url.params["context"]) == ("0", "7", "0")
    assert isinstance(diff, MemoryFileDiff)
    assert (diff.from_version, diff.to_version, diff.added_lines, diff.approximate) == (6, 7, 1, False)
    assert diff.diff.startswith("--- memory.md@v6\n")


@pytest.mark.parametrize("kwargs", [{"from_version": -1}, {"to_version": 0}, {"to_version": True}])
def test_diff_rejects_out_of_range_versions(kwargs):
    server = Server()
    with pytest.raises(ValueError):
        _memory(server).files.diff("memory.md", **kwargs)
    assert server.requests == []


def test_grep_sends_every_default_and_parses_matches():
    body = {
        "matches": [
            {
                "entity_id": OWNER,
                "partition": PARTITION,
                "name": "memory.md",
                "version": 7,
                "line": 5,
                "section": "Agreed terms",
                "section_slug": "agreed-terms",
                "text": "- 2026-09-01 — Net 30 payment terms (alice, ISS-12)",
                "before": ["## Agreed terms"],
                "after": [""],
            }
        ],
        "truncated": False,
        "truncated_reason": None,
        "files_scanned": 1,
        "bytes_scanned": 122,
    }
    server = Server(ok(body))
    result = _memory(server).files.grep("Net 30")
    assert _json(server.only) == {
        "entity_id": OWNER,
        "pattern": "Net 30",
        "regex": False,
        "ignore_case": False,
        "context": 0,
        "max_matches": 100,
    }
    assert isinstance(result, MemoryGrepResult)
    [match] = result.matches
    assert (match.name, match.version, match.line, match.section_slug) == ("memory.md", 7, 5, "agreed-terms")
    assert (match.before, match.after) == (["## Agreed terms"], [""])
    assert (result.truncated, result.truncated_reason, result.files_scanned, result.bytes_scanned) == (False, None, 1, 122)


def test_grep_maps_options_and_files():
    server = Server(ok({"matches": [], "truncated": True, "truncated_reason": "max_matches"}))
    result = _memory(server).files.grep(
        r"net\s+\d+", regex=True, ignore_case=True, files=["memory.md"], context=1, max_matches=5
    )
    assert _json(server.only) == {
        "entity_id": OWNER,
        "pattern": r"net\s+\d+",
        "regex": True,
        "ignore_case": True,
        "files": ["memory.md"],
        "context": 1,
        "max_matches": 5,
    }
    assert result.truncated is True and result.truncated_reason == "max_matches"


def test_grep_validates_pattern_and_files_before_any_http_call():
    server = Server()
    files = _memory(server).files
    with pytest.raises(ValueError, match="pattern"):
        files.grep("")
    with pytest.raises(ValueError, match="pattern"):
        _client(server).grep_memory_files("")
    with pytest.raises(ValueError, match="files"):
        files.grep("net", files="memory.md")
    assert server.requests == []


def test_delete_hard_maps_and_parses():
    body = {
        **{k: INFO[k] for k in ("entity_id", "partition", "name", "file_id")},
        "version": 7,
        "hard": True,
        "versions_deleted": 7,
    }
    server = Server(ok(body))
    result = _memory(server).files.delete("memory.md", hard=True)
    request = server.only
    assert (request.method, request.url.path) == ("DELETE", "/v1/memory/files/memory.md")
    assert request.url.params["hard"] == "true"
    assert isinstance(result, MemoryFileDeleteResult)
    assert (result.version, result.hard, result.versions_deleted) == (7, True, 7)


def test_soft_delete_sends_no_hard_param():
    server = Server(ok({"version": 7, "hard": False, "versions_deleted": 1}))
    _memory(server).files.delete("memory.md")
    assert "hard" not in server.only.url.params


# ── 10. Validation ───────────────────────────────────────────────────


def test_argument_errors_raise_before_any_http_call():
    server = Server()
    files = _memory(server).files
    client = _client(server)
    with pytest.raises(ValueError, match="content"):
        files.append("memory.md", "")
    with pytest.raises(ValueError, match="content"):
        files.append_section("memory.md", "Notes", "")
    with pytest.raises(ValueError, match="section"):
        files.append_section("memory.md", "  ", "x")
    with pytest.raises(ValueError, match="section"):
        files.replace_section("memory.md", "", "x", expected_version=1)
    with pytest.raises(ValueError, match="content"):
        files.write("memory.md", None, create=True)
    for call in (
        lambda: client.get_summary(""),
        lambda: client.put_summary("", "x"),
        lambda: client.regenerate_summary(""),
    ):
        with pytest.raises(ValueError, match="doc_id"):
            call()
    with pytest.raises(ValueError, match="expected_revision"):
        client.put_summary("doc-1", "x", expected_revision="")
    assert server.requests == []


def test_whitespace_content_is_allowed_for_appends():
    server = Server(ok(_write_body()))
    _memory(server).files.append("memory.md", "  ")
    assert _json(server.only)["content"] == "  "


# ── 11. Summaries ────────────────────────────────────────────────────


def test_get_summary_parses_a_ready_summary_through_a_partition_handle():
    server = Server(ok(_summary_body(stale=True, truncated_input=True), etag='"r3f9a1c2b7d4e5f60"'))
    summary = _client(server).partition(PARTITION).get_summary("doc-1")
    request = server.only
    assert (request.method, request.url.path) == ("GET", "/v1/documents/doc-1/summary")
    assert request.url.params["partition"] == PARTITION
    assert summary == DocumentSummary(
        doc_id="doc-1",
        state="ready",
        reason=None,
        error=None,
        summary_doc_id="sum-1",
        content="## Summary\n- Net 30.\n",
        revision="r3f9a1c2b7d4e5f60",
        authored_by="aether",
        generator="aether:model-x:2026-10-10.v1",
        prompt_version="2026-10-10.v1",
        generated_at="2026-10-10T09:00:00Z",
        source_cid="cid-1",
        stale=True,
        truncated_input=True,
    )


def test_get_summary_parses_a_none_state_with_nulls():
    server = Server(ok({"doc_id": "doc-1", "state": "none"}))
    summary = _client(server).get_summary("doc-1")
    assert "partition" not in server.only.url.params
    assert summary == DocumentSummary(doc_id="doc-1", state="none")
    assert summary.content is None and summary.revision is None
    assert summary.stale is False and summary.truncated_input is False


def test_put_summary_sends_json_content():
    server = Server(ok(_summary_body(authored_by="user", generator="user:key:ak_1"), status=201))
    summary = _client(server).put_summary("doc-1", "## Summary\n- Mine.\n")
    request = server.only
    assert (request.method, request.url.path) == ("PUT", "/v1/documents/doc-1/summary")
    assert request.headers["Content-Type"] == "application/json"
    assert _json(request) == {"content": "## Summary\n- Mine.\n"}
    assert "If-Match" not in request.headers
    assert summary.authored_by == "user"


def test_regenerate_summary_force_and_default():
    pending = _summary_body(state="pending")
    server = Server(ok(pending, status=202), ok(pending, status=202))
    client = _client(server)
    summary = client.regenerate_summary("doc-1", force=True)
    client.regenerate_summary("doc-1")
    forced, plain = server.requests
    assert (forced.method, forced.url.path) == ("POST", "/v1/documents/doc-1/summary")
    assert forced.url.params["force"] == "true"
    assert "force" not in plain.url.params
    assert forced.content == b""
    assert summary.state == "pending" and summary.content == "## Summary\n- Net 30.\n"


# ── 12. summarize at ingest ──────────────────────────────────────────


@pytest.mark.parametrize("value,sent", [(True, "true"), (False, "false"), (None, None)])
def test_insert_text_summarize_is_tri_state(value, sent):
    server = Server(ok(_record(summary="pending" if value else None)))
    record = _client(server).insert_text("hello", summarize=value)
    assert server.only.url.params.get("summarize") == sent
    assert record.summary == ("pending" if value else None)


def test_document_record_summary_parses_on_get():
    server = Server(ok(_record(summary="skipped")))
    assert _client(server).get("doc-1").summary == "skipped"


def test_insert_insert_stream_and_insert_async_forward_summarize(tmp_path):
    path = tmp_path / "report.md"
    path.write_text("# Report\n")
    server = Server(
        ok(_record(summary="pending")),
        ok(_record(summary="skipped")),
        ok({"job_id": "job-1", "status": "queued", "poll_url": "/documents/jobs/job-1"}, status=202),
        ok({"status": "completed", "doc_id": "doc-1", "summary_state": "pending"}),
    )
    client = _client(server)
    assert client.insert(path, summarize=True).summary == "pending"
    assert client.insert_stream(iter([b"abc"]), summarize=False).summary == "skipped"
    job = client.insert_async(path, summarize=True)
    status = client.wait_for_job(job["job_id"], poll_interval=0)
    insert, stream, enqueue, poll = server.requests
    assert insert.url.params["summarize"] == "true"
    assert stream.url.params["summarize"] == "false"
    assert (enqueue.url.path, enqueue.url.params["summarize"]) == ("/v1/documents/async", "true")
    assert poll.url.path == "/v1/documents/jobs/job-1"
    assert status["summary_state"] == "pending"


def test_batch_insert_summarize_at_batch_and_item_level():
    server = Server(ok({"results": [_record(summary="pending"), _record(doc_id="doc-2")]}), ok({"results": []}))
    client = _client(server)
    records = client.batch_insert(
        [
            BatchInsertItem(filename="a.txt", content="a", summarize=False),
            BatchInsertItem(filename="b.txt", content="b"),
        ],
        summarize=True,
    )
    client.batch_insert([BatchInsertItem(filename="c.txt", content="c")])
    with_summary, without = server.requests
    body = _json(with_summary)
    assert body["summarize"] is True
    assert body["documents"][0]["summarize"] is False
    assert "summarize" not in body["documents"][1]
    assert "summarize" not in _json(without)
    assert "summarize" not in _json(without)["documents"][0]
    assert [r.summary for r in records] == ["pending", None]


def test_memory_summarize_constructor_default_and_per_call_override():
    server = Server(*(ok(_record()) for _ in range(5)))
    on = _memory(server, summarize=True)
    on.remember("a")
    on.remember("b", summarize=False)
    off = _memory(server)
    off.remember("c")
    off.remember("d", summarize=True)
    _memory(server, summarize=False).remember("e")
    sent = [r.url.params.get("summarize") for r in server.requests]
    assert sent == ["true", "false", None, "true", "false"]


def test_memory_summarize_is_text_only():
    server = Server()
    mem = _memory(server)
    with pytest.raises(ValueError, match="summarize is valid only for text"):
        mem.remember(image=b"\x89PNG\r\n\x1a\n", content_type="image/png", summarize=True)
    assert server.requests == []


# ── 13. Error passthrough ────────────────────────────────────────────


def test_billing_errors_from_a_memory_file_write_keep_their_type():
    server = Server(err(402, "credit_exhausted"))
    with pytest.raises(CreditExhaustedError):
        _memory(server).files.append("memory.md", "- a")


@pytest.mark.parametrize(
    "response,call",
    [
        (err(428, "precondition_required"), lambda f, c: f.write("memory.md", "x", expected_version=1)),
        (
            err(409, "memory_file_section_ambiguous", candidates=[{"heading": "Notes", "slug": "notes", "start_line": 16}]),
            lambda f, c: f.append_section("memory.md", "notes", "- a"),
        ),
        (err(404, "feature_not_enabled", feature="memory_files"), lambda f, c: f.list()),
        (err(404, "memory_file_not_found"), lambda f, c: f.read("memory.md")),
        (err(412, "summary_revision_conflict", current_revision="r2"), lambda f, c: c.put_summary("doc-1", "x", expected_revision="r1")),
    ],
)
def test_other_codes_surface_as_the_base_api_error(response, call):
    server = Server(response)
    mem = _memory(server)
    with pytest.raises(AetherApiError) as exc:
        call(mem.files, mem.client)
    e = exc.value
    assert type(e) is AetherApiError
    assert e.status_code == response.status_code
    assert e.error_code == response.json()["code"]
    if e.error_code == "memory_file_section_ambiguous":
        assert e.body["candidates"][0]["slug"] == "notes"
    if e.error_code == "summary_revision_conflict":
        assert e.body["current_revision"] == "r2"


# ── 14. Search hits ──────────────────────────────────────────────────

MEMORY_FILE_HIT = {"name": "memory.md", "version": 7, "section": "Agreed terms", "section_slug": "agreed-terms"}


def _hit(**extra):
    hit = {"doc_id": "doc-1", "score": 90, "content_type": "text/markdown", "passage": "Net 30", "tags": [], "metadata": {}}
    hit.update(extra)
    return hit


def test_search_hits_parse_memory_file():
    server = Server(
        ok({"query": "q", "results": [_hit(memory_file=MEMORY_FILE_HIT), _hit(doc_id="doc-2")]}),
        ok({"results": [_hit(memory_file=MEMORY_FILE_HIT)]}),
        ok({"results": [{"query": "q", "results": [_hit(memory_file=MEMORY_FILE_HIT)]}]}),
        ok({"results": [_hit(memory_file=MEMORY_FILE_HIT)], "trace": {}}),
        ok({"query": "q", "results": [_hit(memory_file=MEMORY_FILE_HIT, content="# Acme\n")]}),
    )
    client = _client(server)
    with_file, without_file = client.search("q")
    assert with_file.memory_file == MemoryFileHit(
        name="memory.md", version=7, section="Agreed terms", section_slug="agreed-terms"
    )
    assert without_file.memory_file is None
    assert client.search_by_vector([0.1, 0.2])[0].memory_file.name == "memory.md"
    assert client.batch_search([BatchSearchQuery(q="q")])[0].results[0].memory_file.version == 7
    assert client.search_trace("q").results[0].memory_file.section == "Agreed terms"
    [retrieved] = client.retrieve("q")
    assert retrieved.memory_file.section_slug == "agreed-terms"


def test_search_hit_memory_file_tolerates_null_section():
    server = Server(ok({"query": "q", "results": [_hit(memory_file={"name": "memory.md", "version": 3})]}))
    [hit] = _client(server).search("q")
    assert hit.memory_file == MemoryFileHit(name="memory.md", version=3)


# ── 15. Async ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_read_write_and_scoping():
    server = Server(ok(READ_BODY, etag='"7"'), ok(_write_body(), etag='"8"'))
    mem = _async_memory(server)
    assert isinstance(mem.files, AsyncMemoryFiles) and mem.files is mem.files
    file = await mem.files.read("memory.md", section="Agreed terms")
    result = await mem.files.write("memory.md", file.content, expected_version=file.version)
    read, write = server.requests
    assert read.url.params["entity_id"] == OWNER and read.url.params["partition"] == PARTITION
    assert file.section == MemoryFileSection(**AGREED_TERMS)
    assert write.headers["If-Match"] == '"7"' and "If-None-Match" not in write.headers
    assert _is_uuid4(write.headers["Idempotency-Key"])
    assert result.version == 8
    await mem.client._client.aclose()


@pytest.mark.asyncio
async def test_async_preconditions_and_names_validate_before_any_http_call():
    server = Server()
    files = _async_memory(server).files
    with pytest.raises(ValueError, match="exactly one"):
        await files.write("memory.md", "x")
    with pytest.raises(ValueError, match="exactly one"):
        await files.replace_section("memory.md", "Notes", "x", expected_version=1, create=True)
    with pytest.raises(ValueError, match="invalid memory file name"):
        await files.read("grep")
    with pytest.raises(ValueError, match="pattern"):
        await files.grep("")
    assert server.requests == []


@pytest.mark.asyncio
async def test_async_busy_retry_reuses_the_idempotency_key():
    busy = httpx.Response(503, json={"error": "busy", "code": "memory_file_busy"}, headers={"Retry-After": "0"})
    server = Server(busy, ok(_write_body()))
    mem = _async_memory(server, max_retries=1)
    await mem.files.append_section("memory.md", "Agreed terms", "x")
    first, second = server.requests
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]
    assert _json(second) == {
        "op": "append_section",
        "section": "Agreed terms",
        "content": "x",
        "level": 2,
        "create_if_missing": True,
    }


@pytest.mark.asyncio
async def test_async_list_grep_versions_diff_delete():
    server = Server(
        ok({"files": [INFO], "next_cursor": None}),
        ok({"matches": [{"name": "memory.md", "line": 5, "text": "x"}], "files_scanned": 1}),
        ok({"current_version": 7, "versions": []}),
        ok({"from_version": 6, "to_version": 7, "context": 3, "diff": ""}),
        ok({"version": 7, "hard": True, "versions_deleted": 7}),
    )
    files = _async_memory(server).files
    page = await files.list(prefix="mem")
    grep = await files.grep("x", files=["memory.md"])
    versions = await files.versions("memory.md", limit=5)
    diff = await files.diff("memory.md", from_version=6)
    deleted = await files.delete("memory.md", hard=True)
    listing, grep_req, versions_req, diff_req, delete_req = server.requests
    assert listing.url.params["entity_id"] == OWNER and listing.url.params["prefix"] == "mem"
    assert _json(grep_req)["entity_id"] == OWNER and _json(grep_req)["files"] == ["memory.md"]
    assert versions_req.url.params["limit"] == "5"
    assert (diff_req.url.params["from"], diff_req.url.params["context"]) == ("6", "3")
    assert delete_req.method == "DELETE" and delete_req.url.params["hard"] == "true"
    assert page.files[0].name == "memory.md"
    assert grep.matches[0].line == 5 and grep.files_scanned == 1
    assert versions.current_version == 7
    assert diff.from_version == 6
    assert deleted.versions_deleted == 7


@pytest.mark.asyncio
async def test_async_version_conflict_is_typed():
    server = Server(err(412, "memory_file_version_conflict", current_version=8))
    with pytest.raises(VersionConflictError) as exc:
        await _async_memory(server).files.write("memory.md", "x", expected_version=7)
    assert exc.value.current_version == 8 and exc.value.status_code == 412


@pytest.mark.asyncio
async def test_async_raw_cross_entity_calls():
    server = Server(ok({"files": [], "next_cursor": "n"}), ok({"matches": []}))
    client = _async_client(server).partition(PARTITION)
    page = await client.list_memory_files(entity_prefix=PREFIX)
    await client.grep_memory_files("net", entity_prefix=PREFIX, ignore_case=True)
    listing, grep = server.requests
    assert listing.url.params["entity_prefix"] == PREFIX and listing.url.params["partition"] == PARTITION
    assert _json(grep)["entity_prefix"] == PREFIX and _json(grep)["ignore_case"] is True
    assert page.next_cursor == "n"
    with pytest.raises(ValueError, match="mutually exclusive"):
        await client.list_memory_files(entity_id=OWNER, entity_prefix=PREFIX)


@pytest.mark.asyncio
async def test_async_summaries():
    server = Server(
        ok(_summary_body()),
        ok(_summary_body(authored_by="user")),
        ok(_summary_body(state="pending"), status=202),
    )
    client = _async_client(server).partition(PARTITION)
    summary = await client.get_summary("doc-1")
    put = await client.put_summary("doc-1", "mine", expected_revision="r3f9a1c2b7d4e5f60")
    pending = await client.regenerate_summary("doc-1", force=True)
    get_req, put_req, post_req = server.requests
    assert get_req.url.params["partition"] == PARTITION
    assert put_req.method == "PUT" and _json(put_req) == {"content": "mine"}
    assert put_req.headers["If-Match"] == '"r3f9a1c2b7d4e5f60"'
    assert "Idempotency-Key" not in put_req.headers
    assert post_req.method == "POST" and post_req.url.params["force"] == "true"
    assert (summary.state, put.authored_by, pending.state) == ("ready", "user", "pending")
    with pytest.raises(ValueError, match="doc_id"):
        await client.get_summary("")


@pytest.mark.asyncio
async def test_async_summarize_on_insert_batch_and_memory():
    server = Server(
        ok(_record(summary="pending")),
        ok(_record()),
        ok({"results": [_record(summary="pending")]}),
        ok(_record()),
        ok(_record()),
        ok(_record()),
    )
    client = _async_client(server)
    record = await client.insert_text("hello", summarize=True)
    await client.insert_text("hello")
    await client.batch_insert([BatchInsertItem(filename="a.txt", content="a", summarize=True)], summarize=False)
    mem = AsyncMemory(OWNER, client=client, summarize=True)
    await mem.remember("a")
    await mem.remember("b", summarize=False)
    await AsyncMemory(OWNER, client=client).remember("c")
    text_on, text_unset, batch, rem_default, rem_override, rem_unset = server.requests
    assert record.summary == "pending"
    assert text_on.url.params["summarize"] == "true"
    assert "summarize" not in text_unset.url.params
    assert _json(batch)["summarize"] is False and _json(batch)["documents"][0]["summarize"] is True
    assert rem_default.url.params["summarize"] == "true"
    assert rem_override.url.params["summarize"] == "false"
    assert "summarize" not in rem_unset.url.params
    with pytest.raises(ValueError, match="summarize is valid only for text"):
        await mem.remember(audio=b"RIFF0000WAVE", content_type="audio/wav", transcript="hi", summarize=False)


@pytest.mark.asyncio
async def test_async_search_and_retrieve_carry_memory_file():
    server = Server(
        ok({"query": "q", "results": [_hit(memory_file=MEMORY_FILE_HIT)]}),
        ok({"query": "q", "results": [_hit(memory_file=MEMORY_FILE_HIT, content="# Acme\n")]}),
    )
    client = _async_client(server)
    [hit] = await client.search("q")
    [retrieved] = await client.retrieve("q")
    assert hit.memory_file.name == "memory.md"
    assert retrieved.memory_file == MemoryFileHit(**MEMORY_FILE_HIT)
