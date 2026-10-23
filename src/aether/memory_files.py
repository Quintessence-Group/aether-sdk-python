"""Memory files: named, versioned markdown files kept per entity.

``Memory.files`` (:class:`MemoryFiles`, or :class:`AsyncMemoryFiles` on an
``AsyncMemory``) reads and edits the files of one entity.
:meth:`AetherClient.list_memory_files` and
:meth:`AetherClient.grep_memory_files` work across entities.

Every call goes through the raw client's transport, so its partition handle,
acting principal, retries and error types apply unchanged. Argument errors
raise ``ValueError`` before any HTTP call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import quote

from ._internal import new_idempotency_key, parse_etag_version
from .models import (
    MemoryFile,
    MemoryFileDeleteResult,
    MemoryFileDiff,
    MemoryFilePage,
    MemoryFileVersions,
    MemoryFileWriteResult,
    MemoryGrepResult,
    parse_memory_file,
    parse_memory_file_delete_result,
    parse_memory_file_diff,
    parse_memory_file_page,
    parse_memory_file_versions,
    parse_memory_file_write_result,
    parse_memory_grep_result,
)

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_NAME_RULE = (
    "use 1-128 letters, digits, '.', '_' or '-', starting with a letter or "
    'digit, without "..", and not "grep"'
)


def validate_memory_file_name(name: str) -> str:
    """Check a memory-file name and return it unchanged.

    A valid name is 1-128 ASCII letters, digits, ``.``, ``_`` or ``-``,
    starts with a letter or digit, does not contain ``..`` and is not exactly
    ``grep``. Names are case-sensitive (``Memory.md`` and ``memory.md`` are
    different files).

    Raises:
        ValueError: the name is not valid.
    """
    if (
        not isinstance(name, str)
        or not _NAME_RE.fullmatch(name)
        or ".." in name
        or name == "grep"
    ):
        raise ValueError(f"invalid memory file name {name!r}: {_NAME_RULE}")
    return name


# ── Request building (shared by the sync and async clients and facades) ──


@dataclass
class _Call:
    """One memory-file request, validated and mapped to the wire."""

    method: str
    path: str
    params: dict[str, Any]
    #: ``(json_body, etag_header) -> result``.
    parse: Callable[[Any, Optional[str]], Any]
    headers: dict[str, str] = field(default_factory=dict)
    json: Any = None
    content: Optional[bytes] = None


def _request_kwargs(client: Any, call: _Call) -> dict[str, Any]:
    params = dict(call.params)
    partition = getattr(client, "_partition", None)
    if partition:
        params["partition"] = partition
    kwargs: dict[str, Any] = {"params": params}
    if call.headers:
        # Built once per call: the transport reuses these headers (including
        # the Idempotency-Key) on every retry of this call.
        kwargs["headers"] = dict(call.headers)
    if call.json is not None:
        kwargs["json"] = call.json
    if call.content is not None:
        kwargs["content"] = call.content
    return kwargs


def _execute(client: Any, call: _Call) -> Any:
    resp = client._request_with_retry(call.method, call.path, **_request_kwargs(client, call))
    client._raise_for_status(resp)
    return call.parse(resp.json(), resp.headers.get("etag"))


async def _aexecute(client: Any, call: _Call) -> Any:
    resp = await client._request_with_retry(
        call.method, call.path, **_request_kwargs(client, call)
    )
    client._raise_for_status(resp)
    return call.parse(resp.json(), resp.headers.get("etag"))


def _with_etag_version(body: Any, etag: Optional[str]) -> Any:
    """Fill a missing ``version`` from the response ``ETag`` (``"7"`` -> 7)."""
    if isinstance(body, dict) and body.get("version") is None:
        version = parse_etag_version(etag)
        if version is not None:
            body = {**body, "version": version}
    return body


def _parse_read(body: Any, etag: Optional[str]) -> MemoryFile:
    return parse_memory_file(_with_etag_version(body, etag))


def _parse_write(body: Any, etag: Optional[str]) -> MemoryFileWriteResult:
    return parse_memory_file_write_result(_with_etag_version(body, etag))


def _body_only(parse: Callable[[Any], Any]) -> Callable[[Any, Optional[str]], Any]:
    return lambda body, _etag: parse(body)


def _file_path(name: str, suffix: str = "") -> str:
    return f"/memory/files/{quote(validate_memory_file_name(name), safe='')}{suffix}"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_expected_version(expected_version: Optional[int]) -> None:
    if expected_version is not None and (
        not _is_int(expected_version) or expected_version < 1
    ):
        raise ValueError("expected_version must be an integer >= 1")


def _if_match(expected_version: Optional[int]) -> dict[str, str]:
    _check_expected_version(expected_version)
    if expected_version is None:
        return {}
    return {"If-Match": f'"{expected_version}"'}


def _exactly_one_precondition(
    expected_version: Optional[int], create: bool
) -> dict[str, str]:
    _check_expected_version(expected_version)
    if (expected_version is None) == (not create):
        raise ValueError("pass exactly one of expected_version or create")
    if create:
        return {"If-None-Match": "*"}
    return {"If-Match": f'"{expected_version}"'}


def _require_content(content: Any) -> None:
    if not isinstance(content, str) or content == "":
        raise ValueError("content cannot be empty")


def _require_str_content(content: Any) -> None:
    if not isinstance(content, str):
        raise ValueError("content must be a string")


def _require_section(section: Any) -> None:
    if not isinstance(section, str) or not section.strip():
        raise ValueError("section cannot be empty")


def _check_entity_scope(entity_id: Optional[str], entity_prefix: Optional[str]) -> None:
    if entity_id and entity_prefix:
        raise ValueError("entity_id and entity_prefix are mutually exclusive")


def _write_params(entity_id: str, author: Optional[str]) -> dict[str, Any]:
    params: dict[str, Any] = {"entity_id": entity_id}
    if author:
        params["author"] = author
    return params


def _with_idempotency_key(headers: dict[str, str], idempotency_key: Optional[str]) -> dict[str, str]:
    headers["Idempotency-Key"] = idempotency_key or new_idempotency_key()
    return headers


def _read_file_call(
    entity_id: str,
    name: str,
    *,
    version: Optional[int],
    section: Optional[str],
    outline: bool,
    offset: Optional[int],
    max_bytes: Optional[int],
) -> _Call:
    path = _file_path(name)
    params: dict[str, Any] = {"entity_id": entity_id}
    if version is not None:
        params["version"] = version
    if section is not None:
        params["section"] = section
    if outline:
        params["outline"] = "true"
    if offset is not None:
        params["offset"] = offset
    if max_bytes is not None:
        params["max_bytes"] = max_bytes
    return _Call("GET", path, params, _parse_read)


def _write_file_call(
    entity_id: str,
    name: str,
    content: str,
    *,
    expected_version: Optional[int],
    create: bool,
    author: Optional[str],
    idempotency_key: Optional[str],
) -> _Call:
    path = _file_path(name)
    _require_str_content(content)
    headers = _exactly_one_precondition(expected_version, create)
    headers["Content-Type"] = "text/markdown; charset=utf-8"
    return _Call(
        "PUT",
        path,
        _write_params(entity_id, author),
        _parse_write,
        headers=_with_idempotency_key(headers, idempotency_key),
        content=content.encode("utf-8"),
    )


def _append_file_call(
    entity_id: str,
    name: str,
    content: str,
    *,
    expected_version: Optional[int],
    create_if_missing: bool,
    author: Optional[str],
    idempotency_key: Optional[str],
) -> _Call:
    path = _file_path(name)
    _require_content(content)
    headers = _if_match(expected_version)
    body = {"op": "append", "content": content, "create_if_missing": bool(create_if_missing)}
    return _Call(
        "PATCH",
        path,
        _write_params(entity_id, author),
        _parse_write,
        headers=_with_idempotency_key(headers, idempotency_key),
        json=body,
    )


def _append_section_call(
    entity_id: str,
    name: str,
    section: str,
    content: str,
    *,
    level: int,
    expected_version: Optional[int],
    create_if_missing: bool,
    author: Optional[str],
    idempotency_key: Optional[str],
) -> _Call:
    path = _file_path(name)
    _require_section(section)
    _require_content(content)
    headers = _if_match(expected_version)
    body = {
        "op": "append_section",
        "section": section,
        "content": content,
        "level": level,
        "create_if_missing": bool(create_if_missing),
    }
    return _Call(
        "PATCH",
        path,
        _write_params(entity_id, author),
        _parse_write,
        headers=_with_idempotency_key(headers, idempotency_key),
        json=body,
    )


def _replace_section_call(
    entity_id: str,
    name: str,
    section: str,
    content: str,
    *,
    expected_version: Optional[int],
    create: bool,
    level: int,
    create_if_missing: bool,
    author: Optional[str],
    idempotency_key: Optional[str],
) -> _Call:
    path = _file_path(name)
    _require_section(section)
    _require_str_content(content)
    headers = _exactly_one_precondition(expected_version, create)
    body = {
        "op": "replace_section",
        "section": section,
        "content": content,
        "level": level,
        "create_if_missing": bool(create_if_missing),
    }
    return _Call(
        "PATCH",
        path,
        _write_params(entity_id, author),
        _parse_write,
        headers=_with_idempotency_key(headers, idempotency_key),
        json=body,
    )


def _versions_call(entity_id: str, name: str, *, limit: Optional[int]) -> _Call:
    path = _file_path(name, "/versions")
    params: dict[str, Any] = {"entity_id": entity_id}
    if limit is not None:
        params["limit"] = limit
    return _Call("GET", path, params, _body_only(parse_memory_file_versions))


def _diff_call(
    entity_id: str,
    name: str,
    *,
    from_version: Optional[int],
    to_version: Optional[int],
    context: int,
) -> _Call:
    path = _file_path(name, "/diff")
    if from_version is not None and (not _is_int(from_version) or from_version < 0):
        raise ValueError("from_version must be an integer >= 0")
    if to_version is not None and (not _is_int(to_version) or to_version < 1):
        raise ValueError("to_version must be an integer >= 1")
    params: dict[str, Any] = {"entity_id": entity_id}
    if from_version is not None:
        params["from"] = from_version
    if to_version is not None:
        params["to"] = to_version
    params["context"] = context
    return _Call("GET", path, params, _body_only(parse_memory_file_diff))


def _delete_file_call(
    entity_id: str,
    name: str,
    *,
    hard: bool,
    expected_version: Optional[int],
) -> _Call:
    path = _file_path(name)
    headers = _if_match(expected_version)
    params: dict[str, Any] = {"entity_id": entity_id}
    if hard:
        params["hard"] = "true"
    return _Call(
        "DELETE",
        path,
        params,
        _body_only(parse_memory_file_delete_result),
        headers=headers,
    )


def _list_files_call(
    *,
    entity_id: Optional[str],
    entity_prefix: Optional[str],
    prefix: Optional[str],
    limit: Optional[int],
    cursor: Optional[str],
) -> _Call:
    _check_entity_scope(entity_id, entity_prefix)
    params: dict[str, Any] = {}
    if entity_id:
        params["entity_id"] = entity_id
    if entity_prefix:
        params["entity_prefix"] = entity_prefix
    if prefix:
        params["prefix"] = prefix
    if limit is not None:
        params["limit"] = limit
    if cursor:
        params["cursor"] = cursor
    return _Call("GET", "/memory/files", params, _body_only(parse_memory_file_page))


def _grep_files_call(
    pattern: str,
    *,
    entity_id: Optional[str],
    entity_prefix: Optional[str],
    regex: bool,
    ignore_case: bool,
    files: Optional[list[str]],
    context: int,
    max_matches: int,
) -> _Call:
    if not isinstance(pattern, str) or pattern == "":
        raise ValueError("pattern cannot be empty")
    _check_entity_scope(entity_id, entity_prefix)
    body: dict[str, Any] = {
        "pattern": pattern,
        "regex": bool(regex),
        "ignore_case": bool(ignore_case),
        "context": context,
        "max_matches": max_matches,
    }
    if entity_id:
        body["entity_id"] = entity_id
    elif entity_prefix:
        body["entity_prefix"] = entity_prefix
    if files is not None:
        if isinstance(files, str):
            raise ValueError("files must be a list of memory file names, not a string")
        body["files"] = list(files)
    return _Call(
        "POST",
        "/memory/files/grep",
        {},
        _body_only(parse_memory_grep_result),
        json=body,
    )


# ── Entity-scoped facade ─────────────────────────────────────────────


class MemoryFiles:
    """The memory files of one entity: ``Memory("acme").files``.

    A memory file is a named markdown file (``memory.md``, ``notes.md``, …)
    kept per entity and versioned: every change writes a new version, older
    versions stay readable, and you can diff any two of them. Files are
    addressed by ``(partition, entity_id, name)``; the entity is the
    ``Memory``'s and the partition is its client's partition handle, if any.

    Reads return the whole file or one section (by heading text or slug),
    paged by bytes. Writes either replace the whole file under a version
    precondition (:meth:`write`), or edit it section by section
    (:meth:`append`, :meth:`append_section`, :meth:`replace_section`).

    Read-modify-write::

        file = memory.files.read("memory.md")
        new_content = edit(file.content)
        try:
            memory.files.write("memory.md", new_content, expected_version=file.version)
        except VersionConflictError:
            ...  # someone else wrote first: re-read, re-apply, write again

    Every write sends an ``Idempotency-Key`` that is reused when the call is
    retried, so a retried append is applied once. A 404 with code
    ``feature_not_enabled`` means memory files are not enabled on your
    deployment.
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    @property
    def entity_id(self) -> str:
        """The entity whose files this handle reads and writes."""
        return self._memory.entity_id

    def _run(self, call: _Call) -> Any:
        return _execute(self._memory.client, call)

    def list(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> MemoryFilePage:
        """List the entity's current files, ordered by name.

        *prefix* keeps names starting with it (case-sensitive). *limit* is the
        page size (server default 100, at most 1000). Pass the page's
        ``next_cursor`` back as *cursor* for the next page.
        """
        return self._memory.client.list_memory_files(
            entity_id=self._memory.entity_id, prefix=prefix, limit=limit, cursor=cursor
        )

    def read(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        section: Optional[str] = None,
        outline: bool = False,
        offset: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> MemoryFile:
        """Read a file (the current version unless *version* is given).

        *section* narrows the content to one section, selected by heading text
        (trimmed, case-insensitive) or by slug; the section's heading line is
        included. ``outline=True`` returns only ``sections`` (with
        ``content == ""``) and cannot be combined with *section*, *offset* or
        *max_bytes*. Page through a large file with *max_bytes* and
        ``offset=file.next_offset`` while ``file.truncated``; pages end on a
        line boundary where possible.

        Raises:
            ValueError: invalid *name*.
            AetherApiError: 404 ``memory_file_not_found`` /
                ``memory_file_version_not_found`` /
                ``memory_file_section_not_found``; 409
                ``memory_file_section_ambiguous`` (several headings match: use
                the slug; ``body["candidates"]`` lists them).
        """
        return self._run(
            _read_file_call(
                self._memory.entity_id,
                name,
                version=version,
                section=section,
                outline=outline,
                offset=offset,
                max_bytes=max_bytes,
            )
        )

    def write(
        self,
        name: str,
        content: str,
        *,
        expected_version: Optional[int] = None,
        create: bool = False,
        author: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> MemoryFileWriteResult:
        """Replace the whole file with *content*.

        Pass exactly one precondition: ``expected_version=file.version`` to
        overwrite the version you read, or ``create=True`` to create a file
        that must not exist yet. Writing content identical to the current
        version writes nothing and returns ``unchanged=True``.

        *author* is a free-text attribution label stored with the version
        (for example the end user on whose behalf you write); it is never
        used for authorization. *idempotency_key* overrides the key generated
        for this call.

        Raises:
            ValueError: invalid *name*, or not exactly one of
                *expected_version* / *create*.
            VersionConflictError: the precondition failed (the file changed,
                is missing, or already exists); see ``current_version``.
        """
        return self._run(
            _write_file_call(
                self._memory.entity_id,
                name,
                content,
                expected_version=expected_version,
                create=create,
                author=author,
                idempotency_key=idempotency_key,
            )
        )

    def append(
        self,
        name: str,
        content: str,
        *,
        expected_version: Optional[int] = None,
        create_if_missing: bool = True,
        author: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> MemoryFileWriteResult:
        """Append *content* to the end of the file (a newline is added first
        if the file does not end with one, and after *content* if it lacks
        one).

        Needs no precondition: the append applies to whatever the current
        version is, so concurrent appends are never lost. Pass
        *expected_version* to require a specific current version anyway. A
        missing file is created unless ``create_if_missing=False``.

        Raises:
            ValueError: invalid *name* or empty *content*.
            VersionConflictError: *expected_version* was given and did not
                match.
        """
        return self._run(
            _append_file_call(
                self._memory.entity_id,
                name,
                content,
                expected_version=expected_version,
                create_if_missing=create_if_missing,
                author=author,
                idempotency_key=idempotency_key,
            )
        )

    def append_section(
        self,
        name: str,
        section: str,
        content: str,
        *,
        level: int = 2,
        expected_version: Optional[int] = None,
        create_if_missing: bool = True,
        author: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> MemoryFileWriteResult:
        """Add *content* at the end of a section's own text, ahead of its
        sub-sections, so entries land under their heading in order.

        *section* is a heading text or slug. A missing section (and a missing
        file) is created at the end of the file with a level-*level* heading
        unless ``create_if_missing=False``. Like :meth:`append`, it needs no
        precondition.

        Raises:
            ValueError: invalid *name*, blank *section* or empty *content*.
            VersionConflictError: *expected_version* was given and did not
                match.
            AetherApiError: 409 ``memory_file_section_ambiguous``; 404
                ``memory_file_section_not_found`` /
                ``memory_file_not_found`` with ``create_if_missing=False``.
        """
        return self._run(
            _append_section_call(
                self._memory.entity_id,
                name,
                section,
                content,
                level=level,
                expected_version=expected_version,
                create_if_missing=create_if_missing,
                author=author,
                idempotency_key=idempotency_key,
            )
        )

    def replace_section(
        self,
        name: str,
        section: str,
        content: str,
        *,
        expected_version: Optional[int] = None,
        create: bool = False,
        level: int = 2,
        create_if_missing: bool = True,
        author: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> MemoryFileWriteResult:
        """Replace the body of a section (everything under its heading,
        including nested sub-sections) with *content*; the heading is kept.
        ``content=""`` clears the section.

        Pass exactly one precondition, as for :meth:`write`:
        ``expected_version`` for an existing file or ``create=True`` for a new
        one. A missing section is created at the end of the file unless
        ``create_if_missing=False``.

        Raises:
            ValueError: invalid *name*, blank *section*, or not exactly one of
                *expected_version* / *create*.
            VersionConflictError: the precondition failed.
            AetherApiError: 409 ``memory_file_section_ambiguous``; 404
                ``memory_file_section_not_found`` with
                ``create_if_missing=False``.
        """
        return self._run(
            _replace_section_call(
                self._memory.entity_id,
                name,
                section,
                content,
                expected_version=expected_version,
                create=create,
                level=level,
                create_if_missing=create_if_missing,
                author=author,
                idempotency_key=idempotency_key,
            )
        )

    def versions(self, name: str, limit: Optional[int] = None) -> MemoryFileVersions:
        """List a file's retained versions, newest first (*limit*: 1-100)."""
        return self._run(_versions_call(self._memory.entity_id, name, limit=limit))

    def diff(
        self,
        name: str,
        *,
        from_version: Optional[int] = None,
        to_version: Optional[int] = None,
        context: int = 3,
    ) -> MemoryFileDiff:
        """Unified diff between two versions of a file.

        *to_version* defaults to the current version and *from_version* to
        the one before it; ``from_version=0`` diffs against the empty file.
        *context* is the number of unchanged lines shown around each change.

        Raises:
            ValueError: invalid *name*, ``from_version < 0`` or
                ``to_version < 1``.
            AetherApiError: 404 ``memory_file_not_found`` /
                ``memory_file_version_not_found``.
        """
        return self._run(
            _diff_call(
                self._memory.entity_id,
                name,
                from_version=from_version,
                to_version=to_version,
                context=context,
            )
        )

    def grep(
        self,
        pattern: str,
        *,
        regex: bool = False,
        ignore_case: bool = False,
        files: Optional[list[str]] = None,
        context: int = 0,
        max_matches: int = 100,
    ) -> MemoryGrepResult:
        """Search the current version of the entity's files line by line.

        *pattern* is a literal substring unless ``regex=True`` (linear-time
        regular expressions: no back-references or look-around). *files*
        restricts the search to those names. *context* adds up to that many
        lines before and after each match. The scan stops early, with
        ``truncated=True``, at *max_matches* or a server limit.

        Raises:
            ValueError: empty *pattern*.
            AetherApiError: 400 ``invalid_pattern``.
        """
        return self._memory.client.grep_memory_files(
            pattern,
            entity_id=self._memory.entity_id,
            regex=regex,
            ignore_case=ignore_case,
            files=files,
            context=context,
            max_matches=max_matches,
        )

    def delete(
        self,
        name: str,
        *,
        hard: bool = False,
        expected_version: Optional[int] = None,
    ) -> MemoryFileDeleteResult:
        """Delete a file.

        A soft delete (default) hides the file from list, read, grep and
        search; writing the name again continues its version numbering.
        ``hard=True`` irreversibly purges every version (use it for erasure);
        the name then starts again at version 1. Pass *expected_version* to
        delete only if the file is still at that version.

        Raises:
            ValueError: invalid *name*.
            VersionConflictError: *expected_version* did not match.
            AetherApiError: 404 ``memory_file_not_found``.
        """
        return self._run(
            _delete_file_call(
                self._memory.entity_id,
                name,
                hard=hard,
                expected_version=expected_version,
            )
        )


class AsyncMemoryFiles:
    """Async twin of :class:`MemoryFiles`: ``AsyncMemory("acme").files``.

    Same operations, arguments, validation and errors, with ``await``.
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    @property
    def entity_id(self) -> str:
        """The entity whose files this handle reads and writes."""
        return self._memory.entity_id

    async def _run(self, call: _Call) -> Any:
        return await _aexecute(self._memory.client, call)

    async def list(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> MemoryFilePage:
        """See :meth:`MemoryFiles.list`."""
        return await self._memory.client.list_memory_files(
            entity_id=self._memory.entity_id, prefix=prefix, limit=limit, cursor=cursor
        )

    async def read(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        section: Optional[str] = None,
        outline: bool = False,
        offset: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> MemoryFile:
        """See :meth:`MemoryFiles.read`."""
        return await self._run(
            _read_file_call(
                self._memory.entity_id,
                name,
                version=version,
                section=section,
                outline=outline,
                offset=offset,
                max_bytes=max_bytes,
            )
        )

    async def write(
        self,
        name: str,
        content: str,
        *,
        expected_version: Optional[int] = None,
        create: bool = False,
        author: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> MemoryFileWriteResult:
        """See :meth:`MemoryFiles.write`."""
        return await self._run(
            _write_file_call(
                self._memory.entity_id,
                name,
                content,
                expected_version=expected_version,
                create=create,
                author=author,
                idempotency_key=idempotency_key,
            )
        )

    async def append(
        self,
        name: str,
        content: str,
        *,
        expected_version: Optional[int] = None,
        create_if_missing: bool = True,
        author: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> MemoryFileWriteResult:
        """See :meth:`MemoryFiles.append`."""
        return await self._run(
            _append_file_call(
                self._memory.entity_id,
                name,
                content,
                expected_version=expected_version,
                create_if_missing=create_if_missing,
                author=author,
                idempotency_key=idempotency_key,
            )
        )

    async def append_section(
        self,
        name: str,
        section: str,
        content: str,
        *,
        level: int = 2,
        expected_version: Optional[int] = None,
        create_if_missing: bool = True,
        author: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> MemoryFileWriteResult:
        """See :meth:`MemoryFiles.append_section`."""
        return await self._run(
            _append_section_call(
                self._memory.entity_id,
                name,
                section,
                content,
                level=level,
                expected_version=expected_version,
                create_if_missing=create_if_missing,
                author=author,
                idempotency_key=idempotency_key,
            )
        )

    async def replace_section(
        self,
        name: str,
        section: str,
        content: str,
        *,
        expected_version: Optional[int] = None,
        create: bool = False,
        level: int = 2,
        create_if_missing: bool = True,
        author: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> MemoryFileWriteResult:
        """See :meth:`MemoryFiles.replace_section`."""
        return await self._run(
            _replace_section_call(
                self._memory.entity_id,
                name,
                section,
                content,
                expected_version=expected_version,
                create=create,
                level=level,
                create_if_missing=create_if_missing,
                author=author,
                idempotency_key=idempotency_key,
            )
        )

    async def versions(self, name: str, limit: Optional[int] = None) -> MemoryFileVersions:
        """See :meth:`MemoryFiles.versions`."""
        return await self._run(_versions_call(self._memory.entity_id, name, limit=limit))

    async def diff(
        self,
        name: str,
        *,
        from_version: Optional[int] = None,
        to_version: Optional[int] = None,
        context: int = 3,
    ) -> MemoryFileDiff:
        """See :meth:`MemoryFiles.diff`."""
        return await self._run(
            _diff_call(
                self._memory.entity_id,
                name,
                from_version=from_version,
                to_version=to_version,
                context=context,
            )
        )

    async def grep(
        self,
        pattern: str,
        *,
        regex: bool = False,
        ignore_case: bool = False,
        files: Optional[list[str]] = None,
        context: int = 0,
        max_matches: int = 100,
    ) -> MemoryGrepResult:
        """See :meth:`MemoryFiles.grep`."""
        return await self._memory.client.grep_memory_files(
            pattern,
            entity_id=self._memory.entity_id,
            regex=regex,
            ignore_case=ignore_case,
            files=files,
            context=context,
            max_matches=max_matches,
        )

    async def delete(
        self,
        name: str,
        *,
        hard: bool = False,
        expected_version: Optional[int] = None,
    ) -> MemoryFileDeleteResult:
        """See :meth:`MemoryFiles.delete`."""
        return await self._run(
            _delete_file_call(
                self._memory.entity_id,
                name,
                hard=hard,
                expected_version=expected_version,
            )
        )
