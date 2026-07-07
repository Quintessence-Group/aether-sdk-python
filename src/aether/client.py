"""Aether Python SDK client."""

from __future__ import annotations

import copy
import json
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import quote

import httpx

from ._internal import USER_AGENT, enforce_secure_base_url, new_idempotency_key
from .errors import AetherApiError, AetherNetworkError, aether_api_error_from_response
from .schema import SchemaClient

# A partition id must be non-empty and at most this many characters, matching
# the server's constraint. Validated client-side before any HTTP call.
_MAX_PARTITION_LEN = 256

# Canonical public API version prefix. Every data route (documents, search,
# memory, partitions, archive) is served under this prefix. The public probe
# route ``GET /status`` is intentionally unversioned.
_API_VERSION_PREFIX = "/v1"
_UNVERSIONED_PATHS = frozenset({"/status"})


def _versioned_path(url: str) -> str:
    """Prefix a relative request path with the public API version.

    The prefix always goes before the path itself, never into the query
    string. Unversioned probe routes (``/status``) pass through untouched.
    """
    path = url.split("?", 1)[0]
    if path in _UNVERSIONED_PATHS:
        return url
    return _API_VERSION_PREFIX + url


def _json_query_value(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _validate_partition(partition_id: str) -> str:
    """Validate a partition id client-side (never a network round-trip).

    Mirrors the entity_id validation style: reject empty/whitespace-only ids
    and ids longer than 256 characters with a ``ValueError``.
    """
    if not isinstance(partition_id, str) or not partition_id.strip():
        raise ValueError("partition cannot be empty")
    if len(partition_id) > _MAX_PARTITION_LEN:
        raise ValueError(
            f"partition must be at most {_MAX_PARTITION_LEN} characters "
            f"(got {len(partition_id)})"
        )
    return partition_id


def _with_partition_guard(url: str, partition: Optional[str]) -> str:
    """Append the partition guard query param to an ID-addressed route.

    A no-op for an unscoped client (``partition`` is ``None``), which keeps
    the pre-handle behavior byte-identical. On a scoped handle the guard makes
    a document outside the partition indistinguishable from a missing one.
    """
    if not partition:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}partition={quote(partition)}"


from .models import (
    AggregateResult,
    AuditProof,
    AuditRecord,
    BatchInsertItem,
    BatchSearchQuery,
    BatchSearchResponse,
    DocumentPage,
    DocumentRecord,
    QueryGroup,
    EntityBackfillReport,
    IngestResult,
    IsolationCheck,
    Metadata,
    MetadataFilter,
    NodeStatus,
    resolve_content_type,
    PartitionInfo,
    PartitionList,
    PartitionWarning,
    RetrievalResult,
    SearchResult,
    SearchTrace,
    TracedSearch,
)


class AetherClient:
    """Client for the Aether dRAG HTTP API.

    Usage::

        client = AetherClient()  # reads AETHER_API_KEY from env

    Configuration is resolved in priority order:
      base_url: explicit param > AETHER_BASE_URL env var > https://api.aetherdb.ai
      api_key:  explicit param > AETHER_API_KEY env var > None
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
    ):
        self.base_url = (
            base_url or os.environ.get("AETHER_BASE_URL") or "https://api.aetherdb.ai"
        ).rstrip("/")
        api_key = api_key or os.environ.get("AETHER_API_KEY")
        enforce_secure_base_url(self.base_url, api_key)
        headers = {"User-Agent": USER_AGENT}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        # Partition scope: None on the base client (unscoped). Set on a
        # scoped clone produced by ``partition()``. When set it is injected into
        # every partition-aware read/write so a partition can never be forgotten.
        self._partition: Optional[str] = None
        # Ownership: the base client owns and closes the transport; a scoped
        # clone shares it and must not close it (mirrors the Memory facade's
        # ``_owns_client`` pattern).
        self._owns_transport = True

    def partition(self, partition_id: str) -> "AetherClient":
        """Return a partition-scoped clone of this client.

        Every read and write on the returned handle is automatically scoped to
        ``partition_id`` — there is no per-call partition argument, so the scope
        cannot be forgotten. Reaching a different partition requires obtaining a
        separate handle via another ``partition()`` call. The top-level client
        stays unscoped and behaves exactly as before.

        A multi-tenant key requires a partition on every call; an unscoped call
        under such a key is rejected by the server. Under a single-tenant key,
        unscoped calls operate on the default partition.

        ID-addressed calls (``get``, ``download`` / ``download_text``,
        ``delete``, ``restore``, ``update``) carry the scope as a **guard**: a
        document outside this partition is indistinguishable from a missing one
        (the same not-found error), so a bare doc id can never reach another
        partition's document. ``backfill_entity_from_tags`` likewise constrains
        its scan to this partition.

        The clone shares this client's transport and all configuration (base
        url, auth, timeout, retries, backoff). It does **not** own the transport:
        closing the clone is a no-op, and the base client still closes it.
        Re-scoping is allowed — ``client.partition("a").partition("b")`` yields a
        handle scoped to ``"b"``.
        """
        scoped = copy.copy(self)
        scoped._partition = _validate_partition(partition_id)
        scoped._owns_transport = False
        return scoped

    def _raise_for_status(self, resp: httpx.Response) -> None:
        """Raise AetherApiError or AetherNetworkError for non-2xx responses."""
        if resp.is_success:
            return
        request_id = resp.headers.get("x-request-id")
        try:
            body = resp.json()
            message = body.get("error", resp.reason_phrase or "Unknown error")
            error_code = body.get("code")
        except Exception:
            body = {}
            message = resp.reason_phrase or f"HTTP {resp.status_code}"
            error_code = None
        raise aether_api_error_from_response(
            resp.status_code,
            message,
            error_code=error_code,
            request_id=request_id,
            body=body,
        )

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Send a single HTTP request (no retries).

        The relative *url* is rewritten under the ``/v1`` API version prefix
        here, at the transport boundary, so every caller (including the
        Memory facade) versions its data routes in one place.
        """
        return self._client.request(method, _versioned_path(url), **kwargs)

    def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Send an HTTP request with exponential backoff retry for transient errors."""
        last_error: Optional[Exception] = None
        resp: Optional[httpx.Response] = None
        max_attempts = self._max_retries + 1

        # Attach a stable idempotency key to non-idempotent writes so the server
        # can deduplicate a retry whose original response was lost in transit.
        headers = dict(kwargs.pop("headers", None) or {})
        if method.upper() == "POST" and "Idempotency-Key" not in headers:
            headers["Idempotency-Key"] = new_idempotency_key()
        if headers:
            kwargs["headers"] = headers

        for attempt in range(max_attempts):
            try:
                resp = self._request(method, url, **kwargs)
                if resp.is_success:
                    return resp
                # Build error and check if retryable
                request_id = resp.headers.get("x-request-id")
                try:
                    body = resp.json()
                    message = body.get("error", resp.reason_phrase or "Unknown error")
                    error_code = body.get("code")
                except Exception:
                    body = {}
                    message = resp.reason_phrase or f"HTTP {resp.status_code}"
                    error_code = None
                api_err = aether_api_error_from_response(
                    resp.status_code, message,
                    error_code=error_code, request_id=request_id, body=body,
                )
                if not api_err.is_retryable or attempt == max_attempts - 1:
                    raise api_err
                last_error = api_err
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == max_attempts - 1:
                    raise AetherNetworkError(
                        f"Connection failed after {max_attempts} attempts: {e}", cause=e,
                    ) from e
                last_error = e

            # Exponential backoff with jitter (0-50% of base delay added)
            delay = self._retry_base_delay * (2 ** attempt)
            jitter = random.uniform(0, delay * 0.5)
            delay += jitter

            # Respect Retry-After header for 429 responses
            if (
                isinstance(last_error, AetherApiError)
                and last_error.status_code == 429
                and resp is not None
            ):
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass

            time.sleep(delay)

        # Should not reach here, but just in case
        if isinstance(last_error, (AetherApiError, AetherNetworkError)):
            raise last_error
        raise AetherNetworkError(f"Request failed after {max_attempts} attempts")

    def close(self):
        """Close the underlying transport.

        A no-op on a partition-scoped clone, which shares (but does not own) the
        base client's transport — only the base client closes it.
        """
        if self._owns_transport:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Documents ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_document_record(d: dict) -> DocumentRecord:
        """Map a document JSON object to a :class:`DocumentRecord`.

        New optional fields default so payloads from older servers (which omit
        them) still parse: ``tags`` -> ``[]``, ``source`` / ``partition`` ->
        ``None`` (for ``partition``, ``None`` is also the default partition).
        """
        return DocumentRecord(
            doc_id=d["doc_id"],
            cid=d.get("cid", ""),
            title=d.get("title"),
            content_type=d.get("content_type", ""),
            size_bytes=d.get("size_bytes", 0),
            chunks=d.get("chunks", 0),
            vectors=d.get("vectors", 0),
            version=d.get("version", 1),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            entity_id=d.get("entity_id"),
            tags=list(d.get("tags") or []),
            source=d.get("source"),
            partition=d.get("partition"),
            metadata=dict(d.get("metadata") or {}),
        )

    @staticmethod
    def _parse_audit_record(d: dict) -> AuditRecord:
        """Map an audit-record JSON object to an :class:`AuditRecord`.

        ``proof`` is parsed into an :class:`AuditProof` when present and left
        ``None`` otherwise. Inside the proof, ``content_id`` is optional and
        defaults to ``None`` when the key is omitted (e.g. for a tombstone).
        """
        proof_raw = d.get("proof")
        proof = None
        if proof_raw is not None:
            proof = AuditProof(
                content_id=proof_raw.get("content_id"),
                lamport=proof_raw.get("lamport", 0),
                node_id=proof_raw.get("node_id", ""),
                public_key=proof_raw.get("public_key", ""),
                signature=proof_raw.get("signature", ""),
                verified=proof_raw.get("verified", False),
            )
        return AuditRecord(
            at=d["at"],
            actor=d["actor"],
            action=d["action"],
            resource=d["resource"],
            outcome=d["outcome"],
            source=d["source"],
            proof=proof,
        )

    def insert(
        self,
        file_path: str | Path,
        content_type: str | None = None,
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
        source: str | None = None,
        metadata: Metadata | None = None,
    ) -> DocumentRecord:
        """Insert a document from a file path.

        If *content_type* is not given it is guessed from the file extension
        (e.g. ``.pdf`` -> ``application/pdf``).  Falls back to
        ``application/octet-stream`` for unknown extensions.

        Pass *entity_id* to associate the document with an entity (e.g. a
        user or customer id) for later filtering on search and list.

        Pass *source* to tag the document with its origin (e.g. a system or
        channel name) for later filtering on search and list.
        """
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if overlap is not None and overlap < 0:
            raise ValueError("overlap must be non-negative")
        path = Path(file_path)
        data = path.read_bytes()
        filename = path.name
        if content_type is None:
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        url = f"/documents?filename={quote(filename)}&content_type={quote(content_type)}"
        if tags:
            url += f"&tags={quote(','.join(tags))}"
        if chunk_size is not None:
            url += f"&chunk_size={chunk_size}"
        if overlap is not None:
            url += f"&overlap={overlap}"
        if entity_id:
            url += f"&entity_id={quote(entity_id)}"
        if source:
            url += f"&source={quote(source)}"
        if metadata is not None:
            url += f"&metadata={quote(_json_query_value(metadata))}"
        if self._partition:
            url += f"&partition={quote(self._partition)}"

        resp = self._request_with_retry("POST", url, content=data)
        self._raise_for_status(resp)
        body = resp.json()
        return self._parse_document_record(body)

    def ingest_files(
        self,
        paths: "list[str | Path]",
        *,
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
        source: str | None = None,
        metadata: Metadata | None = None,
        raise_on_error: bool = False,
    ) -> list[IngestResult]:
        """Ingest many files in one call.

        Each file is inserted independently with :meth:`insert`; the content
        type is resolved from the extension (``.md`` / ``.txt`` / ``.pdf`` and
        friends). Chunking uses the server defaults unless *chunk_size* /
        *overlap* are given.

        A file the engine cannot ingest — an unsupported or binary type, or one
        that needs the server-side document parser when it is not configured, or
        a file over the size limit — is **reported** in the returned results
        (``status="skipped"``) rather than aborting the batch or failing
        silently. Set ``raise_on_error=True`` to re-raise instead.

        Returns one :class:`IngestResult` per input path, in order.
        """
        results: list[IngestResult] = []
        for p in paths:
            path = Path(p)
            content_type = resolve_content_type(path)
            try:
                record = self.insert(
                    path,
                    content_type=content_type,
                    tags=tags,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    entity_id=entity_id,
                    source=source,
                    metadata=metadata,
                )
                results.append(
                    IngestResult(
                        path=str(path),
                        status="ingested",
                        doc_id=record.doc_id,
                        content_type=content_type,
                    )
                )
            except AetherApiError as e:
                if raise_on_error:
                    raise
                # 413 (too large) / 415 (unsupported media) / 422 (unprocessable
                # — unknown or binary type the parser can't handle) are per-file
                # rejections the caller can't fix by retrying: report, don't abort.
                status = "skipped" if e.status_code in (413, 415, 422) else "error"
                results.append(
                    IngestResult(
                        path=str(path),
                        status=status,
                        content_type=content_type,
                        error=str(e),
                    )
                )
            except OSError as e:
                if raise_on_error:
                    raise
                results.append(
                    IngestResult(path=str(path), status="error", error=str(e))
                )
        return results

    def ingest_directory(
        self,
        directory: str | Path,
        *,
        extensions: list[str] | None = None,
        recursive: bool = True,
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
        source: str | None = None,
        metadata: Metadata | None = None,
        raise_on_error: bool = False,
    ) -> list[IngestResult]:
        """Ingest every file under *directory*.

        Walks *directory* (recursively by default) and ingests each file via
        :meth:`ingest_files`. Pass *extensions* (e.g. ``[".md", ".txt", ".pdf"]``)
        to restrict which files are loaded; leading dots and case are optional.
        See :meth:`ingest_files` for how unsupported files are reported.
        """
        base = Path(directory)
        if not base.is_dir():
            raise ValueError(f"not a directory: {directory}")
        allowed: set[str] | None = None
        if extensions is not None:
            allowed = {
                (e if e.startswith(".") else f".{e}").lower() for e in extensions
            }
        walker = base.rglob("*") if recursive else base.glob("*")
        files = sorted(
            p
            for p in walker
            if p.is_file() and (allowed is None or p.suffix.lower() in allowed)
        )
        return self.ingest_files(
            files,
            tags=tags,
            chunk_size=chunk_size,
            overlap=overlap,
            entity_id=entity_id,
            source=source,
            metadata=metadata,
            raise_on_error=raise_on_error,
        )

    def insert_text(
        self,
        text: str,
        filename: str = "text.txt",
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
        source: str | None = None,
        metadata: Metadata | None = None,
        extract_facts: bool = False,
    ) -> DocumentRecord:
        """Insert raw text content.

        Pass *entity_id* to associate the document with an entity (e.g. a
        user or customer id) for later filtering on search and list.

        Pass *source* to tag the document with its origin (e.g. a system or
        channel name) for later filtering on search and list.

        Pass ``extract_facts=True`` to also distill the text into atomic facts
        server-side; each fact is stored as a sibling document tagged
        ``kind:fact`` and linked to this document. Requires fact extraction to
        be configured on the node, otherwise the request fails.
        """
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if overlap is not None and overlap < 0:
            raise ValueError("overlap must be non-negative")
        url = f"/documents?filename={quote(filename)}&content_type=text%2Fplain"
        if tags:
            url += f"&tags={quote(','.join(tags))}"
        if chunk_size is not None:
            url += f"&chunk_size={chunk_size}"
        if overlap is not None:
            url += f"&overlap={overlap}"
        if entity_id:
            url += f"&entity_id={quote(entity_id)}"
        if source:
            url += f"&source={quote(source)}"
        if metadata is not None:
            url += f"&metadata={quote(_json_query_value(metadata))}"
        if extract_facts:
            url += "&extract_facts=true"
        if self._partition:
            url += f"&partition={quote(self._partition)}"
        resp = self._request_with_retry("POST", url, content=text.encode("utf-8"))
        self._raise_for_status(resp)
        body = resp.json()
        return self._parse_document_record(body)

    def insert_stream(
        self,
        stream,
        filename: str = "upload.bin",
        content_type: str = "application/octet-stream",
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
        source: str | None = None,
        metadata: Metadata | None = None,
    ) -> DocumentRecord:
        """Insert a document from a file-like object or iterator without loading everything into memory.

        ``stream`` can be any object that ``httpx`` accepts as streaming content:
        a file opened in binary mode, a ``bytes`` iterator, or a generator.

        Pass *entity_id* to associate the document with an entity (e.g. a
        user or customer id) for later filtering on search and list.

        Pass *source* to tag the document with its origin (e.g. a system or
        channel name) for later filtering on search and list.

        Note: streaming uploads bypass the retry wrapper because the stream
        may not be re-readable. Ensure the stream is seekable if you need retries.
        """
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if overlap is not None and overlap < 0:
            raise ValueError("overlap must be non-negative")
        url = f"/documents?filename={quote(filename)}&content_type={quote(content_type)}"
        if tags:
            url += f"&tags={quote(','.join(tags))}"
        if chunk_size is not None:
            url += f"&chunk_size={chunk_size}"
        if overlap is not None:
            url += f"&overlap={overlap}"
        if entity_id:
            url += f"&entity_id={quote(entity_id)}"
        if source:
            url += f"&source={quote(source)}"
        if metadata is not None:
            url += f"&metadata={quote(_json_query_value(metadata))}"
        if self._partition:
            url += f"&partition={quote(self._partition)}"
        resp = self._client.post(
            _versioned_path(url),
            content=stream,
            headers={"Idempotency-Key": new_idempotency_key()},
        )
        self._raise_for_status(resp)
        body = resp.json()
        return self._parse_document_record(body)

    def update(
        self,
        doc_id: str,
        file_path: str | Path,
        content_type: str = "text/plain",
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
        source: str | None = None,
        metadata: Metadata | None = None,
    ) -> DocumentRecord:
        """Update an existing document.

        *entity_id* replaces the stored entity id; omitting it clears any
        existing value (mirrors *tags* semantics). *source* behaves the same
        way — it replaces the stored origin, and omitting it clears it.
        """
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if overlap is not None and overlap < 0:
            raise ValueError("overlap must be non-negative")
        path = Path(file_path)
        data = path.read_bytes()
        filename = path.name

        url = f"/documents/{quote(doc_id)}?filename={quote(filename)}&content_type={quote(content_type)}"
        if tags:
            url += f"&tags={quote(','.join(tags))}"
        if chunk_size is not None:
            url += f"&chunk_size={chunk_size}"
        if overlap is not None:
            url += f"&overlap={overlap}"
        if entity_id:
            url += f"&entity_id={quote(entity_id)}"
        if source:
            url += f"&source={quote(source)}"
        if metadata is not None:
            url += f"&metadata={quote(_json_query_value(metadata))}"
        if self._partition:
            url += f"&partition={quote(self._partition)}"

        resp = self._request_with_retry("PUT", url, content=data)
        self._raise_for_status(resp)
        body = resp.json()
        return self._parse_document_record(body)

    def get(self, doc_id: str) -> DocumentRecord:
        """Get document metadata."""
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        resp = self._request_with_retry(
            "GET", _with_partition_guard(f"/documents/{quote(doc_id)}", self._partition)
        )
        self._raise_for_status(resp)
        body = resp.json()
        return self._parse_document_record(body)

    def lineage(self, doc_id: str) -> list[AuditRecord]:
        """Return the signed provenance/lineage trail for a document.

        Calls ``GET /v1/audit/records/{doc_id}`` and returns the document's
        ordered audit records, each carrying a cryptographic :class:`AuditProof`
        for ledger-sourced events. The endpoint is tenant-scoped by the API key
        and takes no partition parameter. Raises the same not-found error as
        :meth:`get` (404) when the document is unknown.
        """
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        resp = self._request_with_retry("GET", f"/audit/records/{quote(doc_id)}")
        self._raise_for_status(resp)
        body = resp.json()
        return [self._parse_audit_record(r) for r in body.get("records", [])]

    def download(self, doc_id: str, output_path: str | Path) -> int:
        """Download a document to a file. Returns bytes written."""
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        resp = self._request_with_retry(
            "GET", _with_partition_guard(f"/documents/{quote(doc_id)}/download", self._partition)
        )
        self._raise_for_status(resp)
        path = Path(output_path)
        path.write_bytes(resp.content)
        return len(resp.content)

    def download_text(self, doc_id: str) -> str:
        """Download a document and return its content as text."""
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        resp = self._request_with_retry(
            "GET", _with_partition_guard(f"/documents/{quote(doc_id)}/download", self._partition)
        )
        self._raise_for_status(resp)
        return resp.content.decode("utf-8")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        tags: list[str] | None = None,
        entity_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        last_n_days: int | None = None,
        max_distance: float | None = None,
        any_tags: list[str] | None = None,
        content_types: list[str] | None = None,
        sources: list[str] | None = None,
        filter: MetadataFilter | None = None,
        recency_weight: float | None = None,
        half_life_days: float | None = None,
        freshness_weight: float | None = None,
        freshness_half_life_days: float | None = None,
    ) -> list[RetrievalResult]:
        """Search and return results with document content included.

        Combines search() + download_text() into a single call for RAG workflows.
        Results are deduplicated by doc_id (best score wins).
        Uses server-side include_content when available, falling back to per-doc downloads.

        The *entity_id*, *since*, *until*, *last_n_days*, *max_distance*,
        *any_tags*, *content_types*, *sources*, *recency_weight*,
        *half_life_days*, *freshness_weight* and *freshness_half_life_days*
        filters are forwarded to search() — see there for semantics.
        """
        if not query:
            raise ValueError("query cannot be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        results = self.search(
            query,
            k=k,
            include_content=True,
            tags=tags,
            entity_id=entity_id,
            since=since,
            until=until,
            last_n_days=last_n_days,
            max_distance=max_distance,
            any_tags=any_tags,
            content_types=content_types,
            sources=sources,
            filter=filter,
            recency_weight=recency_weight,
            half_life_days=half_life_days,
            freshness_weight=freshness_weight,
            freshness_half_life_days=freshness_half_life_days,
        )

        # Deduplicate by doc_id, keeping the best match (results arrive in
        # descending-score order, so the first occurrence wins)
        seen: dict[str, SearchResult] = {}
        for r in results:
            if r.doc_id not in seen:
                seen[r.doc_id] = r

        retrieval_results = []
        for r in seen.values():
            content = r.content if r.content is not None else self.download_text(r.doc_id)
            retrieval_results.append(
                RetrievalResult(
                    doc_id=r.doc_id,
                    score=r.score,
                    content=content,
                    title=r.title,
                    content_type=r.content_type,
                    passage=r.passage,
                    entity_id=r.entity_id,
                    tags=r.tags,
                    source=r.source,
                    partition=r.partition,
                    metadata=r.metadata,
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                )
            )
        return retrieval_results

    def list(
        self,
        offset: int = 0,
        limit: int = 50,
        entity_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        last_n_days: int | None = None,
        tags: list[str] | None = None,
        any_tags: list[str] | None = None,
        content_types: list[str] | None = None,
        sources: list[str] | None = None,
        filter: MetadataFilter | None = None,
    ) -> DocumentPage:
        """List active documents with pagination.

        Args:
            offset: Number of documents to skip. Default: 0.
            limit: Maximum number of documents to return. Default: 50, max: 1000.
            entity_id: Only return documents associated with this entity id.
            since: Only return documents created at or after this RFC 3339
                timestamp (e.g. ``2026-06-01T00:00:00Z``). Inclusive.
            until: Only return documents created at or before this RFC 3339
                timestamp. Inclusive.
            last_n_days: Only return documents created in the last N days.
                Cannot be combined with ``since``.
            tags: Only return documents carrying **all** of these tags (AND).
            any_tags: Only return documents carrying **at least one** of these
                tags (OR).
            content_types: Only return documents whose content type is one of
                these (OR).
            sources: Only return documents whose source is one of these (OR).

        Returns:
            A :class:`DocumentPage` (a ``list`` subclass) of records, with
            ``.total`` and ``.has_more`` pagination metadata attached.
        """
        params: dict = {"offset": offset, "limit": limit}
        if entity_id:
            params["entity_id"] = entity_id
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if last_n_days is not None:
            params["last_n_days"] = last_n_days
        if tags:
            params["tags"] = ",".join(tags)
        if any_tags:
            params["any_tags"] = ",".join(any_tags)
        if content_types:
            params["content_type"] = ",".join(content_types)
        if sources:
            params["source"] = ",".join(sources)
        if filter is not None:
            params["filter"] = _json_query_value(filter)
        if self._partition:
            params["partition"] = self._partition
        resp = self._request_with_retry("GET","/documents", params=params)
        self._raise_for_status(resp)
        body = resp.json()
        documents = [
            self._parse_document_record({"cid": "", **d})
            for d in body.get("documents", [])
        ]
        return DocumentPage(
            documents,
            total=body.get("total", len(documents)),
            has_more=body.get("has_more", False),
        )

    def query(
        self,
        *,
        filter: MetadataFilter | None = None,
        group_by: list[str] | None = None,
        aggregate: list[dict] | None = None,
        sort: list[dict] | None = None,
        limit: int | None = None,
        offset: int = 0,
        partition: str | None = None,
    ) -> Union[DocumentPage, AggregateResult]:
        """Run a structured analytical query over declared typed fields + built-ins.

        Exact and deterministic — the analytical read path never consults an
        embedding.

        **Mode A** (no ``aggregate``): returns a :class:`DocumentPage` of the
        matching documents, typed-sorted and paginated (``.total`` / ``.has_more``
        page the whole population).

        **Mode B** (with ``aggregate``): returns an :class:`AggregateResult` — the
        matching documents grouped by ``group_by`` and folded into the requested
        aggregates.

        Args:
            filter: The filter grammar — ``{"and"|"or"|"not": …}`` combinators over
                ``{"field", "op", "value"}`` leaves (ops: eq, neq, in, gt, gte, lt,
                lte, between, exists, contains, prefix). Omit to match every
                document in scope.
            group_by: Up to two declared/built-in fields to group by (Mode B).
            aggregate: Aggregates to compute per group, e.g.
                ``[{"op": "sum", "field": "amount", "as": "total"}]``. Ops: count,
                count_distinct, sum, avg, min, max (the numeric ops require an
                int/float field). Passing this selects Mode B.
            sort: Typed multi-key sort — ``[{"by": "<field-or-aggregate>", "dir":
                "asc"|"desc"}]``. Mode A sorts documents by field; Mode B sorts
                groups by an aggregate output or a group_by field.
            limit: Mode A page size (max 1000); Mode B max groups returned.
            offset: Mode A page offset.
            partition: Partition to scope the query to. On a partition-scoped
                handle the scope is already fixed and this argument is ignored.

        Raises:
            AetherApiError: 400 for an unknown field, a type-mismatched literal, a
                non-numeric numeric-aggregate, or a guardrail breach (the scan or
                max-groups cap) — never a silently truncated result.
        """
        body: dict[str, Any] = {}
        if filter is not None:
            body["filter"] = filter
        if group_by:
            body["group_by"] = group_by
        if aggregate:
            body["aggregate"] = aggregate
        if sort:
            body["sort"] = sort
        if limit is not None:
            body["limit"] = limit
        if offset:
            body["offset"] = offset
        scope = self._partition or partition
        if scope:
            body["partition"] = scope

        resp = self._request_with_retry("POST", "/query", json=body)
        self._raise_for_status(resp)
        data = resp.json()
        if aggregate:
            grps = [
                QueryGroup(keys=g.get("keys", {}), aggregates=g.get("aggregates", {}))
                for g in data.get("groups", [])
            ]
            return AggregateResult(
                groups=grps,
                total_groups=data.get("total_groups", len(grps)),
                scanned=data.get("scanned", 0),
            )
        documents = [
            self._parse_document_record({"cid": "", **d})
            for d in data.get("documents", [])
        ]
        return DocumentPage(
            documents,
            total=data.get("total", len(documents)),
            has_more=data.get("has_more", False),
        )

    @property
    def schema(self) -> SchemaClient:
        """Field-schema facade — declare / list / delete the typed fields that
        :meth:`query` filters, sorts, and aggregates over."""
        return SchemaClient(self)

    def delete(self, doc_id: str, hard: bool = False) -> None:
        """Delete a document.

        By default this is a soft delete: the document is tombstoned (hidden
        from list/search) and can be brought back with :meth:`restore`.

        Pass ``hard=True`` for a permanent, **irreversible** delete:
        the document is purged from the primary store and removed from both the
        vector and keyword indexes, and its encryption key is shredded. Nothing
        is recoverable afterwards — this is the right-to-be-forgotten path.
        """
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        path = f"/documents/{quote(doc_id)}"
        if hard:
            path += "?hard=true"
        resp = self._request_with_retry("DELETE", _with_partition_guard(path, self._partition))
        self._raise_for_status(resp)

    def restore(self, doc_id: str) -> None:
        """Restore a tombstoned document."""
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        resp = self._request_with_retry(
            "POST", _with_partition_guard(f"/documents/{quote(doc_id)}/restore", self._partition)
        )
        self._raise_for_status(resp)

    def backfill_entity_from_tags(
        self,
        tag_prefix: str,
        *,
        overwrite: bool = False,
    ) -> EntityBackfillReport:
        """Backfill ``entity_id`` on existing documents from a tag convention.

        For every active document, a tag starting with *tag_prefix*
        (e.g. ``"patient:"``) sets ``entity_id`` to the suffix after the
        prefix when exactly one such tag exists; ambiguous (2+) or absent
        matches are skipped. Documents that already have an ``entity_id``
        are left alone unless *overwrite* is True. Metadata-only — documents
        are not re-embedded.

        Under a partition handle the scan is constrained to that partition;
        a multi-tenant key requires the scope.

        Returns an :class:`EntityBackfillReport` with per-document outcome
        counts.
        """
        if not tag_prefix:
            raise ValueError("tag_prefix cannot be empty")
        body: dict = {"tag_prefix": tag_prefix, "overwrite": overwrite}
        resp = self._request_with_retry(
            "POST",
            _with_partition_guard("/documents/backfill-entity", self._partition),
            json=body,
        )
        self._raise_for_status(resp)
        r = resp.json()
        return EntityBackfillReport(
            scanned=r["scanned"],
            updated=r["updated"],
            skipped_existing=r["skipped_existing"],
            skipped_no_match=r["skipped_no_match"],
            skipped_ambiguous=r["skipped_ambiguous"],
            skipped_invalid=r["skipped_invalid"],
        )

    def move_document(
        self,
        doc_id: str,
        *,
        from_partition: str | None,
        to_partition: str | None,
    ) -> DocumentRecord:
        """Move a document between partitions (metadata-only).

        The only way to re-home a document across named partitions.
        *from_partition* asserts where the document lives **now**;
        *to_partition* names the destination. ``None`` means the default
        partition for either — both are always passed explicitly, so a move
        can never be aimed by a forgotten or implicit scope (a partition
        handle deliberately does **not** scope this call). Content, ``cid``,
        chunks and vectors are unchanged (no re-embed); ``version``
        increments on a real move.

        A wrong *from_partition* assertion, a missing id, or a tombstoned
        document all surface the identical not-found error — the call never
        reveals which partition a document lives in.
        ``to_partition == from_partition`` is an idempotent no-op.

        Returns the updated :class:`DocumentRecord`.
        """
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        if from_partition is not None:
            _validate_partition(from_partition)
        if to_partition is not None:
            _validate_partition(to_partition)
        body: dict = {"to_partition": to_partition, "expect_partition": from_partition}
        resp = self._request_with_retry(
            "POST", f"/documents/{quote(doc_id)}/move", json=body
        )
        self._raise_for_status(resp)
        return self._parse_document_record(resp.json())

    # ── Search ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
        include_content: bool = False,
        tags: list[str] | None = None,
        entity_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        last_n_days: int | None = None,
        max_distance: float | None = None,
        any_tags: list[str] | None = None,
        content_types: list[str] | None = None,
        sources: list[str] | None = None,
        filter: MetadataFilter | None = None,
        recency_weight: float | None = None,
        half_life_days: float | None = None,
        freshness_weight: float | None = None,
        freshness_half_life_days: float | None = None,
    ) -> list[SearchResult]:
        """Similarity search across documents.

        Args:
            entity_id: Only match documents associated with this entity id.
            since: Only match documents created at or after this RFC 3339
                timestamp (e.g. ``2026-06-01T00:00:00Z``). Inclusive.
            until: Only match documents created at or before this RFC 3339
                timestamp. Inclusive.
            last_n_days: Only match documents created in the last N days.
                Cannot be combined with ``since``.
            max_distance: Drop results whose distance exceeds this threshold.
            tags: Only match documents carrying **all** of these tags (AND).
            any_tags: Only match documents carrying **at least one** of these
                tags (OR).
            content_types: Only match documents whose content type is one of
                these (OR).
            sources: Only match documents whose source is one of these (OR).
            recency_weight: Blend recency into ranking, ``0.0``–``1.0``
               . ``0`` (default) is pure similarity; ``1`` is pure
                recency. ``final = (1-w)*similarity + w*recency`` where
                ``recency = exp(-age_days / half_life_days)``.
            half_life_days: Age (in days) at which the recency contribution
                halves. Defaults server-side to 30; only consulted when
                ``recency_weight > 0``. Must be > 0.
            freshness_weight: Blend freshness into ranking, ``0.0``–``1.0``.
                Boosts recently *updated* documents (``updated_at``, falling
                back to ``created_at``). Composes with ``recency_weight``;
                the server rejects ``recency_weight + freshness_weight > 1``.
                May require a Scale plan or higher.
            freshness_half_life_days: Age (in days) at which the freshness
                contribution halves. Defaults server-side to 14; only
                consulted when ``freshness_weight > 0``. Must be > 0.
        """
        if not query:
            raise ValueError("query cannot be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        params: dict = {"q": query, "k": k}
        if include_content:
            params["include_content"] = "true"
        if tags:
            params["tags"] = ",".join(tags)
        if entity_id:
            params["entity_id"] = entity_id
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if last_n_days is not None:
            params["last_n_days"] = last_n_days
        if max_distance is not None:
            params["max_distance"] = max_distance
        if any_tags:
            params["any_tags"] = ",".join(any_tags)
        if content_types:
            params["content_type"] = ",".join(content_types)
        if sources:
            params["source"] = ",".join(sources)
        if filter is not None:
            params["filter"] = _json_query_value(filter)
        if recency_weight is not None:
            params["recency_weight"] = recency_weight
        if half_life_days is not None:
            params["half_life_days"] = half_life_days
        if freshness_weight is not None:
            params["freshness_weight"] = freshness_weight
        if freshness_half_life_days is not None:
            params["freshness_half_life_days"] = freshness_half_life_days
        if self._partition:
            params["partition"] = self._partition
        resp = self._request_with_retry("GET","/search", params=params)
        self._raise_for_status(resp)
        body = resp.json()
        query_id = body.get("query_id")
        return [self._parse_search_result(r, query_id) for r in body.get("results", [])]

    @staticmethod
    def _parse_search_result(r: dict, query_id: Optional[str] = None) -> SearchResult:
        """Map a search-hit JSON object to a :class:`SearchResult`.

        *query_id* is the response-level feedback handle (present only when
        usage-feedback capture is enabled for the tenant); it is stamped onto
        every hit so a caller can pass it straight to
        :meth:`send_search_feedback`. Absent -> ``None``, like the other
        optional fields.
        """
        return SearchResult(
            doc_id=r["doc_id"],
            score=r["score"],
            title=r.get("title"),
            content_type=r.get("content_type", ""),
            content=r.get("content"),
            passage=r.get("passage"),
            entity_id=r.get("entity_id"),
            tags=list(r.get("tags") or []),
            source=r.get("source"),
            partition=r.get("partition"),
            metadata=dict(r.get("metadata") or {}),
            created_at=r.get("created_at"),
            updated_at=r.get("updated_at"),
            query_id=query_id,
        )

    def send_search_feedback(self, query_id: str, doc_id: str, signal: str) -> None:
        """Report how a search result was actually used.

        Ties a returned hit back to its real outcome so retrieval quality can
        be measured against actual usage. *signal* is one of ``"used"``,
        ``"cited"`` or ``"ignored"``.

        Requires usage-feedback capture to be enabled for your tenant; search
        results then carry a ``query_id`` to pass here (``None`` otherwise).
        The server rejects an unknown ``query_id`` with 404 and an invalid
        *signal* with 400 (both surface as :class:`AetherApiError`).
        """
        if not query_id:
            raise ValueError("query_id cannot be empty")
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        if not signal:
            raise ValueError("signal cannot be empty")
        body = {"query_id": query_id, "doc_id": doc_id, "signal": signal}
        resp = self._request_with_retry("POST", "/search/feedback", json=body)
        self._raise_for_status(resp)

    @staticmethod
    def _parse_trace(t: dict) -> SearchTrace:
        return SearchTrace(
            scoped_to=t.get("scoped_to"),
            partitions_touched=t.get("partitions_touched", []),
            default_partition_touched=t.get("default_partition_touched", False),
            results=t.get("results", 0),
            candidates_in_scope=t.get("candidates_in_scope"),
            boundary=t.get("boundary", ""),
        )

    def search_trace(
        self,
        query: str,
        k: int = 10,
        include_content: bool = False,
        tags: list[str] | None = None,
        entity_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        last_n_days: int | None = None,
        max_distance: float | None = None,
        filter: MetadataFilter | None = None,
    ) -> TracedSearch:
        """Like :meth:`search`, but also return an isolation :class:`SearchTrace`.

        The trace is computed from the records actually returned, so it is
        evidence — not intent — of which partition(s) the query touched. Under a
        partition handle the scope is injected exactly as in :meth:`search`; the
        trace then proves the boundary held (``partitions_touched`` is ``[]`` or
        ``[scoped_to]``, and ``candidates_in_scope`` is the partition's own size).
        """
        if not query:
            raise ValueError("query cannot be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        params: dict = {"q": query, "k": k, "trace": "true"}
        if include_content:
            params["include_content"] = "true"
        if tags:
            params["tags"] = ",".join(tags)
        if entity_id:
            params["entity_id"] = entity_id
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if last_n_days is not None:
            params["last_n_days"] = last_n_days
        if max_distance is not None:
            params["max_distance"] = max_distance
        if filter is not None:
            params["filter"] = _json_query_value(filter)
        if self._partition:
            params["partition"] = self._partition
        resp = self._request_with_retry("GET", "/search", params=params)
        self._raise_for_status(resp)
        body = resp.json()
        query_id = body.get("query_id")
        results = [self._parse_search_result(r, query_id) for r in body.get("results", [])]
        return TracedSearch(results=results, trace=self._parse_trace(body.get("trace", {})))

    def verify_isolation(
        self,
        query: str,
        k: int = 10,
    ) -> IsolationCheck:
        """Self-test that a scoped search never leaks out of this partition.

        Runs :meth:`search_trace` under this handle's partition and checks that
        every returned record stayed in scope. ``ok`` is true iff nothing
        leaked. Only meaningful on a partition handle and for a query that
        returns results — a 0-result query passes vacuously.

        Drop one line into your own tests to prove isolation against your data::

            assert client.partition("client_42").verify_isolation("returns policy").ok
        """
        if self._partition is None:
            raise ValueError(
                "verify_isolation requires a partition handle — call "
                "client.partition(id).verify_isolation(...)"
            )
        traced = self.search_trace(query, k=k)
        scoped = self._partition
        leaked = [p for p in traced.trace.partitions_touched if p != scoped]
        ok = not leaked and not traced.trace.default_partition_touched
        return IsolationCheck(
            ok=ok,
            scoped_to=scoped,
            partitions_touched=traced.trace.partitions_touched,
            results=traced.trace.results,
            candidates_in_scope=traced.trace.candidates_in_scope,
            leaked=leaked,
        )

    # ── Partitions ──────────────────────────────────────────────────

    def list_partitions(self) -> PartitionList:
        """List this tenant's partitions with active document counts.

        Tenant-level (does **not** use the partition handle). Includes advisory
        ``warnings`` flagging likely typos / ghost partitions; the default
        (unkeyed) partition is not listed.
        """
        resp = self._request_with_retry("GET", "/partitions")
        self._raise_for_status(resp)
        body = resp.json()
        return PartitionList(
            partitions=[
                PartitionInfo(id=p["id"], document_count=p.get("document_count", 0))
                for p in body.get("partitions", [])
            ],
            warnings=[
                PartitionWarning(
                    kind=w["kind"],
                    partitions=w.get("partitions", []),
                    detail=w.get("detail", ""),
                )
                for w in body.get("warnings", [])
            ],
        )

    def delete_partition(self, partition_id: str) -> int:
        """Delete a partition and shred **every** document in it (active and
        tombstoned). One-call GDPR / client-offboarding teardown.

        Returns the number of documents deleted. Idempotent: deleting an
        unknown or empty partition returns ``0`` and is never an error.
        """
        partition_id = _validate_partition(partition_id)
        resp = self._request_with_retry("DELETE", f"/partitions/{quote(partition_id, safe='')}")
        self._raise_for_status(resp)
        return resp.json().get("documents_deleted", 0)

    # ── BYOE (Bring Your Own Embeddings) ────────────────────────────

    def insert_with_embeddings(
        self,
        content: str,
        passages: list[dict] | None = None,
        embedding: list[float] | None = None,
        filename: str = "text.txt",
        content_type: str = "text/plain",
        tags: list[str] | None = None,
        entity_id: str | None = None,
        source: str | None = None,
        metadata: Metadata | None = None,
    ) -> DocumentRecord:
        """Insert a document with caller-provided embeddings.

        Either provide `passages` (list of {"text": str, "embedding": list[float]})
        for passage-level embeddings, or `embedding` for a single whole-document embedding.

        Pass *entity_id* to associate the document with an entity (e.g. a
        user or customer id) for later filtering on search and list.

        Pass *source* to tag the document with its origin (e.g. a system or
        channel name) for later filtering on search and list.
        """
        if not content:
            raise ValueError("content cannot be empty")
        body: dict = {"content": content, "filename": filename, "content_type": content_type}
        if passages is not None:
            body["passages"] = passages
        elif embedding is not None:
            body["embedding"] = embedding
        else:
            raise ValueError("Either 'passages' or 'embedding' must be provided")

        if tags:
            body["tags"] = tags
        if entity_id:
            body["entity_id"] = entity_id
        if source:
            body["source"] = source
        if metadata is not None:
            body["metadata"] = metadata
        if self._partition:
            body["partition"] = self._partition

        resp = self._request_with_retry("POST","/documents/embed", json=body)
        self._raise_for_status(resp)
        r = resp.json()
        return self._parse_document_record(r)

    def search_by_vector(
        self,
        embedding: list[float],
        k: int = 10,
        include_content: bool = False,
        tags: list[str] | None = None,
        entity_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        last_n_days: int | None = None,
        max_distance: float | None = None,
        any_tags: list[str] | None = None,
        content_types: list[str] | None = None,
        sources: list[str] | None = None,
        filter: MetadataFilter | None = None,
        recency_weight: float | None = None,
        half_life_days: float | None = None,
        freshness_weight: float | None = None,
        freshness_half_life_days: float | None = None,
    ) -> list[SearchResult]:
        """Search using a pre-computed query embedding.

        Args:
            entity_id: Only match documents associated with this entity id.
            since: Only match documents created at or after this RFC 3339
                timestamp (e.g. ``2026-06-01T00:00:00Z``). Inclusive.
            until: Only match documents created at or before this RFC 3339
                timestamp. Inclusive.
            last_n_days: Only match documents created in the last N days.
                Cannot be combined with ``since``.
            max_distance: Drop results whose distance exceeds this threshold.
            tags: Only match documents carrying **all** of these tags (AND).
            any_tags: Only match documents carrying **at least one** of these
                tags (OR).
            content_types: Only match documents whose content type is one of
                these (OR).
            sources: Only match documents whose source is one of these (OR).
            recency_weight: Blend recency into ranking, ``0.0``–``1.0``
                See :meth:`search`.
            half_life_days: Recency half-life in days; see :meth:`search`.
            freshness_weight: Blend freshness (recent updates) into ranking,
                ``0.0``–``1.0``; server default half-life is 14 days. See
                :meth:`search`. May require a Scale plan or higher.
            freshness_half_life_days: Freshness half-life in days; see
                :meth:`search`.
        """
        if not embedding:
            raise ValueError("embedding cannot be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        body: dict = {"embedding": embedding, "k": k, "include_content": include_content}
        if tags:
            body["tags"] = tags
        if entity_id:
            body["entity_id"] = entity_id
        if since:
            body["since"] = since
        if until:
            body["until"] = until
        if last_n_days is not None:
            body["last_n_days"] = last_n_days
        if max_distance is not None:
            body["max_distance"] = max_distance
        if any_tags:
            body["any_tags"] = any_tags
        if content_types:
            body["content_type"] = content_types
        if sources:
            body["source"] = sources
        if filter is not None:
            body["filter"] = filter
        if recency_weight is not None:
            body["recency_weight"] = recency_weight
        if half_life_days is not None:
            body["half_life_days"] = half_life_days
        if freshness_weight is not None:
            body["freshness_weight"] = freshness_weight
        if freshness_half_life_days is not None:
            body["freshness_half_life_days"] = freshness_half_life_days
        if self._partition:
            body["partition"] = self._partition
        resp = self._request_with_retry("POST","/search/embed", json=body)
        self._raise_for_status(resp)
        r = resp.json()
        query_id = r.get("query_id")
        return [self._parse_search_result(sr, query_id) for sr in r.get("results", [])]

    # ── Async Processing ──────────────────────────────────────────────
    
    def insert_async(
        self,
        file_path: str | Path,
        content_type: str | None = None,
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
        source: str | None = None,
        metadata: Metadata | None = None,
    ) -> dict:
        """Enqueue a document for asynchronous processing.
        Returns a dict with: job_id, status, poll_url.

        Pass *entity_id* to associate the document with an entity (e.g. a
        user or customer id) for later filtering on search and list.

        Pass *source* to tag the document with its origin (e.g. a system or
        channel name) for later filtering on search and list.
        """
        path = Path(file_path)
        data = path.read_bytes()
        filename = path.name
        if content_type is None:
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        url = f"/documents/async?filename={quote(filename)}&content_type={quote(content_type)}"
        if tags:
            url += f"&tags={quote(','.join(tags))}"
        if chunk_size is not None:
            url += f"&chunk_size={chunk_size}"
        if overlap is not None:
            url += f"&overlap={overlap}"
        if entity_id:
            url += f"&entity_id={quote(entity_id)}"
        if source:
            url += f"&source={quote(source)}"
        if metadata is not None:
            url += f"&metadata={quote(_json_query_value(metadata))}"
        if self._partition:
            url += f"&partition={quote(self._partition)}"

        resp = self._request_with_retry("POST", url, content=data)
        self._raise_for_status(resp)
        return resp.json()

    def wait_for_job(self, job_id: str, timeout: float = 60.0, poll_interval: float = 1.0) -> dict:
        """Wait for a background document job to complete."""
        if not job_id:
            raise ValueError("job_id cannot be empty")
        start = time.time()
        while time.time() - start < timeout:
            resp = self._request_with_retry("GET", f"/documents/jobs/{quote(job_id)}")
            self._raise_for_status(resp)
            job = resp.json()
            if job.get("status") in ("completed", "failed"):
                return job
            time.sleep(poll_interval)
        
        raise AetherApiError(408, "Job Timeout", error_code="timeout")

    # ── Batch Operations ─────────────────────────────────────────────

    def batch_insert(
        self,
        documents: list[BatchInsertItem],
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[DocumentRecord]:
        """Insert multiple text documents in a single batch request."""
        if not documents:
            raise ValueError("documents cannot be empty")
        payload: dict = {
            "documents": [
                {
                    "filename": d.filename,
                    "content": d.content,
                    **({"tags": ",".join(d.tags)} if d.tags else {}),
                    **({"entity_id": d.entity_id} if d.entity_id else {}),
                    **({"source": d.source} if d.source else {}),
                    **({"metadata": d.metadata} if d.metadata is not None else {}),
                    **({"partition": self._partition} if self._partition else {}),
                }
                for d in documents
            ],
        }
        if chunk_size is not None:
            payload["chunk_size"] = chunk_size
        if overlap is not None:
            payload["overlap"] = overlap

        resp = self._request_with_retry("POST", "/documents/batch", json=payload)
        self._raise_for_status(resp)
        body = resp.json()
        return [
            self._parse_document_record(r)
            for r in body.get("results", [])
        ]

    def batch_search(
        self,
        queries: list[BatchSearchQuery],
    ) -> list[BatchSearchResponse]:
        """Run multiple search queries in a single batch request."""
        if not queries:
            raise ValueError("queries cannot be empty")
        payload = {
            "queries": [
                {
                    "q": q.q,
                    "k": q.k,
                    **({"tags": ",".join(q.tags)} if q.tags else {}),
                    **({"include_content": q.include_content} if q.include_content else {}),
                    **({"entity_id": q.entity_id} if q.entity_id else {}),
                    **({"since": q.since} if q.since else {}),
                    **({"until": q.until} if q.until else {}),
                    **({"last_n_days": q.last_n_days} if q.last_n_days is not None else {}),
                    **({"max_distance": q.max_distance} if q.max_distance is not None else {}),
                    **({"any_tags": ",".join(q.any_tags)} if q.any_tags else {}),
                    **({"content_type": ",".join(q.content_types)} if q.content_types else {}),
                    **({"source": ",".join(q.sources)} if q.sources else {}),
                    **({"filter": q.filter} if q.filter is not None else {}),
                    **({"recency_weight": q.recency_weight} if q.recency_weight is not None else {}),
                    **({"half_life_days": q.half_life_days} if q.half_life_days is not None else {}),
                    **({"freshness_weight": q.freshness_weight} if q.freshness_weight is not None else {}),
                    **({"freshness_half_life_days": q.freshness_half_life_days} if q.freshness_half_life_days is not None else {}),
                    **({"partition": self._partition} if self._partition else {}),
                }
                for q in queries
            ],
        }
        resp = self._request_with_retry("POST", "/search/batch", json=payload)
        self._raise_for_status(resp)
        body = resp.json()
        return [
            BatchSearchResponse(
                query=r["query"],
                results=[
                    self._parse_search_result(sr, r.get("query_id"))
                    for sr in r.get("results", [])
                ],
            )
            for r in body.get("results", [])
        ]

    # ── Cluster ───────────────────────────────────────────────────────

    def status(self) -> NodeStatus:
        """Get node status."""
        resp = self._request_with_retry("GET","/status")
        self._raise_for_status(resp)
        body = resp.json()
        return NodeStatus(**{k: body[k] for k in NodeStatus.__dataclass_fields__ if k in body})

    def get_archive_price(self) -> dict[str, Any]:
        """Fetch the live $/GiB price for permanent archive uploads (Arweave/Irys).

        Mirrors the gateway's 5-minute cached upstream price. Useful for
        showing customers their archive cost before flipping the
        ``permanent_archive`` toggle.

        Returns:
            ``{"provider", "unit_price_cents_per_gib", "fetched_at",
            "cache_ttl_seconds"}``. The server returns 404 when the gateway
            is configured without an upstream URL — surfaces here as
            :class:`AetherApiError`.
        """
        resp = self._request_with_retry("GET", "/archive/price")
        self._raise_for_status(resp)
        return resp.json()

    # Note: Cluster operations (sync, snapshot, checkpoint, recover, validate)
    # are admin-only and not exposed in the public SDK. Use the REST API
    # directly with an admin API key for operational tasks.
