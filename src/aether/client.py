"""Aether Python SDK client."""

from __future__ import annotations

import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx

from ._internal import USER_AGENT, enforce_secure_base_url, new_idempotency_key
from .errors import AetherApiError, AetherNetworkError, aether_api_error_from_response
from .models import (
    BatchInsertItem,
    BatchSearchQuery,
    BatchSearchResponse,
    DocumentPage,
    DocumentRecord,
    EntityBackfillReport,
    NodeStatus,
    RetrievalResult,
    SearchResult,
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
        """Send a single HTTP request (no retries)."""
        return self._client.request(method, url, **kwargs)

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
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Documents ─────────────────────────────────────────────────────

    def insert(
        self,
        file_path: str | Path,
        content_type: str | None = None,
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
    ) -> DocumentRecord:
        """Insert a document from a file path.

        If *content_type* is not given it is guessed from the file extension
        (e.g. ``.pdf`` -> ``application/pdf``).  Falls back to
        ``application/octet-stream`` for unknown extensions.

        Pass *entity_id* to associate the document with an entity (e.g. a
        user or customer id) for later filtering on search and list.
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

        resp = self._request_with_retry("POST", url, content=data)
        self._raise_for_status(resp)
        body = resp.json()
        return DocumentRecord(
            doc_id=body["doc_id"],
            cid=body["cid"],
            title=body.get("title"),
            content_type=body.get("content_type", ""),
            size_bytes=body.get("size_bytes", 0),
            chunks=body["chunks"],
            vectors=body["vectors"],
            version=body["version"],
            created_at=body.get("created_at"),
            updated_at=body.get("updated_at"),
            entity_id=body.get("entity_id"),
        )

    def insert_text(
        self,
        text: str,
        filename: str = "text.txt",
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
    ) -> DocumentRecord:
        """Insert raw text content.

        Pass *entity_id* to associate the document with an entity (e.g. a
        user or customer id) for later filtering on search and list.
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
        resp = self._request_with_retry("POST", url, content=text.encode("utf-8"))
        self._raise_for_status(resp)
        body = resp.json()
        return DocumentRecord(
            doc_id=body["doc_id"],
            cid=body["cid"],
            title=body.get("title"),
            content_type=body.get("content_type", ""),
            size_bytes=body.get("size_bytes", 0),
            chunks=body["chunks"],
            vectors=body["vectors"],
            version=body["version"],
            created_at=body.get("created_at"),
            updated_at=body.get("updated_at"),
            entity_id=body.get("entity_id"),
        )

    def insert_stream(
        self,
        stream,
        filename: str = "upload.bin",
        content_type: str = "application/octet-stream",
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
    ) -> DocumentRecord:
        """Insert a document from a file-like object or iterator without loading everything into memory.

        ``stream`` can be any object that ``httpx`` accepts as streaming content:
        a file opened in binary mode, a ``bytes`` iterator, or a generator.

        Pass *entity_id* to associate the document with an entity (e.g. a
        user or customer id) for later filtering on search and list.

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
        resp = self._client.post(
            url, content=stream, headers={"Idempotency-Key": new_idempotency_key()}
        )
        self._raise_for_status(resp)
        body = resp.json()
        return DocumentRecord(
            doc_id=body["doc_id"],
            cid=body["cid"],
            title=body.get("title"),
            content_type=body.get("content_type", ""),
            size_bytes=body.get("size_bytes", 0),
            chunks=body["chunks"],
            vectors=body["vectors"],
            version=body["version"],
            created_at=body.get("created_at"),
            updated_at=body.get("updated_at"),
            entity_id=body.get("entity_id"),
        )

    def update(
        self,
        doc_id: str,
        file_path: str | Path,
        content_type: str = "text/plain",
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
    ) -> DocumentRecord:
        """Update an existing document.

        *entity_id* replaces the stored entity id; omitting it clears any
        existing value (mirrors *tags* semantics).
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

        resp = self._request_with_retry("PUT", url, content=data)
        self._raise_for_status(resp)
        body = resp.json()
        return DocumentRecord(
            doc_id=body["doc_id"],
            cid=body["cid"],
            title=body.get("title"),
            content_type=body.get("content_type", ""),
            size_bytes=body.get("size_bytes", 0),
            chunks=body["chunks"],
            vectors=body["vectors"],
            version=body["version"],
            created_at=body.get("created_at"),
            updated_at=body.get("updated_at"),
            entity_id=body.get("entity_id"),
        )

    def get(self, doc_id: str) -> DocumentRecord:
        """Get document metadata."""
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        resp = self._request_with_retry("GET",f"/documents/{quote(doc_id)}")
        self._raise_for_status(resp)
        body = resp.json()
        return DocumentRecord(
            doc_id=body["doc_id"],
            cid=body["cid"],
            title=body.get("title"),
            content_type=body.get("content_type", ""),
            size_bytes=body.get("size_bytes", 0),
            chunks=body.get("chunks", 0),
            vectors=body.get("vectors", 0),
            version=body.get("version", 1),
            created_at=body.get("created_at"),
            updated_at=body.get("updated_at"),
            entity_id=body.get("entity_id"),
        )

    def download(self, doc_id: str, output_path: str | Path) -> int:
        """Download a document to a file. Returns bytes written."""
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        resp = self._request_with_retry("GET",f"/documents/{quote(doc_id)}/download")
        self._raise_for_status(resp)
        path = Path(output_path)
        path.write_bytes(resp.content)
        return len(resp.content)

    def download_text(self, doc_id: str) -> str:
        """Download a document and return its content as text."""
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        resp = self._request_with_retry("GET",f"/documents/{quote(doc_id)}/download")
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
    ) -> list[RetrievalResult]:
        """Search and return results with full document content included.

        Combines search() + download_text() into a single call for RAG workflows.
        Results are deduplicated by doc_id (closest match wins). Since search no
        longer returns full document content, each unique document's text is
        fetched by id and attached as ``content``.

        Args:
            query: Search query.
            k: Maximum number of results to return.
            tags: Optional tag filter; results must carry all listed tags.
            entity_id: Only match documents associated with this entity id.
            since: Only match documents created at or after this RFC 3339
                timestamp (e.g. ``2026-06-01T00:00:00Z``). Inclusive.
            until: Only match documents created at or before this RFC 3339
                timestamp. Inclusive.
            last_n_days: Only match documents created in the last N days.
                Cannot be combined with ``since``.
            max_distance: Optional cosine-distance ceiling. Results with
                ``distance > max_distance`` are dropped server-side, after
                reranking. Omit (or pass ``None``) to return the top-k regardless
                of distance — the historical behavior.
        """
        if not query:
            raise ValueError("query cannot be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        results = self.search(
            query,
            k=k,
            tags=tags,
            entity_id=entity_id,
            since=since,
            until=until,
            last_n_days=last_n_days,
            max_distance=max_distance,
        )

        # Deduplicate by doc_id, keeping the closest match
        seen: dict[str, SearchResult] = {}
        for r in results:
            if r.doc_id not in seen:
                seen[r.doc_id] = r

        # Search returns only the matched passage now (never full document
        # content), so fetch each unique document's text by id for RAG prompts.
        retrieval_results = []
        for r in seen.values():
            content = self.download_text(r.doc_id)
            retrieval_results.append(
                RetrievalResult(
                    doc_id=r.doc_id,
                    score=r.score,
                    content=content,
                    title=r.title,
                    content_type=r.content_type,
                    passage=r.passage,
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
        resp = self._request_with_retry("GET","/documents", params=params)
        self._raise_for_status(resp)
        body = resp.json()
        documents = [
            DocumentRecord(
                doc_id=d["doc_id"],
                cid="",
                title=d.get("title"),
                content_type=d.get("content_type", ""),
                size_bytes=d.get("size_bytes", 0),
                version=d.get("version", 1),
                created_at=d.get("created_at"),
                entity_id=d.get("entity_id"),
            )
            for d in body.get("documents", [])
        ]
        return DocumentPage(
            documents,
            total=body.get("total", len(documents)),
            has_more=body.get("has_more", False),
        )

    def delete(self, doc_id: str) -> None:
        """Tombstone a document."""
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        resp = self._request_with_retry("DELETE",f"/documents/{quote(doc_id)}")
        self._raise_for_status(resp)

    def restore(self, doc_id: str) -> None:
        """Restore a tombstoned document."""
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        resp = self._request_with_retry("POST",f"/documents/{quote(doc_id)}/restore")
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

        Returns an :class:`EntityBackfillReport` with per-document outcome
        counts.
        """
        if not tag_prefix:
            raise ValueError("tag_prefix cannot be empty")
        body: dict = {"tag_prefix": tag_prefix, "overwrite": overwrite}
        resp = self._request_with_retry("POST", "/documents/backfill-entity", json=body)
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

    # ── Search ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
        tags: list[str] | None = None,
        entity_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        last_n_days: int | None = None,
        max_distance: float | None = None,
    ) -> list[SearchResult]:
        """Similarity search across documents.

        Each hit carries the matched ``passage`` (by default) and a calibrated
        ``score`` (0-100, higher = better). Full document text is never inlined;
        use :py:meth:`retrieve` or :py:meth:`download_text` to fetch it.

        Args:
            query: Search query.
            k: Maximum number of results to return.
            tags: Optional tag filter; results must carry all listed tags.
            entity_id: Only match documents associated with this entity id.
            since: Only match documents created at or after this RFC 3339
                timestamp (e.g. ``2026-06-01T00:00:00Z``). Inclusive.
            until: Only match documents created at or before this RFC 3339
                timestamp. Inclusive.
            last_n_days: Only match documents created in the last N days.
                Cannot be combined with ``since``.
            max_distance: Optional cosine-distance ceiling. Results with
                ``distance > max_distance`` are dropped server-side, after
                reranking. Omit (or pass ``None``) to return the top-k regardless
                of distance — the historical behavior.
        """
        if not query:
            raise ValueError("query cannot be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        params: dict = {"q": query, "k": k}
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
        resp = self._request_with_retry("GET","/search", params=params)
        self._raise_for_status(resp)
        body = resp.json()
        return [
            SearchResult(
                doc_id=r["doc_id"],
                score=r["score"],
                title=r.get("title"),
                content_type=r.get("content_type", ""),
                passage=r.get("passage"),
                entity_id=r.get("entity_id"),
            )
            for r in body.get("results", [])
        ]

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
    ) -> DocumentRecord:
        """Insert a document with caller-provided embeddings.

        Either provide `passages` (list of {"text": str, "embedding": list[float]})
        for passage-level embeddings, or `embedding` for a single whole-document embedding.

        Pass *entity_id* to associate the document with an entity (e.g. a
        user or customer id) for later filtering on search and list.
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

        resp = self._request_with_retry("POST","/documents/embed", json=body)
        self._raise_for_status(resp)
        r = resp.json()
        return DocumentRecord(
            doc_id=r["doc_id"], cid=r["cid"],
            title=r.get("title"), content_type=r.get("content_type", ""),
            size_bytes=r.get("size_bytes", 0),
            chunks=r["chunks"], vectors=r["vectors"], version=r["version"],
            created_at=r.get("created_at"), updated_at=r.get("updated_at"),
            entity_id=r.get("entity_id"),
        )

    def search_by_vector(
        self,
        embedding: list[float],
        k: int = 10,
        tags: list[str] | None = None,
        entity_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        last_n_days: int | None = None,
        max_distance: float | None = None,
    ) -> list[SearchResult]:
        """Search using a pre-computed query embedding.

        See :py:meth:`search` for the result shape and the semantics of the
        ``entity_id``, ``since``, ``until``, ``last_n_days`` and ``max_distance``
        filters.
        """
        if not embedding:
            raise ValueError("embedding cannot be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        body: dict = {"embedding": embedding, "k": k}
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
        resp = self._request_with_retry("POST","/search/embed", json=body)
        self._raise_for_status(resp)
        r = resp.json()
        return [
            SearchResult(
                doc_id=sr["doc_id"], score=sr["score"],
                title=sr.get("title"), content_type=sr.get("content_type", ""),
                passage=sr.get("passage"),
                entity_id=sr.get("entity_id"),
            )
            for sr in r.get("results", [])
        ]

    # ── Async Processing ──────────────────────────────────────────────
    
    def insert_async(
        self,
        file_path: str | Path,
        content_type: str | None = None,
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        entity_id: str | None = None,
    ) -> dict:
        """Enqueue a document for asynchronous processing.
        Returns a dict with: job_id, status, poll_url.

        Pass *entity_id* to associate the document with an entity (e.g. a
        user or customer id) for later filtering on search and list.
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
            DocumentRecord(
                doc_id=r["doc_id"],
                cid=r["cid"],
                title=r.get("title"),
                content_type=r.get("content_type", ""),
                size_bytes=r.get("size_bytes", 0),
                chunks=r["chunks"],
                vectors=r["vectors"],
                version=r["version"],
                created_at=r.get("created_at"),
                updated_at=r.get("updated_at"),
                entity_id=r.get("entity_id"),
            )
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
                    **({"entity_id": q.entity_id} if q.entity_id else {}),
                    **({"since": q.since} if q.since else {}),
                    **({"until": q.until} if q.until else {}),
                    **({"last_n_days": q.last_n_days} if q.last_n_days is not None else {}),
                    **({"max_distance": q.max_distance} if q.max_distance is not None else {}),
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
                    SearchResult(
                        doc_id=sr["doc_id"],
                        score=sr["score"],
                        title=sr.get("title"),
                        content_type=sr.get("content_type", ""),
                        passage=sr.get("passage"),
                        entity_id=sr.get("entity_id"),
                    )
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
