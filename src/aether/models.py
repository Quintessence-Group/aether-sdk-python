"""Data models for Aether SDK responses."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

# typing.Union rather than PEP 604 `|`: this alias is evaluated at runtime and
# the package supports Python 3.9.
MetadataValue = Union[str, int, float, bool]
Metadata = dict[str, MetadataValue]
MetadataFilter = dict[str, Any]


# Extension → content type for batch ingestion. Explicit so common
# document types resolve the same way on every OS regardless of the local
# mimetypes database (e.g. `.md` is not always registered). Anything not listed
# falls back to `mimetypes.guess_type`, then (at the call site) octet-stream.
INGEST_CONTENT_TYPES: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".text": "text/plain",
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
}


def resolve_content_type(path: Path) -> Optional[str]:
    """Best-effort content type for a file path: the explicit ingest map first,
    then the stdlib guess. ``None`` lets `insert` fall back to octet-stream."""
    ext = path.suffix.lower()
    if ext in INGEST_CONTENT_TYPES:
        return INGEST_CONTENT_TYPES[ext]
    return mimetypes.guess_type(path.name)[0]


@dataclass
class DocumentRecord:
    doc_id: str
    cid: str
    title: Optional[str] = None
    content_type: str = "text/plain"
    size_bytes: int = 0
    chunks: int = 0
    vectors: int = 0
    version: int = 1
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    entity_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    source: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)


@dataclass
class SearchResult:
    doc_id: str
    #: Calibrated relevance, 0-100 (higher = better); ~100 for a near-exact
    #: match. Results are ordered by descending ``score``.
    score: int
    title: Optional[str] = None
    content_type: str = "text/plain"
    content: Optional[str] = None
    passage: Optional[str] = None
    #: Identifier of the entity (e.g. user, customer) the matched document
    #: belongs to, echoed back by the engine on every hit. ``None`` if the
    #: document has no associated entity.
    entity_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    source: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)
    #: RFC 3339 timestamp of when the matched document was created.
    created_at: Optional[str] = None
    #: RFC 3339 timestamp of when the matched document was last updated, or
    #: ``None`` if it has never been updated since insert. Lets a caller spot
    #: a freshly-superseded hit without a second ``get`` round-trip.
    updated_at: Optional[str] = None
    #: Feedback handle for the search that returned this hit. Present only
    #: when usage-feedback capture is enabled for your tenant (``None``
    #: otherwise); pass it to ``send_search_feedback`` together with this
    #: hit's ``doc_id``.
    query_id: Optional[str] = None


@dataclass
class IngestResult:
    """Outcome of ingesting a single file via ``ingest_files`` /
    ``ingest_directory``.

    ``status`` is one of:

    * ``"ingested"`` — stored and indexed; ``doc_id`` is set.
    * ``"skipped"`` — the engine could not ingest this file (an unsupported or
      binary type, or one that needs the server-side document parser that is not
      configured, or a file over the size limit). ``error`` explains why. This is
      the graceful path: the batch continues.
    * ``"error"`` — an unexpected failure (e.g. the file could not be read, or a
      transient API/network error). ``error`` carries the detail.
    """

    path: str
    status: str
    doc_id: Optional[str] = None
    content_type: Optional[str] = None
    error: Optional[str] = None


@dataclass
class NodeStatus:
    """Basic node health information."""
    node_id: int = 0
    documents: int = 0
    vectors: int = 0
    version: str = ""


@dataclass
class RetrievalResult:
    """A search result enriched with document content for RAG workflows."""

    doc_id: str
    #: Calibrated relevance, 0-100 (higher = better); ~100 for a near-exact match.
    score: int
    #: Full document content as text, for use in RAG prompts.
    content: str
    title: Optional[str] = None
    content_type: str = "text/plain"
    passage: Optional[str] = None
    #: Identifier of the entity the matched document belongs to, if any.
    entity_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    source: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)
    created_at: Optional[str] = None
    #: RFC 3339 timestamp of the document's last update, or ``None``.
    updated_at: Optional[str] = None


class DocumentPage(list):
    """A page of documents returned by ``list()``.

    Subclasses :class:`list`, so it behaves exactly like a
    ``list[DocumentRecord]`` (iteration, indexing, ``len()``) for backward
    compatibility, while also exposing pagination metadata for parity with the
    TypeScript, Go, and .NET SDKs.

    Attributes:
        total: Total number of active documents across all pages.
        has_more: Whether additional pages exist beyond this one.
    """

    def __init__(self, documents=(), total: int = 0, has_more: bool = False):
        super().__init__(documents)
        self.total = total
        self.has_more = has_more


@dataclass
class BatchInsertItem:
    filename: str
    content: str
    tags: Optional[list[str]] = None
    entity_id: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Metadata] = None

@dataclass
class BatchSearchQuery:
    q: str
    k: int = 10
    tags: Optional[list[str]] = None
    include_content: bool = False
    entity_id: Optional[str] = None
    since: Optional[str] = None
    until: Optional[str] = None
    last_n_days: Optional[int] = None
    max_distance: Optional[float] = None
    any_tags: Optional[list[str]] = None
    content_types: Optional[list[str]] = None
    sources: Optional[list[str]] = None
    filter: Optional[MetadataFilter] = None
    #: Blend recency into ranking, 0.0–1.0. See ``search``.
    recency_weight: Optional[float] = None
    #: Recency half-life in days; see ``search``.
    half_life_days: Optional[float] = None
    #: Blend freshness (recently *updated* documents, via ``updated_at``)
    #: into ranking, 0.0–1.0. See ``search``. May require a Scale plan or
    #: higher.
    freshness_weight: Optional[float] = None
    #: Freshness half-life in days (server default 14); see ``search``.
    freshness_half_life_days: Optional[float] = None

@dataclass
class BatchSearchResponse:
    query: str
    results: list[SearchResult]

@dataclass
class EntityBackfillReport:
    """Counts returned by a tag-driven entity_id backfill run."""
    scanned: int = 0
    updated: int = 0
    skipped_existing: int = 0
    skipped_no_match: int = 0
    skipped_ambiguous: int = 0
    skipped_invalid: int = 0


# ── Partition lifecycle ──────────────────────────────────────────────

@dataclass
class PartitionInfo:
    """A partition and its active (non-tombstoned) document count."""
    id: str
    document_count: int = 0


@dataclass
class PartitionWarning:
    """An advisory flag about a likely-mistyped or ghost partition.

    ``kind`` is ``"single_document"`` (a partition holding one document —
    often a typo or abandoned ghost) or ``"near_duplicate"`` (two ids that
    differ only cosmetically — likely the same end-client under two keys).
    Advisory only; create-on-write is never blocked.
    """
    kind: str
    partitions: list[str]
    detail: str


@dataclass
class PartitionList:
    """Result of :meth:`AetherClient.list_partitions`."""
    partitions: list[PartitionInfo]
    warnings: list[PartitionWarning]


# ── Provable isolation ───────────────────────────────────────────────

@dataclass
class SearchTrace:
    """Evidence of which partition(s) a search actually touched.

    For a scoped query, ``partitions_touched`` is always ``[]`` or exactly
    ``[scoped_to]``, and ``candidates_in_scope`` is the partition's own size
    (proof the scope bounded the search as a hard ceiling, not a post-filter).
    ``boundary`` is ``"partition"`` (scoped) or ``"tenant"`` (unscoped).
    """
    scoped_to: Optional[str]
    partitions_touched: list[str]
    default_partition_touched: bool
    results: int
    candidates_in_scope: Optional[int]
    boundary: str


@dataclass
class TracedSearch:
    """Search results plus the isolation :class:`SearchTrace` that produced them."""
    results: list[SearchResult]
    trace: SearchTrace


@dataclass
class IsolationCheck:
    """Outcome of :meth:`AetherClient.verify_isolation` on a scoped handle.

    ``ok`` is true iff no returned record left the handle's partition. Only
    meaningful for a query that returns results — a 0-result query passes
    vacuously (``results == 0``).
    """
    ok: bool
    scoped_to: Optional[str]
    partitions_touched: list[str]
    results: int
    candidates_in_scope: Optional[int]
    leaked: list[str]
