"""Data models for Aether SDK responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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


@dataclass
class SearchResult:
    doc_id: str
    distance: float
    title: Optional[str] = None
    content_type: str = "text/plain"
    content: Optional[str] = None
    passage: Optional[str] = None


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
    distance: float
    content: str
    title: Optional[str] = None
    content_type: str = "text/plain"
    passage: Optional[str] = None

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
