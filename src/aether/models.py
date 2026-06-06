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

@dataclass
class BatchInsertItem:
    filename: str
    content: str
    tags: Optional[list[str]] = None

@dataclass
class BatchSearchQuery:
    q: str
    k: int = 10
    tags: Optional[list[str]] = None
    include_content: bool = False

@dataclass
class BatchSearchResponse:
    query: str
    results: list[SearchResult]
