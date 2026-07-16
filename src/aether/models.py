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
    #: Conversation identity when this document is an AET-151 thread turn.
    thread_id: Optional[str] = None
    #: Zero-based, server-assigned position within ``thread_id``.
    turn_index: Optional[int] = None
    tags: list[str] = field(default_factory=list)
    source: Optional[str] = None
    #: Partition the document lives in, echoed back by the engine on every
    #: document response. ``None`` means the default partition (mirrors the
    #: ``entity_id``/``source`` convention).
    partition: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)
    #: ``image`` or ``audio`` for a multimodal memory, else ``None``.
    modality: Optional[str] = None
    #: Indexed caption/transcript for a multimodal memory. Original media bytes
    #: remain available through ``download``.
    derived_text: Optional[str] = None


@dataclass
class MediaMemoryRecord:
    """Result of storing an image or audio memory."""

    doc_id: str
    cid: str
    modality: str
    content_type: str
    derived_text: str
    derived_by: str
    created_at: Optional[str] = None
    entity_id: Optional[str] = None
    partition: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)


@dataclass
class AuditProof:
    """Cryptographic proof attached to a ledger audit record.

    Present for ledger-sourced :class:`AuditRecord` entries. ``content_id`` is
    the blake3 content address of the document at the time of the event; it is
    omitted (``None``) for events that do not reference content, such as a
    tombstone/deletion.
    """

    #: blake3 content address (e.g. ``blake3:...``) of the document at the
    #: event, or ``None`` when the event references no content (e.g. a delete).
    content_id: Optional[str] = None
    #: Lamport clock value of the event, giving a total order across the ledger.
    lamport: int = 0
    #: Hex node id of the node that signed the record.
    node_id: str = ""
    #: Hex-encoded public key the signature verifies against.
    public_key: str = ""
    #: Hex-encoded signature over the record.
    signature: str = ""
    #: Whether the SDK/engine verified the signature against ``public_key``.
    verified: bool = False


@dataclass
class AuditRecord:
    """One audit record in the shared envelope used by both audit surfaces.

    Returned by :meth:`AetherClient.lineage` (``source == "ledger"``: signed
    provenance events, each carrying a cryptographic :class:`AuditProof`) and
    by :meth:`AuditClient.access` (``source == "access"``: the operational
    access-audit log — reads, search deliveries, denials, and admin bypasses —
    which carries no proof).
    """

    #: RFC 3339 timestamp of when the event occurred.
    at: str
    #: Who performed the action. For ledger records this is the signing node
    #: (``node:<hex>``); for access records it is the asserted acting principal
    #: (e.g. ``user:alice``), or ``key:<prefix>`` when no principal was asserted.
    actor: str
    #: The action taken (e.g. ``document.inserted``; access records use
    #: ``read`` / ``search_hit`` / ``denied`` / ``admin_bypass``).
    action: str
    #: The resource acted upon (e.g. ``document:<uuid>``).
    resource: str
    #: The outcome of the action (``committed`` for ledger records; ``ok`` /
    #: ``denied`` / ``admin_bypass`` for access records).
    outcome: str
    #: Which audit surface produced this record: ``ledger`` or ``access``.
    source: str
    #: Cryptographic proof for the record (present for ``ledger`` records), or
    #: ``None`` when absent (``access`` records carry no proof).
    proof: Optional[AuditProof] = None


class AccessAuditPage(list):
    """A page of access-audit records returned by :meth:`AuditClient.access`.

    Subclasses :class:`list`, so it behaves exactly like a
    ``list[AuditRecord]`` (iteration, indexing, ``len()``), while also exposing
    the total match count for paging (mirrors :class:`DocumentPage`).

    Attributes:
        total: Total records matching the filter across all pages (ignores
            ``limit`` / ``offset``).
    """

    def __init__(self, records=(), total: int = 0):
        super().__init__(records)
        self.total = total


@dataclass
class GroundingSource:
    """One tenant-private source in a declared answer-grounding set.

    This detail is returned only to the authenticated caller. It is intentionally
    absent from an optional public :class:`ShareableReceipt`.
    """

    document_id: str
    content_id: str
    rank: int
    retained_signed_event_count: int
    current_content_verified: bool
    #: Existing lineage evidence for this current CID. It is engine-verified
    #: traceability data, not a standalone LedgerEvent signing transcript.
    proof: Optional[AuditProof] = None


@dataclass
class GroundingTrustSignal:
    """Integrity state for a declared grounding set.

    ``verified`` means every current source CID was anchored by a retained,
    valid Ed25519 event when the receipt was created. It does not assess factual
    correctness or prove how an external model reasoned from a source.
    """

    status: str
    sources_requested: int
    sources_verified: int
    answer_bound: bool


@dataclass
class GroundingBinding:
    """Authenticated-only material for an opaque receipt commitment.

    ``verification_salt`` is returned only to the authenticated creator and is
    never persisted or emitted by a public receipt URL. Retain it with the
    original answer/source result if you need to recompute the commitment.
    """

    algorithm: str
    source_set_commitment: str
    source_evidence_commitment: str
    binding_commitment: str
    verification_salt: str


@dataclass
class ReceiptAttestation:
    """Publicly checkable Ed25519 node attestation over public receipt fields."""

    signer_node_id: str
    signer_public_key: str
    signature: str
    verified: bool


@dataclass
class GroundingSetAttestation:
    """Ed25519 attestation over every authenticated grounding result."""

    version: str
    issued_at: str
    binding_algorithm: str
    signer_node_id: str
    signer_public_key: str
    signature: str
    verified: bool


@dataclass
class ShareableReceipt:
    """Public-safe, revocable share-link metadata returned after ``share=True``.

    The public link exposes aggregate verification state only: never answer
    text/digest, tenant id, document ids, CIDs, titles, passages, or ledger
    event bytes.
    """

    version: str
    receipt_id: str
    issued_at: str
    expires_at: str
    source_count: int
    verified_source_count: int
    status: str
    binding_commitment: str
    capability_commitment: str
    owner_commitment: str
    attestation: ReceiptAttestation
    share_url: str
    badge_url: str


@dataclass
class GroundingReceipt:
    """Authenticated AET-348 answer-grounding provenance response."""

    answer_digest: str
    sources: list[GroundingSource]
    trust: GroundingTrustSignal
    binding: GroundingBinding
    attestation: GroundingSetAttestation
    receipt: Optional[ShareableReceipt] = None


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
    #: Conversation identity when this hit is an AET-151 thread turn.
    thread_id: Optional[str] = None
    #: Zero-based, server-assigned position within ``thread_id``.
    turn_index: Optional[int] = None
    tags: list[str] = field(default_factory=list)
    source: Optional[str] = None
    #: Partition the matched document lives in, echoed back by the engine on
    #: every hit. ``None`` means the default partition (mirrors the
    #: ``entity_id``/``source`` convention).
    partition: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)
    #: RFC 3339 timestamp of when the matched document was created.
    created_at: Optional[str] = None
    #: RFC 3339 timestamp of when the matched document was last updated, or
    #: ``None`` if it has never been updated since insert. Lets a caller spot
    #: a freshly-superseded hit without a second ``get`` round-trip.
    updated_at: Optional[str] = None
    #: ``image`` or ``audio`` when this hit is a multimodal memory.
    modality: Optional[str] = None
    #: Feedback handle for the search that returned this hit. Present only
    #: when usage-feedback capture is enabled for your tenant (``None``
    #: otherwise); pass it to ``send_search_feedback`` together with this
    #: hit's ``doc_id``.
    query_id: Optional[str] = None


@dataclass
class ConversationThread:
    """Canonical tenant-scoped AET-151 conversation in turn order."""

    thread_id: str
    documents: list[DocumentRecord] = field(default_factory=list)


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
    #: Conversation identity when this result is an AET-151 thread turn.
    thread_id: Optional[str] = None
    #: Zero-based, server-assigned position within ``thread_id``.
    turn_index: Optional[int] = None
    tags: list[str] = field(default_factory=list)
    source: Optional[str] = None
    #: Partition the matched document lives in; ``None`` means the default
    #: partition (mirrors the ``entity_id``/``source`` convention).
    partition: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)
    created_at: Optional[str] = None
    #: RFC 3339 timestamp of the document's last update, or ``None``.
    updated_at: Optional[str] = None
    #: ``image`` or ``audio`` when this result is a multimodal memory.
    modality: Optional[str] = None


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
    #: Read-ACL for this document: ``user:`` / ``group:`` labels naming who may
    #: read it. ``None`` (default) leaves it unlabeled / tenant-visible; an
    #: explicit empty list quarantines it to admin-role keys only.
    acl_readers: Optional[list[str]] = None

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
    #: Restrict results to one canonical conversation thread. Kept last so
    #: existing positional construction retains its prior argument mapping.
    thread_id: Optional[str] = None

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


# ── Structured query & field schema ──────────────────────────────────

@dataclass
class FieldSchema:
    """A declared typed field for the structured-query layer.

    Field values are extracted from document metadata (or passage text via a
    regex) at ingest time, and become filterable / sortable / aggregatable
    through :meth:`AetherClient.query`.
    """
    name: str
    #: One of ``string``, ``int``, ``float``, ``bool``, ``datetime``,
    #: ``string_list``.
    type: str
    #: Where the value comes from: ``{"metadata": "<key>"}`` or
    #: ``{"regex": "<pattern>"}``.
    source: dict = field(default_factory=dict)
    #: Hard-partition scope, or ``None`` for a tenant-wide field.
    partition_scope: Optional[str] = None
    #: Active documents whose source value coerced to the declared type.
    coverage: int = 0
    #: Active documents whose source value was present but failed to coerce.
    mismatch_count: int = 0
    #: Backfill state; ``"complete"`` in v1 (synchronous on declare).
    backfill: str = "complete"


@dataclass
class QueryGroup:
    """One group in an aggregation (Mode B) result."""
    #: Group-key values keyed by ``group_by`` field name; empty for a
    #: whole-population aggregate.
    keys: dict
    #: Computed aggregates keyed by output name (the ``as`` alias or a default).
    aggregates: dict


@dataclass
class AggregateResult:
    """Result of an aggregation (Mode B) :meth:`AetherClient.query`."""
    groups: list["QueryGroup"]
    #: Distinct group count before ``limit`` is applied.
    total_groups: int
    #: Documents folded into the aggregation (post-filter).
    scanned: int
