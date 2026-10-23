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
    #: Conversation identity when this document is a thread turn.
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
    #: Summary request outcome when the insert asked for one (``summarize``):
    #: ``"pending"`` (queued) or ``"skipped"`` (skipped by policy); ``None``
    #: when no summary was requested. Poll ``get_summary`` for the result.
    summary: Optional[str] = None


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
    """Authenticated answer-grounding provenance response."""

    answer_digest: str
    sources: list[GroundingSource]
    trust: GroundingTrustSignal
    binding: GroundingBinding
    attestation: GroundingSetAttestation
    receipt: Optional[ShareableReceipt] = None


@dataclass
class MemoryFileHit:
    """Where a search hit landed inside a memory file (``SearchResult.memory_file``)."""

    #: The memory file's name, e.g. ``memory.md``.
    name: str
    #: The file version the hit was found in (always the current version).
    version: int
    #: Heading of the innermost section containing the match (best effort;
    #: ``None`` when unknown or in the text before the first heading).
    section: Optional[str] = None
    #: Slug of that section, usable as a ``section`` selector.
    section_slug: Optional[str] = None


def parse_memory_file_hit(raw: Any) -> Optional[MemoryFileHit]:
    """Map a search hit's optional ``memory_file`` object; ``None`` when absent."""
    if not isinstance(raw, dict):
        return None
    return MemoryFileHit(
        name=raw.get("name", ""),
        version=int(raw.get("version") or 0),
        section=raw.get("section"),
        section_slug=raw.get("section_slug"),
    )


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
    #: Conversation identity when this hit is a thread turn.
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
    #: Set when the hit is a memory file: its name, version and the section
    #: the match falls in. ``None`` for every other document.
    memory_file: Optional[MemoryFileHit] = None


@dataclass
class ConversationThread:
    """Canonical tenant-scoped conversation in turn order."""

    thread_id: str
    documents: list[DocumentRecord] = field(default_factory=list)


@dataclass
class ThreadLifecycleResult:
    """Result of a whole-thread lifecycle mutation — ``restore_thread``,
    ``set_thread_acl``, ``move_thread``, or ``delete_thread`` (and their
    ``Thread`` / ``AsyncThread`` facade equivalents).
    """

    #: The applied action: ``"tombstoned"``, ``"restored"``, ``"acl_updated"``,
    #: ``"hard_deleted"``, or ``"moved"``.
    status: str
    thread_id: str
    #: Number of turns whose canonical marker (tombstone / ACL / partition) was
    #: rewritten by the op.
    turns: int = 0


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
    #: Conversation identity when this result is a thread turn.
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
    #: Set when the result is a memory file (see ``SearchResult.memory_file``).
    memory_file: Optional[MemoryFileHit] = None


@dataclass
class MemoryFileSection:
    """One section of a markdown outline: a top-level ATX heading and the
    lines it spans. Line numbers are 1-based and inclusive; ``bytes`` counts
    lines ``start_line..end_line`` including their line terminators."""

    #: Heading text; ``""`` for a bare ``#``.
    heading: str
    #: Heading level, 1-6.
    level: int
    #: Slug, unique within the text — usable as a ``section`` selector.
    slug: str
    start_line: int
    end_line: int
    bytes: int


@dataclass
class DocumentText:
    """Result of :meth:`AetherClient.get_text` (``GET /v1/documents/{id}/text``)."""

    doc_id: str
    content_type: str
    #: ``original`` (a text document's own bytes), ``derived`` (a media
    #: memory's caption/transcript) or ``passages`` (rebuilt from a parsed
    #: document's passages, best effort); ``None`` when ``state`` is
    #: ``unsupported``.
    source: Optional[str]
    #: ``ready``, or ``unsupported`` when the document has no readable text.
    state: str
    #: This page of the selected text (``""`` with ``outline=True``).
    text: str
    #: Byte offset of this page within the selected text.
    offset: int = 0
    #: Offset of the next page, or ``None`` on the last page.
    next_offset: Optional[int] = None
    #: UTF-8 byte length of the selected text (the whole text, or the section).
    total_bytes: int = 0
    #: ``True`` when more pages follow (``next_offset is not None``).
    truncated: bool = False
    #: The selected section when ``section`` was requested, else ``None``.
    section: Optional[MemoryFileSection] = None
    #: The text's outline when ``outline=True`` was requested, else ``None``.
    sections: Optional[list[MemoryFileSection]] = None


def _parse_memory_file_section(raw: dict[str, Any]) -> MemoryFileSection:
    return MemoryFileSection(
        heading=raw.get("heading", ""),
        level=int(raw.get("level", 0)),
        slug=raw.get("slug", ""),
        start_line=int(raw.get("start_line", 0)),
        end_line=int(raw.get("end_line", 0)),
        bytes=int(raw.get("bytes", 0)),
    )


def parse_document_text(raw: dict[str, Any]) -> DocumentText:
    """Map a ``/documents/{id}/text`` JSON body to a :class:`DocumentText`."""
    section = raw.get("section")
    sections = raw.get("sections")
    return DocumentText(
        doc_id=raw["doc_id"],
        content_type=raw.get("content_type", ""),
        source=raw.get("source"),
        state=raw.get("state", "ready"),
        text=raw.get("text", ""),
        offset=int(raw.get("offset", 0)),
        next_offset=raw.get("next_offset"),
        total_bytes=int(raw.get("total_bytes", 0)),
        truncated=bool(raw.get("truncated", False)),
        section=_parse_memory_file_section(section) if section else None,
        sections=(
            [_parse_memory_file_section(s) for s in sections]
            if sections is not None
            else None
        ),
    )


# ── Memory files ─────────────────────────────────────────────────────


@dataclass
class MemoryFileInfo:
    """A memory file at one version: the item type of
    :meth:`MemoryFiles.list` / :meth:`AetherClient.list_memory_files`, and the
    fields every memory-file read and write result starts with.

    A memory file is identified by ``(partition, entity_id, name)``. Its
    ``version`` starts at 1 and grows by one with every change; it is not the
    same counter as ``DocumentRecord.version``.
    """

    entity_id: str = ""
    #: ``None`` for the default partition.
    partition: Optional[str] = None
    name: str = ""
    #: Stable id of the file, kept across a soft delete and re-create (a hard
    #: delete starts a new one).
    file_id: str = ""
    #: The document holding this version's content.
    doc_id: str = ""
    version: int = 0
    #: Size of this version's content in bytes.
    bytes: int = 0
    #: Content hash of this version.
    content_hash: str = ""
    #: Free-text attribution label passed as ``author`` on the write, if any.
    author: Optional[str] = None
    #: Who wrote this version: the acting principal, or ``key:<prefix>``.
    principal: str = ""
    #: The write that produced this version: ``put``, ``append``,
    #: ``append_section`` or ``replace_section``.
    op: str = ""
    #: RFC 3339 time the file was created (version 1 of its current lifetime).
    created_at: str = ""
    #: RFC 3339 time this version was written.
    updated_at: str = ""


@dataclass
class MemoryFile(MemoryFileInfo):
    """Result of :meth:`MemoryFiles.read`: one page of a memory file (the
    whole file by default) plus its outline."""

    #: ``False`` when an older ``version`` was requested.
    is_current: bool = True
    #: This page of the selected text (``""`` with ``outline=True``).
    content: str = ""
    #: The selected section when ``section`` was requested, else ``None``.
    section: Optional[MemoryFileSection] = None
    #: Byte offset of this page within the selected text.
    offset: int = 0
    #: Offset of the next page, or ``None`` on the last page.
    next_offset: Optional[int] = None
    #: UTF-8 byte length of the selected text (the whole version, or the
    #: section).
    total_bytes: int = 0
    #: ``True`` when more pages follow (``next_offset is not None``).
    truncated: bool = False
    #: The whole version's outline, with whole-file line numbers.
    sections: list[MemoryFileSection] = field(default_factory=list)


@dataclass
class MemoryFileWriteResult(MemoryFileInfo):
    """Result of a memory-file write (``write``, ``append``,
    ``append_section``, ``replace_section``). The info fields describe the
    file's current version after the write."""

    #: ``True`` when the write would not change the content, so nothing was
    #: written and the info fields describe the existing current version.
    unchanged: bool = False
    #: The version this write replaced; ``None`` when it created the file.
    previous_version: Optional[int] = None
    #: ``True`` when this write created the file.
    file_created: bool = False
    #: Section writes: the affected section after the write, else ``None``.
    section: Optional[MemoryFileSection] = None
    #: Section writes: ``True`` when the section did not exist and was added.
    section_created: bool = False
    #: Outline of the resulting content.
    sections: list[MemoryFileSection] = field(default_factory=list)


@dataclass
class MemoryFileVersion:
    """One retained version in :class:`MemoryFileVersions`."""

    doc_id: str = ""
    version: int = 0
    bytes: int = 0
    content_hash: str = ""
    author: Optional[str] = None
    principal: str = ""
    op: str = ""
    updated_at: str = ""
    is_current: bool = False


@dataclass
class MemoryFileVersions:
    """Result of :meth:`MemoryFiles.versions`: retained versions, newest first."""

    entity_id: str = ""
    partition: Optional[str] = None
    name: str = ""
    file_id: str = ""
    current_version: int = 0
    versions: list[MemoryFileVersion] = field(default_factory=list)


@dataclass
class MemoryFileDiff:
    """Result of :meth:`MemoryFiles.diff`: a line-based unified diff between
    two versions (``from_version`` 0 is the empty file)."""

    entity_id: str = ""
    partition: Optional[str] = None
    name: str = ""
    file_id: str = ""
    from_version: int = 0
    to_version: int = 0
    #: Lines of context around each change.
    context: int = 0
    #: Unified diff text (``--- name@vN`` / ``+++ name@vM`` headers).
    diff: str = ""
    added_lines: int = 0
    removed_lines: int = 0
    #: ``True`` when the diff is correct but possibly not minimal (the server
    #: stopped optimizing it after its time budget).
    approximate: bool = False
    #: ``True`` when the diff text was cut at the server's output limit.
    truncated: bool = False


@dataclass
class MemoryFilePage:
    """One page of memory files; pass ``next_cursor`` back as ``cursor`` for
    the next page (``None`` on the last page)."""

    files: list[MemoryFileInfo] = field(default_factory=list)
    next_cursor: Optional[str] = None


@dataclass
class MemoryGrepMatch:
    """One matching line from a memory-file grep."""

    entity_id: str = ""
    partition: Optional[str] = None
    name: str = ""
    #: The file version that was searched (always the current version).
    version: int = 0
    #: 1-based line number of the match.
    line: int = 0
    #: Heading of the innermost section containing the line, or ``None``
    #: before the first heading.
    section: Optional[str] = None
    section_slug: Optional[str] = None
    #: The matching line, without its line terminator.
    text: str = ""
    #: Up to ``context`` lines before the match.
    before: list[str] = field(default_factory=list)
    #: Up to ``context`` lines after the match.
    after: list[str] = field(default_factory=list)


@dataclass
class MemoryGrepResult:
    """Result of a memory-file grep."""

    matches: list[MemoryGrepMatch] = field(default_factory=list)
    #: ``True`` when the scan stopped at a limit before covering everything.
    truncated: bool = False
    #: Which limit stopped it: ``max_matches``, ``max_files``, ``max_bytes`` or
    #: ``deadline``; ``None`` when not truncated.
    truncated_reason: Optional[str] = None
    files_scanned: int = 0
    bytes_scanned: int = 0


@dataclass
class MemoryFileDeleteResult:
    """Result of :meth:`MemoryFiles.delete`."""

    entity_id: str = ""
    partition: Optional[str] = None
    name: str = ""
    file_id: str = ""
    #: The version that was current when the file was deleted.
    version: int = 0
    #: ``True`` for a hard delete (every version purged).
    hard: bool = False
    #: Number of versions removed: ``1`` for a soft delete.
    versions_deleted: int = 0


@dataclass
class DocumentSummary:
    """A document's summary and its state (:meth:`AetherClient.get_summary`,
    ``put_summary``, ``regenerate_summary``).

    ``state`` is one of ``none``, ``pending``, ``ready``, ``failed``,
    ``skipped`` or ``unsupported``. ``content`` is independent of ``state``: a
    stored summary stays readable while a regeneration is pending or after
    one failed.
    """

    doc_id: str
    state: str
    #: Machine reason for ``skipped``, ``unsupported`` or ``failed``.
    reason: Optional[str] = None
    #: Human-readable message for ``failed``.
    error: Optional[str] = None
    #: The document that stores the summary, if any.
    summary_doc_id: Optional[str] = None
    #: The stored summary markdown, if any.
    content: Optional[str] = None
    #: Opaque revision of the stored summary; pass it as ``expected_revision``
    #: to ``put_summary`` for a conditional write. Compare only for equality.
    revision: Optional[str] = None
    #: ``"aether"`` (generated) or ``"user"`` (written with ``put_summary``).
    authored_by: Optional[str] = None
    generator: Optional[str] = None
    prompt_version: Optional[str] = None
    #: RFC 3339 time the stored summary was written.
    generated_at: Optional[str] = None
    #: The document content id the summary was written against.
    source_cid: Optional[str] = None
    #: ``True`` when the document changed after the stored summary was written.
    stale: bool = False
    #: ``True`` when the document was too long and only its outline and
    #: beginning were summarized.
    truncated_input: bool = False


def _int_field(raw: dict[str, Any], key: str, default: int = 0) -> int:
    value = raw.get(key)
    return int(value) if value is not None else default


def _opt_int_field(raw: dict[str, Any], key: str) -> Optional[int]:
    value = raw.get(key)
    return int(value) if value is not None else None


def _sections_field(raw: dict[str, Any]) -> list[MemoryFileSection]:
    return [_parse_memory_file_section(s) for s in raw.get("sections") or []]


def _section_field(raw: dict[str, Any]) -> Optional[MemoryFileSection]:
    section = raw.get("section")
    return _parse_memory_file_section(section) if section else None


def _memory_file_info_fields(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": raw.get("entity_id") or "",
        "partition": raw.get("partition"),
        "name": raw.get("name") or "",
        "file_id": raw.get("file_id") or "",
        "doc_id": raw.get("doc_id") or "",
        "version": _int_field(raw, "version"),
        "bytes": _int_field(raw, "bytes"),
        "content_hash": raw.get("content_hash") or "",
        "author": raw.get("author"),
        "principal": raw.get("principal") or "",
        "op": raw.get("op") or "",
        "created_at": raw.get("created_at") or "",
        "updated_at": raw.get("updated_at") or "",
    }


def parse_memory_file_info(raw: dict[str, Any]) -> MemoryFileInfo:
    """Map a memory-file list item to a :class:`MemoryFileInfo`."""
    return MemoryFileInfo(**_memory_file_info_fields(raw))


def parse_memory_file(raw: dict[str, Any]) -> MemoryFile:
    """Map a memory-file read body to a :class:`MemoryFile`."""
    return MemoryFile(
        **_memory_file_info_fields(raw),
        is_current=bool(raw.get("is_current", True)),
        content=raw.get("content") or "",
        section=_section_field(raw),
        offset=_int_field(raw, "offset"),
        next_offset=_opt_int_field(raw, "next_offset"),
        total_bytes=_int_field(raw, "total_bytes"),
        truncated=bool(raw.get("truncated", False)),
        sections=_sections_field(raw),
    )


def parse_memory_file_write_result(raw: dict[str, Any]) -> MemoryFileWriteResult:
    """Map a memory-file write body to a :class:`MemoryFileWriteResult`."""
    return MemoryFileWriteResult(
        **_memory_file_info_fields(raw),
        unchanged=bool(raw.get("unchanged", False)),
        previous_version=_opt_int_field(raw, "previous_version"),
        file_created=bool(raw.get("file_created", False)),
        section=_section_field(raw),
        section_created=bool(raw.get("section_created", False)),
        sections=_sections_field(raw),
    )


def parse_memory_file_versions(raw: dict[str, Any]) -> MemoryFileVersions:
    """Map a memory-file versions body to a :class:`MemoryFileVersions`."""
    return MemoryFileVersions(
        entity_id=raw.get("entity_id") or "",
        partition=raw.get("partition"),
        name=raw.get("name") or "",
        file_id=raw.get("file_id") or "",
        current_version=_int_field(raw, "current_version"),
        versions=[
            MemoryFileVersion(
                doc_id=v.get("doc_id") or "",
                version=_int_field(v, "version"),
                bytes=_int_field(v, "bytes"),
                content_hash=v.get("content_hash") or "",
                author=v.get("author"),
                principal=v.get("principal") or "",
                op=v.get("op") or "",
                updated_at=v.get("updated_at") or "",
                is_current=bool(v.get("is_current", False)),
            )
            for v in raw.get("versions") or []
        ],
    )


def parse_memory_file_diff(raw: dict[str, Any]) -> MemoryFileDiff:
    """Map a memory-file diff body to a :class:`MemoryFileDiff`."""
    return MemoryFileDiff(
        entity_id=raw.get("entity_id") or "",
        partition=raw.get("partition"),
        name=raw.get("name") or "",
        file_id=raw.get("file_id") or "",
        from_version=_int_field(raw, "from_version"),
        to_version=_int_field(raw, "to_version"),
        context=_int_field(raw, "context"),
        diff=raw.get("diff") or "",
        added_lines=_int_field(raw, "added_lines"),
        removed_lines=_int_field(raw, "removed_lines"),
        approximate=bool(raw.get("approximate", False)),
        truncated=bool(raw.get("truncated", False)),
    )


def parse_memory_file_page(raw: dict[str, Any]) -> MemoryFilePage:
    """Map a memory-file list body to a :class:`MemoryFilePage`."""
    return MemoryFilePage(
        files=[parse_memory_file_info(f) for f in raw.get("files") or []],
        next_cursor=raw.get("next_cursor"),
    )


def parse_memory_grep_result(raw: dict[str, Any]) -> MemoryGrepResult:
    """Map a memory-file grep body to a :class:`MemoryGrepResult`."""
    return MemoryGrepResult(
        matches=[
            MemoryGrepMatch(
                entity_id=m.get("entity_id") or "",
                partition=m.get("partition"),
                name=m.get("name") or "",
                version=_int_field(m, "version"),
                line=_int_field(m, "line"),
                section=m.get("section"),
                section_slug=m.get("section_slug"),
                text=m.get("text") or "",
                before=list(m.get("before") or []),
                after=list(m.get("after") or []),
            )
            for m in raw.get("matches") or []
        ],
        truncated=bool(raw.get("truncated", False)),
        truncated_reason=raw.get("truncated_reason"),
        files_scanned=_int_field(raw, "files_scanned"),
        bytes_scanned=_int_field(raw, "bytes_scanned"),
    )


def parse_memory_file_delete_result(raw: dict[str, Any]) -> MemoryFileDeleteResult:
    """Map a memory-file delete body to a :class:`MemoryFileDeleteResult`."""
    return MemoryFileDeleteResult(
        entity_id=raw.get("entity_id") or "",
        partition=raw.get("partition"),
        name=raw.get("name") or "",
        file_id=raw.get("file_id") or "",
        version=_int_field(raw, "version"),
        hard=bool(raw.get("hard", False)),
        versions_deleted=_int_field(raw, "versions_deleted"),
    )


def parse_document_summary(raw: dict[str, Any]) -> DocumentSummary:
    """Map a ``/documents/{id}/summary`` body to a :class:`DocumentSummary`."""
    return DocumentSummary(
        doc_id=raw.get("doc_id") or "",
        state=raw.get("state") or "none",
        reason=raw.get("reason"),
        error=raw.get("error"),
        summary_doc_id=raw.get("summary_doc_id"),
        content=raw.get("content"),
        revision=raw.get("revision"),
        authored_by=raw.get("authored_by"),
        generator=raw.get("generator"),
        prompt_version=raw.get("prompt_version"),
        generated_at=raw.get("generated_at"),
        source_cid=raw.get("source_cid"),
        stale=bool(raw.get("stale", False)),
        truncated_input=bool(raw.get("truncated_input", False)),
    )


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
    #: Ask for (``True``) or opt out of (``False``) a generated summary for
    #: this document, overriding the batch-level ``summarize``. ``None``
    #: (default) sends nothing.
    summarize: Optional[bool] = None

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


# ── Connections + connect sessions ────


@dataclass
class ConnectSession:
    """Response of :meth:`AetherClient.create_connect_session`."""
    #: Opaque, single-use. Embedded in ``connect_url``.
    session_token: str
    #: Open this in the end user's browser to start the hosted OAuth flow.
    connect_url: str
    #: Returned exactly once. Store server-side; use it with
    #: :func:`aether.connections.verify_redirect_signature`.
    client_secret: str
    expires_at: str


@dataclass
class Connection:
    """One connection — a developer's own source (mode A) or one end user's
    (mode B). Never carries credential material."""
    connection_id: str
    provider: str
    owner_type: str  # "tenant" | "external_user"
    owner_id: Optional[str]
    provider_account_id: str
    account_display_name: Optional[str]
    #: ``None`` for mode A (the tenant's default partition); the end user's
    #: id for mode B.
    target_partition: Optional[str]
    status: str
    granted_scopes: list[str]
    created_at: str
    last_sync_at: Optional[str]
    last_error: Optional[str]
    files_synced: int
    files_skipped: int
    files_deleted: int
    selected_paths: list[str]
    purge_state: str  # "not_started" | "in_flight" | "complete"
    purge_receipt_id: Optional[str]
    credential_deleted: bool
    #: True when this row's provider-side identity was withheld because the
    #: call was not scoped to the connection's partition: ``provider_account_id``
    #: is then ``""`` and ``account_display_name`` ``None``. Every other field
    #: is real. Pass ``partition=`` — the same scope the by-id routes require —
    #: to get the identity. Always ``False`` for mode-A connections and on
    #: by-id reads. Defaulted so an older engine simply never redacts.
    identity_redacted: bool = False

    @classmethod
    def _from_wire(cls, body: dict) -> "Connection":
        return cls(
            connection_id=body["connection_id"],
            provider=body["provider"],
            owner_type=body["owner_type"],
            owner_id=body.get("owner_id"),
            provider_account_id=body["provider_account_id"],
            account_display_name=body.get("account_display_name"),
            target_partition=body.get("target_partition"),
            status=body["status"],
            granted_scopes=body.get("granted_scopes", []),
            created_at=body["created_at"],
            last_sync_at=body.get("last_sync_at"),
            last_error=body.get("last_error"),
            files_synced=body.get("files_synced", 0),
            files_skipped=body.get("files_skipped", 0),
            files_deleted=body.get("files_deleted", 0),
            selected_paths=body.get("selected_paths", []),
            purge_state=body.get("purge_state", "not_started"),
            purge_receipt_id=body.get("purge_receipt_id"),
            credential_deleted=body.get("credential_deleted", False),
            identity_redacted=body.get("identity_redacted", False),
        )


@dataclass
class PurgeSummary:
    """Summary of what a disconnect destroyed. Only present on
    :attr:`DisconnectResult.purge` when the connection actually existed."""
    receipt_id: str
    documents_purged: int
    merkle_root: str
    completed_at: str
    signer_node_id: str


@dataclass
class DisconnectResult:
    """Result of :meth:`AetherClient.delete_connection`."""
    connection_id: str
    status: str
    #: ``None`` only when the connection did not exist (the idempotent
    #: no-op) — a disconnect that ran always produces a receipt.
    purge: Optional["PurgeSummary"] = None


@dataclass
class ConnectionPurgeReceipt:
    """The full signed proof :meth:`AetherClient.get_purge_receipt` returns."""
    version: str
    receipt_id: str
    tenant_id: str
    connection_id: str
    provider: str
    owner: str
    provider_account_id: str
    documents_purged: int
    documents_failed: int
    merkle_root: str
    merkle_leaf_count: int
    purged_document_ids: list[str]
    partitions_touched: list[str]
    default_partition_touched: bool
    credential_revocation: str
    credential_deleted: bool
    started_at: str
    completed_at: str
    signer_node_id: str
    signer_public_key: str
    signature: str
    #: The node's own re-verification of the row at read time.
    verified: bool


@dataclass
class ConnectionBrowseEntry:
    """One entry in a :meth:`AetherClient.browse_connection` page."""
    name: str
    path_display: str
    is_folder: bool
    size_bytes: Optional[int] = None
    modified: Optional[str] = None


@dataclass
class ConnectionBrowsePage:
    """One page of :meth:`AetherClient.browse_connection`."""
    entries: list["ConnectionBrowseEntry"]
    #: Present iff another page exists; pass back as the next call's ``cursor``.
    next_cursor: Optional[str] = None
