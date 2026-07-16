# Changelog

All notable changes to the `aether-ai` Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0]

### Added

- **Per-user permissions & audit.** Documents can now carry a read-ACL, and a
  client handle can act on behalf of a principal so reads are filtered by that
  ACL:
  - Pass `acl_readers=["user:alice", "group:eng"]` to `insert`, `insert_text`,
    `ingest_files`, and `ingest_directory` to restrict who can read a document.
    Omit it (or pass `[]`) for the admin-only default.
  - `client.as_principal("user:alice", groups=[...])` returns a scoped clone
    whose reads and searches only surface documents the principal is allowed to
    see (unlabeled documents plus those whose ACL names it or one of its
    groups). Composes with `client.partition(...)`. Admin-role keys bypass
    filtering.
  - `client.audit.access(...)` (`AuditClient` / `AsyncAuditClient`) queries the
    tenant's access-audit log — document reads, search deliveries, denials, and
    admin bypasses — returning an `AccessAuditPage`. Requires access-audit
    capture to be enabled for the tenant.
  - New typed error `PrincipalPinMismatchError` (HTTP 403,
    `code="principal_pin_mismatch"`) is raised when a principal-pinned API key
    is asked to assert a different principal. Not retryable.
- **Durable conversation threads.** `client.append_thread(...)` and
  `client.get_thread(...)` store and replay an ordered message history
  (`ConversationThread`) for an agent or chat session, with a
  `memory.thread(thread_id)` facade (`Thread` / `AsyncThread`).
- **Shared grounding provenance receipts.** New `GroundingReceipt`,
  `GroundingBinding`, `GroundingSource`, `GroundingTrustSignal`,
  `GroundingSetAttestation`, `ReceiptAttestation`, and `ShareableReceipt` types
  expose signed, shareable provenance for a generated answer's sources.
- **Multimodal recall.** Image and audio memories can be remembered and
  recalled, surfaced through the new `MediaMemoryRecord` type.

### Notes

- Purely additive: no breaking changes. All new write parameters are optional
  and existing calls behave exactly as before.

## [0.4.0]

### Added

- **Move a document between partitions.** `client.move_document(doc_id,
  from_partition=..., to_partition=...)` relocates an existing document from one
  partition to another in a single call (`POST /v1/documents/{id}/move`).
  Available on both `AetherClient` and `AsyncAetherClient`.
- **Analytical `query()`.** `client.query(...)` runs structured filter, sort, and
  aggregation queries over your documents (`POST /v1/query`), returning grouped
  `AggregateResult` / `QueryGroup` rows or matching records. Mirrored on the async
  client.
- **Field-schema facade.** `client.schema` (`SchemaClient` / `AsyncSchemaClient`)
  declares and manages the typed fields that `query()` filters, sorts, and
  aggregates over — `declare_fields()`, `list_fields()`, and `delete_field()`.
  Fields are extracted from document metadata or passage text and returned as
  `FieldSchema` records.
- **`partition` on document, search, and insert results.** `DocumentRecord`,
  `SearchResult`, and `RetrievalResult` now carry the `partition` the record lives
  in, echoed back by the API (mirrors the existing `entity_id` / `source`
  convention). `None` means the default partition.
- **Typed `PartitionRequiredError`.** A key that requires every call to name a
  partition now raises `PartitionRequiredError` (a subclass of `AetherApiError`)
  on an unscoped call, instead of a generic API error. Scope the call through
  `client.partition("<id>")`. Not retryable.

### Changed

- Partition-scoped handles (`client.partition("x")`) now pin the id-addressed
  operations — `get`, `download`, `delete`, and `restore` — to the handle's
  partition, matching the scoping already applied to search, insert, and list.

[0.5.0]: https://github.com/quintessence-group/aether-sdk-python/releases/tag/v0.5.0
[0.4.0]: https://github.com/quintessence-group/aether-sdk-python/releases/tag/v0.4.0
