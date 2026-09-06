# Changelog

All notable changes to the `aether-ai` Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0]

### Added

- **Thread lifecycle.** Whole-thread operations on the raw client —
  `client.restore_thread(thread_id)`, `client.set_thread_acl(thread_id, acl_readers)`,
  `client.move_thread(thread_id, to_partition=..., expect_partition=...)`, and
  `client.delete_thread(thread_id, hard=False)` — each returning a uniform
  `ThreadLifecycleResult` (`status`, `thread_id`, `turns`). The
  `memory.thread(thread_id)` facade (`Thread` / `AsyncThread`) gains the same
  operations as `restore()`, `set_acl(readers)`, `move(to_partition)`, and
  `delete(hard=False)`. Every operation sends an `Idempotency-Key` (pass
  `idempotency_key=` to make cross-process retries safe). Turn text is never
  rewritten — an edit appends a correction turn — and deletes are soft by
  default; `hard=True` is an irreversible erasure. Mirrored on
  `AsyncAetherClient`.
- **Connections API + connect sessions.** Attach an end user's external account
  (Dropbox today) to their partition from your own backend, without the portal:
  - `client.create_connect_session(external_user_id, return_url, provider="dropbox", target_partition=None)`
    mints a hosted OAuth entry point and returns a `ConnectSession`
    (`session_token`, `connect_url`, a one-time `client_secret`, `expires_at`).
  - `aether.verify_redirect_signature(client_secret, session=..., status=..., connection_id=..., sig=...)`
    verifies the signed redirect back to your `return_url` entirely offline
    (HMAC-SHA256 over `session|status|connection_id`, keyed by
    `SHA-256(client_secret)`). Standard-library crypto only; no new
    dependencies.
  - `client.list_connections(owner_type=..., owner_id=..., include_purged=...)`,
    `client.get_connection(id)`, `client.resync_connection(id)`,
    `client.browse_connection(id, path="", cursor=None)`, and
    `client.update_selection(id, selected_paths)` manage a connection and its
    sync scope. `client.delete_connection(id)` purges the synced content and
    returns a `DisconnectResult`; the signed purge receipt is fetchable with
    `client.get_purge_receipt(receipt_id)` (`ConnectionPurgeReceipt`).
  - New models: `ConnectSession`, `Connection`, `ConnectionBrowseEntry`,
    `ConnectionBrowsePage`, `DisconnectResult`, `PurgeSummary`,
    `ConnectionPurgeReceipt`. All of the above is mirrored on
    `AsyncAetherClient`.
- **Typed connect-session errors.** `SessionInvalidError` (HTTP 400,
  `code="session_invalid"` — the session token is unknown, already used, or
  expired; mint a new session instead of retrying) and
  `PartitionMismatchError` (HTTP 400, `code="partition_mismatch"` — the
  handle's partition disagrees with where the session would resolve). Neither
  is retryable.

### Notes

- Purely additive: no breaking changes. Existing calls behave exactly as before.

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

[0.6.0]: https://github.com/quintessence-group/aether-sdk-python/releases/tag/v0.6.0
[0.5.0]: https://github.com/quintessence-group/aether-sdk-python/releases/tag/v0.5.0
[0.4.0]: https://github.com/quintessence-group/aether-sdk-python/releases/tag/v0.4.0
