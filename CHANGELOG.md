# Changelog

All notable changes to the `aether-ai` Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.4.0]: https://github.com/quintessence-group/aether-sdk-python/releases/tag/v0.4.0
