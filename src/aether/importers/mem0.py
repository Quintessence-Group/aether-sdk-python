"""One-shot importer: a Mem0 export → Aether's memory model.

Maps the shapes Mem0 actually exports (``get_all()`` output, with or without
the graph-store ``relations``) onto Aether's :class:`~aether.Memory` facade —
one facade instance per Mem0 owner — so everything imported is immediately
queryable through ``recall`` / ``list`` / the memory-graph read methods.

The authoritative schema-mapping document lives in the monorepo at
``docs/importers/mem0-mapping.md``. Summary of the mapping implemented here:

- **Owner scoping** — memories are grouped by ``user_id`` (falling back to
  ``agent_id``, then ``run_id``, else a single default bucket). Each owner
  becomes one ``Memory`` facade with
  ``entity_id = f"{entity_prefix}:{kind}:{value}"`` (e.g. ``mem0:user:alice``);
  the default bucket is ``f"{entity_prefix}:default"``.
- **Memories** — each Mem0 memory becomes ``remember(text, metadata=...)``.
  The metadata carries ``source: "mem0"``, ``mem0_id``, the original
  timestamps as ``mem0_created_at`` / ``mem0_updated_at`` (Aether's own
  ``created_at`` is write-time), Mem0 ``categories`` as a joined
  ``mem0_categories`` string plus per-category ``category_<slug>: True``
  filter flags, and the Mem0 ``metadata`` dict merged in (importer keys win
  on collision). Pairs the facade's scalar, tag-mirrorable metadata cannot
  represent are dropped and counted (``metadata_pairs_dropped``).
- **Relations** (graph-enabled Mem0) — each relation's ``source`` / ``target``
  node becomes ``upsert_entity`` (entity type from ``source_type`` /
  ``target_type`` when given, else ``"mem0_node"``; the node name doubles as
  the idempotent ``memory_entity_id``), then the edge becomes
  ``relate(from_id, to_id, relationship)``.

Sync only: this importer drives the synchronous :class:`~aether.AetherClient`
(an async variant is an explicit follow-up). No new dependencies — stdlib
``json`` only. Writes are not transactional: with ``on_error="raise"`` an
error mid-import leaves earlier writes in place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from ..client import AetherClient
from ..errors import AetherError
from ..memory import Memory, _encode_legacy_metadata_tags, _validate_entity_id
from ..models import Metadata

__all__ = ["Mem0ImportReport", "import_mem0"]

_ON_ERROR_MODES = ("skip", "raise")

# Owner precedence (first match wins): user_id → agent_id → run_id.
_OWNER_KEYS = (("user", "user_id"), ("agent", "agent_id"), ("run", "run_id"))

# Default entity type for graph nodes when the relation carries no type.
_DEFAULT_NODE_TYPE = "mem0_node"


@dataclass
class Mem0ImportReport:
    """Outcome of one :func:`import_mem0` run.

    - ``memories_imported`` / ``entities_created`` / ``relationships_created``
      — successful ``remember`` / ``upsert_entity`` / ``relate`` calls (what
      *would* be written when ``dry_run=True``). Graph nodes are deduplicated
      per owner, so a node referenced by many relations counts once.
    - ``skipped`` + ``skips`` — items not imported, as ``(item_id, reason)``
      pairs. ``item_id`` is the memory's Mem0 ``id`` when present, else a
      positional ``memories[i]`` / ``relations[i]`` reference.
    - ``metadata_pairs_dropped`` — individual metadata key/value pairs the
      facade's scalar metadata cannot represent (the memory itself still
      imports; see the mapping doc for the exact rules).
    - ``owners`` — the sorted Aether ``entity_id`` values the import wrote
      (or, for a dry run, would write) under; construct ``Memory`` facades
      with these to query the imported data.
    """

    memories_imported: int = 0
    entities_created: int = 0
    relationships_created: int = 0
    skipped: int = 0
    skips: list[tuple[str, str]] = field(default_factory=list)
    metadata_pairs_dropped: int = 0
    owners: list[str] = field(default_factory=list)
    dry_run: bool = False


class _SkipItem(Exception):
    """Internal: one item cannot be imported; carries the report entry."""

    def __init__(self, ref: str, reason: str) -> None:
        super().__init__(f"{ref}: {reason}")
        self.ref = ref
        self.reason = reason


@dataclass
class _MappedMemory:
    ref: str
    entity_id: str
    text: str
    metadata: Metadata


@dataclass
class _MappedRelation:
    ref: str
    entity_id: str
    source: str
    source_type: str
    relationship: str
    target: str
    target_type: str


# ── input handling ───────────────────────────────────────────────────


def _load_export(data: Union[str, Path, dict, list]) -> tuple[list, list]:
    """Resolve *data* to ``(memories, relations)`` lists.

    Accepted top-level shapes (documented in ``docs/importers/mem0-mapping.md``):

    - a path (``str`` / ``Path``) to a JSON file containing one of the shapes
      below;
    - a bare ``list`` of memory objects (Mem0 v1.0 ``get_all()``);
    - ``{"results": [...]}`` (Mem0 v1.1 ``get_all()``);
    - ``{"results": [...], "relations": [...]}`` (graph-enabled Mem0).

    Anything else is a ``ValueError`` — a malformed *export* always raises,
    regardless of ``on_error`` (which governs per-item problems only).
    """
    if isinstance(data, (str, Path)):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)
    if isinstance(data, list):
        return data, []
    if isinstance(data, dict):
        memories = data.get("results")
        relations = data.get("relations")
        if memories is None and relations is None:
            raise ValueError(
                "unrecognized Mem0 export: expected a list of memories, or an "
                "object with a 'results' (and optionally 'relations') key"
            )
        memories = memories if memories is not None else []
        relations = relations if relations is not None else []
        if not isinstance(memories, list):
            raise ValueError("Mem0 export 'results' must be a list")
        if not isinstance(relations, list):
            raise ValueError("Mem0 export 'relations' must be a list")
        return memories, relations
    raise ValueError(
        f"unsupported Mem0 export type: {type(data).__name__} "
        "(expected a file path, dict, or list)"
    )


# ── mapping helpers ──────────────────────────────────────────────────


def _owner_entity_id(item: dict, entity_prefix: str) -> Optional[str]:
    """The Aether owner ``entity_id`` for one Mem0 object, or ``None`` when it
    carries no owner id (caller decides the fallback)."""
    for kind, key in _OWNER_KEYS:
        value = item.get(key)
        # Mem0 exports usually carry string ids, but integer ids show up in
        # parsed-dict inputs; accept them rather than silently rehoming the
        # item to the default bucket. bool is an int subclass — exclude it.
        if isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
        if isinstance(value, str) and value.strip():
            return f"{entity_prefix}:{kind}:{value}"
    return None


def _sanitize_metadata(raw: dict) -> tuple[Metadata, int]:
    """Keep only pairs ``Memory.remember`` can carry; return ``(kept, dropped)``.

    The facade's metadata is a flat ``{str: str|int|float|bool}`` map whose
    pairs are also mirrored into legacy ``key:value`` tags, so it rejects:
    empty keys, keys containing ``:`` or ``,``, values whose string form
    contains ``,``, and non-scalar (nested / null) values. Offending pairs are
    dropped — the memory itself still imports.
    """
    kept: Metadata = {}
    dropped = 0
    for key, value in raw.items():
        key = str(key)
        if not key or ":" in key or "," in key:
            dropped += 1
            continue
        if not isinstance(value, (str, int, float, bool)):  # None, nested, …
            dropped += 1
            continue
        if "," in str(value):
            dropped += 1
            continue
        kept[key] = value
    return kept, dropped


def _category_slug(name: str) -> str:
    """A category name as a metadata-key-safe slug: whitespace runs, ``:``
    and ``,`` each become ``_`` (original names are kept in ``mem0_categories``)."""
    return "_".join(name.split()).replace(":", "_").replace(",", "_")


def _category_metadata(categories: list) -> Metadata:
    """Mem0 ``categories`` → facade metadata.

    Two representations, both documented in the mapping doc:

    - ``mem0_categories`` — the original names joined with ``|`` (a ``,``
      inside a name becomes ``;``: the legacy tag wire format cannot carry
      commas), preserving order.
    - ``category_<slug>: True`` per category — exact-match filterable via
      ``recall(filter={"category_<slug>": True})`` and mirrored to a
      ``category_<slug>:True`` tag.
    """
    names = [str(c).strip() for c in categories if str(c).strip()]
    if not names:
        return {}
    meta: Metadata = {
        "mem0_categories": "|".join(n.replace(",", ";") for n in names)
    }
    for n in names:
        meta[f"category_{_category_slug(n)}"] = True
    return meta


def _map_memory(
    item: Any, index: int, entity_prefix: str
) -> tuple[_MappedMemory, int]:
    """Validate + map one Mem0 memory object (no I/O). Raises :class:`_SkipItem`."""
    if not isinstance(item, dict):
        raise _SkipItem(f"memories[{index}]", "memory item is not an object")
    mem0_id = item.get("id")
    ref = str(mem0_id) if mem0_id is not None else f"memories[{index}]"

    # ``memory`` is Mem0's canonical text key; older exports used ``text``.
    text = item.get("memory")
    if text is None:
        text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        raise _SkipItem(ref, "missing or empty 'memory' text")

    raw_meta = item.get("metadata")
    if raw_meta is None:
        raw_meta = {}
    if not isinstance(raw_meta, dict):
        raise _SkipItem(ref, "'metadata' is not an object")
    metadata, dropped = _sanitize_metadata(raw_meta)

    categories = item.get("categories")
    if categories is None:
        categories = []
    if not isinstance(categories, list):
        raise _SkipItem(ref, "'categories' is not a list")
    metadata.update(_category_metadata(categories))

    # Importer-provenance keys — applied last, so they win on collision.
    importer_meta: Metadata = {"source": "mem0"}
    if mem0_id is not None:
        importer_meta["mem0_id"] = str(mem0_id)
    if item.get("created_at"):
        importer_meta["mem0_created_at"] = str(item["created_at"])
    if item.get("updated_at"):
        importer_meta["mem0_updated_at"] = str(item["updated_at"])
    metadata.update(importer_meta)

    entity_id = _owner_entity_id(item, entity_prefix) or f"{entity_prefix}:default"

    # Run the facade's own validators here so dry_run reports exactly what a
    # live run would accept (both raise ValueError before any HTTP call).
    try:
        _validate_entity_id(entity_id)
        _encode_legacy_metadata_tags(metadata)
    except ValueError as e:
        raise _SkipItem(ref, str(e)) from None

    # ``score`` (and any other Mem0 field, e.g. ``hash``) is ignored.
    return _MappedMemory(ref, entity_id, text, metadata), dropped


def _map_relation(
    rel: Any, index: int, entity_prefix: str, fallback_owner: str
) -> _MappedRelation:
    """Validate + map one Mem0 graph relation (no I/O). Raises :class:`_SkipItem`.

    Accepted relation shapes (be liberal — both Mem0 graph surfaces):

    - ``{"source", "relationship", "target"}`` — Mem0 graph ``get_all()``;
    - ``{"source", "relationship", "destination"}`` — Mem0 graph ``add()``
      (``added_entities``) triples;
    - ``relation`` is accepted as an alias for ``relationship``;
    - optional ``source_type`` / ``target_type`` / ``destination_type`` set
      the node entity types (default ``"mem0_node"``);
    - optional ``user_id`` / ``agent_id`` / ``run_id`` scope the relation to
      that owner; otherwise it lands under *fallback_owner*.
    """
    ref = f"relations[{index}]"
    if not isinstance(rel, dict):
        raise _SkipItem(ref, "relation is not an object")

    source = rel.get("source")
    relationship = rel.get("relationship")
    if relationship is None:
        relationship = rel.get("relation")
    target = rel.get("target")
    if target is None:
        target = rel.get("destination")
    for name, value in (
        ("source", source),
        ("relationship", relationship),
        ("target' or 'destination", target),
    ):
        if not isinstance(value, str) or not value.strip():
            raise _SkipItem(ref, f"missing or empty '{name}'")

    entity_id = _owner_entity_id(rel, entity_prefix) or fallback_owner
    try:
        _validate_entity_id(entity_id)
    except ValueError as e:
        raise _SkipItem(ref, str(e)) from None

    # Blank/whitespace-only node types fall back to the default so dry-run
    # counts match the live write path, which rejects empty entity types.
    source_type = rel.get("source_type")
    if not isinstance(source_type, str) or not source_type.strip():
        source_type = _DEFAULT_NODE_TYPE
    target_type = rel.get("target_type")
    if not isinstance(target_type, str) or not target_type.strip():
        target_type = rel.get("destination_type")
    if not isinstance(target_type, str) or not target_type.strip():
        target_type = _DEFAULT_NODE_TYPE
    return _MappedRelation(
        ref=ref,
        entity_id=entity_id,
        source=source,
        source_type=source_type,
        relationship=relationship,
        target=target,
        target_type=target_type,
    )


# ── the importer ─────────────────────────────────────────────────────


def import_mem0(
    client: AetherClient,
    data: Union[str, Path, dict, list],
    *,
    entity_prefix: str = "mem0",
    dry_run: bool = False,
    on_error: str = "skip",
) -> Mem0ImportReport:
    """Import a Mem0 export into Aether's memory model through the ``Memory`` facade.

    Args:
        client: A synchronous :class:`~aether.AetherClient`. One
            ``Memory(entity_id, client=client)`` facade is built per Mem0
            owner, so the client's configuration (including any partition
            scope) applies to every write. Async clients are a follow-up.
        data: A path (``str`` / ``Path``) to a JSON file, or an
            already-parsed ``dict`` / ``list``, in one of the accepted Mem0
            export shapes (see :func:`_load_export` / the mapping doc).
        entity_prefix: Prefix for owner entity ids —
            ``f"{entity_prefix}:user:alice"`` etc. Default ``"mem0"``.
        dry_run: Validate and map everything without any HTTP call; the
            report carries the counts a live run would produce.
        on_error: ``"skip"`` (default) records ``(item_id, reason)`` in the
            report and continues; ``"raise"`` re-raises on the first bad item
            or failed write (earlier writes are **not** rolled back).

    Returns:
        A :class:`Mem0ImportReport`.

    Raises:
        ValueError: for a malformed export, an invalid ``on_error`` /
            ``entity_prefix``, or (with ``on_error="raise"``) a bad item.
        AetherError: with ``on_error="raise"``, when a write fails.
        TypeError: when *client* is not a synchronous ``AetherClient``.
    """
    if on_error not in _ON_ERROR_MODES:
        raise ValueError("on_error must be 'skip' or 'raise'")
    if not isinstance(entity_prefix, str) or not entity_prefix.strip():
        raise ValueError("entity_prefix cannot be empty")
    if not isinstance(client, AetherClient):
        raise TypeError(
            "import_mem0 requires a synchronous AetherClient "
            "(async import is not supported yet)"
        )

    memories_raw, relations_raw = _load_export(data)
    report = Mem0ImportReport(dry_run=dry_run)

    def _skip(ref: str, reason: str) -> None:
        report.skipped += 1
        report.skips.append((ref, reason))

    # Phase 1 — map + validate every memory (no I/O), so dry_run and the live
    # run share exactly one validation path.
    mapped_memories: list[_MappedMemory] = []
    for i, item in enumerate(memories_raw):
        try:
            mapped, dropped = _map_memory(item, i, entity_prefix)
        except _SkipItem as e:
            if on_error == "raise":
                raise ValueError(str(e)) from None
            _skip(e.ref, e.reason)
            continue
        report.metadata_pairs_dropped += dropped
        mapped_memories.append(mapped)

    # Relations without their own owner ids: when every memory belongs to one
    # owner, attach them there; otherwise to the default bucket.
    memory_owners = {m.entity_id for m in mapped_memories}
    fallback_owner = (
        next(iter(memory_owners))
        if len(memory_owners) == 1
        else f"{entity_prefix}:default"
    )

    mapped_relations: list[_MappedRelation] = []
    for i, rel in enumerate(relations_raw):
        try:
            mapped_relations.append(
                _map_relation(rel, i, entity_prefix, fallback_owner)
            )
        except _SkipItem as e:
            if on_error == "raise":
                raise ValueError(str(e)) from None
            _skip(e.ref, e.reason)

    report.owners = sorted(
        memory_owners | {r.entity_id for r in mapped_relations}
    )

    # Phase 2 — write through the Memory facade (skipped when dry_run).
    facades: dict[str, Memory] = {}

    def _memory_for(entity_id: str) -> Memory:
        if entity_id not in facades:
            facades[entity_id] = Memory(entity_id, client=client)
        return facades[entity_id]

    for m in mapped_memories:
        if not dry_run:
            try:
                _memory_for(m.entity_id).remember(m.text, metadata=m.metadata)
            except (ValueError, AetherError) as e:
                if on_error == "raise":
                    raise
                _skip(m.ref, f"remember failed: {e}")
                continue
        report.memories_imported += 1

    # Graph nodes are deduplicated per owner: (owner, node name) → the node's
    # memory_entity_id. The node name is used as the idempotent
    # ``memory_entity_id``, so re-running the import updates rather than
    # duplicates nodes. A node seen with two types keeps the first type.
    node_ids: dict[tuple[str, str], str] = {}

    def _ensure_node(mem: Memory, owner: str, name: str, node_type: str) -> str:
        key = (owner, name)
        if key in node_ids:
            return node_ids[key]
        if dry_run:
            node_ids[key] = name
        else:
            entity = mem.upsert_entity(
                node_type, memory_entity_id=name, display_name=name
            )
            node_ids[key] = entity.memory_entity_id
        report.entities_created += 1
        return node_ids[key]

    for r in mapped_relations:
        try:
            mem = _memory_for(r.entity_id) if not dry_run else None
            from_id = _ensure_node(mem, r.entity_id, r.source, r.source_type)
            to_id = _ensure_node(mem, r.entity_id, r.target, r.target_type)
            if not dry_run:
                mem.relate(from_id, to_id, r.relationship)
        except (ValueError, AetherError) as e:
            if on_error == "raise":
                raise
            _skip(r.ref, f"relation import failed: {e}")
            continue
        report.relationships_created += 1

    return report
