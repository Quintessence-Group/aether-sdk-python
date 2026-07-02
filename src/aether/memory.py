"""Entity-scoped ``Memory`` facade over the Aether raw client.

``Memory`` is a thin, ergonomic wrapper around :class:`~aether.AetherClient`
(and ``AsyncMemory`` around :class:`~aether.AsyncAetherClient`). Construct it
once with an ``entity_id`` — a user, customer, patient, or agent session — and
every call is automatically scoped to that entity.

It adds **no new HTTP routes** and changes **no existing behavior**: all
transport, retry, error, and timeout semantics are inherited unchanged from the
raw client, and the raw client's existing error types are surfaced verbatim.

See ``docs/MEMORY_CONTRACT.md`` for the authoritative cross-SDK definition.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Union
from urllib.parse import quote

from .async_client import AsyncAetherClient
from .client import AetherClient
from .models import Metadata, MetadataFilter

# Algorithm constants — see MEMORY_CONTRACT.md §4 Mode B. Identical across all
# four SDKs so the contract test produces the same ordering everywhere.
OVERFETCH = 4
MAX_CANDIDATES = 100

# entity_id constraint mirrors the server: 1–256 chars.
_MAX_ENTITY_ID_LEN = 256

@dataclass
class MemoryItem:
    """A single remembered item — the shared result type for ``remember``,
    ``recall``, and ``list``.

    Fields are populated depending on the call that produced the item (see
    MEMORY_CONTRACT.md §2):

    - ``id`` — the underlying ``doc_id``. Always populated.
    - ``text`` — the remembered text. Always populated.
    - ``created_at`` — RFC 3339 string, **unparsed** (kept as a string per the
      raw contract §7). Populated by ``remember`` and ``list``; by ``recall``
      only when ``recency_weight > 0``, else ``None``.
    - ``entity_id`` — the owning entity (= the ``Memory``'s ``entity_id``).
    - ``score`` — relevance signal, higher = more relevant. **Relative within a
      single ``recall`` call; not comparable across calls.** ``recall`` only;
      ``None`` for ``remember``/``list``.

    - ``metadata`` — structured metadata echoed by the raw document API.
    """

    id: str
    text: str
    created_at: Optional[str] = None
    entity_id: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)
    score: Optional[float] = None


# ── Memory graph result types (Part II, ADR-019) ──────────────────────────
#
# Read models mirroring the engine's /v1/memory/* response DTOs 1:1. Scalar
# ``attributes``/``value`` are str | int | float | bool | None (the engine
# rejects nested objects/arrays). Timestamps are RFC 3339 strings, unparsed.
ScalarValue = Optional[Union[str, int, float, bool]]


@dataclass
class MemoryEntity:
    """A typed node in the owner's memory graph (`/v1/memory/entities`)."""

    memory_entity_id: str
    entity_id: str
    entity_type: str
    partition: Optional[str] = None
    display_name: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, ScalarValue] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class MemoryRelationship:
    """A directed, typed edge between two entities (`/v1/memory/relationships`)."""

    relationship_id: str
    entity_id: str
    from_entity_id: str
    to_entity_id: str
    relationship_type: str
    partition: Optional[str] = None
    attributes: dict[str, ScalarValue] = field(default_factory=dict)
    valid_from: Optional[str] = None
    observed_at: str = ""
    invalid_from: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class MemoryFact:
    """A temporal assertion with contradiction-resolution history (`/v1/memory/facts`)."""

    fact_id: str
    entity_id: str
    subject_type: str
    predicate: str
    value: ScalarValue
    cardinality: str
    partition: Optional[str] = None
    subject_id: Optional[str] = None
    valid_from: Optional[str] = None
    observed_at: str = ""
    invalid_from: Optional[str] = None
    supersedes_fact_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ConsolidationReport:
    """Report returned by ``consolidate`` (`POST /v1/memory/consolidate`)."""

    active_facts_before: int
    active_facts_after: int
    retracted: int


def _parse_entity(d: dict) -> MemoryEntity:
    return MemoryEntity(
        memory_entity_id=d["memory_entity_id"],
        entity_id=d.get("entity_id", ""),
        entity_type=d.get("entity_type", ""),
        partition=d.get("partition"),
        display_name=d.get("display_name"),
        aliases=list(d.get("aliases") or []),
        attributes=dict(d.get("attributes") or {}),
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", ""),
    )


def _parse_relationship(d: dict) -> MemoryRelationship:
    return MemoryRelationship(
        relationship_id=d["relationship_id"],
        entity_id=d.get("entity_id", ""),
        from_entity_id=d.get("from_entity_id", ""),
        to_entity_id=d.get("to_entity_id", ""),
        relationship_type=d.get("relationship_type", ""),
        partition=d.get("partition"),
        attributes=dict(d.get("attributes") or {}),
        valid_from=d.get("valid_from"),
        observed_at=d.get("observed_at", ""),
        invalid_from=d.get("invalid_from"),
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", ""),
    )


def _parse_fact(d: dict) -> MemoryFact:
    return MemoryFact(
        fact_id=d["fact_id"],
        entity_id=d.get("entity_id", ""),
        subject_type=d.get("subject_type", ""),
        predicate=d.get("predicate", ""),
        value=d.get("value"),
        cardinality=d.get("cardinality", "single"),
        partition=d.get("partition"),
        subject_id=d.get("subject_id"),
        valid_from=d.get("valid_from"),
        observed_at=d.get("observed_at", ""),
        invalid_from=d.get("invalid_from"),
        supersedes_fact_id=d.get("supersedes_fact_id"),
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", ""),
    )


def _parse_consolidation(d: dict) -> ConsolidationReport:
    return ConsolidationReport(
        active_facts_before=d.get("active_facts_before", 0),
        active_facts_after=d.get("active_facts_after", 0),
        retracted=d.get("retracted", 0),
    )


_VALID_SUBJECT_TYPES = ("owner", "entity", "relationship")


def _validate_subject(
    subject_type: str, subject_id: Optional[str]
) -> tuple[str, Optional[str]]:
    """Validate (subject_type, subject_id) client-side (MEMORY_CONTRACT.md §13)."""
    if subject_type not in _VALID_SUBJECT_TYPES:
        raise ValueError("subject_type must be 'owner', 'entity', or 'relationship'")
    if subject_type == "owner":
        return subject_type, None
    if not subject_id:
        raise ValueError(
            f"subject_id is required when subject_type is '{subject_type}'"
        )
    return subject_type, subject_id


def _validate_cardinality(cardinality: Optional[str]) -> Optional[str]:
    if cardinality is None:
        return None
    if cardinality not in ("single", "multi"):
        raise ValueError("cardinality must be 'single' or 'multi'")
    return cardinality


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")


# Graph request specs. Each returns ``(method, path, params, json_body, parse)``
# where ``parse`` maps the response JSON to the result type/list. ``params``
# excludes ``entity_id``/``partition`` — those are injected by ``_graph_request``
# from the Memory's scope. Validation lives here so it is shared by Memory and
# AsyncMemory (and raised before any HTTP call). See MEMORY_CONTRACT.md §12–§13.


def _spec_upsert_entity(entity_type, memory_entity_id, display_name, aliases, attributes):
    _require_nonempty("entity_type", entity_type)
    body: dict = {"entity_type": entity_type}
    if memory_entity_id:
        body["memory_entity_id"] = memory_entity_id
    if display_name is not None:
        body["display_name"] = display_name
    if aliases is not None:
        body["aliases"] = list(aliases)
    if attributes is not None:
        body["attributes"] = dict(attributes)
    return "POST", "/memory/entities", {}, body, _parse_entity


def _spec_get_entity(memory_entity_id):
    _require_nonempty("memory_entity_id", memory_entity_id)
    return "GET", f"/memory/entities/{quote(memory_entity_id)}", {}, None, _parse_entity


def _spec_list_entities(entity_type, limit):
    params: dict = {}
    if entity_type:
        params["entity_type"] = entity_type
    if limit is not None:
        params["limit"] = limit
    return (
        "GET",
        "/memory/entities",
        params,
        None,
        lambda d: [_parse_entity(e) for e in d.get("entities", [])],
    )


def _spec_relate(from_entity_id, to_entity_id, relationship_type, relationship_id, attributes, valid_from):
    _require_nonempty("from_entity_id", from_entity_id)
    _require_nonempty("to_entity_id", to_entity_id)
    _require_nonempty("relationship_type", relationship_type)
    body: dict = {
        "from_entity_id": from_entity_id,
        "to_entity_id": to_entity_id,
        "relationship_type": relationship_type,
    }
    if relationship_id:
        body["relationship_id"] = relationship_id
    if attributes is not None:
        body["attributes"] = dict(attributes)
    if valid_from is not None:
        body["valid_from"] = valid_from
    return "POST", "/memory/relationships", {}, body, _parse_relationship


def _spec_list_relationships(from_entity_id, to_entity_id, relationship_type, include_inactive, as_of, limit):
    params: dict = {}
    if from_entity_id:
        params["from_entity_id"] = from_entity_id
    if to_entity_id:
        params["to_entity_id"] = to_entity_id
    if relationship_type:
        params["relationship_type"] = relationship_type
    if include_inactive:
        params["include_inactive"] = "true"
    if as_of:
        params["as_of"] = as_of
    if limit is not None:
        params["limit"] = limit
    return (
        "GET",
        "/memory/relationships",
        params,
        None,
        lambda d: [_parse_relationship(r) for r in d.get("relationships", [])],
    )


def _spec_remember_fact(predicate, value, subject_type, subject_id, cardinality, valid_from, observed_at, supersedes_fact_id):
    _require_nonempty("predicate", predicate)
    subject_type, subject_id = _validate_subject(subject_type, subject_id)
    cardinality = _validate_cardinality(cardinality)
    body: dict = {"subject_type": subject_type, "predicate": predicate, "value": value}
    if subject_id is not None:
        body["subject_id"] = subject_id
    if cardinality is not None:
        body["cardinality"] = cardinality
    if valid_from is not None:
        body["valid_from"] = valid_from
    if observed_at is not None:
        body["observed_at"] = observed_at
    if supersedes_fact_id:
        body["supersedes_fact_id"] = supersedes_fact_id
    return "POST", "/memory/facts", {}, body, _parse_fact


def _spec_list_facts(subject_type, subject_id, predicate, include_inactive, as_of, limit):
    params: dict = {}
    if subject_type is not None:
        subject_type, subject_id = _validate_subject(subject_type, subject_id)
        params["subject_type"] = subject_type
        if subject_id is not None:
            params["subject_id"] = subject_id
    if predicate:
        params["predicate"] = predicate
    if include_inactive:
        params["include_inactive"] = "true"
    if as_of:
        params["as_of"] = as_of
    if limit is not None:
        params["limit"] = limit
    return (
        "GET",
        "/memory/facts",
        params,
        None,
        lambda d: [_parse_fact(f) for f in d.get("facts", [])],
    )


def _spec_fact_history(predicate, subject_type, subject_id):
    _require_nonempty("predicate", predicate)
    subject_type, subject_id = _validate_subject(subject_type, subject_id)
    params: dict = {"history": "true", "subject_type": subject_type, "predicate": predicate}
    if subject_id is not None:
        params["subject_id"] = subject_id
    return (
        "GET",
        "/memory/facts",
        params,
        None,
        lambda d: [_parse_fact(f) for f in d.get("facts", [])],
    )


def _spec_consolidate():
    return "POST", "/memory/consolidate", {}, None, _parse_consolidation


def _validate_entity_id(entity_id: str) -> str:
    """Validate ``entity_id`` client-side (never a network round-trip)."""
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise ValueError("entity_id cannot be empty")
    if len(entity_id) > _MAX_ENTITY_ID_LEN:
        raise ValueError(
            f"entity_id must be at most {_MAX_ENTITY_ID_LEN} characters "
            f"(got {len(entity_id)})"
        )
    return entity_id


def _encode_legacy_metadata_tags(metadata: Optional[Metadata]) -> Optional[list[str]]:
    """Best-effort legacy tag mirror for ``remember(..., metadata)``.

    Structured metadata is authoritative. Tags are emitted only when the old
    comma-joined ``key:value`` format can represent the pair losslessly.
    """
    if not metadata:
        return None
    tags: list[str] = []
    for key in sorted(metadata):
        value = metadata[key]
        if not key:
            raise ValueError("metadata keys must not be empty")
        if ":" in key or "," in key:
            raise ValueError(
                "metadata keys must not contain ':' or ',' "
                "(':' is the tag separator; ',' joins tags on the wire)"
            )
        if "," in str(value):
            raise ValueError(
                "metadata values must not contain ',' "
                "(the comma-joined tag wire format cannot escape it)"
            )
        tags.append(f"{key}:{value}")
    return tags


def _normalize_text(text: str) -> str:
    """Reject empty/whitespace-only text client-side (MEMORY_CONTRACT.md §3)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text cannot be empty")
    return text


def _clamp_recency_weight(recency_weight: float) -> float:
    """Clamp ``recency_weight`` to ``[0, 1]`` (MEMORY_CONTRACT.md §4)."""
    if recency_weight < 0.0:
        return 0.0
    if recency_weight > 1.0:
        return 1.0
    return recency_weight


# A fact's tag marking it as an extracted fact, and the confidence tag
# (corroborating-source count) that consolidation grows.
_FACT_KIND_TAG = "kind:fact"
_FACT_CONFIDENCE_PREFIX = "conf:"


def _fact_confidence(tags: list[str]) -> int:
    """Confidence (corroborating-source count) parsed from a fact's ``conf:`` tag,
    defaulting to 1. Higher = corroborated across more sources."""
    for t in tags or []:
        if t.startswith(_FACT_CONFIDENCE_PREFIX):
            try:
                return max(1, int(t[len(_FACT_CONFIDENCE_PREFIX) :]))
            except ValueError:
                return 1
    return 1


def _parse_rfc3339(value: str) -> datetime:
    """Parse an RFC 3339 timestamp string to an aware UTC ``datetime``.

    This is the **only** place the Memory layer parses a timestamp (the raw
    models keep them as strings, contract §7). Normalizes a trailing ``Z`` to
    ``+00:00`` for runtimes whose ``fromisoformat`` predates 3.11, and treats a
    naive (no-offset) timestamp as UTC.
    """
    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _recency_score(
    created_at: Optional[str],
    now: datetime,
    half_life_days: float,
) -> float:
    """Exponential half-life recency decay in ``[0, 1]`` (contract §4).

    ``None``/unparseable timestamps score ``0.0``; future timestamps clamp to
    age ``0`` → score ``1.0``.
    """
    if not created_at:
        return 0.0
    try:
        created = _parse_rfc3339(created_at)
    except (ValueError, TypeError):
        return 0.0
    age_seconds = (now - created).total_seconds()
    age_days = max(0.0, age_seconds / 86400.0)
    return 0.5 ** (age_days / half_life_days)


def _similarity(score: float) -> float:
    """Normalize a 0–100 relevance ``score`` (higher = better) to ``[0, 1]``.

    Search hits carry a calibrated ``score`` (0–100) since the 0.3.0 redesign;
    dividing by 100 puts it on the same ``[0, 1]`` scale as the recency term so
    the §4 Mode B blend stays well-defined.
    """
    return score / 100.0


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


class _MemoryBase:
    """Shared validation/encoding helpers for the sync and async facades."""

    def __init__(
        self,
        entity_id: str,
        *,
        half_life_days: float,
        extract_facts: bool,
        now: Callable[[], datetime],
    ) -> None:
        self.entity_id = _validate_entity_id(entity_id)
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.half_life_days = half_life_days
        # KEEP        # extraction (contract §3): when true, every ``remember`` on
        # this instance requests ``insert_text(..., extract_facts=True)`` unless
        # the call passes an explicit ``extract=``. Requires fact extraction to
        # be configured on the node. The SDKs carry no LLM dependency — they
        # only pass the flag.
        self.extract_facts = extract_facts
        # Injectable clock — defaults to real UTC now; overridden in tests so the
        # recency-decay ordering is deterministic (contract §4).
        self.now: Callable[[], datetime] = now

    def _blend_and_rank(
        self,
        candidates: list,
        created_at_of: dict[str, Optional[str]],
        recency_weight: float,
        k: int,
    ) -> list[MemoryItem]:
        """Apply the §4 Mode B blended ranking and return the top ``k`` items.

        ``candidates`` are ``RetrievalResult`` objects (``doc_id``, ``score``,
        ``content``). ``created_at_of`` maps ``doc_id`` → its (possibly null)
        ``created_at`` string. Deterministic total order:
        ``(blended DESC, score DESC, doc_id ASC)``.
        """
        now = self.now()
        scored = []
        for c in candidates:
            similarity = _similarity(c.score)
            created = created_at_of.get(c.doc_id)
            recency = _recency_score(created, now, self.half_life_days)
            blended = (1.0 - recency_weight) * similarity + recency_weight * recency
            scored.append((blended, c, created))
        scored.sort(key=lambda t: (-t[0], -t[1].score, t[1].doc_id))
        return [
            MemoryItem(
                id=c.doc_id,
                text=c.content,
                created_at=created,
                entity_id=self.entity_id,
                metadata=c.metadata,
                score=blended,
            )
            for blended, c, created in scored[:k]
        ]


class Memory(_MemoryBase):
    """Entity-scoped, synchronous memory facade over :class:`AetherClient`.

    Construct once with the entity to scope to::

        mem = Memory("patient-john", api_key="ak_...")
        mem.remember("Anxious about flying; uses 4-7-8 breathing")
        hits = mem.recall("anxiety coping")

    Two construction paths (MEMORY_CONTRACT.md §1):

    1. **Convenience** — pass connection options and ``Memory`` builds its own
       :class:`AetherClient`.
    2. **Dependency injection** — pass an already-configured ``client=`` (used by
       tests and apps sharing one client across many entities).

    ``entity_id`` is fixed at construction and validated (non-empty, ≤ 256
    chars). All error/retry/timeout semantics come from the raw client unchanged.
    """

    def __init__(
        self,
        entity_id: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
        half_life_days: float = 30.0,
        extract_facts: bool = False,
        client: Optional[AetherClient] = None,
        now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        super().__init__(
            entity_id,
            half_life_days=half_life_days,
            extract_facts=extract_facts,
            now=now or _default_now,
        )
        if client is not None:
            self.client = client
            self._owns_client = False
        else:
            self.client = AetherClient(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )
            self._owns_client = True

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the owned raw client. No-op for an injected client."""
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Operations ────────────────────────────────────────────────────

    def remember(
        self,
        text: str,
        metadata: Optional[Metadata] = None,
        extract: Optional[bool] = None,
    ) -> MemoryItem:
        """Store one memory for the entity (**one HTTP call**).

        ``metadata`` is sent as structured typed document metadata. For older
        tag-based callers, string-safe metadata is also mirrored into
        ``key:value`` tags where doing so is lossless.

        Pass ``extract=True`` to also distill the text into atomic facts
        server-side; each fact is stored as a sibling memory tagged
        ``kind:fact`` and is recallable like any other. When ``extract`` is
        omitted (``None``), the constructor's ``extract_facts`` flag decides;
        an explicit ``True``/``False`` overrides it for this call. The returned
        :class:`MemoryItem` is the raw stored memory (not the facts); list the
        facts with ``kind="fact"`` on the underlying client. Requires fact
        extraction to be configured on the node.
        """
        text = _normalize_text(text)
        tags = _encode_legacy_metadata_tags(metadata)
        record = self.client.insert_text(
            text,
            entity_id=self.entity_id,
            tags=tags,
            metadata=metadata,
            extract_facts=self.extract_facts if extract is None else extract,
        )
        return MemoryItem(
            id=record.doc_id,
            text=text,
            created_at=record.created_at,
            entity_id=record.entity_id or self.entity_id,
            metadata=record.metadata or metadata or {},
            score=None,
        )

    def recall(
        self,
        query: str,
        k: int = 5,
        recency_weight: float = 0.0,
        since: Optional[str] = None,
        until: Optional[str] = None,
        filter: Optional[MetadataFilter] = None,
    ) -> list[MemoryItem]:
        """Semantic search scoped to the entity, with optional recency decay.

        ``recency_weight`` is clamped to ``[0, 1]``. When ``0`` (default) this
        is a single ``retrieve`` call and ``created_at`` is ``None`` on every
        item. When ``> 0`` it overfetches, resolves each candidate's
        ``created_at`` via ``get``, and re-ranks by the §4 Mode B blend (N+1
        calls).
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        recency_weight = _clamp_recency_weight(recency_weight)

        if recency_weight == 0.0:
            hits = self.client.retrieve(
                query,
                k=k,
                entity_id=self.entity_id,
                since=since,
                until=until,
                filter=filter,
            )
            return [
                MemoryItem(
                    id=h.doc_id,
                    text=h.content,
                    created_at=None,
                    entity_id=self.entity_id,
                    metadata=h.metadata,
                    score=_similarity(h.score),
                )
                for h in hits
            ]

        candidates = self.client.retrieve(
            query,
            k=min(k * OVERFETCH, MAX_CANDIDATES),
            entity_id=self.entity_id,
            since=since,
            until=until,
            filter=filter,
        )
        if not candidates:
            return []

        # Resolve created_at per unique doc_id (sequential for the sync facade).
        created_at_of: dict[str, Optional[str]] = {}
        for c in candidates:
            if c.doc_id not in created_at_of:
                created_at_of[c.doc_id] = self.client.get(c.doc_id).created_at

        return self._blend_and_rank(candidates, created_at_of, recency_weight, k)

    def list(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        filter: Optional[MetadataFilter] = None,
        limit: int = 50,
    ) -> list[MemoryItem]:
        """Chronological view of the entity's memories, **newest first**.

        Cost note: this is **1 + N** calls (one listing plus one content
        download per item). Callers who only need metadata can drop to the raw
        ``client.list(entity_id=...)``.
        """
        records = self.client.list(
            entity_id=self.entity_id,
            since=since,
            until=until,
            filter=filter,
            limit=limit,
        )
        items: list[MemoryItem] = []
        for r in records[:limit]:
            text = self.client.download_text(r.doc_id)
            items.append(
                MemoryItem(
                    id=r.doc_id,
                    text=text,
                    created_at=r.created_at,
                    entity_id=r.entity_id or self.entity_id,
                    metadata=r.metadata,
                    score=None,
                )
            )
        return items

    def list_extracted_facts(self, *, limit: int = 50) -> list[MemoryItem]:
        """Return this entity's consolidated **extracted** facts (``kind:fact``
        memories), highest corroborated confidence first.

        These are the free-text facts produced by ``remember(..., extract=True)``
        and deduped server-side — distinct from the structured memory-graph facts
        returned by :meth:`list_facts`. Ordered most-corroborated first (ties
        broken by recency). Cost is 1 + N (one listing plus a content download per
        fact)."""
        records = self.client.list(
            entity_id=self.entity_id,
            tags=[_FACT_KIND_TAG],
            limit=limit,
        )
        records.sort(
            key=lambda r: (_fact_confidence(r.tags), r.created_at or ""),
            reverse=True,
        )
        items: list[MemoryItem] = []
        for r in records[:limit]:
            text = self.client.download_text(r.doc_id)
            items.append(
                MemoryItem(
                    id=r.doc_id,
                    text=text,
                    created_at=r.created_at,
                    entity_id=r.entity_id or self.entity_id,
                    metadata=r.metadata,
                    score=None,
                )
            )
        return items

    def forget(self, memory_id: str) -> None:
        """Delete one memory by id (soft tombstone). Empty id is a ``ValueError``."""
        if not memory_id:
            raise ValueError("memory_id cannot be empty")
        self.client.delete(memory_id)

    def forget_all(self) -> int:
        """Delete **every** memory for this entity; return the count deleted.

        Pages the entity's listing and deletes each ``doc_id`` until the listing
        is exhausted (deletes are tombstones, so re-listing excludes them).
        """
        deleted = 0
        while True:
            records = self.client.list(entity_id=self.entity_id, limit=1000)
            if not records:
                break
            for r in records:
                self.client.delete(r.doc_id)
                deleted += 1
        return deleted

    # ── Memory graph (Part II) ────────────────────────────────────────

    def _graph_request(self, method, path, params, body):
        """Execute a /v1/memory/* graph request scoped to this Memory.

        Injects ``entity_id`` (the owner) and the owned client's partition, then
        reuses the raw client's retry/error transport unchanged.
        """
        p = dict(params)
        p["entity_id"] = self.entity_id
        partition = getattr(self.client, "_partition", None)
        if partition:
            p["partition"] = partition
        if body is None:
            resp = self.client._request_with_retry(method, path, params=p)
        else:
            resp = self.client._request_with_retry(method, path, params=p, json=body)
        self.client._raise_for_status(resp)
        return resp.json()

    def upsert_entity(
        self,
        entity_type: str,
        *,
        memory_entity_id: Optional[str] = None,
        display_name: Optional[str] = None,
        aliases: Optional[list[str]] = None,
        attributes: Optional[dict[str, ScalarValue]] = None,
    ) -> MemoryEntity:
        """Create or update a typed entity node in this owner's graph.

        Omit ``memory_entity_id`` to mint a new node; pass an existing one (or an
        idempotency key) to update it. ``attributes`` values must be scalar.
        """
        method, path, params, body, parse = _spec_upsert_entity(
            entity_type, memory_entity_id, display_name, aliases, attributes
        )
        return parse(self._graph_request(method, path, params, body))

    def get_entity(self, memory_entity_id: str) -> MemoryEntity:
        """Fetch one entity node by id."""
        method, path, params, body, parse = _spec_get_entity(memory_entity_id)
        return parse(self._graph_request(method, path, params, body))

    def list_entities(
        self,
        *,
        entity_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[MemoryEntity]:
        """List this owner's entity nodes, optionally filtered by ``entity_type``."""
        method, path, params, body, parse = _spec_list_entities(entity_type, limit)
        return parse(self._graph_request(method, path, params, body))

    def relate(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        *,
        relationship_id: Optional[str] = None,
        attributes: Optional[dict[str, ScalarValue]] = None,
        valid_from: Optional[str] = None,
    ) -> MemoryRelationship:
        """Create or update a directed edge between two entity nodes."""
        method, path, params, body, parse = _spec_relate(
            from_entity_id, to_entity_id, relationship_type,
            relationship_id, attributes, valid_from,
        )
        return parse(self._graph_request(method, path, params, body))

    def list_relationships(
        self,
        *,
        from_entity_id: Optional[str] = None,
        to_entity_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        include_inactive: bool = False,
        as_of: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[MemoryRelationship]:
        """List edges, optionally filtered. ``as_of`` returns edges active at that instant."""
        method, path, params, body, parse = _spec_list_relationships(
            from_entity_id, to_entity_id, relationship_type, include_inactive, as_of, limit
        )
        return parse(self._graph_request(method, path, params, body))

    def remember_fact(
        self,
        predicate: str,
        value: ScalarValue,
        *,
        subject_type: str = "owner",
        subject_id: Optional[str] = None,
        cardinality: Optional[str] = None,
        valid_from: Optional[str] = None,
        observed_at: Optional[str] = None,
        supersedes_fact_id: Optional[str] = None,
    ) -> MemoryFact:
        """Assert a temporal fact about the owner (default), an entity, or a relationship.

        A newer single-valued fact with the same (subject, predicate) supersedes
        the prior one server-side, keeping it in history (ADR-019). ``value`` must
        be scalar.
        """
        method, path, params, body, parse = _spec_remember_fact(
            predicate, value, subject_type, subject_id, cardinality,
            valid_from, observed_at, supersedes_fact_id,
        )
        return parse(self._graph_request(method, path, params, body))

    def list_facts(
        self,
        *,
        subject_type: Optional[str] = None,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None,
        include_inactive: bool = False,
        as_of: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[MemoryFact]:
        """List active facts (default), or include superseded/retracted with ``include_inactive``."""
        method, path, params, body, parse = _spec_list_facts(
            subject_type, subject_id, predicate, include_inactive, as_of, limit
        )
        return parse(self._graph_request(method, path, params, body))

    def fact_history(
        self,
        predicate: str,
        *,
        subject_type: str = "owner",
        subject_id: Optional[str] = None,
    ) -> list[MemoryFact]:
        """Full assertion chain (active + superseded) for one (subject, predicate)."""
        method, path, params, body, parse = _spec_fact_history(
            predicate, subject_type, subject_id
        )
        return parse(self._graph_request(method, path, params, body))

    def consolidate(self) -> ConsolidationReport:
        """Soft-retract redundant facts in this scope; returns a report."""
        method, path, params, body, parse = _spec_consolidate()
        return parse(self._graph_request(method, path, params, body))


class AsyncMemory(_MemoryBase):
    """Entity-scoped, asynchronous memory facade over :class:`AsyncAetherClient`.

    Identical surface to :class:`Memory`, with ``await`` on every operation.
    The N+1 paths (``recall`` recency mode, ``list``) parallelize the per-item
    ``get``/``download`` calls with :func:`asyncio.gather`.
    """

    def __init__(
        self,
        entity_id: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
        half_life_days: float = 30.0,
        extract_facts: bool = False,
        client: Optional[AsyncAetherClient] = None,
        now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        super().__init__(
            entity_id,
            half_life_days=half_life_days,
            extract_facts=extract_facts,
            now=now or _default_now,
        )
        if client is not None:
            self.client = client
            self._owns_client = False
        else:
            self.client = AsyncAetherClient(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )
            self._owns_client = True

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the owned raw client. No-op for an injected client."""
        if self._owns_client:
            await self.client.close()

    async def __aenter__(self) -> "AsyncMemory":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    # ── Operations ────────────────────────────────────────────────────

    async def remember(
        self,
        text: str,
        metadata: Optional[Metadata] = None,
        extract: Optional[bool] = None,
    ) -> MemoryItem:
        """Store one memory for the entity (**one HTTP call**). See
        :meth:`Memory.remember`. Pass ``extract=True`` to also distill atomic
        facts server-side; when omitted, the constructor's
        ``extract_facts`` flag decides."""
        text = _normalize_text(text)
        tags = _encode_legacy_metadata_tags(metadata)
        record = await self.client.insert_text(
            text,
            entity_id=self.entity_id,
            tags=tags,
            metadata=metadata,
            extract_facts=self.extract_facts if extract is None else extract,
        )
        return MemoryItem(
            id=record.doc_id,
            text=text,
            created_at=record.created_at,
            entity_id=record.entity_id or self.entity_id,
            metadata=record.metadata or metadata or {},
            score=None,
        )

    async def recall(
        self,
        query: str,
        k: int = 5,
        recency_weight: float = 0.0,
        since: Optional[str] = None,
        until: Optional[str] = None,
        filter: Optional[MetadataFilter] = None,
    ) -> list[MemoryItem]:
        """Semantic search scoped to the entity. See :meth:`Memory.recall`.

        The recency mode resolves each candidate's ``created_at`` concurrently
        with :func:`asyncio.gather`.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        recency_weight = _clamp_recency_weight(recency_weight)

        if recency_weight == 0.0:
            hits = await self.client.retrieve(
                query,
                k=k,
                entity_id=self.entity_id,
                since=since,
                until=until,
                filter=filter,
            )
            return [
                MemoryItem(
                    id=h.doc_id,
                    text=h.content,
                    created_at=None,
                    entity_id=self.entity_id,
                    metadata=h.metadata,
                    score=_similarity(h.score),
                )
                for h in hits
            ]

        candidates = await self.client.retrieve(
            query,
            k=min(k * OVERFETCH, MAX_CANDIDATES),
            entity_id=self.entity_id,
            since=since,
            until=until,
            filter=filter,
        )
        if not candidates:
            return []

        unique_ids = list(dict.fromkeys(c.doc_id for c in candidates))
        records = await asyncio.gather(
            *(self.client.get(doc_id) for doc_id in unique_ids)
        )
        created_at_of: dict[str, Optional[str]] = {
            doc_id: record.created_at for doc_id, record in zip(unique_ids, records)
        }
        return self._blend_and_rank(candidates, created_at_of, recency_weight, k)

    async def list(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        filter: Optional[MetadataFilter] = None,
        limit: int = 50,
    ) -> list[MemoryItem]:
        """Chronological view, newest first. See :meth:`Memory.list`.

        Per-item content downloads run concurrently with :func:`asyncio.gather`.
        """
        records = await self.client.list(
            entity_id=self.entity_id,
            since=since,
            until=until,
            filter=filter,
            limit=limit,
        )
        records = records[:limit]
        if not records:
            return []
        texts = await asyncio.gather(
            *(self.client.download_text(r.doc_id) for r in records)
        )
        return [
            MemoryItem(
                id=r.doc_id,
                text=text,
                created_at=r.created_at,
                entity_id=r.entity_id or self.entity_id,
                metadata=r.metadata,
                score=None,
            )
            for r, text in zip(records, texts)
        ]

    async def list_extracted_facts(self, *, limit: int = 50) -> list[MemoryItem]:
        """Entity's consolidated extracted facts, highest-confidence first. See
        :meth:`Memory.list_extracted_facts`. Content downloads run concurrently."""
        records = await self.client.list(
            entity_id=self.entity_id,
            tags=[_FACT_KIND_TAG],
            limit=limit,
        )
        records.sort(
            key=lambda r: (_fact_confidence(r.tags), r.created_at or ""),
            reverse=True,
        )
        records = records[:limit]
        if not records:
            return []
        texts = await asyncio.gather(
            *(self.client.download_text(r.doc_id) for r in records)
        )
        return [
            MemoryItem(
                id=r.doc_id,
                text=text,
                created_at=r.created_at,
                entity_id=r.entity_id or self.entity_id,
                metadata=r.metadata,
                score=None,
            )
            for r, text in zip(records, texts)
        ]

    async def forget(self, memory_id: str) -> None:
        """Delete one memory by id. See :meth:`Memory.forget`."""
        if not memory_id:
            raise ValueError("memory_id cannot be empty")
        await self.client.delete(memory_id)

    async def forget_all(self) -> int:
        """Delete every memory for this entity; return the count. See
        :meth:`Memory.forget_all`."""
        deleted = 0
        while True:
            records = await self.client.list(entity_id=self.entity_id, limit=1000)
            if not records:
                break
            await asyncio.gather(*(self.client.delete(r.doc_id) for r in records))
            deleted += len(records)
        return deleted

    # ── Memory graph (Part II) ────────────────────────────────────────

    async def _graph_request(self, method, path, params, body):
        """Async twin of :meth:`Memory._graph_request`."""
        p = dict(params)
        p["entity_id"] = self.entity_id
        partition = getattr(self.client, "_partition", None)
        if partition:
            p["partition"] = partition
        if body is None:
            resp = await self.client._request_with_retry(method, path, params=p)
        else:
            resp = await self.client._request_with_retry(method, path, params=p, json=body)
        self.client._raise_for_status(resp)
        return resp.json()

    async def upsert_entity(
        self,
        entity_type: str,
        *,
        memory_entity_id: Optional[str] = None,
        display_name: Optional[str] = None,
        aliases: Optional[list[str]] = None,
        attributes: Optional[dict[str, ScalarValue]] = None,
    ) -> MemoryEntity:
        """See :meth:`Memory.upsert_entity`."""
        method, path, params, body, parse = _spec_upsert_entity(
            entity_type, memory_entity_id, display_name, aliases, attributes
        )
        return parse(await self._graph_request(method, path, params, body))

    async def get_entity(self, memory_entity_id: str) -> MemoryEntity:
        """See :meth:`Memory.get_entity`."""
        method, path, params, body, parse = _spec_get_entity(memory_entity_id)
        return parse(await self._graph_request(method, path, params, body))

    async def list_entities(
        self,
        *,
        entity_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[MemoryEntity]:
        """See :meth:`Memory.list_entities`."""
        method, path, params, body, parse = _spec_list_entities(entity_type, limit)
        return parse(await self._graph_request(method, path, params, body))

    async def relate(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        *,
        relationship_id: Optional[str] = None,
        attributes: Optional[dict[str, ScalarValue]] = None,
        valid_from: Optional[str] = None,
    ) -> MemoryRelationship:
        """See :meth:`Memory.relate`."""
        method, path, params, body, parse = _spec_relate(
            from_entity_id, to_entity_id, relationship_type,
            relationship_id, attributes, valid_from,
        )
        return parse(await self._graph_request(method, path, params, body))

    async def list_relationships(
        self,
        *,
        from_entity_id: Optional[str] = None,
        to_entity_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        include_inactive: bool = False,
        as_of: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[MemoryRelationship]:
        """See :meth:`Memory.list_relationships`."""
        method, path, params, body, parse = _spec_list_relationships(
            from_entity_id, to_entity_id, relationship_type, include_inactive, as_of, limit
        )
        return parse(await self._graph_request(method, path, params, body))

    async def remember_fact(
        self,
        predicate: str,
        value: ScalarValue,
        *,
        subject_type: str = "owner",
        subject_id: Optional[str] = None,
        cardinality: Optional[str] = None,
        valid_from: Optional[str] = None,
        observed_at: Optional[str] = None,
        supersedes_fact_id: Optional[str] = None,
    ) -> MemoryFact:
        """See :meth:`Memory.remember_fact`."""
        method, path, params, body, parse = _spec_remember_fact(
            predicate, value, subject_type, subject_id, cardinality,
            valid_from, observed_at, supersedes_fact_id,
        )
        return parse(await self._graph_request(method, path, params, body))

    async def list_facts(
        self,
        *,
        subject_type: Optional[str] = None,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None,
        include_inactive: bool = False,
        as_of: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[MemoryFact]:
        """See :meth:`Memory.list_facts`."""
        method, path, params, body, parse = _spec_list_facts(
            subject_type, subject_id, predicate, include_inactive, as_of, limit
        )
        return parse(await self._graph_request(method, path, params, body))

    async def fact_history(
        self,
        predicate: str,
        *,
        subject_type: str = "owner",
        subject_id: Optional[str] = None,
    ) -> list[MemoryFact]:
        """See :meth:`Memory.fact_history`."""
        method, path, params, body, parse = _spec_fact_history(
            predicate, subject_type, subject_id
        )
        return parse(await self._graph_request(method, path, params, body))

    async def consolidate(self) -> ConsolidationReport:
        """See :meth:`Memory.consolidate`."""
        method, path, params, body, parse = _spec_consolidate()
        return parse(await self._graph_request(method, path, params, body))
