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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from .async_client import AsyncAetherClient
from .client import AetherClient

# Algorithm constants — see MEMORY_CONTRACT.md §4 Mode B. Identical across all
# four SDKs so the contract test produces the same ordering everywhere.
OVERFETCH = 4
MAX_CANDIDATES = 100

# entity_id constraint mirrors the server: 1–256 chars.
_MAX_ENTITY_ID_LEN = 256

# Metadata values are strings only in v1 (MEMORY_CONTRACT.md §3): numeric/float
# formatting is not portable across the four languages and metadata is write-only
# anyway, so a typed-value convention is deferred to a later version.
MetadataValue = str


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

    .. note::
        ``metadata`` is intentionally **not** a field here. The raw document API
        does not echo ``tags`` on any read model, so ``remember(text, metadata)``
        *writes* metadata as searchable tags but it cannot be read back in v1.
        When the server starts echoing tags, a ``metadata`` field can be added
        without a breaking change.
    """

    id: str
    text: str
    created_at: Optional[str] = None
    entity_id: Optional[str] = None
    score: Optional[float] = None


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


def _encode_metadata(metadata: Optional[dict[str, str]]) -> Optional[list[str]]:
    """Encode a ``{key: value}`` (string→string) metadata map as ``key:value`` tags.

    Values are strings only in v1 (MEMORY_CONTRACT.md §3). Each pair becomes one
    ``key:value`` tag, split on the **first** ``:``. Client-side argument errors
    (raised before any HTTP call): an empty key; a key containing ``:`` or ``,``;
    a value containing ``,`` (the tag wire format is comma-joined and cannot
    escape commas). Tags are emitted **sorted by key ascending** so the wire
    string is byte-identical across all four languages for the same map.
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
        if "," in value:
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
        # ``extract_facts`` is a reserved no-op in v1 (contract §3): there is no
        # server-side fact-extraction endpoint and these SDKs carry no LLM
        # dependency, so ``True`` currently behaves identically to ``False``.
        # The flag keeps the public signature stable for a future extractor.
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
        metadata: Optional[dict[str, MetadataValue]] = None,
    ) -> MemoryItem:
        """Store one memory for the entity (**one HTTP call**).

        ``metadata`` is encoded as ``key:value`` tags (write-only in v1 — it
        cannot be read back, see :class:`MemoryItem`). A value containing a
        comma raises ``ValueError`` before any HTTP call.
        """
        text = _normalize_text(text)
        tags = _encode_metadata(metadata)
        record = self.client.insert_text(
            text,
            entity_id=self.entity_id,
            tags=tags,
        )
        return MemoryItem(
            id=record.doc_id,
            text=text,
            created_at=record.created_at,
            entity_id=record.entity_id or self.entity_id,
            score=None,
        )

    def recall(
        self,
        query: str,
        k: int = 5,
        recency_weight: float = 0.0,
        since: Optional[str] = None,
        until: Optional[str] = None,
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
            )
            return [
                MemoryItem(
                    id=h.doc_id,
                    text=h.content,
                    created_at=None,
                    entity_id=self.entity_id,
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
        metadata: Optional[dict[str, MetadataValue]] = None,
    ) -> MemoryItem:
        """Store one memory for the entity (**one HTTP call**). See
        :meth:`Memory.remember`."""
        text = _normalize_text(text)
        tags = _encode_metadata(metadata)
        record = await self.client.insert_text(
            text,
            entity_id=self.entity_id,
            tags=tags,
        )
        return MemoryItem(
            id=record.doc_id,
            text=text,
            created_at=record.created_at,
            entity_id=record.entity_id or self.entity_id,
            score=None,
        )

    async def recall(
        self,
        query: str,
        k: int = 5,
        recency_weight: float = 0.0,
        since: Optional[str] = None,
        until: Optional[str] = None,
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
            )
            return [
                MemoryItem(
                    id=h.doc_id,
                    text=h.content,
                    created_at=None,
                    entity_id=self.entity_id,
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
        limit: int = 50,
    ) -> list[MemoryItem]:
        """Chronological view, newest first. See :meth:`Memory.list`.

        Per-item content downloads run concurrently with :func:`asyncio.gather`.
        """
        records = await self.client.list(
            entity_id=self.entity_id,
            since=since,
            until=until,
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
