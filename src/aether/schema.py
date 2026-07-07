"""Field-schema facade over the Aether client (structured-query layer).

``SchemaClient`` (accessed via ``client.schema``) declares and manages the typed
fields that :meth:`~aether.AetherClient.query` filters, sorts, and aggregates
over. It adds no new transport behavior — retry, error, timeout, and partition
scoping are inherited from the raw client. ``AsyncSchemaClient`` is the
``async``/``await`` equivalent for :class:`~aether.AsyncAetherClient`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import quote

from .models import FieldSchema

if TYPE_CHECKING:
    from .async_client import AsyncAetherClient
    from .client import AetherClient


def _to_field_schema(d: dict) -> FieldSchema:
    return FieldSchema(
        name=d["name"],
        type=d["type"],
        source=d.get("source", {}),
        partition_scope=d.get("partition_scope"),
        coverage=d.get("coverage", 0),
        mismatch_count=d.get("mismatch_count", 0),
        backfill=d.get("backfill", "complete"),
    )


def _partition_params(partition: Optional[str]) -> dict[str, Any]:
    return {"partition": partition} if partition else {}


class SchemaClient:
    """Declare and manage typed fields for the structured-query layer.

    Access via ``client.schema``. On a partition-scoped handle
    (``client.partition("x").schema``) every call is pinned to that partition,
    exactly like the rest of the client.
    """

    def __init__(self, client: "AetherClient"):
        self._c = client

    def declare_fields(self, fields: list[dict]) -> list[FieldSchema]:
        """Declare or replace typed fields, then return the declared set.

        Each entry is
        ``{"name", "type", "source": {"metadata"|"regex": …}, "partition_scope"?}``.
        Re-declaring an existing name replaces its type/source and re-backfills;
        names absent from ``fields`` are left untouched.
        """
        resp = self._c._request_with_retry(
            "PUT",
            "/schema/fields",
            params=_partition_params(self._c._partition),
            json={"fields": fields},
        )
        self._c._raise_for_status(resp)
        return [_to_field_schema(f) for f in resp.json().get("fields", [])]

    def list_fields(self) -> list[FieldSchema]:
        """List the declared fields visible to this handle (name-sorted)."""
        resp = self._c._request_with_retry(
            "GET", "/schema/fields", params=_partition_params(self._c._partition)
        )
        self._c._raise_for_status(resp)
        return [_to_field_schema(f) for f in resp.json().get("fields", [])]

    def delete_field(self, name: str) -> list[FieldSchema]:
        """Remove a declared field; return the remaining fields."""
        resp = self._c._request_with_retry(
            "DELETE",
            f"/schema/fields/{quote(name)}",
            params=_partition_params(self._c._partition),
        )
        self._c._raise_for_status(resp)
        return [_to_field_schema(f) for f in resp.json().get("fields", [])]


class AsyncSchemaClient:
    """Async equivalent of :class:`SchemaClient`, accessed via ``client.schema``."""

    def __init__(self, client: "AsyncAetherClient"):
        self._c = client

    async def declare_fields(self, fields: list[dict]) -> list[FieldSchema]:
        """Declare or replace typed fields, then return the declared set. See
        :meth:`SchemaClient.declare_fields`."""
        resp = await self._c._request_with_retry(
            "PUT",
            "/schema/fields",
            params=_partition_params(self._c._partition),
            json={"fields": fields},
        )
        self._c._raise_for_status(resp)
        return [_to_field_schema(f) for f in resp.json().get("fields", [])]

    async def list_fields(self) -> list[FieldSchema]:
        """List the declared fields visible to this handle (name-sorted)."""
        resp = await self._c._request_with_retry(
            "GET", "/schema/fields", params=_partition_params(self._c._partition)
        )
        self._c._raise_for_status(resp)
        return [_to_field_schema(f) for f in resp.json().get("fields", [])]

    async def delete_field(self, name: str) -> list[FieldSchema]:
        """Remove a declared field; return the remaining fields."""
        resp = await self._c._request_with_retry(
            "DELETE",
            f"/schema/fields/{quote(name)}",
            params=_partition_params(self._c._partition),
        )
        self._c._raise_for_status(resp)
        return [_to_field_schema(f) for f in resp.json().get("fields", [])]
