"""Access-audit facade over the Aether client (read-ACL layer).

``AuditClient`` (accessed via ``client.audit``) queries the tenant's
access-audit log — the operational record of document reads, search
deliveries, denials, and admin bypasses written when access-audit capture is
enabled for the tenant. It adds no new transport behavior — retry, error, and
timeout handling are inherited from the raw client. ``AsyncAuditClient`` is
the ``async``/``await`` equivalent for :class:`~aether.AsyncAetherClient`.

Access records share the audit envelope with the signed provenance trail
returned by ``client.lineage(...)``: the same :class:`~aether.AuditRecord`
shape with ``source == "access"`` and no cryptographic proof.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .models import AccessAuditPage, AuditProof, AuditRecord

if TYPE_CHECKING:
    from .async_client import AsyncAetherClient
    from .client import AetherClient


def _to_audit_record(d: dict) -> AuditRecord:
    """Map an audit-record JSON object to an :class:`AuditRecord`.

    Access records never carry a ``proof``; it is parsed when present so the
    converter stays byte-compatible with the shared envelope.
    """
    proof_raw = d.get("proof")
    proof = None
    if proof_raw is not None:
        proof = AuditProof(
            content_id=proof_raw.get("content_id"),
            lamport=proof_raw.get("lamport", 0),
            node_id=proof_raw.get("node_id", ""),
            public_key=proof_raw.get("public_key", ""),
            signature=proof_raw.get("signature", ""),
            verified=proof_raw.get("verified", False),
        )
    return AuditRecord(
        at=d["at"],
        actor=d["actor"],
        action=d["action"],
        resource=d["resource"],
        outcome=d["outcome"],
        source=d["source"],
        proof=proof,
    )


def _access_params(
    actor: Optional[str],
    resource: Optional[str],
    action: Optional[str],
    since: Optional[str],
    until: Optional[str],
    limit: Optional[int],
    offset: Optional[int],
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if actor:
        params["actor"] = actor
    if resource:
        params["resource"] = resource
    if action:
        params["action"] = action
    if since:
        params["since"] = since
    if until:
        params["until"] = until
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset
    return params


class AuditClient:
    """Query the tenant's access-audit log. Access via ``client.audit``."""

    def __init__(self, client: "AetherClient"):
        self._c = client

    def access(
        self,
        *,
        actor: str | None = None,
        resource: str | None = None,
        action: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> AccessAuditPage:
        """Query the tenant's access-audit events, newest first.

        Requires access-audit capture to be enabled for your tenant (an
        operator setting); a tenant that has not opted in always returns an
        empty page. All filters are optional and compose with AND:

        Args:
            actor: Only events by this actor — an asserted acting principal
                (e.g. ``user:alice``), or ``key:<prefix>`` for requests that
                asserted none.
            resource: Only events on this resource (a document id, or a query
                id for ``search_hit`` events).
            action: Only this action: ``read`` | ``search_hit`` | ``denied``
                | ``admin_bypass``.
            since: Inclusive lower time bound (RFC 3339).
            until: Inclusive upper time bound (RFC 3339).
            limit: Page size (server default 100, max 1000).
            offset: Page offset.

        Returns:
            An :class:`AccessAuditPage` (a ``list`` of :class:`AuditRecord`
            with ``source == "access"`` and no ``proof``), carrying ``.total``
            — the number of events matching the filter across all pages.
        """
        resp = self._c._request_with_retry(
            "GET",
            "/audit/access",
            params=_access_params(actor, resource, action, since, until, limit, offset),
        )
        self._c._raise_for_status(resp)
        body = resp.json()
        records = [_to_audit_record(r) for r in body.get("records", [])]
        return AccessAuditPage(records, total=body.get("total", len(records)))


class AsyncAuditClient:
    """Async equivalent of :class:`AuditClient`, accessed via ``client.audit``."""

    def __init__(self, client: "AsyncAetherClient"):
        self._c = client

    async def access(
        self,
        *,
        actor: str | None = None,
        resource: str | None = None,
        action: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> AccessAuditPage:
        """Query the tenant's access-audit events, newest first. See
        :meth:`AuditClient.access`."""
        resp = await self._c._request_with_retry(
            "GET",
            "/audit/access",
            params=_access_params(actor, resource, action, since, until, limit, offset),
        )
        self._c._raise_for_status(resp)
        body = resp.json()
        records = [_to_audit_record(r) for r in body.get("records", [])]
        return AccessAuditPage(records, total=body.get("total", len(records)))
