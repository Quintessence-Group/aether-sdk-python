"""Aether SDK exception types."""

from __future__ import annotations

from typing import Any, Optional


class AetherError(Exception):
    """Base exception for all Aether SDK errors."""


class AetherApiError(AetherError):
    """Raised when the Aether API returns an error response (4xx/5xx).

    Attributes:
        status_code: HTTP status code from the API.
        error_code: Machine-readable error code (e.g., "invalid_api_key").
        message: Human-readable error description.
        request_id: Correlation ID for debugging (from x-request-id header).
        body: Raw parsed JSON response body.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None,
        body: Optional[dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.request_id = request_id
        self.body = body or {}
        super().__init__(f"[{status_code}] {message}")

    @property
    def is_retryable(self) -> bool:
        """Whether this error is likely transient and worth retrying."""
        return self.status_code in (429, 502, 503, 504)


class CreditExhaustedError(AetherApiError):
    """Raised when a paid tenant's prepaid credit balance is exhausted
    (HTTP 402, ``code="credit_exhausted"``). Top up via the Portal billing
    page; the SDK never retries — the operation is permanently denied
    until credit is added.
    """


class FreeLimitExceededError(AetherApiError):
    """Raised when a Free-tier tenant exceeds a hard plan limit (HTTP 402,
    ``code="free_limit_exceeded"``). Distinct from
    :class:`CreditExhaustedError` so dashboards can separate abuse signal
    from billing failures. Resolution is a plan upgrade, not a top-up.
    """


class TenantPausedError(AetherApiError):
    """Raised when an operator has paused a tenant via the spike detector
    or admin console (HTTP 403, ``code="tenant_paused"``). Not retryable;
    the tenant must be un-paused out-of-band.
    """


class PartitionRequiredError(AetherApiError):
    """Raised when a multi-tenant key makes an unscoped call (HTTP 400,
    ``code="partition_required"``). The key requires every read/write to name
    a partition; scope the call through a partition handle —
    ``client.partition("<end-client-id>")`` — instead of the top-level client.
    Not retryable: it is a programming error, not a transient failure.
    """


def aether_api_error_from_response(
    status_code: int,
    message: str,
    *,
    error_code: Optional[str] = None,
    request_id: Optional[str] = None,
    body: Optional[dict[str, Any]] = None,
) -> AetherApiError:
    """Build the most-specific :class:`AetherApiError` subclass for a server
    response. Inspects the structured ``code`` field (Phase 8 / ADR-015 wire
    shape); unknown codes fall back to the base :class:`AetherApiError`.
    """
    if status_code == 402 and error_code == "credit_exhausted":
        cls: type[AetherApiError] = CreditExhaustedError
    elif status_code == 402 and error_code == "free_limit_exceeded":
        cls = FreeLimitExceededError
    elif status_code == 403 and error_code == "tenant_paused":
        cls = TenantPausedError
    elif status_code == 400 and error_code == "partition_required":
        cls = PartitionRequiredError
    else:
        cls = AetherApiError
    return cls(
        status_code,
        message,
        error_code=error_code,
        request_id=request_id,
        body=body,
    )


class AetherNetworkError(AetherError):
    """Raised when a request fails due to a network issue (connection, timeout, DNS)."""

    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        self.message = message
        self.__cause__ = cause
        super().__init__(message)
