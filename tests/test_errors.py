"""Phase 8 / ADR-015 error-mapping tests for the Python SDK."""

from __future__ import annotations

from aether.errors import (
    AetherApiError,
    CreditExhaustedError,
    FreeLimitExceededError,
    TenantPausedError,
    aether_api_error_from_response,
)


def test_factory_returns_credit_exhausted_for_402_credit_exhausted():
    err = aether_api_error_from_response(
        402, "Top up your balance", error_code="credit_exhausted"
    )
    assert isinstance(err, CreditExhaustedError)
    assert isinstance(err, AetherApiError)
    assert err.status_code == 402
    assert err.error_code == "credit_exhausted"
    assert not err.is_retryable


def test_factory_returns_free_limit_exceeded_for_402_free_limit_exceeded():
    err = aether_api_error_from_response(
        402, "Free plan limit", error_code="free_limit_exceeded"
    )
    assert isinstance(err, FreeLimitExceededError)
    # Distinct from CreditExhaustedError — siblings, not subclasses
    assert not isinstance(err, CreditExhaustedError)
    assert not err.is_retryable


def test_factory_returns_tenant_paused_for_403_tenant_paused():
    err = aether_api_error_from_response(
        403, "Tenant paused by operator", error_code="tenant_paused"
    )
    assert isinstance(err, TenantPausedError)
    assert err.status_code == 403


def test_factory_falls_back_for_unknown_402_code():
    err = aether_api_error_from_response(402, "Other billing", error_code="something_else")
    assert isinstance(err, AetherApiError)
    assert not isinstance(err, CreditExhaustedError)
    assert not isinstance(err, FreeLimitExceededError)


def test_factory_falls_back_for_402_no_code():
    err = aether_api_error_from_response(402, "Generic")
    assert type(err) is AetherApiError


def test_factory_falls_back_for_unrelated_status():
    err = aether_api_error_from_response(404, "Not found")
    assert type(err) is AetherApiError
    assert not isinstance(err, TenantPausedError)


def test_subclass_is_not_retryable():
    # 402/403 must never end up in the retry classification.
    for cls, status in [
        (CreditExhaustedError, 402),
        (FreeLimitExceededError, 402),
        (TenantPausedError, 403),
    ]:
        err = cls(status, "x")
        assert not err.is_retryable


# ── Canonical billing bodies ─────────────────────────────────────────
#
# These mirror the exact wire shape the engine emits for billing
# rejections: {"error", "code", "request_id", ...optional context}.
# Each fixture is fed through the factory the way the client's
# _raise_for_status() builds its arguments (message <- "error",
# error_code <- "code", request_id <- header/body), and we assert the
# correct typed subclass plus populated attributes.

TENANT_PAUSED_BODY = {
    "error": "Tenant has been paused by the operator",
    "code": "tenant_paused",
    "request_id": "req-123",
}

CREDIT_EXHAUSTED_BODY = {
    "error": "Prepaid credit balance exhausted; top up to continue.",
    "code": "credit_exhausted",
    "request_id": "req-123",
    "resource": "vectors",
    "balance_cents": 0,
}

FREE_LIMIT_EXCEEDED_BODY = {
    "error": "Free vector limit exceeded (1001/1000)",
    "code": "free_limit_exceeded",
    "request_id": "req-123",
    "limit_type": "vectors",
    "plan": "free",
}


def _from_body(status_code: int, body: dict) -> AetherApiError:
    """Build an error the same way the client does from a parsed body."""
    return aether_api_error_from_response(
        status_code,
        body["error"],
        error_code=body.get("code"),
        request_id=body.get("request_id"),
        body=body,
    )


def test_canonical_tenant_paused_body_maps_to_typed_error():
    err = _from_body(403, TENANT_PAUSED_BODY)
    assert type(err) is TenantPausedError
    assert err.status_code == 403
    assert err.error_code == "tenant_paused"
    assert err.request_id == "req-123"
    assert err.message == "Tenant has been paused by the operator"
    assert err.body == TENANT_PAUSED_BODY
    assert not err.is_retryable


def test_canonical_credit_exhausted_body_maps_to_typed_error():
    err = _from_body(402, CREDIT_EXHAUSTED_BODY)
    assert type(err) is CreditExhaustedError
    assert err.status_code == 402
    assert err.error_code == "credit_exhausted"
    assert err.request_id == "req-123"
    assert err.message == "Prepaid credit balance exhausted; top up to continue."
    # Optional context fields survive on the raw body.
    assert err.body["resource"] == "vectors"
    assert err.body["balance_cents"] == 0
    assert not err.is_retryable


def test_canonical_free_limit_exceeded_body_maps_to_typed_error():
    err = _from_body(402, FREE_LIMIT_EXCEEDED_BODY)
    assert type(err) is FreeLimitExceededError
    # Distinct from a credit failure even though both are 402.
    assert not isinstance(err, CreditExhaustedError)
    assert err.status_code == 402
    assert err.error_code == "free_limit_exceeded"
    assert err.request_id == "req-123"
    assert err.body["plan"] == "free"
    assert not err.is_retryable


def test_canonical_unknown_code_falls_back_to_base():
    body = {"error": "Some other billing thing", "code": "mystery", "request_id": "req-999"}
    err = _from_body(402, body)
    assert type(err) is AetherApiError
    assert err.error_code == "mystery"
    assert err.status_code == 402
    assert err.request_id == "req-999"
