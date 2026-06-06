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
