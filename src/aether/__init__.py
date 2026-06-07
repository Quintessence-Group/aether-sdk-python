"""Aether Python SDK."""

from ._internal import __version__
from .async_client import AsyncAetherClient
from .client import AetherClient
from .errors import (
    AetherApiError,
    AetherError,
    AetherNetworkError,
    CreditExhaustedError,
    FreeLimitExceededError,
    TenantPausedError,
    aether_api_error_from_response,
)
from .models import (
    BatchInsertItem,
    BatchSearchQuery,
    BatchSearchResponse,
    DocumentPage,
    DocumentRecord,
    NodeStatus,
    RetrievalResult,
    SearchResult,
)

__all__ = [
    "__version__",
    "AetherClient",
    "AsyncAetherClient",
    "AetherError",
    "AetherApiError",
    "AetherNetworkError",
    "CreditExhaustedError",
    "FreeLimitExceededError",
    "TenantPausedError",
    "aether_api_error_from_response",
    "BatchInsertItem",
    "BatchSearchQuery",
    "BatchSearchResponse",
    "DocumentPage",
    "DocumentRecord",
    "SearchResult",
    "RetrievalResult",
    "NodeStatus",
]
