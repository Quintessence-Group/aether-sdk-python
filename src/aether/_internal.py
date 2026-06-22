"""Internal helpers shared by the sync and async clients.

Not part of the public API; subject to change without notice.
"""

from __future__ import annotations

import ipaddress
import platform
import uuid
from typing import Optional
from urllib.parse import urlparse

__version__ = "0.3.1"

#: Sent on every request so the server can attribute traffic by SDK + version.
USER_AGENT = (
    f"aether-sdk-python/{__version__} "
    f"(python/{platform.python_version()}; httpx)"
)


def new_idempotency_key() -> str:
    """Return a fresh idempotency key for a single logical write operation.

    The same key is reused across retries of one call so the server can
    deduplicate a request whose response was lost in transit; a new key is
    minted for each distinct call.
    """
    return str(uuid.uuid4())


def _is_loopback(host: str) -> bool:
    if host in ("", "localhost") or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def enforce_secure_base_url(base_url: str, api_key: Optional[str]) -> None:
    """Raise ``ValueError`` if an API key would be sent over cleartext HTTP to
    a non-loopback host. Loopback addresses (localhost/127.0.0.0/8/::1) are
    allowed so local development against a non-TLS node still works.
    """
    if not api_key:
        return
    parsed = urlparse(base_url)
    if parsed.scheme == "http" and not _is_loopback(parsed.hostname or ""):
        raise ValueError(
            f"Refusing to send API key over insecure HTTP to {parsed.hostname!r}. "
            "Use an https:// base URL, or omit the API key for local non-TLS endpoints."
        )
