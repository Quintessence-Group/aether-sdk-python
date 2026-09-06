"""Pure, offline helpers for the Connections
API's connect-session redirect signature. No network calls; safe to run in
whatever request handler the application's backend uses for the OAuth
callback.
"""

from __future__ import annotations

import hashlib
import hmac


def verify_redirect_signature(
    client_secret: str,
    *,
    session: str,
    status: str,
    connection_id: str,
    sig: str,
) -> bool:
    """Verify a ``create_connect_session`` redirect's signature
    (``docs/SDK_API_CONTRACT.md`` §4.18).

    ``client_secret`` is the value returned exactly once by
    ``AetherClient.create_connect_session``. The comparison is
    constant-time (:func:`hmac.compare_digest`) so this function itself
    never becomes a timing oracle on the signature.

    Returns ``True`` iff ``sig`` matches the recomputed signature::

        sig = hex(HMAC-SHA256(
                key = SHA-256(client_secret),
                message = "<session>|<status>|<connection_id>"))
    """
    key = hashlib.sha256(client_secret.encode("utf-8")).digest()
    message = f"{session}|{status}|{connection_id}".encode("utf-8")
    expected = hmac.new(key, message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)
