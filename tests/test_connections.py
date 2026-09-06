"""The Connections API surface: mint / list /
get / delete / resync / browse / update_selection / get_purge_receipt, the
typed errors, and the offline redirect-signature verifier. Driven through a
real client over httpx.MockTransport so the genuine request/parse/error-
mapping path runs, mirroring test_partitions.py's pattern."""

import hashlib
import hmac
import json

import httpx
import pytest

from aether import (
    AetherApiError,
    AetherClient,
    PartitionMismatchError,
    SessionInvalidError,
    verify_redirect_signature,
)


def _mock_client(handler) -> AetherClient:
    c = AetherClient(base_url="http://localhost:9000", api_key="test-key", max_retries=0)
    c._client = httpx.Client(
        base_url=c.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )
    return c


# ── create_connect_session ───────────────────────────────────────────

def test_create_connect_session_mints_and_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/connections/sessions"
        body = json.loads(request.content)
        assert body == {
            "provider": "dropbox",
            "external_user_id": "priya",
            "return_url": "https://acme.example.com/cb",
        }
        return httpx.Response(
            200,
            json={
                "session_token": "acs_deadbeef",
                "connect_url": "https://connect.example.com/connect/acs_deadbeef",
                "client_secret": "acsec_secretsecret",
                "expires_at": "2026-08-15T00:00:00Z",
            },
        )

    session = _mock_client(handler).create_connect_session(
        "priya", "https://acme.example.com/cb"
    )
    assert session.session_token == "acs_deadbeef"
    assert session.connect_url.endswith("/connect/acs_deadbeef")
    assert session.client_secret == "acsec_secretsecret"


def test_create_connect_session_sends_target_partition_override():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["target_partition"] == "shared-corpus"
        return httpx.Response(
            200,
            json={
                "session_token": "acs_x",
                "connect_url": "https://connect.example.com/connect/acs_x",
                "client_secret": "acsec_x",
                "expires_at": "2026-08-15T00:00:00Z",
            },
        )

    _mock_client(handler).create_connect_session(
        "priya", "https://acme.example.com/cb", target_partition="shared-corpus"
    )


def test_create_connect_session_on_a_handle_asserts_partition():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("partition") == "priya"
        return httpx.Response(
            200,
            json={
                "session_token": "acs_x",
                "connect_url": "https://connect.example.com/connect/acs_x",
                "client_secret": "acsec_x",
                "expires_at": "2026-08-15T00:00:00Z",
            },
        )

    _mock_client(handler).partition("priya").create_connect_session(
        "priya", "https://acme.example.com/cb"
    )


def test_create_connect_session_rejects_empty_args():
    client = _mock_client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        client.create_connect_session("", "https://acme.example.com/cb")
    with pytest.raises(ValueError):
        client.create_connect_session("priya", "")


def test_create_connect_session_partition_mismatch_is_typed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": "this session would resolve to partition \"priya\"",
                "code": "partition_mismatch",
            },
        )

    with pytest.raises(PartitionMismatchError) as exc:
        _mock_client(handler).partition("someone-else").create_connect_session(
            "priya", "https://acme.example.com/cb"
        )
    assert exc.value.status_code == 400
    assert exc.value.error_code == "partition_mismatch"


# ── list_connections ─────────────────────────────────────────────────

def test_list_connections_parses_every_field():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/connections"
        return httpx.Response(
            200,
            json={
                "connections": [
                    {
                        "connection_id": "11111111-1111-1111-1111-111111111111",
                        "provider": "dropbox",
                        "owner_type": "external_user",
                        "owner_id": "priya",
                        "provider_account_id": "dbid:priya",
                        "account_display_name": "Priya",
                        "target_partition": "priya",
                        "status": "active",
                        "granted_scopes": ["files.metadata.read"],
                        "created_at": "2026-08-15T00:00:00Z",
                        "last_sync_at": None,
                        "last_error": None,
                        "files_synced": 3,
                        "files_skipped": 0,
                        "files_deleted": 0,
                        "selected_paths": [],
                        "purge_state": "not_started",
                        "purge_receipt_id": None,
                        "credential_deleted": False,
                    }
                ]
            },
        )

    conns = _mock_client(handler).list_connections()
    assert len(conns) == 1
    c = conns[0]
    assert c.owner_type == "external_user"
    assert c.owner_id == "priya"
    assert c.target_partition == "priya"
    assert c.files_synced == 3
    assert c.purge_state == "not_started"
    assert c.credential_deleted is False


def test_list_connections_sends_owner_filters_and_include_purged():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("owner_type") == "external_user"
        assert request.url.params.get("owner_id") == "priya"
        assert request.url.params.get("include_purged") == "false"
        return httpx.Response(200, json={"connections": []})

    _mock_client(handler).list_connections(
        owner_type="external_user", owner_id="priya", include_purged=False
    )


def test_list_connections_on_a_handle_sends_partition():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("partition") == "priya"
        return httpx.Response(200, json={"connections": []})

    _mock_client(handler).partition("priya").list_connections()


# ── get_connection ───────────────────────────────────────────────────

def test_get_connection_fetches_by_id():
    cid = "11111111-1111-1111-1111-111111111111"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v1/connections/{cid}"
        return httpx.Response(
            200,
            json={
                "connection_id": cid,
                "provider": "dropbox",
                "owner_type": "tenant",
                "owner_id": None,
                "provider_account_id": "dbid:acme",
                "account_display_name": None,
                "target_partition": None,
                "status": "active",
                "granted_scopes": [],
                "created_at": "2026-08-15T00:00:00Z",
                "last_sync_at": None,
                "last_error": None,
                "files_synced": 0,
                "files_skipped": 0,
                "files_deleted": 0,
                "selected_paths": [],
                "purge_state": "not_started",
                "purge_receipt_id": None,
                "credential_deleted": False,
            },
        )

    conn = _mock_client(handler).get_connection(cid)
    assert conn.connection_id == cid
    assert conn.owner_type == "tenant"
    assert conn.target_partition is None


def test_get_connection_wrong_partition_is_the_plain_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": "unknown connection", "code": "connection_not_found"}
        )

    with pytest.raises(AetherApiError) as exc:
        _mock_client(handler).partition("someone-else").get_connection("11111111-1111-1111-1111-111111111111")
    assert exc.value.status_code == 404
    assert exc.value.error_code == "connection_not_found"


def test_get_connection_rejects_empty_id():
    client = _mock_client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        client.get_connection("")


# ── delete_connection ────────────────────────────────────────────────

def test_delete_connection_parses_purge_summary():
    cid = "11111111-1111-1111-1111-111111111111"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == f"/v1/connections/{cid}"
        return httpx.Response(
            200,
            json={
                "connection_id": cid,
                "status": "revoked",
                "purge": {
                    "receipt_id": "r1",
                    "documents_purged": 5,
                    "merkle_root": "deadbeef",
                    "completed_at": "2026-08-15T00:00:00Z",
                    "signer_node_id": "node-1",
                },
            },
        )

    result = _mock_client(handler).delete_connection(cid)
    assert result.connection_id == cid
    assert result.status == "revoked"
    assert result.purge is not None
    assert result.purge.documents_purged == 5
    assert result.purge.receipt_id == "r1"


def test_delete_connection_idempotent_no_op_has_no_purge():
    cid = "11111111-1111-1111-1111-111111111111"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"connection_id": cid, "status": "revoked", "purge": None})

    result = _mock_client(handler).delete_connection(cid)
    assert result.purge is None


def test_delete_connection_rejects_empty_id():
    client = _mock_client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        client.delete_connection("")


# ── resync_connection ────────────────────────────────────────────────

def test_resync_connection_posts_then_refetches():
    cid = "11111111-1111-1111-1111-111111111111"
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == f"/v1/connections/{cid}/resync":
            assert request.method == "POST"
            return httpx.Response(200, json={"connection_id": cid, "status": "active"})
        return httpx.Response(
            200,
            json={
                "connection_id": cid,
                "provider": "dropbox",
                "owner_type": "tenant",
                "owner_id": None,
                "provider_account_id": "dbid:acme",
                "account_display_name": None,
                "target_partition": None,
                "status": "active",
                "granted_scopes": [],
                "created_at": "2026-08-15T00:00:00Z",
                "last_sync_at": None,
                "last_error": None,
                "files_synced": 0,
                "files_skipped": 0,
                "files_deleted": 0,
                "selected_paths": [],
                "purge_state": "not_started",
                "purge_receipt_id": None,
                "credential_deleted": False,
            },
        )

    conn = _mock_client(handler).resync_connection(cid)
    assert conn.status == "active"
    assert ("POST", f"/v1/connections/{cid}/resync") in calls
    assert ("GET", f"/v1/connections/{cid}") in calls


# ── browse_connection / update_selection ─────────────────────────────

def test_browse_connection_sends_path_and_cursor_and_parses_page():
    cid = "11111111-1111-1111-1111-111111111111"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/v1/connections/{cid}/browse"
        body = json.loads(request.content)
        assert body == {"path": "/Reports", "cursor": None}
        return httpx.Response(
            200,
            json={
                "entries": [
                    {"name": "q3.txt", "path_display": "/Reports/q3.txt", "is_folder": False, "size_bytes": 42}
                ],
                "next_cursor": "cursor-2",
            },
        )

    page = _mock_client(handler).browse_connection(cid, path="/Reports")
    assert page.entries[0].name == "q3.txt"
    assert page.entries[0].is_folder is False
    assert page.next_cursor == "cursor-2"


def test_update_selection_replaces_paths_and_returns_normalized():
    cid = "11111111-1111-1111-1111-111111111111"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert request.url.path == f"/v1/connections/{cid}/selection"
        body = json.loads(request.content)
        assert body == {"selected_paths": ["/Reports"]}
        return httpx.Response(
            200, json={"connection_id": cid, "selected_paths": ["/Reports"]}
        )

    paths = _mock_client(handler).update_selection(cid, ["/Reports"])
    assert paths == ["/Reports"]


# ── get_purge_receipt ────────────────────────────────────────────────

def test_get_purge_receipt_parses_full_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/connections/purge-receipts/r1"
        return httpx.Response(
            200,
            json={
                "version": "1",
                "receipt_id": "r1",
                "tenant_id": "t1",
                "connection_id": "c1",
                "provider": "dropbox",
                "owner": "external_user:priya",
                "provider_account_id": "dbid:priya",
                "documents_purged": 5,
                "documents_failed": 0,
                "merkle_root": "deadbeef",
                "merkle_leaf_count": 5,
                "purged_document_ids": ["d1", "d2"],
                "partitions_touched": ["priya"],
                "default_partition_touched": False,
                "credential_revocation": "revoked",
                "credential_deleted": True,
                "started_at": "2026-08-15T00:00:00Z",
                "completed_at": "2026-08-15T00:00:01Z",
                "signer_node_id": "node-1",
                "signer_public_key": "pub-1",
                "signature": "sig-1",
                "verified": True,
            },
        )

    receipt = _mock_client(handler).get_purge_receipt("r1")
    assert receipt.documents_purged == 5
    assert receipt.partitions_touched == ["priya"]
    assert receipt.verified is True


# ── session_invalid typed error ──────────────────────────────────────

def test_session_invalid_is_typed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": "this connect session is unknown, already used, or expired",
                "code": "session_invalid",
            },
        )

    with pytest.raises(SessionInvalidError) as exc:
        _mock_client(handler).create_connect_session("priya", "https://acme.example.com/cb")
    assert exc.value.status_code == 400
    assert exc.value.error_code == "session_invalid"


# ── verify_redirect_signature (pure, offline) ────────────────────────

def _reference_sig(client_secret: str, session: str, status: str, connection_id: str) -> str:
    key = hashlib.sha256(client_secret.encode("utf-8")).digest()
    message = f"{session}|{status}|{connection_id}".encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def test_verify_redirect_signature_accepts_a_correct_signature():
    secret = "acsec_the-real-secret"
    sig = _reference_sig(secret, "acs_tok", "active", "conn-1")
    assert verify_redirect_signature(
        secret, session="acs_tok", status="active", connection_id="conn-1", sig=sig
    )


def test_verify_redirect_signature_rejects_a_tampered_param():
    secret = "acsec_the-real-secret"
    sig = _reference_sig(secret, "acs_tok", "active", "conn-1")
    assert not verify_redirect_signature(
        secret, session="acs_tok", status="error", connection_id="conn-1", sig=sig
    )


def test_verify_redirect_signature_rejects_the_wrong_secret():
    sig = _reference_sig("acsec_the-real-secret", "acs_tok", "active", "conn-1")
    assert not verify_redirect_signature(
        "acsec_a-different-secret", session="acs_tok", status="active", connection_id="conn-1", sig=sig
    )
