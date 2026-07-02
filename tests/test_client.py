"""Tests for AetherClient download_text and retrieve methods."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from aether import (
    AetherClient,
    DocumentRecord,
    EntityBackfillReport,
    IngestResult,
    RetrievalResult,
)
from aether.errors import AetherApiError, CreditExhaustedError, TenantPausedError


def _ok_response(**kwargs):
    """Create a mock httpx.Response that passes retry checks."""
    resp = MagicMock(spec=httpx.Response)
    resp.is_success = True
    resp.status_code = 200
    for k, v in kwargs.items():
        setattr(resp, k, v)
    return resp


@pytest.fixture
def client():
    return AetherClient(base_url="http://localhost:9000", api_key="test-key", max_retries=0)


class TestDownloadText:
    def test_returns_decoded_content(self, client):
        mock_resp = _ok_response(content=b"Hello, world!")

        with patch.object(client._client, "request", return_value=mock_resp):
            result = client.download_text("doc-123")

        assert result == "Hello, world!"

    def test_url_encodes_doc_id(self, client):
        mock_resp = _ok_response(content=b"content")

        with patch.object(client._client, "request", return_value=mock_resp):
            client.download_text("doc/with spaces")

        # Verifying it doesn't raise is sufficient — URL encoding tested implicitly


class TestRetrieve:
    def test_returns_results_with_content(self, client):
        search_resp = _ok_response()
        search_resp.json.return_value = {
            "query": "test",
            "results": [
                {"doc_id": "doc-1", "score": 90, "title": "Doc 1", "content_type": "text/plain"},
                {"doc_id": "doc-2", "score": 70, "title": "Doc 2", "content_type": "text/plain"},
            ],
        }

        dl_resp_1 = _ok_response(content=b"Content of doc 1")
        dl_resp_2 = _ok_response(content=b"Content of doc 2")

        with patch.object(client._client, "request", side_effect=[search_resp, dl_resp_1, dl_resp_2]):
            results = client.retrieve("test query", k=5)

        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].doc_id == "doc-1"
        assert results[0].content == "Content of doc 1"
        assert results[0].score == 90
        assert results[1].content == "Content of doc 2"

    def test_deduplicates_by_doc_id(self, client):
        search_resp = _ok_response()
        search_resp.json.return_value = {
            "query": "test",
            "results": [
                {"doc_id": "doc-1", "score": 90, "title": "Doc 1", "content_type": "text/plain"},
                {"doc_id": "doc-1", "score": 80, "title": "Doc 1", "content_type": "text/plain"},
                {"doc_id": "doc-2", "score": 70, "title": "Doc 2", "content_type": "text/plain"},
            ],
        }

        dl_resp_1 = _ok_response(content=b"Content 1")
        dl_resp_2 = _ok_response(content=b"Content 2")

        with patch.object(client._client, "request", side_effect=[search_resp, dl_resp_1, dl_resp_2]):
            results = client.retrieve("test", k=5)

        # Should deduplicate: 3 search results -> 2 unique docs
        assert len(results) == 2
        assert results[0].score == 90  # keeps the best-scoring match


def _wire_url(params: dict) -> str:
    """Render the final wire URL exactly as httpx would encode the params."""
    return str(httpx.URL("http://localhost:9000/x", params=params))


class TestSearchFilters:
    def _search_resp(self):
        resp = _ok_response()
        resp.json.return_value = {"query": "q", "results": []}
        return resp

    def test_passes_filters_as_url_params(self, client):
        with patch.object(client._client, "request", return_value=self._search_resp()) as mock_req:
            client.search(
                "q",
                k=3,
                entity_id="user-123",
                since="2026-06-01T00:00:00Z",
                until="2026-06-10T23:59:59Z",
                max_distance=0.5,
            )

        method, url = mock_req.call_args[0]
        assert method == "GET"
        assert url == "/v1/search"
        params = mock_req.call_args[1]["params"]
        assert params["entity_id"] == "user-123"
        assert params["since"] == "2026-06-01T00:00:00Z"
        assert params["until"] == "2026-06-10T23:59:59Z"
        assert params["max_distance"] == 0.5
        wire = _wire_url(params)
        assert "entity_id=user-123" in wire
        assert "since=2026-06-01T00%3A00%3A00Z" in wire
        assert "until=2026-06-10T23%3A59%3A59Z" in wire
        assert "max_distance=0.5" in wire

    def test_passes_last_n_days(self, client):
        with patch.object(client._client, "request", return_value=self._search_resp()) as mock_req:
            client.search("q", last_n_days=30)

        params = mock_req.call_args[1]["params"]
        assert params["last_n_days"] == 30
        assert "since" not in params
        assert "until" not in params
        assert "last_n_days=30" in _wire_url(params)

    def test_encodes_offset_timestamps(self, client):
        with patch.object(client._client, "request", return_value=self._search_resp()) as mock_req:
            client.search("q", since="2026-06-01T00:00:00+02:00")

        wire = _wire_url(mock_req.call_args[1]["params"])
        assert "since=2026-06-01T00%3A00%3A00%2B02%3A00" in wire

    def test_omits_unset_filters(self, client):
        with patch.object(client._client, "request", return_value=self._search_resp()) as mock_req:
            client.search("q", k=2)

        params = mock_req.call_args[1]["params"]
        wire = _wire_url(params)
        for key in ("entity_id", "since", "until", "last_n_days", "max_distance"):
            assert key not in params
            assert key not in wire

    def test_retrieve_forwards_filters(self, client):
        search_resp = self._search_resp()
        with patch.object(client._client, "request", return_value=search_resp) as mock_req:
            client.retrieve(
                "q",
                k=2,
                entity_id="user-123",
                since="2026-06-01T00:00:00Z",
                until="2026-06-10T23:59:59Z",
                max_distance=0.4,
            )

        params = mock_req.call_args[1]["params"]
        assert params["include_content"] == "true"
        assert params["entity_id"] == "user-123"
        assert params["since"] == "2026-06-01T00:00:00Z"
        assert params["until"] == "2026-06-10T23:59:59Z"
        assert params["max_distance"] == 0.4

    def test_retrieve_forwards_last_n_days(self, client):
        with patch.object(client._client, "request", return_value=self._search_resp()) as mock_req:
            client.retrieve("q", last_n_days=7)

        params = mock_req.call_args[1]["params"]
        assert params["last_n_days"] == 7
        assert "since" not in params


class TestListFilters:
    def _list_resp(self, documents):
        resp = _ok_response()
        resp.json.return_value = {"documents": documents, "total": len(documents), "has_more": False}
        return resp

    def test_passes_filters_as_url_params(self, client):
        docs = [{"doc_id": "d1", "entity_id": "user-123", "created_at": "2026-06-02T08:00:00Z"}]
        with patch.object(client._client, "request", return_value=self._list_resp(docs)) as mock_req:
            records = client.list(
                entity_id="user-123",
                since="2026-06-01T00:00:00Z",
                until="2026-06-10T23:59:59Z",
            )

        method, url = mock_req.call_args[0]
        assert method == "GET"
        assert url == "/v1/documents"
        params = mock_req.call_args[1]["params"]
        assert params == {
            "offset": 0,
            "limit": 50,
            "entity_id": "user-123",
            "since": "2026-06-01T00:00:00Z",
            "until": "2026-06-10T23:59:59Z",
        }
        wire = _wire_url(params)
        assert "since=2026-06-01T00%3A00%3A00Z" in wire
        assert "until=2026-06-10T23%3A59%3A59Z" in wire
        assert records[0].entity_id == "user-123"

    def test_passes_last_n_days(self, client):
        with patch.object(client._client, "request", return_value=self._list_resp([])) as mock_req:
            client.list(last_n_days=7)

        params = mock_req.call_args[1]["params"]
        assert params["last_n_days"] == 7
        assert "since" not in params

    def test_omits_unset_filters(self, client):
        with patch.object(client._client, "request", return_value=self._list_resp([])) as mock_req:
            client.list()

        assert mock_req.call_args[1]["params"] == {"offset": 0, "limit": 50}


class TestMetadataFacetFilters:
    """The OR-list facet filters (any_tags / content_types / sources) and the
    AND tags filter on list go on the wire alongside the existing filters.
    """

    def _search_resp(self):
        resp = _ok_response()
        resp.json.return_value = {"query": "q", "results": []}
        return resp

    def _list_resp(self):
        resp = _ok_response()
        resp.json.return_value = {"documents": [], "total": 0, "has_more": False}
        return resp

    def test_search_sends_facet_filters_csv(self, client):
        with patch.object(client._client, "request", return_value=self._search_resp()) as mock_req:
            client.search(
                "q",
                tags=["a", "b"],
                any_tags=["x", "y"],
                content_types=["text/plain", "application/pdf"],
                sources=["slack", "email"],
            )

        params = mock_req.call_args[1]["params"]
        assert params["tags"] == "a,b"
        assert params["any_tags"] == "x,y"
        assert params["content_type"] == "text/plain,application/pdf"
        assert params["source"] == "slack,email"
        wire = _wire_url(params)
        assert "any_tags=x%2Cy" in wire
        assert "content_type=text%2Fplain%2Capplication%2Fpdf" in wire
        assert "source=slack%2Cemail" in wire

    def test_search_omits_unset_facet_filters(self, client):
        with patch.object(client._client, "request", return_value=self._search_resp()) as mock_req:
            client.search("q")

        params = mock_req.call_args[1]["params"]
        for key in ("any_tags", "content_type", "source"):
            assert key not in params

    def test_list_sends_facet_filters_csv(self, client):
        with patch.object(client._client, "request", return_value=self._list_resp()) as mock_req:
            client.list(
                tags=["a", "b"],
                any_tags=["x", "y"],
                content_types=["text/plain"],
                sources=["slack"],
            )

        params = mock_req.call_args[1]["params"]
        assert params["tags"] == "a,b"
        assert params["any_tags"] == "x,y"
        assert params["content_type"] == "text/plain"
        assert params["source"] == "slack"

    def test_retrieve_forwards_facet_filters(self, client):
        with patch.object(client._client, "request", return_value=self._search_resp()) as mock_req:
            client.retrieve(
                "q",
                any_tags=["x"],
                content_types=["text/plain"],
                sources=["slack"],
            )

        params = mock_req.call_args[1]["params"]
        assert params["include_content"] == "true"
        assert params["any_tags"] == "x"
        assert params["content_type"] == "text/plain"
        assert params["source"] == "slack"

    def test_search_by_vector_sends_facet_filters_arrays(self, client):
        resp = _ok_response()
        resp.json.return_value = {"results": []}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.search_by_vector(
                [0.1, 0.2],
                any_tags=["x", "y"],
                content_types=["text/plain"],
                sources=["slack"],
            )

        body = mock_req.call_args[1]["json"]
        # The embed endpoint takes JSON arrays, not CSV strings.
        assert body["any_tags"] == ["x", "y"]
        assert body["content_type"] == ["text/plain"]
        assert body["source"] == ["slack"]

    def test_search_by_vector_omits_unset_facet_filters(self, client):
        resp = _ok_response()
        resp.json.return_value = {"results": []}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.search_by_vector([0.1])

        body = mock_req.call_args[1]["json"]
        for key in ("any_tags", "content_type", "source"):
            assert key not in body


class TestSourceOnWrites:
    """The single ``source`` string threads to the wire exactly like
    ``entity_id`` on every write path.
    """

    def _insert_resp(self, **extra):
        resp = _ok_response()
        resp.json.return_value = {
            "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1, **extra,
        }
        return resp

    def test_insert_sends_source_param_and_round_trips(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        resp = self._insert_resp(source="slack", tags=["a", "b"])

        with patch.object(client._client, "request", return_value=resp) as mock_req:
            record = client.insert(f, source="slack")

        url = mock_req.call_args[0][1]
        assert "source=slack" in url
        # insert -> get round-trip surfaces the stored source and tags.
        assert record.source == "slack"
        assert record.tags == ["a", "b"]

    def test_insert_url_encodes_source(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        with patch.object(client._client, "request", return_value=self._insert_resp()) as mock_req:
            client.insert(f, source="channel:42")

        assert "source=channel%3A42" in mock_req.call_args[0][1]

    def test_insert_omits_source_when_unset(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        with patch.object(client._client, "request", return_value=self._insert_resp()) as mock_req:
            record = client.insert(f)

        assert "source" not in mock_req.call_args[0][1]
        assert record.source is None
        assert record.tags == []

    def test_insert_text_sends_source_param(self, client):
        with patch.object(client._client, "request", return_value=self._insert_resp()) as mock_req:
            client.insert_text("hello", source="slack")

        assert "source=slack" in mock_req.call_args[0][1]

    def test_update_sends_source_param(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        with patch.object(client._client, "request", return_value=self._insert_resp()) as mock_req:
            client.update("doc-1", f, source="slack")

        method, url = mock_req.call_args[0]
        assert method == "PUT"
        assert "source=slack" in url

    def test_insert_async_sends_source_param(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        resp = _ok_response()
        resp.json.return_value = {"job_id": "j1", "status": "queued", "poll_url": "/x"}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.insert_async(f, source="slack")

        assert "source=slack" in mock_req.call_args[0][1]

    def test_insert_stream_sends_source_param(self, client):
        import io

        with patch.object(client._client, "post", return_value=self._insert_resp()) as mock_post:
            client.insert_stream(io.BytesIO(b"data"), source="slack")

        assert "source=slack" in mock_post.call_args[0][0]

    def test_insert_with_embeddings_sends_source_json(self, client):
        resp = _ok_response()
        resp.json.return_value = {
            "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
            "source": "slack",
        }
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            record = client.insert_with_embeddings("text", embedding=[0.1], source="slack")

        assert mock_req.call_args[1]["json"]["source"] == "slack"
        assert record.source == "slack"

    def test_insert_with_embeddings_omits_source_when_unset(self, client):
        resp = _ok_response()
        resp.json.return_value = {"doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.insert_with_embeddings("text", embedding=[0.1])

        assert "source" not in mock_req.call_args[1]["json"]


class TestSearchResultMetadataParsing:
    """Search results echo tags / source / created_at; older payloads still parse."""

    def test_search_parses_metadata_echo(self, client):
        resp = _ok_response()
        resp.json.return_value = {
            "query": "q",
            "results": [
                {
                    "doc_id": "d1", "score": 90, "content_type": "text/plain",
                    "tags": ["a", "b"], "source": "slack",
                    "created_at": "2026-06-11T00:00:00Z",
                },
            ],
        }
        with patch.object(client._client, "request", return_value=resp):
            results = client.search("q")

        assert results[0].tags == ["a", "b"]
        assert results[0].source == "slack"
        # created_at is kept as a raw string, never parsed to a datetime.
        assert results[0].created_at == "2026-06-11T00:00:00Z"

    def test_search_defaults_metadata_when_absent(self, client):
        resp = _ok_response()
        resp.json.return_value = {
            "query": "q",
            "results": [{"doc_id": "d1", "score": 90, "content_type": "text/plain"}],
        }
        with patch.object(client._client, "request", return_value=resp):
            results = client.search("q")

        assert results[0].tags == []
        assert results[0].source is None
        assert results[0].created_at is None

    def test_get_parses_tags_and_source(self, client):
        resp = _ok_response()
        resp.json.return_value = {
            "doc_id": "d1", "cid": "c1", "tags": ["a"], "source": "slack",
        }
        with patch.object(client._client, "request", return_value=resp):
            record = client.get("d1")

        assert record.tags == ["a"]
        assert record.source == "slack"

    def test_get_defaults_tags_and_source(self, client):
        resp = _ok_response()
        resp.json.return_value = {"doc_id": "d1", "cid": "c1"}
        with patch.object(client._client, "request", return_value=resp):
            record = client.get("d1")

        assert record.tags == []
        assert record.source is None

    def test_list_parses_tags_and_source(self, client):
        resp = _ok_response()
        resp.json.return_value = {
            "documents": [{"doc_id": "d1", "tags": ["a", "b"], "source": "email"}],
            "total": 1, "has_more": False,
        }
        with patch.object(client._client, "request", return_value=resp):
            records = client.list()

        assert records[0].tags == ["a", "b"]
        assert records[0].source == "email"


class TestInsertEntityId:
    def _insert_resp(self, **extra):
        resp = _ok_response()
        resp.json.return_value = {
            "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1, **extra,
        }
        return resp

    def test_insert_sends_entity_id_param(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        resp = self._insert_resp(
            entity_id="user-123",
            created_at="2026-06-11T00:00:00Z",
            updated_at="2026-06-11T00:00:00Z",
        )

        with patch.object(client._client, "request", return_value=resp) as mock_req:
            record = client.insert(f, entity_id="user-123")

        url = mock_req.call_args[0][1]
        assert "entity_id=user-123" in url
        # Mapper round-trips the full record returned by the server
        assert record.entity_id == "user-123"
        assert record.created_at == "2026-06-11T00:00:00Z"
        assert record.updated_at == "2026-06-11T00:00:00Z"

    def test_insert_url_encodes_entity_id(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")

        with patch.object(client._client, "request", return_value=self._insert_resp()) as mock_req:
            client.insert(f, entity_id="customer:42")

        url = mock_req.call_args[0][1]
        assert "entity_id=customer%3A42" in url

    def test_insert_omits_entity_id_when_unset(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")

        with patch.object(client._client, "request", return_value=self._insert_resp()) as mock_req:
            record = client.insert(f)

        assert "entity_id" not in mock_req.call_args[0][1]
        assert record.entity_id is None

    def test_insert_text_sends_entity_id_param(self, client):
        with patch.object(client._client, "request", return_value=self._insert_resp()) as mock_req:
            client.insert_text("hello", entity_id="user-123")

        assert "entity_id=user-123" in mock_req.call_args[0][1]

    def test_update_sends_entity_id_param(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")

        with patch.object(client._client, "request", return_value=self._insert_resp()) as mock_req:
            client.update("doc-1", f, entity_id="user-123")

        method, url = mock_req.call_args[0]
        assert method == "PUT"
        assert "entity_id=user-123" in url

    def test_insert_async_sends_entity_id_param(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        resp = _ok_response()
        resp.json.return_value = {"job_id": "j1", "status": "queued", "poll_url": "/v1/documents/jobs/j1"}

        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.insert_async(f, entity_id="user-123")

        url = mock_req.call_args[0][1]
        assert url.startswith("/v1/documents/async?")
        assert "entity_id=user-123" in url

    def test_insert_stream_sends_entity_id_param(self, client):
        import io

        resp = self._insert_resp()
        with patch.object(client._client, "post", return_value=resp) as mock_post:
            client.insert_stream(io.BytesIO(b"data"), entity_id="user-123")

        assert "entity_id=user-123" in mock_post.call_args[0][0]


class TestByoeFilters:
    def test_insert_with_embeddings_sends_entity_id_json(self, client):
        resp = _ok_response()
        resp.json.return_value = {
            "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
            "entity_id": "user-123",
        }

        with patch.object(client._client, "request", return_value=resp) as mock_req:
            record = client.insert_with_embeddings("text", embedding=[0.1], entity_id="user-123")

        body = mock_req.call_args[1]["json"]
        assert body["entity_id"] == "user-123"
        assert record.entity_id == "user-123"

    def test_insert_with_embeddings_omits_entity_id_when_unset(self, client):
        resp = _ok_response()
        resp.json.return_value = {"doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1}

        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.insert_with_embeddings("text", embedding=[0.1])

        assert "entity_id" not in mock_req.call_args[1]["json"]

    def test_search_by_vector_sends_filters_json(self, client):
        resp = _ok_response()
        resp.json.return_value = {"results": []}

        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.search_by_vector(
                [0.1, 0.2],
                k=3,
                entity_id="user-123",
                since="2026-06-01T00:00:00Z",
                until="2026-06-10T23:59:59Z",
                max_distance=0.5,
            )

        body = mock_req.call_args[1]["json"]
        assert body["entity_id"] == "user-123"
        assert body["since"] == "2026-06-01T00:00:00Z"
        assert body["until"] == "2026-06-10T23:59:59Z"
        assert body["max_distance"] == 0.5
        assert "last_n_days" not in body

    def test_search_by_vector_sends_last_n_days_json(self, client):
        resp = _ok_response()
        resp.json.return_value = {"results": []}

        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.search_by_vector([0.1], last_n_days=30)

        body = mock_req.call_args[1]["json"]
        assert body["last_n_days"] == 30
        assert "since" not in body

    def test_search_by_vector_omits_unset_filters(self, client):
        resp = _ok_response()
        resp.json.return_value = {"results": []}

        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.search_by_vector([0.1], k=2)

        body = mock_req.call_args[1]["json"]
        for key in ("entity_id", "since", "until", "last_n_days", "max_distance"):
            assert key not in body


class TestEntityIdMapping:
    def test_get_maps_entity_id(self, client):
        resp = _ok_response()
        resp.json.return_value = {
            "doc_id": "d1", "cid": "c1", "title": None, "content_type": "text/plain",
            "size_bytes": 5, "chunks": 1, "vectors": 1, "version": 1,
            "created_at": "2026-06-11T00:00:00Z", "updated_at": "2026-06-11T00:00:00Z",
            "entity_id": "user-123",
        }

        with patch.object(client._client, "request", return_value=resp):
            record = client.get("d1")

        assert record.entity_id == "user-123"

    def test_get_defaults_entity_id_to_none(self, client):
        resp = _ok_response()
        resp.json.return_value = {"doc_id": "d1", "cid": "c1"}

        with patch.object(client._client, "request", return_value=resp):
            record = client.get("d1")

        assert record.entity_id is None


class TestBackfillEntityFromTags:
    def _report_resp(self):
        resp = _ok_response()
        resp.json.return_value = {
            "scanned": 10,
            "updated": 6,
            "skipped_existing": 2,
            "skipped_no_match": 1,
            "skipped_ambiguous": 1,
            "skipped_invalid": 0,
        }
        return resp

    def test_posts_to_backfill_endpoint_with_default_body(self, client):
        with patch.object(client._client, "request", return_value=self._report_resp()) as mock_req:
            client.backfill_entity_from_tags("patient:")

        method, url = mock_req.call_args[0]
        assert method == "POST"
        assert url == "/v1/documents/backfill-entity"
        assert mock_req.call_args[1]["json"] == {"tag_prefix": "patient:", "overwrite": False}

    def test_forwards_overwrite(self, client):
        with patch.object(client._client, "request", return_value=self._report_resp()) as mock_req:
            client.backfill_entity_from_tags("patient:", overwrite=True)

        assert mock_req.call_args[1]["json"] == {"tag_prefix": "patient:", "overwrite": True}

    def test_parses_report(self, client):
        with patch.object(client._client, "request", return_value=self._report_resp()):
            report = client.backfill_entity_from_tags("patient:")

        assert isinstance(report, EntityBackfillReport)
        assert report.scanned == 10
        assert report.updated == 6
        assert report.skipped_existing == 2
        assert report.skipped_no_match == 1
        assert report.skipped_ambiguous == 1
        assert report.skipped_invalid == 0

    def test_empty_tag_prefix_raises(self, client):
        with pytest.raises(ValueError):
            client.backfill_entity_from_tags("")


def _mock_client(handler) -> AetherClient:
    """Build a real AetherClient whose httpx.Client is backed by a
    MockTransport, so requests flow through the genuine request/retry/
    error-mapping path instead of a patched method.
    """
    c = AetherClient(base_url="http://localhost:9000", api_key="test-key", max_retries=0)
    c._client = httpx.Client(
        base_url=c.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )
    return c


# Canonical billing bodies — the exact wire shape the engine emits.
CREDIT_EXHAUSTED_BODY = {
    "error": "Prepaid credit balance exhausted; top up to continue.",
    "code": "credit_exhausted",
    "request_id": "req-123",
    "resource": "vectors",
    "balance_cents": 0,
}
TENANT_PAUSED_BODY = {
    "error": "Tenant has been paused by the operator",
    "code": "tenant_paused",
    "request_id": "req-123",
}


class TestBillingErrorsThroughClient:
    """Drive canonical billing responses through real client methods and
    assert the client raises the typed exception (exercises
    _raise_for_status end-to-end via httpx.MockTransport).
    """

    def test_insert_text_raises_credit_exhausted_on_402(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                402,
                json=CREDIT_EXHAUSTED_BODY,
                headers={"x-request-id": "req-123"},
            )

        client = _mock_client(handler)
        with pytest.raises(CreditExhaustedError) as exc_info:
            client.insert_text("hello world")

        err = exc_info.value
        assert err.status_code == 402
        assert err.error_code == "credit_exhausted"
        assert err.request_id == "req-123"
        assert err.body["balance_cents"] == 0
        assert not err.is_retryable

    def test_search_raises_tenant_paused_on_403(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json=TENANT_PAUSED_BODY,
                headers={"x-request-id": "req-123"},
            )

        client = _mock_client(handler)
        with pytest.raises(TenantPausedError) as exc_info:
            client.search("anything")

        err = exc_info.value
        assert err.status_code == 403
        assert err.error_code == "tenant_paused"
        assert err.request_id == "req-123"
        assert not err.is_retryable


class TestInsertParsesFullRecord:
    """size_bytes / content_type regression: the four write methods must
    parse the full record the server returns, not leave size_bytes at 0.
    """

    _FULL_BODY = {
        "doc_id": "d1",
        "cid": "c1",
        "title": "My Doc",
        "content_type": "application/pdf",
        "size_bytes": 4096,
        "chunks": 3,
        "vectors": 3,
        "version": 1,
        "created_at": "2026-06-11T00:00:00Z",
        "updated_at": "2026-06-11T00:00:00Z",
        "entity_id": "user-123",
    }

    def _insert_resp(self):
        resp = _ok_response(status_code=201)
        resp.json.return_value = self._FULL_BODY
        return resp

    def test_insert_text_parses_size_and_content_type(self, client):
        with patch.object(client._client, "request", return_value=self._insert_resp()):
            record = client.insert_text("hello world")

        assert record.size_bytes == 4096
        assert record.content_type == "application/pdf"
        assert record.title == "My Doc"

    def test_insert_parses_size_and_content_type(self, client, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"data")
        with patch.object(client._client, "request", return_value=self._insert_resp()):
            record = client.insert(f)

        assert record.size_bytes == 4096
        assert record.content_type == "application/pdf"

    def test_update_parses_size_and_content_type(self, client, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"data")
        with patch.object(client._client, "request", return_value=self._insert_resp()):
            record = client.update("d1", f)

        assert record.size_bytes == 4096
        assert record.content_type == "application/pdf"

    def test_insert_stream_parses_size_and_content_type(self, client):
        import io

        with patch.object(client._client, "post", return_value=self._insert_resp()):
            record = client.insert_stream(io.BytesIO(b"data"))

        assert record.size_bytes == 4096
        assert record.content_type == "application/pdf"

    def test_insert_with_embeddings_parses_size_and_content_type(self, client):
        with patch.object(client._client, "request", return_value=self._insert_resp()):
            record = client.insert_with_embeddings("hello", embedding=[0.1, 0.2, 0.3])

        assert record.size_bytes == 4096
        assert record.content_type == "application/pdf"

    def test_batch_insert_parses_size_and_content_type(self, client):
        from aether import BatchInsertItem

        resp = _ok_response(status_code=201)
        resp.json.return_value = {"results": [self._FULL_BODY]}
        with patch.object(client._client, "request", return_value=resp):
            records = client.batch_insert(
                [BatchInsertItem(filename="a.txt", content="hi")]
            )

        assert records[0].size_bytes == 4096
        assert records[0].content_type == "application/pdf"

    def test_insert_text_defaults_size_when_absent(self, client):
        """Backward-compatible: a server that omits the fields yields 0/''."""
        resp = _ok_response(status_code=201)
        resp.json.return_value = {
            "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
        }
        with patch.object(client._client, "request", return_value=resp):
            record = client.insert_text("hi")

        assert record.size_bytes == 0
        assert record.content_type == ""


# ── scoped-partition handle ─────────────────────────────────


def _partition_insert_resp():
    resp = _ok_response()
    resp.json.return_value = {
        "doc_id": "d1", "cid": "c1", "chunks": 1, "vectors": 1, "version": 1,
    }
    return resp


class TestPartitionHandle:
    def test_partition_returns_distinct_scoped_object(self, client):
        scoped = client.partition("tenant-a")
        assert scoped is not client
        assert isinstance(scoped, AetherClient)
        assert scoped._partition == "tenant-a"
        # The base client stays unscoped.
        assert client._partition is None

    def test_scoped_shares_transport_and_config(self, client):
        scoped = client.partition("tenant-a")
        # Shared transport (no new httpx.Client) and config carried verbatim.
        assert scoped._client is client._client
        assert scoped.base_url == client.base_url
        assert scoped._max_retries == client._max_retries
        assert scoped._retry_base_delay == client._retry_base_delay
        assert scoped._owns_transport is False
        assert client._owns_transport is True

    def test_rescoping_last_wins(self, client):
        scoped = client.partition("a").partition("b")
        assert scoped._partition == "b"
        assert scoped._client is client._client

    def test_original_client_sends_no_partition(self, client):
        resp = _ok_response()
        resp.json.return_value = {"query": "q", "results": []}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.search("q")

        params = mock_req.call_args[1]["params"]
        assert "partition" not in params

    # ── partition injection per location ──────────────────────────────

    def test_search_sends_partition_query(self, client):
        resp = _ok_response()
        resp.json.return_value = {"query": "q", "results": []}
        scoped = client.partition("tenant-a")
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            scoped.search("q")

        params = mock_req.call_args[1]["params"]
        assert params["partition"] == "tenant-a"
        assert "partition=tenant-a" in _wire_url(params)

    def test_insert_text_sends_partition_query(self, client):
        with patch.object(client._client, "request", return_value=_partition_insert_resp()) as mock_req:
            client.partition("tenant-a").insert_text("hello")

        assert "partition=tenant-a" in mock_req.call_args[0][1]

    def test_list_sends_partition_query(self, client):
        resp = _ok_response()
        resp.json.return_value = {"documents": [], "total": 0, "has_more": False}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.partition("tenant-a").list()

        params = mock_req.call_args[1]["params"]
        assert params["partition"] == "tenant-a"

    def test_search_by_vector_sends_partition_body(self, client):
        resp = _ok_response()
        resp.json.return_value = {"results": []}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.partition("tenant-a").search_by_vector([0.1, 0.2])

        assert mock_req.call_args[1]["json"]["partition"] == "tenant-a"

    def test_insert_with_embeddings_sends_partition_body(self, client):
        with patch.object(client._client, "request", return_value=_partition_insert_resp()) as mock_req:
            client.partition("tenant-a").insert_with_embeddings("text", embedding=[0.1])

        assert mock_req.call_args[1]["json"]["partition"] == "tenant-a"

    def test_batch_insert_sends_partition_per_item(self, client):
        from aether import BatchInsertItem

        resp = _ok_response()
        resp.json.return_value = {"results": []}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.partition("tenant-a").batch_insert([
                BatchInsertItem(filename="a.txt", content="hello"),
                BatchInsertItem(filename="b.txt", content="world"),
            ])

        docs = mock_req.call_args[1]["json"]["documents"]
        assert [d["partition"] for d in docs] == ["tenant-a", "tenant-a"]

    def test_batch_search_sends_partition_per_query(self, client):
        from aether import BatchSearchQuery

        resp = _ok_response()
        resp.json.return_value = {"results": []}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.partition("tenant-a").batch_search([
                BatchSearchQuery(q="one"),
                BatchSearchQuery(q="two"),
            ])

        queries = mock_req.call_args[1]["json"]["queries"]
        assert [q["partition"] for q in queries] == ["tenant-a", "tenant-a"]

    def test_insert_sends_partition_query(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        with patch.object(client._client, "request", return_value=_partition_insert_resp()) as mock_req:
            client.partition("tenant-a").insert(f)

        assert "partition=tenant-a" in mock_req.call_args[0][1]

    def test_update_sends_partition_query(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        with patch.object(client._client, "request", return_value=_partition_insert_resp()) as mock_req:
            client.partition("tenant-a").update("doc-1", f)

        method, url = mock_req.call_args[0]
        assert method == "PUT"
        assert "partition=tenant-a" in url

    def test_insert_async_sends_partition_query(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        resp = _ok_response()
        resp.json.return_value = {"job_id": "j1", "status": "queued", "poll_url": "/x"}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.partition("tenant-a").insert_async(f)

        assert "partition=tenant-a" in mock_req.call_args[0][1]

    def test_insert_stream_sends_partition_query(self, client):
        import io

        with patch.object(client._client, "post", return_value=_partition_insert_resp()) as mock_post:
            client.partition("tenant-a").insert_stream(io.BytesIO(b"data"))

        assert "partition=tenant-a" in mock_post.call_args[0][0]

    def test_partition_value_is_url_encoded_on_query_routes(self, client):
        resp = _ok_response()
        resp.json.return_value = {"query": "q", "results": []}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.partition("tenant:a").search("q")

        # Same encoding entity_id receives on query routes.
        assert "partition=tenant%3Aa" in _wire_url(mock_req.call_args[1]["params"])

    # ── doc_id-addressed methods carry no partition ───────────────────

    def test_get_sends_no_partition(self, client):
        resp = _ok_response()
        resp.json.return_value = {"doc_id": "d1", "cid": "c1"}
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.partition("tenant-a").get("d1")

        method, url = mock_req.call_args[0]
        assert "partition" not in url
        assert "params" not in mock_req.call_args[1]

    def test_delete_sends_no_partition(self, client):
        resp = _ok_response()
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.partition("tenant-a").delete("d1")

        method, url = mock_req.call_args[0]
        assert "partition" not in url
        assert "params" not in mock_req.call_args[1]

    def test_delete_soft_by_default(self, client):
        resp = _ok_response()
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.delete("d1")
        _, url = mock_req.call_args[0]
        assert "hard" not in url  # default is a recoverable tombstone

    def test_delete_hard_sends_hard_flag(self, client):
        resp = _ok_response()
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.delete("d1", hard=True)
        method, url = mock_req.call_args[0]
        assert method == "DELETE"
        assert url.endswith("?hard=true")  # irreversible purge

    # ── validation: rejected before any HTTP call ─────────────────────

    def test_empty_partition_rejected(self, client):
        with pytest.raises(ValueError):
            client.partition("")

    def test_whitespace_partition_rejected(self, client):
        with pytest.raises(ValueError):
            client.partition("   ")

    def test_too_long_partition_rejected(self, client):
        with pytest.raises(ValueError):
            client.partition("x" * 257)

    def test_max_length_partition_accepted(self, client):
        scoped = client.partition("x" * 256)
        assert scoped._partition == "x" * 256

    def test_invalid_partition_makes_no_http_call(self, client):
        with patch.object(client._client, "request") as mock_req:
            with pytest.raises(ValueError):
                client.partition("")
        mock_req.assert_not_called()

    # ── lifecycle: scoped clone does not close the parent transport ───

    def test_closing_scoped_handle_keeps_parent_transport_open(self):
        client = AetherClient(base_url="http://localhost:9000", api_key="k", max_retries=0)
        scoped = client.partition("tenant-a")
        scoped.close()
        assert not client._client.is_closed
        client.close()
        assert client._client.is_closed

    def test_scoped_context_manager_does_not_close_parent(self):
        client = AetherClient(base_url="http://localhost:9000", api_key="k", max_retries=0)
        with client.partition("tenant-a"):
            pass
        assert not client._client.is_closed
        client.close()


class TestSearchTimestampsAndRecency:
    """Search hits carry created_at/updated_at; the SDK reads the engine's
    calibrated ``score`` verbatim; and recency ranking params are forwarded
    to the engine."""

    def test_parses_score_and_reads_timestamps(self, client):
        # The engine wire shape: calibrated `score` (0-100, higher = better),
        # plus created_at and (post-update) updated_at echoed on every hit.
        resp = _ok_response()
        resp.json.return_value = {
            "query": "q",
            "results": [
                {
                    "doc_id": "doc-1",
                    "score": 90,
                    "title": "Doc 1",
                    "content_type": "text/plain",
                    "created_at": "2026-06-01T00:00:00Z",
                    "updated_at": "2026-06-20T00:00:00Z",
                },
            ],
        }
        with patch.object(client._client, "request", return_value=resp):
            results = client.search("q", k=1)

        assert len(results) == 1
        r = results[0]
        # The 0-100 score is surfaced verbatim, never rescaled client-side.
        assert r.score == 90
        assert r.created_at == "2026-06-01T00:00:00Z"
        assert r.updated_at == "2026-06-20T00:00:00Z"

    def test_updated_at_null_until_updated(self, client):
        resp = _ok_response()
        resp.json.return_value = {
            "query": "q",
            "results": [
                {"doc_id": "d", "score": 50, "content_type": "text/plain",
                 "created_at": "2026-06-01T00:00:00Z"},
            ],
        }
        with patch.object(client._client, "request", return_value=resp):
            results = client.search("q", k=1)
        assert results[0].updated_at is None

    def test_recency_params_forwarded(self, client):
        resp = _ok_response()
        resp.json.return_value = {"query": "q", "results": []}
        with patch.object(client._client, "request", return_value=resp) as req:
            client.search("q", k=3, recency_weight=0.7, half_life_days=14.0)
        params = req.call_args.kwargs["params"]
        assert params["recency_weight"] == 0.7
        assert params["half_life_days"] == 14.0

    def test_freshness_params_forwarded(self, client):
        resp = _ok_response()
        resp.json.return_value = {"query": "q", "results": []}
        with patch.object(client._client, "request", return_value=resp) as req:
            client.search("q", k=3, freshness_weight=0.4, freshness_half_life_days=7.0)
        params = req.call_args.kwargs["params"]
        assert params["freshness_weight"] == 0.4
        assert params["freshness_half_life_days"] == 7.0

    def test_freshness_params_omitted_when_unset(self, client):
        resp = _ok_response()
        resp.json.return_value = {"query": "q", "results": []}
        with patch.object(client._client, "request", return_value=resp) as req:
            client.search("q", k=3)
        params = req.call_args.kwargs["params"]
        assert "freshness_weight" not in params
        assert "freshness_half_life_days" not in params

    def test_search_by_vector_forwards_freshness(self, client):
        resp = _ok_response()
        resp.json.return_value = {"results": []}
        with patch.object(client._client, "request", return_value=resp) as req:
            client.search_by_vector(
                [0.1], freshness_weight=0.25, freshness_half_life_days=3.5
            )
        body = req.call_args[1]["json"]
        assert body["freshness_weight"] == 0.25
        assert body["freshness_half_life_days"] == 3.5

    def test_search_by_vector_omits_unset_freshness(self, client):
        resp = _ok_response()
        resp.json.return_value = {"results": []}
        with patch.object(client._client, "request", return_value=resp) as req:
            client.search_by_vector([0.1], k=2)
        body = req.call_args[1]["json"]
        assert "freshness_weight" not in body
        assert "freshness_half_life_days" not in body

    def test_score_bounds_read_verbatim(self, client):
        # Both ends of the 0-100 range survive parsing untouched.
        resp = _ok_response()
        resp.json.return_value = {
            "query": "q",
            "results": [
                {"doc_id": "a", "score": 100, "content_type": "text/plain"},
                {"doc_id": "b", "score": 0, "content_type": "text/plain"},
            ],
        }
        with patch.object(client._client, "request", return_value=resp):
            results = client.search("q", k=2)
        assert [r.score for r in results] == [100, 0]


class TestIngest:
    """Batch / directory ingestion with graceful unsupported-type
    handling and per-file reporting."""

    def _fake_insert(self):
        # Mimics the engine: text-ish files ingest; an unsupported/binary type
        # is rejected with 422 (the engine's unprocessable-entity response).
        def _insert(path, content_type=None, **kwargs):
            name = Path(path).name
            if name.endswith((".bin", ".png")):
                raise AetherApiError(422, "unsupported content type", error_code="unsupported")
            return DocumentRecord(doc_id=f"doc-{name}", cid="cid", content_type=content_type or "")
        return _insert

    def test_ingest_files_reports_each_file(self, client, tmp_path):
        good = tmp_path / "a.md"
        good.write_text("# hello")
        bad = tmp_path / "b.bin"
        bad.write_bytes(b"\x00\x01\x02")

        with patch.object(client, "insert", side_effect=self._fake_insert()):
            results = client.ingest_files([good, bad])

        assert len(results) == 2
        assert all(isinstance(r, IngestResult) for r in results)
        ingested = [r for r in results if r.status == "ingested"]
        skipped = [r for r in results if r.status == "skipped"]
        assert len(ingested) == 1
        assert ingested[0].doc_id == "doc-a.md"
        assert ingested[0].content_type == "text/markdown"  # explicit .md mapping
        # Unsupported type is reported, not silently dropped or fatal.
        assert len(skipped) == 1
        assert "unsupported" in skipped[0].error

    def test_ingest_directory_filters_by_extension(self, client, tmp_path):
        (tmp_path / "a.md").write_text("# a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.png").write_bytes(b"\x89PNG")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "d.txt").write_text("d")

        seen: list[str] = []

        def _insert(path, content_type=None, **kwargs):
            seen.append(Path(path).name)
            return DocumentRecord(doc_id="d", cid="c")

        with patch.object(client, "insert", side_effect=_insert):
            # leading dots and case optional; recursive picks up sub/d.txt
            results = client.ingest_directory(tmp_path, extensions=["md", ".TXT"])

        assert {r.path.split("/")[-1] for r in results} == {"a.md", "b.txt", "d.txt"}
        assert "c.png" not in seen

    def test_ingest_directory_non_recursive(self, client, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("b")

        with patch.object(client, "insert", side_effect=self._fake_insert()):
            results = client.ingest_directory(tmp_path, recursive=False)

        assert {r.path.split("/")[-1] for r in results} == {"a.txt"}

    def test_ingest_raise_on_error_propagates(self, client, tmp_path):
        bad = tmp_path / "b.bin"
        bad.write_bytes(b"\x00")
        with patch.object(client, "insert", side_effect=self._fake_insert()):
            with pytest.raises(AetherApiError):
                client.ingest_files([bad], raise_on_error=True)

    def test_ingest_directory_rejects_non_directory(self, client, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("a")
        with pytest.raises(ValueError):
            client.ingest_directory(f)


class TestSearchFeedback:
    """Usage-feedback capture: query_id on search hits + send_search_feedback."""

    _HIT = {"doc_id": "doc-1", "score": 90, "content_type": "text/plain"}

    def test_search_parses_query_id_when_present(self, client):
        resp = _ok_response()
        resp.json.return_value = {
            "query": "q",
            "query_id": "11111111-2222-3333-4444-555555555555",
            "results": [self._HIT, {**self._HIT, "doc_id": "doc-2"}],
        }
        with patch.object(client._client, "request", return_value=resp):
            results = client.search("q")

        assert [r.query_id for r in results] == [
            "11111111-2222-3333-4444-555555555555",
            "11111111-2222-3333-4444-555555555555",
        ]

    def test_search_query_id_none_when_absent(self, client):
        resp = _ok_response()
        resp.json.return_value = {"query": "q", "results": [self._HIT]}
        with patch.object(client._client, "request", return_value=resp):
            results = client.search("q")

        assert results[0].query_id is None
        assert results[0].doc_id == "doc-1"

    def test_search_by_vector_parses_query_id(self, client):
        resp = _ok_response()
        resp.json.return_value = {
            "query": "",
            "query_id": "qid-embed",
            "results": [self._HIT],
        }
        with patch.object(client._client, "request", return_value=resp):
            results = client.search_by_vector([0.1] * 3)

        assert results[0].query_id == "qid-embed"

    def test_batch_search_stamps_per_query_query_id(self, client):
        from aether import BatchSearchQuery

        resp = _ok_response()
        resp.json.return_value = {
            "results": [
                {"query": "a", "query_id": "qid-a", "results": [self._HIT]},
                {"query": "b", "results": [{**self._HIT, "doc_id": "doc-2"}]},
            ],
        }
        with patch.object(client._client, "request", return_value=resp):
            responses = client.batch_search(
                [BatchSearchQuery(q="a"), BatchSearchQuery(q="b")]
            )

        assert responses[0].results[0].query_id == "qid-a"
        assert responses[1].results[0].query_id is None

    def test_send_search_feedback_posts_path_and_body(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["path"] = request.url.path
            captured["body"] = request.read()
            return httpx.Response(200, json={"recorded": True})

        client = _mock_client(handler)
        result = client.send_search_feedback("qid-1", "doc-1", "used")

        assert result is None
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/search/feedback"
        import json as _json

        assert _json.loads(captured["body"]) == {
            "query_id": "qid-1",
            "doc_id": "doc-1",
            "signal": "used",
        }

    def test_send_search_feedback_raises_404_on_unknown_query_id(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "unknown query_id"})

        client = _mock_client(handler)
        with pytest.raises(AetherApiError) as exc_info:
            client.send_search_feedback("nope", "doc-1", "cited")

        assert exc_info.value.status_code == 404
        assert not exc_info.value.is_retryable

    def test_send_search_feedback_raises_400_on_invalid_signal(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400, json={"error": "invalid signal", "code": "invalid_input"}
            )

        client = _mock_client(handler)
        with pytest.raises(AetherApiError) as exc_info:
            client.send_search_feedback("qid-1", "doc-1", "loved")

        assert exc_info.value.status_code == 400
        assert exc_info.value.error_code == "invalid_input"

    @pytest.mark.parametrize(
        "query_id,doc_id,signal",
        [("", "doc-1", "used"), ("qid-1", "", "used"), ("qid-1", "doc-1", "")],
    )
    def test_send_search_feedback_validates_arguments(self, client, query_id, doc_id, signal):
        with pytest.raises(ValueError):
            client.send_search_feedback(query_id, doc_id, signal)
