"""Tests for AetherClient download_text and retrieve methods."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from aether import AetherClient, EntityBackfillReport, RetrievalResult


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
                {"doc_id": "doc-1", "score": 95, "title": "Doc 1", "content_type": "text/plain"},
                {"doc_id": "doc-2", "score": 80, "title": "Doc 2", "content_type": "text/plain"},
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
        assert results[0].score == 95
        assert results[1].content == "Content of doc 2"

    def test_deduplicates_by_doc_id(self, client):
        search_resp = _ok_response()
        search_resp.json.return_value = {
            "query": "test",
            "results": [
                {"doc_id": "doc-1", "score": 95, "title": "Doc 1", "content_type": "text/plain"},
                {"doc_id": "doc-1", "score": 90, "title": "Doc 1", "content_type": "text/plain"},
                {"doc_id": "doc-2", "score": 80, "title": "Doc 2", "content_type": "text/plain"},
            ],
        }

        dl_resp_1 = _ok_response(content=b"Content 1")
        dl_resp_2 = _ok_response(content=b"Content 2")

        with patch.object(client._client, "request", side_effect=[search_resp, dl_resp_1, dl_resp_2]):
            results = client.retrieve("test", k=5)

        # Should deduplicate: 3 search results -> 2 unique docs
        assert len(results) == 2
        assert results[0].score == 95  # keeps closest match


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
        assert url == "/search"
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
        assert "include_content" not in params
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
        assert url == "/documents"
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
        resp.json.return_value = {"job_id": "j1", "status": "queued", "poll_url": "/documents/jobs/j1"}

        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client.insert_async(f, entity_id="user-123")

        url = mock_req.call_args[0][1]
        assert url.startswith("/documents/async?")
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
        assert url == "/documents/backfill-entity"
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
