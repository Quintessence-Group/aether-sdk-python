"""Tests for input validation."""

import pytest
from aether import AetherClient


@pytest.fixture
def client():
    return AetherClient(base_url="http://localhost:9000", api_key="test-key")


def test_get_empty_doc_id(client):
    with pytest.raises(ValueError, match="doc_id cannot be empty"):
        client.get("")


def test_delete_empty_doc_id(client):
    with pytest.raises(ValueError, match="doc_id cannot be empty"):
        client.delete("")


def test_search_empty_query(client):
    with pytest.raises(ValueError, match="query cannot be empty"):
        client.search("")


def test_search_invalid_k(client):
    with pytest.raises(ValueError, match="k must be at least 1"):
        client.search("test", k=0)


def test_retrieve_invalid_k(client):
    with pytest.raises(ValueError, match="k must be at least 1"):
        client.retrieve("test", k=0)


def test_search_by_vector_empty_embedding(client):
    with pytest.raises(ValueError, match="embedding cannot be empty"):
        client.search_by_vector([])


def test_batch_insert_empty_documents(client):
    with pytest.raises(ValueError, match="documents cannot be empty"):
        client.batch_insert([])


def test_batch_search_empty_queries(client):
    with pytest.raises(ValueError, match="queries cannot be empty"):
        client.batch_search([])


def test_wait_for_job_empty_id(client):
    with pytest.raises(ValueError, match="job_id cannot be empty"):
        client.wait_for_job("")


def test_insert_invalid_chunk_size(client):
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        client.insert_text("hello", chunk_size=0)


def test_insert_invalid_overlap(client):
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        client.insert_text("hello", overlap=-1)
