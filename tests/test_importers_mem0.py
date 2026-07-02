"""Tests for the Mem0 → Aether importer (``aether.importers.mem0``).

Mocked at the **same transport layer as the existing client/memory tests** —
``client._client.request`` (the underlying httpx ``Client``) — with the *real*
raw client and the *real* ``Memory`` facade running underneath, so these tests
assert the actual HTTP writes the import produces. One test additionally
patches the ``Memory`` class itself to pin the facade-level call arguments
documented in ``docs/importers/mem0-mapping.md``.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from aether import AetherClient, Mem0ImportReport, import_mem0

DOCS = "/v1/documents"
ENTITIES = "/v1/memory/entities"
RELS = "/v1/memory/relationships"


# ── transport stubs (same style as tests/test_memory.py) ─────────────


def _resp(*, json_data=None, status_code=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.reason_phrase = "OK" if resp.is_success else "Error"
    resp.headers = {}
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


class Transport:
    """Records every ``request(method, url, **kwargs)`` and dispatches a
    scripted response keyed by ``(method, path)`` — the path **excluding the
    query string**, since the importer varies query params per item. Per-key
    responses pop off a queue when a list is supplied, else the single value
    is reused.
    """

    def __init__(self, routes):
        self.routes = routes
        self.calls = []  # list of (method, url, kwargs)

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        key = (method, url.split("?", 1)[0])
        if key not in self.routes:
            raise AssertionError(f"unexpected request: {method} {url}")
        value = self.routes[key]
        if isinstance(value, list):
            return value.pop(0)
        return value

    def calls_to(self, method, path):
        return [
            c
            for c in self.calls
            if c[0] == method and c[1].split("?", 1)[0] == path
        ]


def _client(transport):
    client = AetherClient(base_url="http://localhost:9000", api_key="k", max_retries=0)
    patch.object(client._client, "request", side_effect=transport).start()
    return client


@pytest.fixture(autouse=True)
def _stop_patches():
    yield
    patch.stopall()


def _query(url):
    """Parsed query params of a request URL, single-valued."""
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def _insert_resp(doc_id="doc-1"):
    return _resp(json_data={
        "doc_id": doc_id,
        "cid": "cid-1",
        "chunks": 1,
        "vectors": 1,
        "version": 1,
        "created_at": "2026-07-01T00:00:00Z",
    })


def _entity_resp(memory_entity_id):
    return _resp(json_data={
        "memory_entity_id": memory_entity_id,
        "entity_id": "mem0:user:alice",
        "entity_type": "mem0_node",
    })


def _rel_resp(relationship_id="rel-1"):
    return _resp(json_data={
        "relationship_id": relationship_id,
        "entity_id": "mem0:user:alice",
        "from_entity_id": "a",
        "to_entity_id": "b",
        "relationship_type": "likes",
    })


def _mem(id="m-1", text="likes pizza", **kw):
    return {"id": id, "memory": text, **kw}


# ── input shapes ─────────────────────────────────────────────────────


class TestInputShapes:
    def test_bare_list_shape(self):
        t = Transport({("POST", DOCS): [_insert_resp("d1"), _insert_resp("d2")]})
        report = import_mem0(_client(t), [
            _mem("m-1", "likes pizza", user_id="alice"),
            _mem("m-2", "vegetarian", user_id="alice"),
        ])
        assert isinstance(report, Mem0ImportReport)
        assert report.memories_imported == 2
        assert report.skipped == 0
        assert report.owners == ["mem0:user:alice"]
        assert len(t.calls_to("POST", DOCS)) == 2
        for method, url, _ in t.calls_to("POST", DOCS):
            assert _query(url)["entity_id"] == "mem0:user:alice"

    def test_results_wrapper_shape(self):
        t = Transport({("POST", DOCS): _insert_resp()})
        report = import_mem0(
            _client(t), {"results": [_mem(user_id="alice")]}
        )
        assert report.memories_imported == 1
        assert _query(t.calls[0][1])["entity_id"] == "mem0:user:alice"

    def test_file_path_input_str_and_path(self, tmp_path):
        export = {"results": [_mem(user_id="alice")]}
        path = tmp_path / "export.json"
        path.write_text(json.dumps(export))
        for data in (str(path), Path(path)):
            t = Transport({("POST", DOCS): _insert_resp()})
            report = import_mem0(_client(t), data)
            assert report.memories_imported == 1

    def test_unrecognized_export_raises(self):
        t = Transport({})
        with pytest.raises(ValueError, match="unrecognized Mem0 export"):
            import_mem0(_client(t), {"memories": []})
        with pytest.raises(ValueError, match="unsupported Mem0 export type"):
            import_mem0(_client(t), 42)
        assert t.calls == []

    def test_invalid_on_error_and_prefix_raise(self):
        t = Transport({})
        with pytest.raises(ValueError, match="on_error"):
            import_mem0(_client(t), [], on_error="explode")
        with pytest.raises(ValueError, match="entity_prefix"):
            import_mem0(_client(t), [], entity_prefix="  ")
        assert t.calls == []


# ── owner grouping ───────────────────────────────────────────────────


class TestOwnerGrouping:
    def test_user_agent_run_default_precedence(self):
        t = Transport({("POST", DOCS): [_insert_resp() for _ in range(5)]})
        report = import_mem0(_client(t), [
            _mem("m-1", "a", user_id="alice"),
            _mem("m-2", "b", agent_id="bot-1"),
            _mem("m-3", "c", run_id="run-9"),
            _mem("m-4", "d"),
            # user_id wins over agent_id/run_id when several are present
            _mem("m-5", "e", user_id="alice", agent_id="bot-1", run_id="run-9"),
        ])
        owners = [_query(url)["entity_id"] for _, url, _ in t.calls]
        assert owners == [
            "mem0:user:alice",
            "mem0:agent:bot-1",
            "mem0:run:run-9",
            "mem0:default",
            "mem0:user:alice",
        ]
        assert report.owners == sorted(
            {"mem0:user:alice", "mem0:agent:bot-1", "mem0:run:run-9", "mem0:default"}
        )

    def test_entity_prefix_override(self):
        t = Transport({("POST", DOCS): _insert_resp()})
        import_mem0(_client(t), [_mem(user_id="alice")], entity_prefix="legacy")
        assert _query(t.calls[0][1])["entity_id"] == "legacy:user:alice"

    def test_integer_owner_id_is_accepted_not_rehomed(self):
        t = Transport({("POST", DOCS): [_insert_resp() for _ in range(2)]})
        report = import_mem0(_client(t), [
            _mem("m-1", "a", user_id=123),
            # bool is an int subclass but is not an owner id
            _mem("m-2", "b", user_id=True),
        ])
        owners = [_query(url)["entity_id"] for _, url, _ in t.calls]
        assert owners == ["mem0:user:123", "mem0:default"]
        assert report.memories_imported == 2


# ── metadata mapping ─────────────────────────────────────────────────


class TestMetadataMapping:
    def test_metadata_merge_categories_and_tag_mirror(self):
        t = Transport({("POST", DOCS): _insert_resp()})
        import_mem0(_client(t), [{
            "id": "m-1",
            "memory": "likes pizza",
            "user_id": "alice",
            "created_at": "2025-01-02T03:04:05Z",
            "updated_at": "2025-02-03T04:05:06Z",
            "categories": ["food", "travel plans"],
            "score": 0.87,  # ignored
            "metadata": {
                "mood": "happy",
                "source": "should-lose",   # importer key wins
                "nested": {"x": 1},        # dropped: non-scalar
                "csv": "a,b",              # dropped: comma in value
                "bad,key": "v",            # dropped: comma in key
            },
        }])
        _, url, kwargs = t.calls[0]
        q = _query(url)
        meta = json.loads(q["metadata"])
        assert meta["mood"] == "happy"
        assert meta["source"] == "mem0"
        assert meta["mem0_id"] == "m-1"
        assert meta["mem0_created_at"] == "2025-01-02T03:04:05Z"
        assert meta["mem0_updated_at"] == "2025-02-03T04:05:06Z"
        assert meta["mem0_categories"] == "food|travel plans"
        assert meta["category_food"] is True
        assert meta["category_travel_plans"] is True
        assert "nested" not in meta and "csv" not in meta and "bad,key" not in meta
        # metadata is also mirrored into legacy key:value tags by the facade
        tags = q["tags"].split(",")
        assert "source:mem0" in tags
        assert "category_food:True" in tags
        # the memory text itself is the request body
        assert kwargs["content"] == b"likes pizza"

    def test_dropped_pairs_are_counted(self):
        t = Transport({("POST", DOCS): _insert_resp()})
        report = import_mem0(_client(t), [
            _mem(metadata={"ok": 1, "nested": [1, 2], "b,ad": "x"}),
        ])
        assert report.memories_imported == 1
        assert report.metadata_pairs_dropped == 2


# ── graph relations ──────────────────────────────────────────────────


class TestGraphRelations:
    def test_relations_create_entities_and_edges(self):
        t = Transport({
            ("POST", DOCS): _insert_resp(),
            ("POST", ENTITIES): [_entity_resp("node-alice"), _entity_resp("node-pizza")],
            ("POST", RELS): _rel_resp(),
        })
        report = import_mem0(_client(t), {
            "results": [_mem(user_id="alice")],
            "relations": [
                {"source": "alice", "relationship": "likes", "target": "pizza"},
            ],
        })
        assert report.entities_created == 2
        assert report.relationships_created == 1

        ent_calls = t.calls_to("POST", ENTITIES)
        # relations without their own owner ids land under the single memory owner
        assert all(c[2]["params"]["entity_id"] == "mem0:user:alice" for c in ent_calls)
        bodies = [c[2]["json"] for c in ent_calls]
        assert bodies[0] == {
            "entity_type": "mem0_node",
            "memory_entity_id": "alice",
            "display_name": "alice",
        }
        assert bodies[1]["memory_entity_id"] == "pizza"

        rel_call = t.calls_to("POST", RELS)[0]
        assert rel_call[2]["params"]["entity_id"] == "mem0:user:alice"
        assert rel_call[2]["json"] == {
            "from_entity_id": "node-alice",
            "to_entity_id": "node-pizza",
            "relationship_type": "likes",
        }

    def test_destination_alias_and_explicit_types(self):
        t = Transport({
            ("POST", ENTITIES): [_entity_resp("n-a"), _entity_resp("n-b")],
            ("POST", RELS): _rel_resp(),
        })
        report = import_mem0(_client(t), {
            "results": [],
            "relations": [{
                "source": "alice",
                "source_type": "person",
                "relationship": "works_at",
                "destination": "acme",
                "destination_type": "company",
            }],
        })
        assert report.relationships_created == 1
        bodies = [c[2]["json"] for c in t.calls_to("POST", ENTITIES)]
        assert bodies[0]["entity_type"] == "person"
        assert bodies[1]["entity_type"] == "company"
        # no memories → relations land in the default bucket
        assert t.calls[0][2]["params"]["entity_id"] == "mem0:default"

    def test_nodes_deduplicated_across_relations(self):
        t = Transport({
            ("POST", ENTITIES): [
                _entity_resp("n-alice"), _entity_resp("n-pizza"), _entity_resp("n-rome"),
            ],
            ("POST", RELS): [_rel_resp("r-1"), _rel_resp("r-2")],
        })
        report = import_mem0(_client(t), {
            "results": [],
            "relations": [
                {"source": "alice", "relationship": "likes", "target": "pizza"},
                {"source": "alice", "relationship": "visited", "target": "rome"},
            ],
        })
        assert report.entities_created == 3  # alice upserted once
        assert report.relationships_created == 2
        assert len(t.calls_to("POST", ENTITIES)) == 3

    def test_relations_fall_back_to_default_when_owners_ambiguous(self):
        t = Transport({
            ("POST", DOCS): [_insert_resp(), _insert_resp()],
            ("POST", ENTITIES): [_entity_resp("n-a"), _entity_resp("n-b")],
            ("POST", RELS): _rel_resp(),
        })
        import_mem0(_client(t), {
            "results": [_mem("m-1", "a", user_id="alice"), _mem("m-2", "b", user_id="bob")],
            "relations": [{"source": "a", "relationship": "r", "target": "b"}],
        })
        assert t.calls_to("POST", ENTITIES)[0][2]["params"]["entity_id"] == "mem0:default"

    def test_malformed_relation_skipped_with_reason(self):
        t = Transport({
            ("POST", ENTITIES): [_entity_resp("n-a"), _entity_resp("n-b")],
            ("POST", RELS): _rel_resp(),
        })
        report = import_mem0(_client(t), {
            "results": [],
            "relations": [
                {"source": "alice", "relationship": "likes"},  # no target
                "not-an-object",
                {"source": "alice", "relationship": "likes", "target": "pizza"},
            ],
        })
        assert report.relationships_created == 1
        assert report.skipped == 2
        assert report.skips[0][0] == "relations[0]"
        assert "target" in report.skips[0][1]
        assert report.skips[1] == ("relations[1]", "relation is not an object")


# ── dry run ──────────────────────────────────────────────────────────


class TestDryRun:
    def test_dry_run_makes_no_http_calls_but_counts(self):
        t = Transport({})  # any request would raise AssertionError
        report = import_mem0(_client(t), {
            "results": [
                _mem("m-1", "a", user_id="alice"),
                _mem("m-2", "b", user_id="bob"),
                {"id": "m-3"},  # malformed: no text
            ],
            "relations": [
                {"source": "alice", "relationship": "likes", "target": "pizza"},
                {"source": "alice", "relationship": "visited", "target": "rome"},
            ],
        }, dry_run=True)
        assert t.calls == []
        assert report.dry_run is True
        assert report.memories_imported == 2
        assert report.entities_created == 3
        assert report.relationships_created == 2
        assert report.skipped == 1
        assert report.skips == [("m-3", "missing or empty 'memory' text")]
        assert report.owners == ["mem0:default", "mem0:user:alice", "mem0:user:bob"]

    def test_dry_run_still_validates(self):
        t = Transport({})
        with pytest.raises(ValueError):
            import_mem0(_client(t), [42], dry_run=True, on_error="raise")
        assert t.calls == []

    def test_whitespace_node_type_falls_back_and_matches_live_counts(self):
        rel = {
            "source": "alice",
            "source_type": "   ",
            "relationship": "likes",
            "target": "pizza",
            "target_type": "\t",
        }
        dry = import_mem0(
            _client(Transport({})),
            {"results": [], "relations": [rel]},
            dry_run=True,
        )
        t = Transport({
            ("POST", ENTITIES): [_entity_resp("e-1"), _entity_resp("e-2")],
            ("POST", RELS): _rel_resp(),
        })
        live = import_mem0(_client(t), {"results": [], "relations": [rel]})
        for report in (dry, live):
            assert report.entities_created == 2
            assert report.relationships_created == 1
            assert report.skipped == 0
        # blank types fell back to the default node type on the wire
        sent_types = [c[2]["json"]["entity_type"] for c in t.calls_to("POST", ENTITIES)]
        assert sent_types == ["mem0_node", "mem0_node"]


# ── error handling ───────────────────────────────────────────────────


class TestErrorHandling:
    BAD_ITEMS = [
        42,                                   # not an object
        {"id": "m-2"},                        # missing text
        {"id": "m-3", "memory": "   "},       # empty text
        {"id": "m-4", "memory": "ok", "metadata": "nope"},   # metadata not a dict
        {"id": "m-5", "memory": "ok", "categories": "nope"}, # categories not a list
        _mem("m-6", "fine", user_id="alice"),  # the one good item
    ]

    def test_skip_records_item_id_and_reason(self):
        t = Transport({("POST", DOCS): _insert_resp()})
        report = import_mem0(_client(t), list(self.BAD_ITEMS))
        assert report.memories_imported == 1
        assert report.skipped == 5
        assert report.skips == [
            ("memories[0]", "memory item is not an object"),
            ("m-2", "missing or empty 'memory' text"),
            ("m-3", "missing or empty 'memory' text"),
            ("m-4", "'metadata' is not an object"),
            ("m-5", "'categories' is not a list"),
        ]
        assert len(t.calls) == 1  # only the good item hit the wire

    def test_on_error_raise_stops_before_any_write(self):
        t = Transport({})
        with pytest.raises(ValueError, match=r"memories\[0\]"):
            import_mem0(_client(t), list(self.BAD_ITEMS), on_error="raise")
        assert t.calls == []

    def test_http_failure_is_skipped_with_reason(self):
        t = Transport({("POST", DOCS): [
            _resp(json_data={"error": "boom"}, status_code=400),
            _insert_resp(),
        ]})
        report = import_mem0(_client(t), [
            _mem("m-1", "first", user_id="alice"),
            _mem("m-2", "second", user_id="alice"),
        ])
        assert report.memories_imported == 1
        assert report.skipped == 1
        ref, reason = report.skips[0]
        assert ref == "m-1"
        assert reason.startswith("remember failed:")

    def test_http_failure_raises_when_on_error_raise(self):
        from aether import AetherApiError

        t = Transport({("POST", DOCS): _resp(json_data={"error": "boom"}, status_code=400)})
        with pytest.raises(AetherApiError):
            import_mem0(
                _client(t),
                [_mem("m-1", "first", user_id="alice")],
                on_error="raise",
            )


# ── facade-level contract (end-to-end-ish) ───────────────────────────


class TestFacadeCalls:
    def test_facade_methods_called_with_documented_arguments(self):
        """The importer's writes are exactly the documented facade calls:
        one ``Memory`` per owner, ``remember(text, metadata=...)`` per memory,
        ``upsert_entity``/``relate`` per relation."""
        facades = {}

        def _factory(entity_id, *, client):
            mem = MagicMock(name=f"Memory({entity_id})")
            mem.upsert_entity.side_effect = lambda *a, **kw: MagicMock(
                memory_entity_id=f"id-{kw['memory_entity_id']}"
            )
            facades[entity_id] = mem
            return mem

        with patch("aether.importers.mem0.Memory", side_effect=_factory):
            client = AetherClient(base_url="http://localhost:9000", api_key="k")
            report = import_mem0(client, {
                "results": [{
                    "id": "m-1",
                    "memory": "likes pizza",
                    "user_id": "alice",
                    "created_at": "2025-01-02T03:04:05Z",
                    "categories": ["food"],
                    "metadata": {"mood": "happy"},
                }],
                "relations": [
                    {"source": "alice", "relationship": "likes", "target": "pizza"},
                ],
            })

        assert list(facades) == ["mem0:user:alice"]
        mem = facades["mem0:user:alice"]
        mem.remember.assert_called_once_with(
            "likes pizza",
            metadata={
                "mood": "happy",
                "mem0_categories": "food",
                "category_food": True,
                "source": "mem0",
                "mem0_id": "m-1",
                "mem0_created_at": "2025-01-02T03:04:05Z",
            },
        )
        assert mem.upsert_entity.call_count == 2
        mem.upsert_entity.assert_any_call(
            "mem0_node", memory_entity_id="alice", display_name="alice"
        )
        mem.upsert_entity.assert_any_call(
            "mem0_node", memory_entity_id="pizza", display_name="pizza"
        )
        mem.relate.assert_called_once_with("id-alice", "id-pizza", "likes")
        assert report.memories_imported == 1
        assert report.entities_created == 2
        assert report.relationships_created == 1
        assert report.owners == ["mem0:user:alice"]
