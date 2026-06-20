"""Tests for aether.rag.format_context."""

from aether import RetrievalResult, SearchResult, format_context


def _r(*, doc_id="d1", score=90, content="full body", passage=None, title=None):
    return RetrievalResult(
        doc_id=doc_id,
        score=score,
        content=content,
        passage=passage,
        title=title,
    )


class TestFormatContextDefaults:
    def test_empty_returns_empty_string(self):
        assert format_context([]) == ""

    def test_default_template_numbers_sources_from_one(self):
        results = [_r(doc_id="a"), _r(doc_id="b")]
        out = format_context(results)
        assert "[Source 1]" in out
        assert "[Source 2]" in out
        assert "[Source 0]" not in out

    def test_default_separator_is_blank_line(self):
        results = [_r(doc_id="a", content="alpha"), _r(doc_id="b", content="beta")]
        out = format_context(results)
        assert out == "[Source 1]\nalpha\n\n[Source 2]\nbeta"

    def test_prefers_passage_over_content_by_default(self):
        # Long-form docs: passage is the matched chunk; content is the whole doc.
        results = [_r(content="100-page handbook", passage="the matched paragraph")]
        out = format_context(results)
        assert "the matched paragraph" in out
        assert "100-page handbook" not in out

    def test_falls_back_to_content_when_passage_missing(self):
        # Short single-chunk inserts (the quickstart shape) have no separate passage.
        results = [_r(content="short doc", passage=None)]
        out = format_context(results)
        assert "short doc" in out


class TestFormatContextOptions:
    def test_prefer_passage_false_uses_content(self):
        results = [_r(content="full body", passage="chunk")]
        out = format_context(results, prefer_passage=False)
        assert "full body" in out
        assert "chunk" not in out

    def test_custom_template_with_title_and_score(self):
        results = [_r(doc_id="d1", title="PTO policy", score=87, content="20 days")]
        out = format_context(
            results,
            template="<{title} | s={score}>\n{text}",
        )
        assert out == "<PTO policy | s=87>\n20 days"

    def test_custom_template_falls_back_to_doc_id_when_title_missing(self):
        results = [_r(doc_id="d1", title=None, content="body")]
        out = format_context(results, template="[{title}] {text}")
        assert out == "[d1] body"

    def test_custom_separator(self):
        results = [_r(content="a"), _r(content="b")]
        out = format_context(results, separator=" --- ")
        assert " --- " in out
        assert "\n\n" not in out


class TestFormatContextResultTypes:
    def test_accepts_search_result_with_passage(self):
        # search() returns SearchResult carrying the matched passage (no content).
        sr = SearchResult(doc_id="d1", score=90, passage="matched passage")
        out = format_context([sr])
        assert "matched passage" in out

    def test_search_result_without_passage_renders_empty_text(self):
        # A SearchResult with no passage has no text to render.
        sr = SearchResult(doc_id="d1", score=90, passage=None)
        out = format_context([sr])
        assert "[Source 1]\n" in out
        assert out.endswith("\n")
