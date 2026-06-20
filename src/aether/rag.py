"""RAG-specific helpers for the Aether SDK.

These utilities sit on top of :meth:`AetherClient.retrieve` to remove the
last-mile boilerplate of building an LLM prompt context: numbering sources,
choosing between the matched passage and the full document content, and
joining them into a single string.
"""

from __future__ import annotations

from typing import Sequence, Union

from .models import RetrievalResult, SearchResult

_Result = Union[RetrievalResult, SearchResult]

DEFAULT_TEMPLATE = "[Source {i}]\n{text}"


def format_context(
    results: Sequence[_Result],
    *,
    template: str = DEFAULT_TEMPLATE,
    separator: str = "\n\n",
    prefer_passage: bool = True,
) -> str:
    """Format retrieve()/search() results into an LLM-ready context string.

    The default output looks like::

        [Source 1]
        <matched passage 1>

        [Source 2]
        <matched passage 2>

    Args:
        results: Sequence of :class:`RetrievalResult` (from
            :meth:`AetherClient.retrieve`, which carries full ``content``) or
            :class:`SearchResult` (from :meth:`AetherClient.search`, which
            carries the matched ``passage``).
        template: Format string per source. Available placeholders:
            ``{i}`` (1-based source number), ``{doc_id}``, ``{title}``,
            ``{text}`` (passage or content), ``{score}``.
        separator: String joined between formatted sources.
        prefer_passage: When ``True`` (default), use the matched passage if
            present and fall back to ``content``. When ``False``, use
            ``content`` if present and fall back to ``passage``. Passages are
            the right choice for chunked long-form documents; content is
            fine for short single-chunk inserts.

    Returns:
        A single string ready to drop into an LLM system prompt.

    Example:
        >>> from aether import AetherClient, format_context
        >>> client = AetherClient()
        >>> results = client.retrieve("How many vacation days do I get?", k=3)
        >>> context = format_context(results)
    """
    chunks: list[str] = []
    for i, r in enumerate(results, start=1):
        passage = getattr(r, "passage", None) or ""
        # ``content`` (full document text) is present only on RetrievalResult,
        # which retrieve() returns; plain search() results carry just the passage.
        content = getattr(r, "content", None) or ""
        text = (passage or content) if prefer_passage else (content or passage)
        chunks.append(
            template.format(
                i=i,
                doc_id=r.doc_id,
                title=getattr(r, "title", None) or r.doc_id,
                text=text,
                score=r.score,
            )
        )
    return separator.join(chunks)
