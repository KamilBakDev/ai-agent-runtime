"""Turn retrieved chunks into a numbered, citable context block."""

from __future__ import annotations

from rag.retriever import RetrievedChunk


def format_context_with_citations(chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    """Return (context_text_with_[n]_markers, citation_records)."""
    lines = []
    citations = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] {c.text}")
        citations.append(
            {
                "id": i,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "score": c.score,
                "text": c.text[:280],
            }
        )
    return "\n\n".join(lines), citations
