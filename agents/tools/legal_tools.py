"""The two RAG-backed tools shared by the researcher agent, the HTTP tools API, and MCP."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from agents.config import get_settings
from rag.retriever import HybridRetriever


async def search_cases(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid vector+BM25 search over the ingested legal document corpus."""
    retriever = HybridRetriever()
    try:
        chunks = await retriever.retrieve(query, top_k=top_k)
    finally:
        await retriever.close()
    return [
        {"text": c.text, "source": c.source, "chunk_index": c.chunk_index, "score": c.score}
        for c in chunks
    ]


async def extract_citations(doc_ids: list[str]) -> list[dict]:
    """Fetch all indexed chunks belonging to the given source document filenames."""
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        points, _ = await client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="source", match=qmodels.MatchAny(any=doc_ids))]
            ),
            limit=1000,
            with_payload=True,
        )
    except Exception:
        return []
    finally:
        await client.close()

    results = [p.payload for p in points if p.payload]
    results.sort(key=lambda r: (r["source"], r["chunk_index"]))
    return results
