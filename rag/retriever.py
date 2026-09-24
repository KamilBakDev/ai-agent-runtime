"""Hybrid (vector + BM25) retriever over the ingested Qdrant collection.

Fuses Qdrant's cosine-similarity ranking with a BM25 lexical ranking via reciprocal
rank fusion (RRF) -- simple, parameter-light, and doesn't require normalizing two
very different score scales.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from rank_bm25 import BM25Okapi

from agents.config import Settings, get_settings
from rag.embeddings import get_embeddings

RRF_K = 60


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float


class HybridRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = AsyncQdrantClient(url=self.settings.qdrant_url)
        self.embeddings = get_embeddings(self.settings)
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] = []
        self._loaded = False

    async def _load_corpus(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            points, _ = await self.client.scroll(
                collection_name=self.settings.qdrant_collection, limit=2000, with_payload=True
            )
        except Exception:
            points = []
        self._corpus = [p.payload for p in points if p.payload]
        tokenized = [c["text"].lower().split() for c in self._corpus]
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        await self._load_corpus()
        if not self._corpus:
            return []

        def key(payload: dict) -> tuple[str, int]:
            return (payload["source"], payload["chunk_index"])

        vector = (await self.embeddings.embed([query]))[0]
        response = await self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            limit=max(top_k * 3, 10),
            with_payload=True,
        )
        vector_rank = {
            key(hit.payload): i for i, hit in enumerate(response.points) if hit.payload
        }

        bm25_rank: dict[tuple[str, int], int] = {}
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query.lower().split())
            order = sorted(range(len(self._corpus)), key=lambda i: scores[i], reverse=True)
            bm25_rank = {key(self._corpus[i]): rank for rank, i in enumerate(order)}

        fused: dict[tuple[str, int], float] = {}
        for k, rank in vector_rank.items():
            fused[k] = fused.get(k, 0.0) + 1.0 / (RRF_K + rank)
        for k, rank in bm25_rank.items():
            fused[k] = fused.get(k, 0.0) + 1.0 / (RRF_K + rank)

        by_key = {key(c): c for c in self._corpus}
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        return [
            RetrievedChunk(
                text=by_key[k]["text"],
                source=by_key[k]["source"],
                chunk_index=by_key[k]["chunk_index"],
                score=score,
            )
            for k, score in top
            if k in by_key
        ]

    async def close(self) -> None:
        await self.client.close()
