"""Pluggable embeddings providers.

``fake`` is a deterministic, dependency-free hash embedding used by default so RAG
is fully testable offline; ``ollama``/``openai`` call a real embedding model when
configured.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import httpx

from agents.config import Settings, get_settings


class EmbeddingsProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbeddings:
    """Deterministic hash-bucket embedding. Not semantically meaningful on its own --
    the hybrid retriever's BM25 half carries lexical relevance; this just needs to be
    stable and cheap so offline tests are deterministic."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


class OllamaEmbeddings:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        async with httpx.AsyncClient(timeout=60) as client:
            for text in texts:
                resp = await client.post(
                    f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": text}
                )
                resp.raise_for_status()
                out.append(resp.json()["embedding"])
        return out


class OpenAIEmbeddings:
    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        resp = await client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def get_embeddings(settings: Settings | None = None) -> EmbeddingsProvider:
    settings = settings or get_settings()
    provider = settings.embeddings_provider

    if provider == "fake":
        return FakeEmbeddings(dim=settings.embeddings_dim)
    if provider == "ollama":
        return OllamaEmbeddings(settings.embeddings_model, settings.ollama_base_url)
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDINGS_PROVIDER=openai requires OPENAI_API_KEY to be set")
        return OpenAIEmbeddings(settings.embeddings_model or "text-embedding-3-small", settings.openai_api_key)

    raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {provider!r}")
