"""Ingest PDF/Markdown documents into Qdrant: chunk, embed, upsert."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from pypdf import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from agents.config import Settings, get_settings
from rag.embeddings import get_embeddings

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(normalized):
        chunks.append(normalized[start : start + chunk_size])
        start += step
    return [c for c in chunks if c.strip()]


async def ensure_collection(client: AsyncQdrantClient, name: str, dim: int) -> None:
    collections = await client.get_collections()
    if name not in {c.name for c in collections.collections}:
        await client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )


async def ingest_directory(
    directory: Path, settings: Settings | None = None, collection: str | None = None
) -> int:
    """Ingest every supported file in ``directory`` and return the number of chunks written."""
    settings = settings or get_settings()
    collection = collection or settings.qdrant_collection
    embeddings = get_embeddings(settings)
    client = AsyncQdrantClient(url=settings.qdrant_url)

    try:
        await ensure_collection(client, collection, settings.embeddings_dim)

        points: list[qmodels.PointStruct] = []
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            chunks = chunk_text(_read_text(path))
            if not chunks:
                continue
            vectors = await embeddings.embed(chunks)
            for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path.name}-{i}"))
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={"text": chunk, "source": path.name, "chunk_index": i},
                    )
                )

        if points:
            await client.upsert(collection_name=collection, points=points)
        return len(points)
    finally:
        await client.close()


async def _main() -> None:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_docs")
    count = await ingest_directory(directory)
    print(f"Ingested {count} chunks from {directory}")


if __name__ == "__main__":
    asyncio.run(_main())
