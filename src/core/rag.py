"""Orchestration: ingestion."""

from __future__ import annotations

from typing import Any

from core.config import get_settings
from core.chunking import chunk_text
from core.embeddings import embed_texts, embed_query
from core.store import insert_chunks, search


def ingest(source: str, text: str, metadata: dict[str, Any] | None = None) -> int:
    """chunk -> embed -> store. Returns the number of chunks ingested."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    return insert_chunks(source, chunks, embeddings, metadata)

def retrieve(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Embed the question and return the top_k most similar chunks."""
    k = top_k or get_settings().top_k
    query_embedding = embed_query(question)
    return search(query_embedding, k)