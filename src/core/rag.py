"""Orchestration: ingestion."""

from __future__ import annotations

from typing import Any

from core.chunking import chunk_text
from core.embeddings import embed_texts
from core.store import insert_chunks


def ingest(source: str, text: str, metadata: dict[str, Any] | None = None) -> int:
    """chunk -> embed -> store. Returns the number of chunks ingested."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    return insert_chunks(source, chunks, embeddings, metadata)