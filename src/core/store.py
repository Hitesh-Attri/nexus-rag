"""The vector store: persist chunks + their embeddings, and search them."""

from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from core.db import get_pool


def insert_chunks(
    source: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: dict[str, Any] | None = None,
) -> int:
    """Insert one row per chunk. Returns the number of rows written."""
    meta = Jsonb(metadata or {})
    rows = [
        (source, chunk, meta, emb)
        for chunk, emb in zip(chunks, embeddings, strict=True)
    ]

    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO documents (source, content, metadata, embedding) "
            "VALUES (%s, %s, %s, %s)",
            rows,
        )
    return len(rows)