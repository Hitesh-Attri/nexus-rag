"""The vector store: persist chunks + their embeddings, and search them."""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
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


def search(embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    """Return the top_k chunks most similar to the query embedding (cosine).
    The one thing to understand here:

        <=> is cosine distance in pgvector - smaller = more alike (0 = identical direction, 2 = opposite).
        So we ORDER BY embedding <=> query ascending and take the first top_k.
        That ordering is exactly what the HNSW cosine index accelerates.
        1 - (embedding <=> %s) = cosine similarity - we flip distance into a 0-1 "how similar" score to return to the caller.
        That's why the query embedding appears twice (once for the score, once for the ordering).
        dict_row makes each row come back as a dict instead of a tuple.
    """
    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, source, content, metadata, "
            "       1 - (embedding <=> %s) AS similarity "
            "FROM documents "
            "ORDER BY embedding <=> %s "
            "LIMIT %s",
            (embedding, embedding, top_k),
        )
        return cur.fetchall()