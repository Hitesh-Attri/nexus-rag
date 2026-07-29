"""Neon Postgres connection pool with pgvector registered.

register_vector teaches psycopg to adapt Python lists <-> the Postgres `vector`
type, so we pass embeddings as plain lists and read them back the same way.
It runs on every pooled connection (the `configure` hook).
"""

from __future__ import annotations

from functools import lru_cache

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from core.config import get_settings


@lru_cache
def get_pool() -> ConnectionPool:
    settings = get_settings()
    return ConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=5,
        configure=register_vector,  # runs per new connection
        open=True,
    )