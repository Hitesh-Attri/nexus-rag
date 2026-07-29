"""Typed configuration, loaded once from the environment / .env.

Same config-over-code instinct as the gateway: behaviour (chunk size, top_k,
which embedding model, where the gateway lives) is env, not hardcoded.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Neon Postgres connection string (postgresql://...?sslmode=require).
    # No default -> if it's missing, the app fails fast on startup.
    database_url: str

    # Where the resilient-llm-gateway is reachable (local dev: run it on :8080).
    gateway_url: str = "http://localhost:8080"

    # Embedding model. embedding_dim MUST equal the vector(N) column in the DB,
    # and the SAME model must embed both chunks and queries.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Chunking, in characters, with overlap so context isn't cut mid-thought.
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # How many chunks retrieval returns per query.
    top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()