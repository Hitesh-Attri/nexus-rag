"""Local text embeddings via fastembed (BAAI/bge-small-en-v1.5, 384-dim).

fastembed runs a small ONNX model on CPU: no API key, no network call at embed
time — only a one-time model download on first use. The model is loaded once
(lru_cache) and reused for the process lifetime.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

from core.config import get_settings


@lru_cache
def get_embedder() -> TextEmbedding:
    settings = get_settings()
    return TextEmbedding(model_name=settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts (chunks). Returns one 384-float vector per text."""
    model = get_embedder()
    return [vec.tolist() for vec in model.embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a single query. Same model as chunks -> same meaning-space."""
    return embed_texts([text])[0]