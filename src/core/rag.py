"""Orchestration: ingestion."""

from __future__ import annotations

from typing import Any

from core.config import get_settings
from core.chunking import chunk_text
from core.embeddings import embed_texts, embed_query
from core.store import insert_chunks, search
from core.gateway import generate


PROMPT_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "provided. If the context does not contain the answer, say you don't know. "
    "Be concise."
)


def _build_context(chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[source: {c['source']}] {c['content']}" for c in chunks)


def answer_question(question: str, top_k: int | None = None) -> dict[str, Any]:
    """Retrieve context, have the gateway answer from it, return answer + sources."""
    chunks = retrieve(question, top_k)
    if not chunks:
        return {"answer": "I don't have any documents to answer from.",
                "provider": None, "model": None, "sources": []}
    user = f"Context:\n{_build_context(chunks)}\n\nQuestion: {question}"
    result = generate(PROMPT_SYSTEM, user)
    return {"answer": result["content"], "provider": result["provider"],
            "model": result["model"], "sources": chunks}

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