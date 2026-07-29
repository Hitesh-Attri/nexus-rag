"""Split a document's text into overlapping chunks.

Character-based sliding window: simple, dependency-free, good enough for a first
cut. The overlap keeps a sentence that straddles a boundary present in BOTH
neighbouring chunks, so retrieval doesn't lose meaning at the seams.
"""

from __future__ import annotations

from core.config import get_settings


def chunk_text(text: str) -> list[str]:
    settings = get_settings()
    size = settings.chunk_size
    step = max(1, size - settings.chunk_overlap)  # how far the window advances

    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):  # this window reached the end — stop, no slivers
            break
        start += step
    return chunks