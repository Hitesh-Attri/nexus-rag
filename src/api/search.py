"""POST /search - retrieval only, no generation.

Callers that do their own reasoning (the agent) want raw context to work with,
not a finished answer. Routing them through /query would trigger a second,
redundant LLM call inside this service.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.rag import retrieve

router = APIRouter()


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class SearchResult(BaseModel):
    id: int
    source: str
    content: str
    similarity: float


class SearchResponse(BaseModel):
    question: str
    results: list[SearchResult]


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    rows = retrieve(body.question, body.top_k)
    results = [
        SearchResult(id=r["id"], source=r["source"], content=r["content"],
                     similarity=r["similarity"])
        for r in rows
    ]
    return SearchResponse(question=body.question, results=results)