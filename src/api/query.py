"""POST /query — embed the question, retrieve the top-k matching chunks."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.rag import retrieve

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class Source(BaseModel):
    id: int
    source: str
    content: str
    similarity: float


class QueryResponse(BaseModel):
    question: str
    sources: list[Source]


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    results = retrieve(body.question, body.top_k)
    sources = [
        Source(id=r["id"], source=r["source"], content=r["content"], similarity=r["similarity"])
        for r in results
    ]
    return QueryResponse(question=body.question, sources=sources)