"""POST /query — embed the question, retrieve the top-k matching chunks."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.gateway import GatewayError
from core.rag import answer_question

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
    answer: str
    provider: str | None = None
    model: str | None = None
    sources: list[Source]


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    try:
        result = answer_question(body.question, body.top_k)
    except GatewayError as e:
        raise HTTPException(status_code=503, detail=f"generation failed: {e}") from e
    sources = [
        Source(id=r["id"], source=r["source"], content=r["content"], similarity=r["similarity"])
        for r in result["sources"]
    ]
    return QueryResponse(
        question=body.question,
        answer=result["answer"],
        provider=result["provider"],
        model=result["model"],
        sources=sources,
    )