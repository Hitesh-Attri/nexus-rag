"""POST /documents — ingestion pipeline"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.rag import ingest

router = APIRouter()


class IngestRequest(BaseModel):
    source: str = Field(min_length=1)   # a name/id for the document
    text: str = Field(min_length=1)     # the raw document text
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    source: str
    chunks_ingested: int


@router.post("/documents", response_model=IngestResponse)
def create_document(body: IngestRequest) -> IngestResponse:
    n = ingest(body.source, body.text, body.metadata)
    return IngestResponse(source=body.source, chunks_ingested=n)