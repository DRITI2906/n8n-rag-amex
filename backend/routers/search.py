"""GET/POST /search — pure retrieval, no LLM generation."""
from datetime import date

import asyncpg
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from config import get_settings
from db.connection import get_conn
from services.retriever import search, SearchResult

router = APIRouter()
_settings = get_settings()


class SearchRequest(BaseModel):
    question: str
    top_k: int = 5
    similarity_threshold: float | None = None
    source: str | None = None
    doc_id: str | None = None
    author: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    use_hybrid: bool = True


class ChunkResult(BaseModel):
    chunk_id: str
    chunk_text: str
    relevance_score: float
    doc_id: str
    doc_title: str | None
    source: str
    parent_heading: str


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    total: int


@router.post("", response_model=SearchResponse)
async def search_docs(
    req: SearchRequest,
    conn: asyncpg.Connection = Depends(get_conn),
):
    threshold = req.similarity_threshold or _settings.SIMILARITY_THRESHOLD
    results: list[SearchResult] = await search(
        conn=conn,
        question=req.question,
        top_k=req.top_k,
        similarity_threshold=threshold,
        source_filter=req.source,
        doc_id_filter=req.doc_id,
        author_filter=req.author,
        date_from=req.date_from,
        date_to=req.date_to,
        use_hybrid=req.use_hybrid,
    )

    return SearchResponse(
        query=req.question,
        results=[
            ChunkResult(
                chunk_id=r.chunk_id,
                chunk_text=r.chunk_text,
                relevance_score=r.relevance_score,
                doc_id=r.doc_id,
                doc_title=r.doc_title,
                source=r.source,
                parent_heading=r.parent_heading,
            )
            for r in results
        ],
        total=len(results),
    )
