"""
POST /query — full RAG chain: retrieve chunks → call LLM → stream answer.
Supports SSE streaming and non-streaming modes.
Logs every query to the query_logs table.
Also exposes POST /query/feedback for thumbs-up/down.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import date

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import get_settings
from db.connection import get_conn
from services.retriever import search, SearchResult
from services.llm import generate

router = APIRouter()
_settings = get_settings()


# ─── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_local_llm: bool | None = None   # None = use LLM_PROVIDER env default
    provider: str | None = None         # override: "ollama" | "anthropic" | "openai"
    stream: bool = True
    user_id: str | None = None
    source: str | None = None
    doc_id: str | None = None
    date_from: date | None = None
    date_to: date | None = None


class SourceRef(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str | None
    source: str
    parent_heading: str
    relevance_score: float


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    sources: list[SourceRef]
    retrieval_time_ms: int
    generation_time_ms: int
    model_used: str


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int          # 1 = thumbs up, -1 = thumbs down
    user_id: str | None = None
    comment: str | None = None


# ─── Main /query endpoint ─────────────────────────────────────────────────────

@router.post("")
async def query_docs(
    req: QueryRequest,
    conn: asyncpg.Connection = Depends(get_conn),
):
    # 1. Retrieval
    t0 = time.monotonic()
    chunks: list[SearchResult] = await search(
        conn=conn,
        question=req.question,
        top_k=req.top_k,
        similarity_threshold=_settings.SIMILARITY_THRESHOLD,
        source_filter=req.source,
        doc_id_filter=req.doc_id,
        date_from=req.date_from,
        date_to=req.date_to,
    )
    retrieval_ms = int((time.monotonic() - t0) * 1000)

    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "chunk_text": c.chunk_text,
            "doc_id": c.doc_id,
            "doc_title": c.doc_title,
            "source": c.source,
            "parent_heading": c.parent_heading,
            "relevance_score": c.relevance_score,
        }
        for c in chunks
    ]

    # Determine provider
    if req.provider:
        provider = req.provider
    elif req.use_local_llm is False:
        provider = "anthropic" if _settings.ANTHROPIC_API_KEY else "openai"
    elif req.use_local_llm is True:
        provider = "ollama"
    else:
        provider = _settings.LLM_PROVIDER

    model_used = _resolve_model_name(provider)

    # 2. Generation — streaming
    if req.stream:
        return EventSourceResponse(
            _stream_response(
                conn=conn,
                question=req.question,
                chunks=chunk_dicts,
                retrieval_ms=retrieval_ms,
                provider=provider,
                model_used=model_used,
                user_id=req.user_id,
            )
        )

    # 2b. Non-streaming
    t1 = time.monotonic()
    result = await generate(req.question, chunk_dicts, provider=provider, stream=False)
    generation_ms = int((time.monotonic() - t1) * 1000)

    answer = result if isinstance(result, str) else ""
    fallback = not answer

    query_id = await _log_query(
        conn=conn,
        question=req.question,
        answer=answer,
        sources=chunk_dicts,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        model_used=model_used,
        top_k=req.top_k,
        user_id=req.user_id,
    )

    if fallback:
        answer = "Could not generate an answer — here are the most relevant document sections found."

    return QueryResponse(
        query_id=query_id,
        question=req.question,
        answer=answer,
        sources=[
            SourceRef(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                doc_title=c["doc_title"],
                source=c["source"],
                parent_heading=c["parent_heading"],
                relevance_score=c["relevance_score"],
            )
            for c in chunk_dicts
        ],
        retrieval_time_ms=retrieval_ms,
        generation_time_ms=generation_ms,
        model_used=model_used,
    )


# ─── SSE streaming ────────────────────────────────────────────────────────────

async def _stream_response(
    conn, question, chunks, retrieval_ms, provider, model_used, user_id
):
    """Yields SSE events: token by token, then a final 'done' event with metadata."""
    query_id = str(uuid.uuid4())

    # Send retrieval metadata first
    yield {
        "event": "metadata",
        "data": json.dumps({
            "query_id": query_id,
            "sources": chunks,
            "retrieval_time_ms": retrieval_ms,
            "model_used": model_used,
        }),
    }

    t1 = time.monotonic()
    full_answer = ""

    try:
        stream = await generate(question, chunks, provider=provider, stream=True)
        if hasattr(stream, "__aiter__"):
            async for token in stream:
                full_answer += token
                yield {"event": "token", "data": token}
        else:
            full_answer = stream or ""
            yield {"event": "token", "data": full_answer}
    except Exception as e:
        full_answer = "Could not generate an answer — here are the most relevant document sections found."
        yield {"event": "token", "data": full_answer}

    generation_ms = int((time.monotonic() - t1) * 1000)

    await _log_query(
        conn=conn,
        question=question,
        answer=full_answer,
        sources=chunks,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        model_used=model_used,
        top_k=len(chunks),
        user_id=user_id,
        query_id=query_id,
    )

    yield {"event": "done", "data": json.dumps({"generation_time_ms": generation_ms})}


# ─── Feedback endpoint ────────────────────────────────────────────────────────

@router.post("/feedback")
async def submit_feedback(
    req: FeedbackRequest,
    conn: asyncpg.Connection = Depends(get_conn),
):
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be 1 (up) or -1 (down)")

    await conn.execute(
        """
        INSERT INTO feedback (query_id, user_id, rating, comment)
        VALUES ($1::uuid, $2, $3, $4)
        """,
        req.query_id, req.user_id, req.rating, req.comment,
    )
    return {"status": "recorded"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_model_name(provider: str) -> str:
    s = _settings
    if provider == "anthropic":
        return s.ANTHROPIC_MODEL
    if provider == "openai":
        return s.OPENAI_CHAT_MODEL
    return f"ollama/{s.OLLAMA_MODEL}"


async def _log_query(
    conn, question, answer, sources, retrieval_ms, generation_ms,
    model_used, top_k, user_id, query_id: str | None = None,
) -> str:
    qid = query_id or str(uuid.uuid4())
    await conn.execute(
        """
        INSERT INTO query_logs
            (query_id, user_id, question, answer, sources,
             retrieval_ms, generation_ms, model_used, top_k)
        VALUES ($1::uuid,$2,$3,$4,$5::jsonb,$6,$7,$8,$9)
        ON CONFLICT (query_id) DO NOTHING
        """,
        qid, user_id, question, answer,
        json.dumps(sources), retrieval_ms, generation_ms, model_used, top_k,
    )
    return qid
