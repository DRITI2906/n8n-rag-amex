"""
POST /query — full RAG chain: retrieve chunks → call LLM → stream answer.
Supports SSE streaming and non-streaming modes.
Logs every query to the query_logs table.
Also exposes POST /query/feedback for thumbs-up/down.
Also exposes GET /query/cache/stats and DELETE /query/cache for cache management.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import date

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import get_settings
from db.connection import get_conn, get_pool
from services.retriever import search, SearchResult
from services.llm import generate
from services.cache import get_cache

router = APIRouter()
_settings = get_settings()


# ─── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_local_llm: bool | None = None
    provider: str | None = None
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
    cache_hit: bool = False


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
    provider = _resolve_provider(req)
    model_used = _resolve_model_name(provider)

    cache = get_cache()
    cache_key = cache.make_key(
        req.question, req.top_k, req.source, req.doc_id,
        provider, req.date_from, req.date_to,
    )
    cached = cache.get(cache_key)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cached:
        if req.stream:
            return EventSourceResponse(
                _stream_cached(cached, model_used, req.user_id, req.top_k)
            )
        query_id = str(uuid.uuid4())
        asyncio.create_task(
            _log_query_with_conn(
                question=req.question,
                answer=cached["answer"],
                sources=cached["sources"],
                retrieval_ms=cached["retrieval_ms"],
                generation_ms=cached["generation_ms"],
                model_used=model_used,
                top_k=req.top_k,
                user_id=req.user_id,
                query_id=query_id,
            )
        )
        return QueryResponse(
            query_id=query_id,
            question=req.question,
            answer=cached["answer"],
            sources=[SourceRef(**_src_ref_fields(c)) for c in cached["sources"]],
            retrieval_time_ms=cached["retrieval_ms"],
            generation_time_ms=cached["generation_ms"],
            model_used=model_used,
            cache_hit=True,
        )

    # ── Cache miss — full pipeline ────────────────────────────────────────────

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

    # 2. Generation — streaming
    if req.stream:
        return EventSourceResponse(
            _stream_response(
                question=req.question,
                chunks=chunk_dicts,
                retrieval_ms=retrieval_ms,
                provider=provider,
                model_used=model_used,
                user_id=req.user_id,
                cache_key=cache_key,
            )
        )

    # 2b. Non-streaming
    t1 = time.monotonic()
    result = await generate(req.question, chunk_dicts, provider=provider, stream=False)
    generation_ms = int((time.monotonic() - t1) * 1000)

    answer = result if isinstance(result, str) else ""
    fallback = not answer

    if not fallback:
        cache.set(cache_key, {
            "answer": answer,
            "sources": chunk_dicts,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
        })

    query_id = str(uuid.uuid4())
    asyncio.create_task(
        _log_query_with_conn(
            question=req.question,
            answer=answer,
            sources=chunk_dicts,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            model_used=model_used,
            top_k=req.top_k,
            user_id=req.user_id,
            query_id=query_id,
        )
    )

    if fallback:
        answer = "Could not generate an answer — here are the most relevant document sections found."

    return QueryResponse(
        query_id=query_id,
        question=req.question,
        answer=answer,
        sources=[SourceRef(**_src_ref_fields(c)) for c in chunk_dicts],
        retrieval_time_ms=retrieval_ms,
        generation_time_ms=generation_ms,
        model_used=model_used,
        cache_hit=False,
    )


# ─── SSE streaming — cache miss ───────────────────────────────────────────────

async def _stream_response(
    question, chunks, retrieval_ms, provider, model_used, user_id, cache_key,
):
    query_id = str(uuid.uuid4())

    yield {
        "event": "metadata",
        "data": json.dumps({
            "query_id": query_id,
            "sources": chunks,
            "retrieval_time_ms": retrieval_ms,
            "model_used": model_used,
            "cache_hit": False,
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
    except Exception:
        full_answer = "Could not generate an answer — here are the most relevant document sections found."
        yield {"event": "token", "data": full_answer}

    generation_ms = int((time.monotonic() - t1) * 1000)

    # Store in cache only on successful generation
    if full_answer and "Could not generate" not in full_answer:
        get_cache().set(cache_key, {
            "answer": full_answer,
            "sources": chunks,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
        })

    try:
        async with get_pool().acquire() as log_conn:
            await _log_query(
                conn=log_conn,
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
    except Exception as e:
        print(f"[query] Failed to log query: {e}")

    yield {"event": "done", "data": json.dumps({"generation_time_ms": generation_ms})}


# ─── SSE streaming — cache hit ────────────────────────────────────────────────

async def _stream_cached(cached: dict, model_used: str, user_id: str | None, top_k: int):
    query_id = str(uuid.uuid4())

    yield {
        "event": "metadata",
        "data": json.dumps({
            "query_id": query_id,
            "sources": cached["sources"],
            "retrieval_time_ms": cached["retrieval_ms"],
            "model_used": model_used,
            "cache_hit": True,
        }),
    }

    yield {"event": "token", "data": cached["answer"]}

    # Log cache hit as a near-zero-latency query
    try:
        async with get_pool().acquire() as log_conn:
            await _log_query(
                conn=log_conn,
                question="",   # already in the original log entry
                answer=cached["answer"],
                sources=cached["sources"],
                retrieval_ms=0,
                generation_ms=0,
                model_used=model_used,
                top_k=top_k,
                user_id=user_id,
                query_id=query_id,
            )
    except Exception as e:
        print(f"[query] Failed to log cached query: {e}")

    yield {"event": "done", "data": json.dumps({"generation_time_ms": 0, "cache_hit": True})}


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


# ─── Cache management endpoints ───────────────────────────────────────────────

@router.get("/cache/stats")
async def cache_stats():
    return get_cache().stats


@router.delete("/cache")
async def clear_cache():
    get_cache().clear()
    return {"status": "cleared"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_provider(req: QueryRequest) -> str:
    if req.provider:
        return req.provider
    if req.use_local_llm is False:
        return "anthropic" if _settings.ANTHROPIC_API_KEY else "openai"
    if req.use_local_llm is True:
        return "ollama"
    return _settings.LLM_PROVIDER


def _resolve_model_name(provider: str) -> str:
    s = _settings
    if provider == "anthropic":
        return s.ANTHROPIC_MODEL
    if provider == "openai":
        return s.OPENAI_CHAT_MODEL
    return f"ollama/{s.OLLAMA_MODEL}"


def _src_ref_fields(c: dict) -> dict:
    return {
        "chunk_id": c["chunk_id"],
        "doc_id": c["doc_id"],
        "doc_title": c["doc_title"],
        "source": c["source"],
        "parent_heading": c["parent_heading"],
        "relevance_score": c["relevance_score"],
    }


async def _log_query_with_conn(
    question, answer, sources, retrieval_ms, generation_ms,
    model_used, top_k, user_id, query_id: str,
) -> None:
    try:
        async with get_pool().acquire() as conn:
            await _log_query(
                conn=conn,
                question=question,
                answer=answer,
                sources=sources,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                model_used=model_used,
                top_k=top_k,
                user_id=user_id,
                query_id=query_id,
            )
    except Exception as e:
        print(f"[query] Failed to log query: {e}")


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
        sources, retrieval_ms, generation_ms, model_used, top_k,
    )
    return qid
