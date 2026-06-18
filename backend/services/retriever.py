"""
Retrieval service.
Performs hybrid search: vector similarity (pgvector) + full-text (PostgreSQL ts_vector).
Results are fused via Reciprocal Rank Fusion (RRF) and de-duplicated.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import asyncpg

from db.connection import get_pool
from services.embedder import embed_texts


@dataclass
class SearchResult:
    chunk_id: str
    chunk_text: str
    relevance_score: float
    doc_id: str
    doc_title: str | None
    source: str
    parent_heading: str
    chunk_metadata: dict[str, Any]
    doc_metadata: dict[str, Any]


async def search(
    conn: asyncpg.Connection,
    question: str,
    top_k: int = 5,
    similarity_threshold: float = 0.35,
    source_filter: str | None = None,
    doc_id_filter: str | None = None,
    author_filter: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    use_hybrid: bool = True,
) -> list[SearchResult]:
    """
    Embed the question, search top-K chunks by vector similarity,
    optionally blend with full-text BM25 results via RRF.
    Vector and FTS queries run concurrently when use_hybrid=True.
    """
    query_vectors = await embed_texts([question])
    query_vector = query_vectors[0]
    vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

    # Build optional WHERE filters
    where_clauses = ["1 - (ce.embedding_vector <=> $1::vector) >= $2"]
    params: list[Any] = [vector_str, similarity_threshold]
    p = 3  # next param index

    if source_filter:
        where_clauses.append(f"d.source = ${p}")
        params.append(source_filter)
        p += 1
    if doc_id_filter:
        where_clauses.append(f"d.doc_id = ${p}::uuid")
        params.append(doc_id_filter)
        p += 1
    if author_filter:
        where_clauses.append(f"d.author ILIKE ${p}")
        params.append(f"%{author_filter}%")
        p += 1
    if date_from:
        where_clauses.append(f"d.fetch_timestamp >= ${p}")
        params.append(date_from)
        p += 1
    if date_to:
        where_clauses.append(f"d.fetch_timestamp <= ${p}")
        params.append(date_to)
        p += 1

    where_sql = " AND ".join(where_clauses)
    fetch_limit = top_k * 4  # over-fetch for re-ranking

    vector_sql = f"""
        SELECT
            dc.chunk_id::text,
            dc.chunk_text,
            dc.parent_heading,
            dc.metadata as chunk_metadata,
            d.doc_id::text,
            d.title as doc_title,
            d.source,
            d.metadata as doc_metadata,
            1 - (ce.embedding_vector <=> $1::vector) as score
        FROM chunk_embeddings ce
        JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
        JOIN documents d ON dc.doc_id = d.doc_id
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT {fetch_limit}
    """

    if use_hybrid:
        # Full-text search with same filters (skip vector params $1, $2)
        ft_where = " AND ".join(
            c for c in where_clauses if "embedding_vector" not in c
        ) or "TRUE"
        ft_params = params[2:]  # skip vector + threshold

        ft_sql = f"""
            SELECT
                dc.chunk_id::text,
                dc.chunk_text,
                dc.parent_heading,
                dc.metadata as chunk_metadata,
                d.doc_id::text,
                d.title as doc_title,
                d.source,
                d.metadata as doc_metadata,
                ts_rank(dc.fts_vector, plainto_tsquery('english', $1)) as score
            FROM document_chunks dc
            JOIN documents d ON dc.doc_id = d.doc_id
            WHERE dc.fts_vector @@ plainto_tsquery('english', $1)
              AND {ft_sql_where(ft_where, offset=1)}
            ORDER BY score DESC
            LIMIT {fetch_limit}
        """

        # Run vector and FTS queries concurrently on separate connections.
        pool = get_pool()
        async with pool.acquire() as ft_conn:
            results_pair = await asyncio.gather(
                conn.fetch(vector_sql, *params),
                ft_conn.fetch(ft_sql, question, *ft_params),
                return_exceptions=True,
            )

        vector_rows = results_pair[0] if not isinstance(results_pair[0], Exception) else []
        ft_rows = results_pair[1] if not isinstance(results_pair[1], Exception) else []
        results = _rrf_fuse(vector_rows, ft_rows, top_k=top_k)
    else:
        vector_rows = await conn.fetch(vector_sql, *params)
        results = [_row_to_result(r) for r in vector_rows[:top_k]]

    return results


def ft_sql_where(where: str, offset: int) -> str:
    """Re-number $N params in the FT where clause, skipping the first `offset` positions."""
    def replace(m):
        n = int(m.group(1))
        return f"${n + offset - 2}"
    return re.sub(r"\$(\d+)", replace, where)


def _rrf_fuse(
    vector_rows: list,
    ft_rows: list,
    top_k: int,
    k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank)."""
    scores: dict[str, float] = {}
    row_map: dict[str, Any] = {}

    for rank, row in enumerate(vector_rows):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        row_map[cid] = row

    for rank, row in enumerate(ft_rows):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        row_map.setdefault(cid, row)

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    results = []
    for cid in sorted_ids:
        r = _row_to_result(row_map[cid])
        r.relevance_score = round(scores[cid], 4)
        results.append(r)
    return results


def _row_to_result(row) -> SearchResult:
    # asyncpg's JSONB codec already decodes these as dicts; just guard against NULL.
    return SearchResult(
        chunk_id=row["chunk_id"],
        chunk_text=row["chunk_text"],
        relevance_score=round(float(row["score"]), 4),
        doc_id=row["doc_id"],
        doc_title=row["doc_title"],
        source=row["source"],
        parent_heading=row["parent_heading"] or "",
        chunk_metadata=row["chunk_metadata"] or {},
        doc_metadata=row["doc_metadata"] or {},
    )
