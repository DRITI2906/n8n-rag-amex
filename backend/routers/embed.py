"""POST /embed_chunks — embed chunks (with caching) and store in pgvector."""
from __future__ import annotations

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config import get_settings
from db.connection import get_conn
from services.embedder import embed_texts, compute_text_hash

router = APIRouter()
_settings = get_settings()

MODEL_NAME = (
    _settings.HF_MODEL_NAME
    if _settings.EMBEDDING_MODEL == "huggingface"
    else _settings.OPENAI_EMBEDDING_MODEL
)


class EmbedRequest(BaseModel):
    chunk_ids: list[str] | None = None  # If None, embed all un-embedded chunks for doc_id
    doc_id: str | None = None
    batch_size: int = 64


class EmbedResponse(BaseModel):
    embedded: int
    skipped_cached: int
    failed: int


@router.post("", response_model=EmbedResponse)
async def embed_chunks(
    req: EmbedRequest,
    conn: asyncpg.Connection = Depends(get_conn),
):
    if req.chunk_ids:
        rows = await conn.fetch(
            "SELECT chunk_id::text, chunk_text FROM document_chunks WHERE chunk_id = ANY($1::uuid[])",
            req.chunk_ids,
        )
    elif req.doc_id:
        rows = await conn.fetch(
            """
            SELECT dc.chunk_id::text, dc.chunk_text
            FROM document_chunks dc
            LEFT JOIN chunk_embeddings ce ON dc.chunk_id = ce.chunk_id
            WHERE dc.doc_id = $1::uuid AND ce.chunk_id IS NULL
            """,
            req.doc_id,
        )
    else:
        raise HTTPException(status_code=400, detail="Provide chunk_ids or doc_id")

    if not rows:
        return EmbedResponse(embedded=0, skipped_cached=0, failed=0)

    # Check cache: skip chunks whose text_hash already exists
    text_hashes = {r["chunk_id"]: compute_text_hash(r["chunk_text"]) for r in rows}
    existing = await conn.fetch(
        "SELECT text_hash FROM chunk_embeddings WHERE text_hash = ANY($1::text[])",
        list(text_hashes.values()),
    )
    cached_hash_set = {r["text_hash"] for r in existing}

    to_embed = [r for r in rows if text_hashes[r["chunk_id"]] not in cached_hash_set]
    skipped = len(rows) - len(to_embed)

    embedded = 0
    failed = 0

    for i in range(0, len(to_embed), req.batch_size):
        batch = to_embed[i : i + req.batch_size]
        texts = [r["chunk_text"] for r in batch]

        try:
            vectors = await embed_texts(texts, batch_size=req.batch_size)
        except Exception as e:
            print(f"[embed] Batch failed: {e}")
            failed += len(batch)
            continue

        async with conn.transaction():
            for row, vector in zip(batch, vectors):
                vector_str = "[" + ",".join(str(v) for v in vector) + "]"
                await conn.execute(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, text_hash, embedding_vector, model_name)
                    VALUES ($1::uuid, $2, $3::vector, $4)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        embedding_vector = EXCLUDED.embedding_vector,
                        model_name = EXCLUDED.model_name,
                        updated_at = NOW()
                    """,
                    row["chunk_id"],
                    text_hashes[row["chunk_id"]],
                    vector_str,
                    MODEL_NAME,
                )
                embedded += 1

    # Update document status
    if req.doc_id and embedded > 0:
        await conn.execute(
            "UPDATE documents SET status='embedded' WHERE doc_id=$1::uuid", req.doc_id
        )

    return EmbedResponse(embedded=embedded, skipped_cached=skipped, failed=failed)
