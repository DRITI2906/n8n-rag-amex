"""POST /chunk_document — chunk a previously parsed document."""
import json

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config import get_settings
from db.connection import get_conn
from services.chunker import chunk_document

router = APIRouter()
_settings = get_settings()


class ChunkRequest(BaseModel):
    doc_id: str
    max_tokens: int = 512
    overlap_tokens: int = 100


class ChunkResponse(BaseModel):
    doc_id: str
    chunks_created: int
    chunk_ids: list[str]


@router.post("", response_model=ChunkResponse)
async def chunk_doc(
    req: ChunkRequest,
    conn: asyncpg.Connection = Depends(get_conn),
):
    # Fetch parsed text stored in metadata during /parse_document
    row = await conn.fetchrow(
        "SELECT metadata FROM documents WHERE doc_id = $1::uuid", req.doc_id
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"Document {req.doc_id} not found")

    metadata = dict(row["metadata"] or {})
    text = metadata.get("parsed_text")
    if not text:
        raise HTTPException(
            status_code=422, detail="No parsed text found — run /parse_document first"
        )

    # Remove any old chunks for this doc before re-chunking
    await conn.execute(
        "DELETE FROM document_chunks WHERE doc_id = $1::uuid", req.doc_id
    )

    max_tokens = req.max_tokens or _settings.CHUNK_SIZE_TOKENS
    overlap = req.overlap_tokens or _settings.CHUNK_OVERLAP_TOKENS
    chunks = chunk_document(text, req.doc_id, max_tokens=max_tokens, overlap_tokens=overlap)

    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks produced from document")

    chunk_ids: list[str] = []
    async with conn.transaction():
        for c in chunks:
            cid = await conn.fetchval(
                """
                INSERT INTO document_chunks
                    (doc_id, chunk_index, chunk_text, token_count,
                     char_start, char_end, parent_heading, parent_section, metadata)
                VALUES ($1::uuid,$2,$3,$4,$5,$6,$7,$8,$9::jsonb)
                RETURNING chunk_id::text
                """,
                req.doc_id, c.chunk_index, c.chunk_text, c.token_count,
                c.char_start, c.char_end, c.parent_heading, c.parent_section,
                json.dumps(c.metadata),
            )
            chunk_ids.append(cid)

        await conn.execute(
            "UPDATE documents SET status='chunked' WHERE doc_id=$1::uuid", req.doc_id
        )

    return ChunkResponse(doc_id=req.doc_id, chunks_created=len(chunks), chunk_ids=chunk_ids)
