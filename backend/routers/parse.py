"""POST /parse_document — accepts a file upload, returns parsed text + metadata."""
import hashlib
import json
from datetime import datetime

import asyncpg
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from db.connection import get_conn
from services.parser import parse_document

router = APIRouter()


class ParseResponse(BaseModel):
    doc_id: str
    title: str | None
    author: str | None
    language: str
    file_hash: str
    raw_content_size: int
    already_exists: bool
    text_preview: str
    metadata: dict


@router.post("", response_model=ParseResponse)
async def parse_doc(
    file: UploadFile = File(...),
    source: str = Form("upload"),
    file_id: str = Form(""),
    file_url: str = Form(""),
    conn: asyncpg.Connection = Depends(get_conn),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    mime_type = file.content_type or "text/plain"
    filename = file.filename or ""

    try:
        parsed = parse_document(content, mime_type, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parsing failed: {e}")

    # Deduplication check
    existing = await conn.fetchrow(
        "SELECT doc_id FROM documents WHERE file_hash = $1", parsed.file_hash
    )
    if existing:
        return ParseResponse(
            doc_id=str(existing["doc_id"]),
            title=parsed.title,
            author=parsed.author,
            language=parsed.language,
            file_hash=parsed.file_hash,
            raw_content_size=parsed.raw_size,
            already_exists=True,
            text_preview=parsed.text[:300],
            metadata=parsed.metadata,
        )

    doc_id = await conn.fetchval(
        """
        INSERT INTO documents (
            source, file_id, file_hash, title, author, language,
            mime_type, file_url, creation_date, modified_date,
            raw_content_size, status, metadata
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'parsed',$12)
        RETURNING doc_id::text
        """,
        source,
        file_id or filename,
        parsed.file_hash,
        parsed.title,
        parsed.author,
        parsed.language,
        mime_type,
        file_url,
        parsed.creation_date,
        parsed.modified_date,
        parsed.raw_size,
        parsed.metadata,
    )

    # Store full parsed text in metadata for chunking step
    await conn.execute(
        "UPDATE documents SET metadata = metadata || $1::jsonb WHERE doc_id = $2::uuid",
        {"parsed_text": parsed.text},
        doc_id,
    )

    return ParseResponse(
        doc_id=doc_id,
        title=parsed.title,
        author=parsed.author,
        language=parsed.language,
        file_hash=parsed.file_hash,
        raw_content_size=parsed.raw_size,
        already_exists=False,
        text_preview=parsed.text[:300],
        metadata=parsed.metadata,
    )
