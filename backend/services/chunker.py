"""
Intelligent recursive chunker.
Strategy: heading → paragraph → sentence
Preserves parent_heading context and applies token overlap between chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken
import nltk

# Download once (Dockerfile pre-downloads, but guard here too)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    chunk_text: str
    token_count: int
    char_start: int
    char_end: int
    parent_heading: str
    parent_section: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── Public API ──────────────────────────────────────────────────────────────

def chunk_document(
    text: str,
    doc_id: str,
    max_tokens: int = 512,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Split document text into chunks, returning a list of Chunk objects."""
    sections = _split_by_headings(text)
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        heading = section["heading"]
        body = section["body"]
        char_offset = section["char_offset"]

        new_chunks = _chunk_section(
            body=body,
            doc_id=doc_id,
            start_index=chunk_index,
            char_offset=char_offset,
            parent_heading=heading,
            parent_section=heading,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        chunks.extend(new_chunks)
        chunk_index += len(new_chunks)

    return chunks


# ─── Heading splitter ─────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _split_by_headings(text: str) -> list[dict]:
    """
    Split text into sections by Markdown headings.
    Each section = {heading, body, char_offset}.
    Handles documents with no headings as one big section.
    """
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return [{"heading": "", "body": text, "char_offset": 0}]

    sections: list[dict] = []

    # Text before first heading
    if matches[0].start() > 0:
        sections.append({
            "heading": "",
            "body": text[:matches[0].start()].strip(),
            "char_offset": 0,
        })

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append({
            "heading": heading,
            "body": body,
            "char_offset": body_start,
        })

    return [s for s in sections if s["body"]]


# ─── Section → chunks ────────────────────────────────────────────────────────

def _chunk_section(
    body: str,
    doc_id: str,
    start_index: int,
    char_offset: int,
    parent_heading: str,
    parent_section: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    if _count_tokens(body) <= max_tokens:
        return [_make_chunk(
            doc_id, start_index, body,
            char_offset, char_offset + len(body),
            parent_heading, parent_section,
        )]

    # Split by paragraph first
    paragraphs = _split_paragraphs(body)
    return _pack_into_chunks(
        segments=paragraphs,
        doc_id=doc_id,
        start_index=start_index,
        char_offset=char_offset,
        parent_heading=parent_heading,
        parent_section=parent_section,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def _pack_into_chunks(
    segments: list[str],
    doc_id: str,
    start_index: int,
    char_offset: int,
    parent_heading: str,
    parent_section: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    local_char = char_offset
    chunk_start = char_offset
    idx = start_index

    for seg in segments:
        seg_tokens = _count_tokens(seg)

        # Segment itself is too long → split by sentence
        if seg_tokens > max_tokens:
            # Flush what we have
            if current_parts:
                text = "\n\n".join(current_parts)
                chunks.append(_make_chunk(doc_id, idx, text, chunk_start, local_char, parent_heading, parent_section))
                idx += 1
                overlap_text = _take_overlap(text, overlap_tokens)
                current_parts = [overlap_text] if overlap_text else []
                current_tokens = _count_tokens(overlap_text) if overlap_text else 0
                chunk_start = local_char

            # Recurse on sentences
            sentences = _split_sentences(seg)
            sentence_chunks = _pack_into_chunks(
                segments=sentences,
                doc_id=doc_id,
                start_index=idx,
                char_offset=local_char,
                parent_heading=parent_heading,
                parent_section=parent_section,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
            chunks.extend(sentence_chunks)
            idx += len(sentence_chunks)
            local_char += len(seg) + 2
            chunk_start = local_char
            continue

        if current_tokens + seg_tokens > max_tokens and current_parts:
            # Flush current chunk
            text = "\n\n".join(current_parts)
            chunks.append(_make_chunk(doc_id, idx, text, chunk_start, local_char, parent_heading, parent_section))
            idx += 1
            # Start next chunk with overlap from previous
            overlap_text = _take_overlap(text, overlap_tokens)
            current_parts = [overlap_text] if overlap_text else []
            current_tokens = _count_tokens(overlap_text) if overlap_text else 0
            chunk_start = local_char

        current_parts.append(seg)
        current_tokens += seg_tokens
        local_char += len(seg) + 2  # +2 for "\n\n"

    if current_parts:
        text = "\n\n".join(current_parts)
        chunks.append(_make_chunk(doc_id, idx, text, chunk_start, local_char, parent_heading, parent_section))

    return chunks


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on ". "
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _take_overlap(text: str, overlap_tokens: int) -> str:
    """Return the last `overlap_tokens` tokens of text as a string."""
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    overlap_token_ids = tokens[-overlap_tokens:]
    return _TOKENIZER.decode(overlap_token_ids)


def _make_chunk(
    doc_id: str,
    chunk_index: int,
    text: str,
    char_start: int,
    char_end: int,
    parent_heading: str,
    parent_section: str,
) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_index=chunk_index,
        chunk_text=text.strip(),
        token_count=_count_tokens(text),
        char_start=char_start,
        char_end=char_end,
        parent_heading=parent_heading,
        parent_section=parent_section,
    )
