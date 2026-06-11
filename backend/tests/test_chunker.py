"""
Unit tests for the chunker service.
Run with: pytest tests/test_chunker.py -v
"""
import pytest
from services.chunker import chunk_document, _count_tokens


DOC_ID = "test-doc-001"


# ── Basic tests ──────────────────────────────────────────────────────────────

def test_short_document_becomes_one_chunk():
    text = "This is a short document with minimal content."
    chunks = chunk_document(text, DOC_ID, max_tokens=512)
    assert len(chunks) == 1
    assert chunks[0].chunk_text == text.strip()
    assert chunks[0].chunk_index == 0
    assert chunks[0].doc_id == DOC_ID


def test_empty_document_returns_no_chunks():
    chunks = chunk_document("", DOC_ID)
    assert chunks == []


def test_whitespace_only_returns_no_chunks():
    chunks = chunk_document("   \n\n\n   ", DOC_ID)
    assert chunks == []


# ── Heading splitting ─────────────────────────────────────────────────────────

def test_headings_preserved_as_parent():
    text = "## Introduction\n\nThis section introduces the topic.\n\n## Methods\n\nHere we describe methods."
    chunks = chunk_document(text, DOC_ID)
    headings = {c.parent_heading for c in chunks}
    assert "Introduction" in headings
    assert "Methods" in headings


def test_nested_headings():
    text = "# Chapter 1\n\nIntro text.\n\n## Section 1.1\n\nDetailed content here.\n\n## Section 1.2\n\nMore content."
    chunks = chunk_document(text, DOC_ID)
    assert any(c.parent_heading == "Chapter 1" for c in chunks)
    assert any(c.parent_heading == "Section 1.1" for c in chunks)


# ── Token limits ──────────────────────────────────────────────────────────────

def test_chunk_respects_max_tokens():
    """No chunk should exceed max_tokens (allowing small overshoot at sentence boundary)."""
    long_para = " ".join(["word"] * 600)  # ~600 tokens
    text = f"## Long Section\n\n{long_para}"
    chunks = chunk_document(text, DOC_ID, max_tokens=200, overlap_tokens=20)
    for c in chunks:
        # Allow 10% overshoot due to sentence boundary rounding
        assert c.token_count <= 220, f"Chunk {c.chunk_index} has {c.token_count} tokens > 220"


def test_overlap_content_shared_between_chunks():
    """Consecutive chunks should share some tokens from the overlap."""
    sentences = [f"This is sentence number {i} in the document." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunk_document(text, DOC_ID, max_tokens=100, overlap_tokens=20)
    if len(chunks) > 1:
        # The end of chunk[0] should appear somewhere in chunk[1]
        end_of_first = chunks[0].chunk_text[-80:]
        # Overlap means some words from end_of_first appear in start_of_second
        start_of_second = chunks[1].chunk_text[:100]
        # At least one word overlap
        words_first = set(end_of_first.split())
        words_second = set(start_of_second.split())
        assert words_first & words_second, "No overlap detected between consecutive chunks"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_very_long_single_paragraph():
    """A single paragraph longer than max_tokens must be split by sentence."""
    long_text = ". ".join([f"Sentence {i} of the very long paragraph" for i in range(100)]) + "."
    chunks = chunk_document(long_text, DOC_ID, max_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1


def test_table_like_content():
    """Pipe-delimited table rows should not crash the chunker."""
    table = "Column A | Column B | Column C\n" + "\n".join(
        f"Row {i} data | Value {i} | Result {i}" for i in range(50)
    )
    chunks = chunk_document(table, DOC_ID, max_tokens=200)
    assert len(chunks) >= 1
    assert all(c.chunk_text for c in chunks)


def test_chunk_indices_are_sequential():
    long_text = " ".join([f"Word{i}" for i in range(2000)])
    chunks = chunk_document(long_text, DOC_ID, max_tokens=100, overlap_tokens=10)
    for expected, c in enumerate(chunks):
        assert c.chunk_index == expected


def test_char_positions_non_negative():
    text = "## Heading\n\nSome content here.\n\n## Another Heading\n\nMore content."
    chunks = chunk_document(text, DOC_ID)
    for c in chunks:
        assert c.char_start >= 0
        assert c.char_end >= c.char_start


def test_markdown_document():
    md = """# Project Overview

Build a production-ready document intelligence system.

## Phase 1: Ingestion

Connect multiple data sources including Google Drive and Slack.
This phase involves building n8n workflows for each source.

## Phase 2: Processing

Parse documents and chunk them intelligently.
Apply recursive splitting by heading, paragraph, and sentence.

## Phase 3: Embedding

Generate vector embeddings using HuggingFace models.
Store embeddings in PostgreSQL with pgvector extension."""

    chunks = chunk_document(md, DOC_ID, max_tokens=100, overlap_tokens=20)
    assert len(chunks) >= 3
    assert all(c.chunk_text.strip() for c in chunks)


def test_token_count_accuracy():
    text = "Hello world, this is a test."
    chunks = chunk_document(text, DOC_ID)
    assert chunks[0].token_count == _count_tokens(text)
