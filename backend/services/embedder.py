"""
Embedding service.
Default: HuggingFace all-MiniLM-L6-v2 (384 dims, free, runs on CPU).
Optional: OpenAI text-embedding-3-small (1536 dims, API key required).
Caches embeddings by MD5 hash of chunk text to avoid redundant calls.
"""
from __future__ import annotations

import asyncio
import hashlib
from functools import lru_cache
from typing import Any

from config import get_settings

_settings = get_settings()

# Module-level singleton for the HuggingFace model
_hf_model: Any = None
_model_lock = asyncio.Lock()


async def get_embedder():
    """Warm-up: load HF model into memory (called once at startup)."""
    global _hf_model
    if _settings.EMBEDDING_MODEL == "huggingface" and _hf_model is None:
        async with _model_lock:
            if _hf_model is None:
                from sentence_transformers import SentenceTransformer
                loop = asyncio.get_event_loop()
                _hf_model = await loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer(_settings.HF_MODEL_NAME),
                )
    return _hf_model


async def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a list of strings, return list of float vectors."""
    if not texts:
        return []

    if _settings.EMBEDDING_MODEL == "openai":
        return await _embed_openai(texts, batch_size)
    return await _embed_huggingface(texts, batch_size)


def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ─── HuggingFace ─────────────────────────────────────────────────────────────

async def _embed_huggingface(texts: list[str], batch_size: int) -> list[list[float]]:
    model = await get_embedder()
    if model is None:
        raise RuntimeError("HuggingFace model not loaded")

    loop = asyncio.get_event_loop()
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = await loop.run_in_executor(
            None,
            lambda b=batch: model.encode(b, normalize_embeddings=True).tolist(),
        )
        all_vectors.extend(vectors)

    return all_vectors


# ─── OpenAI ──────────────────────────────────────────────────────────────────

async def _embed_openai(texts: list[str], batch_size: int) -> list[list[float]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=_settings.OPENAI_API_KEY)
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(
            model=_settings.OPENAI_EMBEDDING_MODEL,
            input=batch,
        )
        all_vectors.extend([item.embedding for item in response.data])

    return all_vectors


# ─── Embedding dimension helper ──────────────────────────────────────────────

def embedding_dim() -> int:
    if _settings.EMBEDDING_MODEL == "openai":
        return 1536
    return 384
