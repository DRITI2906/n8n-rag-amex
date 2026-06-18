"""
LLM generation service.
Supports: Ollama (local default) | Anthropic Claude | OpenAI GPT
All paths support streaming via async generators.
Falls back gracefully: if LLM fails, returns None so the router
can return ranked chunks without a generated answer.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator, Any

import httpx

from config import get_settings

_settings = get_settings()

# ─── Singleton clients — created once, reused across all requests ─────────────

_ollama_client: httpx.AsyncClient | None = None
_anthropic_client: Any = None
_openai_client: Any = None


def _get_ollama_client() -> httpx.AsyncClient:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
    return _ollama_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.AsyncAnthropic(api_key=_settings.ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=_settings.OPENAI_API_KEY)
    return _openai_client


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an enterprise document assistant.
Your job is to answer questions based ONLY on the context documents provided.
Rules:
- If the answer is in the context, cite the source (document title + section).
- If the answer spans multiple sources, cite each one.
- If you cannot find the answer, say exactly: "I don't have information about this in the indexed documents."
- Be concise, factual, and professional.
- Do not hallucinate or invent facts not present in the context.
- When quoting figures, dates, or names, copy them exactly from the source."""

# Domain-neutral few-shot grounded in financial/enterprise document style
_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Context:\n"
            "[Doc: Q3 2024 Financial Report › Revenue Summary]\n"
            "Total net revenue for Q3 2024 was $15.3 billion, a 9% increase year-over-year, "
            "driven by growth in Card Member spending and net interest income.\n\n"
            "Question: What was the net revenue and growth rate in Q3 2024?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "According to the **Q3 2024 Financial Report** (Revenue Summary), "
            "total net revenue was **$15.3 billion**, representing a **9% year-over-year increase**, "
            "driven by Card Member spending growth and net interest income."
        ),
    },
]

# Token budget for context: 8192 (num_ctx) minus ~300 for system+few-shot,
# ~200 for question, and 1024 reserved for the model's output.
_CONTEXT_TOKEN_BUDGET = 1200


# ─── Public API ───────────────────────────────────────────────────────────────

def build_prompt_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Construct the message list, trimming chunks to fit the context budget."""
    fitted = _fit_chunks_to_budget(chunks)

    context_parts = []
    for i, chunk in enumerate(fitted, 1):
        title = chunk.get("doc_title") or "Unknown Document"
        heading = chunk.get("parent_heading") or ""
        source = chunk.get("source") or ""
        section_label = (
            title
            + (f" › {heading}" if heading else "")
            + (f" ({source})" if source else "")
        )
        context_parts.append(f"[{i}] {section_label}\n{chunk['chunk_text']}")

    context_block = "\n\n---\n\n".join(context_parts)
    user_message = f"Context:\n{context_block}\n\nQuestion: {question}"

    return [
        *_FEW_SHOT,
        {"role": "user", "content": user_message},
    ]


def _fit_chunks_to_budget(chunks: list[dict]) -> list[dict]:
    """
    Drop the lowest-ranked chunks that would overflow the context token budget.
    Uses a rough 4-chars-per-token estimate to avoid importing tiktoken here.
    Always keeps at least one chunk.
    """
    result: list[dict] = []
    used = 0
    for chunk in chunks:
        estimate = max(1, len(chunk.get("chunk_text", "")) // 4)
        if used + estimate > _CONTEXT_TOKEN_BUDGET and result:
            break
        result.append(chunk)
        used += estimate
    return result


async def generate(
    question: str,
    chunks: list[dict],
    provider: str | None = None,
    stream: bool = True,
) -> AsyncGenerator[str, None] | str | None:
    """
    Generate an answer.
    Returns an async generator (stream=True) or a plain string.
    Returns None on failure so callers can fall back to chunk-only responses.
    """
    p = provider or _settings.LLM_PROVIDER
    messages = build_prompt_messages(question, chunks)

    try:
        if p == "anthropic":
            if stream:
                return _stream_anthropic(messages)
            return await _complete_anthropic(messages)
        elif p == "openai":
            if stream:
                return _stream_openai(messages)
            return await _complete_openai(messages)
        else:  # ollama
            if stream:
                return _stream_ollama(messages)
            return await _complete_ollama(messages)
    except Exception as e:
        print(f"[LLM] Provider '{p}' failed: {e}")
        return None


# ─── Anthropic ────────────────────────────────────────────────────────────────

async def _complete_anthropic(messages: list[dict]) -> str:
    client = _get_anthropic_client()
    resp = await client.messages.create(
        model=_settings.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=messages,
    )
    return resp.content[0].text


async def _stream_anthropic(messages: list[dict]) -> AsyncGenerator[str, None]:
    client = _get_anthropic_client()
    async with client.messages.stream(
        model=_settings.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


# ─── OpenAI ───────────────────────────────────────────────────────────────────

async def _complete_openai(messages: list[dict]) -> str:
    client = _get_openai_client()
    system_msg = {"role": "system", "content": _SYSTEM_PROMPT}
    resp = await client.chat.completions.create(
        model=_settings.OPENAI_CHAT_MODEL,
        messages=[system_msg, *messages],
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""


async def _stream_openai(messages: list[dict]) -> AsyncGenerator[str, None]:
    client = _get_openai_client()
    system_msg = {"role": "system", "content": _SYSTEM_PROMPT}
    stream = await client.chat.completions.create(
        model=_settings.OPENAI_CHAT_MODEL,
        messages=[system_msg, *messages],
        max_tokens=1024,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ─── Ollama ───────────────────────────────────────────────────────────────────

async def _complete_ollama(messages: list[dict]) -> str:
    client = _get_ollama_client()
    full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}, *messages]
    resp = await client.post(
        f"{_settings.OLLAMA_BASE_URL}/api/chat",
        json={
            "model": _settings.OLLAMA_MODEL,
            "messages": full_messages,
            "stream": False,
            "options": {"num_predict": 512, "num_ctx": 2048, "temperature": 0.1},
        },
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


async def _stream_ollama(messages: list[dict]) -> AsyncGenerator[str, None]:
    client = _get_ollama_client()
    full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}, *messages]
    async with client.stream(
        "POST",
        f"{_settings.OLLAMA_BASE_URL}/api/chat",
        json={
            "model": _settings.OLLAMA_MODEL,
            "messages": full_messages,
            "stream": True,
            "options": {"num_predict": 512, "num_ctx": 2048, "temperature": 0.1},
        },
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if line:
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
