"""
LLM generation service.
Supports: Ollama (local default) | Anthropic Claude | OpenAI GPT
All paths support streaming via async generators.
Falls back gracefully: if LLM fails, returns None so the router
can return ranked chunks without a generated answer.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

from config import get_settings

_settings = get_settings()

# ─── System prompt + few-shot ─────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an enterprise document assistant.
Your job is to answer questions based ONLY on the context documents provided.
Rules:
- If the answer is in the context, cite the source (document title + section).
- If you cannot find the answer, say: "I don't have information about this in the indexed documents."
- Be concise, factual, and professional.
- Do not hallucinate or invent facts."""

_FEW_SHOT = [
    {
        "role": "user",
        "content": "Context:\n[Doc: HR Policy 2024, Section: Leave Policy]\nEmployees are entitled to 20 days paid leave per year.\n\nQuestion: How many leave days do employees get?",
    },
    {
        "role": "assistant",
        "content": "According to the HR Policy 2024 (Leave Policy section), employees are entitled to **20 days of paid leave per year**.",
    },
]


# ─── Public API ───────────────────────────────────────────────────────────────

def build_prompt_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Construct the message list for any provider."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("doc_title") or "Unknown Document"
        heading = chunk.get("parent_heading") or ""
        source = chunk.get("source") or ""
        section_label = f"{title}" + (f" › {heading}" if heading else "") + (f" ({source})" if source else "")
        context_parts.append(f"[{i}] {section_label}\n{chunk['chunk_text']}")

    context_block = "\n\n---\n\n".join(context_parts)
    user_message = f"Context:\n{context_block}\n\nQuestion: {question}"

    return [
        *_FEW_SHOT,
        {"role": "user", "content": user_message},
    ]


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
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=_settings.ANTHROPIC_API_KEY)
    resp = await client.messages.create(
        model=_settings.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=messages,
    )
    return resp.content[0].text


async def _stream_anthropic(messages: list[dict]) -> AsyncGenerator[str, None]:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=_settings.ANTHROPIC_API_KEY)
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
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=_settings.OPENAI_API_KEY)
    system_msg = {"role": "system", "content": _SYSTEM_PROMPT}
    resp = await client.chat.completions.create(
        model=_settings.OPENAI_CHAT_MODEL,
        messages=[system_msg, *messages],
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""


async def _stream_openai(messages: list[dict]) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=_settings.OPENAI_API_KEY)
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
    import httpx

    full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}, *messages]
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{_settings.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": _settings.OLLAMA_MODEL,
                "messages": full_messages,
                "stream": False,
                "options": {"num_predict": 400, "num_ctx": 4096, "temperature": 0.1},
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


async def _stream_ollama(messages: list[dict]) -> AsyncGenerator[str, None]:
    import httpx

    full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}, *messages]
    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream(
            "POST",
            f"{_settings.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": _settings.OLLAMA_MODEL,
                "messages": full_messages,
                "stream": True,
                "options": {"num_predict": 400, "num_ctx": 4096, "temperature": 0.1},
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
