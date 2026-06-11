import json
import asyncpg
from typing import AsyncGenerator
from config import get_settings

_pool: asyncpg.Pool | None = None


async def _init_conn(conn: asyncpg.Connection) -> None:
    """Register JSON codecs so asyncpg encodes/decodes JSONB as Python dicts."""
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    await conn.set_type_codec(
        "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def init_pool() -> None:
    global _pool
    settings = get_settings()
    # asyncpg wants a raw postgres:// URL, not the SQLAlchemy +asyncpg variant
    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    _pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10, init=_init_conn)


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    assert _pool is not None, "DB pool not initialised — call init_pool() first"
    return _pool


async def get_conn() -> AsyncGenerator[asyncpg.Connection, None]:
    assert _pool is not None, "DB pool not initialised — call init_pool() first"
    async with _pool.acquire() as conn:
        yield conn
