import asyncpg
from typing import AsyncGenerator
from config import get_settings

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    global _pool
    settings = get_settings()
    # asyncpg wants a raw postgres:// URL, not the SQLAlchemy +asyncpg variant
    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    _pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def get_conn() -> AsyncGenerator[asyncpg.Connection, None]:
    assert _pool is not None, "DB pool not initialised — call init_pool() first"
    async with _pool.acquire() as conn:
        yield conn
