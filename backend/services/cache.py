"""
In-memory query result cache.
LRU eviction (max 256 entries) + TTL expiry (1 hour by default).
Keyed on a normalized hash of (question, top_k, source, doc_id, provider, date_from, date_to).
No external dependencies — uses only stdlib OrderedDict.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any


class QueryCache:
    def __init__(self, max_size: int = 256, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def make_key(
        self,
        question: str,
        top_k: int,
        source: str | None,
        doc_id: str | None,
        provider: str,
        date_from: Any = None,
        date_to: Any = None,
    ) -> str:
        payload = json.dumps({
            "q": question.strip().lower(),
            "k": top_k,
            "src": source,
            "doc": doc_id,
            "prv": provider,
            "df": str(date_from),
            "dt": str(date_to),
        }, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            self.misses += 1
            return None
        ts, value = self._cache[key]
        if time.monotonic() - ts > self._ttl:
            del self._cache[key]
            self.misses += 1
            return None
        self._cache.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.monotonic(), value)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": self.size,
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


# Module-level singleton
_query_cache = QueryCache()


def get_cache() -> QueryCache:
    return _query_cache
