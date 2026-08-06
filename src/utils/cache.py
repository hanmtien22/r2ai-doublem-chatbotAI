from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional


class SimpleCache:
    def __init__(self, max_size: int = 1000, enabled: bool = True):
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._enabled = enabled
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        if not self._enabled:
            return None
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if not self._enabled:
            return
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._store),
        }
