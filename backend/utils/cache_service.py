"""
Thread-safe LRU + TTL cache with dict-like API.

Designed as a drop-in replacement for the ad-hoc module-level dicts
previously used across the backend (RASTER_FILE_CACHE, INDEX_DATA_CACHE,
_GPU_IMAGE_CACHE, _MAP_ID_CACHE, TARGET_DETECTION_CACHE, MANGROVE_SEG_CACHE).

Key properties:
- Bounded by `maxsize` (LRU eviction when exceeded)
- Per-entry expiry via `ttl` seconds; expired entries are purged lazily on
  access and proactively on set
- Thread-safe: every read/write holds an internal lock
- Optional `on_evict(value)` hook for releasing resources (e.g. GPU memory,
  temp files) when an entry is evicted or cleared
- Supports dict-style access: `cache[k] = v`, `cache[k]`, `k in cache`,
  `cache.get(k)`, `cache.pop(k)`, `cache.clear()`, `len(cache)`, `cache.keys()`
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Iterator, Optional


class TTLCache:
    def __init__(
        self,
        maxsize: int = 128,
        ttl: float = 3600.0,
        on_evict: Optional[Callable[[Any], None]] = None,
        name: str = "cache",
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        if ttl <= 0:
            raise ValueError("ttl must be > 0")
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._on_evict = on_evict
        self._name = name
        self._lock = threading.RLock()

    # ---- dict-like API ----------------------------------------------------
    def __contains__(self, key: str) -> bool:
        with self._lock:
            return self._peek(key) is not None

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            v = self._peek(key)
            if v is None:
                raise KeyError(key)
            self._store.move_to_end(key)
            return v

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            # Evict previous entry (may release resources) before overwrite
            if key in self._store:
                _, old = self._store.pop(key)
                self._fire_evict(old)
            self._store[key] = (time.time() + self._ttl, value)
            self._evict_if_needed()

    def __delitem__(self, key: str) -> None:
        with self._lock:
            if key not in self._store:
                raise KeyError(key)
            _, old = self._store.pop(key)
            self._fire_evict(old)

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._store)

    def __iter__(self) -> Iterator[str]:
        # Snapshot to avoid holding the lock while caller iterates
        with self._lock:
            self._purge_expired()
            return iter(list(self._store.keys()))

    # ---- convenience ------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            v = self._peek(key)
            if v is None:
                return default
            self._store.move_to_end(key)
            return v

    def set(self, key: str, value: Any) -> None:
        self.__setitem__(key, value)

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key not in self._store:
                return default
            _, value = self._store.pop(key)
            # NOTE: pop does not fire on_evict — caller is taking ownership
            return value

    def clear(self) -> None:
        with self._lock:
            values = [v for _, v in self._store.values()]
            self._store.clear()
        # fire hooks outside the lock to avoid reentrancy issues
        for v in values:
            self._fire_evict(v)

    def keys(self) -> list[str]:
        with self._lock:
            self._purge_expired()
            return list(self._store.keys())

    def stats(self) -> dict:
        with self._lock:
            self._purge_expired()
            return {
                "name": self._name,
                "size": len(self._store),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl,
            }

    # ---- internal ---------------------------------------------------------
    def _peek(self, key: str) -> Optional[Any]:
        self._purge_expired()
        entry = self._store.get(key)
        if entry is None:
            return None
        return entry[1]

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [k for k, (exp, _) in self._store.items() if exp < now]
        for k in expired:
            _, v = self._store.pop(k)
            self._fire_evict(v)

    def _evict_if_needed(self) -> None:
        while len(self._store) > self._maxsize:
            k, (_, v) = self._store.popitem(last=False)
            self._fire_evict(v)

    def _fire_evict(self, value: Any) -> None:
        if self._on_evict is None:
            return
        try:
            self._on_evict(value)
        except Exception as e:  # pragma: no cover
            print(f"[TTLCache:{self._name}] on_evict hook failed: {e}")
