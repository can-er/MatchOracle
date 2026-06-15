"""Cache backend contract (phase 2).

All cache backends expose the same minimal get/set/delete surface, so callers
(e.g. the WeightManager) never care whether state lives in Redis, Postgres or
process memory.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any, ttl: int = 300) -> None: ...

    def delete(self, key: str) -> None: ...
