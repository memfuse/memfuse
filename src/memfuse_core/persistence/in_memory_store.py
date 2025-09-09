"""In-memory Store implementation for development and tests.

Implements StorePort with a simple token Jaccard similarity for queries.
Safe for unit tests: no external dependencies.
"""
from __future__ import annotations

from typing import Any, Dict, List
from dataclasses import asdict

from ..models.core import Item, Query, QueryResult
from .store_port import StorePort


class InMemoryStore(StorePort):
    def __init__(self) -> None:
        self._items: Dict[str, Item] = {}

    async def initialize(self) -> bool:
        self._items.clear()
        return True

    async def add(self, item: Item) -> str:
        self._items[item.id] = item
        return item.id

    async def query(self, query: Query) -> QueryResult:
        """Return the best matching item as a single QueryResult (per StorePort signature).

        If no items exist, returns a zero-score placeholder result.
        """
        best = None
        best_score = -1.0
        qtext = (query.text or "").strip()
        for it in self._items.values():
            score = _jaccard_score(qtext, it.content)
            if score > best_score:
                best_score = score
                best = it
        if best is None:
            return QueryResult(id="", content="", score=0.0, metadata={"source": "in_memory_store"})
        return QueryResult(
            id=best.id,
            content=best.content,
            score=best_score,
            metadata={**(best.metadata or {}), "source": "in_memory_store"},
            store_type="vector",
        )

    async def delete(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

    # Convenience method for tests and adapters
    async def query_topk(self, query_text: str, top_k: int = 5) -> List[QueryResult]:
        scored: List[QueryResult] = []
        for it in self._items.values():
            score = _jaccard_score(query_text, it.content)
            scored.append(
                QueryResult(
                    id=it.id, content=it.content, score=score,
                    metadata={**(it.metadata or {}), "source": "in_memory_store"},
                    store_type="vector",
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: max(1, top_k)]


def _jaccard_score(q: str, c: str) -> float:
    if not q or not c:
        return 0.0
    qt = set(q.lower().split())
    ct = set(c.lower().split())
    if not qt or not ct:
        return 0.0
    inter = len(qt & ct)
    union = len(qt | ct)
    if union == 0:
        return 0.0
    return inter / union

