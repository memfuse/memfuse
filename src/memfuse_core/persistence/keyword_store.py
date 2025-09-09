"""KeywordStore: minimal keyword-based store for tests."""
from __future__ import annotations

from typing import Dict, List
from .store_port import StorePort
from ..models.core import Item, Query, QueryResult


class KeywordStore(StorePort):
    def __init__(self) -> None:
        self._items: Dict[str, Item] = {}

    async def initialize(self) -> bool:
        self._items.clear()
        return True

    async def add(self, item: Item) -> str:
        self._items[item.id] = item
        return item.id

    async def query(self, query: Query) -> QueryResult:
        text = (query.text or "").lower()
        for it in self._items.values():
            if any(tok in it.content.lower() for tok in text.split() if tok):
                return QueryResult(id=it.id, content=it.content, score=1.0, metadata={"source": "keyword"}, store_type="keyword")
        return QueryResult(id="", content="", score=0.0, metadata={"source": "keyword"}, store_type="keyword")

    async def delete(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

    async def query_topk(self, query_text: str, top_k: int = 5) -> List[QueryResult]:
        q = (query_text or "").lower()
        out: List[QueryResult] = []
        for it in self._items.values():
            score = 1.0 if any(tok in it.content.lower() for tok in q.split() if tok) else 0.0
            out.append(QueryResult(id=it.id, content=it.content, score=score, metadata={"source": "keyword"}, store_type="keyword"))
        out.sort(key=lambda r: r.score, reverse=True)
        return out[: max(1, top_k)]

