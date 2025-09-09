"""GraphStore: minimal graph-like store for tests (node content indexed).
Not a real graph DB; just satisfies StorePort for unit tests.
"""
from __future__ import annotations

from typing import Dict, List
from .store_port import StorePort
from ..models.core import Item, Query, QueryResult


class GraphStore(StorePort):
    def __init__(self) -> None:
        self._nodes: Dict[str, Item] = {}

    async def initialize(self) -> bool:
        self._nodes.clear()
        return True

    async def add(self, item: Item) -> str:
        self._nodes[item.id] = item
        return item.id

    async def query(self, query: Query) -> QueryResult:
        text = (query.text or "").lower()
        for it in self._nodes.values():
            if text and text in it.content.lower():
                return QueryResult(id=it.id, content=it.content, score=1.0, metadata={"source": "graph"}, store_type="graph")
        return QueryResult(id="", content="", score=0.0, metadata={"source": "graph"}, store_type="graph")

    async def delete(self, item_id: str) -> bool:
        return self._nodes.pop(item_id, None) is not None

    async def query_topk(self, query_text: str, top_k: int = 5) -> List[QueryResult]:
        q = (query_text or "").lower()
        out: List[QueryResult] = []
        for it in self._nodes.values():
            score = 1.0 if q and q in it.content.lower() else 0.0
            out.append(QueryResult(id=it.id, content=it.content, score=score, metadata={"source": "graph"}, store_type="graph"))
        out.sort(key=lambda r: r.score, reverse=True)
        return out[: max(1, top_k)]

