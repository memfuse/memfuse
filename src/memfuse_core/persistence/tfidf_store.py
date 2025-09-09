"""TFIDF-like minimal store without external deps.
Uses simple term-frequency cosine similarity (no IDF) for tests.
"""
from __future__ import annotations

from typing import Dict, List
from math import sqrt

from .store_port import StorePort
from ..models.core import Item, Query, QueryResult


def _tokenize(text: str) -> List[str]:
    return [t for t in (text or "").lower().split() if t]


class TfidfStore(StorePort):
    def __init__(self) -> None:
        self._items: Dict[str, Item] = {}
        self._tf: Dict[str, Dict[str, float]] = {}
        self._norm: Dict[str, float] = {}

    async def initialize(self) -> bool:
        self._items.clear()
        self._tf.clear()
        self._norm.clear()
        return True

    async def add(self, item: Item) -> str:
        self._items[item.id] = item
        tf: Dict[str, float] = {}
        for tok in _tokenize(item.content):
            tf[tok] = tf.get(tok, 0.0) + 1.0
        self._tf[item.id] = tf
        self._norm[item.id] = sqrt(sum(v * v for v in tf.values())) or 1.0
        return item.id

    async def query(self, query: Query) -> QueryResult:
        q = (query.text or "")
        scores = await self.query_topk(q, 1)
        return scores[0] if scores else QueryResult(id="", content="", score=0.0, metadata={"source": "tfidf"}, store_type="tfidf")

    async def delete(self, item_id: str) -> bool:
        self._tf.pop(item_id, None)
        self._norm.pop(item_id, None)
        return self._items.pop(item_id, None) is not None

    async def query_topk(self, query_text: str, top_k: int = 5) -> List[QueryResult]:
        qtf: Dict[str, float] = {}
        for tok in _tokenize(query_text):
            qtf[tok] = qtf.get(tok, 0.0) + 1.0
        qnorm = sqrt(sum(v * v for v in qtf.values())) or 1.0

        scores: List[QueryResult] = []
        for doc_id, item in self._items.items():
            dot = 0.0
            dtf = self._tf.get(doc_id, {})
            for tok, qv in qtf.items():
                dot += qv * dtf.get(tok, 0.0)
            sim = dot / (qnorm * (self._norm.get(doc_id) or 1.0))
            scores.append(
                QueryResult(id=doc_id, content=item.content, score=float(sim), metadata={"source": "tfidf"}, store_type="tfidf")
            )
        scores.sort(key=lambda r: r.score, reverse=True)
        return scores[: max(1, top_k)]

