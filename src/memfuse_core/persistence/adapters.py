"""Adapters to integrate StorePort implementations with Buffer/Query path."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Awaitable

from .store_port import StorePort
from ..models.core import Query


class RetrievalAdapter:
    """Class adapter that normalizes StorePort outputs to List[dict] for QueryBuffer.

    Usage:
      adapter = RetrievalAdapter(store)
      handler = adapter.as_handler()  # async (query_text, max_results) -> List[dict]
    """

    def __init__(self, store: StorePort) -> None:
        self.store = store

    async def __call__(self, query_text: str, max_results: int) -> List[Dict[str, Any]]:
        return await self._retrieve(query_text, max_results)

    async def _retrieve(self, query_text: str, max_results: int) -> List[Dict[str, Any]]:
        try:
            # Prefer optional query_topk if available for efficiency
            if hasattr(self.store, "query_topk"):
                results = await getattr(self.store, "query_topk")(query_text, max_results)
                return [
                    {"id": r.id, "content": r.content, "score": r.score, "metadata": r.metadata}
                    for r in results
                ]
            # Fallback: single-result query
            q = Query(text=query_text, metadata={"top_k": max_results})
            r = await self.store.query(q)
            return (
                [{"id": r.id, "content": r.content, "score": r.score, "metadata": r.metadata}]
                if r else []
            )
        except Exception:
            # Be safe in adapter to avoid breaking upstream pipelines
            return []

    def as_handler(self) -> Callable[[str, int], Awaitable[List[Dict[str, Any]]]]:
        async def handler(query_text: str, max_results: int) -> List[Dict[str, Any]]:
            return await self._retrieve(query_text, max_results)
        return handler


def make_retrieval_handler_from_store(store: StorePort) -> Callable[[str, int], Any]:
    """Create a retrieval handler compatible with QueryBuffer from a StorePort.

    Returns an async function(query_text, max_results) -> List[dict]
    where each dict has id, content, score, metadata.
    """
    return RetrievalAdapter(store).as_handler()

