"""Adapters to integrate StorePort implementations with Buffer/Query path."""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from .store_port import StorePort
from ..models.core import Query


def make_retrieval_handler_from_store(store: StorePort) -> Callable[[str, int], Any]:
    """Create a retrieval handler compatible with QueryBuffer from a StorePort.

    Returns an async function(query_text, max_results) -> List[dict]
    where each dict has id, content, score, metadata.
    """

    async def retrieval_handler(query_text: str, max_results: int) -> List[Dict[str, Any]]:
        # Prefer optional query_topk if available for efficiency
        if hasattr(store, "query_topk"):
            results = await getattr(store, "query_topk")(query_text, max_results)
            return [
                {"id": r.id, "content": r.content, "score": r.score, "metadata": r.metadata}
                for r in results
            ]
        # Fallback: call single-result query multiple times is not ideal; return best only
        q = Query(text=query_text, metadata={"top_k": max_results})
        r = await store.query(q)
        return [{"id": r.id, "content": r.content, "score": r.score, "metadata": r.metadata}] if r else []

    return retrieval_handler

