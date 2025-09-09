"""Adapters to integrate StorePort implementations with Buffer/Query path."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Awaitable, Optional
import asyncio

from .store_port import StorePort
from ..models.core import Query
from ..utils.global_config_manager import get_global_config_manager


class RetrievalAdapter:
    """Class adapter that normalizes StorePort outputs to List[dict] for QueryBuffer.

    Usage:
      adapter = RetrievalAdapter(store, timeout_seconds=0.2)
      handler = adapter.as_handler()  # async (query_text, max_results) -> List[dict]
    """

    def __init__(self, store: StorePort, timeout_seconds: Optional[float] = None) -> None:
        self.store = store
        # If not provided, try read from global config (buffer.retrieval_timeout_seconds)
        self.timeout_seconds = timeout_seconds
        if self.timeout_seconds is None:
            try:
                gcm = get_global_config_manager()
                if gcm.is_initialized():
                    buf_cfg = gcm.get_section("buffer") or {}
                    rts = buf_cfg.get("retrieval_timeout_seconds")
                    if rts is not None:
                        self.timeout_seconds = float(rts)
            except Exception:
                self.timeout_seconds = None

    async def __call__(self, query_text: str, max_results: int) -> List[Dict[str, Any]]:
        return await self._retrieve(query_text, max_results)

    async def _retrieve(self, query_text: str, max_results: int) -> List[Dict[str, Any]]:
        try:
            # Prefer optional query_topk if available for efficiency
            if hasattr(self.store, "query_topk"):
                coro = getattr(self.store, "query_topk")(query_text, max_results)
                if self.timeout_seconds:
                    results = await asyncio.wait_for(coro, timeout=self.timeout_seconds)
                else:
                    results = await coro
                return [
                    {"id": r.id, "content": r.content, "score": r.score, "metadata": r.metadata}
                    for r in results
                ]
            # Fallback: single-result query
            q = Query(text=query_text, metadata={"top_k": max_results})
            coro2 = self.store.query(q)
            if self.timeout_seconds:
                r = await asyncio.wait_for(coro2, timeout=self.timeout_seconds)
            else:
                r = await coro2
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

