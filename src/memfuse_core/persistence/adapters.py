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
        self.store_class_name = store.__class__.__name__
        # If not provided, try read from global config (buffer.retrieval_timeout_seconds)
        self.timeout_seconds = timeout_seconds
        # Retry config (buffer.retrieval_retry)
        self.retry_enabled: bool = False
        self.retry_attempts: int = 1
        self.retry_backoff_ms: int = 0
        try:
            gcm = get_global_config_manager()
            if gcm.is_initialized():
                buf_cfg = gcm.get_section("buffer") or {}
                # timeout (global default, then per-store override)
                if self.timeout_seconds is None:
                    rts = buf_cfg.get("retrieval_timeout_seconds")
                    if rts is not None:
                        self.timeout_seconds = float(rts)
                # per-store timeout override
                per_store_cfg = buf_cfg.get("retrieval_per_store", {}) or {}
                store_cfg = per_store_cfg.get(self.store_class_name, {}) or {}
                if "timeout_seconds" in store_cfg:
                    self.timeout_seconds = float(store_cfg["timeout_seconds"])
                # retries (global default, then per-store override)
                rr = buf_cfg.get("retrieval_retry", {}) or {}
                self.retry_enabled = bool(rr.get("enabled", False))
                self.retry_attempts = int(rr.get("max_attempts", 1))
                self.retry_backoff_ms = int(rr.get("backoff_ms", 0))
                # per-store retry override
                if "retry" in store_cfg:
                    store_retry = store_cfg["retry"] or {}
                    if "enabled" in store_retry:
                        self.retry_enabled = bool(store_retry["enabled"])
                    if "max_attempts" in store_retry:
                        self.retry_attempts = int(store_retry["max_attempts"])
                    if "backoff_ms" in store_retry:
                        self.retry_backoff_ms = int(store_retry["backoff_ms"])
        except Exception:
            # Use safe defaults
            if self.timeout_seconds is None:
                self.timeout_seconds = None
            self.retry_enabled = False
            self.retry_attempts = 1
            self.retry_backoff_ms = 0

    async def __call__(self, query_text: str, max_results: int) -> List[Dict[str, Any]]:
        return await self._retrieve(query_text, max_results)

    async def _attempt_once(self, query_text: str, max_results: int) -> List[Dict[str, Any]]:
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

    async def _retrieve(self, query_text: str, max_results: int) -> List[Dict[str, Any]]:
        attempts = self.retry_attempts if self.retry_enabled else 1
        for i in range(attempts):
            try:
                return await self._attempt_once(query_text, max_results)
            except Exception:
                # If this was the last attempt, break to return []
                if i >= attempts - 1:
                    break
                # Backoff before retry
                if self.retry_backoff_ms > 0:
                    await asyncio.sleep(self.retry_backoff_ms / 1000.0)
                continue
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

