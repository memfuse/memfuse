import asyncio
import pytest
from typing import List

from src.memfuse_core.persistence.adapters import RetrievalAdapter
from src.memfuse_core.persistence.store_port import StorePort
from src.memfuse_core.models.core import Query, QueryResult
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class FlakyStore(StorePort):
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    async def query_topk(self, query_text: str, max_results: int) -> List[QueryResult]:
        self.calls += 1
        if self.calls <= self.fail_times:
            # simulate transient error
            raise RuntimeError("transient failure")
        return [QueryResult(id="ok", content="ok", score=0.9, metadata={})]

    # Provide query fallback to satisfy Protocol if needed
    async def query(self, query: Query) -> QueryResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient failure")
        return QueryResult(id="ok", content="ok", score=0.9, metadata={})


@pytest.mark.asyncio
async def test_retry_succeeds_before_exhaustion_with_topk():
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "buffer": {
            "retrieval_timeout_seconds": 0.2,
            "retrieval_retry": {"enabled": True, "max_attempts": 3, "backoff_ms": 1}
        }
    })

    store = FlakyStore(fail_times=2)
    adapter = RetrievalAdapter(store)
    out = await adapter("hi", 3)
    assert len(out) == 1 and out[0]["id"] == "ok"
    assert store.calls == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_retry_exhausts_and_returns_empty():
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "buffer": {
            "retrieval_timeout_seconds": 0.05,
            "retrieval_retry": {"enabled": True, "max_attempts": 2, "backoff_ms": 1}
        }
    })

    store = FlakyStore(fail_times=5)
    adapter = RetrievalAdapter(store)
    out = await adapter("hi", 3)
    assert out == []
    assert store.calls == 2

