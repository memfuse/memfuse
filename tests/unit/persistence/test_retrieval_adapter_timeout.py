import asyncio
import pytest
from typing import List

from src.memfuse_core.persistence.adapters import RetrievalAdapter
from src.memfuse_core.persistence.store_port import StorePort
from src.memfuse_core.models.core import Query, QueryResult
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class SlowStore(StorePort):
    async def query_topk(self, query_text: str, max_results: int) -> List[QueryResult]:
        await asyncio.sleep(0.2)
        return [QueryResult(id="1", content="ok", score=0.5, metadata={})]

    async def query(self, query: Query) -> QueryResult:
        await asyncio.sleep(0.2)
        return QueryResult(id="1", content="ok", score=0.5, metadata={})


@pytest.mark.asyncio
async def test_adapter_timeout_param_overrides():
    adapter = RetrievalAdapter(SlowStore(), timeout_seconds=0.05)
    out = await adapter("hello", 3)
    assert out == []  # timed out -> empty list fallback


@pytest.mark.asyncio
async def test_adapter_timeout_from_config():
    gcm = get_global_config_manager()
    await gcm.hot_reload({"buffer": {"retrieval_timeout_seconds": 0.05}})

    # Without explicit param, should pick from config
    adapter = RetrievalAdapter(SlowStore())
    out = await adapter("hello", 3)
    assert out == []  # timed out -> fallback

    # Increase timeout to allow result
    await gcm.hot_reload({"buffer": {"retrieval_timeout_seconds": 0.3}})
    adapter2 = RetrievalAdapter(SlowStore())
    out2 = await adapter2("hello", 3)
    assert len(out2) == 1 and out2[0]["id"] == "1"

