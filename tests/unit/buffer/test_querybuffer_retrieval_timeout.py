import asyncio
import pytest
from typing import Any, Dict, List

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class FakeChunk:
    def __init__(self, messages: List[Dict[str, Any]], session_id: str):
        self.messages = messages
        self.metadata = {"session_id": session_id}


class FakeHybrid:
    def __init__(self, session_id: str):
        self.chunks = [
            FakeChunk([
                {"id": "h1", "content": "hyb-1", "created_at": "2024-01-02T00:00:00", "metadata": {"session_id": session_id}},
                {"id": "h2", "content": "hyb-2", "created_at": "2024-01-03T00:00:00", "metadata": {"session_id": session_id}},
            ], session_id=session_id)
        ]


@pytest.mark.asyncio
async def test_query_timeout_returns_empty_without_hybrid():
    gcm = get_global_config_manager()
    await gcm.hot_reload({"buffer": {"retrieval_timeout_seconds": 0.05}})

    async def slow_handler(query_text: str, max_results: int) -> List[Dict[str, Any]]:
        await asyncio.sleep(0.2)
        return [{"id": "s", "content": "slow", "score": 1.0}]

    qb = QueryBuffer(retrieval_handler=slow_handler)
    out = await qb.query("q", top_k=5)
    assert out == []


@pytest.mark.asyncio
async def test_session_timeout_falls_back_to_hybrid_only():
    gcm = get_global_config_manager()
    await gcm.hot_reload({"buffer": {"retrieval_timeout_seconds": 0.05}})

    async def slow_handler(query_text: str, max_results: int) -> List[Dict[str, Any]]:
        await asyncio.sleep(0.2)
        return [{"id": "s1", "content": "slow", "metadata": {"session_id": "S"}, "created_at": "2024-01-01T00:00:00"}]

    qb = QueryBuffer(retrieval_handler=slow_handler)
    qb.set_hybrid_buffer(FakeHybrid("S"))

    out = await qb.query_by_session("S", limit=5, sort_by="timestamp", order="desc")
    # Should return only hybrid messages due to storage timeout
    assert [m["id"] for m in out] == ["h2", "h1"]

