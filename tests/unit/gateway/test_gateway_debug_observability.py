import pytest
from typing import Any, Dict

from src.memfuse_core.gateway.api_gateway import MemoryApiGateway
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class FakeBufferService:
    def __init__(self, results):
        self.results = results

    async def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        return {
            "status": "success",
            "code": 200,
            "data": {"results": self.results, "total": len(self.results)},
            "message": "ok",
            "errors": None,
        }


@pytest.mark.asyncio
async def test_gateway_debug_aggregates_rerank_cache_hit_when_enabled():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {"inbound": [], "outbound": []},
                "debug": {"enabled": True, "include_rerank_cache_hit": True},
            },
            "guardrail": {"output": {"enabled": False}},
        }
    )

    results = [
        {
            "id": "1",
            "content": "hello",
            "score": 0.9,
            "metadata": {"observability": {"rerank_cache_hit": False}},
        },
        {
            "id": "2",
            "content": "world",
            "score": 0.8,
            "metadata": {"observability": {"rerank_cache_hit": True}},
        },
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    assert resp["status"] == "success"
    assert resp["data"]["total"] == 2
    # Aggregated flag should be True because one item had True
    assert resp["data"].get("metadata", {}).get("observability", {}).get("rerank_cache_hit") is True


@pytest.mark.asyncio
async def test_gateway_debug_not_present_when_disabled():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {"pipeline": {"inbound": [], "outbound": []}},
            "guardrail": {"output": {"enabled": False}},
        }
    )

    results = [
        {
            "id": "1",
            "content": "hello",
            "score": 0.9,
            "metadata": {"observability": {"rerank_cache_hit": True}},
        }
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    assert resp["status"] == "success"
    # When disabled, top-level data.metadata should not be added by gateway
    assert "metadata" not in resp["data"]

