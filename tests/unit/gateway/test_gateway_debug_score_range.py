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
async def test_gateway_debug_aggregates_score_range_when_enabled():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {"inbound": [], "outbound": []},
                "debug": {"enabled": True, "include_score_range": True},
            },
            "guardrail": {"output": {"enabled": False}},
        }
    )

    results = [
        {"id": "1", "content": "a", "score": 0.1},
        {"id": "2", "content": "b", "score": 0.8},
        {"id": "3", "content": "c", "score": 0.5},
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    obs = resp["data"].get("metadata", {}).get("observability", {})
    assert obs.get("score_range", {}).get("min") == 0.1
    assert obs.get("score_range", {}).get("max") == 0.8


@pytest.mark.asyncio
async def test_gateway_debug_score_range_not_present_when_disabled():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {"pipeline": {"inbound": [], "outbound": []}},
            "guardrail": {"output": {"enabled": False}},
        }
    )

    results = [{"id": "1", "content": "a", "score": 0.3}]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    assert resp["status"] == "success"
    assert "metadata" not in resp["data"]

