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
async def test_gateway_debug_aggregates_dedup_and_clip_stats():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {"inbound": [], "outbound": []},
                "debug": {
                    "enabled": True,
                    "include_dedup_removed_count": True,
                    "include_score_clip_stats": True,
                },
            },
            "guardrail": {"output": {"enabled": False}},
        }
    )

    results = [
        {
            "id": "1",
            "content": "a",
            "score": 0.3,
            "metadata": {
                "observability": {
                    "dedup_removed_count": 2,
                    "score_clip_stats": {
                        "count_clipped": 1,
                        "min_before": 0.1,
                        "max_before": 0.95,
                        "min_after": 0.1,
                        "max_after": 0.9,
                    },
                }
            },
        },
        {"id": "2", "content": "b", "score": 0.2},
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    obs = resp["data"].get("metadata", {}).get("observability", {})
    assert obs.get("dedup_removed_count") == 2
    scs = obs.get("score_clip_stats")
    assert isinstance(scs, dict) and scs.get("count_clipped") == 1 and scs.get("max_after") == 0.9

