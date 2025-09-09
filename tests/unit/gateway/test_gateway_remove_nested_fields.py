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
async def test_outbound_remove_nested_fields():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {"inbound": [], "outbound": [{"name": "output_remove", "enabled": True}]}
            },
            "guardrail": {
                "output": {
                    "enabled": True,
                    "remove_fields": [
                        "metadata.observability.stage",
                        "metadata.observability.query_len",
                        "metadata.source",
                    ],
                }
            },
        }
    )

    results = [
        {
            "id": "1",
            "content": "hello",
            "score": 0.9,
            "metadata": {
                "source": "hybrid",
                "observability": {"stage": "merge", "query_len": 12, "keep": True},
            },
        },
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    out = resp["data"]["results"][0]
    md = out.get("metadata", {})
    assert "source" not in md
    assert "observability" in md and "stage" not in md["observability"] and "query_len" not in md["observability"]
    assert md["observability"].get("keep") is True

