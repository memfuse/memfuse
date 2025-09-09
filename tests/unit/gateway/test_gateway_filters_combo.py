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
async def test_outbound_filters_combined_order():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {
                    "inbound": [],
                    # Ensure order: pii_redact -> toxicity_mark -> output_remove
                    "outbound": [
                        {"name": "pii_redact", "enabled": True},
                        {"name": "toxicity_mark", "enabled": True},
                        {"name": "output_remove", "enabled": True},
                    ],
                }
            },
            "guardrail": {
                "pii": {"enabled": True, "redact": True},
                "toxicity": {"enabled": True, "threshold": 0.0},
                "output": {"enabled": True, "remove_fields": ["metadata.source"]},
            },
        }
    )

    results = [
        {"id": "1", "content": "email a@b.com toxic", "score": 0.9, "metadata": {"source": "hybrid"}},
        {"id": "2", "content": "harmless content", "score": 0.1, "metadata": {"source": "storage"}},
    ]
    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    out = resp["data"]["results"]
    # PII in content should be redacted
    assert "[REDACTED:EMAIL]" in out[0]["content"]
    # Toxicity flag should be set on item containing keyword
    assert out[0]["metadata"].get("toxicity_flag") is True
    # metadata.source should be removed finally
    for r in out:
        assert "source" not in r.get("metadata", {})


@pytest.mark.asyncio
async def test_gateway_handles_empty_results():
    gcm = get_global_config_manager()
    await gcm.hot_reload({"gateway": {"pipeline": {"inbound": [], "outbound": []}}})
    gw = MemoryApiGateway(buffer_service=FakeBufferService([]), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "anything", "top_k": 3})
    assert resp["status"] == "success"
    assert isinstance(resp.get("data", {}).get("results", []), list)
    assert resp["data"]["results"] == []

