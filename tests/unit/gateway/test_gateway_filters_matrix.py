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
async def test_outbound_output_remove_only():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {"inbound": [], "outbound": [{"name": "output_remove", "enabled": True}]}
            },
            "guardrail": {"output": {"enabled": True, "remove_fields": ["metadata.source", "content"]}},
        }
    )

    results = [
        {"id": "1", "content": "keep?", "score": 0.9, "metadata": {"source": "hybrid", "x": 1}},
        {"id": "2", "content": "keep?", "score": 0.8, "metadata": {"source": "storage", "y": 2}},
    ]
    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})
    for r in resp["data"]["results"]:
        assert "source" not in r.get("metadata", {})
        assert "content" not in r  # removed


@pytest.mark.asyncio
async def test_outbound_pii_only():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {"pipeline": {"inbound": [], "outbound": [{"name": "pii_redact", "enabled": True}]}},
            "guardrail": {"pii": {"enabled": True, "redact": True}},
        }
    )

    results = [{"id": "1", "content": "email me at a@b.com", "score": 0.5, "metadata": {}}]
    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})
    assert "[REDACTED:EMAIL]" in resp["data"]["results"][0]["content"]


@pytest.mark.asyncio
async def test_outbound_toxicity_only():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {"pipeline": {"inbound": [], "outbound": [{"name": "toxicity_mark", "enabled": True}]}},
            "guardrail": {"toxicity": {"enabled": True, "threshold": 0.0}},
        }
    )

    results = [{"id": "1", "content": "This is toxic", "score": 0.5, "metadata": {}}]
    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})
    md = resp["data"]["results"][0]["metadata"]
    assert md.get("toxicity_flag") is True

