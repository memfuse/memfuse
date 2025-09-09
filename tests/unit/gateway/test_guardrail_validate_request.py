import pytest
from typing import Any, Dict

from src.memfuse_core.gateway.api_gateway import MemoryApiGateway
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class EchoBufferService:
    async def query(self, query: Any, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        return {
            "status": "success",
            "code": 200,
            "data": {"results": [{"id": "1", "content": str(query), "score": 0.0, "metadata": {}}], "total": 1},
            "message": "ok",
            "errors": None,
        }


@pytest.mark.asyncio
async def test_guardrail_validate_request_blocks_invalid_types():
    # No inbound normalizer enabled; query is int -> should be blocked by guardrail
    gcm = get_global_config_manager()
    await gcm.hot_reload({"gateway": {"pipeline": {"inbound": [], "outbound": []}}})

    gw = MemoryApiGateway(buffer_service=EchoBufferService(), db_service=None)
    resp = await gw.process_request({"user_id": "u", "query": 123, "top_k": "abc"})
    assert resp["status"] == "error"


@pytest.mark.asyncio
async def test_guardrail_validate_request_allows_after_normalization():
    # Enable inbound normalizer to coerce/limit, then guardrail accepts
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "request_normalizer": {"enabled": True},
                "pipeline": {"inbound": [{"name": "request_normalizer", "enabled": True}], "outbound": []},
            }
        }
    )

    gw = MemoryApiGateway(buffer_service=EchoBufferService(), db_service=None)
    resp = await gw.process_request({"user_id": "u", "query": 123, "top_k": "abc"})
    assert resp["status"] == "success"

