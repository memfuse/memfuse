import pytest
from typing import Any, Dict

from src.memfuse_core.gateway.api_gateway import MemoryApiGateway
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class EchoBufferService:
    async def query(self, query: Any, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        # Echo back what we received to verify inbound normalization
        result = {
            "id": "echo",
            "content": str(query),
            "score": 0.0,
            "metadata": {"echo": {"received_top_k": top_k, "type_query": type(query).__name__}},
        }
        return {
            "status": "success",
            "code": 200,
            "data": {"results": [result], "total": 1},
            "message": "ok",
            "errors": None,
        }


@pytest.mark.asyncio
async def test_inbound_request_normalizer_clamps_and_coerces():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "request_normalizer": {
                    "enabled": True,
                    "min_top_k": 1,
                    "max_top_k": 3,
                    "default_top_k": 2,
                },
                "pipeline": {"inbound": [{"name": "request_normalizer", "enabled": True}], "outbound": []},
            }
        }
    )

    gw = MemoryApiGateway(buffer_service=EchoBufferService(), db_service=None)

    # Provide invalid top_k and non-string query; expect normalization
    resp = await gw.process_request({
        "user_id": "u",
        "agent_id": "a",
        "session_id": "s",
        "query": None,
        "top_k": 0,
    })

    assert resp["status"] == "success"
    res = resp["data"]["results"][0]
    # Query coerced to string
    assert isinstance(res["content"], str)
    # top_k clamped to min_top_k=1
    assert res["metadata"]["echo"]["received_top_k"] == 1

