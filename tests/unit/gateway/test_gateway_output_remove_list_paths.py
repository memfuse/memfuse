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
async def test_output_remove_supports_list_paths_and_wildcard():
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
                        "metadata.tags.0.k",      # remove key from first element
                        "metadata.tags.*.v",      # remove v from all elements
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
                "tags": [
                    {"k": "A", "v": 1, "x": 10},
                    {"k": "B", "v": 2, "x": 20},
                ]
            },
        }
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    out = resp["data"]["results"][0]
    tags = out["metadata"]["tags"]
    assert "k" not in tags[0]
    assert "v" not in tags[0] and "v" not in tags[1]
    # untouched field remains
    assert tags[0]["x"] == 10 and tags[1]["x"] == 20

