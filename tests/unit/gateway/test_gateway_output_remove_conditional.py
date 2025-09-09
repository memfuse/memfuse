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
async def test_output_remove_conditional_wildcard_on_list_items():
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
                        "metadata.tags.*{k=B}.v",    # only remove v where k == B
                        "metadata.tags.*{k=A}.x",    # only remove x where k == A
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
                    {"k": "C", "v": 3, "x": 30},
                ]
            },
        }
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    out = resp["data"]["results"][0]
    tags = out["metadata"]["tags"]

    # For k == B, v removed, x kept
    tB = next(t for t in tags if t.get("k") == "B")
    assert "v" not in tB and tB.get("x") == 20

    # For k == A, x removed, v kept
    tA = next(t for t in tags if t.get("k") == "A")
    assert "x" not in tA and tA.get("v") == 1

    # For k == C, untouched
    tC = next(t for t in tags if t.get("k") == "C")
    assert tC.get("v") == 3 and tC.get("x") == 30

