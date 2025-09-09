import pytest
from typing import Any, Dict

from src.memfuse_core.gateway.api_gateway import MemoryApiGateway
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class FakeBufferService:
    def __init__(self, results):
        self.results = results

    async def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        return {"status": "success", "code": 200, "data": {"results": self.results, "total": len(self.results)}, "message": "ok", "errors": None}


@pytest.mark.asyncio
async def test_filters_handle_missing_metadata_and_nonstring_content():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {
                    "inbound": [],
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

    # metadata 缺失、content 为非字符串
    results = [
        {"id": "1", "content": 12345, "score": 0.1},
        {"id": "2", "content": {"text": "a@b.com toxic"}, "score": 0.2, "metadata": {}},
        {"id": "3", "content": "clean", "score": 0.3, "metadata": {"source": "x"}},
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    out = resp["data"]["results"]
    # 非字符串不应崩溃
    assert any(r["id"] == "1" for r in out)
    assert any(r["id"] == "2" for r in out)
    # 字段移除对缺失 metadata 安全
    for r in out:
        assert "metadata" in r  # gateway/processors 应保证 metadata 存在
        assert "source" not in r["metadata"]

