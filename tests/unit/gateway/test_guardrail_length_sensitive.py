import pytest

from memfuse_core.gateway.api_gateway import MemoryApiGateway
from memfuse_core.interfaces.gateway_interface import OperationType
from memfuse_core.utils.global_config_manager import get_global_config_manager


class MockBufferService:
    def __init__(self, content: str):
        self._content = content

    async def query(self, query: str, top_k: int = 5, **kwargs):
        return {
            "status": "success",
            "code": 200,
            "data": {
                "results": [
                    {
                        "id": "1",
                        "relevance_score": 0.5,
                        "memory_type": "episodic",
                        "content": self._content,
                        "created_at": None,
                        "metadata": {"user_id": "u", "scope": "in_session"},
                    }
                ],
                "total": 1,
            },
            "message": "ok",
        }


@pytest.mark.asyncio
async def test_max_length_truncates_and_marks():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {
                    "inbound": [],
                    "outbound": [
                        {"name": "max_length", "enabled": True},
                    ],
                }
            },
            "guardrail": {
                "length": {"enabled": True, "max_content_length": 10, "suffix": "..."}
            },
        }
    )

    long_text = "this is a very long content"
    gateway = MemoryApiGateway(buffer_service=MockBufferService(long_text), db_service=None)
    resp = await gateway.process_request({"user_id": "u", "query": "hello"}, operation_type=OperationType.QUERY)

    res0 = resp["data"]["results"][0]
    assert res0["content"].startswith("this is a ") and res0["content"].endswith("...")
    assert res0["metadata"].get("length_truncated") is True


@pytest.mark.asyncio
async def test_sensitive_word_masking_marks_and_masks():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {
                    "inbound": [],
                    "outbound": [
                        {"name": "sensitive_word", "enabled": True},
                    ],
                }
            },
            "guardrail": {
                "sensitive": {"enabled": True, "words": ["forbidden", "bad"], "mask_token": "[BLOCKED]"}
            },
        }
    )

    text = "This contains a forbidden phrase and a BAD word."
    gateway = MemoryApiGateway(buffer_service=MockBufferService(text), db_service=None)
    resp = await gateway.process_request({"user_id": "u", "query": "hello"}, operation_type=OperationType.QUERY)

    res0 = resp["data"]["results"][0]
    assert "[BLOCKED]" in res0["content"]
    assert res0["metadata"].get("sensitive_hit") is True

