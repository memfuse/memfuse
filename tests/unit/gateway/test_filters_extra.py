"""Unit tests for additional outbound filters: PII redaction and Toxicity annotator."""

import asyncio
import pytest

from memfuse_core.gateway.api_gateway import MemoryApiGateway
from memfuse_core.interfaces.gateway_interface import OperationType
from memfuse_core.utils.global_config_manager import get_global_config_manager


class MockBufferService:
    def __init__(self, content: str, metadata: dict):
        self._content = content
        self._metadata = metadata

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
                        "metadata": {"user_id": "u", "scope": "in_session", **self._metadata},
                    }
                ],
                "total": 1,
            },
            "message": "ok",
        }


@pytest.mark.asyncio
async def test_pii_redact_filter_masks_email_and_phone():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {
                    "inbound": [],
                    "outbound": [
                        {"name": "pii_redact", "enabled": True},
                    ],
                }
            },
            "guardrail": {
                "pii": {"enabled": True, "redact": True}
            },
        }
    )

    content = "Contact me at john.doe@example.com or +1 415-555-1234"
    metadata = {"email": "john.doe@example.com", "phone": "+14155551234"}

    gateway = MemoryApiGateway(buffer_service=MockBufferService(content, metadata), db_service=None)
    resp = await gateway.process_request({"user_id": "u", "query": "hello"}, operation_type=OperationType.QUERY)

    res0 = resp["data"]["results"][0]
    assert "[REDACTED:EMAIL]" in res0["content"]
    assert "[REDACTED:PHONE]" in res0["content"]
    assert res0["metadata"]["email"] == "[REDACTED:EMAIL]"
    assert res0["metadata"]["phone"] == "[REDACTED:PHONE]"


@pytest.mark.asyncio
async def test_toxicity_annotator_sets_flag():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {
                    "inbound": [],
                    "outbound": [
                        {"name": "toxicity_mark", "enabled": True},
                    ],
                }
            },
            "guardrail": {
                "toxicity": {"enabled": True, "threshold": 0.5}
            },
        }
    )

    content = "This is extremely toxic message"
    metadata = {}

    gateway = MemoryApiGateway(buffer_service=MockBufferService(content, metadata), db_service=None)
    resp = await gateway.process_request({"user_id": "u", "query": "hello"}, operation_type=OperationType.QUERY)

    res0 = resp["data"]["results"][0]
    assert res0["metadata"].get("toxicity_flag") is True
    assert 0.0 <= res0["metadata"].get("toxicity_score", 0.0) <= 1.0

