import pytest
from typing import Any, Dict

from memfuse_core.gateway.api_gateway import MemoryApiGateway
from memfuse_core.utils.global_config_manager import get_global_config_manager


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
async def test_sensitive_word_action_mask_default():
    """Test default mask action on content."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "gateway": {
            "pipeline": {"inbound": [], "outbound": [{"name": "sensitive_word", "enabled": True}]},
        },
        "guardrail": {
            "sensitive": {
                "enabled": True,
                "words": ["secret", "forbidden"],
                "mask_token": "[MASKED]",
                "action": "mask"  # explicit, though it's default
            }
        }
    })

    results = [
        {"id": "1", "content": "This contains secret information and forbidden data"},
        {"id": "2", "content": "Normal content here"}
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    res = resp["data"]["results"]
    assert len(res) == 2
    assert res[0]["content"] == "This contains [MASKED] information and [MASKED] data"
    assert res[0]["metadata"]["sensitive_hit"] is True
    assert res[1]["content"] == "Normal content here"
    assert "sensitive_hit" not in res[1].get("metadata", {})


@pytest.mark.asyncio
async def test_sensitive_word_action_flag():
    """Test flag action - content unchanged, only metadata marked."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "gateway": {
            "pipeline": {"inbound": [], "outbound": [{"name": "sensitive_word", "enabled": True}]},
        },
        "guardrail": {
            "sensitive": {
                "enabled": True,
                "words": ["secret"],
                "action": "flag"
            }
        }
    })

    results = [{"id": "1", "content": "This contains secret information"}]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    res = resp["data"]["results"]
    assert len(res) == 1
    assert res[0]["content"] == "This contains secret information"  # unchanged
    assert res[0]["metadata"]["sensitive_hit"] is True


@pytest.mark.asyncio
async def test_sensitive_word_action_drop():
    """Test drop action - sensitive results removed entirely."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "gateway": {
            "pipeline": {"inbound": [], "outbound": [{"name": "sensitive_word", "enabled": True}]},
        },
        "guardrail": {
            "sensitive": {
                "enabled": True,
                "words": ["secret"],
                "action": "drop"
            }
        }
    })

    results = [
        {"id": "1", "content": "This contains secret information"},
        {"id": "2", "content": "Normal content here"},
        {"id": "3", "content": "Another secret thing"}
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    res = resp["data"]["results"]
    assert len(res) == 1  # only the normal content remains
    assert res[0]["id"] == "2"
    assert res[0]["content"] == "Normal content here"


@pytest.mark.asyncio
async def test_sensitive_word_recurse_metadata():
    """Test recursive metadata processing."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "gateway": {
            "pipeline": {"inbound": [], "outbound": [{"name": "sensitive_word", "enabled": True}]},
        },
        "guardrail": {
            "sensitive": {
                "enabled": True,
                "words": ["secret"],
                "mask_token": "[REDACTED]",
                "recurse_metadata": True,
                "action": "mask"
            }
        }
    })

    results = [
        {
            "id": "1", 
            "content": "Normal content",
            "metadata": {
                "title": "secret document",
                "nested": {
                    "description": "contains secret info"
                },
                "tags": ["public", "secret", "internal"]
            }
        }
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    res = resp["data"]["results"]
    assert len(res) == 1
    r = res[0]
    assert r["content"] == "Normal content"  # unchanged
    assert r["metadata"]["title"] == "[REDACTED] document"
    assert r["metadata"]["nested"]["description"] == "contains [REDACTED] info"
    assert r["metadata"]["tags"] == ["public", "[REDACTED]", "internal"]
    assert r["metadata"]["sensitive_hit"] is True


@pytest.mark.asyncio
async def test_sensitive_word_recurse_metadata_disabled():
    """Test that metadata is not processed when recurse_metadata is False."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "gateway": {
            "pipeline": {"inbound": [], "outbound": [{"name": "sensitive_word", "enabled": True}]},
        },
        "guardrail": {
            "sensitive": {
                "enabled": True,
                "words": ["secret"],
                "recurse_metadata": False,  # explicit disable
                "action": "mask"
            }
        }
    })

    results = [
        {
            "id": "1", 
            "content": "Normal content",
            "metadata": {"title": "secret document"}
        }
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    res = resp["data"]["results"]
    assert len(res) == 1
    r = res[0]
    assert r["metadata"]["title"] == "secret document"  # unchanged
    assert "sensitive_hit" not in r["metadata"]  # no hit detected
