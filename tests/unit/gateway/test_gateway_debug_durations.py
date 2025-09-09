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
async def test_gateway_debug_includes_durations_when_enabled():
    """Test that durations are included in debug metadata when enabled."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "gateway": {
            "pipeline": {"inbound": [], "outbound": []},
            "debug": {
                "enabled": True,
                "include_durations": True,
            },
        },
        "guardrail": {"output": {"enabled": False}},
    })

    results = [
        {"id": "1", "content": "test content", "score": 0.8},
        {"id": "2", "content": "another test", "score": 0.7}
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({
        "user_id": "u", "agent_id": "a", "session_id": "s", 
        "query": "test query", "top_k": 5
    })

    # Check that durations are present in top-level observability
    obs = resp["data"].get("metadata", {}).get("observability", {})
    durations = obs.get("durations")
    
    assert isinstance(durations, dict)
    assert "response_processor" in durations
    assert "metadata_enricher" in durations
    assert "scope_calculator" in durations
    assert "field_remover" in durations
    assert "overall" in durations
    
    # Check that all durations are positive numbers (in milliseconds)
    for stage, duration in durations.items():
        assert isinstance(duration, (int, float))
        assert duration >= 0
        # Should be reasonable timing (less than 1 second = 1000ms for unit tests)
        assert duration < 1000


@pytest.mark.asyncio
async def test_gateway_debug_no_durations_when_disabled():
    """Test that durations are not included when debug is disabled."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "gateway": {
            "pipeline": {"inbound": [], "outbound": []},
            "debug": {
                "enabled": False,  # disabled
                "include_durations": True,  # this should be ignored
            },
        },
        "guardrail": {"output": {"enabled": False}},
    })

    results = [{"id": "1", "content": "test content", "score": 0.8}]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({
        "user_id": "u", "agent_id": "a", "session_id": "s", 
        "query": "test query", "top_k": 5
    })

    # Check that no debug metadata is present
    obs = resp["data"].get("metadata", {}).get("observability", {})
    assert "durations" not in obs


@pytest.mark.asyncio
async def test_gateway_debug_no_durations_when_include_durations_false():
    """Test that durations are not included when include_durations is false."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "gateway": {
            "pipeline": {"inbound": [], "outbound": []},
            "debug": {
                "enabled": True,
                "include_durations": False,  # explicitly disabled
            },
        },
        "guardrail": {"output": {"enabled": False}},
    })

    results = [{"id": "1", "content": "test content", "score": 0.8}]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({
        "user_id": "u", "agent_id": "a", "session_id": "s", 
        "query": "test query", "top_k": 5
    })

    # Check that durations are not present
    obs = resp["data"].get("metadata", {}).get("observability", {})
    assert "durations" not in obs


@pytest.mark.asyncio
async def test_gateway_debug_durations_combined_with_other_debug_fields():
    """Test that durations work alongside other debug fields."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "gateway": {
            "pipeline": {"inbound": [], "outbound": []},
            "debug": {
                "enabled": True,
                "include_durations": True,
                "include_rerank_cache_hit": True,
            },
        },
        "guardrail": {"output": {"enabled": False}},
    })

    results = [
        {
            "id": "1", 
            "content": "test content", 
            "score": 0.8,
            "metadata": {
                "observability": {"rerank_cache_hit": True}
            }
        }
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({
        "user_id": "u", "agent_id": "a", "session_id": "s", 
        "query": "test query", "top_k": 5
    })

    # Check that both durations and cache hit are present
    obs = resp["data"].get("metadata", {}).get("observability", {})
    assert "durations" in obs
    assert "rerank_cache_hit" in obs
    assert obs["rerank_cache_hit"] is True
    
    durations = obs["durations"]
    assert isinstance(durations, dict)
    assert len(durations) == 5  # four transformation stages + overall
