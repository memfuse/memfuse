"""Tests for ConfigOutputRemovalFilter outbound filter."""

import asyncio
from memfuse_core.gateway.filters import ConfigOutputRemovalFilter
from memfuse_core.interfaces.gateway_interface import RequestContext
from memfuse_core.utils.global_config_manager import get_global_config_manager


def test_config_output_removal_filter_applies_fields():
    gcm = get_global_config_manager()
    asyncio.run(gcm.hot_reload({
        "guardrail": {
            "output": {"enabled": True, "remove_fields": ["metadata.level", "metadata.source"]}
        }
    }))

    flt = ConfigOutputRemovalFilter()
    ctx = RequestContext(user_id="u")

    resp = {
        "status": "success",
        "code": 200,
        "data": {
            "results": [
                {"id": "1", "relevance_score": 0.8, "memory_type": "episodic", "metadata": {"user_id": "u", "scope": "in_session", "level": "debug", "source": "internal"}},
                {"id": "2", "relevance_score": 0.7, "memory_type": "episodic", "metadata": {"user_id": "u", "scope": "in_session", "source": "ext"}},
            ],
            "total": 2,
        },
        "message": "ok",
    }

    out = flt.apply(resp, ctx)
    meta0 = out["data"]["results"][0]["metadata"]
    meta1 = out["data"]["results"][1]["metadata"]
    assert "level" not in meta0
    assert "source" not in meta0
    assert "source" not in meta1


def test_config_output_removal_filter_disabled_noop():
    gcm = get_global_config_manager()
    asyncio.run(gcm.hot_reload({
        "guardrail": {
            "output": {"enabled": False, "remove_fields": ["metadata.level"]}
        }
    }))

    flt = ConfigOutputRemovalFilter()
    ctx = RequestContext(user_id="u")

    resp = {
        "status": "success",
        "code": 200,
        "data": {
            "results": [
                {"id": "1", "relevance_score": 0.8, "memory_type": "episodic", "metadata": {"user_id": "u", "scope": "in_session", "level": "debug"}},
            ],
            "total": 1,
        },
        "message": "ok",
    }

    out = flt.apply(resp, ctx)
    meta0 = out["data"]["results"][0]["metadata"]
    assert "level" in meta0

