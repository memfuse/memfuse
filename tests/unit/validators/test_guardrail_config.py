"""Tests for guardrail wiring with global config."""

import asyncio
from memfuse_core.validators.guardrails import MemoryGuardrail
from memfuse_core.interfaces.gateway_interface import RequestContext
from memfuse_core.utils.global_config_manager import get_global_config_manager


def _sample_response():
    return {
        "status": "success",
        "code": 200,
        "data": {
            "results": [
                {
                    "id": "1",
                    "relevance_score": 0.9,
                    "memory_type": "episodic",
                    "content": "hello",
                    "created_at": None,
                    "metadata": {
                        "user_id": "u",
                        "scope": "in_session",
                        "source": "internal",
                        "retrieval": "debug",
                    },
                }
            ],
            "total": 1,
        },
        "message": "ok",
    }


def test_guardrail_output_remove_fields_applied():
    gcm = get_global_config_manager()
    asyncio.run(gcm.hot_reload({
        "guardrail": {
            "output": {
                "enabled": True,
                "remove_fields": ["metadata.source", "metadata.retrieval"],
            }
        }
    }))

    guardrail = MemoryGuardrail()
    resp = _sample_response()
    ctx = RequestContext(user_id="u")

    # Note: In the real pipeline, field removal should happen via outbound filters
    # before validation. Here we directly audit (which applies output removal) and
    # then assert the fields are removed.
    guardrail.audit_response(resp, ctx)

    meta = resp["data"]["results"][0]["metadata"]
    assert "source" not in meta
    assert "retrieval" not in meta


def test_guardrail_output_disabled_no_change():
    gcm = get_global_config_manager()
    asyncio.run(gcm.hot_reload({
        "guardrail": {
            "output": {
                "enabled": False,
                "remove_fields": ["metadata.source"],
            }
        }
    }))

    guardrail = MemoryGuardrail()
    resp = _sample_response()
    ctx = RequestContext(user_id="u")

    guardrail.audit_response(resp, ctx)
    meta = resp["data"]["results"][0]["metadata"]
    # field should still exist when disabled
    assert "source" in meta

