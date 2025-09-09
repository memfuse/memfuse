import asyncio
import pytest

from src.memfuse_core.gateway.api_gateway import MemoryApiGateway
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class FakeBufferService:
    async def query(self, query: str, top_k: int = 5, **kwargs):
        # Return a MemoryService-like response shape
        results = [
            {
                "id": "r1",
                "score": 0.95,
                "type": "episodic",
                "content": "Contact me at john.doe@example.com",
                "created_at": "2024-01-01T00:00:00Z",
                "metadata": {"source": "buffer"},
            },
            {
                "id": "r2",
                "score": 0.65,
                "type": "episodic",
                "content": "This looks toxic to me",
                "created_at": "2024-01-02T00:00:00Z",
                "metadata": {"source": "buffer"},
            },
        ]
        return {
            "status": "success",
            "code": 200,
            "data": {"results": results, "total": len(results)},
            "message": "ok",
            "errors": None,
        }


@pytest.mark.asyncio
async def test_gateway_pipeline_end_to_end_transforms_and_filters():
    # Initialize global config with gateway filters and guardrail settings
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
                "output": {"enabled": True, "remove_fields": ["metadata.source"]},
                "pii": {"enabled": True, "redact": True},
                "toxicity": {"enabled": True, "threshold": 0.5},
            },
        }
    )

    gateway = MemoryApiGateway(buffer_service=FakeBufferService(), db_service=None)

    request = {
        "user_id": "user-123",
        "agent_id": "agent-xyz",
        "session_id": "sess-1",
        "query": "test query",
        "top_k": 2,
        "metadata": {"task": "qa"},
    }

    resp = await gateway.process_request(request)

    assert resp["status"] == "success"
    assert resp["code"] == 200
    assert "data" in resp and "results" in resp["data"]

    results = resp["data"]["results"]
    assert len(results) == 2

    # Validate schema-transformed fields
    for r in results:
        assert "relevance_score" in r and "score" not in r
        assert "memory_type" in r and r["memory_type"] in {"episodic", "message", "chunk", "semantic"}
        assert "created_at" in r
        assert "metadata" in r
        md = r["metadata"]
        # Enriched required metadata by processors
        assert md.get("user_id") == "user-123"
        # Scope must be set based on session
        assert md.get("scope") in {"in_session", "cross_session", None}
        # Removed fields
        assert "source" not in md  # removed by processors or outbound filter

    # PII redaction should occur in content
    assert results[0]["content"].find("[REDACTED:EMAIL]") != -1

    # Toxicity annotation should be present on the second result
    md2 = results[1]["metadata"]
    assert md2.get("toxicity_flag") is True
    assert md2.get("toxicity_score", 0) >= 0.5

