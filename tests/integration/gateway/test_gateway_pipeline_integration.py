"""Integration test for API→Gateway pipeline filter execution order.

This test ensures outbound filters configured via gateway.pipeline are applied
before guardrail validation and auditing, removing configured fields from
response payloads.
"""

import asyncio
import pytest

from memfuse_core.gateway.api_gateway import MemoryApiGateway
from memfuse_core.interfaces.gateway_interface import OperationType
from memfuse_core.utils.global_config_manager import get_global_config_manager


class MockBufferService:
    async def query(self, query: str, top_k: int = 5, **kwargs):
        # Return a minimal, valid response that will pass the gateway processors
        return {
            "status": "success",
            "code": 200,
            "data": {
                "results": [
                    {
                        "id": "1",
                        "relevance_score": 0.88,
                        "memory_type": "episodic",
                        "content": "hello world",
                        "created_at": None,
                        "metadata": {
                            "user_id": "u",
                            "scope": "in_session",
                            "internal_note": "secret",  # to be removed by outbound filter
                        },
                    }
                ],
                "total": 1,
            },
            "message": "ok",
        }


@pytest.mark.asyncio
async def test_outbound_output_remove_filter_runs_before_guardrail():
    # Setup global config: enable outbound output_remove and configure field
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {
                    "inbound": [],
                    "outbound": [
                        {"name": "output_remove", "enabled": True},
                    ],
                }
            },
            "guardrail": {
                "output": {
                    "enabled": True,
                    "remove_fields": ["metadata.internal_note"],
                }
            },
        }
    )

    gateway = MemoryApiGateway(buffer_service=MockBufferService(), db_service=None)

    response = await gateway.process_request(
        {"user_id": "u", "query": "hello"}, operation_type=OperationType.QUERY
    )

    assert response.get("status") == "success"
    results = response.get("data", {}).get("results", [])
    assert results and isinstance(results, list)
    meta = results[0].get("metadata", {})
    assert "internal_note" not in meta  # removed by outbound filter prior to validation

