#!/usr/bin/env python3
"""
Core manual test for Schema + M3 integration (gateway-level, no HTTP server).

Usage:
  poetry run python scripts/test_schema_m3.py

What it does:
  - Creates a MemoryApiGateway with a mock BufferService
  - Exercises:
      1) Query without session_id (scope must be null)
      2) Query with session_id (scope in_session / cross_session)
      3) Query with metadata.task (ensures task flows into result.metadata)
  - Prints normalized responses and validates key schema requirements

Note:
  This avoids real DB/LLM calls and is fast. For full API tests, consider
  running tests/contract/test_memory_api_contract.py under poetry/pytest.
"""

import json
from typing import Any, Dict, List

import asyncio

from memfuse_core.gateway.api_gateway import MemoryApiGateway
from memfuse_core.interfaces.gateway_interface import OperationType


class MockBufferService:
    def __init__(self, results: List[Dict[str, Any]]):
        self._results = results

    async def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        return {
            "status": "success",
            "data": {
                "results": self._results,
                "total": len(self._results),
            }
        }


def pretty(title: str, obj: Any):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


async def run():
    # Two episodic-like results with different session_ids, and one semantic-like
    raw_results = [
        {
            "id": "episodic-1",
            "content": "E1 content",
            "score": 0.88,
            "type": "chunk",
            "created_at": "2025-09-02T13:52:46.552383+00:00",
            "metadata": {"user_id": "user-1", "session_id": "sess-1", "source": "memory_database"},
        },
        {
            "id": "episodic-2",
            "content": "E2 content",
            "score": 0.73,
            "type": "message",
            "created_at": "2025-09-02T13:52:46.552383+00:00",
            "metadata": {"user_id": "user-1", "session_id": "sess-2", "level": "debug"},
        },
        {
            "id": "semantic-1",
            "score": 0.95,
            "type": "semantic",
            "created_at": "2025-09-02T13:52:46.552383+00:00",
            "derived_from": ["episodic-1"],
            "metadata": {"user_id": "user-1", "session_id": "sess-1", "retrieval": "semantic"},
            "fact": {"text": "semantic fact", "triples": None},
        },
    ]

    gw = MemoryApiGateway(buffer_service=MockBufferService(raw_results), db_service=None)

    # Case 1: Without session_id → scope must be null
    req1 = {
        "query": "environment",
        "user_id": "user-1",
        "agent_id": "agent-1",
        "top_k": 5,
    }
    resp1 = await gw.process_request(req1, OperationType.QUERY)
    pretty("Without session_id", resp1)
    assert resp1["status"] == "success"
    for r in resp1["data"]["results"]:
        assert r["metadata"]["scope"] is None
        assert "score" not in r and "type" not in r
        for bad in ("level", "retrieval", "source"):
            assert bad not in r["metadata"]

    # Case 2: With session_id → scope in_session/cross_session
    req2 = {
        "query": "environment",
        "user_id": "user-1",
        "agent_id": "agent-1",
        "session_id": "sess-1",
        "top_k": 5,
        "metadata": {"task": "op_websearch_memory", "mode": None}
    }
    resp2 = await gw.process_request(req2, OperationType.QUERY)
    pretty("With session_id", resp2)
    assert resp2["status"] == "success"
    metas = [r["metadata"] for r in resp2["data"]["results"]]
    assert any(m.get("scope") == "in_session" for m in metas)
    assert any(m.get("scope") == "cross_session" for m in metas)
    # Ensure schema-required metadata present
    for m in metas:
        for key in ("user_id", "agent_id", "session_id", "session_name"):
            assert key in m

    # Case 3: With task only (no guidance required) → metadata.task should flow through
    req3 = {
        "query": "environment",
        "user_id": "user-1",
        "agent_id": "agent-1",
        "top_k": 3,
        "metadata": {"task": "op_websearch_memory", "mode": None}
    }
    resp3 = await gw.process_request(req3, OperationType.QUERY)
    pretty("With task (no guidance by default)", resp3)
    assert resp3["status"] == "success"
    for r in resp3["data"]["results"]:
        assert r["metadata"].get("task") == "op_websearch_memory"

    print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(run())

