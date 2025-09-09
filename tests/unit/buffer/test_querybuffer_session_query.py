import pytest
from typing import Any, Dict, List

from src.memfuse_core.buffer.query_buffer import QueryBuffer


class FakeChunk:
    def __init__(self, messages: List[Dict[str, Any]], session_id: str):
        self.messages = messages
        self.metadata = {"session_id": session_id}


class FakeHybrid:
    def __init__(self, session_id: str):
        self.chunks = [
            FakeChunk([
                {"id": "x", "content": "hyb-x", "created_at": "2024-01-02T00:00:00", "metadata": {"session_id": session_id}},
                {"id": "y", "content": "hyb-y", "created_at": "2024-01-03T00:00:00", "metadata": {"session_id": session_id}},
            ], session_id=session_id)
        ]


@pytest.mark.asyncio
async def test_query_by_session_merges_dedups_and_sorts_desc():
    session_id = "s1"

    async def retrieval_handler(query_text: str, max_results: int) -> List[Dict[str, Any]]:
        # storage returns duplicate id 'x' (should be overridden by hybrid copy), and unique 'z'
        return [
            {"id": "x", "content": "db-x", "created_at": "2024-01-01T00:00:00", "metadata": {"session_id": session_id}},
            {"id": "z", "content": "db-z", "created_at": "2024-01-01T12:00:00", "metadata": {"session_id": session_id}},
        ]

    qb = QueryBuffer(retrieval_handler=retrieval_handler)
    qb.set_hybrid_buffer(FakeHybrid(session_id))

    out = await qb.query_by_session(session_id, limit=5, sort_by="timestamp", order="desc")

    # Dedup by id: expect ids y, x (from hybrid), z (from storage)
    assert [m["id"] for m in out] == ["y", "x", "z"]
    # 'x' content should be from hybrid (hyb-x), not storage (db-x)
    x = next(o for o in out if o["id"] == "x")
    assert x["content"] == "hyb-x"

