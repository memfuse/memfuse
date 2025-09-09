import pytest
from typing import List, Dict, Any

from src.memfuse_core.buffer.query_buffer import QueryBuffer


@pytest.mark.asyncio
async def test_querybuffer_sort_by_timestamp_handles_missing_and_varied_types():
    # retrieval handler returns only storage results; no buffer results
    async def retrieval_handler(query_text: str, max_results: int) -> List[Dict[str, Any]]:
        return [
            {"id": "n", "content": {"k": 1}, "score": 0.1, "created_at": None},
            {"id": "s", "content": "str", "score": 0.2, "created_at": "2024-01-02T00:00:00"},
            {"id": "t", "content": 123, "score": 0.3, "created_at": 1703980800},  # 2023-12-31
        ]

    qb = QueryBuffer(retrieval_handler=retrieval_handler)

    out_asc = await qb.query("q", top_k=3, sort_by="timestamp", order="asc", use_rerank=False)
    assert [r["id"] for r in out_asc] == ["n", "t", "s"]

    out_desc = await qb.query("q2", top_k=3, sort_by="timestamp", order="desc", use_rerank=False)
    assert [r["id"] for r in out_desc] == ["s", "t", "n"]

    # Ensure no crash and structure preserved even with non-string content
    assert isinstance(out_asc[0], dict) and isinstance(out_asc[0].get("content"), dict)

