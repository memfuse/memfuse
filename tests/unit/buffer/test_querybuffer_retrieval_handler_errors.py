import pytest
from typing import Any, Dict, List

from src.memfuse_core.buffer.query_buffer import QueryBuffer


@pytest.mark.asyncio
async def test_query_returns_empty_when_retrieval_handler_raises():
    async def bad_handler(query_text: str, max_results: int) -> List[Dict[str, Any]]:
        raise RuntimeError("boom")

    qb = QueryBuffer(retrieval_handler=bad_handler)

    out = await qb.query("q", top_k=5)
    assert out == []

