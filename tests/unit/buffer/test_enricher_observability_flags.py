import pytest
import asyncio
from typing import List, Dict, Any

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.buffer.plugins import ResultEnricherPlugin


async def _async_return(x):
    return x


def make_results() -> List[Dict[str, Any]]:
    return [
        {"id": "r1", "content": "A", "score": 0.2, "metadata": {}},
        {"id": "r2", "content": "B", "score": 0.8, "metadata": {}},
    ]


@pytest.mark.asyncio
async def test_enricher_includes_rerank_cache_and_plugin_order():
    # Rerank handler that reverses list to simulate work
    async def rr(query_text: str, results: List[Dict[str, Any]]):
        await asyncio.sleep(0)  # yield
        return list(reversed(results))

    qb = QueryBuffer(retrieval_handler=None, rerank_handler=rr)

    # Monkeypatch buffer retrieval to provide deterministic buffer results
    results = make_results()

    async def fake_retrieve(**kwargs):
        return list(results)

    qb.buffer_retrieval.retrieve = fake_retrieve  # type: ignore

    # Inject only the result_enricher plugin with flags
    qb._plugins = [
        ResultEnricherPlugin(
            include_query_len=True,
            include_rerank_cache=True,
            include_plugin_order=True,
        )
    ]

    # First call: cache miss for rerank -> rerank_cache_hit False
    out1 = await qb.query("hello", top_k=2, use_rerank=True)
    assert len(out1) == 2
    obs1 = out1[0]["metadata"]["observability"]
    assert obs1.get("stage") == "after_merge"
    assert isinstance(obs1.get("query_len"), int)
    assert obs1.get("plugin_order") == ["ResultEnricherPlugin"]
    assert obs1.get("rerank_cache_hit") is False

    # Second call: clear QueryBuffer result cache to force pipeline, rerank cache should hit
    await qb.clear_cache()
    out2 = await qb.query("hello", top_k=2, use_rerank=True)
    obs2 = out2[0]["metadata"]["observability"]
    assert obs2.get("rerank_cache_hit") is True

