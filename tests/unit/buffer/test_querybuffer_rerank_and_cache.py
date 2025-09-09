import pytest

from src.memfuse_core.buffer.query_buffer import QueryBuffer


@pytest.mark.asyncio
async def test_querybuffer_rerank_and_cache(monkeypatch):
    calls = {"rerank": 0}

    async def fake_buffer_retrieve(query: str, user_id=None, session_id=None, top_k=5, hybrid_buffer=None, round_buffer=None):
        # Deliberately unsorted by score
        return [
            {"id": "b", "content": "short", "score": 0.2, "metadata": {}},
            {"id": "a", "content": "a much longer content", "score": 0.9, "metadata": {}},
        ]

    async def simple_reranker(query_text: str, results):
        calls["rerank"] += 1
        # Sort by content length ascending to flip the order
        return sorted(results, key=lambda r: len(r.get("content", "")))

    qb = QueryBuffer(retrieval_handler=None, rerank_handler=simple_reranker, max_size=10)
    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_buffer_retrieve)

    # First query with rerank
    out1 = await qb.query("q1", top_k=2, use_rerank=True)
    assert [r["id"] for r in out1] == ["b", "a"]  # reordered by reranker
    assert qb.rerank_operations == 1
    assert calls["rerank"] == 1

    # Second query identical should hit rerank cache; no new rerank
    out2 = await qb.query("q1", top_k=2, use_rerank=True)
    assert [r["id"] for r in out2] == ["b", "a"]
    assert qb.rerank_operations == 1
    assert calls["rerank"] == 1

