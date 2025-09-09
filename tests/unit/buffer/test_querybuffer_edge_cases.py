import pytest

from src.memfuse_core.buffer.query_buffer import QueryBuffer


@pytest.mark.asyncio
async def test_topk_zero_returns_empty(monkeypatch):
    qb = QueryBuffer(retrieval_handler=None, rerank_handler=None, max_size=10)

    async def fake_buffer_retrieve(query: str, user_id=None, session_id=None, top_k=5, hybrid_buffer=None, round_buffer=None):
        return [
            {"id": "a", "content": "x", "score": 0.9, "metadata": {}},
            {"id": "b", "content": "y", "score": 0.8, "metadata": {}},
        ]

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_buffer_retrieve)
    out = await qb.query("q", top_k=0, use_rerank=False)
    assert out == []


@pytest.mark.asyncio
async def test_timestamp_sorting(monkeypatch):
    qb = QueryBuffer(retrieval_handler=None, rerank_handler=None, max_size=10, default_sort_by="timestamp")

    async def fake_buffer_retrieve(query: str, user_id=None, session_id=None, top_k=5, hybrid_buffer=None, round_buffer=None):
        return [
            {"id": "a", "content": "x", "score": 0.5, "metadata": {}, "created_at": "2025-01-01T00:00:00"},
            {"id": "b", "content": "y", "score": 0.9, "metadata": {}, "created_at": "2024-01-01T00:00:00"},
        ]

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_buffer_retrieve)

    out_desc = await qb.query("q", top_k=2, sort_by="timestamp", order="desc", use_rerank=False)
    assert [r["id"] for r in out_desc] == ["a", "b"]

    out_asc = await qb.query("q2", top_k=2, sort_by="timestamp", order="asc", use_rerank=False)
    assert [r["id"] for r in out_asc] == ["b", "a"]

