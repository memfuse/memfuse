import pytest

from src.memfuse_core.persistence.in_memory_store import InMemoryStore
from src.memfuse_core.persistence.adapters import RetrievalAdapter
from src.memfuse_core.models.core import Item
from src.memfuse_core.buffer.query_buffer import QueryBuffer


@pytest.mark.asyncio
async def test_retrieval_adapter_topk_and_fallback(monkeypatch):
    store = InMemoryStore()
    await store.initialize()
    await store.add(Item(id="a", content="alpha beta", metadata={}))
    await store.add(Item(id="b", content="gamma delta", metadata={}))

    adapter = RetrievalAdapter(store)
    out = await adapter("alpha", 1)
    assert isinstance(out, list) and len(out) == 1
    assert set(["id", "content", "score", "metadata"]).issubset(out[0].keys())

    # Remove query_topk on class to hit fallback path
    monkeypatch.delattr(InMemoryStore, "query_topk", raising=False)
    out2 = await adapter("alpha", 5)
    assert isinstance(out2, list) and len(out2) <= 1


@pytest.mark.asyncio
async def test_retrieval_adapter_as_handler_with_querybuffer(monkeypatch):
    store = InMemoryStore()
    await store.initialize()
    await store.add(Item(id="a", content="email me: a@b.com", metadata={}))

    adapter = RetrievalAdapter(store)
    qb = QueryBuffer(retrieval_handler=adapter.as_handler(), rerank_handler=None, max_size=5)

    async def fake_buffer_retrieve(query: str, user_id=None, session_id=None, top_k=5, hybrid_buffer=None, round_buffer=None):
        # No buffer results to force storage usage
        return []

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_buffer_retrieve)

    res = await qb.query("email", top_k=3, use_rerank=False)
    assert isinstance(res, list) and len(res) >= 1
    assert res[0]["id"] == "a"

