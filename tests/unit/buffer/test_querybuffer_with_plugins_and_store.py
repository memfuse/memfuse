import pytest

from memfuse_core.buffer.query_buffer import QueryBuffer
from memfuse_core.persistence.in_memory_store import InMemoryStore
from memfuse_core.persistence.adapters import make_retrieval_handler_from_store
from memfuse_core.utils.global_config_manager import get_global_config_manager


@pytest.mark.asyncio
async def test_querybuffer_uses_store_and_plugins(monkeypatch):
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "buffer_plugins": {
            "plugins": [
                {"name": "session_annotator", "enabled": True, "params": {"default_session_id": "s1", "default_agent_id": "a1"}},
                {"name": "score_clip", "enabled": True, "params": {"max": 0.9}},
                {"name": "deduplicate", "enabled": True, "params": {"key": "id"}},
            ]
        }
    })

    # Prepare in-memory store with items
    store = InMemoryStore()
    await store.initialize()
    from memfuse_core.models.core import Item
    await store.add(Item(id="x", content="hello alpha", metadata={}))
    await store.add(Item(id="y", content="hello beta", metadata={}))
    await store.add(Item(id="z", content="gamma delta", metadata={}))

    qb = QueryBuffer()
    qb.retrieval_handler = make_retrieval_handler_from_store(store)

    # Monkeypatch buffer retrieval to add potential duplicates with different scores
    async def fake_buffer_retrieve(**kwargs):
        return [
            {"id": "x", "content": "hello alpha", "score": 0.95, "metadata": {}},
            {"id": "x", "content": "hello alpha", "score": 0.90, "metadata": {}},
        ]

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_buffer_retrieve)

    results = await qb.query("hello", top_k=5, use_rerank=False)

    # Deduplicate should collapse duplicates of id=x
    ids = [r.get("id") for r in results]
    assert ids.count("x") == 1

    # SessionAnnotator should add defaults
    md = results[0].get("metadata", {})
    assert md.get("session_id") == "s1"
    assert md.get("agent_id") == "a1"

    # ScoreClip should cap any score > 0.9 to 0.9
    assert all(r.get("score", 0) <= 0.9 for r in results)

