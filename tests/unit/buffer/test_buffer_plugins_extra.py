import asyncio
import pytest

from memfuse_core.buffer.query_buffer import QueryBuffer
from memfuse_core.utils.global_config_manager import get_global_config_manager


@pytest.mark.asyncio
async def test_session_annotator_plugin_adds_defaults(monkeypatch):
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "buffer_plugins": {
            "plugins": [
                {"name": "session_annotator", "enabled": True, "params": {"default_session_id": "s1", "default_agent_id": "a1"}},
            ]
        }
    })

    qb = QueryBuffer()

    async def fake_retrieve(**kwargs):
        return [
            {"id": "a", "score": 0.5, "content": "hello", "metadata": {}},
        ]

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_retrieve)

    results = await qb.query("q", top_k=1, use_rerank=False)
    md = results[0]["metadata"]
    assert md.get("session_id") == "s1"
    assert md.get("agent_id") == "a1"


@pytest.mark.asyncio
async def test_deduplicate_plugin_removes_duplicates(monkeypatch):
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "buffer_plugins": {
            "plugins": [
                {"name": "deduplicate", "enabled": True, "params": {"key": "id"}},
            ]
        }
    })

    qb = QueryBuffer()

    async def fake_retrieve(**kwargs):
        return [
            {"id": "x", "score": 0.8, "content": "a", "metadata": {}},
            {"id": "x", "score": 0.7, "content": "a", "metadata": {}},
            {"score": 0.6, "content": "b", "metadata": {}},
            {"score": 0.6, "content": "b", "metadata": {}},
        ]

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_retrieve)

    results = await qb.query("q", top_k=10, use_rerank=False)

    # Expect 1 for id=x, and 1 for content b tuple fallback => total 2 unique
    assert len(results) == 2

