import asyncio
import pytest

from memfuse_core.buffer.query_buffer import QueryBuffer
from memfuse_core.utils.global_config_manager import get_global_config_manager


@pytest.mark.asyncio
async def test_rag_annotator_and_score_clip_plugins(monkeypatch):
    # Enable buffer plugins via hot reload
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "buffer_plugins": {
            "plugins": [
                {"name": "rag_annotator", "enabled": True, "params": {"source": "buffer"}},
                {"name": "score_clip", "enabled": True, "params": {"min": 0.0, "max": 0.9}},
            ]
        }
    })

    qb = QueryBuffer()

    async def fake_retrieve(**kwargs):
        return [
            {"id": "a", "score": 1.5, "content": "hello", "metadata": {}},
            {"id": "b", "content": "world", "metadata": {"source": "hybrid"}},
        ]

    # Monkeypatch buffer retrieval to avoid touching buffers/storage
    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_retrieve)

    results = await qb.query("q", top_k=2, use_rerank=False)

    # Plugin effects:
    # - rag_annotator adds source to first item (missing)
    # - score_clip clamps score to <= 0.9
    assert results[0]["metadata"].get("source") in {"buffer", "hybrid"}
    assert results[1]["metadata"].get("source") in {"buffer", "hybrid"}

    scores = [r.get("score") for r in results]
    assert any(s is not None for s in scores)
    for s in scores:
        if s is not None:
            assert 0.0 <= s <= 0.9

