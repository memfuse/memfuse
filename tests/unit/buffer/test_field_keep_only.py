import pytest

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


@pytest.mark.asyncio
async def test_field_keep_only(monkeypatch):
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "buffer_plugins": {
                "plugins": [
                    {"name": "field_keep_or_remove", "enabled": True, "params": {"keep_fields": ["id", "metadata"], "remove_fields": ["metadata.source"]}},
                ]
            }
        }
    )

    qb = QueryBuffer(retrieval_handler=None, rerank_handler=None, max_size=5)

    async def fake_buffer_retrieve(query: str, user_id=None, session_id=None, top_k=5, hybrid_buffer=None, round_buffer=None):
        return [
            {"id": "x", "content": "c", "score": 0.5, "metadata": {"source": "hybrid", "foo": "bar"}},
        ]

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_buffer_retrieve)

    res = await qb.query("q", top_k=3, use_rerank=False)
    assert len(res) == 1
    r = res[0]
    # Only id and metadata kept
    assert set(r.keys()) == {"id", "metadata"}
    assert "source" not in r["metadata"]

