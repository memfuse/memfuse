import pytest

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


@pytest.mark.asyncio
async def test_result_enricher_and_field_keep_remove(monkeypatch):
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "buffer_plugins": {
                "plugins": [
                    {"name": "result_enricher", "enabled": True, "params": {"stage": "after_merge", "include_query_len": True}},
                    {"name": "field_keep_or_remove", "enabled": True, "params": {"remove_fields": ["metadata.internal", "metadata.debug"]}},
                ]
            }
        }
    )

    qb = QueryBuffer(retrieval_handler=None, rerank_handler=None, max_size=10)

    async def fake_buffer_retrieve(query: str, user_id=None, session_id=None, top_k=5, hybrid_buffer=None, round_buffer=None):
        return [
            {"id": "x1", "content": "hello", "score": 0.42, "metadata": {"internal": "keep?", "foo": "bar"}},
            {"id": "x2", "content": "world", "score": 0.35, "metadata": {"debug": "drop", "foo": "baz"}},
        ]

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_buffer_retrieve)

    res = await qb.query("hello world", top_k=5, use_rerank=False)

    assert len(res) == 2
    # FieldKeepOrRemovePlugin should remove metadata.internal and metadata.debug
    assert "internal" not in res[0].get("metadata", {})
    assert "debug" not in res[1].get("metadata", {})

    # ResultEnricherPlugin should add observability info
    for r in res:
        md = r.get("metadata", {})
        obs = md.get("observability", {})
        assert obs.get("stage") == "after_merge"
        assert obs.get("query_len") == len("hello world")

