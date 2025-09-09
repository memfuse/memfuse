import pytest

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


@pytest.mark.asyncio
async def test_plugin_order_dedupe_then_enrich_then_remove(monkeypatch):
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "buffer_plugins": {
                "plugins": [
                    {"name": "deduplicate", "enabled": True, "params": {}},
                    {"name": "result_enricher", "enabled": True, "params": {"stage": "merge", "include_query_len": True}},
                    {"name": "field_keep_or_remove", "enabled": True, "params": {"remove_fields": ["metadata.observability.query_len"]}},
                ]
            }
        }
    )

    qb = QueryBuffer(retrieval_handler=None, rerank_handler=None, max_size=10)

    async def fake_buffer_retrieve(query: str, user_id=None, session_id=None, top_k=5, hybrid_buffer=None, round_buffer=None):
        return [
            {"id": "a", "content": "x", "score": 0.9, "metadata": {}},
            {"id": "a", "content": "x", "score": 0.5, "metadata": {}},
        ]

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_buffer_retrieve)

    res = await qb.query("q", top_k=10, use_rerank=False)
    assert len(res) == 1  # deduplicated first
    md = res[0]["metadata"]
    assert "observability" in md  # enriched second
    assert "query_len" not in md.get("observability", {})  # removed last

