import pytest

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


@pytest.mark.asyncio
async def test_session_annotator_uses_ctx_values(monkeypatch):
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "buffer_plugins": {
                "plugins": [
                    {"name": "session_annotator", "enabled": True},
                ]
            }
        }
    )

    qb = QueryBuffer(retrieval_handler=None, rerank_handler=None, max_size=5)

    async def fake_buffer_retrieve(query: str, user_id=None, session_id=None, top_k=5, hybrid_buffer=None, round_buffer=None):
        return [
            {"id": "x", "content": "c", "score": 0.5, "metadata": {}},
        ]

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_buffer_retrieve)

    res = await qb.query("q", top_k=3, use_rerank=False, user_id="u1", session_id="s1", agent_id="a1")
    assert len(res) == 1
    md = res[0]["metadata"]
    assert md.get("session_id") == "s1"
    assert md.get("agent_id") == "a1"

