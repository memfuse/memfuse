import pytest
from typing import List, Dict, Any

from src.memfuse_core.gateway.api_gateway import MemoryApiGateway
from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.persistence.in_memory_store import InMemoryStore
from src.memfuse_core.persistence.adapters import make_retrieval_handler_from_store
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class FakeBufferServiceWithQB:
    def __init__(self, qb: QueryBuffer):
        self.qb = qb

    async def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        results = await self.qb.query(query_text=query, top_k=top_k, sort_by="score", order="desc", use_rerank=False)
        return {
            "status": "success",
            "code": 200,
            "data": {"results": results, "total": len(results)},
            "message": "ok",
            "errors": None,
        }


@pytest.mark.asyncio
async def test_gateway_with_querybuffer_plugins_and_outbound_filters(monkeypatch):
    # 1) Configure plugins and gateway outbound filters via global config
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "buffer_plugins": {
                "plugins": [
                    {"name": "score_clip", "enabled": True, "params": {"max": 0.8}},
                    {"name": "session_annotator", "enabled": True, "params": {"default_session_id": "sess-default", "default_agent_id": "agent-default"}},
                    {"name": "deduplicate", "enabled": True},
                ]
            },
            "gateway": {
                "pipeline": {
                    "inbound": [],
                    "outbound": [
                        {"name": "pii_redact", "enabled": True},
                        {"name": "output_remove", "enabled": True},
                    ],
                }
            },
            "guardrail": {
                "output": {"enabled": True, "remove_fields": ["metadata.source"]},
                "pii": {"enabled": True, "redact": True},
            },
        }
    )

    # 2) Build storage and retrieval handler
    store = InMemoryStore()
    await store.initialize()
    # Two items, one with PII
    await store.add(item=__import__("src.memfuse_core.models.core", fromlist=["Item"]).Item(id="a", content="email me at a@b.com", metadata={}))
    await store.add(item=__import__("src.memfuse_core.models.core", fromlist=["Item"]).Item(id="b", content="just some text", metadata={}))

    rh = make_retrieval_handler_from_store(store)

    # 3) Create QueryBuffer AFTER config reload so it loads plugins
    qb = QueryBuffer(retrieval_handler=rh, rerank_handler=None, max_size=10)

    # 4) Monkeypatch buffer retrieval to introduce duplicates overlapping with storage
    async def fake_buffer_retrieve(query: str, user_id=None, session_id=None, top_k=5, hybrid_buffer=None, round_buffer=None):
        return [
            {"id": "a", "content": "email me at a@b.com", "score": 0.95, "metadata": {"source": "hybrid"}},
            {"id": "a", "content": "duplicate same id", "score": 0.7, "metadata": {"source": "hybrid"}},
        ]

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_buffer_retrieve)

    # 5) Wire gateway with fake buffer service that uses our QueryBuffer
    gateway = MemoryApiGateway(buffer_service=FakeBufferServiceWithQB(qb), db_service=None)

    # 6) Send a request through gateway
    request = {"user_id": "user-1", "agent_id": "agent-1", "session_id": "sess-1", "query": "email", "top_k": 5}
    resp = await gateway.process_request(request)

    assert resp["status"] == "success"
    results = resp["data"]["results"]

    # DeduplicatePlugin should ensure only one item with id "a" remains
    ids = [r.get("id") for r in results]
    assert ids.count("a") == 1

    # ScoreClipPlugin should clip the top score to <= 0.8 before being renamed
    # After gateway transform, score is at relevance_score and 'score' removed
    for r in results:
        assert "relevance_score" in r and "score" not in r
        assert r["relevance_score"] <= 0.8

    # PII should be redacted by outbound filter
    contents = " ".join([r.get("content", "") for r in results if r.get("content")])
    assert "[REDACTED:EMAIL]" in contents

    # metadata.source should be removed by outbound/guardrail output removal
    for r in results:
        assert "metadata" in r
        assert "source" not in r["metadata"]

