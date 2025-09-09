import pytest
from typing import Any, Dict

from src.memfuse_core.gateway.api_gateway import MemoryApiGateway
from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.persistence.keyword_store import KeywordStore
from src.memfuse_core.persistence.adapters import RetrievalAdapter
from src.memfuse_core.models.core import Item
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class FakeBufferServiceWithQB:
    def __init__(self, qb: QueryBuffer):
        self.qb = qb

    async def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        results = await self.qb.query(query_text=query, top_k=top_k, use_rerank=False)
        return {"status": "success", "code": 200, "data": {"results": results, "total": len(results)}, "message": "ok", "errors": None}


@pytest.mark.asyncio
async def test_end2end_keyword_store_with_plugins_and_filters(monkeypatch):
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "buffer_plugins": {
                "plugins": [
                    {"name": "result_enricher", "enabled": True, "params": {"stage": "merge", "include_query_len": True}},
                    {"name": "field_keep_or_remove", "enabled": True, "params": {"remove_fields": ["metadata.source"]}},
                ]
            },
            "gateway": {
                "pipeline": {"inbound": [], "outbound": [{"name": "pii_redact", "enabled": True}]}
            },
            "guardrail": {"pii": {"enabled": True, "redact": True}},
        }
    )

    store = KeywordStore()
    await store.initialize()
    await store.add(Item(id="k1", content="contact: a@b.com keyword", metadata={"trace": "t1"}))
    await store.add(Item(id="k2", content="no match", metadata={}))

    adapter = RetrievalAdapter(store).as_handler()
    qb = QueryBuffer(retrieval_handler=adapter, rerank_handler=None, max_size=5)

    async def fake_buffer_retrieve(query: str, user_id=None, session_id=None, top_k=5, hybrid_buffer=None, round_buffer=None):
        # Force storage usage only
        return []

    monkeypatch.setattr(qb.buffer_retrieval, "retrieve", fake_buffer_retrieve)

    gateway = MemoryApiGateway(buffer_service=FakeBufferServiceWithQB(qb), db_service=None)
    resp = await gateway.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "keyword", "top_k": 3})

    assert resp["status"] == "success"
    results = resp["data"]["results"]
    assert len(results) >= 1 and results[0]["id"] == "k1"
    # PII redacted
    assert "[REDACTED:EMAIL]" in results[0]["content"]
    # source removed by plugin
    assert "source" not in results[0]["metadata"]
    # observability added
    assert "observability" in results[0]["metadata"]

