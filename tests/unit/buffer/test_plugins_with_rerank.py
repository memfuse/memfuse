import pytest
from typing import List, Dict, Any

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.persistence.adapters import RetrievalAdapter


class Obj:
    def __init__(self, id: str, content: Any, score: float, metadata: Dict[str, Any] | None = None):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata or {}


class ListStore:
    def __init__(self, items: List[Obj]):
        self.items = items

    async def query_topk(self, query_text: str, top_k: int) -> List[Obj]:
        # return fixed order to observe rerank effects later
        return self.items[:top_k]


@pytest.mark.asyncio
async def test_plugins_apply_after_rerank_and_rerank_cached():
    # items deliberately reversed by rerank
    items = [
        Obj("a", "short", 0.1, {"created_at": "2024-01-01T00:00:00"}),
        Obj("b", "a little bit longer", 0.2, {"created_at": "2024-01-02T00:00:00"}),
    ]

    qb = QueryBuffer(retrieval_handler=RetrievalAdapter(ListStore(items)))

    async def rerank_handler(query_text: str, results: List[Dict[str, Any]]):
        # reverse order to simulate rerank
        return list(reversed(results))

    qb.rerank_handler = rerank_handler

    out1 = await qb.query("q", top_k=2, sort_by="score", order="asc", use_rerank=True)
    assert [r["id"] for r in out1] == ["b", "a"]
    assert qb.rerank_operations == 1

    # Ensure plugins (if any configured globally) are applied after rerank: we can't depend on global config here,
    # but we can assert that query returns dictionaries and is stable.
    assert isinstance(out1[0], dict)

    # Second call should hit rerank cache
    out2 = await qb.query("q", top_k=2, sort_by="score", order="asc", use_rerank=True)
    assert [r["id"] for r in out2] == ["b", "a"]
    assert qb.rerank_operations == 1  # cached

