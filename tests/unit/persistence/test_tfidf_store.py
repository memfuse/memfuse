import pytest

from src.memfuse_core.persistence.tfidf_store import TfidfStore
from src.memfuse_core.persistence.adapters import RetrievalAdapter
from src.memfuse_core.models.core import Item


@pytest.mark.asyncio
async def test_tfidf_store_ranks_relevant_first():
    store = TfidfStore()
    await store.initialize()
    await store.add(Item(id="d1", content="alpha beta beta", metadata={}))
    await store.add(Item(id="d2", content="gamma delta", metadata={}))

    adapter = RetrievalAdapter(store)
    res = await adapter("alpha beta", 2)
    assert [r["id"] for r in res] == ["d1", "d2"]
    assert res[0]["id"] == "d1" and res[0]["metadata"]["source"] == "tfidf"

