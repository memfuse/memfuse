import pytest

from src.memfuse_core.persistence.keyword_store import KeywordStore
from src.memfuse_core.persistence.graph_store import GraphStore
from src.memfuse_core.persistence.adapters import RetrievalAdapter
from src.memfuse_core.models.core import Item


@pytest.mark.asyncio
async def test_keyword_store_with_retrieval_adapter():
    store = KeywordStore()
    await store.initialize()
    await store.add(Item(id="k1", content="hello keyword world", metadata={}))
    await store.add(Item(id="k2", content="nothing matches", metadata={}))

    adapter = RetrievalAdapter(store)
    res = await adapter("keyword", 2)
    assert isinstance(res, list) and len(res) >= 1
    assert res[0]["id"] == "k1"


@pytest.mark.asyncio
async def test_graph_store_with_retrieval_adapter():
    store = GraphStore()
    await store.initialize()
    await store.add(Item(id="g1", content="node about alpha", metadata={}))
    await store.add(Item(id="g2", content="beta node", metadata={}))

    adapter = RetrievalAdapter(store)
    res = await adapter("alpha", 2)
    assert isinstance(res, list) and len(res) >= 1
    assert res[0]["id"] == "g1"

