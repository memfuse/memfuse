import pytest

from src.memfuse_core.persistence.in_memory_store import InMemoryStore
from src.memfuse_core.persistence.adapters import make_retrieval_handler_from_store
from src.memfuse_core.models.core import Item, Query, QueryResult
from src.memfuse_core.persistence.store_port import StorePort


@pytest.mark.asyncio
async def test_adapter_query_topk_happy_path():
    store = InMemoryStore()
    await store.add(Item(id="a", content="hello world", metadata={}))
    await store.add(Item(id="b", content="hello there", metadata={}))

    handler = make_retrieval_handler_from_store(store)
    results = await handler("hello", 1)

    assert isinstance(results, list)
    assert len(results) == 1
    r = results[0]
    assert set(["id", "content", "score", "metadata"]).issubset(r.keys())


@pytest.mark.asyncio
async def test_adapter_handles_empty_store_and_zero_results():
    store = InMemoryStore()
    handler = make_retrieval_handler_from_store(store)
    results = await handler("nonexistent", 3)
    # When store empty, still returns list (possibly empty)
    assert isinstance(results, list)


class DummyStoreWithoutTopK(StorePort):
    def __init__(self):
        self._items = {}

    async def initialize(self) -> bool:
        self._items.clear()
        return True

    async def add(self, item: Item) -> str:
        self._items[item.id] = item
        return item.id

    async def query(self, query: Query) -> QueryResult:
        # Return first item that contains text
        for it in self._items.values():
            if query.text.lower() in it.content.lower():
                return QueryResult(id=it.id, content=it.content, score=1.0, metadata={})
        return QueryResult(id="", content="", score=0.0, metadata={})

    async def delete(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None


@pytest.mark.asyncio
async def test_adapter_fallback_when_query_topk_missing():
    store = DummyStoreWithoutTopK()
    await store.add(Item(id="x", content="alpha", metadata={}))

    handler = make_retrieval_handler_from_store(store)

    # Fallback uses single-result query; should return at most one item
    res = await handler("alpha", 5)
    assert isinstance(res, list)
    assert len(res) <= 1

