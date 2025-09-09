import pytest

from src.memfuse_core.persistence.adapters import RetrievalAdapter
from src.memfuse_core.persistence.store_port import StorePort
from src.memfuse_core.models.core import Item, Query, QueryResult


class BoomStore(StorePort):
    async def initialize(self) -> bool:
        return True
    async def add(self, item: Item) -> str:
        return item.id
    async def query(self, query: Query) -> QueryResult:
        raise RuntimeError("boom")
    async def delete(self, item_id: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_adapter_handles_store_exception_returns_empty():
    adapter = RetrievalAdapter(BoomStore())
    out = await adapter("anything", 5)
    assert out == []

