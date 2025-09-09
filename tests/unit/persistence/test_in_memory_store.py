import pytest
from memfuse_core.persistence.in_memory_store import InMemoryStore
from memfuse_core.models.core import Item, Query


@pytest.mark.asyncio
async def test_in_memory_store_add_query_delete():
    store = InMemoryStore()
    ok = await store.initialize()
    assert ok

    # Add items
    i1 = Item(id="a", content="hello world", metadata={"session_id": "s1"})
    i2 = Item(id="b", content="hello there", metadata={"session_id": "s1"})
    i3 = Item(id="c", content="goodbye moon", metadata={"session_id": "s2"})

    assert await store.add(i1) == "a"
    assert await store.add(i2) == "b"
    assert await store.add(i3) == "c"

    # Single best result
    qr = await store.query(Query(text="hello"))
    assert qr.id in {"a", "b"}
    assert qr.score > 0
    assert qr.metadata.get("source") == "in_memory_store"

    # Top-k
    top2 = await store.query_topk("hello", 2)
    assert len(top2) == 2
    assert top2[0].score >= top2[1].score

    # Delete
    assert await store.delete("b") is True
    assert await store.delete("b") is False

