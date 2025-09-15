import pytest


class DummyMemoryService:
    def __init__(self):
        self.last_query_kwargs = None

    async def query(self, **kwargs):
        self.last_query_kwargs = kwargs
        # return minimal success payload
        return {"status": "success", "code": 200, "data": {"results": [], "total": 0}}


@pytest.mark.asyncio
async def test_gateway_passes_task_to_buffer(monkeypatch):
    captured = {}

    class DummyBuffer:
        async def query(self, **kwargs):
            captured.update(kwargs)
            return {"status": "success", "code": 200, "data": {"results": [], "total": 0}}

    from memfuse_core.gateway.api_gateway import MemoryApiGateway

    gw = MemoryApiGateway(buffer_service=DummyBuffer(), db_service=None)
    request = {
        "query": "hello",
        "top_k": 5,
        "metadata": {"task": "op_websearch_memory", "mode": None},
    }
    out = await gw.process_request(request, operation_type=None)
    assert out["status"] == "success"
    assert captured.get("task") == "op_websearch_memory"


@pytest.mark.asyncio
async def test_buffer_bypass_passes_task_to_memory_service(monkeypatch):
    # Disable buffer mode by config
    mem = DummyMemoryService()

    from memfuse_core.services.buffer_service import BufferService

    buf = BufferService(memory_service=mem, user="u", config={"buffer": {"enabled": False}})
    await buf.initialize()

    await buf.query(
        query="hello",
        top_k=3,
        session_id="sess1",
        task="op_x",
        include_messages=True,
        include_knowledge=False,
        include_chunks=True,
    )
    assert mem.last_query_kwargs is not None
    assert mem.last_query_kwargs.get("task") == "op_x"

