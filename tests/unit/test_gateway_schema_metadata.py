import pytest

from memfuse_core.gateway.api_gateway import MemoryApiGateway
from memfuse_core.gateway.processors import (
    QueryResponseProcessor,
    MetadataEnricher,
    ScopeCalculator,
    FieldRemover,
)
from memfuse_core.interfaces.gateway_interface import RequestContext, OperationType, ServiceType


def build_context(user_id="u1", agent_id="a1", session_id=None, session_name=None, metadata=None):
    return RequestContext(
        user_id=user_id,
        user_name="user",
        agent_id=agent_id,
        agent_name="agent",
        session_id=session_id,
        session_name=session_name,
        operation_type=OperationType.QUERY,
        request_metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_processors_schema_and_scope_in_session():
    # Arrange
    processors = (
        QueryResponseProcessor(),
        MetadataEnricher(),
        ScopeCalculator(),
        FieldRemover(fields_to_remove=["metadata.level", "metadata.retrieval", "metadata.source"]),
    )
    request_ctx = build_context(session_id="sess-1", session_name="some-session-name", metadata={"task": None, "mode": None})
    # result from storage (pre-transform): includes score/type and extra fields
    raw = {
        "id": "mid-1",
        "content": "hello",
        "score": 0.73,
        "type": "chunk",
        "created_at": "2025-09-02T13:52:46.552383+00:00",
        "metadata": {
            "user_id": "u1",
            "session_id": "sess-1",
            "level": "debug",
            "retrieval": "vector",
            "source": "memory_database",
        },
    }
    data = {"results": [raw], "total": 1}

    # Act
    cur = data
    for p in processors:
        if isinstance(p, QueryResponseProcessor):
            cur = p.transform(cur, request_ctx)
        elif isinstance(p, MetadataEnricher):
            cur = p.transform(cur, request_ctx)
        elif isinstance(p, ScopeCalculator):
            cur = p.transform(cur, request_ctx)
        else:
            cur = p.transform(cur, request_ctx)

    # Assert schema
    assert "results" in cur and len(cur["results"]) == 1
    item = cur["results"][0]
    assert "relevance_score" in item and "score" not in item
    assert item["memory_type"] == "episodic"
    assert "updated_at" in item  # may be same as created_at or None
    md = item["metadata"]
    assert md.get("user_id") == "u1"
    assert md.get("session_id") == "sess-1"
    assert "agent_id" in md  # filled from context
    assert "session_name" in md  # filled from context
    # scope in_session when session matches
    assert md.get("scope") == "in_session"
    # removed fields
    assert "level" not in md and "retrieval" not in md and "source" not in md


@pytest.mark.asyncio
async def test_processors_schema_and_scope_cross_session():
    processors = (
        QueryResponseProcessor(),
        MetadataEnricher(),
        ScopeCalculator(),
        FieldRemover(fields_to_remove=["metadata.level", "metadata.retrieval", "metadata.source"]),
    )
    request_ctx = build_context(session_id="sess-REQ", session_name="req-session")
    raw = {
        "id": "mid-2",
        "content": "world",
        "score": 0.66,
        "type": "message",
        "created_at": "2025-09-02T13:52:46.552383+00:00",
        "metadata": {
            "user_id": "u1",
            "session_id": "sess-OTHER",
            "source": "memory_database",
        },
    }
    data = {"results": [raw], "total": 1}
    cur = data
    for p in processors:
        cur = p.transform(cur, request_ctx)
    item = cur["results"][0]
    assert item["memory_type"] == "episodic"
    assert item["metadata"]["scope"] == "cross_session"


@pytest.mark.asyncio
async def test_gateway_transform_echo_query_and_metadata():
    gw = MemoryApiGateway(buffer_service=None, db_service=None)
    # service_response shape before transform
    service_response = {
        "status": "success",
        "code": 200,
        "data": {
            "results": [
                {"id": "x", "content": "c", "score": 0.5, "type": "chunk", "metadata": {"session_id": None}}
            ],
            "total": 1,
        },
        "message": "ok",
        "errors": None,
    }
    ctx = build_context(session_id=None)
    request = {"query": "What?", "metadata": {"task": "t1", "mode": None}}
    out = await gw._transform_response(service_response, ctx, type("R", (), {})(), request)
    assert out["status"] == "success"
    assert out["data"].get("query") == "What?"
    item = out["data"]["results"][0]
    assert item.get("relevance_score") is not None and item.get("memory_type") == "episodic"
    # request metadata merged
    assert item["metadata"].get("task") == "t1"

