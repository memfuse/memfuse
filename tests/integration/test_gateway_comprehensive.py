import pytest
pytestmark = pytest.mark.skip(reason="Deprecated async integration tests; use tests/integration/test_gateway_sync_only.py")

"""Comprehensive Gateway integration tests with Buffer and Database scenarios."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List

from memfuse_core.gateway.api_gateway import MemoryApiGateway, create_memory_gateway
from memfuse_core.interfaces.gateway_interface import OperationType


@pytest.mark.asyncio
async def test_gateway_creation():
    """Test gateway creation with proper dependencies."""
    # Mock dependencies
    buffer_service = AsyncMock()
    db_service = AsyncMock()

    # Create gateway
    gateway = create_memory_gateway(
        buffer_service=buffer_service,
        db_service=db_service
    )

    assert isinstance(gateway, MemoryApiGateway)
    assert gateway.buffer_service == buffer_service
    assert gateway.db_service == db_service


@pytest.mark.asyncio
async def test_gateway_m1_episodic_transformation():
    """Test M1 episodic memory transformation (messages from buffer)."""
    # Mock dependencies
    buffer_service = AsyncMock()
    db_service = AsyncMock()

    # Mock buffer service response with M1 episodic data
    buffer_service.query.return_value = {
        "status": "success",
        "code": 200,
        "data": {
            "results": [
                {
                    "id": "round_0_0",
                    "content": "Hello, I am testing the buffer system.",
                    "score": 1.0,
                    "type": "message",
                    "role": "user",
                    "created_at": "2025-09-08T20:00:00Z",
                    "metadata": {
                        "source": "round_buffer",
                        "similarity_score": 1.0
                    }
                }
            ],
            "total": 1
        },
        "message": "Query completed"
    }

    # Create gateway
    gateway = create_memory_gateway(
        buffer_service=buffer_service,
        db_service=db_service
    )

    # Test query
    request_data = {
        "query": "buffer system testing",
        "top_k": 5,
        "user_id": "a4b063c1-9e0e-46c4-bcfc-4aeb4b1317cb",
        "session_id": "192a2f52-0682-4ff3-856b-a454bfd47856",
        "agent_id": "a3f10716-c9e2-4a29-88f4-bebb1d3f031a",
        "metadata": {"task": "search", "mode": "episodic"}
    }

    response = await gateway.process_request(
        request_data=request_data,
        operation_type=OperationType.QUERY
    )

    # Verify response structure
    assert response["status"] == "success"
    assert response["code"] == 200
    assert "data" in response
    assert "results" in response["data"]

    # Verify M1 episodic transformation
    results = response["data"]["results"]
    assert len(results) == 1

    result = results[0]
    # Check field renaming
    assert "relevance_score" in result
    assert "memory_type" in result
    assert result["memory_type"] == "message"
    assert "score" not in result
    assert "type" not in result

    # Check M1 episodic format (keeps content, no fact)
    assert "content" in result
    assert "fact" not in result

    # Check metadata enrichment
    assert "metadata" in result
    metadata = result["metadata"]
    assert metadata.get("user_id") == "a4b063c1-9e0e-46c4-bcfc-4aeb4b1317cb"
    assert metadata.get("session_id") == "192a2f52-0682-4ff3-856b-a454bfd47856"
    assert metadata.get("agent_id") == "a3f10716-c9e2-4a29-88f4-bebb1d3f031a"
    assert metadata.get("scope") == "in_session"

    # Check field removal
    assert "source" not in metadata
    assert "similarity_score" not in metadata
