"""Integration tests for query schema compliance."""

import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock

from memfuse_core.gateway.api_gateway import MemoryApiGateway
from memfuse_core.services.buffer_service import BufferService
from memfuse_core.services.database_service import DatabaseService
from memfuse_core.interfaces.gateway_interface import OperationType


class MockBufferService:
    """Mock buffer service for testing."""
    
    def __init__(self, mock_results=None):
        self.mock_results = mock_results or []
    
    async def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Mock query method."""
        return {
            "status": "success",
            "data": {
                "results": self.mock_results[:top_k],
                "total": len(self.mock_results)
            }
        }


class MockDatabaseService:
    """Mock database service for testing."""
    
    def __init__(self):
        self.users = {
            "user-1": {"id": "user-1", "name": "Test User"}
        }
        self.agents = {
            "agent-1": {"id": "agent-1", "name": "Test Agent"}
        }
        self.sessions = {
            "session-1": {"id": "session-1", "name": "Test Session"},
            "session-2": {"id": "session-2", "name": "Other Session"}
        }
    
    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """Mock get user method."""
        return self.users.get(user_id)
    
    async def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """Mock get agent method."""
        return self.agents.get(agent_id)
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Mock get session method."""
        return self.sessions.get(session_id)


@pytest.mark.asyncio
class TestQuerySchemaIntegration:
    """Integration tests for query schema compliance."""
    
    def create_mock_episodic_results(self):
        """Create mock episodic memory results."""
        return [
            {
                "id": "episodic-1",
                "content": "[USER]: What did you know about environmental conservation?",
                "score": 0.85,
                "type": "message",
                "created_at": "2025-09-02T13:52:46.552383+00:00",
                "metadata": {
                    "user_id": "user-1",
                    "session_id": "session-1",
                    "level": "info",
                    "retrieval": "vector",
                    "source": "memory_database"
                }
            },
            {
                "id": "episodic-2", 
                "content": "[ASSISTANT]: Environmental conservation involves protecting natural resources...",
                "score": 0.78,
                "type": "chunk",
                "created_at": "2025-09-02T13:53:46.552383+00:00",
                "metadata": {
                    "user_id": "user-1",
                    "session_id": "session-2",
                    "level": "info",
                    "retrieval": "vector",
                    "source": "memory_database"
                }
            }
        ]
    
    def create_mock_semantic_results(self):
        """Create mock semantic memory results."""
        return [
            {
                "id": "semantic-1",
                "content": "Environmental conservation is the practice of protecting natural resources.",
                "score": 0.92,
                "type": "semantic",
                "created_at": "2025-09-02T13:52:46.552383+00:00",
                "metadata": {
                    "user_id": "user-1",
                    "session_id": "session-1",
                    "derived_from": ["episodic-1", "episodic-2"],
                    "level": "info",
                    "retrieval": "semantic",
                    "source": "memory_database"
                }
            }
        ]
    
    async def test_gateway_episodic_memory_without_session(self):
        """Test gateway processing of episodic memories without session context."""
        # Arrange
        mock_results = self.create_mock_episodic_results()
        buffer_service = MockBufferService(mock_results)
        db_service = MockDatabaseService()
        gateway = MemoryApiGateway(buffer_service, db_service)
        
        request_data = {
            "query": "environmental conservation",
            "user_id": "user-1",
            "agent_id": "agent-1",
            "top_k": 5
        }
        
        # Act
        response = await gateway.process_request(request_data, OperationType.QUERY)
        
        # Assert
        assert response["status"] == "success"
        assert response["code"] == 200
        assert response["message"] is not None
        assert response["errors"] is None
        
        data = response["data"]
        assert "query" in data
        assert "results" in data
        assert "total" in data
        
        results = data["results"]
        assert len(results) == 2
        
        # Check first result (episodic)
        result1 = results[0]
        assert result1["memory_type"] == "episodic"
        assert "content" in result1
        assert "relevance_score" in result1
        assert "fact" not in result1
        
        metadata1 = result1["metadata"]
        assert metadata1["user_id"] == "user-1"
        assert metadata1["agent_id"] == "agent-1"
        assert metadata1["session_id"] == "session-1"
        assert metadata1["session_name"] == "Test Session"
        assert metadata1["scope"] is None  # No session in request
        
        # Forbidden fields should be removed
        assert "score" not in result1
        assert "type" not in result1
        assert "level" not in metadata1
        assert "retrieval" not in metadata1
        assert "source" not in metadata1
    
    async def test_gateway_episodic_memory_with_session_context(self):
        """Test gateway processing with session context for scope calculation."""
        # Arrange
        mock_results = self.create_mock_episodic_results()
        buffer_service = MockBufferService(mock_results)
        db_service = MockDatabaseService()
        gateway = MemoryApiGateway(buffer_service, db_service)
        
        request_data = {
            "query": "environmental conservation",
            "user_id": "user-1",
            "agent_id": "agent-1",
            "session_id": "session-1",  # Same as first result
            "top_k": 5,
            "metadata": {
                "task": "op_websearch_memory",
                "mode": None
            }
        }
        
        # Act
        response = await gateway.process_request(request_data, OperationType.QUERY)
        
        # Assert
        results = response["data"]["results"]
        
        # First result should be in_session
        result1 = results[0]
        assert result1["metadata"]["scope"] == "in_session"
        assert result1["metadata"]["task"] == "op_websearch_memory"
        assert result1["metadata"]["mode"] is None
        
        # Second result should be cross_session
        result2 = results[1]
        assert result2["metadata"]["scope"] == "cross_session"
        assert result2["metadata"]["session_id"] == "session-2"
    
    async def test_gateway_semantic_memory_processing(self):
        """Test gateway processing of semantic memories."""
        # Arrange
        mock_results = self.create_mock_semantic_results()
        buffer_service = MockBufferService(mock_results)
        db_service = MockDatabaseService()
        gateway = MemoryApiGateway(buffer_service, db_service)
        
        request_data = {
            "query": "environmental conservation",
            "user_id": "user-1",
            "agent_id": "agent-1",
            "top_k": 5
        }
        
        # Act
        response = await gateway.process_request(request_data, OperationType.QUERY)
        
        # Assert
        results = response["data"]["results"]
        assert len(results) == 1
        
        result = results[0]
        assert result["memory_type"] == "semantic"
        assert "content" not in result  # Should be removed for semantic
        assert "fact" in result
        
        fact = result["fact"]
        assert fact["text"] == "Environmental conservation is the practice of protecting natural resources."
        assert fact["triples"] is None
        
        metadata = result["metadata"]
        assert "derived_from" in metadata
        assert metadata["derived_from"] == ["episodic-1", "episodic-2"]
        assert metadata["scope"] is None  # No session in request
    
    async def test_gateway_mixed_memory_types(self):
        """Test gateway processing of mixed memory types."""
        # Arrange
        episodic_results = self.create_mock_episodic_results()
        semantic_results = self.create_mock_semantic_results()
        mixed_results = episodic_results + semantic_results
        
        buffer_service = MockBufferService(mixed_results)
        db_service = MockDatabaseService()
        gateway = MemoryApiGateway(buffer_service, db_service)
        
        request_data = {
            "query": "environmental conservation",
            "user_id": "user-1",
            "agent_id": "agent-1",
            "session_id": "session-1",
            "top_k": 10
        }
        
        # Act
        response = await gateway.process_request(request_data, OperationType.QUERY)
        
        # Assert
        results = response["data"]["results"]
        assert len(results) == 3
        
        # Check that we have both types
        memory_types = [r["memory_type"] for r in results]
        assert "episodic" in memory_types
        assert "semantic" in memory_types
        
        # Check episodic results
        episodic_results = [r for r in results if r["memory_type"] == "episodic"]
        for result in episodic_results:
            assert "content" in result
            assert "fact" not in result
        
        # Check semantic results
        semantic_results = [r for r in results if r["memory_type"] == "semantic"]
        for result in semantic_results:
            assert "content" not in result
            assert "fact" in result
    
    async def test_gateway_task_eos_handling(self):
        """Test gateway handling of task_eos metadata."""
        # Arrange
        mock_results = self.create_mock_episodic_results()
        buffer_service = MockBufferService(mock_results)
        db_service = MockDatabaseService()
        gateway = MemoryApiGateway(buffer_service, db_service)
        
        request_data = {
            "query": "Job done about web search.",
            "user_id": "user-1",
            "agent_id": "agent-1",
            "session_id": "session-1",
            "top_k": 10,
            "metadata": {
                "task": "op_websearch_memory",
                "mode": None,
                "task_eos": True
            }
        }
        
        # Act
        response = await gateway.process_request(request_data, OperationType.QUERY)
        
        # Assert
        assert response["status"] == "success"
        
        # task_eos should be handled but not propagated to results
        results = response["data"]["results"]
        for result in results:
            metadata = result["metadata"]
            assert metadata.get("task") == "op_websearch_memory"
            assert "task_eos" not in metadata  # Should not be in result metadata
    
    async def test_gateway_error_handling(self):
        """Test gateway error handling."""
        # Arrange
        buffer_service = AsyncMock()
        buffer_service.query.side_effect = Exception("Service error")
        
        db_service = MockDatabaseService()
        gateway = MemoryApiGateway(buffer_service, db_service)
        
        request_data = {
            "query": "test query",
            "user_id": "user-1",
            "agent_id": "agent-1",
            "top_k": 5
        }
        
        # Act
        response = await gateway.process_request(request_data, OperationType.QUERY)
        
        # Assert
        assert response["status"] == "error"
        assert response["code"] == 500
        assert "error" in response["message"].lower()
        assert response["errors"] is not None
        
        # Data should still have proper structure
        data = response["data"]
        assert "results" in data
        assert "total" in data
        assert data["results"] == []
        assert data["total"] == 0
    
    async def test_gateway_empty_results(self):
        """Test gateway handling of empty results."""
        # Arrange
        buffer_service = MockBufferService([])  # Empty results
        db_service = MockDatabaseService()
        gateway = MemoryApiGateway(buffer_service, db_service)
        
        request_data = {
            "query": "nonexistent query",
            "user_id": "user-1",
            "agent_id": "agent-1",
            "top_k": 5
        }
        
        # Act
        response = await gateway.process_request(request_data, OperationType.QUERY)
        
        # Assert
        assert response["status"] == "success"
        assert response["code"] == 200
        assert response["message"] is not None
        assert response["errors"] is None
        
        data = response["data"]
        assert data["results"] == []
        assert data["total"] == 0
    
    async def test_gateway_context_enrichment(self):
        """Test that gateway properly enriches context from database."""
        # Arrange
        mock_results = self.create_mock_episodic_results()
        buffer_service = MockBufferService(mock_results)
        db_service = MockDatabaseService()
        gateway = MemoryApiGateway(buffer_service, db_service)
        
        request_data = {
            "query": "test query",
            "user_id": "user-1",
            "agent_id": "agent-1",
            "session_id": "session-1",
            "top_k": 5
        }
        
        # Act
        response = await gateway.process_request(request_data, OperationType.QUERY)
        
        # Assert - context should be enriched with names from database
        results = response["data"]["results"]
        for result in results:
            metadata = result["metadata"]
            # These should be enriched from the database
            if result["metadata"]["session_id"] == "session-1":
                assert metadata["session_name"] == "Test Session"
            elif result["metadata"]["session_id"] == "session-2":
                assert metadata["session_name"] == "Other Session"


@pytest.mark.asyncio
class TestQuerySchemaValidation:
    """Test query schema validation and edge cases."""
    
    async def test_malformed_result_handling(self):
        """Test handling of malformed results."""
        # Arrange
        malformed_results = [
            {"id": "valid-1", "content": "valid", "score": 0.8, "type": "chunk"},
            {"invalid": "no required fields"},  # Missing required fields
            None,  # Null result
            "not a dict",  # Wrong type
            {"id": "valid-2", "content": "also valid", "score": 0.7, "type": "message"}
        ]
        
        buffer_service = MockBufferService(malformed_results)
        db_service = MockDatabaseService()
        gateway = MemoryApiGateway(buffer_service, db_service)
        
        request_data = {
            "query": "test query",
            "user_id": "user-1",
            "agent_id": "agent-1",
            "top_k": 10
        }
        
        # Act
        response = await gateway.process_request(request_data, OperationType.QUERY)
        
        # Assert - should handle malformed results gracefully
        assert response["status"] == "success"
        results = response["data"]["results"]
        
        # Should only include valid results
        assert len(results) <= 2  # Only valid results should remain
        for result in results:
            assert isinstance(result, dict)
            assert "id" in result
            assert "memory_type" in result
            assert "relevance_score" in result
    
    async def test_missing_metadata_handling(self):
        """Test handling of results with missing metadata.""" 
        # Arrange
        results_with_missing_metadata = [
            {
                "id": "no-metadata",
                "content": "content without metadata", 
                "score": 0.8,
                "type": "chunk"
                # No metadata field
            }
        ]
        
        buffer_service = MockBufferService(results_with_missing_metadata)
        db_service = MockDatabaseService()
        gateway = MemoryApiGateway(buffer_service, db_service)
        
        request_data = {
            "query": "test query",
            "user_id": "user-1",
            "agent_id": "agent-1",
            "session_id": "session-1",
            "top_k": 5
        }
        
        # Act
        response = await gateway.process_request(request_data, OperationType.QUERY)
        
        # Assert
        results = response["data"]["results"]
        assert len(results) == 1
        
        result = results[0]
        assert "metadata" in result
        
        # Metadata should be enriched from context
        metadata = result["metadata"]
        assert metadata["user_id"] == "user-1"
        assert metadata["agent_id"] == "agent-1"
        assert metadata["session_id"] == "session-1"
        assert metadata["scope"] == "in_session"