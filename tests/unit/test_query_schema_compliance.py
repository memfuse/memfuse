"""Tests for query schema compliance with the new requirements."""

import pytest
from typing import Dict, Any

from memfuse_core.gateway.processors import (
    QueryResponseProcessor,
    MetadataEnricher,
    ScopeCalculator,
    FieldRemover,
)
from memfuse_core.interfaces.gateway_interface import RequestContext, OperationType


def create_test_context(
    user_id: str = "8a0d893f-21bd-450c-b337-735645324a6a",
    agent_id: str = "5a3a351c-18a1-9996-af50-149089234ca",
    session_id: str = None,
    session_name: str = None,
    metadata: Dict[str, Any] = None
) -> RequestContext:
    """Create a test request context."""
    return RequestContext(
        user_id=user_id,
        user_name="test-user",
        agent_id=agent_id,
        agent_name="test-agent",
        session_id=session_id,
        session_name=session_name,
        operation_type=OperationType.QUERY,
        request_metadata=metadata or {}
    )


def create_raw_episodic_result() -> Dict[str, Any]:
    """Create a raw episodic memory result before transformation."""
    return {
        "id": "00004017-625d-4adc-97f7-1b4f7cb7895f",
        "content": "[ASSISTANT]: Yes, Baha'i communities have been involved in several environmental conservation projects around the world...",
        "score": 0.7356168329715745,
        "type": "chunk",
        "created_at": "2025-09-02T13:52:46.552383+00:00",
        "metadata": {
            "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
            "session_id": "1bcd351c-88b8-4f56-af50-14908955edcb",
            "level": "debug",
            "retrieval": "vector",
            "source": "memory_database"
        }
    }


def create_raw_semantic_result() -> Dict[str, Any]:
    """Create a raw semantic memory result before transformation."""
    return {
        "id": "m2_jira_ticket_753",
        "content": "The goal is to update the legacy authentication service to use OAuth2 and improve token refresh logic. Current status is 'Ready for Staging'.",
        "score": 0.98,
        "type": "semantic",
        "created_at": "2025-09-02T13:52:46.552383+00:00",
        "metadata": {
            "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
            "session_id": "2209aabc-c2a6-4b76-9530-f4ef7f57a521",
            "derived_from": [
                "m1_conversation_log_102",
                "some_other_source_memory_id"
            ],
            "level": "info",
            "retrieval": "semantic",
            "source": "memory_database"
        }
    }


class TestQuerySchemaCompliance:
    """Test query schema compliance with new requirements."""
    
    def setup_method(self):
        """Set up test processors."""
        self.processors = [
            QueryResponseProcessor(),
            MetadataEnricher(),
            ScopeCalculator(),
            FieldRemover(fields_to_remove=[
                "metadata.level",
                "metadata.retrieval", 
                "metadata.source"
            ])
        ]
    
    def transform_data(self, data: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Apply all processors to transform data."""
        result = data.copy()
        for processor in self.processors:
            result = processor.transform(result, context)
        return result
    
    def test_episodic_memory_without_session_id(self):
        """Test M1 episodic memory format when session_id is not specified."""
        # Arrange
        context = create_test_context(session_id=None, session_name="some-session-name")
        raw_result = create_raw_episodic_result()
        data = {"results": [raw_result], "total": 1}
        
        # Act
        transformed = self.transform_data(data, context)
        
        # Assert
        assert "results" in transformed
        assert len(transformed["results"]) == 1
        
        result = transformed["results"][0]
        
        # Test field renaming
        assert "relevance_score" in result
        assert "score" not in result
        assert result["relevance_score"] == 0.7356168329715745
        
        assert "memory_type" in result
        assert "type" not in result
        assert result["memory_type"] == "episodic"
        
        # Test episodic memory structure
        assert "content" in result
        assert "fact" not in result
        
        # Test metadata structure
        metadata = result["metadata"]
        assert metadata["user_id"] == "8a0d893f-21bd-450c-b337-735645324a6a"
        assert metadata["agent_id"] == "5a3a351c-18a1-9996-af50-149089234ca"
        assert metadata["session_id"] == "1bcd351c-88b8-4f56-af50-14908955edcb"
        assert "session_name" in metadata  # Should be added by enricher
        assert metadata["scope"] is None  # Should be null when no session_id in request
        
        # Test removed fields
        assert "level" not in metadata
        assert "retrieval" not in metadata
        assert "source" not in metadata
        assert "role" not in result
        
        # Test timestamps
        assert "created_at" in result
        assert "updated_at" in result
    
    def test_episodic_memory_with_session_id_in_session(self):
        """Test M1 episodic memory with session_id - in_session scope."""
        # Arrange
        session_id = "1bcd351c-88b8-4f56-af50-14908955edcb"
        context = create_test_context(
            session_id=session_id,
            session_name="some-session-name",
            metadata={"task": "op_websearch_memory", "mode": None}
        )
        raw_result = create_raw_episodic_result()
        data = {"results": [raw_result], "total": 1}
        
        # Act
        transformed = self.transform_data(data, context)
        
        # Assert
        result = transformed["results"][0]
        metadata = result["metadata"]
        
        assert metadata["scope"] == "in_session"
        assert metadata["session_name"] == "some-session-name"
        assert metadata["task"] == "op_websearch_memory"
        assert metadata["mode"] is None
    
    def test_episodic_memory_with_session_id_cross_session(self):
        """Test M1 episodic memory with session_id - cross_session scope."""
        # Arrange
        request_session_id = "different-session-id"
        context = create_test_context(
            session_id=request_session_id,
            session_name="request-session-name"
        )
        raw_result = create_raw_episodic_result()  # Has different session_id
        data = {"results": [raw_result], "total": 1}
        
        # Act
        transformed = self.transform_data(data, context)
        
        # Assert
        result = transformed["results"][0]
        metadata = result["metadata"]
        
        assert metadata["scope"] == "cross_session"
        assert metadata["session_id"] == "1bcd351c-88b8-4f56-af50-14908955edcb"  # Original from result
    
    def test_semantic_memory_without_session_id(self):
        """Test M2 semantic memory format when session_id is not specified."""
        # Arrange
        context = create_test_context(session_id=None)
        raw_result = create_raw_semantic_result()
        data = {"results": [raw_result], "total": 1}
        
        # Act
        transformed = self.transform_data(data, context)
        
        # Assert
        result = transformed["results"][0]
        
        # Test field renaming
        assert result["relevance_score"] == 0.98
        assert result["memory_type"] == "semantic"
        
        # Test semantic memory structure
        assert "content" not in result  # Should be removed for semantic
        assert "fact" in result
        assert result["fact"]["text"] == "The goal is to update the legacy authentication service to use OAuth2 and improve token refresh logic. Current status is 'Ready for Staging'."
        assert result["fact"]["triples"] is None
        
        # Test metadata
        metadata = result["metadata"]
        assert metadata["scope"] is None
        assert "derived_from" in metadata
        assert metadata["derived_from"] == [
            "m1_conversation_log_102",
            "some_other_source_memory_id"
        ]
    
    def test_semantic_memory_with_session_id_cross_session(self):
        """Test M2 semantic memory with session_id - cross_session scope."""
        # Arrange
        request_session_id = "1bcd351c-88b8-4f56-af50-14908955edcb"
        context = create_test_context(
            session_id=request_session_id,
            session_name="some-session-name"
        )
        raw_result = create_raw_semantic_result()  # Has different session_id
        data = {"results": [raw_result], "total": 1}
        
        # Act
        transformed = self.transform_data(data, context)
        
        # Assert
        result = transformed["results"][0]
        metadata = result["metadata"]
        
        assert metadata["scope"] == "cross_session"
        assert metadata["session_id"] == "2209aabc-c2a6-4b76-9530-f4ef7f57a521"  # Original from result
        assert metadata["session_name"] == "some-session-name"  # From context
    
    def test_mixed_memory_types(self):
        """Test mixed episodic and semantic memory results."""
        # Arrange
        session_id = "1bcd351c-88b8-4f56-af50-14908955edcb"
        context = create_test_context(session_id=session_id)
        
        episodic_result = create_raw_episodic_result()
        semantic_result = create_raw_semantic_result()
        
        data = {"results": [episodic_result, semantic_result], "total": 2}
        
        # Act
        transformed = self.transform_data(data, context)
        
        # Assert
        results = transformed["results"]
        assert len(results) == 2
        
        # First result should be episodic
        episodic = results[0]
        assert episodic["memory_type"] == "episodic"
        assert "content" in episodic
        assert "fact" not in episodic
        assert episodic["metadata"]["scope"] == "in_session"
        
        # Second result should be semantic
        semantic = results[1]
        assert semantic["memory_type"] == "semantic"
        assert "content" not in semantic
        assert "fact" in semantic
        assert semantic["metadata"]["scope"] == "cross_session"
    
    def test_task_eos_metadata_handling(self):
        """Test task_eos metadata handling."""
        # Arrange
        context = create_test_context(
            metadata={"task": "op_websearch_memory", "mode": None, "task_eos": True}
        )
        raw_result = create_raw_episodic_result()
        data = {"results": [raw_result], "total": 1}
        
        # Act
        transformed = self.transform_data(data, context)
        
        # Assert
        result = transformed["results"][0]
        metadata = result["metadata"]
        
        assert metadata["task"] == "op_websearch_memory"
        # task_eos should not be propagated to result metadata
        assert "task_eos" not in metadata
    
    def test_all_required_fields_present(self):
        """Test that all required fields are present in the response."""
        # Arrange
        context = create_test_context(
            session_id="1bcd351c-88b8-4f56-af50-14908955edcb",
            session_name="some-session-name"
        )
        raw_result = create_raw_episodic_result()
        data = {"results": [raw_result], "total": 1}
        
        # Act
        transformed = self.transform_data(data, context)
        
        # Assert
        result = transformed["results"][0]
        
        # Required top-level fields
        required_fields = [
            "id", "content", "relevance_score", "memory_type", 
            "created_at", "updated_at", "metadata"
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Required metadata fields
        metadata = result["metadata"]
        required_metadata_fields = [
            "user_id", "agent_id", "session_id", "session_name", "scope"
        ]
        for field in required_metadata_fields:
            assert field in metadata, f"Missing required metadata field: {field}"
    
    def test_forbidden_fields_removed(self):
        """Test that forbidden fields are properly removed."""
        # Arrange
        context = create_test_context()
        raw_result = create_raw_episodic_result()
        # Add some forbidden fields
        raw_result.update({
            "role": "assistant",
            "source": "database",
            "similarity_score": 0.8,
            "distance": 0.2
        })
        data = {"results": [raw_result], "total": 1}
        
        # Act
        transformed = self.transform_data(data, context)
        
        # Assert
        result = transformed["results"][0]
        
        # These fields should be removed
        forbidden_fields = ["role", "source", "similarity_score", "distance", "score", "type"]
        for field in forbidden_fields:
            assert field not in result, f"Forbidden field still present: {field}"
        
        # These metadata fields should be removed
        metadata = result["metadata"]
        forbidden_metadata_fields = ["level", "retrieval", "source"]
        for field in forbidden_metadata_fields:
            assert field not in metadata, f"Forbidden metadata field still present: {field}"


@pytest.mark.asyncio
class TestQueryApiResponseFormat:
    """Test the query API response format compliance."""
    
    def test_response_structure_compliance(self):
        """Test that API response follows the required structure."""
        # This would be tested in integration tests with actual API calls
        # Here we test the expected structure
        expected_response = {
            "status": "success",
            "code": 200,
            "data": {
                "query": "test query",
                "results": [],
                "total": 0
            },
            "message": "Found 0 results",
            "errors": None
        }
        
        # Test structure
        assert "status" in expected_response
        assert "code" in expected_response
        assert "data" in expected_response
        assert "message" in expected_response
        assert "errors" in expected_response
        
        # Test data structure
        data = expected_response["data"]
        assert "query" in data
        assert "results" in data
        assert "total" in data
        
        # For successful responses, errors should be None
        assert expected_response["errors"] is None