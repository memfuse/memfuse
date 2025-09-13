"""End-to-end tests for complete schema compliance."""

import pytest
import json
from typing import Dict, Any

from memfuse_core.gateway.api_gateway import MemoryApiGateway
from memfuse_core.gateway.processors import (
    QueryResponseProcessor,
    MetadataEnricher,
    ScopeCalculator,
    FieldRemover
)


def validate_response_schema(response: Dict[str, Any], has_session_id: bool = False) -> None:
    """Validate that response matches the required schema exactly."""
    
    # Top-level response structure
    assert "status" in response
    assert "code" in response
    assert "data" in response
    assert "message" in response
    assert "errors" in response
    
    # For successful responses
    if response["status"] == "success":
        assert response["code"] == 200
        assert response["errors"] is None
        assert isinstance(response["message"], str)
    
    # Data structure
    data = response["data"]
    assert "query" in data
    assert "results" in data
    assert "total" in data
    
    assert isinstance(data["query"], str)
    assert isinstance(data["results"], list)
    assert isinstance(data["total"], int)
    assert data["total"] == len(data["results"])
    
    # Validate each result
    for result in data["results"]:
        validate_result_schema(result, has_session_id)


def validate_result_schema(result: Dict[str, Any], has_session_id: bool = False) -> None:
    """Validate that a single result matches the required schema."""
    
    # Required top-level fields
    required_fields = ["id", "relevance_score", "memory_type", "created_at", "updated_at", "metadata"]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    # Field types
    assert isinstance(result["id"], str)
    assert isinstance(result["relevance_score"], (int, float))
    assert isinstance(result["memory_type"], str)
    assert isinstance(result["created_at"], str)
    assert result["updated_at"] is None or isinstance(result["updated_at"], str)
    assert isinstance(result["metadata"], dict)
    
    # Memory type specific validation
    memory_type = result["memory_type"]
    if memory_type == "episodic":
        validate_episodic_result(result)
    elif memory_type == "semantic":
        validate_semantic_result(result)
    else:
        pytest.fail(f"Unknown memory type: {memory_type}")
    
    # Metadata validation
    validate_metadata_schema(result["metadata"], has_session_id)
    
    # Forbidden fields should not be present
    forbidden_fields = ["score", "type", "role", "source", "similarity_score", "distance"]
    for field in forbidden_fields:
        assert field not in result, f"Forbidden field present: {field}"


def validate_episodic_result(result: Dict[str, Any]) -> None:
    """Validate episodic memory result structure."""
    assert result["memory_type"] == "episodic"
    assert "content" in result
    assert "fact" not in result
    assert isinstance(result["content"], str)


def validate_semantic_result(result: Dict[str, Any]) -> None:
    """Validate semantic memory result structure."""
    assert result["memory_type"] == "semantic"
    assert "content" not in result
    assert "fact" in result
    
    fact = result["fact"]
    assert isinstance(fact, dict)
    assert "text" in fact
    assert "triples" in fact
    assert isinstance(fact["text"], str)
    # triples can be None or a list
    assert fact["triples"] is None or isinstance(fact["triples"], list)


def validate_metadata_schema(metadata: Dict[str, Any], has_session_id: bool = False) -> None:
    """Validate metadata structure."""
    
    # Required metadata fields
    required_fields = ["user_id", "agent_id", "session_id", "session_name", "scope"]
    for field in required_fields:
        assert field in metadata, f"Missing required metadata field: {field}"
    
    # Field types
    assert isinstance(metadata["user_id"], str)
    assert isinstance(metadata["agent_id"], str)
    assert isinstance(metadata["session_id"], str)
    assert isinstance(metadata["session_name"], str)
    
    # Scope validation
    if has_session_id:
        # When session_id is provided in request, scope should be "in_session" or "cross_session"
        assert metadata["scope"] in ["in_session", "cross_session"]
    else:
        # When no session_id in request, scope should be null
        assert metadata["scope"] is None
    
    # Optional fields
    if "task" in metadata:
        assert metadata["task"] is None or isinstance(metadata["task"], str)
    if "mode" in metadata:
        assert metadata["mode"] is None or isinstance(metadata["mode"], str)
    if "derived_from" in metadata:
        assert isinstance(metadata["derived_from"], list)
    
    # Forbidden metadata fields
    forbidden_fields = ["level", "retrieval", "source"]
    for field in forbidden_fields:
        assert field not in metadata, f"Forbidden metadata field present: {field}"


class TestEndToEndSchemaCompliance:
    """End-to-end schema compliance tests."""
    
    def create_test_data_without_session(self) -> Dict[str, Any]:
        """Create test data for request without session_id."""
        return {
            "query": "What's the status of the 'auth-refactor' ticket?",
            "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
            "agent_id": "5a3a351c-18a1-9996-af50-149089234ca",
            "top_k": 10
        }
    
    def create_test_data_with_session(self) -> Dict[str, Any]:
        """Create test data for request with session_id."""
        return {
            "query": "What did you know about environmental conservation?",
            "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
            "agent_id": "d6d60bdf-ba1e-46d8-9b98-38529c6e85aa",
            "session_id": "1bcd351c-88b8-4f56-af50-14908955edcb",
            "top_k": 10,
            "metadata": {
                "task": "op_websearch_memory",
                "mode": None
            }
        }
    
    def create_test_data_with_task_eos(self) -> Dict[str, Any]:
        """Create test data for request with task_eos."""
        return {
            "query": "Job done about web search.",
            "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
            "agent_id": "d6d60bdf-ba1e-46d8-9b98-38529c6e85aa",
            "session_id": "1bcd351c-88b8-4f56-af50-14908955edcb",
            "top_k": 10,
            "metadata": {
                "task": "op_websearch_memory",
                "mode": None,
                "task_eos": True
            }
        }
    
    def create_mock_service_results(self) -> Dict[str, Any]:
        """Create mock service results."""
        return {
            "status": "success",
            "data": {
                "results": [
                    {
                        "id": "00004017-625d-4adc-97f7-1b4f7cb7895f",
                        "content": "[ASSISTANT]: Yes, Baha'i communities have been involved in several environmental conservation projects around the world...",
                        "score": 0.7356168329715745,
                        "type": "chunk",
                        "created_at": "2025-09-02T13:52:46.552383+00:00",
                        "metadata": {
                            "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
                            "agent_id": "5a3a351c-18a1-9996-af50-149089234ca",
                            "session_id": "1bcd351c-88b8-4f56-af50-14908955edcb",
                            "session_name": "some-session-name",
                            "level": "info",
                            "retrieval": "vector",
                            "source": "memory_database"
                        }
                    },
                    {
                        "id": "1833a3fa-b6a9-4477-bdb2-58453ac1f098",
                        "content": "[ASSISTANT]: As an AI, I remain completely in agreement with you.",
                        "score": 0.6660368343569761,
                        "type": "message",
                        "created_at": "2025-09-02T13:52:46.552383+00:00",
                        "metadata": {
                            "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
                            "agent_id": "5a3a351c-18a1-9996-af50-149089234ca",
                            "session_id": "2209aabc-c2a6-4b76-9530-f4ef7f57a521",
                            "session_name": "other-session-name",
                            "level": "info",
                            "retrieval": "vector",
                            "source": "memory_database"
                        }
                    },
                    {
                        "id": "m2_jira_ticket_753",
                        "content": "The goal is to update the legacy authentication service to use OAuth2 and improve token refresh logic. Current status is 'Ready for Staging'.",
                        "score": 0.98,
                        "type": "semantic",
                        "created_at": "2025-09-02T13:52:46.552383+00:00",
                        "metadata": {
                            "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
                            "agent_id": "5a3a351c-18a1-9996-af50-149089234ca",
                            "session_id": "2209aabc-c2a6-4b76-9530-f4ef7f57a521",
                            "session_name": "other-session-name",
                            "derived_from": [
                                "m1_conversation_log_102",
                                "some_other_source_memory_id"
                            ],
                            "level": "info",
                            "retrieval": "semantic",
                            "source": "memory_database"
                        }
                    }
                ],
                "total": 3
            }
        }
    
    @pytest.mark.asyncio
    async def test_complete_schema_without_session_id(self):
        """Test complete schema compliance when session_id is not specified in request."""
        # This test validates the exact schema from the requirements
        
        # Create mock gateway that returns our test results
        class MockGateway(MemoryApiGateway):
            def __init__(self):
                super().__init__(None, None)
            
            async def _call_service(self, service_type, request_data, service_params):
                return self.create_mock_service_results()
            
            def create_mock_service_results(self):
                return {
                    "status": "success",
                    "data": {
                        "results": [
                            {
                                "id": "00004017-625d-4adc-97f7-1b4f7cb7895f",
                                "content": "[ASSISTANT]: Yes, Baha'i communities have been involved in several environmental conservation projects around the world...",
                                "score": 0.7356168329715745,
                                "type": "chunk",
                                "created_at": "2025-09-02T13:52:46.552383+00:00",
                                "metadata": {
                                    "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
                                    "session_id": "1bcd351c-88b8-4f56-af50-14908955edcb",
                                    "level": "info",
                                    "retrieval": "vector",
                                    "source": "memory_database"
                                }
                            }
                        ],
                        "total": 1
                    }
                }
        
        # Arrange
        gateway = MockGateway()
        request_data = self.create_test_data_without_session()
        
        # Act
        response = await gateway.process_request(request_data)
        
        # Assert
        validate_response_schema(response, has_session_id=False)
        
        # Specific validation for no session case
        data = response["data"]
        for result in data["results"]:
            assert result["metadata"]["scope"] is None
    
    @pytest.mark.asyncio
    async def test_complete_schema_with_session_id(self):
        """Test complete schema compliance when session_id is specified in request."""
        
        class MockGateway(MemoryApiGateway):
            def __init__(self):
                super().__init__(None, None)
            
            async def _call_service(self, service_type, request_data, service_params):
                return {
                    "status": "success", 
                    "data": {
                        "results": [
                            {
                                "id": "00004017-625d-4adc-97f7-1b4f7cb7895f",
                                "content": "[ASSISTANT]: Yes, Baha'i communities have been involved in several environmental conservation projects around the world...",
                                "score": 0.7356168329715745,
                                "type": "chunk",
                                "created_at": "2025-09-02T13:52:46.552383+00:00",
                                "metadata": {
                                    "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
                                    "session_id": "1bcd351c-88b8-4f56-af50-14908955edcb",  # Same as request
                                    "level": "info",
                                    "retrieval": "vector",
                                    "source": "memory_database"
                                }
                            },
                            {
                                "id": "1833a3fa-b6a9-4477-bdb2-58453ac1f098",
                                "content": "[ASSISTANT]: As an AI, I remain completely in agreement with you.",
                                "score": 0.6660368343569761,
                                "type": "message",
                                "created_at": "2025-09-02T13:52:46.552383+00:00",
                                "metadata": {
                                    "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
                                    "session_id": "2209aabc-c2a6-4b76-9530-f4ef7f57a521",  # Different from request
                                    "level": "info",
                                    "retrieval": "vector",
                                    "source": "memory_database"
                                }
                            }
                        ],
                        "total": 2
                    }
                }
        
        # Arrange
        gateway = MockGateway()
        request_data = self.create_test_data_with_session()
        
        # Act
        response = await gateway.process_request(request_data)
        
        # Assert
        validate_response_schema(response, has_session_id=True)
        
        # Specific validation for session case
        data = response["data"]
        results = data["results"]
        
        # First result should be in_session
        assert results[0]["metadata"]["scope"] == "in_session"
        
        # Second result should be cross_session
        assert results[1]["metadata"]["scope"] == "cross_session"
    
    @pytest.mark.asyncio
    async def test_semantic_memory_schema_compliance(self):
        """Test semantic memory (M2) schema compliance."""
        
        class MockGateway(MemoryApiGateway):
            def __init__(self):
                super().__init__(None, None)
            
            async def _call_service(self, service_type, request_data, service_params):
                return {
                    "status": "success",
                    "data": {
                        "results": [
                            {
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
                        ],
                        "total": 1
                    }
                }
        
        # Arrange
        gateway = MockGateway()
        request_data = self.create_test_data_without_session()
        
        # Act
        response = await gateway.process_request(request_data)
        
        # Assert
        validate_response_schema(response, has_session_id=False)
        
        # Specific semantic memory validation
        result = response["data"]["results"][0]
        validate_semantic_result(result)
        
        # Check derived_from is preserved in metadata
        assert "derived_from" in result["metadata"]
        assert result["metadata"]["derived_from"] == [
            "m1_conversation_log_102", 
            "some_other_source_memory_id"
        ]
    
    @pytest.mark.asyncio
    async def test_task_eos_handling_schema_compliance(self):
        """Test task_eos handling maintains schema compliance."""
        
        class MockGateway(MemoryApiGateway):
            def __init__(self):
                super().__init__(None, None)
            
            async def _call_service(self, service_type, request_data, service_params):
                return {
                    "status": "success",
                    "data": {
                        "results": [
                            {
                                "id": "task-result-1",
                                "content": "Task completion result",
                                "score": 0.9,
                                "type": "chunk",
                                "created_at": "2025-09-02T13:52:46.552383+00:00",
                                "metadata": {
                                    "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
                                    "session_id": "1bcd351c-88b8-4f56-af50-14908955edcb"
                                }
                            }
                        ],
                        "total": 1
                    }
                }
        
        # Arrange
        gateway = MockGateway()
        request_data = self.create_test_data_with_task_eos()
        
        # Act
        response = await gateway.process_request(request_data)
        
        # Assert
        validate_response_schema(response, has_session_id=True)
        
        # task_eos should not appear in result metadata
        result = response["data"]["results"][0]
        assert "task_eos" not in result["metadata"]
        
        # But task and mode should be present
        assert result["metadata"]["task"] == "op_websearch_memory"
        assert result["metadata"]["mode"] is None
    
    def test_schema_validation_functions(self):
        """Test that our schema validation functions work correctly."""
        
        # Valid episodic result
        valid_episodic = {
            "id": "test-1",
            "content": "Test content",
            "relevance_score": 0.8,
            "memory_type": "episodic",
            "created_at": "2025-09-02T13:52:46.552383+00:00",
            "updated_at": None,
            "metadata": {
                "user_id": "user-1",
                "agent_id": "agent-1", 
                "session_id": "session-1",
                "session_name": "test-session",
                "scope": None,
                "task": None,
                "mode": None
            }
        }
        
        # Should not raise any assertions
        validate_result_schema(valid_episodic, has_session_id=False)
        
        # Valid semantic result
        valid_semantic = {
            "id": "test-2",
            "relevance_score": 0.9,
            "memory_type": "semantic",
            "created_at": "2025-09-02T13:52:46.552383+00:00",
            "updated_at": None,
            "fact": {
                "text": "Semantic knowledge",
                "triples": None
            },
            "metadata": {
                "user_id": "user-1",
                "agent_id": "agent-1",
                "session_id": "session-1", 
                "session_name": "test-session",
                "scope": "in_session",
                "derived_from": ["source-1"]
            }
        }
        
        # Should not raise any assertions
        validate_result_schema(valid_semantic, has_session_id=True)
        
        # Invalid result (missing required field)
        invalid_result = {
            "id": "test-3",
            "content": "Test content"
            # Missing required fields
        }
        
        # Should raise assertion
        with pytest.raises(AssertionError):
            validate_result_schema(invalid_result, has_session_id=False)


if __name__ == "__main__":
    # For manual testing
    test_instance = TestEndToEndSchemaCompliance()
    
    # Test the validation functions
    test_instance.test_schema_validation_functions()
    print("Schema validation functions work correctly!")
    
    # Note: Async tests need to be run with pytest