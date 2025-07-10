"""
Contract tests for Memory API.

These tests validate the API contract for the Memory Query endpoint - ensuring correct 
HTTP status codes and response schemas match the documented API specification.
"""

import pytest
import json
from typing import Dict, Any, Optional
from fastapi.testclient import TestClient
from jsonschema import validate, ValidationError
from unittest.mock import patch, MagicMock, AsyncMock

# Import the FastAPI app factory
from memfuse_core.server import create_app


class TestMemoryAPIContract:
    """Contract tests for Memory API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(create_app())
    
    @pytest.fixture
    def valid_api_key(self):
        """Return a valid API key for testing."""
        return "test-api-key"
    
    @pytest.fixture
    def headers(self, valid_api_key):
        """Return headers with valid API key."""
        return {"X-API-Key": valid_api_key}
    
    def create_test_user(self, client, headers, name_suffix=""):
        """Create a test user and return its ID."""
        user_data = {
            "name": f"testuser_memory{name_suffix}",
            "description": "Test user for memory API testing"
        }
        response = client.post("/api/v1/users", json=user_data, headers=headers)
        assert response.status_code == 201
        return response.json()["data"]["user"]["id"]
    
    def create_test_agent(self, client, headers, name_suffix=""):
        """Create a test agent and return its ID."""
        agent_data = {
            "name": f"testagent_memory{name_suffix}",
            "description": "Test agent for memory API testing"
        }
        response = client.post("/api/v1/agents", json=agent_data, headers=headers)
        assert response.status_code == 201
        return response.json()["data"]["agent"]["id"]
    
    def create_test_session(self, client, headers, user_id, agent_id, name_suffix=""):
        """Create a test session and return its ID."""
        session_data = {
            "user_id": user_id,
            "agent_id": agent_id,
            "name": f"Test Memory Session{name_suffix}"
        }
        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        assert response.status_code == 201
        return response.json()["data"]["session"]["id"]

    @pytest.fixture
    def mock_memory_service(self):
        """Mock memory service for testing."""
        with patch('memfuse_core.services.service_factory.ServiceFactory.get_buffer_service_for_user') as mock_buffer, \
             patch('memfuse_core.services.service_factory.ServiceFactory.get_memory_service') as mock_memory:
            
            mock_service = AsyncMock()
            
            # Configure the async methods to return awaitables
            async def mock_get_buffer_service_for_user(*args, **kwargs):
                return mock_service
            
            async def mock_get_memory_service(*args, **kwargs):
                return mock_service
            
            mock_buffer.side_effect = mock_get_buffer_service_for_user
            mock_memory.side_effect = mock_get_memory_service
            
            yield mock_service

    # JSON Schema definitions based on the API documentation
    
    @property
    def memory_result_schema(self) -> Dict[str, Any]:
        """Schema for a single memory query result."""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "string"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "type": {"type": "string", "enum": ["message", "knowledge"]},
                "role": {"type": ["string", "null"]},
                "created_at": {"type": ["string", "null"]},
                "updated_at": {"type": ["string", "null"]},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "agent_id": {"type": ["string", "null"]},
                        "session_id": {"type": ["string", "null"]},
                        "session_name": {"type": ["string", "null"]},
                        "scope": {"type": ["string", "null"], "enum": ["in_session", "cross_session", None]},
                        "level": {"type": "integer"},
                        "retrieval": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "similarity": {"type": "number"}
                            },
                            "required": ["source"],
                            "additionalProperties": True
                        }
                    },
                    "required": ["user_id", "level", "retrieval"],
                    "additionalProperties": True
                }
            },
            "required": ["id", "content", "score", "type", "metadata"],
            "additionalProperties": False
        }
    
    @property
    def api_success_response_schema(self) -> Dict[str, Any]:
        """Schema for successful API responses."""
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success"]},
                "code": {"type": "integer"},
                "data": {"type": "object"},
                "message": {"type": "string"},
                "errors": {"type": "null"}
            },
            "required": ["status", "code", "data", "message", "errors"],
            "additionalProperties": False
        }
    
    @property
    def api_error_response_schema(self) -> Dict[str, Any]:
        """Schema for error API responses."""
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["error"]},
                "code": {"type": "integer"},
                "data": {"type": "null"},
                "message": {"type": "string"},
                "errors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "message": {"type": "string"}
                        },
                        "required": ["field", "message"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["status", "code", "data", "message", "errors"],
            "additionalProperties": False
        }
    
    @property
    def memory_query_success_schema(self) -> Dict[str, Any]:
        """Schema for successful memory query response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": self.memory_result_schema
                },
                "total": {"type": "integer", "minimum": 0}
            },
            "required": ["results", "total"],
            "additionalProperties": False
        }
        return base_schema
    
    def validate_response_schema(self, response_data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate response data against schema."""
        try:
            validate(instance=response_data, schema=schema)
        except ValidationError as e:
            pytest.fail(f"Response schema validation failed: {e.message}")
    
    def create_mock_results(self, count: int = 2, session_id: str = None, user_id: str = None) -> list:
        """Create mock memory query results."""
        results = []
        for i in range(count):
            result = {
                "id": f"msg-{i+1}",
                "content": f"This is test message {i+1}",
                "score": 0.95 - (i * 0.1),
                "type": "message",
                "role": "user" if i % 2 == 0 else "assistant",
                "created_at": "2023-01-01T12:00:00Z",
                "updated_at": "2023-01-01T12:00:00Z",
                "metadata": {
                    "user_id": user_id or "test-user",
                    "agent_id": "test-agent",
                    "session_id": session_id or "test-session",
                    "session_name": "Test Session",
                    "scope": "in_session" if session_id else None,
                    "level": 0,
                    "retrieval": {
                        "source": "vector_store",
                        "similarity": 0.95 - (i * 0.1)
                    }
                }
            }
            results.append(result)
        return results

    def test_query_memory_success_contract(self, client, headers, mock_memory_service):
        """Test POST /api/v1/users/{user_id}/query returns correct schema and status code."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_success")
        
        # Setup mock response
        mock_results = self.create_mock_results(count=2, user_id=user_id)
        mock_memory_service.query.return_value = {
            "status": "success",
            "data": {
                "results": mock_results,
                "total": len(mock_results)
            }
        }
        
        query_data = {
            "query": "test search query",
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.memory_query_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["errors"] is None
        assert "results" in response_data["data"]
        assert "total" in response_data["data"]
        assert isinstance(response_data["data"]["results"], list)
        assert isinstance(response_data["data"]["total"], int)

    def test_query_memory_with_session_scope_contract(self, client, headers, mock_memory_service):
        """Test memory query with session_id for scope tagging."""
        # Create test user, agent, and session
        user_id = self.create_test_user(client, headers, "_scope")
        agent_id = self.create_test_agent(client, headers, "_scope")
        session_id = self.create_test_session(client, headers, user_id, agent_id, "_scope")
        
        # Setup mock response with scope tagging
        mock_results = self.create_mock_results(count=2, session_id=session_id, user_id=user_id)
        mock_memory_service.query.return_value = {
            "status": "success",
            "data": {
                "results": mock_results,
                "total": len(mock_results)
            }
        }
        
        query_data = {
            "query": "test search query",
            "session_id": session_id,
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.memory_query_success_schema)
        
        # Validate scope tagging in results
        results = response_data["data"]["results"]
        if results:
            for result in results:
                assert "scope" in result["metadata"]
                assert result["metadata"]["scope"] in ["in_session", "cross_session"]

    def test_query_memory_with_agent_filter_contract(self, client, headers, mock_memory_service):
        """Test memory query with agent_id filter."""
        # Create test user and agent
        user_id = self.create_test_user(client, headers, "_agent")
        agent_id = self.create_test_agent(client, headers, "_agent")
        
        # Setup mock response
        mock_results = self.create_mock_results(count=1, user_id=user_id)
        mock_memory_service.query.return_value = {
            "status": "success",
            "data": {
                "results": mock_results,
                "total": len(mock_results)
            }
        }
        
        query_data = {
            "query": "test search query",
            "agent_id": agent_id,
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.memory_query_success_schema)

    def test_query_memory_with_store_type_contract(self, client, headers, mock_memory_service):
        """Test memory query with specific store_type."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_store")
        
        # Setup mock response
        mock_results = self.create_mock_results(count=1, user_id=user_id)
        mock_results[0]["metadata"]["retrieval"]["source"] = "vector_store"
        mock_memory_service.query.return_value = {
            "status": "success",
            "data": {
                "results": mock_results,
                "total": len(mock_results)
            }
        }
        
        query_data = {
            "query": "test search query",
            "store_type": "vector",
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.memory_query_success_schema)

    def test_query_memory_with_filters_contract(self, client, headers, mock_memory_service):
        """Test memory query with include/exclude filters."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_filters")
        
        # Setup mock response - only knowledge items
        mock_results = [{
            "id": "knowledge-1",
            "content": "This is a knowledge item",
            "score": 0.88,
            "type": "knowledge",
            "role": None,
            "created_at": "2023-01-01T10:00:00Z",
            "updated_at": "2023-01-01T10:00:00Z",
            "metadata": {
                "user_id": user_id,
                "agent_id": None,
                "session_id": None,
                "session_name": None,
                "scope": None,
                "level": 0,
                "retrieval": {
                    "source": "keyword_store",
                    "similarity": 0.88
                }
            }
        }]
        
        mock_memory_service.query.return_value = {
            "status": "success",
            "data": {
                "results": mock_results,
                "total": len(mock_results)
            }
        }
        
        query_data = {
            "query": "test search query",
            "include_messages": False,
            "include_knowledge": True,
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.memory_query_success_schema)

    def test_query_memory_empty_results_contract(self, client, headers, mock_memory_service):
        """Test memory query with no results."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_empty")
        
        # Setup mock response with empty results
        mock_memory_service.query.return_value = {
            "status": "success",
            "data": {
                "results": [],
                "total": 0
            }
        }
        
        query_data = {
            "query": "nonexistent search query",
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.memory_query_success_schema)
        
        # Validate empty results
        assert response_data["data"]["results"] == []
        assert response_data["data"]["total"] == 0

    def test_query_memory_user_not_found_contract(self, client, headers):
        """Test memory query with non-existent user returns 404."""
        query_data = {
            "query": "test search query",
            "top_k": 5
        }
        
        response = client.post(
            "/api/v1/users/nonexistent-user/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code
        assert response.status_code == 404
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["data"] is None
        assert "not found" in response_data["message"].lower()

    def test_query_memory_session_not_found_contract(self, client, headers):
        """Test memory query with non-existent session returns 404."""
        # Create test user, agent, and a valid session
        user_id = self.create_test_user(client, headers, "_session_not_found")
        agent_id = self.create_test_agent(client, headers, "_session_not_found")
        valid_session_id = self.create_test_session(client, headers, user_id, agent_id, "_valid")
        
        query_data = {
            "query": "test search query",
            "session_id": "nonexistent-session",  # Different from the valid session
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code
        assert response.status_code == 404
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["data"] is None
        assert "session" in response_data["message"].lower()

    def test_query_memory_agent_not_found_contract(self, client, headers):
        """Test memory query with non-existent agent returns 404."""
        # Create test user and a valid agent
        user_id = self.create_test_user(client, headers, "_agent_not_found")
        valid_agent_id = self.create_test_agent(client, headers, "_valid_agent")
        
        query_data = {
            "query": "test search query",
            "agent_id": "nonexistent-agent",  # Different from the valid agent
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code
        assert response.status_code == 404
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["data"] is None
        assert "agent" in response_data["message"].lower()

    def test_query_memory_missing_query_parameter_contract(self, client, headers):
        """Test memory query without required query parameter returns 422."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_missing_query")
        
        query_data = {
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code (422 for validation errors)
        assert response.status_code == 422
        
        # Should return validation error response
        response_data = response.json()
        assert "detail" in response_data  # FastAPI validation error format

    def test_query_memory_invalid_store_type_contract(self, client, headers):
        """Test memory query with invalid store_type returns 422."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_invalid_store")
        
        query_data = {
            "query": "test search query",
            "store_type": "invalid_store",
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code (422 for validation errors)
        assert response.status_code == 422
        
        # Should return validation error response
        response_data = response.json()
        assert "detail" in response_data  # FastAPI validation error format

    def test_query_memory_negative_top_k_contract(self, client, headers, mock_memory_service):
        """Test memory query with negative top_k (currently accepted by API)."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_negative_top_k")
        
        # Setup mock response (empty results for negative top_k)
        mock_memory_service.query.return_value = {
            "status": "success",
            "data": {
                "results": [],
                "total": 0
            }
        }
        
        query_data = {
            "query": "test search query",
            "top_k": -1  # Negative value - currently accepted by API
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Current API behavior: accepts negative values and returns 200
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.memory_query_success_schema)

    def test_query_memory_without_api_key_contract(self, client, headers):
        """Test memory query without API key returns 200 (API doesn't enforce auth)."""
        # Create test user (need headers for user creation)
        user_id = self.create_test_user(client, headers, "_no_api_key")
        
        query_data = {
            "query": "test search query",
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data
        )
        
        # Current API behavior: returns 200 even without API key
        assert response.status_code == 200

    def test_query_memory_invalid_json_contract(self, client, headers):
        """Test memory query with invalid JSON returns 422."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_invalid_json")
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            data="invalid json",
            headers=headers
        )
        
        # Validate HTTP status code (422 for validation errors)
        assert response.status_code == 422

    def test_query_memory_large_top_k_contract(self, client, headers, mock_memory_service):
        """Test memory query with large top_k value."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_large_top_k")
        
        # Setup mock response
        mock_results = self.create_mock_results(count=1, user_id=user_id)
        mock_memory_service.query.return_value = {
            "status": "success",
            "data": {
                "results": mock_results,
                "total": len(mock_results)
            }
        }
        
        query_data = {
            "query": "test search query",
            "top_k": 1000  # Large value
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Should still return 200 with results limited by actual data
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.memory_query_success_schema)

    def test_query_memory_invalid_store_type_hybrid_contract(self, client, headers):
        """Test memory query with invalid 'hybrid' store type returns 422."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_hybrid_invalid")
        
        query_data = {
            "query": "test search query",
            "store_type": "hybrid",  # Invalid - not in StoreType enum
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code (422 for validation errors)
        assert response.status_code == 422
        
        # Should return validation error response
        response_data = response.json()
        assert "detail" in response_data  # FastAPI validation error format

    def test_query_memory_graph_store_type_contract(self, client, headers, mock_memory_service):
        """Test memory query with valid 'graph' store type."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_graph")
        
        # Setup mock response
        mock_results = self.create_mock_results(count=1, user_id=user_id)
        mock_results[0]["metadata"]["retrieval"]["source"] = "graph_store"
        mock_memory_service.query.return_value = {
            "status": "success",
            "data": {
                "results": mock_results,
                "total": len(mock_results)
            }
        }
        
        query_data = {
            "query": "test search query",
            "store_type": "graph",  # Valid store type
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query",
            json=query_data,
            headers=headers
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.memory_query_success_schema)

    def test_query_memory_trailing_slash_contract(self, client, headers, mock_memory_service):
        """Test memory query endpoint with trailing slash."""
        # Create test user
        user_id = self.create_test_user(client, headers, "_trailing_slash")
        
        # Setup mock response
        mock_results = self.create_mock_results(count=1, user_id=user_id)
        mock_memory_service.query.return_value = {
            "status": "success",
            "data": {
                "results": mock_results,
                "total": len(mock_results)
            }
        }
        
        query_data = {
            "query": "test search query",
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/query/",  # Note trailing slash
            json=query_data,
            headers=headers
        )
        
        # Should handle trailing slash correctly
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.memory_query_success_schema) 