"""
Contract tests for Knowledge API.

These tests validate the API contract - ensuring correct HTTP status codes
and response schemas match the documented API specification.
"""

import pytest

from typing import Dict, Any
from fastapi.testclient import TestClient
from jsonschema import validate, ValidationError

# Import the FastAPI app factory
from memfuse_core.server import create_app


class CustomTestClient(TestClient):
    """Custom TestClient that supports DELETE with payload."""
    
    def delete_with_payload(self, **kwargs):
        """Delete request with payload support."""
        return self.request(method="DELETE", **kwargs)


class TestKnowledgeAPIContract:
    """Contract tests for Knowledge API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return CustomTestClient(create_app())

    @pytest.fixture
    def valid_api_key(self):
        """Return a valid API key for testing."""
        return "test-api-key"

    @pytest.fixture
    def headers(self, valid_api_key):
        """Return headers with valid API key."""
        return {"X-API-Key": valid_api_key}

    @pytest.fixture
    def test_user_id(self, client, headers):
        """Create a test user and return its ID."""
        import uuid
        
        # Create unique name to avoid conflicts
        unique_suffix = str(uuid.uuid4())[:8]
        
        # Create a user
        user_response = client.post(
            "/api/v1/users",
            json={"name": f"test-user-{unique_suffix}", "description": "Test user"},
            headers=headers,
        )
        assert user_response.status_code == 201
        return user_response.json()["data"]["user"]["id"]

    # JSON Schema definitions based on the API documentation

    @property
    def knowledge_schema(self) -> Dict[str, Any]:
        """Schema for a single knowledge object."""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "user_id": {"type": "string"},
                "content": {"type": "string"},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"},
            },
            "required": ["id", "user_id", "content", "created_at", "updated_at"],
            "additionalProperties": False,
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
                "errors": {"type": "null"},
            },
            "required": ["status", "code", "data", "message", "errors"],
            "additionalProperties": False,
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
                        "properties": {"field": {"type": "string"}, "message": {"type": "string"}},
                        "required": ["field", "message"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["status", "code", "data", "message", "errors"],
            "additionalProperties": False,
        }

    @property
    def knowledge_list_success_schema(self) -> Dict[str, Any]:
        """Schema for successful knowledge list response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"knowledge": {"type": "array", "items": self.knowledge_schema}},
            "required": ["knowledge"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def knowledge_add_success_schema(self) -> Dict[str, Any]:
        """Schema for successful knowledge addition response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"knowledge_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["knowledge_ids"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def knowledge_read_success_schema(self) -> Dict[str, Any]:
        """Schema for successful knowledge read response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"knowledge": {"type": "array", "items": self.knowledge_schema}},
            "required": ["knowledge"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def knowledge_update_success_schema(self) -> Dict[str, Any]:
        """Schema for successful knowledge update response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"knowledge_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["knowledge_ids"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def knowledge_delete_success_schema(self) -> Dict[str, Any]:
        """Schema for successful knowledge deletion response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"knowledge_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["knowledge_ids"],
            "additionalProperties": False,
        }
        return base_schema

    def validate_response_schema(self, response_data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate response data against schema."""
        try:
            validate(instance=response_data, schema=schema)
        except ValidationError as e:
            pytest.fail(f"Response schema validation failed: {e.message}")

    def test_list_knowledge_success_contract(self, client, headers, test_user_id):
        """Test GET /api/v1/users/{user_id}/knowledge returns correct schema and status code."""
        response = client.get(f"/api/v1/users/{test_user_id}/knowledge", headers=headers)
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.knowledge_list_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Knowledge items retrieved successfully"
        assert response_data["errors"] is None
        assert isinstance(response_data["data"]["knowledge"], list)

    def test_list_knowledge_user_not_found_contract(self, client, headers):
        """Test GET /api/v1/users/{user_id}/knowledge with non-existent user returns 404 error."""
        fake_user_id = "non-existent-user"
        response = client.get(f"/api/v1/users/{fake_user_id}/knowledge", headers=headers)
        
        # Validate HTTP status code
        assert response.status_code == 404  # API returns proper HTTP status code
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == f"User with ID '{fake_user_id}' not found"
        assert response_data["data"] is None
        assert len(response_data["errors"]) > 0

    def test_add_knowledge_success_contract(self, client, headers, test_user_id):
        """Test POST /api/v1/users/{user_id}/knowledge returns correct schema and status code."""
        knowledge_data = {
            "knowledge": [
                "This is test knowledge item 1",
                "This is test knowledge item 2",
                "This is test knowledge item 3"
            ]
        }
        
        response = client.post(
            f"/api/v1/users/{test_user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 201
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.knowledge_add_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 201
        assert response_data["message"] == "Knowledge items added successfully"
        assert response_data["errors"] is None
        assert len(response_data["data"]["knowledge_ids"]) == 3
        
        # All knowledge IDs should be strings
        for knowledge_id in response_data["data"]["knowledge_ids"]:
            assert isinstance(knowledge_id, str)

    def test_add_knowledge_user_not_found_contract(self, client, headers):
        """Test POST /api/v1/users/{user_id}/knowledge with non-existent user returns 404 error."""
        fake_user_id = "non-existent-user"
        knowledge_data = {
            "knowledge": ["Test knowledge item"]
        }
        
        response = client.post(
            f"/api/v1/users/{fake_user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 404  # API returns proper HTTP status code
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == f"User with ID '{fake_user_id}' not found"
        assert response_data["data"] is None

    def test_add_knowledge_empty_list_contract(self, client, headers, test_user_id):
        """Test POST /api/v1/knowledge with empty knowledge list."""
        knowledge_data = {
            "knowledge": []
        }
        
        response = client.post(
            f"/api/v1/users/{test_user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 201
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.knowledge_add_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 201
        assert response_data["message"] == "Knowledge items added successfully"
        assert response_data["errors"] is None
        assert len(response_data["data"]["knowledge_ids"]) == 0

    def test_read_knowledge_success_contract(self, client, headers, test_user_id):
        """Test POST /api/v1/knowledge/read returns correct schema and status code."""
        # First, add some knowledge items
        knowledge_data = {
            "knowledge": [
                "Test knowledge item 1",
                "Test knowledge item 2"
            ]
        }
        
        add_response = client.post(
            f"/api/v1/users/{test_user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        assert add_response.status_code == 201
        knowledge_ids = add_response.json()["data"]["knowledge_ids"]
        
        # Now read the knowledge items
        read_data = {
            "knowledge_ids": knowledge_ids
        }
        
        response = client.post(
            f"/api/v1/users/{test_user_id}/knowledge/read",
            json=read_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.knowledge_read_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Knowledge items retrieved successfully"
        assert response_data["errors"] is None
        assert len(response_data["data"]["knowledge"]) == 2
        
        # Validate knowledge item structure
        for knowledge_item in response_data["data"]["knowledge"]:
            assert knowledge_item["user_id"] == test_user_id
            assert knowledge_item["content"] in ["Test knowledge item 1", "Test knowledge item 2"]

    def test_read_knowledge_user_not_found_contract(self, client, headers):
        """Test POST /api/v1/knowledge/read with non-existent user returns 404 error."""
        fake_user_id = "non-existent-user"
        read_data = {
            "knowledge_ids": ["fake-id"]
        }
        
        response = client.post(
            f"/api/v1/users/{fake_user_id}/knowledge/read",
            json=read_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 404  # API returns proper HTTP status code
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == f"User with ID '{fake_user_id}' not found"
        assert response_data["data"] is None

    def test_read_knowledge_empty_list_contract(self, client, headers, test_user_id):
        """Test POST /api/v1/knowledge/read with empty knowledge_ids list."""
        read_data = {
            "knowledge_ids": []
        }
        
        response = client.post(
            f"/api/v1/users/{test_user_id}/knowledge/read",
            json=read_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.knowledge_read_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Knowledge items retrieved successfully"
        assert response_data["errors"] is None
        assert len(response_data["data"]["knowledge"]) == 0

    def test_update_knowledge_success_contract(self, client, headers, test_user_id):
        """Test PUT /api/v1/knowledge returns correct schema and status code."""
        # First, add some knowledge items
        knowledge_data = {
            "knowledge": [
                "Original knowledge item 1",
                "Original knowledge item 2"
            ]
        }
        
        add_response = client.post(
            f"/api/v1/users/{test_user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        assert add_response.status_code == 201
        knowledge_ids = add_response.json()["data"]["knowledge_ids"]
        
        # Now update the knowledge items
        update_data = {
            "knowledge_ids": knowledge_ids,
            "new_knowledge": [
                "Updated knowledge item 1",
                "Updated knowledge item 2"
            ]
        }
        
        response = client.put(
            f"/api/v1/users/{test_user_id}/knowledge",
            json=update_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.knowledge_update_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Knowledge items updated successfully"
        assert response_data["errors"] is None
        assert len(response_data["data"]["knowledge_ids"]) == 2
        
        # All knowledge IDs should be strings
        for knowledge_id in response_data["data"]["knowledge_ids"]:
            assert isinstance(knowledge_id, str)

    def test_update_knowledge_user_not_found_contract(self, client, headers):
        """Test PUT /api/v1/knowledge with non-existent user returns 404 error."""
        fake_user_id = "non-existent-user"
        update_data = {
            "knowledge_ids": ["fake-id"],
            "new_knowledge": ["Updated knowledge"]
        }
        
        response = client.put(
            f"/api/v1/users/{fake_user_id}/knowledge",
            json=update_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 404  # API returns proper HTTP status code
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == f"User with ID '{fake_user_id}' not found"
        assert response_data["data"] is None

    def test_update_knowledge_mismatched_arrays_contract(self, client, headers, test_user_id):
        """Test PUT /api/v1/knowledge with mismatched knowledge_ids and new_knowledge arrays."""
        # First, add a knowledge item
        knowledge_data = {
            "knowledge": ["Original knowledge item"]
        }
        
        add_response = client.post(
            f"/api/v1/users/{test_user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        assert add_response.status_code == 201
        knowledge_ids = add_response.json()["data"]["knowledge_ids"]
        
        # Try to update with mismatched arrays (more new_knowledge than knowledge_ids)
        update_data = {
            "knowledge_ids": knowledge_ids,
            "new_knowledge": [
                "Updated knowledge item 1",
                "Updated knowledge item 2"  # Extra item
            ]
        }
        
        response = client.put(
            f"/api/v1/users/{test_user_id}/knowledge",
            json=update_data,
            headers=headers,
        )
        
        # The API should handle this gracefully (may update only matching items)
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        # Could be success or error depending on implementation
        if response_data["status"] == "success":
            self.validate_response_schema(response_data, self.knowledge_update_success_schema)
        else:
            self.validate_response_schema(response_data, self.api_error_response_schema)

    def test_delete_knowledge_success_contract(self, client, headers, test_user_id):
        """Test DELETE /api/v1/knowledge returns correct schema and status code."""
        # First, add some knowledge items
        knowledge_data = {
            "knowledge": [
                "Knowledge item to delete 1",
                "Knowledge item to delete 2"
            ]
        }
        
        add_response = client.post(
            f"/api/v1/users/{test_user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        assert add_response.status_code == 201
        knowledge_ids = add_response.json()["data"]["knowledge_ids"]
        
        # Now delete the knowledge items
        delete_data = {
            "knowledge_ids": knowledge_ids
        }
        
        response = client.delete_with_payload(
            url=f"/api/v1/users/{test_user_id}/knowledge",
            json=delete_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.knowledge_delete_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Knowledge items deleted successfully"
        assert response_data["errors"] is None
        assert len(response_data["data"]["knowledge_ids"]) == 2
        
        # All knowledge IDs should be strings
        for knowledge_id in response_data["data"]["knowledge_ids"]:
            assert isinstance(knowledge_id, str)

    def test_delete_knowledge_user_not_found_contract(self, client, headers):
        """Test DELETE /api/v1/knowledge with non-existent user returns 404 error."""
        fake_user_id = "non-existent-user"
        delete_data = {
            "knowledge_ids": ["fake-id"]
        }
        
        response = client.delete_with_payload(
            url=f"/api/v1/users/{fake_user_id}/knowledge",
            json=delete_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 404  # API returns proper HTTP status code
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == f"User with ID '{fake_user_id}' not found"
        assert response_data["data"] is None

    def test_delete_knowledge_empty_list_contract(self, client, headers, test_user_id):
        """Test DELETE /api/v1/users/{user_id}/knowledge with empty knowledge_ids list."""
        delete_data = {
            "knowledge_ids": []
        }
        
        response = client.delete_with_payload(
            url=f"/api/v1/users/{test_user_id}/knowledge",
            json=delete_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.knowledge_delete_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Knowledge items deleted successfully"
        assert response_data["errors"] is None
        assert len(response_data["data"]["knowledge_ids"]) == 0

    def test_delete_knowledge_non_existent_ids_contract(self, client, headers, test_user_id):
        """Test DELETE /api/v1/users/{user_id}/knowledge with non-existent knowledge IDs."""
        delete_data = {
            "knowledge_ids": ["non-existent-id-1", "non-existent-id-2"]
        }
        
        response = client.delete_with_payload(
            url=f"/api/v1/users/{test_user_id}/knowledge",
            json=delete_data,
            headers=headers,
        )
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.knowledge_delete_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Knowledge items deleted successfully"
        assert response_data["errors"] is None
        # Should return empty list since no items were actually deleted
        assert len(response_data["data"]["knowledge_ids"]) == 0

    def test_knowledge_api_invalid_json_contract(self, client, headers, test_user_id):
        """Test knowledge API endpoints with invalid JSON."""
        # Test with invalid JSON structure
        invalid_data = {"invalid_field": "value"}
        
        response = client.post(
            f"/api/v1/users/{test_user_id}/knowledge",
            json=invalid_data,
            headers=headers,
        )
        
        # Should return 422 for validation error
        assert response.status_code == 422 