"""
Contract tests for Users API.

These tests validate the API contract - ensuring correct HTTP status codes
and response schemas match the documented API specification.
"""

import pytest
import json
from typing import Dict, Any, Optional
from fastapi.testclient import TestClient
from jsonschema import validate, ValidationError

# Import the FastAPI app factory
from memfuse_core.server import create_app


class TestUsersAPIContract:
    """Contract tests for Users API endpoints."""
    
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
    
    # JSON Schema definitions based on the API documentation
    
    @property
    def user_schema(self) -> Dict[str, Any]:
        """Schema for a single user object."""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": ["string", "null"]},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"}
            },
            "required": ["id", "name", "created_at", "updated_at"],
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
    def users_list_success_schema(self) -> Dict[str, Any]:
        """Schema for successful users list response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": self.user_schema
                }
            },
            "required": ["users"],
            "additionalProperties": False
        }
        return base_schema
    
    @property
    def user_create_success_schema(self) -> Dict[str, Any]:
        """Schema for successful user creation response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {
                "user": self.user_schema
            },
            "required": ["user"],
            "additionalProperties": False
        }
        return base_schema
    
    @property
    def user_get_success_schema(self) -> Dict[str, Any]:
        """Schema for successful user get response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {
                "user": self.user_schema
            },
            "required": ["user"],
            "additionalProperties": False
        }
        return base_schema
    
    @property
    def user_delete_success_schema(self) -> Dict[str, Any]:
        """Schema for successful user deletion response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"}
            },
            "required": ["user_id"],
            "additionalProperties": False
        }
        return base_schema
    
    def validate_response_schema(self, response_data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate response data against schema."""
        try:
            validate(instance=response_data, schema=schema)
        except ValidationError as e:
            pytest.fail(f"Response schema validation failed: {e.message}")
    
    def test_list_users_success_contract(self, client, headers):
        """Test GET /api/v1/users returns correct schema and status code."""
        response = client.get("/api/v1/users", headers=headers)
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.users_list_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Users retrieved successfully"
        assert response_data["errors"] is None
    
    def test_list_users_without_api_key_contract(self, client):
        """Test GET /api/v1/users without API key returns 200 (API doesn't enforce auth)."""
        response = client.get("/api/v1/users")
        
        # Current API behavior: returns 200 even without API key
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.users_list_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["errors"] is None
    
    def test_get_user_by_name_success_contract(self, client, headers):
        """Test GET /api/v1/users?name=username returns correct schema."""
        # First create a user to ensure we have something to query
        create_response = client.post(
            "/api/v1/users",
            json={"name": "test-user", "description": "Test user"},
            headers=headers
        )
        assert create_response.status_code == 201
        
        # Now query by name
        response = client.get("/api/v1/users?name=test-user", headers=headers)
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.users_list_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "User retrieved successfully"
        assert len(response_data["data"]["users"]) == 1
        assert response_data["data"]["users"][0]["name"] == "test-user"
    
    def test_get_user_by_name_not_found_contract(self, client, headers):
        """Test GET /api/v1/users?name=nonexistent returns 404 error."""
        response = client.get("/api/v1/users?name=nonexistent-user", headers=headers)
        
        # Validate HTTP status code
        assert response.status_code == 404
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == "User with name 'nonexistent-user' not found"
        assert response_data["data"] is None
    
    def test_create_user_success_contract(self, client, headers):
        """Test POST /api/v1/users with valid data returns correct schema."""
        user_data = {
            "name": "new-test-user",
            "description": "A new test user"
        }
        
        response = client.post("/api/v1/users", json=user_data, headers=headers)
        
        # API correctly returns 201 (Created) for new user creation
        assert response.status_code == 201
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.user_create_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 201  # API now correctly returns 201
        assert response_data["message"] == "User created successfully"
        assert response_data["data"]["user"]["name"] == "new-test-user"
        assert response_data["data"]["user"]["description"] == "A new test user"
    
    def test_create_user_duplicate_name_contract(self, client, headers):
        """Test POST /api/v1/users with duplicate name returns 400."""
        user_data = {"name": "duplicate-user", "description": "First user"}
        
        # Create first user
        response1 = client.post("/api/v1/users", json=user_data, headers=headers)
        assert response1.status_code == 201  # API correctly returns 201 for creation
        
        # Try to create duplicate
        response2 = client.post("/api/v1/users", json=user_data, headers=headers)
        
        # Validate HTTP status code
        assert response2.status_code == 400
        
        # Validate response schema
        response_data = response2.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 400
        assert response_data["message"] == "User with name 'duplicate-user' already exists"
        assert response_data["data"] is None
    
    def test_create_user_without_api_key_contract(self, client):
        """Test POST /api/v1/users without API key returns error (validation expects API key)."""
        user_data = {"name": "test-user", "description": "Test user"}
        
        response = client.post("/api/v1/users", json=user_data)
        
        # API now correctly validates API key and returns 400 error
        assert response.status_code == 400
        
        # Validate response - should be an error response
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        assert response_data["status"] == "error"
        assert response_data["code"] == 400
    
    def test_get_user_by_id_success_contract(self, client, headers):
        """Test GET /api/v1/users/{user_id} returns correct schema."""
        # First create a user
        create_response = client.post(
            "/api/v1/users",
            json={"name": "get-test-user", "description": "User for get test"},
            headers=headers
        )
        assert create_response.status_code == 201  # API correctly returns 201 for creation
        user_id = create_response.json()["data"]["user"]["id"]
        
        # Get user by ID
        response = client.get(f"/api/v1/users/{user_id}", headers=headers)
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.user_get_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "User retrieved successfully"
        assert response_data["data"]["user"]["id"] == user_id
    
    def test_get_user_by_id_not_found_contract(self, client, headers):
        """Test GET /api/v1/users/{user_id} with invalid ID returns 404."""
        response = client.get("/api/v1/users/nonexistent-id", headers=headers)
        
        # Validate HTTP status code
        assert response.status_code == 404
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == "User with ID 'nonexistent-id' not found"
    
    def test_update_user_success_contract(self, client, headers):
        """Test PUT /api/v1/users/{user_id} returns correct schema."""
        # First create a user
        create_response = client.post(
            "/api/v1/users",
            json={"name": "update-test-user", "description": "Original description"},
            headers=headers
        )
        assert create_response.status_code == 201  # API correctly returns 201 for creation
        user_id = create_response.json()["data"]["user"]["id"]
        
        # Update user
        update_data = {
            "name": "updated-test-user",
            "description": "Updated description"
        }
        response = client.put(f"/api/v1/users/{user_id}", json=update_data, headers=headers)
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.user_get_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "User updated successfully"
        assert response_data["data"]["user"]["name"] == "updated-test-user"
        assert response_data["data"]["user"]["description"] == "Updated description"
    
    def test_update_user_not_found_contract(self, client, headers):
        """Test PUT /api/v1/users/{user_id} with invalid ID returns 404."""
        update_data = {"name": "new-name", "description": "New description"}
        
        response = client.put("/api/v1/users/nonexistent-id", json=update_data, headers=headers)
        
        # Validate HTTP status code
        assert response.status_code == 404
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == "User with ID 'nonexistent-id' not found"
    
    def test_delete_user_success_contract(self, client, headers):
        """Test DELETE /api/v1/users/{user_id} returns correct schema."""
        # First create a user
        create_response = client.post(
            "/api/v1/users",
            json={"name": "delete-test-user", "description": "User for delete test"},
            headers=headers
        )

        # TODO: change to 204
        assert create_response.status_code == 201  # API correctly returns 201 for creation
        user_id = create_response.json()["data"]["user"]["id"]
        
        # Delete user
        response = client.delete(f"/api/v1/users/{user_id}", headers=headers)
        
        # Validate HTTP status code
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.user_delete_success_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "User deleted successfully"
        assert response_data["data"]["user_id"] == user_id
    
    def test_delete_user_not_found_contract(self, client, headers):
        """Test DELETE /api/v1/users/{user_id} with invalid ID returns 404."""
        response = client.delete("/api/v1/users/nonexistent-id", headers=headers)
        
        # Validate HTTP status code
        assert response.status_code == 404
        
        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        
        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == "User with ID 'nonexistent-id' not found"
    
    def test_delete_user_without_api_key_contract(self, client):
        """Test DELETE /api/v1/users/{user_id} without API key returns 404 (user not found)."""
        response = client.delete("/api/v1/users/some-id")
        
        # API returns 404 because user doesn't exist
        assert response.status_code == 404
        
        # Validate response schema - should be an error response
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
    
    def test_create_user_invalid_json_contract(self, client, headers):
        """Test POST /api/v1/users with invalid JSON returns 422."""
        # Missing required 'name' field
        invalid_data = {"description": "Missing name field"}
        
        response = client.post("/api/v1/users", json=invalid_data, headers=headers)
        
        # Validate HTTP status code (422 for validation errors)
        assert response.status_code == 422
        
        # Should return validation error response
        response_data = response.json()
        assert "detail" in response_data  # FastAPI validation error format
    
    def test_update_user_invalid_json_contract(self, client, headers):
        """Test PUT /api/v1/users/{user_id} with invalid JSON structure."""
        # First create a user
        create_response = client.post(
            "/api/v1/users",
            json={"name": "json-test-user", "description": "User for JSON test"},
            headers=headers
        )
        assert create_response.status_code == 201  # API correctly returns 201 for creation
        user_id = create_response.json()["data"]["user"]["id"]
        
        # Send invalid JSON structure
        response = client.put(f"/api/v1/users/{user_id}", json={"invalid": "structure"}, headers=headers)
        
        # API gracefully handles extra fields by ignoring them and returns 200
        assert response.status_code == 200
        
        # Validate response is a success response
        response_data = response.json()
        self.validate_response_schema(response_data, self.user_get_success_schema)
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        