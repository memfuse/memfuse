"""
Contract tests for Agents API.

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


class TestAgentsAPIContract:
    """Contract tests for Agents API endpoints."""

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
    def agent_schema(self) -> Dict[str, Any]:
        """Schema for a single agent object."""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": ["string", "null"]},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"},
            },
            "required": ["id", "name", "created_at", "updated_at"],
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
    def agents_list_success_schema(self) -> Dict[str, Any]:
        """Schema for successful agents list response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"agents": {"type": "array", "items": self.agent_schema}},
            "required": ["agents"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def agent_create_success_schema(self) -> Dict[str, Any]:
        """Schema for successful agent creation response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"agent": self.agent_schema},
            "required": ["agent"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def agent_get_success_schema(self) -> Dict[str, Any]:
        """Schema for successful agent get response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"agent": self.agent_schema},
            "required": ["agent"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def agent_delete_success_schema(self) -> Dict[str, Any]:
        """Schema for successful agent deletion response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"agent_id": {"type": "string"}},
            "required": ["agent_id"],
            "additionalProperties": False,
        }
        return base_schema

    def validate_response_schema(self, response_data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate response data against schema."""
        try:
            validate(instance=response_data, schema=schema)
        except ValidationError as e:
            pytest.fail(f"Response schema validation failed: {e.message}")

    def test_list_agents_success_contract(self, client, headers):
        """Test GET /api/v1/agents returns correct schema and status code."""
        response = client.get("/api/v1/agents", headers=headers)

        # Validate HTTP status code
        assert response.status_code == 200

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.agents_list_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Agents retrieved successfully"
        assert response_data["errors"] is None

    def test_get_agent_by_name_success_contract(self, client, headers):
        """Test GET /api/v1/agents?name=agentname returns correct schema."""
        # First create an agent to ensure we have something to query
        create_response = client.post(
            "/api/v1/agents", json={"name": "test-agent", "description": "Test agent"}, headers=headers
        )
        assert create_response.status_code == 201

        # Now query by name
        response = client.get("/api/v1/agents?name=test-agent", headers=headers)

        # Validate HTTP status code
        assert response.status_code == 200

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.agents_list_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Agent retrieved successfully"
        assert len(response_data["data"]["agents"]) == 1
        assert response_data["data"]["agents"][0]["name"] == "test-agent"

    def test_get_agent_by_name_not_found_contract(self, client, headers):
        """Test GET /api/v1/agents?name=nonexistent returns 404 error."""
        response = client.get("/api/v1/agents?name=nonexistent-agent", headers=headers)

        # Validate HTTP status code
        assert response.status_code == 404

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == "Agent with name 'nonexistent-agent' not found"
        assert response_data["data"] is None

    def test_create_agent_success_contract(self, client, headers):
        """Test POST /api/v1/agents with valid data returns correct schema."""
        agent_data = {"name": "new-test-agent", "description": "A new test agent"}

        response = client.post("/api/v1/agents", json=agent_data, headers=headers)

        # API correctly returns 201 (Created) for new agent creation
        assert response.status_code == 201

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.agent_create_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 201
        assert response_data["message"] == "Agent created successfully"
        assert response_data["data"]["agent"]["name"] == "new-test-agent"
        assert response_data["data"]["agent"]["description"] == "A new test agent"

    def test_create_agent_duplicate_name_contract(self, client, headers):
        """Test POST /api/v1/agents with duplicate name returns 400."""
        agent_data = {"name": "duplicate-agent", "description": "First agent"}

        # Create first agent
        response1 = client.post("/api/v1/agents", json=agent_data, headers=headers)
        assert response1.status_code == 201

        # Try to create duplicate
        response2 = client.post("/api/v1/agents", json=agent_data, headers=headers)

        # Validate HTTP status code
        assert response2.status_code == 400

        # Validate response schema
        response_data = response2.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 400
        assert response_data["message"] == "Agent with name 'duplicate-agent' already exists"
        assert response_data["data"] is None

    def test_get_agent_by_id_success_contract(self, client, headers):
        """Test GET /api/v1/agents/{agent_id} returns correct schema."""
        # First create a agent
        create_response = client.post(
            "/api/v1/agents",
            json={"name": "get-test-agent", "description": "Agent for get test"},
            headers=headers,
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["data"]["agent"]["id"]

        # Get agent by ID
        response = client.get(f"/api/v1/agents/{agent_id}", headers=headers)

        # Validate HTTP status code
        assert response.status_code == 200

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.agent_get_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Agent retrieved successfully"
        assert response_data["data"]["agent"]["id"] == agent_id

    def test_get_agent_by_id_not_found_contract(self, client, headers):
        """Test GET /api/v1/agents/{agent_id} with invalid ID returns 404."""
        response = client.get("/api/v1/agents/nonexistent-id", headers=headers)

        # Validate HTTP status code
        assert response.status_code == 404

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == "Agent with ID 'nonexistent-id' not found"

    def test_update_agent_success_contract(self, client, headers):
        """Test PUT /api/v1/agents/{agent_id} returns correct schema."""
        # First create an agent
        create_response = client.post(
            "/api/v1/agents",
            json={"name": "update-test-agent", "description": "Original description"},
            headers=headers,
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["data"]["agent"]["id"]

        # Update agent
        update_data = {"name": "updated-test-agent", "description": "Updated description"}
        response = client.put(f"/api/v1/agents/{agent_id}", json=update_data, headers=headers)

        # Validate HTTP status code
        assert response.status_code == 200

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.agent_get_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Agent updated successfully"
        assert response_data["data"]["agent"]["name"] == "updated-test-agent"
        assert response_data["data"]["agent"]["description"] == "Updated description"

    def test_update_agent_not_found_contract(self, client, headers):
        """Test PUT /api/v1/agents/{agent_id} with invalid ID returns 404."""
        update_data = {"name": "new-name", "description": "New description"}

        response = client.put("/api/v1/agents/nonexistent-id", json=update_data, headers=headers)

        # Validate HTTP status code
        assert response.status_code == 404

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == "Agent with ID 'nonexistent-id' not found"

    def test_delete_agent_success_contract(self, client, headers):
        """Test DELETE /api/v1/agents/{agent_id} returns correct schema."""
        # First create an agent
        create_response = client.post(
            "/api/v1/agents",
            json={"name": "delete-test-agent", "description": "Agent for delete test"},
            headers=headers,
        )

        assert create_response.status_code == 201
        agent_id = create_response.json()["data"]["agent"]["id"]

        # Delete agent
        response = client.delete(f"/api/v1/agents/{agent_id}", headers=headers)

        # Validate HTTP status code
        assert response.status_code == 200

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.agent_delete_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Agent deleted successfully"
        assert response_data["data"]["agent_id"] == agent_id

    def test_delete_agent_not_found_contract(self, client, headers):
        """Test DELETE /api/v1/agents/{agent_id} with invalid ID returns 404."""
        response = client.delete("/api/v1/agents/nonexistent-id", headers=headers)

        # Validate HTTP status code
        assert response.status_code == 404

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["message"] == "Agent with ID 'nonexistent-id' not found"

    def test_create_agent_invalid_json_contract(self, client, headers):
        """Test POST /api/v1/agents with invalid JSON returns 422."""
        # Missing required 'name' field
        invalid_data = {"description": "Missing name field"}

        response = client.post("/api/v1/agents", json=invalid_data, headers=headers)

        # Validate HTTP status code (422 for validation errors)
        assert response.status_code == 422

        # Should return validation error response
        response_data = response.json()
        assert "detail" in response_data  # FastAPI validation error format 