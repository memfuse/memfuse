"""
Contract tests for Sessions API.

These tests validate the API contract - ensuring correct HTTP status codes
and response schemas match the documented API specification.
"""

import pytest
import uuid
from typing import Dict, Any
from fastapi.testclient import TestClient
from jsonschema import validate, ValidationError

from memfuse_core.server import create_app


class TestSessionsAPIContract:
    """Contract tests for Sessions API endpoints."""

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

    @pytest.fixture
    def setup_user_and_agent(self, client, headers):
        """Create a user and an agent for session tests."""
        # Create unique names for each test
        unique_id = str(uuid.uuid4())[:8]
        user_name = f"test-user-session-{unique_id}"
        agent_name = f"test-agent-session-{unique_id}"
        
        user_res = client.post("/api/v1/users", json={"name": user_name}, headers=headers)
        assert user_res.status_code == 201
        user_id = user_res.json()["data"]["user"]["id"]

        agent_res = client.post("/api/v1/agents", json={"name": agent_name}, headers=headers)
        assert agent_res.status_code == 201
        agent_id = agent_res.json()["data"]["agent"]["id"]
        
        return user_id, agent_id

    # JSON Schema definitions based on the API documentation

    @property
    def session_schema(self) -> Dict[str, Any]:
        """Schema for a single session object."""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "user_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "name": {"type": ["string", "null"]},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"},
            },
            "required": ["id", "user_id", "agent_id", "created_at", "updated_at"],
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
                "data": {"type": ["object", "null"]},
                "message": {"type": "string"},
                "errors": {"type": "null"},
            },
            "required": ["status", "code", "message", "errors"],
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
    def sessions_list_success_schema(self) -> Dict[str, Any]:
        """Schema for successful sessions list response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"sessions": {"type": "array", "items": self.session_schema}},
            "required": ["sessions"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def session_create_success_schema(self) -> Dict[str, Any]:
        """Schema for successful session creation response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"session": self.session_schema},
            "required": ["session"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def session_get_success_schema(self) -> Dict[str, Any]:
        """Schema for successful session get response."""
        return self.session_create_success_schema

    @property
    def session_delete_success_schema(self) -> Dict[str, Any]:
        """Schema for successful session deletion response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
            "additionalProperties": False,
        }
        return base_schema

    def validate_response_schema(self, response_data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate response data against schema."""
        try:
            validate(instance=response_data, schema=schema)
        except ValidationError as e:
            pytest.fail(f"Response schema validation failed: {e.message}")

    def test_list_sessions_success_contract(self, client, headers):
        """Test GET /api/v1/sessions returns correct schema and status code."""
        response = client.get("/api/v1/sessions", headers=headers)
        assert response.status_code == 200
        response_data = response.json()
        self.validate_response_schema(response_data, self.sessions_list_success_schema)
        assert response_data["status"] == "success"
        assert response_data["message"] == "Sessions retrieved successfully"

    def test_create_session_success_contract(self, client, headers, setup_user_and_agent):
        """Test POST /api/v1/sessions with valid data returns correct schema."""
        user_id, agent_id = setup_user_and_agent
        session_name = f"test-session-{str(uuid.uuid4())[:8]}"
        session_data = {"user_id": user_id, "agent_id": agent_id, "name": session_name}

        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        
        # The API can return either 201 (new session) or 200 (updated existing session)
        assert response.status_code in [200, 201]
        response_data = response.json()
        self.validate_response_schema(response_data, self.session_create_success_schema)
        assert response_data["status"] == "success"
        assert response_data["data"]["session"]["name"] == session_name

    def test_create_session_auto_generated_name_contract(self, client, headers, setup_user_and_agent):
        """Test POST /api/v1/sessions without name auto-generates name."""
        user_id, agent_id = setup_user_and_agent
        session_data = {"user_id": user_id, "agent_id": agent_id}

        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        
        assert response.status_code in [200, 201]
        response_data = response.json()
        self.validate_response_schema(response_data, self.session_create_success_schema)
        assert response_data["status"] == "success"
        # Name should be auto-generated
        assert response_data["data"]["session"]["name"] is not None

    def test_create_session_invalid_user_contract(self, client, headers, setup_user_and_agent):
        """Test POST /api/v1/sessions with invalid user_id returns 404."""
        _, agent_id = setup_user_and_agent
        session_data = {"user_id": "invalid-user", "agent_id": agent_id, "name": "test-session"}

        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        
        assert response.status_code == 404
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        assert response_data["status"] == "error"
        assert "User with ID 'invalid-user' not found" in response_data["message"]

    def test_create_session_invalid_agent_contract(self, client, headers, setup_user_and_agent):
        """Test POST /api/v1/sessions with invalid agent_id returns 404."""
        user_id, _ = setup_user_and_agent
        session_data = {"user_id": user_id, "agent_id": "invalid-agent", "name": "test-session"}

        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        
        assert response.status_code == 404
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        assert response_data["status"] == "error"
        assert "Agent with ID 'invalid-agent' not found" in response_data["message"]

    def test_get_session_by_id_success_contract(self, client, headers, setup_user_and_agent):
        """Test GET /api/v1/sessions/{session_id} returns correct schema."""
        user_id, agent_id = setup_user_and_agent
        session_name = f"get-test-session-{str(uuid.uuid4())[:8]}"
        session_data = {"user_id": user_id, "agent_id": agent_id, "name": session_name}
        create_response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        session_id = create_response.json()["data"]["session"]["id"]

        response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
        
        assert response.status_code == 200
        response_data = response.json()
        self.validate_response_schema(response_data, self.session_get_success_schema)
        assert response_data["status"] == "success"
        assert response_data["data"]["session"]["id"] == session_id

    def test_get_session_by_id_not_found_contract(self, client, headers):
        """Test GET /api/v1/sessions/{session_id} with invalid ID returns 404."""
        response = client.get("/api/v1/sessions/nonexistent-id", headers=headers)
        
        assert response.status_code == 404
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        assert response_data["status"] == "error"
        assert "Session with ID 'nonexistent-id' not found" in response_data["message"]

    def test_get_session_by_name_success_contract(self, client, headers, setup_user_and_agent):
        """Test GET /api/v1/sessions?name=session_name returns correct schema."""
        user_id, agent_id = setup_user_and_agent
        session_name = f"query-test-session-{str(uuid.uuid4())[:8]}"
        session_data = {"user_id": user_id, "agent_id": agent_id, "name": session_name}
        create_response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        assert create_response.status_code in [200, 201]

        response = client.get(f"/api/v1/sessions?name={session_name}", headers=headers)
        
        assert response.status_code == 200
        response_data = response.json()
        self.validate_response_schema(response_data, self.sessions_list_success_schema)
        assert response_data["status"] == "success"
        assert response_data["message"] == "Session retrieved successfully"
        assert len(response_data["data"]["sessions"]) == 1
        assert response_data["data"]["sessions"][0]["name"] == session_name

    def test_get_session_by_name_not_found_contract(self, client, headers):
        """Test GET /api/v1/sessions?name=nonexistent returns 404."""
        response = client.get("/api/v1/sessions?name=nonexistent-session", headers=headers)
        
        assert response.status_code == 404
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        assert response_data["status"] == "error"
        assert "Session with name 'nonexistent-session' not found" in response_data["message"]

    def test_list_sessions_with_filters_contract(self, client, headers, setup_user_and_agent):
        """Test GET /api/v1/sessions with user_id and agent_id filters."""
        user_id, agent_id = setup_user_and_agent
        session_name = f"filter-test-session-{str(uuid.uuid4())[:8]}"
        session_data = {"user_id": user_id, "agent_id": agent_id, "name": session_name}
        create_response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        assert create_response.status_code in [200, 201]

        # Test filtering by user_id
        response = client.get(f"/api/v1/sessions?user_id={user_id}", headers=headers)
        assert response.status_code == 200
        response_data = response.json()
        self.validate_response_schema(response_data, self.sessions_list_success_schema)
        assert response_data["status"] == "success"
        
        # Test filtering by agent_id
        response = client.get(f"/api/v1/sessions?agent_id={agent_id}", headers=headers)
        assert response.status_code == 200
        response_data = response.json()
        self.validate_response_schema(response_data, self.sessions_list_success_schema)
        assert response_data["status"] == "success"

    def test_update_session_success_contract(self, client, headers, setup_user_and_agent):
        """Test PUT /api/v1/sessions/{session_id} returns correct schema."""
        user_id, agent_id = setup_user_and_agent
        session_name = f"update-test-session-{str(uuid.uuid4())[:8]}"
        session_data = {"user_id": user_id, "agent_id": agent_id, "name": session_name}
        create_response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        session_id = create_response.json()["data"]["session"]["id"]

        new_name = f"updated-session-name-{str(uuid.uuid4())[:8]}"
        update_data = {"name": new_name}
        response = client.put(f"/api/v1/sessions/{session_id}", json=update_data, headers=headers)
        
        assert response.status_code == 200
        response_data = response.json()
        self.validate_response_schema(response_data, self.session_get_success_schema)
        assert response_data["status"] == "success"
        assert response_data["data"]["session"]["name"] == new_name

    def test_update_session_not_found_contract(self, client, headers):
        """Test PUT /api/v1/sessions/{session_id} with invalid ID returns 404."""
        update_data = {"name": "new-name"}
        response = client.put("/api/v1/sessions/nonexistent-id", json=update_data, headers=headers)
        
        assert response.status_code == 404
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        assert response_data["status"] == "error"
        assert "Session with ID 'nonexistent-id' not found" in response_data["message"]

    def test_delete_session_success_contract(self, client, headers, setup_user_and_agent):
        """Test DELETE /api/v1/sessions/{session_id} returns correct schema."""
        user_id, agent_id = setup_user_and_agent
        session_name = f"delete-test-session-{str(uuid.uuid4())[:8]}"
        session_data = {"user_id": user_id, "agent_id": agent_id, "name": session_name}
        create_response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        session_id = create_response.json()["data"]["session"]["id"]

        response = client.delete(f"/api/v1/sessions/{session_id}", headers=headers)
        
        assert response.status_code == 200
        response_data = response.json()
        self.validate_response_schema(response_data, self.session_delete_success_schema)
        assert response_data["status"] == "success"
        assert response_data["data"]["session_id"] == session_id

    def test_delete_session_not_found_contract(self, client, headers):
        """Test DELETE /api/v1/sessions/{session_id} with invalid ID returns 404."""
        response = client.delete("/api/v1/sessions/nonexistent-id", headers=headers)
        
        assert response.status_code == 404
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)
        assert response_data["status"] == "error"
        assert "Session with ID 'nonexistent-id' not found" in response_data["message"]

    def test_create_session_invalid_json_contract(self, client, headers):
        """Test POST /api/v1/sessions with invalid JSON returns 422."""
        # Missing required fields
        invalid_data = {"name": "test-session"}  # missing user_id and agent_id

        response = client.post("/api/v1/sessions", json=invalid_data, headers=headers)
        
        # FastAPI returns 422 for validation errors
        assert response.status_code == 422
        response_data = response.json()
        assert "detail" in response_data  # FastAPI validation error format
