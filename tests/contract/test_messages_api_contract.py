"""
Contract tests for Messages API.

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


class TestMessagesAPIContract:
    """Contract tests for Messages API endpoints."""

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
    def test_session_id(self, client, headers):
        """Create a test session and return its ID."""
        import uuid
        
        # Create unique names to avoid conflicts
        unique_suffix = str(uuid.uuid4())[:8]
        
        # Create a user
        user_response = client.post(
            "/api/v1/users",
            json={"name": f"test-user-{unique_suffix}", "description": "Test user"},
            headers=headers,
        )
        assert user_response.status_code == 201
        user_id = user_response.json()["data"]["user"]["id"]

        # Create an agent
        agent_response = client.post(
            "/api/v1/agents",
            json={"name": f"test-agent-{unique_suffix}", "description": "Test agent"},
            headers=headers,
        )
        assert agent_response.status_code == 201
        agent_id = agent_response.json()["data"]["agent"]["id"]

        # Create a session
        session_response = client.post(
            "/api/v1/sessions",
            json={"user_id": user_id, "agent_id": agent_id, "name": f"test-session-{unique_suffix}"},
            headers=headers,
        )
        assert session_response.status_code == 201
        return session_response.json()["data"]["session"]["id"]

    # JSON Schema definitions based on the API documentation

    @property
    def message_schema(self) -> Dict[str, Any]:
        """Schema for a single message object."""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "session_id": {"type": "string"},
                "role": {"type": "string", "enum": ["user", "assistant", "system"]},
                "content": {"type": "string"},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"},
            },
            "required": ["id", "session_id", "role", "content", "created_at", "updated_at"],
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
    def messages_list_success_schema(self) -> Dict[str, Any]:
        """Schema for successful messages list response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"messages": {"type": "array", "items": self.message_schema}},
            "required": ["messages"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def messages_add_success_schema(self) -> Dict[str, Any]:
        """Schema for successful messages addition response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"message_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["message_ids"],
            "additionalProperties": True,  # Allow additional fields like transfer_triggered
        }
        return base_schema

    @property
    def message_get_success_schema(self) -> Dict[str, Any]:
        """Schema for successful message get response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"message": self.message_schema},
            "required": ["message"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def messages_read_success_schema(self) -> Dict[str, Any]:
        """Schema for successful messages read response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"messages": {"type": "array", "items": self.message_schema}},
            "required": ["messages"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def messages_update_success_schema(self) -> Dict[str, Any]:
        """Schema for successful messages update response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"message_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["message_ids"],
            "additionalProperties": False,
        }
        return base_schema

    @property
    def messages_delete_success_schema(self) -> Dict[str, Any]:
        """Schema for successful messages deletion response."""
        base_schema = self.api_success_response_schema.copy()
        base_schema["properties"]["data"] = {
            "type": "object",
            "properties": {"message_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["message_ids"],
            "additionalProperties": False,
        }
        return base_schema

    def validate_response_schema(self, response_data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate response data against schema."""
        try:
            validate(instance=response_data, schema=schema)
        except ValidationError as e:
            pytest.fail(f"Response schema validation failed: {e.message}")

    def test_list_messages_success_contract(self, client, headers, test_session_id):
        """Test GET /api/v1/sessions/{session_id}/messages returns correct schema and status code."""
        response = client.get(f"/api/v1/sessions/{test_session_id}/messages", headers=headers)

        # Validate HTTP status code
        assert response.status_code == 200

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.messages_list_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Messages retrieved successfully"
        assert response_data["errors"] is None

    def test_list_messages_with_limit_contract(self, client, headers, test_session_id):
        """Test GET /api/v1/sessions/{session_id}/messages with limit parameter."""
        response = client.get(f"/api/v1/sessions/{test_session_id}/messages?limit=10", headers=headers)

        # Validate HTTP status code
        assert response.status_code == 200

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.messages_list_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200

    def test_list_messages_with_sorting_contract(self, client, headers, test_session_id):
        """Test GET /api/v1/sessions/{session_id}/messages with sorting parameters."""
        response = client.get(
            f"/api/v1/sessions/{test_session_id}/messages?sort_by=timestamp&order=asc",
            headers=headers
        )

        # Validate HTTP status code
        assert response.status_code == 200

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.messages_list_success_schema)

    def test_list_messages_invalid_limit_contract(self, client, headers, test_session_id):
        """Test GET /api/v1/sessions/{session_id}/messages with invalid limit returns 400."""
        response = client.get(f"/api/v1/sessions/{test_session_id}/messages?limit=invalid", headers=headers)

        # Validate HTTP status code
        assert response.status_code == 400

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 400
        assert response_data["data"] is None

    def test_list_messages_invalid_sort_by_contract(self, client, headers, test_session_id):
        """Test GET /api/v1/sessions/{session_id}/messages with invalid sort_by returns 400."""
        response = client.get(
            f"/api/v1/sessions/{test_session_id}/messages?sort_by=invalid_field",
            headers=headers
        )

        # Validate HTTP status code
        assert response.status_code == 400

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)

    def test_list_messages_session_not_found_contract(self, client, headers):
        """Test GET /api/v1/sessions/{session_id}/messages with invalid session returns 404."""
        response = client.get("/api/v1/sessions/nonexistent-session/messages", headers=headers)

        # Debug: Print the actual response
        print(f"Actual status code: {response.status_code}")
        print(f"Actual response: {response.json()}")

        # Validate HTTP status code
        assert response.status_code == 404

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["data"] is None

    def test_add_messages_success_contract(self, client, headers, test_session_id):
        """Test POST /api/v1/sessions/{session_id}/messages with valid data returns correct schema."""
        message_data = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ]
        }

        response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages",
            json=message_data,
            headers=headers
        )

        # Validate HTTP status code
        assert response.status_code == 201

        # Validate response schema
        response_data = response.json()
        self.validate_response_schema(response_data, self.messages_add_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 201
        assert response_data["message"] == "Messages added successfully"
        assert len(response_data["data"]["message_ids"]) == 2

    def test_add_messages_invalid_role_contract(self, client, headers, test_session_id):
        """Test POST /api/v1/sessions/{session_id}/messages with invalid role returns 422."""
        message_data = {
            "messages": [
                {"role": "invalid_role", "content": "Hello, how are you?"}
            ]
        }

        response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages",
            json=message_data,
            headers=headers
        )

        # Validate HTTP status code (422 for validation errors)
        assert response.status_code == 422

        # Should return validation error response
        response_data = response.json()
        assert "detail" in response_data  # FastAPI validation error format

    def test_add_messages_empty_content_contract(self, client, headers, test_session_id):
        """Test POST /api/v1/sessions/{session_id}/messages with empty content returns 422."""
        message_data = {
            "messages": [
                {"role": "user", "content": ""}
            ]
        }

        response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages",
            json=message_data,
            headers=headers
        )

        # Validate HTTP status code (422 for validation errors)
        assert response.status_code == 422

        # Should return validation error response
        response_data = response.json()
        assert "detail" in response_data  # FastAPI validation error format

    def test_add_messages_session_not_found_contract(self, client, headers):
        """Test POST /api/v1/sessions/{session_id}/messages with invalid session returns 404."""
        message_data = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        }

        response = client.post(
            "/api/v1/sessions/nonexistent-session/messages",
            json=message_data,
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

    def test_read_messages_success_contract(self, client, headers, test_session_id):
        """Test POST /api/v1/sessions/{session_id}/messages/read with valid message IDs."""
        # First add some messages
        add_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages",
            json={
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"}
                ]
            },
            headers=headers
        )
        assert add_response.status_code == 201
        message_ids = add_response.json()["data"]["message_ids"]

        # Now read specific messages
        read_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages/read",
            json={"message_ids": message_ids},
            headers=headers
        )

        # Validate HTTP status code
        assert read_response.status_code == 200

        # Validate response schema
        response_data = read_response.json()
        self.validate_response_schema(response_data, self.messages_read_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Messages read successfully"

    def test_update_messages_success_contract(self, client, headers, test_session_id):
        """Test PUT /api/v1/sessions/{session_id}/messages with valid data."""
        # First add a message
        add_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages",
            json={
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            },
            headers=headers
        )
        assert add_response.status_code == 201
        message_ids = add_response.json()["data"]["message_ids"]

        # Update the message
        update_response = client.put(
            f"/api/v1/sessions/{test_session_id}/messages",
            json={
                "message_ids": message_ids,
                "new_messages": [
                    {"role": "user", "content": "Hello, how are you today?"}
                ]
            },
            headers=headers
        )

        # Validate HTTP status code
        assert update_response.status_code == 200

        # Validate response schema
        response_data = update_response.json()
        self.validate_response_schema(response_data, self.messages_update_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Messages updated successfully"

    def test_delete_messages_success_contract(self, client, headers, test_session_id):
        """Test DELETE /api/v1/sessions/{session_id}/messages with valid message IDs."""
        # First add a message
        add_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages",
            json={
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            },
            headers=headers
        )
        assert add_response.status_code == 201
        message_ids = add_response.json()["data"]["message_ids"]

        # Delete the message
        import json
        delete_response = client.request(
            "DELETE",
            f"/api/v1/sessions/{test_session_id}/messages",
            content=json.dumps({"message_ids": message_ids}),
            headers={**headers, "Content-Type": "application/json"}
        )

        # Validate HTTP status code
        assert delete_response.status_code == 200

        # Validate response schema
        response_data = delete_response.json()
        self.validate_response_schema(response_data, self.messages_delete_success_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "success"
        assert response_data["code"] == 200
        assert response_data["message"] == "Messages deleted successfully"

    def test_delete_messages_not_found_contract(self, client, headers, test_session_id):
        """Test DELETE /api/v1/sessions/{session_id}/messages with invalid message IDs returns 404."""
        import json
        delete_response = client.request(
            "DELETE",
            f"/api/v1/sessions/{test_session_id}/messages",
            content=json.dumps({"message_ids": ["nonexistent-message-id"]}),
            headers={**headers, "Content-Type": "application/json"}
        )

        # Validate HTTP status code
        assert delete_response.status_code == 404

        # Validate response schema
        response_data = delete_response.json()
        self.validate_response_schema(response_data, self.api_error_response_schema)

        # Validate specific contract requirements
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["data"] is None
