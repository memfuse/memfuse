"""
Integration tests for message functionality across API calls.

These tests verify that messages are properly handled and can be retrieved
across separate API requests, simulating real-world usage patterns.
"""

import pytest
import asyncio
import time
from typing import Dict, Any
from fastapi.testclient import TestClient

from memfuse_core.server import create_app
from memfuse_core.services.service_factory import ServiceFactory


class TestMessageFunctionality:
    """Integration tests for message functionality across API calls."""

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
        user_data = user_response.json()
        if user_data is None:
            pytest.skip("User creation failed - API response is None")
        user_id = user_data["data"]["user"]["id"]

        # Create an agent
        agent_response = client.post(
            "/api/v1/agents",
            json={"name": f"test-agent-{unique_suffix}", "description": "Test agent"},
            headers=headers,
        )
        assert agent_response.status_code == 201
        agent_data = agent_response.json()
        if agent_data is None:
            pytest.skip("Agent creation failed - API response is None")
        agent_id = agent_data["data"]["agent"]["id"]

        # Create a session
        session_response = client.post(
            "/api/v1/sessions",
            json={"user_id": user_id, "agent_id": agent_id, "name": f"test-session-{unique_suffix}"},
            headers=headers,
        )
        assert session_response.status_code == 201
        session_data = session_response.json()
        if session_data is None:
            pytest.skip("Session creation failed - API response is None")
        return session_data["data"]["session"]["id"]

    def test_message_immediate_access(self, client, headers, test_session_id):
        """Test that messages added can be read immediately (buffer behavior)."""
        # Add messages
        add_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages",
            json={
                "messages": [
                    {"role": "user", "content": "Hello, immediate test"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            },
            headers=headers
        )
        assert add_response.status_code == 200
        message_ids = add_response.json()["data"]["message_ids"]
        assert len(message_ids) == 2

        # Read messages immediately
        read_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages/read",
            json={"message_ids": message_ids},
            headers=headers
        )
        assert read_response.status_code == 200
        
        messages = read_response.json()["data"]["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] in ["Hello, immediate test", "Hi there!"]

    def test_message_delayed_access(self, client, headers, test_session_id):
        """Test that messages remain accessible after a delay (simulating separate requests)."""
        # Add messages
        add_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages",
            json={
                "messages": [
                    {"role": "user", "content": "Hello, delayed test"},
                    {"role": "assistant", "content": "Hi from delayed test!"}
                ]
            },
            headers=headers
        )
        assert add_response.status_code == 200
        message_ids = add_response.json()["data"]["message_ids"]
        assert len(message_ids) == 2

        # Wait a bit to simulate delay between requests
        time.sleep(2)

        # Read messages after delay
        read_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages/read",
            json={"message_ids": message_ids},
            headers=headers
        )
        assert read_response.status_code == 200
        
        messages = read_response.json()["data"]["messages"]
        assert len(messages) == 2
        assert any(msg["content"] == "Hello, delayed test" for msg in messages)
        assert any(msg["content"] == "Hi from delayed test!" for msg in messages)

    def test_message_cross_session_access(self, client, headers, test_session_id):
        """Test that messages remain accessible across different API sessions."""
        # Add messages in first client instance
        add_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages",
            json={
                "messages": [
                    {"role": "user", "content": "Hello, session test"},
                    {"role": "assistant", "content": "Hi from session test!"}
                ]
            },
            headers=headers
        )
        assert add_response.status_code == 200
        message_ids = add_response.json()["data"]["message_ids"]

        # Create a new client instance to simulate separate session
        new_client = TestClient(create_app())
        
        # Read messages with new client
        read_response = new_client.post(
            f"/api/v1/sessions/{test_session_id}/messages/read",
            json={"message_ids": message_ids},
            headers=headers
        )
        assert read_response.status_code == 200
        
        messages = read_response.json()["data"]["messages"]
        assert len(messages) == 2
        assert any(msg["content"] == "Hello, session test" for msg in messages)

    def test_message_buffer_flush_handling(self, client, headers, test_session_id):
        """Test that messages remain accessible after buffer flush operations."""
        # Add messages
        add_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages",
            json={
                "messages": [
                    {"role": "user", "content": "Hello, flush test"},
                    {"role": "assistant", "content": "Hi from flush test!"}
                ]
            },
            headers=headers
        )
        assert add_response.status_code == 200
        message_ids = add_response.json()["data"]["message_ids"]

        # Trigger buffer flush by creating many messages (to exceed buffer limits)
        for i in range(10):
            client.post(
                f"/api/v1/sessions/{test_session_id}/messages",
                json={
                    "messages": [
                        {"role": "user", "content": f"Filler message {i}"}
                    ]
                },
                headers=headers
            )

        # Wait for potential flush operations
        time.sleep(1)

        # Read original messages
        read_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages/read",
            json={"message_ids": message_ids},
            headers=headers
        )
        assert read_response.status_code == 200
        
        messages = read_response.json()["data"]["messages"]
        assert len(messages) == 2
        assert any(msg["content"] == "Hello, flush test" for msg in messages)

    def test_message_list_includes_added_messages(self, client, headers, test_session_id):
        """Test that the list messages API includes recently added messages."""
        # Add messages
        add_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages",
            json={
                "messages": [
                    {"role": "user", "content": "Hello, list test"},
                    {"role": "assistant", "content": "Hi from list test!"}
                ]
            },
            headers=headers
        )
        assert add_response.status_code == 200

        # Wait a bit to ensure messages are available
        time.sleep(1)

        # List messages
        list_response = client.get(
            f"/api/v1/sessions/{test_session_id}/messages",
            headers=headers
        )
        assert list_response.status_code == 200
        
        messages = list_response.json()["data"]["messages"]
        content_list = [msg["content"] for msg in messages]
        
        # Check that our added messages are in the list
        assert "Hello, list test" in content_list
        assert "Hi from list test!" in content_list

    def test_message_error_handling(self, client, headers, test_session_id):
        """Test error handling for non-existent message IDs."""
        # Try to read non-existent message IDs
        read_response = client.post(
            f"/api/v1/sessions/{test_session_id}/messages/read",
            json={"message_ids": ["nonexistent-id-1", "nonexistent-id-2"]},
            headers=headers
        )
        
        # Should return 404 for non-existent messages
        assert read_response.status_code == 404
        
        response_data = read_response.json()
        assert response_data["status"] == "error"
        assert response_data["code"] == 404

    @pytest.mark.asyncio
    async def test_buffer_service_integration(self, test_session_id):
        """Test BufferService behavior directly."""
        # Reset service instances to ensure clean state
        ServiceFactory.reset()
        
        # Get BufferService instance
        buffer_service = await ServiceFactory.get_buffer_service_for_user("test-user")
        assert buffer_service is not None
        
        # Add messages
        messages = [
            {"role": "user", "content": "Direct buffer test"},
            {"role": "assistant", "content": "Direct buffer response"}
        ]
        
        result = await buffer_service.add(messages, session_id=test_session_id)
        assert result["status"] == "success"
        message_ids = result["data"]["message_ids"]
        
        # Read messages back
        read_result = await buffer_service.read(message_ids)
        assert read_result["status"] == "success"
        
        read_messages = read_result["data"]["messages"]
        assert len(read_messages) == 2
        assert any(msg["content"] == "Direct buffer test" for msg in read_messages) 