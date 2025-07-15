"""
Integration tests for Agents API.

These tests validate that Agents API operations actually persist data 
to the database and work correctly with real database operations.
"""

import pytest
from typing import Dict, Any
from fastapi.testclient import TestClient


class TestAgentsAPIIntegration:
    """Integration tests for Agents API endpoints."""

    def test_create_agent_persistence(self, client: TestClient, headers: Dict[str, str],
                                     test_agent_data: Dict[str, Any], database_connection,
                                     integration_helper, mock_embedding_service):
        """Test that creating an agent actually persists to database."""
        # Create agent via API
        response = client.post("/api/v1/agents", json=test_agent_data, headers=headers)
        
        # Verify API response
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "agent" in response_data["data"]
        
        agent_data = response_data["data"]["agent"]
        agent_id = agent_data["id"]
        
        # Verify agent actually exists in database
        assert integration_helper.verify_database_record_exists(
            database_connection, "agents", agent_id
        )

    def test_agent_cascade_deletion_with_sessions(self, client: TestClient,
                                                 headers: Dict[str, str],
                                                 test_user_data: Dict[str, Any],
                                                 test_agent_data: Dict[str, Any],
                                                 test_session_data,
                                                 database_connection,
                                                 integration_helper,
                                                 mock_embedding_service):
        """Test that deleting an agent properly cascades to related sessions."""
        # Create user and agent
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session with agent
        session_data = test_session_data(user["id"], agent["id"])
        session = integration_helper.create_session_via_api(client, headers, session_data)
        
        # Verify session exists
        assert integration_helper.verify_database_record_exists(
            database_connection, "sessions", session["id"]
        )
        
        # Delete agent
        response = client.delete(f"/api/v1/agents/{agent['id']}", headers=headers)
        assert response.status_code == 204
        
        # Verify agent is deleted
        assert not integration_helper.verify_database_record_exists(
            database_connection, "agents", agent["id"]
        )
        
        # Verify session is also deleted (cascade)
        assert not integration_helper.verify_database_record_exists(
            database_connection, "sessions", session["id"]
        ) 