"""
Integration tests for Sessions API.

These tests validate that Sessions API operations actually persist data 
to the database and work correctly with real database operations.
"""

import pytest
from typing import Dict, Any
from fastapi.testclient import TestClient


class TestSessionsAPIIntegration:
    """Integration tests for Sessions API endpoints."""

    def test_create_session_persistence(self, client: TestClient, headers: Dict[str, str],
                                       test_user_data: Dict[str, Any], 
                                       test_agent_data: Dict[str, Any],
                                       test_session_data, database_connection,
                                       integration_helper, mock_embedding_service):
        """Test that creating a session actually persists to database."""
        # Create user and agent first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session
        session_data = test_session_data(user["id"], agent["id"])
        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        
        # Verify API response
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "session" in response_data["data"]
        
        session = response_data["data"]["session"]
        session_id = session["id"]
        
        # Verify session actually exists in database
        assert integration_helper.verify_database_record_exists(
            database_connection, "sessions", session_id
        )
        
        # Verify foreign key relationships
        cursor = database_connection.connection.cursor()
        cursor.execute(
            "SELECT user_id, agent_id FROM sessions WHERE id = %s",
            (session_id,)
        )
        db_record = cursor.fetchone()
        cursor.close()
        
        assert db_record is not None
        assert db_record[0] == user["id"]
        assert db_record[1] == agent["id"]

    def test_session_isolation_between_users(self, client: TestClient, headers: Dict[str, str],
                                           test_user_data: Dict[str, Any], 
                                           test_agent_data: Dict[str, Any],
                                           test_session_data, integration_helper,
                                           mock_embedding_service):
        """Test that sessions are properly isolated between users."""
        # Create two users and one agent
        user1_data = {**test_user_data, "name": "integration_user_1"}
        user2_data = {**test_user_data, "name": "integration_user_2"}
        
        user1 = integration_helper.create_user_via_api(client, headers, user1_data)
        user2 = integration_helper.create_user_via_api(client, headers, user2_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create sessions for each user
        session1_data = test_session_data(user1["id"], agent["id"], "_user1")
        session2_data = test_session_data(user2["id"], agent["id"], "_user2")
        
        session1 = integration_helper.create_session_via_api(client, headers, session1_data)
        session2 = integration_helper.create_session_via_api(client, headers, session2_data)
        
        # Get sessions list - should show all sessions
        response = client.get("/api/v1/sessions", headers=headers)
        assert response.status_code == 200
        
        all_sessions = response.json()["data"]["sessions"]
        session_ids = {s["id"] for s in all_sessions}
        
        # Both sessions should exist
        assert session1["id"] in session_ids
        assert session2["id"] in session_ids
        
        # Verify sessions have correct user associations
        session1_from_list = next(s for s in all_sessions if s["id"] == session1["id"])
        session2_from_list = next(s for s in all_sessions if s["id"] == session2["id"])
        
        assert session1_from_list["user_id"] == user1["id"]
        assert session2_from_list["user_id"] == user2["id"] 