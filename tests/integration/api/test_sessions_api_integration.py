"""
Integration tests for Sessions API.

These tests validate that Sessions API operations actually persist data 
to the database and work correctly with real database operations.
"""

import pytest
from typing import Dict, Any


class TestSessionsAPIIntegration:
    """Integration tests for Sessions API endpoints."""

    def test_create_session(self, client, headers: Dict[str, str],
                                       test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                       database_connection, integration_helper, mock_embedding_service):
        """Test that creating a session works correctly through API."""
        # Create user and agent first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session data
        import uuid
        unique_suffix = str(uuid.uuid4())
        session_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": f"integration_test_session_{unique_suffix}"
        }
        
        # Debug: Check if session exists before creation
        print(f"DEBUG: About to create session with name: {session_data['name']}")

        # Create session via API
        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        
        # Verify API response with enhanced error handling
        assert response.status_code in [200, 201], f"Expected 200 or 201, got {response.status_code}. Response: {response.text}"

        try:
            response_data = response.json()
            if not response_data or response_data.get("status") != "success":
                assert False, f"API returned error: {response_data}"
            if not response_data.get("data") or not response_data["data"].get("session"):
                assert False, f"Missing session in response data: {response_data}"
        except Exception as e:
            assert False, f"Failed to parse response JSON: {e}. Response: {response.text}"

        session_response = response_data["data"]["session"]
        session_id = session_response["id"]
        
        # Verify session data in response
        assert session_response["user_id"] == user["id"]
        assert session_response["agent_id"] == agent["id"]
        assert session_response["name"] == session_data["name"]
        assert "created_at" in session_response
        assert "updated_at" in session_response
        
        # Verify session actually exists in database
        assert integration_helper.verify_database_record_exists(
            database_connection, "sessions", session_id
        )
        
        # Verify database record details
        cursor = database_connection.cursor()
        cursor.execute(
            "SELECT id, user_id, agent_id, name, created_at, updated_at FROM sessions WHERE id = %s",
            (session_id,)
        )
        db_record = cursor.fetchone()
        cursor.close()
        
        assert db_record is not None
        assert db_record["id"] == session_id
        assert db_record["user_id"] == user["id"]
        assert db_record["agent_id"] == agent["id"]
        assert db_record["name"] == session_data["name"]
        assert db_record["created_at"] is not None
        assert db_record["updated_at"] is not None

    def test_create_session_auto_generated_name(self, client, headers: Dict[str, str],
                                               test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                               database_connection, integration_helper, mock_embedding_service):
        """Test that creating a session without name auto-generates one."""
        # Create user and agent first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session data without name
        session_data = {
            "user_id": user["id"],
            "agent_id": agent["id"]
        }
        
        # Create session via API
        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        
        # Verify API response
        assert response.status_code in [200, 201]
        response_data = response.json()
        assert response_data["status"] == "success"
        
        session_response = response_data["data"]["session"]
        session_id = session_response["id"]
        
        # Verify name was auto-generated
        assert session_response["name"] is not None
        assert session_response["name"] != ""
        
        # Verify in database
        cursor = database_connection.cursor()
        cursor.execute("SELECT id, name FROM sessions WHERE id = %s", (session_id,))
        db_record = cursor.fetchone()
        cursor.close()

        assert db_record is not None
        assert db_record["name"] is not None
        assert db_record["name"] != ""

    def test_get_session_from_database(self, client, headers: Dict[str, str],
                                      test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                      integration_helper, mock_embedding_service):
        """Test that getting a session retrieves actual database data."""
        # Create user and agent first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session data
        import uuid
        unique_suffix = str(uuid.uuid4())
        session_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": f"integration_test_session_{unique_suffix}"
        }
        
        # Create session first
        session = integration_helper.create_session_via_api(client, headers, session_data)
        session_id = session["id"]
        
        # Get session via API
        response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        retrieved_session = response_data["data"]["session"]
        assert retrieved_session["id"] == session_id
        assert retrieved_session["user_id"] == user["id"]
        assert retrieved_session["agent_id"] == agent["id"]
        assert retrieved_session["name"] == session_data["name"]

    def test_get_session_by_name_from_database(self, client, headers: Dict[str, str],
                                             test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                             integration_helper, mock_embedding_service):
        """Test that getting a session by name retrieves actual database data."""
        # Create user and agent first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session data
        import uuid
        unique_suffix = str(uuid.uuid4())
        session_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": f"integration_test_session_{unique_suffix}"
        }
        
        # Create session first
        session = integration_helper.create_session_via_api(client, headers, session_data)
        
        # Get session by name via API
        response = client.get(f"/api/v1/sessions?name={session_data['name']}", headers=headers)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        sessions_list = response_data["data"]["sessions"]
        assert len(sessions_list) == 1
        
        retrieved_session = sessions_list[0]
        assert retrieved_session["id"] == session["id"]
        assert retrieved_session["user_id"] == user["id"]
        assert retrieved_session["agent_id"] == agent["id"]
        assert retrieved_session["name"] == session_data["name"]

    def test_update_session(self, client, headers: Dict[str, str],
                                       test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                       database_connection, integration_helper, mock_embedding_service):
        """Test that updating a session works correctly through API."""
        # Create user and agent first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session data
        import uuid
        unique_suffix = str(uuid.uuid4())
        session_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": f"integration_test_session_{unique_suffix}"
        }
        
        # Create session first
        session = integration_helper.create_session_via_api(client, headers, session_data)
        session_id = session["id"]
        
        # Update session data with unique name to avoid conflicts
        updated_unique_suffix = str(uuid.uuid4())
        updated_data = {
            "name": f"updated_integration_test_session_{updated_unique_suffix}"
        }
        
        # Update via API
        response = client.put(f"/api/v1/sessions/{session_id}", json=updated_data, headers=headers)
        
        # Verify API response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        updated_session = response_data["data"]["session"]
        assert updated_session["name"] == updated_data["name"]
        
        # Verify database record was actually updated
        cursor = database_connection.cursor()
        cursor.execute(
            "SELECT name, updated_at FROM sessions WHERE id = %s",
            (session_id,)
        )
        db_record = cursor.fetchone()
        cursor.close()
        
        assert db_record is not None
        assert db_record["name"] == updated_data["name"]
        assert db_record["updated_at"] is not None  # updated_at should be set

    def test_delete_session(self, client, headers: Dict[str, str],
                                       test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                       database_connection, integration_helper, mock_embedding_service):
        """Test that deleting a session works correctly through API."""
        # Create user and agent first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session data
        import uuid
        unique_suffix = str(uuid.uuid4())
        session_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": f"integration_test_session_{unique_suffix}"
        }
        
        # Create session first
        session = integration_helper.create_session_via_api(client, headers, session_data)
        session_id = session["id"]
        
        # Verify session exists in database
        assert integration_helper.verify_database_record_exists(
            database_connection, "sessions", session_id
        )
        
        # Delete via API
        response = client.delete(f"/api/v1/sessions/{session_id}", headers=headers)
        
        # Verify API response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["data"]["session_id"] == session_id
        
        # Verify session no longer exists in database
        assert not integration_helper.verify_database_record_exists(
            database_connection, "sessions", session_id
        )

    def test_delete_session_with_error_handling(self, client, headers: Dict[str, str],
                                               test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                               integration_helper, mock_embedding_service):
        """Test that session deletion properly handles errors with correct HTTP status codes."""
        # Create user and agent first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session data
        import uuid
        unique_suffix = str(uuid.uuid4())
        session_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": f"integration_test_session_{unique_suffix}"
        }
        
        # Create session first
        session = integration_helper.create_session_via_api(client, headers, session_data)
        session_id = session["id"]
        
        # Delete the session successfully
        response = client.delete(f"/api/v1/sessions/{session_id}", headers=headers)
        assert response.status_code == 200
        
        # Try to delete the same session again (should fail with 404)
        response = client.delete(f"/api/v1/sessions/{session_id}", headers=headers)
        assert response.status_code == 404
        
        # Should have proper error response
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "not found" in response_data["message"].lower()
        
        # Try to delete a non-existent session (should fail with 404)
        fake_session_id = "00000000-0000-0000-0000-000000000000"
        response = client.delete(f"/api/v1/sessions/{fake_session_id}", headers=headers)
        assert response.status_code == 404

    def test_list_sessions_from_database(self, client, headers: Dict[str, str],
                                        test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                        integration_helper, mock_embedding_service):
        """Test that listing sessions returns actual database records."""
        # Create user and agent first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create multiple sessions with unique names
        import uuid
        unique_suffix = str(uuid.uuid4())
        
        session1_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": f"integration_test_session_1_{unique_suffix}"
        }
        session2_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": f"integration_test_session_2_{unique_suffix}"
        }
        
        session1 = integration_helper.create_session_via_api(client, headers, session1_data)
        session2 = integration_helper.create_session_via_api(client, headers, session2_data)
        
        # Get sessions list via API
        response = client.get("/api/v1/sessions", headers=headers)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        sessions_list = response_data["data"]["sessions"]
        assert len(sessions_list) >= 2  # At least our two test sessions
        
        # Verify our test sessions are in the list
        session_ids = {session["id"] for session in sessions_list}
        assert session1["id"] in session_ids
        assert session2["id"] in session_ids

    def test_list_sessions_with_user_filter(self, client, headers: Dict[str, str],
                                           test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                           integration_helper, mock_embedding_service):
        """Test that listing sessions with user_id filter returns only sessions for that user."""
        # Create two users and one agent
        import uuid
        unique_suffix = str(uuid.uuid4())
        
        user1_data = {**test_user_data, "name": f"integration_test_user_1_{unique_suffix}"}
        user2_data = {**test_user_data, "name": f"integration_test_user_2_{unique_suffix}"}
        
        user1 = integration_helper.create_user_via_api(client, headers, user1_data)
        user2 = integration_helper.create_user_via_api(client, headers, user2_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create sessions for both users
        session1_data = {
            "user_id": user1["id"],
            "agent_id": agent["id"],
            "name": f"user1_session_{unique_suffix}"
        }
        session2_data = {
            "user_id": user2["id"],
            "agent_id": agent["id"],
            "name": f"user2_session_{unique_suffix}"
        }
        
        session1 = integration_helper.create_session_via_api(client, headers, session1_data)
        session2 = integration_helper.create_session_via_api(client, headers, session2_data)
        
        # Get sessions for user1 only
        response = client.get(f"/api/v1/sessions?user_id={user1['id']}", headers=headers)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        sessions_list = response_data["data"]["sessions"]
        
        # All sessions should belong to user1
        for session in sessions_list:
            assert session["user_id"] == user1["id"]
        
        # Should include our test session for user1
        session_ids = {session["id"] for session in sessions_list}
        assert session1["id"] in session_ids

    def test_list_sessions_with_agent_filter(self, client, headers: Dict[str, str],
                                            test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                            integration_helper, mock_embedding_service):
        """Test that listing sessions with agent_id filter returns only sessions for that agent."""
        # Create one user and two agents
        import uuid
        unique_suffix = str(uuid.uuid4())
        
        agent1_data = {**test_agent_data, "name": f"integration_test_agent_1_{unique_suffix}"}
        agent2_data = {**test_agent_data, "name": f"integration_test_agent_2_{unique_suffix}"}
        
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent1 = integration_helper.create_agent_via_api(client, headers, agent1_data)
        agent2 = integration_helper.create_agent_via_api(client, headers, agent2_data)
        
        # Create sessions for both agents
        session1_data = {
            "user_id": user["id"],
            "agent_id": agent1["id"],
            "name": f"agent1_session_{unique_suffix}"
        }
        session2_data = {
            "user_id": user["id"],
            "agent_id": agent2["id"],
            "name": f"agent2_session_{unique_suffix}"
        }
        
        session1 = integration_helper.create_session_via_api(client, headers, session1_data)
        session2 = integration_helper.create_session_via_api(client, headers, session2_data)
        
        # Get sessions for agent1 only
        response = client.get(f"/api/v1/sessions?agent_id={agent1['id']}", headers=headers)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        sessions_list = response_data["data"]["sessions"]
        
        # All sessions should belong to agent1
        for session in sessions_list:
            assert session["agent_id"] == agent1["id"]
        
        # Should include our test session for agent1
        session_ids = {session["id"] for session in sessions_list}
        assert session1["id"] in session_ids

    def test_create_session_with_invalid_user_id(self, client, headers: Dict[str, str],
                                                 test_agent_data: Dict[str, Any], integration_helper,
                                                 mock_embedding_service):
        """Test that creating a session with invalid user_id returns 404."""
        # Create agent first
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Try to create session with invalid user_id
        session_data = {
            "user_id": "invalid-user-id",
            "agent_id": agent["id"],
            "name": "test_session"
        }
        
        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        
        # Verify error response
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "User with ID 'invalid-user-id' not found" in response_data["message"]

    def test_create_session_with_invalid_agent_id(self, client, headers: Dict[str, str],
                                                  test_user_data: Dict[str, Any], integration_helper,
                                                  mock_embedding_service):
        """Test that creating a session with invalid agent_id returns 404."""
        # Create user first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        
        # Try to create session with invalid agent_id
        session_data = {
            "user_id": user["id"],
            "agent_id": "invalid-agent-id",
            "name": "test_session"
        }
        
        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        
        # Verify error response
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "Agent with ID 'invalid-agent-id' not found" in response_data["message"]

    def test_user_cascade_deletion_affects_sessions(self, client, headers: Dict[str, str],
                                                   test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                                   database_connection, integration_helper, mock_embedding_service):
        """Test that deleting a user properly cascades to related sessions."""
        # Create user and agent
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session with user
        import uuid
        unique_suffix = str(uuid.uuid4())
        session_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": f"integration_test_session_{unique_suffix}"
        }
        
        session = integration_helper.create_session_via_api(client, headers, session_data)
        
        # Verify session exists
        assert integration_helper.verify_database_record_exists(
            database_connection, "sessions", session["id"]
        )
        
        # Delete user
        response = client.delete(f"/api/v1/users/{user['id']}", headers=headers)
        assert response.status_code == 204
        
        # Verify user is deleted
        assert not integration_helper.verify_database_record_exists(
            database_connection, "users", user["id"]
        )
        
        # Verify session is also deleted (cascade)
        assert not integration_helper.verify_database_record_exists(
            database_connection, "sessions", session["id"]
        )

    def test_agent_cascade_deletion_affects_sessions(self, client, headers: Dict[str, str],
                                                    test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                                    database_connection, integration_helper, mock_embedding_service):
        """Test that deleting an agent properly cascades to related sessions."""
        # Create user and agent
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session with agent
        import uuid
        unique_suffix = str(uuid.uuid4())
        session_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": f"integration_test_session_{unique_suffix}"
        }
        
        session = integration_helper.create_session_via_api(client, headers, session_data)
        
        # Verify session exists
        assert integration_helper.verify_database_record_exists(
            database_connection, "sessions", session["id"]
        )
        
        # Delete agent
        response = client.delete(f"/api/v1/agents/{agent['id']}", headers=headers)
        assert response.status_code == 200
        
        # Verify agent is deleted
        assert not integration_helper.verify_database_record_exists(
            database_connection, "agents", agent["id"]
        )
        
        # Verify session is also deleted (cascade)
        assert not integration_helper.verify_database_record_exists(
            database_connection, "sessions", session["id"]
        )

    def test_database_transaction_rollback_on_error(self, client, headers: Dict[str, str],
                                                   test_user_data: Dict[str, Any], test_agent_data: Dict[str, Any],
                                                   database_connection, integration_helper, mock_embedding_service):
        """Test that database transactions are properly rolled back on errors."""
        # Create user and agent first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Get initial session count
        initial_count = integration_helper.verify_database_record_count(
            database_connection, "sessions", 0
        )
        
        # Try to create session with invalid data that should cause rollback
        invalid_session_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": ""  # Invalid empty name
        }
        
        response = client.post("/api/v1/sessions", json=invalid_session_data, headers=headers)
        
        # Verify request failed
        assert response.status_code == 422  # FastAPI validation error
        
        # Verify no partial data was committed to database
        final_count = integration_helper.verify_database_record_count(
            database_connection, "sessions", 0
        )
        
        # Count should be the same (no partial commits)
        assert final_count == initial_count 