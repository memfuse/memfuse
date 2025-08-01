"""
Integration tests for Agents API.

These tests validate that Agents API operations actually persist data 
to the database and work correctly with real database operations.
"""

import pytest
from typing import Dict, Any


class TestAgentsAPIIntegration:
    """Integration tests for Agents API endpoints."""

    def test_create_agent(self, client, headers: Dict[str, str],
                                    test_agent_data: Dict[str, Any], database_connection,
                                    integration_helper, mock_embedding_service):
        """Test that creating an agent works correctly through API."""
        # Create agent via API
        response = client.post("/api/v1/agents", json=test_agent_data, headers=headers)
        
        # Verify API response
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "agent" in response_data["data"]
        
        agent_data = response_data["data"]["agent"]
        agent_id = agent_data["id"]
        
        # Verify agent data in response
        assert agent_data["name"] == test_agent_data["name"]
        assert agent_data["description"] == test_agent_data["description"]
        assert "created_at" in agent_data
        assert "updated_at" in agent_data
        
        # Verify agent actually exists in database
        assert integration_helper.verify_database_record_exists(
            database_connection, "agents", agent_id
        )
        
        # Verify database record details
        with database_connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, name, description, created_at, updated_at FROM agents WHERE id = %s",
                (agent_id,)
            )
            db_record = cursor.fetchone()

        assert db_record is not None
        assert db_record["id"] == agent_id
        assert db_record["name"] == test_agent_data["name"]
        assert db_record["description"] == test_agent_data["description"]
        assert db_record["created_at"] is not None  # created_at
        assert db_record["updated_at"] is not None  # updated_at

    def test_get_agent_from_database(self, client, headers: Dict[str, str],
                                   test_agent_data: Dict[str, Any], integration_helper,
                                   mock_embedding_service):
        """Test that getting an agent retrieves actual database data."""
        # Create agent first
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        agent_id = agent["id"]
        
        # Get agent via API
        response = client.get(f"/api/v1/agents/{agent_id}", headers=headers)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        retrieved_agent = response_data["data"]["agent"]
        assert retrieved_agent["id"] == agent_id
        assert retrieved_agent["name"] == test_agent_data["name"]
        assert retrieved_agent["description"] == test_agent_data["description"]

    def test_update_agent(self, client, headers: Dict[str, str],
                                    test_agent_data: Dict[str, Any], database_connection,
                                    integration_helper, mock_embedding_service):
        """Test that updating an agent works correctly through API."""
        # Create agent first
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        agent_id = agent["id"]
        
        # Update agent data with unique name to avoid conflicts
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        updated_data = {
            "name": f"updated_integration_test_agent_{unique_suffix}",
            "description": "Updated description for integration testing"
        }
        
        # Update via API
        response = client.put(f"/api/v1/agents/{agent_id}", json=updated_data, headers=headers)
        
        # Verify API response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        updated_agent = response_data["data"]["agent"]
        assert updated_agent["name"] == updated_data["name"]
        assert updated_agent["description"] == updated_data["description"]
        
        # Verify database record was actually updated
        cursor = database_connection.execute(
            "SELECT name, description, updated_at FROM agents WHERE id = %s",
            (agent_id,)
        )
        db_record = cursor.fetchone()
        cursor.close()

        assert db_record is not None
        assert db_record["name"] == updated_data["name"]
        assert db_record["description"] == updated_data["description"]
        assert db_record["updated_at"] is not None  # updated_at should be set

    def test_delete_agent(self, client, headers: Dict[str, str],
                                    test_agent_data: Dict[str, Any], database_connection,
                                    integration_helper, mock_embedding_service):
        """Test that deleting an agent works correctly through API."""
        # Create agent first
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        agent_id = agent["id"]
        
        # Verify agent exists in database
        assert integration_helper.verify_database_record_exists(
            database_connection, "agents", agent_id
        )
        
        # Delete via API
        response = client.delete(f"/api/v1/agents/{agent_id}", headers=headers)
        
        # Verify API response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["data"]["agent_id"] == agent_id
        
        # Verify agent no longer exists in database
        assert not integration_helper.verify_database_record_exists(
            database_connection, "agents", agent_id
        )

    def test_delete_agent_with_error_handling(self, client, headers: Dict[str, str],
                                           test_agent_data: Dict[str, Any], database_connection,
                                           integration_helper, mock_embedding_service):
        """Test that agent deletion properly handles errors with correct HTTP status codes."""
        # Create agent first
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        agent_id = agent["id"]
        
        # Delete the agent successfully
        response = client.delete(f"/api/v1/agents/{agent_id}", headers=headers)
        assert response.status_code == 200
        
        # Try to delete the same agent again (should fail with 404)
        response = client.delete(f"/api/v1/agents/{agent_id}", headers=headers)
        assert response.status_code == 404
        
        # Should have proper error response
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "not found" in response_data["message"].lower()
        
        # Try to delete a non-existent agent (should fail with 404)
        fake_agent_id = "00000000-0000-0000-0000-000000000000"
        response = client.delete(f"/api/v1/agents/{fake_agent_id}", headers=headers)
        assert response.status_code == 404

    def test_list_agents_from_database(self, client, headers: Dict[str, str],
                                     test_agent_data: Dict[str, Any], database_connection,
                                     integration_helper, mock_embedding_service):
        """Test that listing agents returns actual database records."""
        # Create multiple agents with unique names
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        
        agent1_data = {**test_agent_data, "name": f"integration_test_agent_1_{unique_suffix}"}
        agent2_data = {**test_agent_data, "name": f"integration_test_agent_2_{unique_suffix}"}
        
        agent1 = integration_helper.create_agent_via_api(client, headers, agent1_data)
        agent2 = integration_helper.create_agent_via_api(client, headers, agent2_data)
        
        # Get agents list via API
        response = client.get("/api/v1/agents", headers=headers)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        agents_list = response_data["data"]["agents"]
        assert len(agents_list) >= 2  # At least our two test agents
        
        # Verify our test agents are in the list
        agent_ids = {agent["id"] for agent in agents_list}
        assert agent1["id"] in agent_ids
        assert agent2["id"] in agent_ids

    def test_agent_name_uniqueness_database_constraint(self, client,
                                                     headers: Dict[str, str],
                                                     test_agent_data: Dict[str, Any],
                                                     integration_helper,
                                                     mock_embedding_service):
        """Test that duplicate agent names are handled by database constraints."""
        # Create first agent
        agent1 = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Try to create second agent with same name
        response = client.post("/api/v1/agents", json=test_agent_data, headers=headers)
        
        # Verify duplicate name is rejected
        assert response.status_code == 400
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "already exists" in response_data["message"].lower()

    def test_agent_cascade_deletion_with_sessions(self, client,
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
        assert response.status_code == 200
        
        # Verify agent is deleted
        assert not integration_helper.verify_database_record_exists(
            database_connection, "agents", agent["id"]
        )
        
        # Verify session is also deleted (cascade)
        assert not integration_helper.verify_database_record_exists(
            database_connection, "sessions", session["id"]
        )

    def test_get_agent_by_name_from_database(self, client, headers: Dict[str, str],
                                           test_agent_data: Dict[str, Any], integration_helper,
                                           mock_embedding_service):
        """Test that getting an agent by name retrieves actual database data."""
        # Create agent first
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Get agent by name via API
        response = client.get(f"/api/v1/agents?name={test_agent_data['name']}", headers=headers)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        agents_list = response_data["data"]["agents"]
        assert len(agents_list) == 1
        
        retrieved_agent = agents_list[0]
        assert retrieved_agent["id"] == agent["id"]
        assert retrieved_agent["name"] == test_agent_data["name"]
        assert retrieved_agent["description"] == test_agent_data["description"]

    def test_get_agent_by_name_not_found_from_database(self, client, headers: Dict[str, str],
                                                      integration_helper, mock_embedding_service):
        """Test that getting a non-existent agent by name returns 404."""
        # Try to get agent by non-existent name
        response = client.get("/api/v1/agents?name=nonexistent-agent", headers=headers)
        
        # Verify response
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "not found" in response_data["message"].lower()

    def test_database_transaction_rollback_on_error(self, client,
                                                   headers: Dict[str, str],
                                                   database_connection,
                                                   integration_helper,
                                                   mock_embedding_service):
        """Test that database transactions are properly rolled back on errors."""
        # Get initial agent count
        initial_count = integration_helper.verify_database_record_count(
            database_connection, "agents", 0
        )
        
        # Try to create agent with invalid data that should cause rollback
        invalid_agent_data = {
            "name": "",  # Invalid empty name
            "description": "This should not be created"
        }
        
        response = client.post("/api/v1/agents", json=invalid_agent_data, headers=headers)
        
        # Verify request failed
        assert response.status_code == 422  # FastAPI validation error
        
        # Verify no partial data was committed to database
        final_count = integration_helper.verify_database_record_count(
            database_connection, "agents", 0
        )
        
        # Count should be the same (no partial commits)
        assert final_count == initial_count 