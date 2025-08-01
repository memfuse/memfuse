"""
Integration tests for Users API.

These tests validate that Users API operations actually persist data 
to the database and work correctly with real database operations.
"""

import pytest
from typing import Dict, Any


class TestUsersAPIIntegration:
    """Integration tests for Users API endpoints."""

    def test_create_user_persistence(self, client, headers: Dict[str, str],
                                   test_user_data: Dict[str, Any], database_connection,
                                   integration_helper, mock_embedding_service):
        """Test that creating a user actually persists to database."""
        # Create user via API
        response = client.post("/api/v1/users", json=test_user_data, headers=headers)

        # Verify API response with enhanced error handling
        response_data = integration_helper.validate_api_response(response, 201)
        assert "user" in response_data["data"], f"Missing 'user' in response data: {response_data}"

        user_data = response_data["data"]["user"]
        user_id = user_data["id"]
        
        # Verify user data in response
        assert user_data["name"] == test_user_data["name"]
        assert user_data["description"] == test_user_data["description"]
        assert "created_at" in user_data
        assert "updated_at" in user_data
        
        # Verify user actually exists in database
        assert integration_helper.verify_database_record_exists(
            database_connection, "users", user_id
        )
        
        # Verify database record details
        with database_connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, name, description, created_at, updated_at FROM users WHERE id = %s",
                (user_id,)
            )
            db_record = cursor.fetchone()
        
        assert db_record is not None
        assert db_record["id"] == user_id
        assert db_record["name"] == test_user_data["name"]
        assert db_record["description"] == test_user_data["description"]
        assert db_record["created_at"] is not None  # created_at
        assert db_record["updated_at"] is not None  # updated_at



    def test_get_user_from_database(self, client, headers: Dict[str, str],
                                   test_user_data: Dict[str, Any], integration_helper,
                                   mock_embedding_service):
        """Test that getting a user retrieves actual database data."""
        # Create user first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        user_id = user["id"]
        
        # Get user via API
        response = client.get(f"/api/v1/users/{user_id}", headers=headers)

        # Verify response with enhanced error handling
        response_data = integration_helper.validate_api_response(response, 200)
        assert "user" in response_data["data"], f"Missing 'user' in response data: {response_data}"

        retrieved_user = response_data["data"]["user"]
        assert retrieved_user["id"] == user_id
        assert retrieved_user["name"] == test_user_data["name"]
        assert retrieved_user["description"] == test_user_data["description"]

    def test_update_user_persistence(self, client, headers: Dict[str, str],
                                    test_user_data: Dict[str, Any], database_connection,
                                    integration_helper, mock_embedding_service):
        """Test that updating a user actually modifies database record."""
        # Create user first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        user_id = user["id"]
        
        # Update user data with unique name to avoid conflicts
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        updated_data = {
            "name": f"updated_integration_test_user_{unique_suffix}",
            "description": "Updated description for integration testing"
        }
        
        # Update via API
        response = client.put(f"/api/v1/users/{user_id}", json=updated_data, headers=headers)

        # Verify API response with enhanced error handling
        response_data = integration_helper.validate_api_response(response, 200)
        assert "user" in response_data["data"], f"Missing 'user' in response data: {response_data}"

        updated_user = response_data["data"]["user"]
        assert updated_user["name"] == updated_data["name"]
        assert updated_user["description"] == updated_data["description"]
        
        # Verify database record was actually updated
        with database_connection.cursor() as cursor:
            cursor.execute(
                "SELECT name, description, updated_at FROM users WHERE id = %s",
                (user_id,)
            )
            db_record = cursor.fetchone()
        
        assert db_record is not None
        assert db_record["name"] == updated_data["name"]
        assert db_record["description"] == updated_data["description"]
        assert db_record["updated_at"] is not None  # updated_at should be set

    def test_delete_user_persistence(self, client, headers: Dict[str, str],
                                    test_user_data: Dict[str, Any], database_connection,
                                    integration_helper, mock_embedding_service):
        """Test that deleting a user actually removes it from database."""
        # Create user first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        user_id = user["id"]
        
        # Verify user exists in database
        assert integration_helper.verify_database_record_exists(
            database_connection, "users", user_id
        )
        
        # Delete via API
        response = client.delete(f"/api/v1/users/{user_id}", headers=headers)

        # Verify API response (204 No Content for successful deletion)
        assert response.status_code == 204
        
        # Verify user no longer exists in database
        assert not integration_helper.verify_database_record_exists(
            database_connection, "users", user_id
        )

    def test_delete_user_with_error_handling(self, client, headers: Dict[str, str],
                                           test_user_data: Dict[str, Any], database_connection,
                                           integration_helper, mock_embedding_service):
        """Test that user deletion properly handles errors with correct HTTP status codes."""
        # Create user first
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        user_id = user["id"]
        
        # Delete the user successfully
        response = client.delete(f"/api/v1/users/{user_id}", headers=headers)
        assert response.status_code == 204
        
        # Try to delete the same user again (should fail with 404)
        response = client.delete(f"/api/v1/users/{user_id}", headers=headers)
        assert response.status_code == 404
        
        # Should have proper error response
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "not found" in response_data["message"].lower()
        
        # Try to delete a non-existent user (should fail with 404)
        fake_user_id = "00000000-0000-0000-0000-000000000000"
        response = client.delete(f"/api/v1/users/{fake_user_id}", headers=headers)
        assert response.status_code == 404

    def test_list_users_from_database(self, client, headers: Dict[str, str],
                                     test_user_data: Dict[str, Any], database_connection,
                                     integration_helper, mock_embedding_service):
        """Test that listing users returns actual database records."""
        # Create multiple users with unique names
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        
        user1_data = {**test_user_data, "name": f"integration_test_user_1_{unique_suffix}"}
        user2_data = {**test_user_data, "name": f"integration_test_user_2_{unique_suffix}"}
        
        user1 = integration_helper.create_user_via_api(client, headers, user1_data)
        user2 = integration_helper.create_user_via_api(client, headers, user2_data)
        
        # Get users list via API
        response = client.get("/api/v1/users", headers=headers)

        # Verify response with enhanced error handling
        response_data = integration_helper.validate_api_response(response, 200)
        assert "users" in response_data["data"], f"Missing 'users' in response data: {response_data}"

        users_list = response_data["data"]["users"]
        assert len(users_list) >= 2  # At least our two test users
        
        # Verify our test users are in the list
        user_ids = {user["id"] for user in users_list}
        assert user1["id"] in user_ids
        assert user2["id"] in user_ids

    def test_user_name_uniqueness_database_constraint(self, client,
                                                     headers: Dict[str, str],
                                                     test_user_data: Dict[str, Any],
                                                     integration_helper,
                                                     mock_embedding_service):
        """Test that duplicate user names are handled by database constraints."""
        # Create first user
        user1 = integration_helper.create_user_via_api(client, headers, test_user_data)
        
        # Try to create second user with same name
        response = client.post("/api/v1/users", json=test_user_data, headers=headers)
        
        # Verify duplicate name is rejected
        assert response.status_code == 400
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "already exists" in response_data["message"].lower()

    def test_user_cascade_deletion_with_sessions(self, client,
                                                headers: Dict[str, str],
                                                test_user_data: Dict[str, Any],
                                                test_agent_data: Dict[str, Any],
                                                test_session_data,
                                                database_connection,
                                                integration_helper,
                                                mock_embedding_service):
        """Test that deleting a user properly cascades to related sessions."""
        # Create user and agent
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session with user
        session_data = test_session_data(user["id"], agent["id"])
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



    def test_database_transaction_rollback_on_error(self, client,
                                                   headers: Dict[str, str],
                                                   database_connection,
                                                   integration_helper,
                                                   mock_embedding_service):
        """Test that database transactions are properly rolled back on errors."""
        # Get initial user count
        initial_count = integration_helper.verify_database_record_count(
            database_connection, "users", 0
        )
        
        # Try to create user with invalid data that should cause rollback
        invalid_user_data = {
            "name": "",  # Invalid empty name
            "description": "This should not be created"
        }
        
        response = client.post("/api/v1/users", json=invalid_user_data, headers=headers)
        
        # Verify request failed
        assert response.status_code == 422  # FastAPI validation error
        
        # Verify no partial data was committed to database
        final_count = integration_helper.verify_database_record_count(
            database_connection, "users", 0
        )
        
        # Count should be the same (no partial commits)
        assert final_count == initial_count 