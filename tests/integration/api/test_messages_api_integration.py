"""
Integration tests for Messages API.

These tests validate that Messages API operations actually persist data 
to the database and work correctly with real database operations.
"""

import pytest
from typing import Dict, Any, List
import json


class TestMessagesAPIIntegration:
    """Integration tests for Messages API endpoints."""

    @pytest.fixture
    def test_session_setup(self, client, headers: Dict[str, str], test_user_data: Dict[str, Any],
                          test_agent_data: Dict[str, Any], integration_helper, mock_embedding_service):
        """Set up a test session for message operations."""
        # Create user and agent
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create session
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        session_data = {
            "user_id": user["id"],
            "agent_id": agent["id"],
            "name": f"integration_test_session_{unique_suffix}"
        }
        
        session = integration_helper.create_session_via_api(client, headers, session_data)
        
        return {
            "user": user,
            "agent": agent,
            "session": session
        }

    def test_add_messages_persistence(self, client, headers: Dict[str, str], test_session_setup,
                                     database_connection, integration_helper, mock_embedding_service):
        """Test that adding messages actually persists to database."""
        session = test_session_setup["session"]
        session_id = session["id"]
        
        # Add messages via API
        message_data = {
            "messages": [
                {"role": "user", "content": "Hello, this is a test message for integration testing."},
                {"role": "assistant", "content": "Hello! I'm responding to your test message."}
            ]
        }
        
        response = client.post(f"/api/v1/sessions/{session_id}/messages", json=message_data, headers=headers)
        
        # Verify API response
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "message_ids" in response_data["data"]
        
        message_ids = response_data["data"]["message_ids"]
        assert len(message_ids) == 2
        
        # Verify messages actually exist in database
        # Note: Messages are stored with round_id, not session_id directly
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT m.id, m.round_id, m.role, m.content, m.created_at, m.updated_at, r.session_id "
            "FROM messages m "
            "JOIN rounds r ON m.round_id = r.id "
            "WHERE m.id = ANY(%s) ORDER BY m.created_at",
            (message_ids,)
        )
        db_records = cursor.fetchall()
        cursor.close()
        
        assert len(db_records) == 2
        
        # Verify first message (user)
        user_message = db_records[0]
        assert user_message[0] == message_ids[0]  # message id
        assert user_message[1] is not None  # round_id exists
        assert user_message[2] == "user"  # role
        assert user_message[3] == "Hello, this is a test message for integration testing."  # content
        assert user_message[4] is not None  # created_at
        assert user_message[5] is not None  # updated_at
        assert user_message[6] == session_id  # session_id from joined round table
        
        # Verify second message (assistant)
        assistant_message = db_records[1]
        assert assistant_message[0] == message_ids[1]  # message id
        assert assistant_message[1] is not None  # round_id exists
        assert assistant_message[2] == "assistant"  # role
        assert assistant_message[3] == "Hello! I'm responding to your test message."  # content
        assert assistant_message[4] is not None  # created_at
        assert assistant_message[5] is not None  # updated_at
        assert assistant_message[6] == session_id  # session_id from joined round table

    def test_list_messages_from_database(self, client, headers: Dict[str, str], test_session_setup,
                                        database_connection, integration_helper, mock_embedding_service):
        """Test that listing messages retrieves actual database data."""
        session = test_session_setup["session"]
        session_id = session["id"]
        
        # Add messages via API first
        message_data = {
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second message"}
            ]
        }
        
        add_response = client.post(f"/api/v1/sessions/{session_id}/messages", json=message_data, headers=headers)
        assert add_response.status_code == 201
        
        # List messages via API
        response = client.get(f"/api/v1/sessions/{session_id}/messages", headers=headers)
        
        # Verify API response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["message"] == "Messages retrieved successfully"
        
        messages = response_data["data"]["messages"]
        assert len(messages) == 3
        
        # Verify message order (should be chronological)
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "First message"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "First response"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Second message"
        
        # Verify all messages belong to the session
        for message in messages:
            assert message["session_id"] == session_id
            assert "id" in message
            assert "created_at" in message
            assert "updated_at" in message

    def test_list_messages_with_limit(self, client, headers: Dict[str, str], test_session_setup,
                                     integration_helper, mock_embedding_service):
        """Test that listing messages with limit works correctly."""
        session = test_session_setup["session"]
        session_id = session["id"]
        
        # Add multiple messages
        message_data = {
            "messages": [
                {"role": "user", "content": f"Message {i}"}
                for i in range(5)
            ]
        }
        
        add_response = client.post(f"/api/v1/sessions/{session_id}/messages", json=message_data, headers=headers)
        assert add_response.status_code == 201
        
        # List messages with limit
        response = client.get(f"/api/v1/sessions/{session_id}/messages?limit=3", headers=headers)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        messages = response_data["data"]["messages"]
        assert len(messages) == 3
        
        # Verify messages are in correct order
        assert messages[0]["content"] == "Message 0"
        assert messages[1]["content"] == "Message 1"
        assert messages[2]["content"] == "Message 2"

    def test_read_specific_messages(self, client, headers: Dict[str, str], test_session_setup,
                                   database_connection, integration_helper, mock_embedding_service):
        """Test reading specific messages by ID."""
        session = test_session_setup["session"]
        session_id = session["id"]
        
        # Add messages via API
        message_data = {
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second message"}
            ]
        }
        
        add_response = client.post(f"/api/v1/sessions/{session_id}/messages", json=message_data, headers=headers)
        assert add_response.status_code == 201
        message_ids = add_response.json()["data"]["message_ids"]
        
        # Read specific messages
        read_data = {
            "message_ids": [message_ids[0], message_ids[2]]  # First and third messages
        }
        
        response = client.post(f"/api/v1/sessions/{session_id}/messages/read", json=read_data, headers=headers)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["message"] == "Messages read successfully"
        
        messages = response_data["data"]["messages"]
        assert len(messages) == 2
        
        # Verify correct messages were returned
        message_contents = {msg["content"] for msg in messages}
        assert "First message" in message_contents
        assert "Second message" in message_contents
        assert "First response" not in message_contents

    def test_update_messages_persistence(self, client, headers: Dict[str, str], test_session_setup,
                                        database_connection, integration_helper, mock_embedding_service):
        """Test that updating messages actually modifies database records."""
        session = test_session_setup["session"]
        session_id = session["id"]
        
        # Add messages via API
        message_data = {
            "messages": [
                {"role": "user", "content": "Original message"}
            ]
        }
        
        add_response = client.post(f"/api/v1/sessions/{session_id}/messages", json=message_data, headers=headers)
        assert add_response.status_code == 201
        message_ids = add_response.json()["data"]["message_ids"]
        
        # Update the message
        update_data = {
            "message_ids": message_ids,
            "new_messages": [
                {"role": "user", "content": "Updated message content"}
            ]
        }
        
        response = client.put(f"/api/v1/sessions/{session_id}/messages", json=update_data, headers=headers)
        
        # Verify API response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["message"] == "Messages updated successfully"
        
        # Verify database was actually updated
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT content, updated_at FROM messages WHERE id = %s",
            (message_ids[0],)
        )
        db_record = cursor.fetchone()
        cursor.close()
        
        assert db_record is not None
        assert db_record[0] == "Updated message content"
        assert db_record[1] is not None  # updated_at should exist

    def test_delete_messages_persistence(self, client, headers: Dict[str, str], test_session_setup,
                                        database_connection, integration_helper, mock_embedding_service):
        """Test that deleting messages actually removes records from database."""
        session = test_session_setup["session"]
        session_id = session["id"]
        
        # Add messages via API
        message_data = {
            "messages": [
                {"role": "user", "content": "Message to delete"},
                {"role": "assistant", "content": "Message to keep"}
            ]
        }
        
        add_response = client.post(f"/api/v1/sessions/{session_id}/messages", json=message_data, headers=headers)
        assert add_response.status_code == 201
        message_ids = add_response.json()["data"]["message_ids"]
        
        # Delete the first message
        delete_data = {
            "message_ids": [message_ids[0]]
        }
        
        response = client.request(
            "DELETE",
            f"/api/v1/sessions/{session_id}/messages",
            content=json.dumps(delete_data),
            headers={**headers, "Content-Type": "application/json"}
        )
        
        # Verify API response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["message"] == "Messages deleted successfully"
        
        # Verify message was actually deleted from database
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM messages WHERE id = %s",
            (message_ids[0],)
        )
        deleted_count = cursor.fetchone()[0]
        
        cursor.execute(
            "SELECT COUNT(*) FROM messages WHERE id = %s",
            (message_ids[1],)
        )
        remaining_count = cursor.fetchone()[0]
        cursor.close()
        
        assert deleted_count == 0  # First message should be deleted
        assert remaining_count == 1  # Second message should remain

    def test_message_session_isolation(self, client, headers: Dict[str, str], test_user_data: Dict[str, Any],
                                      test_agent_data: Dict[str, Any], integration_helper, mock_embedding_service):
        """Test that messages are properly isolated by session."""
        # Create two separate sessions
        import uuid
        
        # Create users and agents
        user1 = integration_helper.create_user_via_api(client, headers, {
            **test_user_data,
            "name": f"integration_test_user_1_{str(uuid.uuid4())[:8]}"
        })
        user2 = integration_helper.create_user_via_api(client, headers, {
            **test_user_data,
            "name": f"integration_test_user_2_{str(uuid.uuid4())[:8]}"
        })
        
        agent1 = integration_helper.create_agent_via_api(client, headers, {
            **test_agent_data,
            "name": f"integration_test_agent_1_{str(uuid.uuid4())[:8]}"
        })
        agent2 = integration_helper.create_agent_via_api(client, headers, {
            **test_agent_data,
            "name": f"integration_test_agent_2_{str(uuid.uuid4())[:8]}"
        })
        
        # Create sessions
        session1 = integration_helper.create_session_via_api(client, headers, {
            "user_id": user1["id"],
            "agent_id": agent1["id"],
            "name": f"session_1_{str(uuid.uuid4())[:8]}"
        })
        session2 = integration_helper.create_session_via_api(client, headers, {
            "user_id": user2["id"],
            "agent_id": agent2["id"],
            "name": f"session_2_{str(uuid.uuid4())[:8]}"
        })
        
        # Add messages to session1
        message_data1 = {
            "messages": [
                {"role": "user", "content": "Session 1 message"}
            ]
        }
        
        response1 = client.post(f"/api/v1/sessions/{session1['id']}/messages", json=message_data1, headers=headers)
        assert response1.status_code == 201
        
        # Add messages to session2
        message_data2 = {
            "messages": [
                {"role": "user", "content": "Session 2 message"}
            ]
        }
        
        response2 = client.post(f"/api/v1/sessions/{session2['id']}/messages", json=message_data2, headers=headers)
        assert response2.status_code == 201
        
        # Get messages from session1
        session1_messages = client.get(f"/api/v1/sessions/{session1['id']}/messages", headers=headers)
        assert session1_messages.status_code == 200
        session1_data = session1_messages.json()["data"]["messages"]
        
        # Get messages from session2
        session2_messages = client.get(f"/api/v1/sessions/{session2['id']}/messages", headers=headers)
        assert session2_messages.status_code == 200
        session2_data = session2_messages.json()["data"]["messages"]
        
        # Verify isolation
        assert len(session1_data) == 1
        assert len(session2_data) == 1
        assert session1_data[0]["content"] == "Session 1 message"
        assert session2_data[0]["content"] == "Session 2 message"
        assert session1_data[0]["session_id"] == session1["id"]
        assert session2_data[0]["session_id"] == session2["id"]

    def test_message_ordering_persistence(self, client, headers: Dict[str, str], test_session_setup,
                                         database_connection, integration_helper, mock_embedding_service):
        """Test that messages are stored and retrieved in correct chronological order."""
        session = test_session_setup["session"]
        session_id = session["id"]
        
        # Add messages in sequence
        for i in range(5):
            message_data = {
                "messages": [
                    {"role": "user", "content": f"Message {i}"}
                ]
            }
            
            response = client.post(f"/api/v1/sessions/{session_id}/messages", json=message_data, headers=headers)
            assert response.status_code == 201
            
            # Add small delay to ensure different timestamps
            import time
            time.sleep(0.1)
        
        # Retrieve messages
        response = client.get(f"/api/v1/sessions/{session_id}/messages", headers=headers)
        assert response.status_code == 200
        
        messages = response.json()["data"]["messages"]
        assert len(messages) == 5
        
        # Verify chronological order
        for i, message in enumerate(messages):
            assert message["content"] == f"Message {i}"
            assert message["role"] == "user"
            
        # Verify timestamps are in ascending order
        timestamps = [message["created_at"] for message in messages]
        assert timestamps == sorted(timestamps)

    def test_list_messages_nonexistent_session(self, client, headers: Dict[str, str], mock_embedding_service):
        """Test that listing messages for non-existent session returns 404."""
        response = client.get("/api/v1/sessions/nonexistent-session/messages", headers=headers)
        
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["data"] is None

    def test_add_messages_nonexistent_session(self, client, headers: Dict[str, str], mock_embedding_service):
        """Test that adding messages to non-existent session returns 404."""
        message_data = {
            "messages": [
                {"role": "user", "content": "Test message"}
            ]
        }
        
        response = client.post("/api/v1/sessions/nonexistent-session/messages", json=message_data, headers=headers)
        
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["data"] is None

    def test_read_nonexistent_messages(self, client, headers: Dict[str, str], test_session_setup,
                                      integration_helper, mock_embedding_service):
        """Test that reading non-existent messages returns 404."""
        session = test_session_setup["session"]
        session_id = session["id"]
        
        read_data = {
            "message_ids": ["nonexistent-message-id"]
        }
        
        response = client.post(f"/api/v1/sessions/{session_id}/messages/read", json=read_data, headers=headers)
        
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["data"] is None

    def test_update_nonexistent_messages(self, client, headers: Dict[str, str], test_session_setup,
                                        integration_helper, mock_embedding_service):
        """Test that updating non-existent messages returns 404."""
        session = test_session_setup["session"]
        session_id = session["id"]
        
        update_data = {
            "message_ids": ["nonexistent-message-id"],
            "new_messages": [
                {"role": "user", "content": "Updated content"}
            ]
        }
        
        response = client.put(f"/api/v1/sessions/{session_id}/messages", json=update_data, headers=headers)
        
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["data"] is None

    def test_delete_nonexistent_messages(self, client, headers: Dict[str, str], test_session_setup,
                                        integration_helper, mock_embedding_service):
        """Test that deleting non-existent messages returns 404."""
        session = test_session_setup["session"]
        session_id = session["id"]
        
        delete_data = {
            "message_ids": ["nonexistent-message-id"]
        }
        
        response = client.request(
            "DELETE",
            f"/api/v1/sessions/{session_id}/messages",
            content=json.dumps(delete_data),
            headers={**headers, "Content-Type": "application/json"}
        )
        
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["code"] == 404
        assert response_data["data"] is None

    def test_message_cascade_deletion_on_session_deletion(self, client, headers: Dict[str, str], test_session_setup,
                                                         database_connection, integration_helper, mock_embedding_service):
        """Test that deleting a session cascades to delete related messages."""
        session = test_session_setup["session"]
        session_id = session["id"]
        
        # Add messages to session
        message_data = {
            "messages": [
                {"role": "user", "content": "Message 1"},
                {"role": "assistant", "content": "Response 1"}
            ]
        }
        
        add_response = client.post(f"/api/v1/sessions/{session_id}/messages", json=message_data, headers=headers)
        assert add_response.status_code == 201
        message_ids = add_response.json()["data"]["message_ids"]
        
        # Verify messages exist
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = %s",
            (session_id,)
        )
        message_count_before = cursor.fetchone()[0]
        cursor.close()
        
        assert message_count_before == 2
        
        # Delete the session
        delete_response = client.delete(f"/api/v1/sessions/{session_id}", headers=headers)
        assert delete_response.status_code == 200
        
        # Verify messages were cascaded deleted
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = %s",
            (session_id,)
        )
        message_count_after = cursor.fetchone()[0]
        cursor.close()
        
        assert message_count_after == 0 