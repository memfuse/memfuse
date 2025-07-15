"""
Integration tests for Messages API.

These tests validate that Messages API operations actually persist data 
to the database and work correctly with real database operations.
"""

import pytest
from typing import Dict, Any, List
from fastapi.testclient import TestClient


class TestMessagesAPIIntegration:
    """Integration tests for Messages API endpoints."""

    def test_add_messages_persistence(self, client: TestClient, headers: Dict[str, str],
                                     test_user_data: Dict[str, Any], 
                                     test_agent_data: Dict[str, Any],
                                     test_session_data, test_message_data: List[Dict[str, Any]],
                                     database_connection, integration_helper,
                                     mock_embedding_service):
        """Test that adding messages actually persists to database."""
        # Create user, agent, and session
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        session_data = test_session_data(user["id"], agent["id"])
        session = integration_helper.create_session_via_api(client, headers, session_data)
        
        # Add messages to session
        response = client.post(
            f"/api/v1/sessions/{session['id']}/messages",
            json={"messages": test_message_data},
            headers=headers
        )
        
        # Verify API response
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "messages" in response_data["data"]
        
        created_messages = response_data["data"]["messages"]
        assert len(created_messages) == len(test_message_data)
        
        # Verify messages actually exist in database
        for i, message in enumerate(created_messages):
            assert integration_helper.verify_database_record_exists(
                database_connection, "messages", message["id"]
            )
            
            # Verify message content in database
            cursor = database_connection.connection.cursor()
            cursor.execute(
                "SELECT session_id, role, content FROM messages WHERE id = %s",
                (message["id"],)
            )
            db_record = cursor.fetchone()
            cursor.close()
            
            assert db_record is not None
            assert db_record[0] == session["id"]
            assert db_record[1] == test_message_data[i]["role"]
            assert db_record[2] == test_message_data[i]["content"]

    def test_message_ordering_persistence(self, client: TestClient, headers: Dict[str, str],
                                         test_user_data: Dict[str, Any], 
                                         test_agent_data: Dict[str, Any],
                                         test_session_data, database_connection,
                                         integration_helper, mock_embedding_service):
        """Test that messages maintain correct ordering in database."""
        # Create user, agent, and session
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        session_data = test_session_data(user["id"], agent["id"])
        session = integration_helper.create_session_via_api(client, headers, session_data)
        
        # Add messages in specific order
        ordered_messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
            {"role": "assistant", "content": "Fourth message"}
        ]
        
        client.post(
            f"/api/v1/sessions/{session['id']}/messages",
            json={"messages": ordered_messages},
            headers=headers
        )
        
        # Retrieve messages and verify order
        response = client.get(
            f"/api/v1/sessions/{session['id']}/messages",
            headers=headers
        )
        
        assert response.status_code == 200
        retrieved_messages = response.json()["data"]["messages"]
        
        # Verify order is maintained
        for i, message in enumerate(retrieved_messages):
            assert message["role"] == ordered_messages[i]["role"]
            assert message["content"] == ordered_messages[i]["content"]

    def test_message_session_isolation(self, client: TestClient, headers: Dict[str, str],
                                      test_user_data: Dict[str, Any], 
                                      test_agent_data: Dict[str, Any],
                                      test_session_data, integration_helper,
                                      mock_embedding_service):
        """Test that messages are properly isolated between sessions."""
        # Create user, agent, and two sessions
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        session1_data = test_session_data(user["id"], agent["id"], "_session1")
        session2_data = test_session_data(user["id"], agent["id"], "_session2")
        
        session1 = integration_helper.create_session_via_api(client, headers, session1_data)
        session2 = integration_helper.create_session_via_api(client, headers, session2_data)
        
        # Add unique messages to each session
        session1_messages = [
            {"role": "user", "content": "Message for session 1"},
            {"role": "assistant", "content": "Response in session 1"}
        ]
        
        session2_messages = [
            {"role": "user", "content": "Message for session 2"},
            {"role": "assistant", "content": "Response in session 2"}
        ]
        
        # Add messages to respective sessions
        client.post(
            f"/api/v1/sessions/{session1['id']}/messages",
            json={"messages": session1_messages},
            headers=headers
        )
        
        client.post(
            f"/api/v1/sessions/{session2['id']}/messages",
            json={"messages": session2_messages},
            headers=headers
        )
        
        # Verify session 1 only has its messages
        response1 = client.get(
            f"/api/v1/sessions/{session1['id']}/messages",
            headers=headers
        )
        
        messages1 = response1.json()["data"]["messages"]
        content1 = [msg["content"] for msg in messages1]
        
        assert "Message for session 1" in content1
        assert "Response in session 1" in content1
        assert "Message for session 2" not in content1
        assert "Response in session 2" not in content1
        
        # Verify session 2 only has its messages
        response2 = client.get(
            f"/api/v1/sessions/{session2['id']}/messages",
            headers=headers
        )
        
        messages2 = response2.json()["data"]["messages"]
        content2 = [msg["content"] for msg in messages2]
        
        assert "Message for session 2" in content2
        assert "Response in session 2" in content2
        assert "Message for session 1" not in content2
        assert "Response in session 1" not in content2 