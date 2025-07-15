"""
Integration tests for Knowledge API.

These tests validate that Knowledge API operations actually persist data 
to the database and work correctly with real database operations.
"""

import pytest
from typing import Dict, Any, List
from fastapi.testclient import TestClient


class TestKnowledgeAPIIntegration:
    """Integration tests for Knowledge API endpoints."""

    def test_add_knowledge_persistence(self, client: TestClient, headers: Dict[str, str],
                                      test_user_data: Dict[str, Any], 
                                      test_knowledge_data: List[str],
                                      database_connection, integration_helper,
                                      mock_embedding_service):
        """Test that adding knowledge actually persists to database."""
        # Create user
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        
        # Add knowledge
        response = client.post(
            f"/api/v1/users/{user['id']}/knowledge",
            json={"knowledge": test_knowledge_data},
            headers=headers
        )
        
        # Verify API response
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "knowledge" in response_data["data"]
        
        created_knowledge = response_data["data"]["knowledge"]
        assert len(created_knowledge) == len(test_knowledge_data)
        
        # Verify knowledge actually exists in database
        for i, knowledge in enumerate(created_knowledge):
            assert integration_helper.verify_database_record_exists(
                database_connection, "knowledge", knowledge["id"]
            )
            
            # Verify knowledge content in database
            cursor = database_connection.connection.cursor()
            cursor.execute(
                "SELECT user_id, content FROM knowledge WHERE id = %s",
                (knowledge["id"],)
            )
            db_record = cursor.fetchone()
            cursor.close()
            
            assert db_record is not None
            assert db_record[0] == user["id"]
            assert db_record[1] == test_knowledge_data[i]

    def test_knowledge_user_scoping(self, client: TestClient, headers: Dict[str, str],
                                   test_user_data: Dict[str, Any], 
                                   integration_helper, mock_embedding_service):
        """Test that knowledge is properly scoped to users."""
        # Create two users
        user1_data = {**test_user_data, "name": "integration_user_1"}
        user2_data = {**test_user_data, "name": "integration_user_2"}
        
        user1 = integration_helper.create_user_via_api(client, headers, user1_data)
        user2 = integration_helper.create_user_via_api(client, headers, user2_data)
        
        # Add knowledge to user1
        user1_knowledge = ["User 1 specific knowledge", "User 1 private information"]
        client.post(
            f"/api/v1/users/{user1['id']}/knowledge",
            json={"knowledge": user1_knowledge},
            headers=headers
        )
        
        # Add knowledge to user2
        user2_knowledge = ["User 2 specific knowledge", "User 2 private information"]
        client.post(
            f"/api/v1/users/{user2['id']}/knowledge",
            json={"knowledge": user2_knowledge},
            headers=headers
        )
        
        # Verify user1 only sees their knowledge
        response1 = client.get(f"/api/v1/users/{user1['id']}/knowledge", headers=headers)
        assert response1.status_code == 200
        
        user1_retrieved = response1.json()["data"]["knowledge"]
        user1_content = [k["content"] for k in user1_retrieved]
        
        assert "User 1 specific knowledge" in user1_content
        assert "User 1 private information" in user1_content
        assert "User 2 specific knowledge" not in user1_content
        assert "User 2 private information" not in user1_content
        
        # Verify user2 only sees their knowledge
        response2 = client.get(f"/api/v1/users/{user2['id']}/knowledge", headers=headers)
        assert response2.status_code == 200
        
        user2_retrieved = response2.json()["data"]["knowledge"]
        user2_content = [k["content"] for k in user2_retrieved]
        
        assert "User 2 specific knowledge" in user2_content
        assert "User 2 private information" in user2_content
        assert "User 1 specific knowledge" not in user2_content
        assert "User 1 private information" not in user2_content

    def test_knowledge_deletion_persistence(self, client: TestClient, headers: Dict[str, str],
                                           test_user_data: Dict[str, Any], 
                                           test_knowledge_data: List[str],
                                           database_connection, integration_helper,
                                           mock_embedding_service):
        """Test that deleting knowledge actually removes it from database."""
        # Create user and add knowledge
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        
        response = client.post(
            f"/api/v1/users/{user['id']}/knowledge",
            json={"knowledge": test_knowledge_data},
            headers=headers
        )
        
        created_knowledge = response.json()["data"]["knowledge"]
        knowledge_ids = [k["id"] for k in created_knowledge]
        
        # Verify knowledge exists in database
        for knowledge_id in knowledge_ids:
            assert integration_helper.verify_database_record_exists(
                database_connection, "knowledge", knowledge_id
            )
        
        # Delete knowledge
        delete_response = client.delete(
            f"/api/v1/users/{user['id']}/knowledge",
            json={"knowledge_ids": knowledge_ids},
            headers=headers
        )
        
        assert delete_response.status_code == 204
        
        # Verify knowledge no longer exists in database
        for knowledge_id in knowledge_ids:
            assert not integration_helper.verify_database_record_exists(
                database_connection, "knowledge", knowledge_id
            ) 