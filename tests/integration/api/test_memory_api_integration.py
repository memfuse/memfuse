"""
Integration tests for Memory API.

These tests validate that Memory API operations work correctly with real embeddings
and actual database operations, including the Taylor Swift reference test.
"""

import pytest
from typing import Dict, Any, List
from fastapi.testclient import TestClient


class TestMemoryAPIIntegration:
    """Integration tests for Memory API endpoints."""

    def test_memory_query_with_real_embeddings(self, client: TestClient, headers: Dict[str, str],
                                              test_user_data: Dict[str, Any], 
                                              test_agent_data: Dict[str, Any],
                                              test_session_data, test_message_data: List[Dict[str, Any]],
                                              integration_helper, real_embedding_service):
        """Test memory query with real embedding service for accurate retrieval."""
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
        assert response.status_code == 201
        
        # Query memory with real embeddings
        query_data = {
            "query": "integration testing",
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user['id']}/memory",
            json=query_data,
            headers=headers
        )
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "results" in response_data["data"]
        
        # Verify results are not empty (should find relevant content)
        results = response_data["data"]["results"]
        assert len(results) > 0
        
        # Verify result structure
        for result in results:
            assert "content" in result
            assert "score" in result
            assert "metadata" in result

    def test_taylor_swift_memory_retrieval(self, client: TestClient, headers: Dict[str, str],
                                          test_user_data: Dict[str, Any], 
                                          test_agent_data: Dict[str, Any],
                                          test_session_data, taylor_swift_test_data: Dict[str, Any],
                                          integration_helper, real_embedding_service):
        """Test Taylor Swift memory retrieval using MSC-MC10 dataset."""
        # Create user, agent, and session
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        session_data = test_session_data(user["id"], agent["id"])
        session = integration_helper.create_session_via_api(client, headers, session_data)
        
        # Add Taylor Swift conversation data to session
        for session_messages in taylor_swift_test_data["haystack_sessions"]:
            response = client.post(
                f"/api/v1/sessions/{session['id']}/messages",
                json={"messages": session_messages},
                headers=headers
            )
            assert response.status_code == 201
        
        # Query memory with Taylor Swift question
        query_data = {
            "query": taylor_swift_test_data["question"],
            "top_k": 10
        }
        
        response = client.post(
            f"/api/v1/users/{user['id']}/memory",
            json=query_data,
            headers=headers
        )
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "results" in response_data["data"]
        
        # Verify results are not empty
        results = response_data["data"]["results"]
        assert len(results) > 0, "Memory query should return results for Taylor Swift conversation"
        
        # Verify relevant content is retrieved
        all_content = " ".join([result["content"] for result in results]).lower()
        expected_keywords = taylor_swift_test_data["expected_keywords"]
        
        # Check for presence of expected keywords
        for keyword in expected_keywords:
            assert keyword.lower() in all_content, f"Expected keyword '{keyword}' not found in retrieved content"

    def test_memory_scoping_user_isolation(self, client: TestClient, headers: Dict[str, str],
                                          test_user_data: Dict[str, Any], 
                                          test_agent_data: Dict[str, Any],
                                          test_session_data, integration_helper,
                                          real_embedding_service):
        """Test that memory queries are properly scoped to users."""
        # Create two users
        user1_data = {**test_user_data, "name": "integration_user_1"}
        user2_data = {**test_user_data, "name": "integration_user_2"}
        
        user1 = integration_helper.create_user_via_api(client, headers, user1_data)
        user2 = integration_helper.create_user_via_api(client, headers, user2_data)
        
        # Create agents and sessions for both users
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        session1_data = test_session_data(user1["id"], agent["id"], "_user1")
        session2_data = test_session_data(user2["id"], agent["id"], "_user2")
        
        session1 = integration_helper.create_session_via_api(client, headers, session1_data)
        session2 = integration_helper.create_session_via_api(client, headers, session2_data)
        
        # Add unique messages to each user's session
        user1_messages = [
            {"role": "user", "content": "I love playing basketball and NBA games."},
            {"role": "assistant", "content": "Basketball is a great sport! Do you have a favorite NBA team?"}
        ]
        
        user2_messages = [
            {"role": "user", "content": "I'm really into classical music and piano concerts."},
            {"role": "assistant", "content": "Classical music is beautiful! Who's your favorite composer?"}
        ]
        
        # Add messages to respective sessions
        client.post(
            f"/api/v1/sessions/{session1['id']}/messages",
            json={"messages": user1_messages},
            headers=headers
        )
        
        client.post(
            f"/api/v1/sessions/{session2['id']}/messages",
            json={"messages": user2_messages},
            headers=headers
        )
        
        # Query User 1's memory for basketball
        response1 = client.post(
            f"/api/v1/users/{user1['id']}/memory",
            json={"query": "basketball NBA", "top_k": 5},
            headers=headers
        )
        
        # Query User 2's memory for basketball
        response2 = client.post(
            f"/api/v1/users/{user2['id']}/memory",
            json={"query": "basketball NBA", "top_k": 5},
            headers=headers
        )
        
        # Verify both responses are successful
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # User 1 should have basketball results
        user1_results = response1.json()["data"]["results"]
        user1_content = " ".join([result["content"] for result in user1_results]).lower()
        assert "basketball" in user1_content or "nba" in user1_content
        
        # User 2 should have no basketball results (or much lower relevance)
        user2_results = response2.json()["data"]["results"]
        user2_content = " ".join([result["content"] for result in user2_results]).lower()
        
        # User 2's results should not contain basketball content (scoping works)
        assert "basketball" not in user2_content and "nba" not in user2_content

    def test_memory_scoping_session_filtering(self, client: TestClient, headers: Dict[str, str],
                                             test_user_data: Dict[str, Any], 
                                             test_agent_data: Dict[str, Any],
                                             test_session_data, integration_helper,
                                             real_embedding_service):
        """Test that memory queries can be filtered by session."""
        # Create user and agent
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        
        # Create two sessions
        session1_data = test_session_data(user["id"], agent["id"], "_session1")
        session2_data = test_session_data(user["id"], agent["id"], "_session2")
        
        session1 = integration_helper.create_session_via_api(client, headers, session1_data)
        session2 = integration_helper.create_session_via_api(client, headers, session2_data)
        
        # Add unique content to each session
        session1_messages = [
            {"role": "user", "content": "I'm planning a trip to Tokyo next month."},
            {"role": "assistant", "content": "Tokyo is amazing! Are you excited about visiting Japan?"}
        ]
        
        session2_messages = [
            {"role": "user", "content": "I'm learning Python programming and data science."},
            {"role": "assistant", "content": "Python is great for data science! What libraries are you using?"}
        ]
        
        # Add messages to sessions
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
        
        # Query memory with session filtering
        query_data = {
            "query": "programming Python",
            "top_k": 5,
            "session_id": session2["id"]
        }
        
        response = client.post(
            f"/api/v1/users/{user['id']}/memory",
            json=query_data,
            headers=headers
        )
        
        # Verify response
        assert response.status_code == 200
        results = response.json()["data"]["results"]
        assert len(results) > 0
        
        # Verify results are from correct session
        content = " ".join([result["content"] for result in results]).lower()
        assert "python" in content or "programming" in content
        
        # Verify Tokyo content is not included (different session)
        assert "tokyo" not in content and "japan" not in content

    def test_memory_store_type_filtering(self, client: TestClient, headers: Dict[str, str],
                                        test_user_data: Dict[str, Any], 
                                        test_agent_data: Dict[str, Any],
                                        test_session_data, integration_helper,
                                        real_embedding_service):
        """Test memory queries with different store types."""
        # Create user, agent, and session
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        session_data = test_session_data(user["id"], agent["id"])
        session = integration_helper.create_session_via_api(client, headers, session_data)
        
        # Add messages with diverse content
        messages = [
            {"role": "user", "content": "I work as a software engineer at a tech startup."},
            {"role": "assistant", "content": "That sounds exciting! What technologies do you work with?"},
            {"role": "user", "content": "We use Python, React, and PostgreSQL for our main application."},
            {"role": "assistant", "content": "Great stack! How do you like working with PostgreSQL?"}
        ]
        
        client.post(
            f"/api/v1/sessions/{session['id']}/messages",
            json={"messages": messages},
            headers=headers
        )
        
        # Test vector store query
        vector_query = {
            "query": "technology stack",
            "top_k": 3,
            "store_type": "vector"
        }
        
        response = client.post(
            f"/api/v1/users/{user['id']}/memory",
            json=vector_query,
            headers=headers
        )
        
        assert response.status_code == 200
        vector_results = response.json()["data"]["results"]
        assert len(vector_results) > 0
        
        # Test keyword store query
        keyword_query = {
            "query": "PostgreSQL",
            "top_k": 3,
            "store_type": "keyword"
        }
        
        response = client.post(
            f"/api/v1/users/{user['id']}/memory",
            json=keyword_query,
            headers=headers
        )
        
        assert response.status_code == 200
        keyword_results = response.json()["data"]["results"]
        assert len(keyword_results) > 0
        
        # Verify keyword results contain exact match
        keyword_content = " ".join([result["content"] for result in keyword_results]).lower()
        assert "postgresql" in keyword_content

    def test_memory_empty_results_handling(self, client: TestClient, headers: Dict[str, str],
                                          test_user_data: Dict[str, Any], 
                                          integration_helper, real_embedding_service):
        """Test memory query when no relevant content exists."""
        # Create user with no content
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        
        # Query for something that doesn't exist
        query_data = {
            "query": "nonexistent quantum entanglement discussion",
            "top_k": 5
        }
        
        response = client.post(
            f"/api/v1/users/{user['id']}/memory",
            json=query_data,
            headers=headers
        )
        
        # Verify response structure is correct even with no results
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "results" in response_data["data"]
        
        # Results should be empty
        results = response_data["data"]["results"]
        assert len(results) == 0

    def test_memory_pagination_and_limits(self, client: TestClient, headers: Dict[str, str],
                                         test_user_data: Dict[str, Any], 
                                         test_agent_data: Dict[str, Any],
                                         test_session_data, integration_helper,
                                         real_embedding_service):
        """Test memory query pagination and top_k limits."""
        # Create user, agent, and session
        user = integration_helper.create_user_via_api(client, headers, test_user_data)
        agent = integration_helper.create_agent_via_api(client, headers, test_agent_data)
        session_data = test_session_data(user["id"], agent["id"])
        session = integration_helper.create_session_via_api(client, headers, session_data)
        
        # Add many messages
        messages = []
        for i in range(10):
            messages.extend([
                {"role": "user", "content": f"This is test message number {i} about testing."},
                {"role": "assistant", "content": f"Response {i}: Testing is important for quality."}
            ])
        
        client.post(
            f"/api/v1/sessions/{session['id']}/messages",
            json={"messages": messages},
            headers=headers
        )
        
        # Test different top_k values
        for top_k in [1, 3, 5, 10]:
            query_data = {
                "query": "testing",
                "top_k": top_k
            }
            
            response = client.post(
                f"/api/v1/users/{user['id']}/memory",
                json=query_data,
                headers=headers
            )
            
            assert response.status_code == 200
            results = response.json()["data"]["results"]
            assert len(results) <= top_k, f"Should return at most {top_k} results" 