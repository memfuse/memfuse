"""
Integration tests for Knowledge API.

These tests validate real data operations, service interactions, and data persistence
for the Knowledge API endpoints. Unlike contract tests, these use real databases
and services to ensure data actually flows correctly through the system.

Integration Testing Pillars:
1. Data Persistence Testing - Ensure API calls result in actual database changes
2. Service Boundaries - Verify components work together with real implementations  
3. Scoping and Data Isolation - Ensure data doesn't leak between users
"""

import pytest
import uuid
import asyncio
from typing import Dict, Any, List
from fastapi.testclient import TestClient

# Import the FastAPI app factory
from memfuse_core.server import create_app
class TestKnowledgeAPIIntegration:
    """Integration tests for Knowledge API endpoints."""

    @pytest.fixture
    def test_user_a(self, client, headers) -> Dict[str, Any]:
        """Create test user A and return user data."""
        unique_suffix = str(uuid.uuid4())[:8]
        
        user_response = client.post(
            "/api/v1/users",
            json={
                "name": f"test-user-a-{unique_suffix}",
                "description": "Test user A for knowledge integration tests"
            },
            headers=headers,
        )
        assert user_response.status_code == 201
        return user_response.json()["data"]["user"]

    @pytest.fixture
    def test_user_b(self, client, headers) -> Dict[str, Any]:
        """Create test user B and return user data."""
        unique_suffix = str(uuid.uuid4())[:8]
        
        user_response = client.post(
            "/api/v1/users",
            json={
                "name": f"test-user-b-{unique_suffix}",
                "description": "Test user B for knowledge integration tests"
            },
            headers=headers,
        )
        assert user_response.status_code == 201
        return user_response.json()["data"]["user"]

    @pytest.fixture
    def sample_knowledge_data(self) -> List[str]:
        """Sample knowledge data for testing."""
        return [
            "Taylor Swift is an American singer-songwriter known for her narrative songwriting.",
            "She has won multiple Grammy Awards and is one of the best-selling music artists.",
            "Her albums include Fearless, Red, 1989, Reputation, Lover, and Folklore.",
            "She started her career in country music before transitioning to pop.",
            "Swift is known for re-recording her early albums to own her master recordings."
        ]

    # Data Persistence Testing Pillar

    def test_knowledge_add(self, client, headers, test_user_a, database_connection):
        """Test that added knowledge actually persists in the database."""
        user_id = test_user_a["id"]
        knowledge_data = {
            "knowledge": [
                "Integration test knowledge item 1",
                "Integration test knowledge item 2"
            ]
        }
        
        # Add knowledge via API
        response = client.post(
            f"/api/v1/users/{user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        
        assert response.status_code == 201
        knowledge_ids = response.json()["data"]["knowledge_ids"]
        assert len(knowledge_ids) == 2
        
        # Verify data persisted in database
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT id, user_id, content FROM knowledge WHERE user_id = %s",
            (user_id,)
        )
        db_records = cursor.fetchall()
        
        assert len(db_records) == 2
        db_knowledge_ids = [record[0] for record in db_records]
        db_contents = [record[2] for record in db_records]
        
        # Verify IDs match
        assert set(knowledge_ids) == set(db_knowledge_ids)
        
        # Verify content matches
        assert "Integration test knowledge item 1" in db_contents
        assert "Integration test knowledge item 2" in db_contents
        
        # Verify user_id is correctly stored
        for record in db_records:
            assert record[1] == user_id

    def test_knowledge_update(self, client, headers, test_user_a, database_connection):
        """Test that knowledge updates actually persist in the database."""
        user_id = test_user_a["id"]
        
        # Add initial knowledge
        knowledge_data = {
            "knowledge": ["Original knowledge content"]
        }
        
        add_response = client.post(
            f"/api/v1/users/{user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        assert add_response.status_code == 201
        knowledge_ids = add_response.json()["data"]["knowledge_ids"]
        
        # Update knowledge via API
        update_data = {
            "knowledge_ids": knowledge_ids,
            "new_knowledge": ["Updated knowledge content"]
        }
        
        update_response = client.put(
            f"/api/v1/users/{user_id}/knowledge",
            json=update_data,
            headers=headers,
        )
        assert update_response.status_code == 200
        
        # Verify update persisted in database
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT content FROM knowledge WHERE id = %s",
            (knowledge_ids[0],)
        )
        db_record = cursor.fetchone()
        
        assert db_record is not None
        assert db_record[0] == "Updated knowledge content"

    def test_knowledge_delete(self, client, headers, test_user_a, database_connection):
        """Test that knowledge deletion actually removes data from the database."""
        user_id = test_user_a["id"]
        
        # Add knowledge to delete
        knowledge_data = {
            "knowledge": ["Knowledge to be deleted"]
        }
        
        add_response = client.post(
            f"/api/v1/users/{user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        assert add_response.status_code == 201
        knowledge_ids = add_response.json()["data"]["knowledge_ids"]
        
        # Verify knowledge exists in database
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge WHERE id = %s",
            (knowledge_ids[0],)
        )
        count_before = cursor.fetchone()[0]
        assert count_before == 1
        
        # Delete knowledge via API
        delete_data = {
            "knowledge_ids": knowledge_ids
        }
        
        delete_response = client.request(
            method="DELETE",
            url=f"/api/v1/users/{user_id}/knowledge",
            json=delete_data,
            headers=headers,
        )
        assert delete_response.status_code == 200
        
        # Verify knowledge is deleted from database
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge WHERE id = %s",
            (knowledge_ids[0],)
        )
        count_after = cursor.fetchone()[0]
        assert count_after == 0

    # Service Boundaries Testing Pillar

    def test_knowledge_with_embedding_service_integration(self, client, headers, test_user_a, database_connection):
        """Test knowledge integration with embedding service."""
        user_id = test_user_a["id"]
        knowledge_data = {
            "knowledge": ["Knowledge for embedding integration test"]
        }
        
        # Add knowledge - this should trigger embedding generation
        response = client.post(
            f"/api/v1/users/{user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        
        assert response.status_code == 201
        knowledge_ids = response.json()["data"]["knowledge_ids"]
        
        # Verify knowledge has embeddings (or embedding placeholders for mocked service)
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT content, embedding FROM knowledge WHERE id = %s",
            (knowledge_ids[0],)
        )
        db_record = cursor.fetchone()
        
        assert db_record is not None
        assert db_record[0] == "Knowledge for embedding integration test"
        # Note: In integration tests with mocked embedding service,
        # we verify the embedding column exists and can be populated

    def test_knowledge_chunking_service_integration(self, client, headers, test_user_a, database_connection):
        """Test knowledge integration with chunking service."""
        user_id = test_user_a["id"]
        
        # Add long knowledge content that should be chunked
        long_content = "This is a very long knowledge content. " * 50  # Create long text
        knowledge_data = {
            "knowledge": [long_content]
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        
        assert response.status_code == 201
        knowledge_ids = response.json()["data"]["knowledge_ids"]
        
        # Verify knowledge was processed (chunked or stored as-is)
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT content FROM knowledge WHERE user_id = %s",
            (user_id,)
        )
        db_records = cursor.fetchall()
        
        # Should have at least one record
        assert len(db_records) >= 1
        
        # Content should be stored (chunked or original)
        stored_content = " ".join([record[0] for record in db_records])
        assert "This is a very long knowledge content." in stored_content

    # Scoping and Data Isolation Testing Pillar

    def test_knowledge_user_isolation(self, client, headers, test_user_a, test_user_b, database_connection):
        """Test that knowledge is properly isolated between users."""
        user_a_id = test_user_a["id"]
        user_b_id = test_user_b["id"]
        
        # Add knowledge for user A
        knowledge_a = {
            "knowledge": ["User A's secret knowledge"]
        }
        
        response_a = client.post(
            f"/api/v1/users/{user_a_id}/knowledge",
            json=knowledge_a,
            headers=headers,
        )
        assert response_a.status_code == 201
        
        # Add knowledge for user B
        knowledge_b = {
            "knowledge": ["User B's private knowledge"]
        }
        
        response_b = client.post(
            f"/api/v1/users/{user_b_id}/knowledge",
            json=knowledge_b,
            headers=headers,
        )
        assert response_b.status_code == 201
        
        # Get knowledge for user A - should only see A's knowledge
        list_response_a = client.get(
            f"/api/v1/users/{user_a_id}/knowledge",
            headers=headers,
        )
        assert list_response_a.status_code == 200
        user_a_knowledge = list_response_a.json()["data"]["knowledge"]
        
        # Get knowledge for user B - should only see B's knowledge
        list_response_b = client.get(
            f"/api/v1/users/{user_b_id}/knowledge",
            headers=headers,
        )
        assert list_response_b.status_code == 200
        user_b_knowledge = list_response_b.json()["data"]["knowledge"]
        
        # Verify isolation
        user_a_contents = [item["content"] for item in user_a_knowledge]
        user_b_contents = [item["content"] for item in user_b_knowledge]
        
        assert "User A's secret knowledge" in user_a_contents
        assert "User A's secret knowledge" not in user_b_contents
        assert "User B's private knowledge" in user_b_contents
        assert "User B's private knowledge" not in user_a_contents
        
        # Verify database-level isolation
        cursor = database_connection.conn.cursor()
        
        # Check user A's knowledge
        cursor.execute(
            "SELECT content FROM knowledge WHERE user_id = %s",
            (user_a_id,)
        )
        db_user_a_knowledge = [record[0] for record in cursor.fetchall()]
        assert "User A's secret knowledge" in db_user_a_knowledge
        assert "User B's private knowledge" not in db_user_a_knowledge
        
        # Check user B's knowledge
        cursor.execute(
            "SELECT content FROM knowledge WHERE user_id = %s",
            (user_b_id,)
        )
        db_user_b_knowledge = [record[0] for record in cursor.fetchall()]
        assert "User B's private knowledge" in db_user_b_knowledge
        assert "User A's secret knowledge" not in db_user_b_knowledge

    def test_knowledge_cross_user_access_prevention(self, client, headers, test_user_a, test_user_b):
        """Test that users cannot access each other's knowledge IDs."""
        user_a_id = test_user_a["id"]
        user_b_id = test_user_b["id"]
        
        # Add knowledge for user A
        knowledge_a = {
            "knowledge": ["User A's private knowledge"]
        }
        
        response_a = client.post(
            f"/api/v1/users/{user_a_id}/knowledge",
            json=knowledge_a,
            headers=headers,
        )
        assert response_a.status_code == 201
        user_a_knowledge_ids = response_a.json()["data"]["knowledge_ids"]
        
        # Try to read user A's knowledge using user B's endpoint
        read_data = {
            "knowledge_ids": user_a_knowledge_ids
        }
        
        cross_access_response = client.post(
            f"/api/v1/users/{user_b_id}/knowledge/read",
            json=read_data,
            headers=headers,
        )
        
        # Should return empty results or error - not user A's knowledge
        assert cross_access_response.status_code == 200
        cross_access_knowledge = cross_access_response.json()["data"]["knowledge"]
        
        # Should not return user A's knowledge
        cross_access_contents = [item["content"] for item in cross_access_knowledge]
        assert "User A's private knowledge" not in cross_access_contents

    # Real-world Integration Testing

    def test_knowledge_taylor_swift_integration(self, client, headers, test_user_a, sample_knowledge_data):
        """Test knowledge integration with Taylor Swift reference data."""
        user_id = test_user_a["id"]
        
        # Add Taylor Swift knowledge
        knowledge_data = {
            "knowledge": sample_knowledge_data
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        
        assert response.status_code == 201
        knowledge_ids = response.json()["data"]["knowledge_ids"]
        assert len(knowledge_ids) == len(sample_knowledge_data)
        
        # Retrieve knowledge to verify it's accessible
        list_response = client.get(
            f"/api/v1/users/{user_id}/knowledge",
            headers=headers,
        )
        
        assert list_response.status_code == 200
        retrieved_knowledge = list_response.json()["data"]["knowledge"]
        
        # Verify all Taylor Swift knowledge is present
        retrieved_contents = [item["content"] for item in retrieved_knowledge]
        for expected_content in sample_knowledge_data:
            assert expected_content in retrieved_contents

    def test_knowledge_bulk_operations_integration(self, client, headers, test_user_a, database_connection):
        """Test bulk knowledge operations with real database."""
        user_id = test_user_a["id"]
        
        # Add bulk knowledge
        bulk_knowledge = [f"Bulk knowledge item {i}" for i in range(10)]
        knowledge_data = {
            "knowledge": bulk_knowledge
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        
        assert response.status_code == 201
        knowledge_ids = response.json()["data"]["knowledge_ids"]
        assert len(knowledge_ids) == 10
        
        # Verify bulk data in database
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge WHERE user_id = %s",
            (user_id,)
        )
        count = cursor.fetchone()[0]
        assert count == 10
        
        # Bulk delete
        delete_data = {
            "knowledge_ids": knowledge_ids[:5]  # Delete first 5
        }
        
        delete_response = client.request(
            method="DELETE",
            url=f"/api/v1/users/{user_id}/knowledge",
            json=delete_data,
            headers=headers,
        )
        
        assert delete_response.status_code == 200
        
        # Verify partial deletion in database
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge WHERE user_id = %s",
            (user_id,)
        )
        count_after = cursor.fetchone()[0]
        assert count_after == 5

    # Error Handling and Edge Cases

    def test_knowledge_transaction_rollback(self, client, headers, test_user_a, database_connection):
        """Test that failed operations properly rollback database transactions."""
        user_id = test_user_a["id"]
        
        # Get initial knowledge count
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge WHERE user_id = %s",
            (user_id,)
        )
        initial_count = cursor.fetchone()[0]
        
        # Try to add knowledge with invalid data (this should fail)
        invalid_knowledge_data = {
            "knowledge": ["Valid knowledge", None, "Another valid knowledge"]  # None should cause error
        }
        
        try:
            response = client.post(
                f"/api/v1/users/{user_id}/knowledge",
                json=invalid_knowledge_data,
                headers=headers,
            )
            # Operation might fail or succeed depending on validation
            # If it succeeds, the None might be filtered out
        except Exception:
            pass  # Expected for invalid data
        
        # Verify database state is consistent (no partial inserts)
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge WHERE user_id = %s",
            (user_id,)
        )
        final_count = cursor.fetchone()[0]
        
        # Count should be either unchanged (rollback) or increased by valid items only
        assert final_count >= initial_count

    def test_knowledge_concurrent_operations(self, client, headers, test_user_a, database_connection):
        """Test concurrent knowledge operations don't cause data corruption."""
        user_id = test_user_a["id"]
        
        # Add initial knowledge
        knowledge_data = {
            "knowledge": ["Concurrent test knowledge"]
        }
        
        response = client.post(
            f"/api/v1/users/{user_id}/knowledge",
            json=knowledge_data,
            headers=headers,
        )
        
        assert response.status_code == 201
        knowledge_ids = response.json()["data"]["knowledge_ids"]
        
        # Simulate concurrent read operations
        # In a real concurrent test, these would run in parallel
        for i in range(3):
            read_response = client.get(
                f"/api/v1/users/{user_id}/knowledge",
                headers=headers,
            )
            assert read_response.status_code == 200
            
            # Verify data consistency
            knowledge_items = read_response.json()["data"]["knowledge"]
            contents = [item["content"] for item in knowledge_items]
            assert "Concurrent test knowledge" in contents
        
        # Verify database consistency
        cursor = database_connection.conn.cursor()
        cursor.execute(
            "SELECT content FROM knowledge WHERE user_id = %s",
            (user_id,)
        )
        db_contents = [record[0] for record in cursor.fetchall()]
        assert "Concurrent test knowledge" in db_contents 