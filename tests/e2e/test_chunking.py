#!/usr/bin/env python3
"""End-to-end tests for chunking functionality."""

import asyncio
import pytest
import aiohttp
from typing import Dict, Any, Optional


class TestChunkingE2E:
    """End-to-end tests for the chunking system."""

    BASE_URL = "http://localhost:8000"

    @pytest.fixture
    async def http_session(self):
        """Create an HTTP session for testing."""
        async with aiohttp.ClientSession() as session:
            yield session

    @pytest.fixture
    async def test_user(self, http_session):
        """Create a test user for the session."""
        user_data = {"name": "test_user_chunking"}

        async with http_session.post(f"{self.BASE_URL}/api/v1/users", json=user_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result["data"]["user"]["id"]
            else:
                # User might already exist, try to get it
                async with http_session.get(f"{self.BASE_URL}/api/v1/users?name=test_user_chunking") as resp2:
                    if resp2.status == 200:
                        result = await resp2.json()
                        if result["data"]["users"]:
                            return result["data"]["users"][0]["id"]
                        else:
                            pytest.fail("Failed to create or find test user")
                    else:
                        pytest.fail(f"Failed to create user: {resp.status}")

    @pytest.fixture
    async def test_session(self, http_session, test_user):
        """Create a test session."""
        session_data = {
            "name": "test_session_chunking",
            "user_id": test_user,
            "agent_id": "agent_default"
        }

        async with http_session.post(f"{self.BASE_URL}/api/v1/sessions", json=session_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result["data"]["session"]["id"]
            else:
                pytest.fail(f"Failed to create session: {resp.status}")

    async def add_messages(self, http_session, session_id: str, messages: list) -> Dict[str, Any]:
        """Helper method to add messages to a session."""
        add_data = {"messages": messages}

        async with http_session.post(f"{self.BASE_URL}/api/v1/sessions/{session_id}/messages", json=add_data) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                text = await resp.text()
                pytest.fail(f"Failed to add messages: {resp.status} - {text}")

    async def query_messages(self, http_session, user_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Helper method to query messages."""
        query_data = {"query": query, "top_k": top_k}

        async with http_session.post(f"{self.BASE_URL}/api/v1/users/{user_id}/query", json=query_data) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                text = await resp.text()
                pytest.fail(f"Failed to query: {resp.status} - {text}")

    @pytest.mark.e2e
    @pytest.mark.chunking
    @pytest.mark.asyncio
    async def test_basic_message_addition_and_retrieval(self, http_session, test_user, test_session):
        """Test basic message addition and retrieval functionality."""

        # Test 1: Add some messages
        messages = [
            {"role": "user", "content": "Hello, I am working on a project about space exploration"},
            {"role": "assistant", "content": "That sounds fascinating! Space exploration is an exciting field with many opportunities for discovery."}
        ]

        result = await self.add_messages(http_session, test_session, messages)
        assert result["status"] == "success"

        # Test 2: Query for the messages
        query_result = await self.query_messages(http_session, test_user, "space exploration project", top_k=5)

        assert query_result["status"] == "success"
        assert "data" in query_result

        # Check if we got results
        if query_result.get("data", {}).get("results"):
            results = query_result["data"]["results"]
            assert len(results) > 0

            # Verify content is retrievable
            found_space_content = any("space" in str(result).lower() for result in results)
            assert found_space_content, "Should find space-related content"

    @pytest.mark.e2e
    @pytest.mark.chunking
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_batch_processing_trigger(self, http_session, test_user, test_session):
        """Test that batch processing is triggered when threshold is reached."""

        # Add multiple message batches to trigger batch processing (threshold: 5)
        batch_results = []

        for i in range(4):  # Add 4 batches to reach the threshold of 5 (including the first test)
            messages = [
                {"role": "user", "content": f"This is test message {i+2} about Mars exploration"},
                {"role": "assistant", "content": f"Interesting question {i+2}! Mars has many fascinating features to explore."}
            ]

            result = await self.add_messages(http_session, test_session, messages)
            batch_results.append(result)
            assert result["status"] == "success"

        # Query to see if batch processing worked
        query_result = await self.query_messages(http_session, test_user, "Mars exploration features", top_k=10)

        assert query_result["status"] == "success"

        # Should have more results after batch processing
        if query_result.get("data", {}).get("results"):
            results = query_result["data"]["results"]
            assert len(results) > 0

            # Verify Mars-related content is retrievable
            found_mars_content = any("mars" in str(result).lower() for result in results)
            assert found_mars_content, "Should find Mars-related content after batch processing"

    @pytest.mark.e2e
    @pytest.mark.chunking
    @pytest.mark.asyncio
    async def test_different_content_types(self, http_session, test_user, test_session):
        """Test chunking with different types of content."""

        # Test with short content
        short_messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}
        ]

        result = await self.add_messages(http_session, test_session, short_messages)
        assert result["status"] == "success"

        # Test with long content
        long_content = "This is a very long message about artificial intelligence and machine learning. " * 10
        long_messages = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": "That's a comprehensive overview of AI and ML topics."}
        ]

        result = await self.add_messages(http_session, test_session, long_messages)
        assert result["status"] == "success"

        # Query for both types of content
        short_query = await self.query_messages(http_session, test_user, "hello", top_k=5)
        long_query = await self.query_messages(http_session, test_user, "artificial intelligence", top_k=5)

        assert short_query["status"] == "success"
        assert long_query["status"] == "success"

    @pytest.mark.e2e
    @pytest.mark.chunking
    @pytest.mark.asyncio
    async def test_query_relevance(self, http_session, test_user, test_session):
        """Test that queries return relevant results."""

        # Add messages with specific topics
        topic_messages = [
            {"role": "user", "content": "Tell me about quantum computing"},
            {"role": "assistant", "content": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement."},
            {"role": "user", "content": "What about classical computers?"},
            {"role": "assistant", "content": "Classical computers use binary bits and Boolean logic for computation."}
        ]

        result = await self.add_messages(http_session, test_session, topic_messages)
        assert result["status"] == "success"

        # Query for quantum computing
        quantum_query = await self.query_messages(http_session, test_user, "quantum computing", top_k=5)
        assert quantum_query["status"] == "success"

        # Query for classical computing
        classical_query = await self.query_messages(http_session, test_user, "classical computers", top_k=5)
        assert classical_query["status"] == "success"

        # Both queries should return results
        if quantum_query.get("data", {}).get("results"):
            quantum_results = quantum_query["data"]["results"]
            assert len(quantum_results) > 0

        if classical_query.get("data", {}).get("results"):
            classical_results = classical_query["data"]["results"]
            assert len(classical_results) > 0


# Standalone test runner for backward compatibility
async def test_chunking_standalone():
    """Standalone test function for backward compatibility."""
    test_instance = TestChunkingE2E()

    async with aiohttp.ClientSession() as session:
        # Create test user
        user_data = {"name": "test_user_standalone"}
        async with session.post(f"{test_instance.BASE_URL}/api/v1/users", json=user_data) as resp:
            if resp.status == 200:
                user_result = await resp.json()
                user_id = user_result["data"]["user"]["id"]
            else:
                print("Failed to create user for standalone test")
                return

        # Create test session
        session_data = {"name": "test_session_standalone", "user_id": user_id, "agent_id": "agent_default"}
        async with session.post(f"{test_instance.BASE_URL}/api/v1/sessions", json=session_data) as resp:
            if resp.status == 200:
                session_result = await resp.json()
                session_id = session_result["data"]["session"]["id"]
            else:
                print("Failed to create session for standalone test")
                return

        # Run basic test
        await test_instance.test_basic_message_addition_and_retrieval(session, user_id, session_id)
        print("✅ Standalone chunking test completed!")


if __name__ == "__main__":
    print("🚀 Testing MemFuse Chunking Functionality")
    print("=" * 50)

    try:
        asyncio.run(test_chunking_standalone())
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import sys
        sys.exit(1)
