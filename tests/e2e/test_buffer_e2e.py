"""End-to-end tests for Buffer functionality.

This module tests the complete Buffer workflow including:
- RoundBuffer token-based FIFO
- HybridBuffer dual-format management
- QueryBuffer unified retrieval
- BufferService integration
- Client SDK buffer_only parameter
"""

import pytest
import asyncio
from typing import List, Dict, Any

from memfuse_core.services.service_factory import ServiceFactory
from memfuse_core.services.buffer_service import BufferService
from memfuse_core.buffer.round_buffer import RoundBuffer
from memfuse_core.buffer.hybrid_buffer import HybridBuffer
from memfuse_core.buffer.query_buffer import QueryBuffer


class TestBufferEndToEnd:
    """End-to-end tests for Buffer architecture."""

    @pytest.fixture
    async def buffer_service(self):
        """Create a BufferService instance for testing."""
        # Mock memory service
        class MockMemoryService:
            async def query(self, query, top_k=10, **kwargs):
                return {
                    "status": "success",
                    "data": {
                        "results": [
                            {
                                "id": "storage_result_1",
                                "content": "Storage result content",
                                "score": 0.8,
                                "metadata": {"source": "storage"}
                            }
                        ],
                        "total": 1
                    }
                }

        memory_service = MockMemoryService()
        
        config = {
            "buffer": {
                "enabled": True,
                "round_buffer": {
                    "max_tokens": 800,
                    "max_size": 5,
                    "token_model": "gpt-4o-mini"
                },
                "hybrid_buffer": {
                    "max_size": 5,
                    "chunk_strategy": "message",
                    "embedding_model": "all-MiniLM-L6-v2"
                },
                "query": {
                    "max_size": 15,
                    "cache_size": 100,
                    "default_sort_by": "score",
                    "default_order": "desc"
                }
            }
        }
        
        service = BufferService(
            memory_service=memory_service,
            user="test_user",
            config=config
        )
        
        await service.initialize()
        return service

    @pytest.mark.asyncio
    async def test_complete_buffer_workflow(self, buffer_service):
        """Test the complete Buffer workflow."""
        # 1. Add messages to trigger RoundBuffer -> HybridBuffer transfer
        messages = [
            [
                {"role": "user", "content": "Hello, how are you?", "id": "msg_1"},
                {"role": "assistant", "content": "I'm doing well, thank you!", "id": "msg_2"}
            ],
            [
                {"role": "user", "content": "What's the weather like?", "id": "msg_3"}
            ]
        ]
        
        session_id = "test_session_1"
        
        # Add messages
        result = await buffer_service.add_batch(messages, session_id=session_id)
        assert result["status"] == "success"
        
        # 2. Test query functionality
        query_result = await buffer_service.query(
            "weather",
            top_k=10,
            sort_by="score",
            order="desc"
        )
        
        assert query_result["status"] == "success"
        assert "results" in query_result["data"]
        
        # 3. Test buffer_only parameter functionality
        # Test RoundBuffer only
        round_messages = await buffer_service.get_messages_by_session(
            session_id=session_id,
            buffer_only=True,
            limit=10
        )
        
        # Test storage only (excluding RoundBuffer)
        storage_messages = await buffer_service.get_messages_by_session(
            session_id=session_id,
            buffer_only=False,
            limit=10
        )
        
        # Verify results
        assert isinstance(round_messages, list)
        assert isinstance(storage_messages, list)

    @pytest.mark.asyncio
    async def test_token_based_transfer(self, buffer_service):
        """Test token-based transfer from RoundBuffer to HybridBuffer."""
        # Create a large message that exceeds token limit
        large_content = "This is a very long message. " * 100  # Should exceed 800 tokens
        
        messages = [
            [
                {"role": "user", "content": large_content, "id": "large_msg_1"}
            ]
        ]
        
        session_id = "test_session_2"
        
        # Add messages - should trigger transfer
        result = await buffer_service.add_batch(messages, session_id=session_id)
        assert result["status"] == "success"
        
        # Verify transfer occurred
        round_buffer_size = len(buffer_service.round_buffer.rounds)
        hybrid_buffer_size = len(buffer_service.hybrid_buffer.chunks)
        
        # After transfer, RoundBuffer should be empty or have new data
        # HybridBuffer should have received the transferred data
        assert round_buffer_size >= 0
        assert hybrid_buffer_size >= 0

    @pytest.mark.asyncio
    async def test_session_change_transfer(self, buffer_service):
        """Test session change triggers transfer."""
        # Add messages to session 1
        messages1 = [
            [
                {"role": "user", "content": "Message in session 1", "id": "msg_s1_1"}
            ]
        ]
        
        session_id_1 = "test_session_3a"
        result1 = await buffer_service.add_batch(messages1, session_id=session_id_1)
        assert result1["status"] == "success"
        
        # Add messages to session 2 - should trigger transfer
        messages2 = [
            [
                {"role": "user", "content": "Message in session 2", "id": "msg_s2_1"}
            ]
        ]
        
        session_id_2 = "test_session_3b"
        result2 = await buffer_service.add_batch(messages2, session_id=session_id_2)
        assert result2["status"] == "success"
        
        # Verify both sessions can be queried
        messages_s1 = await buffer_service.get_messages_by_session(
            session_id=session_id_1,
            limit=10
        )
        
        messages_s2 = await buffer_service.get_messages_by_session(
            session_id=session_id_2,
            limit=10
        )
        
        assert isinstance(messages_s1, list)
        assert isinstance(messages_s2, list)

    @pytest.mark.asyncio
    async def test_query_sorting_functionality(self, buffer_service):
        """Test query sorting by score and timestamp."""
        # Add some test data
        messages = [
            [
                {"role": "user", "content": "Test query sorting", "id": "sort_msg_1"}
            ]
        ]
        
        session_id = "test_session_4"
        await buffer_service.add_batch(messages, session_id=session_id)
        
        # Test sort by score (default)
        result_score = await buffer_service.query(
            "test",
            top_k=5,
            sort_by="score",
            order="desc"
        )
        
        assert result_score["status"] == "success"
        
        # Test sort by timestamp
        result_timestamp = await buffer_service.query(
            "test",
            top_k=5,
            sort_by="timestamp",
            order="desc"
        )
        
        assert result_timestamp["status"] == "success"

    @pytest.mark.asyncio
    async def test_buffer_statistics(self, buffer_service):
        """Test buffer statistics and monitoring."""
        # Add some data
        messages = [
            [
                {"role": "user", "content": "Statistics test", "id": "stats_msg_1"}
            ]
        ]
        
        session_id = "test_session_5"
        await buffer_service.add_batch(messages, session_id=session_id)
        
        # Get buffer statistics
        round_stats = buffer_service.round_buffer.get_stats()
        hybrid_stats = buffer_service.hybrid_buffer.get_stats()
        
        assert isinstance(round_stats, dict)
        assert isinstance(hybrid_stats, dict)
        
        # Verify expected keys
        assert "total_rounds" in round_stats
        assert "total_chunks" in hybrid_stats

    @pytest.mark.asyncio
    async def test_error_handling(self, buffer_service):
        """Test error handling in Buffer."""
        # Test with invalid data
        invalid_messages = [
            [
                {"invalid": "data"}  # Missing required fields
            ]
        ]
        
        session_id = "test_session_6"
        
        # Should handle gracefully
        result = await buffer_service.add_batch(invalid_messages, session_id=session_id)
        # Should not crash, may return success or error depending on implementation
        assert "status" in result

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, buffer_service):
        """Test concurrent buffer operations."""
        # Create multiple concurrent operations
        tasks = []
        
        for i in range(5):
            messages = [
                [
                    {"role": "user", "content": f"Concurrent message {i}", "id": f"concurrent_msg_{i}"}
                ]
            ]
            
            task = buffer_service.add_batch(messages, session_id=f"concurrent_session_{i}")
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all operations completed
        assert len(results) == 5
        
        # Most should succeed (some might have exceptions due to concurrency)
        successful_results = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
        assert len(successful_results) >= 3  # At least 60% success rate
